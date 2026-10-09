"""
Phase 1 Agent Nodes

This module implements the core nodes for the Phase 1 Code Interpreter Agent:
1. generate_code: The Brain - generates Python code based on SOP and metadata
2. execute_code: The Executor - runs code and updates state with results

Privacy-Preserving Architecture:
- LLM only sees data_summary (metadata), never full CSV
- After execution, metadata is refreshed from the updated dataset
- State updates via "Magic Output Protocol" (JSON in stdout)

Magic Output Protocol:
    Generated code prints JSON to stdout:
    print(json.dumps({"__METABO_UPDATE__": {"is_balanced": True, "current_data_path": "/tmp/new.csv"}}))
    
    The execute_node parses this and updates context_variables and current_data_path.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from .state import Phase1State, ExecutionRecord, DataSummary, get_current_step
from .executor import LocalPythonExecutor
from .tool_contracts import TOOL_CONTRACTS, get_tools_documentation
from .ast_validator import validate_code_with_contracts
from src.utils.config_manager import get_config
from src.utils.feature_name_standardizer import standardize_feature_name
from src.tools.analysis.data_analysis_tools import impute_missing_tool, preprocessing_qa_reporter_tool

ROW_LEVEL_PREVIEW_DISABLED = "<disabled: row-level values are never sent to the LLM>"
CURRENT_DATA_PATH_TOKEN = "__METABO_CURRENT_DATA_PATH__"
MAX_PROMPT_COLUMN_NAMES = 80
MAX_PROMPT_DTYPE_SAMPLES = 40
MAX_PROMPT_MISSING_VALUE_SAMPLES = 25
MAX_PROMPT_LIST_ITEMS = 20
MAX_PROMPT_DICT_ITEMS = 40
MAX_PROMPT_STRING_CHARS = 2000
MAX_PROMPT_RECURSION_DEPTH = 4


def _get_phase1_stability_temp_dir() -> str:
    """Get canonical stability temp directory for Phase1."""
    cfg = get_config()
    return (
        cfg.get_phase1_path("stability_temp_dir")
        or "output/phase1/intermediate/latest/stability_logs"
    )


def _enforce_stability_temp_dir(code: str) -> str:
    """
    Normalize stability temp-dir usage to configured canonical path.

    This guards against hardcoded legacy paths like `temp_stab_results` that may
    still appear in SOP examples or generated snippets.
    """
    if not code:
        return code

    if (
        "perform_stability_selection" not in code
        and "calculate_frequencies_from_logs" not in code
        and "run_train_only_stability_selection" not in code
        and "build_stable_panel_from_logs" not in code
        and "temp_stab" not in code
        and "temp_stability" not in code
    ):
        return code

    stability_temp_dir = _get_phase1_stability_temp_dir()

    # Normalize direct assignment.
    code = re.sub(
        r"temp_dir\s*=\s*['\"][^'\"]+['\"]",
        f"temp_dir = '{stability_temp_dir}'",
        code,
    )

    # Normalize kwarg usage.
    code = re.sub(
        r"temp_dir\s*=\s*['\"][^'\"]+['\"]",
        f"temp_dir='{stability_temp_dir}'",
        code,
    )

    # Normalize direct frequency-log reads.
    code = re.sub(
        r"calculate_frequencies_from_logs\(\s*['\"][^'\"]+['\"]\s*\)",
        f"calculate_frequencies_from_logs('{stability_temp_dir}')",
        code,
    )

    return code


def _phase1_runtime_path_map() -> Dict[str, str]:
    """Return legacy Phase 1 path prefixes mapped to the active job runtime."""
    try:
        cfg = get_config()
        path_map = {
            "output/phase1/intermediate/latest": cfg.get_phase1_path("intermediate_latest_dir"),
            "output/phase1/artifacts": cfg.get_phase1_path("artifacts_dir"),
            "output/phase1/final": cfg.get_phase1_path("final_dir"),
        }
    except Exception:
        path_map = {}

    return {
        legacy: str(canonical)
        for legacy, canonical in path_map.items()
        if canonical
    }


def _enforce_phase1_runtime_paths(code: str) -> str:
    """Rewrite legacy relative Phase 1 output paths before subprocess execution.

    Generated snippets historically used ``output/phase1/...``.  Jobs now use
    an isolated ``runtime/phase1`` tree, so leaving those literals unchanged
    writes artifacts into the shared project directory and makes them invisible
    to the next step.  The replacement is deliberately limited to known Phase
    1 roots and does not alter arbitrary user data paths.
    """
    if not code:
        return code

    normalized = code
    for legacy, canonical in _phase1_runtime_path_map().items():
        normalized = normalized.replace(legacy, canonical)
    return normalized


def _normalize_phase1_state_paths(value: Any) -> Any:
    """Normalize canonical Phase 1 path prefixes in Magic Output values."""
    if isinstance(value, str):
        normalized = value
        for legacy, canonical in _phase1_runtime_path_map().items():
            normalized = normalized.replace(legacy, canonical)
        return normalized
    if isinstance(value, dict):
        return {key: _normalize_phase1_state_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_phase1_state_paths(item) for item in value]
    return value


def _maybe_prepare_stage15_scaling_input(state: Phase1State, step_id: str) -> Phase1State:
    """
    Ensure Step 1.5.5 sees the merged engineered matrix rather than the
    pre-engineering or imputed file when a valid merged artifact already exists.
    """
    if str(step_id) != "1.5.5":
        return state

    current_data_path = str(state.get("current_data_path", "") or "").strip()
    if not current_data_path or not os.path.exists(current_data_path):
        return state

    current_path_norm = os.path.basename(current_data_path).lower()
    if "data_with_engineered_features" in current_path_norm:
        return state

    context_variables = dict(state.get("context_variables", {}))
    target_column = state.get("data_summary", {}).get("target_column")

    candidate_paths: List[str] = []
    for key in ("unscaled_enriched_data_path", "engineered_features_path", "latest_engineered_features_path"):
        value = str(context_variables.get(key, "") or "").strip()
        if value:
            candidate_paths.append(value)

    candidate_paths.extend(
        [
            "output/phase1/intermediate/latest/engineered/data_with_engineered_features.csv",
            "output/phase1/intermediate/latest/engineered/data_with_engineered_features_unscaled.csv",
        ]
    )

    try:
        current_df = pd.read_csv(current_data_path, nrows=5)
        current_cols = list(current_df.columns)
        current_header_count = len(current_cols)
        current_row_count = len(pd.read_csv(current_data_path))
    except Exception:
        return state

    best_candidate = None
    seen = set()
    for candidate_path in candidate_paths:
        candidate_path = os.path.abspath(candidate_path)
        if candidate_path in seen or not os.path.exists(candidate_path) or candidate_path == os.path.abspath(current_data_path):
            continue
        seen.add(candidate_path)
        try:
            candidate_df = pd.read_csv(candidate_path, nrows=5)
            candidate_cols = list(candidate_df.columns)
            candidate_row_count = len(pd.read_csv(candidate_path))
        except Exception:
            continue

        new_feature_count = len([col for col in candidate_cols if col not in current_cols])
        if candidate_row_count != current_row_count or new_feature_count <= 0:
            continue

        score = (new_feature_count, len(candidate_cols))
        if best_candidate is None or score > best_candidate["score"]:
            best_candidate = {
                "path": candidate_path,
                "new_feature_count": new_feature_count,
                "candidate_col_count": len(candidate_cols),
                "score": score,
            }

    if best_candidate is None:
        return state

    updated_context = dict(context_variables)
    updated_context["unscaled_enriched_data_path"] = best_candidate["path"]
    updated_context["current_data_path"] = best_candidate["path"]

    return {
        **state,
        "current_data_path": best_candidate["path"],
        "data_summary": get_data_summary(best_candidate["path"], target_column),
        "context_variables": updated_context,
    }


def _autofix_engineered_feature_tool_calls(code: str) -> str:
    """
    Narrow auto-fix for common LLM misuse on engineered feature generators.

    The LLM occasionally treats function tools as mapping-like objects and emits
    invalid calls such as `generate_reaction_ratios.get(...)`. Those calls fail
    AST validation before execution and can prematurely terminate Stage 1.5.

    This helper intentionally applies a minimal rewrite only for the three
    centralized engineered feature generators introduced in the refactor.
    """
    if not code:
        return code

    replacements = {
        "generate_reaction_ratios": "generate_reaction_ratios",
        "generate_taxonomy_sums": "generate_taxonomy_sums",
        "generate_pathway_scores": "generate_pathway_scores",
    }

    fixed_code = code

    # The canonical configuration helper lives in ``src.utils``.  Generated
    # repair snippets occasionally use the older/guessed ``src.config`` path;
    # normalize that import before execution so a path typo does not obscure
    # the original analytical failure.
    fixed_code = re.sub(
        r"from\s+src\.config\s+import\s+get_config",
        "from src.utils.config_manager import get_config",
        fixed_code,
    )
    for tool_name in replacements:
        # Common bad patterns:
        # - generate_reaction_ratios.get(...)
        # - (generate_reaction_ratios).get(...)
        fixed_code = re.sub(
            rf"\(?\s*\b{tool_name}\b\s*\)?\s*\.\s*get\s*\(",
            f"{tool_name}(",
            fixed_code,
        )

    return fixed_code


def _inject_phase1_runtime_bindings(code: str, state: Phase1State) -> str:
    """Provide stable local bindings expected by SOP-generated code.

    Each generated snippet runs in a fresh subprocess.  Bindings such as
    ``current_data_path`` and ``context_variables`` therefore need to be
    materialized explicitly from local state.  Only paths/metadata are added.
    """
    context_vars = dict(state.get("context_variables", {}) or {})
    current_path = str(state.get("current_data_path", "") or "")
    context_vars.setdefault("current_data_path", current_path)
    context_vars.setdefault("target_column", state.get("data_summary", {}).get("target_column"))

    def _ctx_path(*keys: str) -> str:
        for key in keys:
            value = context_vars.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    # Generated snippets run in a fresh subprocess and may be emitted before a
    # prior SOP step has written these directory aliases into context.  Resolve
    # them from the active runtime configuration as a deterministic fallback.
    try:
        cfg = get_config()
        configured_intermediate = str(
            cfg.get_phase1_path("intermediate_latest_dir")
            or "output/phase1/intermediate/latest"
        )
        configured_stability = str(
            cfg.get_phase1_path("stability_temp_dir")
            or os.path.join(configured_intermediate, "stability_logs")
        )
        configured_artifacts = str(
            cfg.get_phase1_path("artifacts_dir")
            or "output/phase1/artifacts"
        )
    except Exception:
        configured_intermediate = "output/phase1/intermediate/latest"
        configured_stability = os.path.join(configured_intermediate, "stability_logs")
        configured_artifacts = "output/phase1/artifacts"

    resolved_intermediate = _ctx_path(
        "phase1_intermediate_latest", "intermediate_latest_dir"
    ) or configured_intermediate
    resolved_stability = _ctx_path("stability_temp_dir") or configured_stability
    context_vars.setdefault("phase1_intermediate_latest", resolved_intermediate)
    context_vars.setdefault("intermediate_latest_dir", resolved_intermediate)
    context_vars.setdefault("stability_temp_dir", resolved_stability)

    # Step 6.2 returns training results through the state protocol.  Materialize
    # that local JSON before Step 6.3 executes so report-generation code does
    # not depend on a relative working-directory side effect.
    training_results = (
        context_vars.get("training_results_json")
        or state.get("training_results_json")
    )
    if isinstance(training_results, dict):
        try:
            artifacts_dir = Path(configured_artifacts)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            training_results_path = artifacts_dir / "autogluon_training_results.json"
            if not training_results_path.exists():
                training_results_path.write_text(
                    json.dumps(training_results, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
        except Exception:
            # The generated code still has its own discovery/recovery logic.
            pass

    try:
        context_literal = json.loads(json.dumps(context_vars, default=str))
    except Exception:
        context_literal = {"current_data_path": current_path}

    bindings = {
        "current_data_path": current_path,
        "context_variables": context_literal,
        "pre_engineering_data_path": _ctx_path("pre_engineering_data_path", "unscaled_enriched_data_path"),
        "phase1_intermediate_latest": resolved_intermediate,
        "stability_temp_dir": resolved_stability,
        "target_column": _ctx_path("target_column") or state.get("data_summary", {}).get("target_column", ""),
        "sample_id_column": _ctx_path("sample_id_column", "id_column"),
        "batch_column": _ctx_path("batch_column"),
        "phase0_output": context_literal.get("phase0_output"),
        "selected_dataset_path": _ctx_path("selected_dataset_path") or current_path,
        "selected_holdout_data_path": _ctx_path("selected_holdout_data_path", "holdout_data_path"),
        # These aliases are used by the legacy label-encoding SOP examples.
        # Materializing them here keeps a repair focused on the injected fault
        # instead of failing secondarily on an omitted boilerplate assignment.
        "clinical_scenario": context_vars.get("clinical_scenario") or "Clinical biomarker study",
        "scenario": context_vars.get("clinical_scenario") or "Clinical biomarker study",
    }
    lines = [
        "# MetaboAgent local runtime bindings (metadata/state only)",
        "from src.utils.config_manager import get_config",
        "cfg = get_config()",
    ]
    for name, value in bindings.items():
        lines.append(f"{name} = {value!r}")
    lines.append("HOLDOUT_DATA_PATH = selected_holdout_data_path")
    return "\n".join(lines) + "\n\n" + code


def _ensure_phase1_intermediate_mirror() -> None:
    """Ensure `output/phase1/intermediate/latest` exists and mirror legacy dirs."""
    cfg = get_config()
    phase1_paths = cfg.get_phase1_paths()
    legacy_dirs = phase1_paths.get("intermediate_legacy_dirs", [])
    mirror_root = phase1_paths.get("intermediate_latest_dir", "output/phase1/intermediate/latest")
    os.makedirs(mirror_root, exist_ok=True)

    mirrored = []
    missing = []
    for legacy_dir in legacy_dirs:
        if not legacy_dir:
            continue

        src_dir = os.path.abspath(legacy_dir)
        dst_dir = os.path.join(mirror_root, os.path.basename(legacy_dir.rstrip("/")))

        if os.path.isdir(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            mirrored.append({"source": src_dir, "target": os.path.abspath(dst_dir)})
        else:
            # Always materialize a placeholder folder to keep structure stable.
            os.makedirs(dst_dir, exist_ok=True)
            missing.append({"source": src_dir, "target": os.path.abspath(dst_dir)})

    manifest_path = os.path.join(mirror_root, "_mirror_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "completed",
                "mirrored_count": len(mirrored),
                "missing_count": len(missing),
                "mirrored": mirrored,
                "missing": missing,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[Intermediate Mirror] ✅ Mirror root ensured: {mirror_root}")
    print(f"[Intermediate Mirror] ✅ Mirrored: {len(mirrored)}, Missing(source): {len(missing)}")
    print(f"[Intermediate Mirror] ✅ Manifest: {manifest_path}")


def _read_tabular_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def _write_tabular_file(df: pd.DataFrame, file_path: str) -> None:
    if file_path.endswith((".xlsx", ".xls")):
        df.to_excel(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)


def _truncate_prompt_text(text: Any, max_chars: int = MAX_PROMPT_STRING_CHARS) -> str:
    """Trim long prompt strings to a bounded size."""
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... [truncated {omitted} chars]"


def _build_prompt_path_aliases(
    context_variables: Dict[str, Any],
    current_data_path: str,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Replace local paths with deterministic opaque tokens before an LLM call."""
    path_to_token: Dict[str, str] = {}
    if current_data_path:
        path_to_token[str(current_data_path)] = CURRENT_DATA_PATH_TOKEN

    def _is_path(value: str) -> bool:
        value = str(value or "").strip()
        if not value or value.startswith(("http://", "https://")):
            return False
        return os.path.isabs(value) or os.path.exists(value)

    def _token_for(path_value: str) -> str:
        path_value = str(path_value)
        if path_value not in path_to_token:
            path_to_token[path_value] = f"__METABO_LOCAL_PATH_{len(path_to_token):03d}__"
        return path_to_token[path_value]

    def _walk(value: Any, depth: int = 0) -> Any:
        if depth >= MAX_PROMPT_RECURSION_DEPTH:
            return "<max_depth_reached>"
        if isinstance(value, str):
            return _token_for(value) if _is_path(value) else value
        if isinstance(value, dict):
            return {str(key): _walk(item, depth + 1) for key, item in value.items()}
        if isinstance(value, list):
            return [_walk(item, depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(_walk(item, depth + 1) for item in value)
        return value

    sanitized_context = _walk(context_variables)
    token_to_path = {token: path for path, token in path_to_token.items()}
    return sanitized_context, token_to_path


def _alias_paths_in_text(text: str, token_to_path: Dict[str, str]) -> str:
    """Convert known local paths in diagnostic text back to their opaque tokens."""
    sanitized = str(text or "")
    for token, path in sorted(token_to_path.items(), key=lambda item: len(item[1]), reverse=True):
        sanitized = sanitized.replace(path, token)
    return sanitized


def _materialize_local_paths(code: str, token_to_path: Dict[str, str]) -> str:
    """Resolve LLM-visible path tokens locally immediately before execution."""
    materialized = str(code or "")
    for token, path in sorted(token_to_path.items(), key=lambda item: len(item[0]), reverse=True):
        materialized = materialized.replace(token, path)
    return materialized


def _summarize_prompt_value(value: Any, depth: int = 0) -> Any:
    """Recursively compress large context payloads before sending them to the LLM."""
    if depth >= MAX_PROMPT_RECURSION_DEPTH:
        return "<max_depth_reached>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_prompt_text(value)

    if isinstance(value, list):
        items = [
            _summarize_prompt_value(item, depth + 1)
            for item in value[:MAX_PROMPT_LIST_ITEMS]
        ]
        if len(value) > MAX_PROMPT_LIST_ITEMS:
            items.append(f"... [{len(value) - MAX_PROMPT_LIST_ITEMS} more items omitted]")
        return items

    if isinstance(value, tuple):
        return _summarize_prompt_value(list(value), depth)

    if isinstance(value, set):
        return _summarize_prompt_value(sorted(value, key=lambda item: str(item)), depth)

    if isinstance(value, dict):
        summarized: Dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:MAX_PROMPT_DICT_ITEMS]:
            summarized[str(key)] = _summarize_prompt_value(item, depth + 1)
        if len(items) > MAX_PROMPT_DICT_ITEMS:
            summarized["__truncated__"] = (
                f"{len(items) - MAX_PROMPT_DICT_ITEMS} additional keys omitted"
            )
        return summarized

    return _truncate_prompt_text(repr(value))


def _select_columns_for_prompt(
    columns: List[str],
    target_column: Optional[str],
    protected_columns: List[str],
) -> List[str]:
    """Keep prompt-visible columns informative without dumping ultra-wide schemas."""
    if len(columns) <= MAX_PROMPT_COLUMN_NAMES:
        return list(columns)

    priority_names = [
        target_column,
        *protected_columns,
        "Sample_ID",
        "sample_id",
        "SampleID",
        "sampleid",
        "subject_id",
        "patient_id",
        "row_id",
        "ROW_ID",
        "id",
        "ID",
        "group",
        "Group",
    ]

    selected: List[str] = []
    seen = set()

    def _append(name: Optional[str]) -> None:
        name = str(name or "").strip()
        if not name or name in seen or name not in columns:
            return
        seen.add(name)
        selected.append(name)

    for name in priority_names:
        _append(name)

    tail_budget = min(10, max(0, MAX_PROMPT_COLUMN_NAMES - len(selected)))
    head_budget = max(0, MAX_PROMPT_COLUMN_NAMES - len(selected) - tail_budget)

    for name in columns[:head_budget]:
        _append(name)

    if tail_budget > 0:
        for name in columns[-tail_budget:]:
            _append(name)

    for name in columns:
        if len(selected) >= MAX_PROMPT_COLUMN_NAMES:
            break
        _append(name)

    return selected


def _build_dtype_summary_for_prompt(
    dtypes: Dict[str, str],
    prompt_columns: List[str],
) -> Dict[str, Any]:
    """Summarize dtypes without serializing every column in ultra-wide matrices."""
    dtype_counts: Dict[str, int] = {}
    for dtype in dtypes.values():
        dtype_name = str(dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

    highlighted = {
        column: str(dtypes.get(column, "unknown"))
        for column in prompt_columns[:MAX_PROMPT_DTYPE_SAMPLES]
        if column in dtypes
    }

    return {
        "dtype_counts": dtype_counts,
        "prompt_column_dtypes": highlighted,
        "omitted_dtype_columns": max(0, len(dtypes) - len(highlighted)),
    }


def _build_missing_value_summary_for_prompt(
    missing_values: Optional[Dict[str, int]],
    prompt_columns: List[str],
) -> Dict[str, Any]:
    """Keep only the most actionable missingness hints for the prompt."""
    if not missing_values:
        return {}

    prompt_hits = {
        column: int(missing_values[column])
        for column in prompt_columns
        if column in missing_values
    }
    if prompt_hits:
        return {
            "prompt_columns_with_missing_values": prompt_hits,
            "total_columns_with_missing_values": len(missing_values),
        }

    top_missing = sorted(
        ((str(column), int(count)) for column, count in missing_values.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:MAX_PROMPT_MISSING_VALUE_SAMPLES]
    return {
        "top_missing_value_columns": dict(top_missing),
        "total_columns_with_missing_values": len(missing_values),
    }


def _build_prompt_data_summary(
    data_summary: DataSummary,
    context_variables: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a prompt-safe view of dataset metadata."""
    target_col = str(data_summary.get("target_column") or "unknown")
    protected_columns = _collect_phase1_protected_columns(context_variables, target_col)
    prompt_columns = _select_columns_for_prompt(
        data_summary.get("columns", []),
        target_col,
        protected_columns,
    )
    omitted_columns = max(0, int(data_summary.get("n_cols", 0)) - len(prompt_columns))

    notes: List[str] = []
    if omitted_columns > 0:
        notes.append(
            "Schema truncated for prompt efficiency: "
            f"showing {len(prompt_columns)} of {data_summary.get('n_cols', 0)} columns."
        )
    if protected_columns:
        notes.append(f"Protected columns: {', '.join(protected_columns[:10])}")

    return {
        "target_column": target_col,
        "protected_columns": protected_columns,
        "prompt_columns": prompt_columns,
        "dtype_summary": _build_dtype_summary_for_prompt(
            data_summary.get("dtypes", {}),
            prompt_columns,
        ),
        "missing_value_summary": _build_missing_value_summary_for_prompt(
            data_summary.get("missing_values"),
            prompt_columns,
        ),
        "notes": notes,
    }


def _apply_train_fitted_standard_scaling(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    protected_cols: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    train_scaled = train_df.copy()
    holdout_scaled = holdout_df.copy()
    protected_set = set(protected_cols)

    numeric_features = [
        col
        for col in train_df.columns
        if col not in protected_set and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    scaling_stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_features:
        train_series = pd.to_numeric(train_df[col], errors="coerce")
        holdout_series = pd.to_numeric(holdout_df[col], errors="coerce")

        mean_value = float(train_series.mean())
        std_value = float(train_series.std(ddof=0))
        if not std_value or abs(std_value) < 1e-12:
            std_value = 1.0

        train_scaled[col] = (train_series - mean_value) / std_value
        holdout_scaled[col] = (holdout_series - mean_value) / std_value
        scaling_stats[col] = {"mean": mean_value, "std": std_value}

    return train_scaled, holdout_scaled, scaling_stats


def _extract_phase0_biomarker_terms(context_variables: Dict[str, Any]) -> List[str]:
    """Extract Phase 0 biomarker IDs/names as match terms for Phase 1 protection."""
    phase0_output = context_variables.get("phase0_output")
    from src.tools.domain.metabolite_name_mapper_enhanced import extract_phase0_biomarker_terms

    return extract_phase0_biomarker_terms(phase0_output)


def _resolve_phase1_protected_anchor_features(
    dataset_columns: List[str],
    target_column: Optional[str],
    context_variables: Dict[str, Any],
) -> Dict[str, Any]:
    """Map Phase 0 biomarkers onto current Phase 1 dataset columns."""
    biomarker_terms = _extract_phase0_biomarker_terms(context_variables)
    if not biomarker_terms:
        return {
            "protected_anchor_features": [],
            "available_prior_anchor_features": [],
            "anchor_mapping_report": {},
            "phase0_prior_resolution_report": {},
        }

    excluded_columns = {
        col
        for col in [
            "Sample_ID",
            "sample_id",
            "SampleID",
            "sampleid",
            "ID",
            "id",
            "ROW_ID",
            "row_id",
            "__row_id__",
            target_column or "",
        ]
        if col
    }
    feature_columns = [col for col in dataset_columns if col not in excluded_columns]

    try:
        from src.tools.domain.metabolite_name_mapper_enhanced import resolve_prior_biomarkers_to_columns

        resolution = resolve_prior_biomarkers_to_columns(
            biomarker_terms,
            feature_columns,
        )
        matched_features = list(resolution.get("protected_features", []) or [])
        available_features = list(resolution.get("available_features", []) or matched_features)
        mapping_report = dict(resolution.get("resolution_report", {}) or {})
    except Exception as exc:
        return {
            "protected_anchor_features": [],
            "available_prior_anchor_features": [],
            "anchor_mapping_report": {
                "error": str(exc),
                "match_rate": 0.0,
                "matched_columns": [],
            },
            "phase0_prior_resolution_report": {
                "error": str(exc),
                "match_rate": 0.0,
                "matched_columns": [],
                "unmatched_metabolites": biomarker_terms,
            },
        }

    validated = [feature for feature in matched_features if feature in feature_columns]
    validated_available = [feature for feature in available_features if feature in feature_columns]
    return {
        "protected_anchor_features": list(dict.fromkeys(validated)),
        "available_prior_anchor_features": list(dict.fromkeys(validated_available)),
        "anchor_mapping_report": mapping_report if isinstance(mapping_report, dict) else {},
        "phase0_prior_resolution_report": mapping_report if isinstance(mapping_report, dict) else {},
    }


def _merge_phase1_protected_anchor_context(
    context_variables: Dict[str, Any],
    dataset_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Inject Phase 0 protected anchors into Phase 1 mandatory/final feature context."""
    updated_context = dict(context_variables or {})
    dataset_columns = list(dataset_columns or [])
    if not dataset_columns:
        return updated_context

    resolved = _resolve_phase1_protected_anchor_features(
        dataset_columns=dataset_columns,
        target_column=target_column,
        context_variables=updated_context,
    )
    protected_anchor_features = list(resolved.get("protected_anchor_features", []) or [])
    available_prior_anchor_features = list(resolved.get("available_prior_anchor_features", []) or [])
    mapping_report = resolved.get("anchor_mapping_report", {}) or {}
    prior_resolution_report = resolved.get("phase0_prior_resolution_report", {}) or {}

    existing_mandatory = [
        str(feature or "").strip()
        for feature in updated_context.get("mandatory_features", []) or []
        if str(feature or "").strip()
    ]
    updated_context["protected_anchor_features"] = protected_anchor_features
    updated_context["n_protected_anchor_features"] = len(protected_anchor_features)
    updated_context["available_prior_anchor_features"] = available_prior_anchor_features
    updated_context["n_available_prior_anchor_features"] = len(available_prior_anchor_features)
    updated_context["protected_anchor_mapping_report"] = mapping_report
    updated_context["phase0_prior_resolution_report"] = prior_resolution_report
    updated_context["mandatory_features"] = list(
        dict.fromkeys(existing_mandatory + protected_anchor_features)
    )
    updated_context["n_mandatory_features"] = len(updated_context["mandatory_features"])

    for feature_key in ("consensus_features", "final_selected_features"):
        existing = [
            str(feature or "").strip()
            for feature in updated_context.get(feature_key, []) or []
            if str(feature or "").strip()
        ]
        if existing:
            updated_context[feature_key] = list(
                dict.fromkeys(existing + protected_anchor_features)
            )

    return updated_context


def _sync_phase1_consensus_artifacts(context_variables: Dict[str, Any]) -> None:
    """
    Keep the canonical consensus handoff file aligned with the in-memory context.

    Step 5.3 correctly unions `mandatory_features` with the voted consensus list,
    but some intermediate files can still reflect the pre-union state produced by
    Step 5.2.2. Persist the merged canonical list so downstream validation and
    provenance consumers see the same feature set.
    """
    context_variables = context_variables or {}
    consensus_features = [
        str(feature or "").strip()
        for feature in context_variables.get("consensus_features", []) or []
        if str(feature or "").strip()
    ]
    if not consensus_features:
        return

    cfg = get_config()
    feature_selection_dir = os.path.join(
        cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest",
        "feature_selection",
    )
    os.makedirs(feature_selection_dir, exist_ok=True)

    consensus_path = os.path.join(feature_selection_dir, "temp_consensus_features.json")
    with open(consensus_path, "w", encoding="utf-8") as f:
        json.dump(consensus_features, f, indent=2, ensure_ascii=False)


def _write_phase1_final_selection_summary(
    *,
    target_column: str,
    protected_cols: List[str],
    consensus_features: List[str],
    mandatory_features: List[str],
    selected_features: List[str],
    missing_train_features: List[str],
    missing_holdout_features: List[str],
    train_pool_path: str,
    train_output_path: str,
    holdout_data_path: str,
    holdout_output_path: str,
    selection_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persist the canonical Step 5.3 final-selection summary for downstream consumers.

    Phase 4 still reads `final_selection_summary.json` for methodology/reporting.
    After the Phase 1 Step 5.3 refactor, the final dataset was updated but this
    compatibility artifact was no longer being refreshed.
    """
    cfg = get_config()
    feature_selection_dir = os.path.join(
        cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest",
        "feature_selection",
    )
    os.makedirs(feature_selection_dir, exist_ok=True)

    summary_path = os.path.join(feature_selection_dir, "final_selection_summary.json")
    summary_payload = {
        "schema_version": "phase1.final_selection_summary.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_stage": "phase1.step_5_3_programmatic_finalization",
        "selection_method": "consensus_plus_mandatory_projection",
        "fallback_strategy_used": False,
        "target_column": str(target_column),
        "protected_columns": list(protected_cols),
        "consensus_features_count": int(len(consensus_features)),
        "consensus_features": list(consensus_features),
        "mandatory_features_count": int(len(mandatory_features)),
        "mandatory_features": list(mandatory_features),
        "selected_features_count": int(len(selected_features)),
        "selected_features": list(selected_features),
        "missing_train_features_count": int(len(missing_train_features)),
        "missing_train_features": list(missing_train_features),
        "missing_holdout_features_filled_with_zero_count": int(len(missing_holdout_features)),
        "missing_holdout_features_filled_with_zero": list(missing_holdout_features),
        "input_train_pool_path": str(train_pool_path),
        "input_holdout_pool_path": str(holdout_data_path),
        "output_path": str(train_output_path),
        "holdout_output_path": str(holdout_output_path),
    }
    if isinstance(selection_details, dict) and selection_details:
        summary_payload["selection_details"] = _safe_json_payload(selection_details)
    _atomic_write_json(summary_path, summary_payload)
    return summary_path


def _resolve_phase1_selector_family(context_vars: Dict[str, Any]) -> List[str]:
    from src.tools.analysis.feature_selection_tools import (
        PHASE1_DEFAULT_SELECTOR_FAMILY,
        PHASE1_SELECTOR_ALIASES,
    )

    default_family = list(PHASE1_DEFAULT_SELECTOR_FAMILY)

    explicit_family = context_vars.get("phase1_selector_family")
    if isinstance(explicit_family, list):
        normalized_set = set()
        for method in explicit_family:
            raw_name = str(method).strip()
            if not raw_name:
                continue
            canonical_name = PHASE1_SELECTOR_ALIASES.get(raw_name.lower())
            if canonical_name:
                normalized_set.add(canonical_name)

        if normalized_set == set(default_family):
            return [method for method in default_family if method in normalized_set]

    return default_family


def _load_phase1_engineered_feature_registry() -> Dict[str, Dict[str, Any]]:
    cfg = get_config()
    candidate_paths = []
    configured_path = cfg.get_phase1_path("engineered_feature_registry")
    if configured_path:
        candidate_paths.append(str(configured_path))

    intermediate_latest_dir = cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest"
    candidate_paths.append(os.path.join(intermediate_latest_dir, "engineered", "engineered_feature_registry.json"))

    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            features = payload.get("features", []) if isinstance(payload, dict) else []
            registry_index = {}
            for record in features:
                if not isinstance(record, dict):
                    continue
                feature_name = str(record.get("feature_name", "") or "").strip()
                if feature_name:
                    registry_index[feature_name] = record
            if registry_index:
                return registry_index
        except Exception:
            continue
    return {}


def _build_phase1_candidate_universe(
    df: pd.DataFrame,
    feature_cols: List[str],
    mandatory_features: List[str],
) -> Dict[str, Any]:
    registry_index = _load_phase1_engineered_feature_registry()
    mandatory_set = set(mandatory_features)

    records = []
    for feature in feature_cols:
        standardized_name = standardize_feature_name(feature)
        registry_record = registry_index.get(feature, {}) if isinstance(registry_index, dict) else {}
        engineered_type = str(registry_record.get("feature_type", "") or "").strip().lower()
        records.append(
            {
                "feature": feature,
                "standardized_name": standardized_name,
                "is_numeric": bool(pd.api.types.is_numeric_dtype(df[feature])),
                "is_prior_protected": bool(standardized_name in mandatory_set),
                "feature_origin": "engineered" if engineered_type else "raw",
                "feature_subtype": engineered_type or "raw",
                "source_phase": "phase0+phase1" if standardized_name in mandatory_set else "phase1",
            }
        )

    summary = {
        "candidate_count": int(len(records)),
        "protected_prior_count": int(sum(1 for row in records if row["is_prior_protected"])),
        "raw_feature_count": int(sum(1 for row in records if row["feature_origin"] == "raw")),
        "engineered_feature_count": int(sum(1 for row in records if row["feature_origin"] == "engineered")),
    }
    return {"records": records, "summary": summary}


def _resolve_phase1_feature_selection_params(
    context_vars: Dict[str, Any],
    n_features_in_pool: int,
    n_mandatory_features: int,
) -> Dict[str, Any]:
    final_panel_target_size = int(context_vars.get("phase1_final_panel_target_size", 15) or 15)
    final_panel_target_size = max(final_panel_target_size, n_mandatory_features)

    per_method_select_k = int(
        context_vars.get("phase1_per_method_select_k", max(final_panel_target_size, 15)) or max(final_panel_target_size, 15)
    )
    per_method_select_k = max(1, min(per_method_select_k, max(1, n_features_in_pool)))

    min_final_feature_count = int(context_vars.get("phase1_min_final_feature_count", 11) or 11)
    min_final_feature_count = max(1, min(min_final_feature_count, max(1, n_features_in_pool)))

    return {
        "phase1_stability_iterations": int(context_vars.get("phase1_stability_iterations", context_vars.get("stability_iterations", 30)) or 30),
        "phase1_resampling_fraction": float(context_vars.get("phase1_resampling_fraction", 0.8) or 0.8),
        "phase1_final_panel_target_size": int(final_panel_target_size),
        "phase1_per_method_select_k": int(per_method_select_k),
        "phase1_selection_frequency_threshold": float(context_vars.get("phase1_selection_frequency_threshold", 0.60) or 0.60),
        "phase1_method_consensus_threshold": int(context_vars.get("phase1_method_consensus_threshold", 3) or 3),
        "phase1_max_pairwise_correlation": float(context_vars.get("phase1_max_pairwise_correlation", 0.90) or 0.90),
        "phase1_min_final_feature_count": int(min_final_feature_count),
    }


def _run_unified_phase1_stability_selection(
    state: Phase1State,
    *,
    step_id: str,
    expected_balance: Optional[bool],
) -> Phase1State:
    try:
        cfg = get_config()
        context_vars = dict(state.get("context_variables", {}))
        observed_balance = context_vars.get("is_balanced")
        if expected_balance is not None and observed_balance is not None and bool(observed_balance) != bool(expected_balance):
            raise ValueError(f"Step {step_id} precondition violated: expected is_balanced == {expected_balance}")

        data_path = str(context_vars.get("current_data_path") or state.get("current_data_path") or "").strip()
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Step {step_id} input not found: {data_path}")

        df_cols = list(_read_tabular_file(data_path).columns)
        target_column = str(context_vars.get("target_column") or "").strip()
        if target_column not in df_cols:
            for candidate in ("group", "Group", "target"):
                if candidate in df_cols:
                    target_column = candidate
                    break
        if target_column not in df_cols:
            raise ValueError(f"Step {step_id} could not resolve a target column (expected one of group/Group/target)")

        mandatory_features = [
            standardize_feature_name(feature)
            for feature in (context_vars.get("mandatory_features", []) or [])
            if str(feature or "").strip()
        ]
        mandatory_features = list(dict.fromkeys([feature for feature in mandatory_features if feature]))

        selector_family = _resolve_phase1_selector_family(context_vars)
        selection_params = _resolve_phase1_feature_selection_params(
            context_vars=context_vars,
            n_features_in_pool=int(context_vars.get("n_features_in_pool", 0) or 0),
            n_mandatory_features=len(mandatory_features),
        )

        feature_selection_dir = os.path.join(
            cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest",
            "feature_selection",
        )
        os.makedirs(feature_selection_dir, exist_ok=True)

        stability_temp_root = str(context_vars.get("stability_temp_dir") or _get_phase1_stability_temp_dir())

        from src.tools.analysis.feature_selection_tools import (
            run_train_only_stability_selection,
            build_stable_panel_from_logs,
        )

        stability_run_result = json.loads(
            run_train_only_stability_selection(
                data_path=data_path,
                group_col=target_column,
                iterations=int(selection_params["phase1_stability_iterations"]),
                methods=selector_family,
                n_features_per_method=int(selection_params["phase1_per_method_select_k"]),
                temp_dir=stability_temp_root,
                sample_fraction=float(selection_params["phase1_resampling_fraction"]),
                mandatory_features=mandatory_features,
                data_already_scaled=bool(context_vars.get("train_fit_scaling_applied", False)),
            )
        )
        if not stability_run_result.get("success"):
            raise ValueError(str(stability_run_result.get("error") or "stability selection failed"))

        panel_summary = json.loads(
            build_stable_panel_from_logs(
                temp_dir=stability_temp_root,
                iterations=int(selection_params["phase1_stability_iterations"]),
                methods=selector_family,
                final_panel_size=int(selection_params["phase1_final_panel_target_size"]),
                mandatory_features=mandatory_features,
                selection_frequency_threshold=float(selection_params["phase1_selection_frequency_threshold"]),
                method_consensus_threshold=int(selection_params["phase1_method_consensus_threshold"]),
                data_path=data_path,
                group_col=target_column,
                max_pairwise_correlation=float(selection_params["phase1_max_pairwise_correlation"]),
                n_features_per_method=int(selection_params["phase1_per_method_select_k"]),
            )
        )
        if not panel_summary.get("success"):
            raise ValueError(str(panel_summary.get("error") or "panel summary failed"))

        stability_summary_path = os.path.join(feature_selection_dir, "stability_selection_summary.json")
        stability_scores_path = os.path.join(feature_selection_dir, "stability_scores.json")
        stable_core_path = os.path.join(feature_selection_dir, "stable_core_features.json")
        panel_selection_path = os.path.join(feature_selection_dir, "final_panel_selection.json")
        temp_consensus_path = os.path.join(feature_selection_dir, "temp_consensus_features.json")
        temp_strategy_path = os.path.join(feature_selection_dir, "temp_selection_strategy.json")

        final_panel_features = [
            standardize_feature_name(feature)
            for feature in (panel_summary.get("final_panel_features") or [])
            if str(feature or "").strip()
        ]
        final_panel_features = list(dict.fromkeys([feature for feature in final_panel_features if feature]))
        stable_core_features = [
            standardize_feature_name(feature)
            for feature in (panel_summary.get("stable_core_features") or [])
            if str(feature or "").strip()
        ]
        stable_core_features = list(dict.fromkeys([feature for feature in stable_core_features if feature]))

        if not final_panel_features:
            raise ValueError(f"Step {step_id} produced an empty final panel")

        _atomic_write_json(stability_summary_path, panel_summary)
        _atomic_write_json(stability_scores_path, {"stability_scores": panel_summary.get("stability_scores", [])})
        _atomic_write_json(stable_core_path, {"stable_core_features": stable_core_features})
        _atomic_write_json(panel_selection_path, {"final_panel_features": final_panel_features})
        _atomic_write_json(temp_consensus_path, final_panel_features)
        _atomic_write_json(
            temp_strategy_path,
            {
                "selection_strategy": "train_only_stability_panel",
                "step_id": step_id,
                "selector_family": selector_family,
                "iterations": int(selection_params["phase1_stability_iterations"]),
                "n_features_per_method": int(selection_params["phase1_per_method_select_k"]),
                "thresholds": panel_summary.get("thresholds", {}),
                "panel_completion": panel_summary.get("panel_completion", {}),
            },
        )

        state_updates = {
            "selection_strategy": "train_only_stability_panel",
            "consensus_features": final_panel_features,
            "final_selected_features": final_panel_features,
            "stable_core_features": stable_core_features,
            "n_features_selected": int(len(final_panel_features)),
            "n_methods_used": int(len(selector_family)),
            "stability_iterations": int(selection_params["phase1_stability_iterations"]),
            "phase1_stability_iterations": int(selection_params["phase1_stability_iterations"]),
            "phase1_resampling_fraction": float(selection_params["phase1_resampling_fraction"]),
            "phase1_selector_family": selector_family,
            "phase1_per_method_select_k": int(selection_params["phase1_per_method_select_k"]),
            "phase1_final_panel_target_size": int(selection_params["phase1_final_panel_target_size"]),
            "phase1_selection_frequency_threshold": float(selection_params["phase1_selection_frequency_threshold"]),
            "phase1_method_consensus_threshold": int(selection_params["phase1_method_consensus_threshold"]),
            "phase1_max_pairwise_correlation": float(selection_params["phase1_max_pairwise_correlation"]),
            "stability_temp_dir": stability_temp_root,
            "stability_summary_path": stability_summary_path,
            "stability_scores_path": stability_scores_path,
            "stable_core_path": stable_core_path,
            "phase1_panel_selection_path": panel_selection_path,
            "mandatory_features": mandatory_features,
        }

        execution_record = ExecutionRecord(
            step_id=step_id,
            generated_code=f"# Programmatic Step {step_id} execution",
            stdout=json.dumps({"__METABO_UPDATE__": _safe_json_payload(state_updates)}, ensure_ascii=False),
            stderr="",
            success=True,
            execution_time=0.0,
            state_updates=state_updates,
        )

        updated_context = {**context_vars, **state_updates}
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "context_variables": updated_context,
            "retry_count": 0,
            "error": None,
            "last_error": None,
        }
    except Exception as e:
        execution_record = ExecutionRecord(
            step_id=step_id,
            generated_code=f"# Programmatic Step {step_id} execution",
            stdout="",
            stderr=str(e),
            success=False,
            execution_time=0.0,
            state_updates=None,
        )
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "error": f"Programmatic Step {step_id} failed: {str(e)}",
            "last_error": f"Programmatic Step {step_id} failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "completed": False,
        }


def _load_phase1_consensus_features(context_variables: Dict[str, Any]) -> List[str]:
    """
    Load the canonical consensus feature list for Step 5.3 finalization.

    Preferred order:
    1. In-memory `context_variables['consensus_features']`
    2. Canonical temp handoff file `temp_consensus_features.json`
    3. Best-effort fallback keys from `consensus_validation.json`

    The validation report is intentionally treated as a fallback only because it
    may omit the full feature list and retain only counts/previews.
    """
    context_variables = context_variables or {}
    raw_context_features = context_variables.get("consensus_features", []) or []
    consensus_features = [
        str(feature or "").strip()
        for feature in raw_context_features
        if str(feature or "").strip()
    ]
    if consensus_features:
        return list(dict.fromkeys(consensus_features))

    cfg = get_config()
    feature_selection_dir = os.path.join(
        cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest",
        "feature_selection",
    )
    temp_consensus_path = os.path.join(feature_selection_dir, "temp_consensus_features.json")
    if os.path.exists(temp_consensus_path):
        try:
            with open(temp_consensus_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return list(dict.fromkeys([
                    str(feature or "").strip()
                    for feature in payload
                    if str(feature or "").strip()
                ]))
        except Exception:
            pass

    validation_path = os.path.join(feature_selection_dir, "consensus_validation.json")
    if not os.path.exists(validation_path):
        return []

    try:
        with open(validation_path, "r", encoding="utf-8") as handle:
            validation_payload = json.load(handle)
    except Exception:
        return []

    for candidate_key in (
        "consensus_features",
        "validated_consensus_features",
        "final_consensus_features",
        "selected_features",
        "features",
    ):
        candidate_value = validation_payload.get(candidate_key)
        if isinstance(candidate_value, list) and candidate_value:
            return list(dict.fromkeys([
                str(feature or "").strip()
                for feature in candidate_value
                if str(feature or "").strip()
            ]))

    return []


def _resolve_phase1_selected_feature_paths(state: Phase1State) -> Dict[str, str]:
    """Resolve canonical/legacy Phase 1 final dataset paths with config-aware priority."""
    cfg = get_config()
    io_policy = cfg.get_phase1_io_policy()
    prefer_new_read = bool(io_policy.get("prefer_new_read", False))
    fallback_old_read = bool(io_policy.get("fallback_old_read", True))

    canonical_path = (
        cfg.get_phase1_path("selected_features_csv")
        or "output/phase1/final/selected_features_final.csv"
    )
    legacy_path = (
        cfg.get_phase1_path("legacy_selected_features")
        or cfg.get_path("phase1_selected_features")
    )

    current_data_path = state.get("current_data_path", "")
    current_looks_final = (
        bool(current_data_path)
        and os.path.exists(current_data_path)
        and os.path.basename(current_data_path) == "selected_features_final.csv"
    )

    candidates = []
    if current_looks_final:
        candidates.append(current_data_path)

    if prefer_new_read:
        candidates.extend([canonical_path, legacy_path])
    else:
        candidates.append(legacy_path)
        if fallback_old_read:
            candidates.append(canonical_path)
        else:
            candidates.insert(0, canonical_path)

    active_path = canonical_path
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            active_path = candidate
            break

    return {
        "active": active_path,
        "canonical": canonical_path,
        "legacy": legacy_path,
    }


def _atomic_write_json(file_path: str, payload: Any) -> None:
    """Atomically write JSON to disk to avoid partial artifact updates."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(file_path),
        prefix=".tmp-",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temp_path, file_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _safe_json_payload(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default))


def _collect_phase1_protected_columns(
    context_variables: Dict[str, Any],
    target_column: Optional[str],
) -> List[str]:
    """Build the protected-column list used across preprocessing sync steps."""
    protected = [
        str(col) for col in (context_variables.get("protected_columns") or [])
        if str(col).strip()
    ]
    protected.extend([
        "Sample_ID",
        "sample_id",
        "SampleID",
        "sampleid",
        "ROW_ID",
        "row_id",
        "__row_id__",
        "target",
    ])
    if target_column:
        protected.append(str(target_column))
    return list(dict.fromkeys([col for col in protected if col]))


def _resolve_phase1_preprocessing_dir(
    context_variables: Dict[str, Any],
    current_data_path: str,
) -> Path:
    """Resolve the canonical directory that stores preprocessing QA artifacts."""
    context_report_path = str(context_variables.get("preprocessing_context_report_path", "") or "").strip()
    if context_report_path:
        return Path(context_report_path).resolve().parent
    if current_data_path:
        return Path(current_data_path).resolve().parent

    cfg = get_config()
    return Path(cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest").resolve()


def _copy_phase1_snapshot(source_path: str, output_dir: Path, snapshot_name: str) -> str:
    """Persist a stable preprocessing snapshot without mutating the source file."""
    if not source_path or not os.path.exists(source_path):
        return ""
    output_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source_path).resolve()
    ext = src.suffix or ".csv"
    snapshot_path = output_dir / f"{snapshot_name}{ext}"
    shutil.copy2(str(src), str(snapshot_path))
    return str(snapshot_path)


def _write_legacy_preprocessing_stage_reports(
    step_id: str,
    context_variables: Dict[str, Any],
    current_data_path: str,
    data_summary: Dict[str, Any],
) -> Dict[str, str]:
    """
    Materialize lightweight reports for legacy preprocessing steps 1.2/1.3/1.4.

    These steps still run through legacy code generation, but their facts should
    be reflected in the new preprocessing QA artifacts consumed by Phase 4.
    """
    if step_id not in {"1.2", "1.3", "1.4"}:
        return {}
    if not current_data_path or not os.path.exists(current_data_path):
        return {}

    output_dir = _resolve_phase1_preprocessing_dir(context_variables, current_data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_tabular_file(current_data_path)
    target_column = data_summary.get("target_column")
    protected_columns = _collect_phase1_protected_columns(context_variables, target_column)
    feature_columns = [
        col for col in df.columns
        if col not in protected_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    report_paths: Dict[str, str] = {}

    if step_id == "1.2":
        sample_column = next(
            (col for col in ["Sample_ID", "sample_id", "SampleID", "sampleid"] if col in df.columns),
            None,
        )
        pqn_snapshot_path = _copy_phase1_snapshot(current_data_path, output_dir, "stage1_2_pqn_only")
        normalization_report_path = output_dir / "normalization_decision_report.json"
        normalization_report = _safe_json_payload({
            "success": True,
            "stage_id": "1.2",
            "method": "PQN",
            "normalization_method": "PQN",
            "sample_column": sample_column,
            "output_path": current_data_path,
            "pqn_snapshot_path": pqn_snapshot_path,
            "normalized_columns": feature_columns,
            "normalized_columns_count": len(feature_columns),
            "protected_columns": protected_columns,
        })
        _atomic_write_json(str(normalization_report_path), normalization_report)
        report_paths["normalization_decision_report_path"] = str(normalization_report_path)
        if pqn_snapshot_path:
            report_paths["pqn_snapshot_path"] = pqn_snapshot_path

    elif step_id == "1.3":
        log2_snapshot_path = _copy_phase1_snapshot(current_data_path, output_dir, "stage1_3_log2_only")
        transformation_report_path = output_dir / "transformation_report.json"
        transformation_report = _safe_json_payload({
            "success": True,
            "stage_id": "1.3",
            "method": "log2",
            "transformation_method": "log2",
            "output_path": current_data_path,
            "log2_snapshot_path": log2_snapshot_path,
            "transformed_columns": feature_columns,
            "transformed_columns_count": len(feature_columns),
            "protected_columns": protected_columns,
        })
        _atomic_write_json(str(transformation_report_path), transformation_report)
        report_paths["transformation_report_path"] = str(transformation_report_path)
        if log2_snapshot_path:
            report_paths["log2_snapshot_path"] = log2_snapshot_path

        final_qc_snapshot_path = _copy_phase1_snapshot(current_data_path, output_dir, "stage1_3_final_qc")
        outlier_report_path = output_dir / "outlier_audit_report.json"
        outlier_report = _safe_json_payload({
            "success": True,
            "stage_id": "1.3",
            "method": "iqr",
            "outlier_method": "iqr_cap",
            "output_path": current_data_path,
            "final_qc_snapshot_path": final_qc_snapshot_path,
            "capped_columns": feature_columns,
            "capped_columns_count": len(feature_columns),
            "protected_columns": protected_columns,
        })
        _atomic_write_json(str(outlier_report_path), outlier_report)
        report_paths["outlier_audit_report_path"] = str(outlier_report_path)
        if final_qc_snapshot_path:
            report_paths["final_qc_snapshot_path"] = final_qc_snapshot_path

    elif step_id == "1.4":
        integrity_report_path = output_dir / "integrity_check_report.json"
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_values = numeric_df.to_numpy(dtype=float, copy=False) if not numeric_df.empty else np.empty((0, 0))
        nan_count = int(df.isna().sum().sum())
        inf_count = int(np.isinf(numeric_values).sum()) if numeric_values.size else 0

        sample_column = next(
            (col for col in ["Sample_ID", "sample_id", "SampleID", "sampleid"] if col in df.columns),
            None,
        )
        required_protected = [col for col in [sample_column, target_column] if col]
        protected_columns_status: Dict[str, str] = {}
        issues: List[str] = []

        for col in required_protected:
            if col not in df.columns:
                protected_columns_status[col] = "MISSING"
                issues.append(f"Protected column missing: {col}")
                continue
            if bool(df[col].isna().any()):
                protected_columns_status[col] = "HAS_NULL"
                issues.append(f"Protected column contains null values: {col}")
                continue
            if col == sample_column and bool(df[col].duplicated().any()):
                protected_columns_status[col] = "DUPLICATED"
                issues.append(f"Protected column contains duplicated values: {col}")
                continue
            if col == target_column:
                observed = set(df[col].dropna().tolist())
                if observed and not observed.issubset({0, 1}):
                    protected_columns_status[col] = "INVALID_LABELS"
                    issues.append(f"Protected target column has unexpected labels: {sorted(observed)}")
                    continue
            protected_columns_status[col] = "OK"

        if nan_count:
            issues.append(f"Detected {nan_count} missing values after integrity check.")
        if inf_count:
            issues.append(f"Detected {inf_count} infinite values after integrity check.")

        integrity_report = _safe_json_payload({
            "success": True,
            "stage_id": "1.4",
            "input_path": current_data_path,
            "total_rows": int(df.shape[0]),
            "total_cols": int(df.shape[1]),
            "checks_passed": not issues,
            "issues": issues,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "protected_columns_status": protected_columns_status,
            "protected_columns_intact": all(status == "OK" for status in protected_columns_status.values()),
            "value_stats": {
                "min_value": float(np.nanmin(numeric_values)) if numeric_values.size else None,
                "max_value": float(np.nanmax(numeric_values)) if numeric_values.size else None,
                "mean_value": float(np.nanmean(numeric_values)) if numeric_values.size else None,
            },
        })
        _atomic_write_json(str(integrity_report_path), integrity_report)
        report_paths["integrity_check_report_path"] = str(integrity_report_path)

    return report_paths


def _refresh_preprocessing_qa_after_legacy_step(
    step_id: str,
    context_variables: Dict[str, Any],
    current_data_path: str,
    data_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Refresh unified preprocessing artifacts after legacy Stage 1.2/1.3/1.4."""
    if step_id not in {"1.2", "1.3", "1.4"}:
        return context_variables

    context_report_path = str(context_variables.get("preprocessing_context_report_path", "") or "").strip()
    if not context_report_path or not os.path.exists(context_report_path):
        return context_variables

    try:
        stage_report_paths = _write_legacy_preprocessing_stage_reports(
            step_id=step_id,
            context_variables=context_variables,
            current_data_path=current_data_path,
            data_summary=data_summary,
        )

        qa_result = json.loads(preprocessing_qa_reporter_tool(
            context_report_path=context_report_path,
            zero_pattern_report_path=str(context_variables.get("zero_pattern_report_path", "") or ""),
            missingness_report_path=str(context_variables.get("missingness_report_path", "") or ""),
            imputation_report_path=str(context_variables.get("imputation_report_path", "") or ""),
            normalization_decision_report_path=str(
                stage_report_paths.get("normalization_decision_report_path")
                or context_variables.get("normalization_decision_report_path", "")
                or ""
            ),
            transformation_report_path=str(
                stage_report_paths.get("transformation_report_path")
                or context_variables.get("transformation_report_path", "")
                or ""
            ),
            outlier_audit_report_path=str(
                stage_report_paths.get("outlier_audit_report_path")
                or context_variables.get("outlier_audit_report_path", "")
                or ""
            ),
            feature_engineering_report_path=str(context_variables.get("feature_engineering_report_path", "") or ""),
            scaling_report_path=str(context_variables.get("scaling_report_path", "") or ""),
        ))
        if not qa_result.get("success"):
            print(f"[Preprocessing Sync] Skip QA refresh after step {step_id}: {qa_result.get('error')}")
            return context_variables

        refreshed_context = {
            **context_variables,
            **stage_report_paths,
            "preprocessing_report_path": qa_result.get("preprocessing_report_path", context_variables.get("preprocessing_report_path")),
            "preprocessing_summary_path": qa_result.get("preprocessing_summary_path", context_variables.get("preprocessing_summary_path")),
            "preprocessing_decision_pack_path": qa_result.get("preprocessing_decision_pack_path", context_variables.get("preprocessing_decision_pack_path")),
            "preprocessing_branch_used": qa_result.get("branch_used", context_variables.get("preprocessing_branch_used")),
            "phase4_reportable_facts": qa_result.get("phase4_reportable_facts", context_variables.get("phase4_reportable_facts", {})),
        }
        return refreshed_context
    except Exception as exc:
        print(f"[Preprocessing Sync] Failed to refresh QA after step {step_id}: {exc}")
        return context_variables


def _persist_phase1_training_results_artifact(training_results: Any) -> List[str]:
    """
    Persist the latest Stage 6 training results to canonical and legacy artifacts.

    This prevents stale `autogluon_training_results.json` files from surviving when
    Step 6.2 returns fresh results through the Magic Output Protocol but Step 6.3
    or downstream evaluators still read from disk.
    """
    if isinstance(training_results, str):
        training_results = json.loads(training_results)

    if not isinstance(training_results, dict):
        raise ValueError("training_results_json must be a dict or JSON string")

    cfg = get_config()
    candidate_paths = [
        os.path.join(
            cfg.get_phase1_path("artifacts_dir") or "output/phase1/artifacts",
            "autogluon_training_results.json",
        ),
        cfg.get_phase1_path("autogluon_results") or "data/autogluon_training_results.json",
    ]

    written_paths: List[str] = []
    seen_paths = set()
    for candidate in candidate_paths:
        if not candidate:
            continue
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in seen_paths:
            continue
        _atomic_write_json(abs_candidate, training_results)
        written_paths.append(abs_candidate)
        seen_paths.add(abs_candidate)
        print(f"[Artifact Sync] ✅ AutoGluon results persisted: {abs_candidate}")

    return written_paths


def execute_programmatic_stage11(state: Phase1State) -> Phase1State:
    """
    Deterministic implementation for Stage 1.1.

    This replaces fragile LLM-generated code for the most structured preprocessing
    step: zero-to-NaN conversion, missing-rate filtering, imputation, and report
    refresh.
    """
    if state.get("completed"):
        return state

    current_step = get_current_step(state)
    step_id = str(current_step.get("step_id", current_step.get("id", "unknown"))) if current_step else "unknown"
    if step_id != "1.1":
        return state

    try:
        current_data_path = state.get("current_data_path", "")
        if not current_data_path or not os.path.exists(current_data_path):
            raise FileNotFoundError(f"Stage 1.1 input not found: {current_data_path}")

        cfg = get_config()
        latest_dir = cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest"
        os.makedirs(latest_dir, exist_ok=True)

        context_vars = dict(state.get("context_variables", {}))
        data_summary = state.get("data_summary", {})
        target_column = data_summary.get("target_column")
        protected_columns = list(dict.fromkeys(
            [str(col) for col in (context_vars.get("protected_columns") or []) if str(col).strip()] +
            [str(col) for col in ["Sample_ID", "sample_id", "SampleID", "sampleid", "ROW_ID", "row_id", "__row_id__", target_column] if col]
        ))

        zero_handling_mode = str(
            context_vars.get("zero_handling_mode_used")
            or context_vars.get("zero_handling_strategy")
            or "keep_zero"
        )
        recommended_imputation_family = str(
            context_vars.get("recommended_imputation_family") or "median"
        ).strip() or "median"
        recommended_thresholds = context_vars.get("recommended_missingness_thresholds", {}) or {}
        feature_missing_threshold = float(recommended_thresholds.get("feature_missing_rate_threshold", 0.5))
        sample_missing_threshold = float(recommended_thresholds.get("sample_missing_rate_threshold", 0.35))

        df = _read_tabular_file(current_data_path)
        original_shape = tuple(df.shape)
        feature_columns = [
            col for col in df.columns
            if col not in protected_columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        zeros_converted = 0
        if zero_handling_mode == "convert_zero_to_nan":
            for col in feature_columns:
                zero_count = int((df[col] == 0).sum())
                if zero_count > 0:
                    df.loc[df[col] == 0, col] = pd.NA
                    zeros_converted += zero_count

        feature_missing_rates = df[feature_columns].isna().mean() if feature_columns else pd.Series(dtype=float)
        high_missing_features = [
            col for col, rate in feature_missing_rates.items()
            if float(rate) > feature_missing_threshold
        ]
        if high_missing_features:
            df = df.drop(columns=high_missing_features)

        remaining_feature_columns = [
            col for col in df.columns
            if col not in protected_columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        sample_missing_rates = df[remaining_feature_columns].isna().mean(axis=1) if remaining_feature_columns else pd.Series(dtype=float)
        high_missing_row_indices = [
            int(idx) for idx, rate in sample_missing_rates.items()
            if float(rate) > sample_missing_threshold
        ]
        if high_missing_row_indices:
            df = df.drop(index=high_missing_row_indices).reset_index(drop=True)

        remaining_feature_columns = [
            col for col in df.columns
            if col not in protected_columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        columns_needing_imputation = [
            col for col in remaining_feature_columns
            if bool(df[col].isna().any())
        ]

        imputation_method_used = "none_needed"
        if columns_needing_imputation:
            method_map = {
                "QRILC": "qrilc",
                "KNN": "knn",
                "MEDIAN": "median",
                "MEAN": "mean",
            }
            preferred_method = method_map.get(recommended_imputation_family.upper(), "median")
            result = None
            fallback_chain = [preferred_method]
            if preferred_method != "median":
                fallback_chain.append("median")

            last_error = None
            for method in fallback_chain:
                result_json = impute_missing_tool(
                    columns=columns_needing_imputation,
                    data_path=_persist_stage11_temp_input(df, latest_dir),
                    method=method,
                )
                result = json.loads(result_json)
                if result.get("success"):
                    df = _read_tabular_file(result["output_path"])
                    imputation_method_used = str(result.get("method", method))
                    last_error = None
                    break
                last_error = result.get("error")

            if last_error:
                raise ValueError(f"Stage 1.1 imputation failed: {last_error}")

        imputed_output_path = os.path.join(latest_dir, "stage1_1_imputed.csv")
        if current_data_path.endswith((".xlsx", ".xls")):
            df.to_excel(imputed_output_path, index=False)
        else:
            df.to_csv(imputed_output_path, index=False)

        imputation_report_path = os.path.join(latest_dir, "imputation_report.json")
        imputation_report = _safe_json_payload({
            "success": True,
            "stage_id": "1.1",
            "input_path": current_data_path,
            "output_path": imputed_output_path,
            "original_shape": list(original_shape),
            "final_shape": list(df.shape),
            "protected_columns": protected_columns,
            "zero_handling_mode_used": zero_handling_mode,
            "zeros_converted_to_missing_count": int(zeros_converted),
            "feature_missing_rate_threshold": float(feature_missing_threshold),
            "sample_missing_rate_threshold": float(sample_missing_threshold),
            "removed_features": high_missing_features,
            "removed_feature_count": len(high_missing_features),
            "removed_row_indices": high_missing_row_indices,
            "removed_row_count": len(high_missing_row_indices),
            "columns_needing_imputation": columns_needing_imputation,
            "columns_imputed_count": len(columns_needing_imputation),
            "recommended_imputation_family": recommended_imputation_family,
            "imputation_method_used": imputation_method_used,
            "remaining_missing_values": int(df[remaining_feature_columns].isna().sum().sum()) if remaining_feature_columns else 0,
        })
        _atomic_write_json(imputation_report_path, imputation_report)

        # Keep the legacy missing-value metadata artifact aligned with the new
        # deterministic Stage 1.1 output so downstream consumers do not read
        # stale "none_needed" facts from previous pipelines.
        normalized_imputation_method = str(imputation_method_used or "").strip()
        if normalized_imputation_method.lower() == "qrilc":
            normalized_imputation_method = "QRILC"
        elif normalized_imputation_method.lower() == "knn":
            normalized_imputation_method = "KNN"
        elif normalized_imputation_method.lower() == "median":
            normalized_imputation_method = "median"
        elif normalized_imputation_method.lower() == "mean":
            normalized_imputation_method = "mean"
        elif not normalized_imputation_method:
            normalized_imputation_method = "none_needed"

        missing_value_metadata_path = os.path.join(latest_dir, "missing_value_metadata.json")
        missing_value_metadata = _safe_json_payload({
            "step": "1.1_missing_value_strategy",
            "input_shape": list(original_shape),
            "output_shape": list(df.shape),
            "features_removed_high_missing": len(high_missing_features),
            "samples_removed_high_missing": len(high_missing_row_indices),
            "columns_removed": len(high_missing_features),
            "rows_removed": len(high_missing_row_indices),
            "columns_with_missing": len(columns_needing_imputation),
            "columns_imputed": len(columns_needing_imputation),
            "imputation_method": normalized_imputation_method,
            "mnar_detected": str(recommended_imputation_family).strip().upper() == "QRILC",
            "final_missing_values": int(df[remaining_feature_columns].isna().sum().sum()) if remaining_feature_columns else 0,
            "steps": [
                {
                    "action": "zero_to_missing",
                    "strategy": zero_handling_mode,
                    "zero_converted_to_missing_count": int(zeros_converted),
                },
                {
                    "action": "mnar_assessment",
                    "columns_with_missing": len(columns_needing_imputation),
                    "recommended_imputation_family": recommended_imputation_family,
                },
                {
                    "action": "imputation",
                    "method": normalized_imputation_method,
                    "columns_imputed": len(columns_needing_imputation),
                    "remaining_missing_values": int(df[remaining_feature_columns].isna().sum().sum()) if remaining_feature_columns else 0,
                },
            ],
        })
        _atomic_write_json(missing_value_metadata_path, missing_value_metadata)

        preprocessing_context_report_path = context_vars.get("preprocessing_context_report_path")
        zero_pattern_report_path = context_vars.get("zero_pattern_report_path", "")
        missingness_report_path = context_vars.get("missingness_report_path", "")
        qa_result = {}
        if preprocessing_context_report_path and os.path.exists(preprocessing_context_report_path):
            qa_result = json.loads(preprocessing_qa_reporter_tool(
                context_report_path=preprocessing_context_report_path,
                zero_pattern_report_path=zero_pattern_report_path,
                missingness_report_path=missingness_report_path,
                imputation_report_path=imputation_report_path,
            ))
            if not qa_result.get("success"):
                raise ValueError(f"preprocessing_qa_reporter_tool failed after Stage 1.1: {qa_result.get('error')}")

        state_updates = {
            "current_data_path": imputed_output_path,
            "imputation_report_path": imputation_report_path,
            "preprocessing_report_path": qa_result.get("preprocessing_report_path", context_vars.get("preprocessing_report_path")),
            "preprocessing_summary_path": qa_result.get("preprocessing_summary_path", context_vars.get("preprocessing_summary_path")),
            "preprocessing_decision_pack_path": qa_result.get("preprocessing_decision_pack_path", context_vars.get("preprocessing_decision_pack_path")),
            "zero_handling_mode_used": zero_handling_mode,
            "recommended_imputation_family": recommended_imputation_family,
            "imputation_method_used": imputation_method_used,
            "missing_value_metadata_path": missing_value_metadata_path,
            "removed_feature_count_stage11": len(high_missing_features),
            "removed_row_count_stage11": len(high_missing_row_indices),
        }

        execution_record = ExecutionRecord(
            step_id="1.1",
            generated_code="# Programmatic Stage 1.1 execution",
            stdout=json.dumps({"__METABO_UPDATE__": _safe_json_payload(state_updates)}, ensure_ascii=False),
            stderr="",
            success=True,
            execution_time=0.0,
            state_updates=state_updates,
        )

        updated_context = {**context_vars, **state_updates}
        if qa_result.get("phase4_reportable_facts"):
            updated_context["phase4_reportable_facts"] = qa_result["phase4_reportable_facts"]

        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "context_variables": updated_context,
            "current_data_path": imputed_output_path,
            "data_summary": get_data_summary(imputed_output_path, target_column),
            "retry_count": 0,
            "error": None,
            "last_error": None,
        }
    except Exception as e:
        execution_record = ExecutionRecord(
            step_id="1.1",
            generated_code="# Programmatic Stage 1.1 execution",
            stdout="",
            stderr=str(e),
            success=False,
            execution_time=0.0,
            state_updates=None,
        )
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "error": f"Programmatic Stage 1.1 failed: {str(e)}",
            "last_error": f"Programmatic Stage 1.1 failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "completed": False,
        }


def execute_programmatic_step50(state: Phase1State) -> Phase1State:
    current_step = get_current_step(state)
    step_id = str(current_step.get("step_id", current_step.get("id", "unknown"))) if current_step else "unknown"
    if step_id != "5.0":
        return state

    try:
        cfg = get_config()
        context_vars = dict(state.get("context_variables", {}))

        train_pool_path = str(state.get("current_data_path", "") or "").strip()
        if not train_pool_path or not os.path.exists(train_pool_path):
            raise FileNotFoundError(f"Step 5.0 input not found: {train_pool_path}")

        df = _read_tabular_file(train_pool_path)
        if "__row_id__" in df.columns:
            df = df.drop(columns=["__row_id__"])

        target_column = (
            str(context_vars.get("target_column") or state.get("data_summary", {}).get("target_column") or "").strip()
        )
        if target_column not in df.columns:
            for candidate in ("group", "Group", "target"):
                if candidate in df.columns:
                    target_column = candidate
                    break
        if target_column not in df.columns:
            raise ValueError("Step 5.0 could not resolve a target column (expected one of group/Group/target)")

        holdout_data_path = str(context_vars.get("holdout_data_path") or "").strip()
        train_pool_data_path = str(context_vars.get("train_pool_data_path") or "").strip()
        final_dir = cfg.get_phase1_path("final_dir") or "output/phase1/final"
        os.makedirs(final_dir, exist_ok=True)
        default_train_path = os.path.join(final_dir, "selected_features_train_pool.csv")
        default_holdout_path = os.path.join(final_dir, "selected_features_holdout_pool.csv")

        if not train_pool_data_path:
            train_pool_data_path = train_pool_path
        if not holdout_data_path:
            holdout_data_path = default_holdout_path

        if (
            os.path.abspath(train_pool_path) != os.path.abspath(default_train_path)
            and not context_vars.get("holdout_split_created")
            and not os.path.exists(holdout_data_path)
        ):
            from sklearn.model_selection import train_test_split
            from src.tools.domain.feature_generator import generate_train_only_engineered_feature_bundle

            split_source_path = (
                str(context_vars.get("pre_engineering_data_path") or "").strip()
                or str(context_vars.get("unscaled_enriched_data_path") or "").strip()
                or train_pool_path
            )
            if not os.path.exists(split_source_path):
                split_source_path = train_pool_path
            split_source_df = _read_tabular_file(split_source_path)
            if "__row_id__" in split_source_df.columns:
                split_source_df = split_source_df.drop(columns=["__row_id__"])

            if len(split_source_df) < 20:
                raise ValueError(f"Step 5.0 cannot create holdout split: dataset too small ({len(split_source_df)} rows)")

            unique_labels = split_source_df[target_column].dropna().unique().tolist()
            if len(unique_labels) < 2:
                raise ValueError(
                    f"Step 5.0 cannot create holdout split: target column '{target_column}' has <2 classes: {unique_labels}"
                )

            train_df, holdout_df = train_test_split(
                split_source_df,
                test_size=0.2,
                stratify=split_source_df[target_column],
                random_state=int(os.environ.get("METABOAGENT_RANDOM_STATE", "42")),
            )
            train_df = train_df.reset_index(drop=True)
            holdout_df = holdout_df.reset_index(drop=True)

            latest_dir = cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest"
            engineered_dir = os.path.join(latest_dir, "engineered")
            os.makedirs(engineered_dir, exist_ok=True)
            train_raw_path = os.path.join(engineered_dir, "train_pool_pre_engineering.csv")
            holdout_raw_path = os.path.join(engineered_dir, "holdout_pool_pre_engineering.csv")
            train_only_bundle_dir = os.path.join(engineered_dir, "train_only_bundle")
            train_only_registry_path = cfg.get_phase1_path("engineered_feature_registry") or os.path.join(
                engineered_dir,
                "engineered_feature_registry.json",
            )
            _write_tabular_file(train_df, train_raw_path)
            _write_tabular_file(holdout_df, holdout_raw_path)

            engineering_result = generate_train_only_engineered_feature_bundle(
                train_data_path=train_raw_path,
                holdout_data_path=holdout_raw_path,
                output_dir=train_only_bundle_dir,
                phase0_output=context_vars.get("phase0_output"),
                registry_path=train_only_registry_path,
                rules_path=context_vars.get("engineered_feature_rules_path"),
            )
            if not isinstance(engineering_result, dict) or engineering_result.get("status") != "success":
                raise ValueError(f"Step 5.0 train-only engineered feature bundle failed: {engineering_result}")

            train_df = _read_tabular_file(engineering_result["train_merged_path"])
            holdout_df = _read_tabular_file(engineering_result["holdout_merged_path"])

            protected_candidates = [
                "Sample_ID",
                "sample_id",
                "SampleID",
                "sampleid",
                "ROW_ID",
                "row_id",
                "__row_id__",
                "ID",
                "id",
                target_column,
            ]
            protected_cols_for_split = [col for col in protected_candidates if col in train_df.columns]
            train_df, holdout_df, scaling_stats = _apply_train_fitted_standard_scaling(
                train_df=train_df,
                holdout_df=holdout_df,
                protected_cols=protected_cols_for_split,
            )

            _write_tabular_file(train_df, default_train_path)
            _write_tabular_file(holdout_df, default_holdout_path)
            train_pool_data_path = default_train_path
            holdout_data_path = default_holdout_path
            df = train_df
            context_vars["step50_split_source_path"] = split_source_path
            context_vars["train_fit_scaling_applied"] = True
            context_vars["train_fit_scaling_method"] = "standard"
            context_vars["train_fit_scaling_feature_count"] = int(len(scaling_stats))
            context_vars["holdout_split_created"] = True
            context_vars["train_pool_pre_engineering_data_path"] = train_raw_path
            context_vars["holdout_pre_engineering_data_path"] = holdout_raw_path
            context_vars["train_pool_unscaled_engineered_data_path"] = engineering_result["train_merged_path"]
            context_vars["holdout_unscaled_engineered_data_path"] = engineering_result["holdout_merged_path"]
            context_vars["engineered_feature_registry_path"] = engineering_result["registry_path"]
            context_vars["train_only_engineering_summary_path"] = engineering_result["summary_path"]
            context_vars["n_ratio_features"] = int((engineering_result.get("counts", {}) or {}).get("train_ratio_features", 0) or 0)
            context_vars["n_sum_features"] = int((engineering_result.get("counts", {}) or {}).get("train_taxonomy_features", 0) or 0)
            context_vars["n_pathway_scores"] = int((engineering_result.get("counts", {}) or {}).get("train_pathway_features", 0) or 0)
            context_vars["engineered_features_merged"] = True
            context_vars["unscaled_enriched_data_path"] = engineering_result["train_merged_path"]

        protected_candidates = [
            "Sample_ID",
            "sample_id",
            "SampleID",
            "sampleid",
            "ROW_ID",
            "row_id",
            "__row_id__",
            "ID",
            "id",
            target_column,
        ]
        protected_cols = [col for col in protected_candidates if col in df.columns]
        feature_cols = [
            col for col in df.columns
            if col not in protected_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        class_counts_series = df[target_column].value_counts(dropna=False)
        class_counts = {str(k): int(v) for k, v in class_counts_series.items()}
        nonzero_counts = [v for v in class_counts.values() if v > 0]
        if len(nonzero_counts) < 2:
            raise ValueError(f"Step 5.0 invalid target distribution for '{target_column}': {class_counts}")

        class_ratio = float(max(nonzero_counts) / max(1, min(nonzero_counts)))
        imbalance_threshold = float(context_vars.get("imbalance_threshold", 2.0) or 2.0)
        is_balanced = bool(class_ratio <= imbalance_threshold)

        mandatory_features = [
            standardize_feature_name(feature)
            for feature in (context_vars.get("mandatory_features", []) or [])
            if str(feature or "").strip()
        ]
        feature_cols_set = {standardize_feature_name(col) for col in feature_cols if col}
        mandatory_features = [
            feature for feature in mandatory_features
            if feature and feature in feature_cols_set
        ]

        feature_selection_dir = os.path.join(
            cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest",
            "feature_selection",
        )
        os.makedirs(feature_selection_dir, exist_ok=True)

        candidate_universe = _build_phase1_candidate_universe(
            df=df,
            feature_cols=feature_cols,
            mandatory_features=mandatory_features,
        )
        candidate_universe_path = os.path.join(feature_selection_dir, "candidate_universe.csv")
        candidate_universe_metadata_path = os.path.join(feature_selection_dir, "candidate_universe_metadata.json")
        pd.DataFrame(candidate_universe["records"]).to_csv(candidate_universe_path, index=False)
        _atomic_write_json(
            candidate_universe_metadata_path,
            {
                "schema_version": "phase1.candidate_universe.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "target_column": target_column,
                "train_pool_data_path": train_pool_data_path,
                "holdout_data_path": holdout_data_path,
                "class_ratio": class_ratio,
                "is_balanced": bool(is_balanced),
                "summary": candidate_universe["summary"],
                "protected_anchor_mapping_report": context_vars.get("protected_anchor_mapping_report", {}),
                "phase0_prior_resolution_report": context_vars.get("phase0_prior_resolution_report", {}),
            },
        )

        selection_params = _resolve_phase1_feature_selection_params(
            context_vars=context_vars,
            n_features_in_pool=len(feature_cols),
            n_mandatory_features=len(mandatory_features),
        )
        selector_family = _resolve_phase1_selector_family(context_vars)

        state_updates = {
            "current_data_path": train_pool_data_path,
            "train_pool_data_path": train_pool_data_path,
            "holdout_data_path": holdout_data_path,
            "feature_pool_path": candidate_universe_path,
            "candidate_universe_path": candidate_universe_path,
            "candidate_universe_metadata_path": candidate_universe_metadata_path,
            "n_features_in_pool": int(len(feature_cols)),
            "target_column": target_column,
            "is_balanced": is_balanced,
            "class_ratio": class_ratio,
            "class_counts": class_counts,
            "mandatory_features": mandatory_features,
            "n_mandatory_features": int(len(mandatory_features)),
            "phase1_selector_family": selector_family,
            **selection_params,
            "step50_split_source_path": str(context_vars.get("step50_split_source_path") or train_pool_data_path),
            "train_fit_scaling_applied": bool(context_vars.get("train_fit_scaling_applied", False)),
            "train_fit_scaling_method": str(context_vars.get("train_fit_scaling_method") or ""),
            "train_fit_scaling_feature_count": int(context_vars.get("train_fit_scaling_feature_count", 0) or 0),
        }

        execution_record = ExecutionRecord(
            step_id="5.0",
            generated_code="# Programmatic Step 5.0 execution",
            stdout=json.dumps({"__METABO_UPDATE__": _safe_json_payload(state_updates)}, ensure_ascii=False),
            stderr="",
            success=True,
            execution_time=0.0,
            state_updates=state_updates,
        )

        updated_context = {**context_vars, **state_updates}
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "context_variables": updated_context,
            "current_data_path": train_pool_data_path,
            "data_summary": get_data_summary(train_pool_data_path, target_column),
            "retry_count": 0,
            "error": None,
            "last_error": None,
        }
    except Exception as e:
        execution_record = ExecutionRecord(
            step_id="5.0",
            generated_code="# Programmatic Step 5.0 execution",
            stdout="",
            stderr=str(e),
            success=False,
            execution_time=0.0,
            state_updates=None,
        )
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "error": f"Programmatic Step 5.0 failed: {str(e)}",
            "last_error": f"Programmatic Step 5.0 failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "completed": False,
        }


def execute_programmatic_stage154(state: Phase1State) -> Phase1State:
    """Deterministic Stage 1.5.4 merge of the three engineered feature tables.

    The merge is pure file orchestration and therefore does not need an LLM.
    Keeping it deterministic prevents generated snippets from losing the
    ``pre_engineering_data_path`` binding or counting join keys incorrectly.
    """
    current_step = get_current_step(state)
    step_id = str(current_step.get("step_id", current_step.get("id", "unknown"))) if current_step else "unknown"
    if step_id != "1.5.4":
        return state
    try:
        context_vars = dict(state.get("context_variables", {}) or {})
        cfg = get_config()
        latest_dir = cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest"
        engineered_dir = os.path.join(latest_dir, "engineered")
        bundle_dir = os.path.join(engineered_dir, "train_only_bundle")
        os.makedirs(engineered_dir, exist_ok=True)

        def _first_path(keys, candidates):
            for key in keys:
                value = str(context_vars.get(key) or "").strip()
                if value and os.path.exists(value):
                    return value
            for value in candidates:
                if value and os.path.exists(value):
                    return value
            return ""

        base_path = _first_path(
            ("pre_engineering_data_path", "unscaled_enriched_data_path", "current_data_path"),
            (str(state.get("current_data_path") or ""), os.path.join(latest_dir, "stage1_3_final_qc.csv")),
        )
        ratio_path = _first_path(
            ("temp_ratio_features_path", "ratio_features_path", "train_temp_ratio_path"),
            (os.path.join(bundle_dir, "train_temp_ratios.csv"), os.path.join(engineered_dir, "temp_ratios.csv"), os.path.join(latest_dir, "temp_ratios.csv")),
        )
        taxonomy_path = _first_path(
            ("temp_taxonomy_features_path", "taxonomy_features_path", "train_temp_taxonomy_path"),
            (os.path.join(bundle_dir, "train_temp_taxonomy_sums.csv"), os.path.join(engineered_dir, "temp_taxonomy_sums.csv"), os.path.join(latest_dir, "temp_taxonomy_sums.csv")),
        )
        pathway_path = _first_path(
            ("temp_pathway_scores_path", "pathway_scores_path", "train_temp_pathway_path"),
            (os.path.join(bundle_dir, "train_temp_pathway_scores.csv"), os.path.join(engineered_dir, "temp_pathway_scores.csv"), os.path.join(latest_dir, "temp_pathway_scores.csv")),
        )
        if not base_path:
            raise FileNotFoundError("Step 1.5.4 could not resolve the base pre-engineering matrix")
        output_path = os.path.join(engineered_dir, "data_with_engineered_features.csv")
        metadata_path = os.path.join(engineered_dir, "feature_metadata.json")
        from src.tools.domain.feature_generator import merge_engineered_features
        result = merge_engineered_features(
            base_data_path=base_path,
            ratio_path=ratio_path or os.path.join(bundle_dir, "train_temp_ratios.csv"),
            taxonomy_path=taxonomy_path or os.path.join(bundle_dir, "train_temp_taxonomy_sums.csv"),
            pathway_path=pathway_path or os.path.join(bundle_dir, "train_temp_pathway_scores.csv"),
            output_path=output_path,
            metadata_path=metadata_path,
        )
        if not isinstance(result, dict) or result.get("status") not in ("success", "ok"):
            raise RuntimeError(f"merge_engineered_features failed: {result}")
        merged_path = str(result.get("output_path") or output_path)
        breakdown = dict(result.get("feature_breakdown") or {})
        updates = {
            "pre_engineering_data_path": base_path,
            "unscaled_enriched_data_path": merged_path,
            "engineered_data_path": merged_path,
            "engineered_features_merged": True,
            "engineered_feature_metadata_path": str(result.get("metadata_path") or metadata_path),
            "n_ratio_features": int(breakdown.get("ratio", 0) or 0),
            "n_sum_features": int(breakdown.get("taxonomy_sum", 0) or 0),
            "n_pathway_scores": int(breakdown.get("pathway_score", 0) or 0),
            "feature_breakdown": breakdown,
        }
        record = ExecutionRecord(step_id="1.5.4", generated_code="# Programmatic Stage 1.5.4 merge", stdout=json.dumps({"__METABO_UPDATE__": updates}), stderr="", success=True, execution_time=0.0, state_updates=updates)
        return {**state, "execution_history": state.get("execution_history", []) + [record], "context_variables": {**context_vars, **updates}, "current_data_path": merged_path, "data_summary": get_data_summary(merged_path, str(context_vars.get("target_column") or state.get("data_summary", {}).get("target_column") or "group")), "retry_count": 0, "error": None, "last_error": None}
    except Exception as exc:
        msg = f"Programmatic Step 1.5.4 failed: {exc}"
        record = ExecutionRecord(step_id="1.5.4", generated_code="# Programmatic Stage 1.5.4 merge", stdout="", stderr=str(exc), success=False, execution_time=0.0, state_updates=None)
        return {**state, "execution_history": state.get("execution_history", []) + [record], "error": msg, "last_error": msg, "retry_count": state.get("retry_count", 0) + 1, "completed": False}


def execute_programmatic_step521(state: Phase1State) -> Phase1State:
    current_step = get_current_step(state)
    step_id = str(current_step.get("step_id", current_step.get("id", "unknown"))) if current_step else "unknown"
    if step_id != "5.2.1":
        return state
    return _run_unified_phase1_stability_selection(
        state,
        step_id="5.2.1",
        expected_balance=None,
    )


def _expand_feature_aliases_from_name_mapping(
    features: set[str],
    name_mapping: Dict[str, Any],
) -> tuple[set[str], Dict[str, str]]:
    """Expand a feature set with verified original-name/HMDB equivalents."""
    expanded = {
        standardize_feature_name(feature)
        for feature in features
        if str(feature or "").strip()
    }
    matched_aliases: Dict[str, str] = {}
    if not isinstance(name_mapping, dict):
        return expanded, matched_aliases

    normalized_pairs = []
    for source, target in name_mapping.items():
        if not str(source or "").strip() or not str(target or "").strip():
            continue
        normalized_pairs.append((
            standardize_feature_name(source),
            standardize_feature_name(target),
        ))

    # Mapping files are normally one-hop original-name -> HMDB mappings. Two
    # passes also cover a harmless chained alias without accepting an unrelated
    # feature: at least one endpoint must already be in the allowed set.
    for _ in range(2):
        for source, target in normalized_pairs:
            if source in expanded or target in expanded:
                expanded.update((source, target))
                matched_aliases[source] = target
    return expanded, matched_aliases


def execute_programmatic_step522(state: Phase1State) -> Phase1State:
    """
    Deterministic implementation for Step 5.2.2 stability audit.

    This replaces fragile LLM-generated audit code with a canonical consistency
    check across the unified train-only stability artifacts.
    """
    current_step = get_current_step(state)
    step_id = str(current_step.get("step_id", current_step.get("id", "unknown"))) if current_step else "unknown"
    if step_id != "5.2.2":
        return state

    try:
        context_vars = dict(state.get("context_variables", {}))
        cfg = get_config()
        feature_selection_dir = os.path.join(
            cfg.get_phase1_path("intermediate_latest_dir") or "output/phase1/intermediate/latest",
            "feature_selection",
        )
        os.makedirs(feature_selection_dir, exist_ok=True)

        summary_path = os.path.join(feature_selection_dir, "stability_selection_summary.json")
        stable_core_path = os.path.join(feature_selection_dir, "stable_core_features.json")
        final_panel_selection_path = os.path.join(feature_selection_dir, "final_panel_selection.json")
        temp_consensus_path = os.path.join(feature_selection_dir, "temp_consensus_features.json")
        stability_scores_path = os.path.join(feature_selection_dir, "stability_scores.json")
        candidate_universe_path = (
            str(context_vars.get("candidate_universe_path") or "").strip()
            or os.path.join(feature_selection_dir, "candidate_universe.csv")
        )
        audit_report_path = os.path.join(feature_selection_dir, "stability_audit_report.json")

        def _load_json(path: str, *, required: bool = True) -> Any:
            if not path or not os.path.exists(path):
                if required:
                    raise FileNotFoundError(f"Required Step 5.2.2 artifact not found: {path}")
                return None
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        def _normalize_feature_list(payload: Any, primary_key: str | None = None) -> List[str]:
            if isinstance(payload, list):
                return list(dict.fromkeys([
                    standardize_feature_name(feature)
                    for feature in payload
                    if str(feature or "").strip()
                ]))
            if isinstance(payload, dict):
                candidate_keys = []
                if primary_key:
                    candidate_keys.append(primary_key)
                candidate_keys.extend([
                    "final_panel_features",
                    "stable_core_features",
                    "consensus_features",
                    "features",
                    "selected_features",
                ])
                for key in candidate_keys:
                    value = payload.get(key)
                    if isinstance(value, list):
                        return list(dict.fromkeys([
                            standardize_feature_name(feature)
                            for feature in value
                            if str(feature or "").strip()
                        ]))
            return []

        stability_summary = _load_json(summary_path)
        stable_core_payload = _load_json(stable_core_path)
        final_panel_payload = _load_json(final_panel_selection_path)
        temp_consensus_payload = _load_json(temp_consensus_path)
        _ = _load_json(stability_scores_path, required=False)

        stable_core_features = _normalize_feature_list(stable_core_payload, primary_key="stable_core_features")
        final_panel_features = _normalize_feature_list(final_panel_payload, primary_key="final_panel_features")
        consensus_features = _normalize_feature_list(temp_consensus_payload, primary_key="consensus_features")
        mandatory_features = list(dict.fromkeys([
            standardize_feature_name(feature)
            for feature in (context_vars.get("mandatory_features", []) or [])
            if str(feature or "").strip()
        ]))

        if not consensus_features:
            raise ValueError("Step 5.2.2 audit could not load temp_consensus_features.json")
        if not final_panel_features:
            raise ValueError("Step 5.2.2 audit could not load final_panel_selection.json")

        universe_features: List[str] = []
        if candidate_universe_path and os.path.exists(candidate_universe_path):
            candidate_df = pd.read_csv(candidate_universe_path)
            if "standardized_name" in candidate_df.columns:
                universe_features = [
                    standardize_feature_name(feature)
                    for feature in candidate_df["standardized_name"].dropna().astype(str).tolist()
                    if str(feature).strip()
                ]
            elif "feature" in candidate_df.columns:
                universe_features = [
                    standardize_feature_name(feature)
                    for feature in candidate_df["feature"].dropna().astype(str).tolist()
                    if str(feature).strip()
                ]
        universe_features = list(dict.fromkeys(universe_features))
        base_allowed_core_features = set(universe_features) | set(mandatory_features)

        # Step 1 standardization may represent a raw metabolite by HMDB ID in
        # candidate_universe.csv while the stability selector reports the
        # original metabolite name. Treat only explicit pairs from the run's
        # train-name mapping as equivalent during the audit.
        name_mapping_path = str(
            context_vars.get("metabolite_name_mapping_path")
            or context_vars.get("train_name_mapping_path")
            or ""
        ).strip()
        name_mapping = context_vars.get("metabolite_name_mapping")
        if not isinstance(name_mapping, dict) or not name_mapping:
            name_mapping = {}
            if name_mapping_path and os.path.exists(name_mapping_path):
                loaded_mapping = _load_json(name_mapping_path, required=False)
                if isinstance(loaded_mapping, dict):
                    name_mapping = loaded_mapping
        allowed_core_features, matched_name_aliases = _expand_feature_aliases_from_name_mapping(
            base_allowed_core_features,
            name_mapping,
        )

        final_panel_matches = final_panel_features == consensus_features
        missing_mandatory = [
            feature for feature in mandatory_features
            if feature not in stable_core_features and feature not in final_panel_features
        ]
        missing_core_features = [
            feature for feature in stable_core_features
            if feature not in allowed_core_features
        ]

        audit_payload = {
            "paths": {
                "stability_summary_path": summary_path,
                "stable_core_path": stable_core_path,
                "final_panel_selection_path": final_panel_selection_path,
                "temp_consensus_path": temp_consensus_path,
                "stability_scores_path": stability_scores_path,
                "candidate_universe_path": candidate_universe_path,
                "metabolite_name_mapping_path": name_mapping_path,
            },
            "checks": {
                "final_panel_matches_temp_consensus": {
                    "expected_size": len(final_panel_features),
                    "final_panel_size": len(final_panel_features),
                    "consensus_panel_size": len(consensus_features),
                    "match": final_panel_matches,
                    "final_only_minus_consensus": sorted(set(final_panel_features) - set(consensus_features)),
                    "consensus_only_minus_final": sorted(set(consensus_features) - set(final_panel_features)),
                },
                "mandatory_features_retained": {
                    "n_mandatory": len(mandatory_features),
                    "missing_count": len(missing_mandatory),
                    "missing_mandatory": missing_mandatory,
                    "passed": len(missing_mandatory) == 0,
                },
                "stable_core_subset_of_universe_or_protected": {
                    "stable_core_size": len(stable_core_features),
                    "universe_size": len(universe_features),
                    "missing_count": len(missing_core_features),
                    "missing_core_features": missing_core_features,
                    "matched_name_aliases": matched_name_aliases,
                    "passed": len(missing_core_features) == 0,
                },
            },
            "warnings": [],
            "success": bool(final_panel_matches and not missing_mandatory and not missing_core_features),
        }

        # Copy stable-core thresholds and method summary into the audit artifact for traceability.
        if isinstance(stability_summary, dict):
            audit_payload["summary_snapshot"] = {
                "selection_thresholds": _safe_json_payload(stability_summary.get("selection_thresholds", {})),
                "methods_used": _safe_json_payload(stability_summary.get("methods_used", [])),
                "final_panel_target_size": _safe_json_payload(stability_summary.get("final_panel_target_size")),
            }

        _atomic_write_json(audit_report_path, audit_payload)

        if not audit_payload["success"]:
            raise ValueError(
                "Step 5.2.2 stability audit failed: "
                + json.dumps(audit_payload["checks"], ensure_ascii=False)
            )

        state_updates = {
            "stable_core_features": stable_core_features,
            "consensus_features": consensus_features,
            "final_selected_features": final_panel_features,
            "n_features_selected": int(len(consensus_features)),
            "stability_audit_report_path": audit_report_path,
            "stability_summary_path": summary_path,
            "stability_scores_path": stability_scores_path,
            "phase1_panel_selection_path": final_panel_selection_path,
        }

        execution_record = ExecutionRecord(
            step_id="5.2.2",
            generated_code="# Programmatic Step 5.2.2 execution",
            stdout=json.dumps({"__METABO_UPDATE__": _safe_json_payload(state_updates)}, ensure_ascii=False),
            stderr="",
            success=True,
            execution_time=0.0,
            state_updates=state_updates,
        )

        updated_context = {**context_vars, **state_updates}
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "context_variables": updated_context,
            "retry_count": 0,
            "error": None,
            "last_error": None,
        }
    except Exception as e:
        execution_record = ExecutionRecord(
            step_id="5.2.2",
            generated_code="# Programmatic Step 5.2.2 execution",
            stdout="",
            stderr=str(e),
            success=False,
            execution_time=0.0,
            state_updates=None,
        )
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "error": f"Programmatic Step 5.2.2 failed: {str(e)}",
            "last_error": f"Programmatic Step 5.2.2 failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "completed": False,
        }


def execute_programmatic_stage53(state: Phase1State) -> Phase1State:
    """
    Deterministic implementation for Step 5.3 final dataset creation.

    This replaces fragile LLM-generated finalization code with a canonical merge:
    - read the finalized single-panel features from in-memory state or temp file
    - union them with mandatory Phase 0 anchors
    - project the learned train feature list onto train/holdout datasets
    - enforce the minimum handoff size configured for the new stability pipeline
    """
    current_step = get_current_step(state)
    step_id = str(current_step.get("step_id", current_step.get("id", "unknown"))) if current_step else "unknown"
    if step_id != "5.3":
        return state

    try:
        context_vars = dict(state.get("context_variables", {}))
        train_pool_path = state.get("current_data_path", "")
        if not train_pool_path or not os.path.exists(train_pool_path):
            raise FileNotFoundError(f"Step 5.3 input not found: {train_pool_path}")

        cfg = get_config()
        final_dir = cfg.get_phase1_path("final_dir") or "output/phase1/final"
        os.makedirs(final_dir, exist_ok=True)

        target_column = (
            context_vars.get("target_column")
            or state.get("data_summary", {}).get("target_column")
            or "Group"
        )
        holdout_data_path = str(context_vars.get("holdout_data_path") or "").strip()
        min_final_feature_count = int(context_vars.get("phase1_min_final_feature_count", 11) or 11)

        train_df = _read_tabular_file(train_pool_path)
        context_vars = _merge_phase1_protected_anchor_context(
            context_vars,
            dataset_columns=list(train_df.columns),
            target_column=target_column,
        )

        passthrough_protected_cols = [
            col for col in [
                "Sample_ID",
                "sample_id",
                "SampleID",
                "sampleid",
                "ID",
                "id",
                "ROW_ID",
                "row_id",
                "__row_id__",
                target_column,
            ]
            if col in train_df.columns
        ]
        blocked_leakage_cols = []
        protected_cols = passthrough_protected_cols + blocked_leakage_cols

        excluded_feature_names = {
            standardize_feature_name(name)
            for name in ["ROW_ID", "row_id", "__row_id__", "ROW_ID_", "ID", "id"]
        }

        consensus_features = [
            standardize_feature_name(feature)
            for feature in _load_phase1_consensus_features(context_vars)
            if str(feature or "").strip()
        ]
        consensus_features = [
            feature for feature in consensus_features
            if feature and feature not in excluded_feature_names
        ]

        mandatory_features = [
            standardize_feature_name(feature)
            for feature in (context_vars.get("mandatory_features", []) or [])
            if str(feature or "").strip()
        ]
        mandatory_features = [
            feature for feature in mandatory_features
            if feature and feature not in excluded_feature_names
        ]

        if not consensus_features:
            raise ValueError(
                "Step 5.3 could not load consensus features from context_variables "
                "or temp_consensus_features.json"
            )

        requested_final_features = list(dict.fromkeys(consensus_features + mandatory_features))
        train_feature_map = {
            standardize_feature_name(col): col
            for col in train_df.columns
            if col not in protected_cols
        }
        available_train_features = [
            feature for feature in requested_final_features
            if feature in train_feature_map
        ]
        missing_train_features = [
            feature for feature in requested_final_features
            if feature not in train_feature_map
        ]

        if len(available_train_features) < min_final_feature_count:
            raise ValueError(
                "Step 5.3 final feature set violates the minimum feature rule (minimum handoff rule): "
                f"got {len(available_train_features)} features, expected >= {min_final_feature_count}. "
                f"Panel target={int(context_vars.get('phase1_final_panel_target_size', 0) or 0)}, "
                f"stable core={len(context_vars.get('stable_core_features', []) or [])}, "
                f"mandatory={len(mandatory_features)}, missing from train pool={len(missing_train_features)}"
            )

        train_selected_df = pd.DataFrame()
        for col in passthrough_protected_cols:
            train_selected_df[col] = train_df[col]
        for feature in available_train_features:
            train_selected_df[feature] = train_df[train_feature_map[feature]]

        selected_feature_paths = _resolve_phase1_selected_feature_paths(state)
        train_output_path = selected_feature_paths["canonical"]
        os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
        train_selected_df.to_csv(train_output_path, index=False)

        io_policy = cfg.get_phase1_io_policy()
        _mirror_phase1_final_dataset(
            source_path=train_output_path,
            canonical_path=selected_feature_paths["canonical"],
            legacy_path=selected_feature_paths["legacy"],
            dual_write_enabled=bool(io_policy.get("dual_write_enabled", False)),
        )

        holdout_output_path = ""
        missing_holdout_features: List[str] = []
        if holdout_data_path and os.path.exists(holdout_data_path):
            holdout_df = _read_tabular_file(holdout_data_path)
            holdout_passthrough_protected_cols = [
                col for col in [
                    "Sample_ID",
                    "sample_id",
                    "SampleID",
                    "sampleid",
                    "ID",
                    "id",
                    "ROW_ID",
                    "row_id",
                    "__row_id__",
                    target_column,
                ]
                if col in holdout_df.columns
            ]
            holdout_blocked_leakage_cols = []
            holdout_protected_cols = holdout_passthrough_protected_cols + holdout_blocked_leakage_cols
            holdout_feature_map = {
                standardize_feature_name(col): col
                for col in holdout_df.columns
                if col not in holdout_protected_cols
            }

            holdout_selected_df = pd.DataFrame()
            for col in holdout_passthrough_protected_cols:
                holdout_selected_df[col] = holdout_df[col]
            for feature in available_train_features:
                if feature in holdout_feature_map:
                    holdout_selected_df[feature] = holdout_df[holdout_feature_map[feature]]
                else:
                    holdout_selected_df[feature] = 0.0
                    missing_holdout_features.append(feature)

            holdout_output_path = os.path.join(final_dir, "selected_features_holdout.csv")
            holdout_selected_df.to_csv(holdout_output_path, index=False)

        final_selection_summary_path = _write_phase1_final_selection_summary(
            target_column=target_column,
            protected_cols=protected_cols,
            consensus_features=consensus_features,
            mandatory_features=mandatory_features,
            selected_features=available_train_features,
            missing_train_features=missing_train_features,
            missing_holdout_features=missing_holdout_features,
            train_pool_path=train_pool_path,
            train_output_path=train_output_path,
            holdout_data_path=holdout_data_path,
            holdout_output_path=holdout_output_path,
            selection_details={
                "phase1_final_panel_target_size": int(context_vars.get("phase1_final_panel_target_size", 0) or 0),
                "phase1_min_final_feature_count": int(min_final_feature_count),
                "phase1_selector_family": context_vars.get("phase1_selector_family", []),
                "phase1_stability_iterations": int(context_vars.get("phase1_stability_iterations", 0) or 0),
                "phase1_per_method_select_k": int(context_vars.get("phase1_per_method_select_k", 0) or 0),
                "phase1_selection_frequency_threshold": float(context_vars.get("phase1_selection_frequency_threshold", 0.0) or 0.0),
                "phase1_method_consensus_threshold": int(context_vars.get("phase1_method_consensus_threshold", 0) or 0),
                "stable_core_features": context_vars.get("stable_core_features", []),
                "stability_summary_path": context_vars.get("stability_summary_path", ""),
                "stability_scores_path": context_vars.get("stability_scores_path", ""),
                "candidate_universe_path": context_vars.get("candidate_universe_path", ""),
                "candidate_universe_metadata_path": context_vars.get("candidate_universe_metadata_path", ""),
            },
        )

        context_vars["consensus_features"] = available_train_features
        context_vars["final_selected_features"] = available_train_features
        context_vars["mandatory_features"] = list(dict.fromkeys(mandatory_features))
        context_vars["n_features_selected"] = len(available_train_features)
        context_vars["n_final_selected_features"] = len(available_train_features)
        context_vars["selected_data_path"] = train_output_path
        context_vars["selected_dataset_path"] = train_output_path
        context_vars["selected_holdout_data_path"] = holdout_output_path
        context_vars["final_selection_summary_path"] = final_selection_summary_path
        context_vars["feature_selection_finalized"] = True
        context_vars["step_5_3_completed"] = True
        context_vars["ready_for_ml_pipeline"] = True

        _sync_phase1_consensus_artifacts(context_vars)

        state_updates = {
            "selected_data_path": train_output_path,
            "selected_dataset_path": train_output_path,
            "current_data_path": train_output_path,
            "selected_holdout_data_path": holdout_output_path,
            "final_selection_summary_path": final_selection_summary_path,
            "n_final_selected_features": len(available_train_features),
            "final_selected_features": available_train_features,
            "consensus_features": available_train_features,
            "mandatory_features": context_vars["mandatory_features"],
            "feature_selection_finalized": True,
            "step_5_3_completed": True,
            "ready_for_ml_pipeline": True,
            "missing_train_features_during_finalization": missing_train_features,
            "missing_holdout_features_filled_with_zero": missing_holdout_features,
        }

        execution_record = ExecutionRecord(
            step_id="5.3",
            generated_code="# Programmatic Step 5.3 execution",
            stdout=json.dumps({"__METABO_UPDATE__": _safe_json_payload(state_updates)}, ensure_ascii=False),
            stderr="",
            success=True,
            execution_time=0.0,
            state_updates=state_updates,
        )

        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "context_variables": context_vars,
            "current_data_path": train_output_path,
            "data_summary": get_data_summary(train_output_path, target_column),
            "retry_count": 0,
            "error": None,
            "last_error": None,
        }
    except Exception as e:
        execution_record = ExecutionRecord(
            step_id="5.3",
            generated_code="# Programmatic Step 5.3 execution",
            stdout="",
            stderr=str(e),
            success=False,
            execution_time=0.0,
            state_updates=None,
        )
        return {
            **state,
            "execution_history": state.get("execution_history", []) + [execution_record],
            "error": f"Programmatic Step 5.3 failed: {str(e)}",
            "last_error": f"Programmatic Step 5.3 failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "completed": False,
        }


def _persist_stage11_temp_input(df: pd.DataFrame, latest_dir: str) -> str:
    path = os.path.join(latest_dir, "stage1_1_pre_impute.csv")
    df.to_csv(path, index=False)
    return path


def _mirror_phase1_final_dataset(
    source_path: str,
    canonical_path: str,
    legacy_path: str,
    dual_write_enabled: bool,
) -> None:
    """Keep canonical/legacy selected-features files aligned after Phase 1 completion."""
    if not dual_write_enabled or not source_path or not os.path.exists(source_path):
        return

    source_abs = os.path.abspath(source_path)
    for mirror_path in (canonical_path, legacy_path):
        if not mirror_path:
            continue
        mirror_abs = os.path.abspath(mirror_path)
        if mirror_abs == source_abs:
            continue
        os.makedirs(os.path.dirname(mirror_abs), exist_ok=True)
        shutil.copy2(source_abs, mirror_abs)
        print(f"[Final Standardization] ✅ Dual-write mirror: {mirror_abs}")


def get_data_summary(file_path: str, target_column: Optional[str] = None) -> DataSummary:
    """
    Generate metadata summary of a CSV file without exposing full data.
    
    This function reads the CSV locally and extracts only metadata that is
    safe to share with the LLM. Row-level values are never included.
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target/group column (optional)
    
    Returns:
        DataSummary with metadata only
    
    Privacy Note:
        - Only basename, byte size, column names, types, shape and missing counts are exposed
        - No row-level values or local directory paths are included
    
    Example:
        >>> summary = get_data_summary("data.csv", "Group")
        >>> print(summary["columns"])
        ['Sample_ID', 'Metabolite_1', 'Metabolite_2', 'Group']
    """
    try:
        df = _read_tabular_file(file_path)
        
        # Extract metadata
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        n_rows, n_cols = df.shape
        
        # Calculate missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items() if v > 0}
        
        return DataSummary(
            columns=columns,
            dtypes=dtypes,
            n_rows=n_rows,
            n_cols=n_cols,
            file_name=os.path.basename(file_path),
            file_size_bytes=int(os.path.getsize(file_path)),
            head_preview=ROW_LEVEL_PREVIEW_DISABLED,
            missing_values=missing_values if missing_values else None,
            target_column=target_column
        )
    
    except Exception as e:
        # Return minimal summary on error
        return DataSummary(
            columns=[],
            dtypes={},
            n_rows=0,
            n_cols=0,
            file_name=os.path.basename(file_path),
            file_size_bytes=int(os.path.getsize(file_path)) if os.path.exists(file_path) else 0,
            head_preview=ROW_LEVEL_PREVIEW_DISABLED,
            missing_values=None,
            target_column=target_column
        )


def _get_tool_documentation(
    lib_hints: str,
    tools_required: List[str] = None,
    retry_level: int = 0
) -> str:
    """
    === P0-2: 三层渐进式工具文档 ===
    
    根据重试级别动态返回不同详细程度的工具文档：
    - Level 0 (首次尝试): Minimal - 只有函数签名
    - Level 1 (第一次重试): Standard - 签名 + 参数说明 + 返回格式
    - Level 2+ (后续重试): Full - 完整示例代码
    
    这种设计可以：
    1. 首次尝试时节省 80% 的 Token
    2. 重试时逐步增加细节，帮助 LLM 理解
    3. 避免信息过载导致的注意力分散
    
    Args:
        lib_hints: SOP 中的 lib_hint 字段（用于向后兼容）
        tools_required: 当前步骤需要的工具列表（从 SOP 中提取）
        retry_level: 重试级别 (0=首次, 1=第一次重试, 2+=后续重试)
    
    Returns:
        格式化的工具文档字符串
    """
    # 确定文档详细级别
    if retry_level == 0:
        detail_level = "minimal"
    elif retry_level == 1:
        detail_level = "standard"
    else:
        detail_level = "full"
    
    # 如果提供了 tools_required，使用精确的工具列表
    if tools_required:
        return get_tools_documentation(tools_required, level=detail_level)
    
    # 否则，从 lib_hints 推断工具（向后兼容旧版 SOP）
    if not lib_hints:
        return "\n**Available Tools:** (No specific tools hinted for this step)\n"
    
    # 从 lib_hints 推断可能需要的工具
    inferred_tools = []
    lib_names = [lib.strip() for lib in lib_hints.split(",")]
    
    for lib_name in lib_names:
        # 根据库名推断工具
        if "data_analysis_tools" in lib_name:
            inferred_tools.extend([
                "analyze_dataset_quality",
                "data_context_resolver_tool",
                "zero_semantics_analyzer_tool",
                "missingness_assessment_tool",
                "preprocessing_qa_reporter_tool",
                "delete_columns_tool",
                "delete_rows_tool",
                "impute_missing_tool",
                "pqn_normalization_tool",
                "log2_transformation_tool",
                "cap_outliers_tool",
                "auto_scaling_tool"
            ])
        elif "feature_selection_tools" in lib_name:
            inferred_tools.extend([
                "run_elasticnet_selector",
                "run_lasso_selector",
                "run_random_forest_selector",
                "run_lightgbm_selector",
                "run_mrmr_selector",
                "run_t_test_selector",
                "run_fdr_effect_size_selector",
                "run_train_only_stability_selection",
                "build_stable_panel_from_logs",
                "perform_stability_selection",
                "calculate_frequencies_from_logs"
            ])
        elif "model_building_tools" in lib_name:
            inferred_tools.extend([
                "analyze_data_for_modeling",
                "train_with_autogluon"
            ])
        elif "json_metabolite_mapper" in lib_name:
            inferred_tools.append("map_metabolite_fast")
        elif "reaction_checker" in lib_name:
            inferred_tools.append("ReactionCheckerTool")
        elif "taxonomy_group_tool" in lib_name:
            inferred_tools.append("TaxonomyGroupTool")
        elif "pathway_scorer" in lib_name:
            inferred_tools.append("calculate_pathway_scores")
    
    # 去重
    inferred_tools = list(dict.fromkeys(inferred_tools))
    
    if inferred_tools:
        return get_tools_documentation(inferred_tools, level=detail_level)
    else:
        return "\n**Available Tools:** (Could not infer tools from lib_hint)\n"


def generate_code(state: Phase1State) -> Phase1State:
    """
    Node A: The Brain - Generate Python code based on SOP and metadata.
    
    === P0-2 & P0-3 集成 ===
    - 使用三层渐进式工具文档（根据 retry_level 调整详细程度）
    - 在代码生成后立即进行 AST 验证
    - 如果验证失败，立即重试并提供精确的错误信息
    
    This node constructs a prompt for the LLM that includes:
    1. The current SOP step instruction
    2. The data_summary (metadata only, not full data)
    3. Context variables (is_balanced, protected_columns, etc.)
    4. Instructions for the Magic Output Protocol
    5. Dynamic tool documentation (Optimization 1)
    6. Previous step output (Optimization 2 - Chain of Memory)
    
    The LLM generates Python code that:
    - Uses tools from src.tools.domain and src.tools.analysis
    - Operates on the file path (not DataFrame objects)
    - Prints state updates using the Magic Output Protocol
    
    Args:
        state: Current Phase1State
    
    Returns:
        Updated Phase1State with generated code in messages
    
    Privacy Note:
        The LLM only sees:
        - Column names and types (from data_summary)
        - Number of rows and columns
        - Context variables
        
        The LLM NEVER sees the full dataset.
    """
    # Get current step from SOP
    current_step = get_current_step(state)
    
    if not current_step:
        return {
            **state,
            "error": "No current step found in SOP",
            "completed": False
        }
    
    # Extract step information
    step_instruction = current_step.get("instruction", "")
    step_details = (
        current_step.get("details_override")
        or current_step.get("details", "")
    )  # Prefer override text when SOP keeps legacy long-form examples for compatibility
    step_id = current_step.get("step_id", current_step.get("id", "unknown"))
    state = _maybe_prepare_stage15_scaling_input(state, str(step_id))
    
    # === P0-2: 提取 tools_required 和确定 retry_level ===
    tools_required = current_step.get("tools_required", [])
    lib_hint = current_step.get("lib_hint", "")
    retry_level = state.get("retry_count", 0)
    
    # 获取渐进式工具文档
    tool_docs = _get_tool_documentation(lib_hint, tools_required, retry_level)
    
    # Optimization 2: Chain of Memory - Get previous step output
    last_exec = state["execution_history"][-1] if state["execution_history"] else None
    previous_step_output = ""
    
    if last_exec and last_exec["success"]:
        # Format the state updates nicely
        updates_str = json.dumps(last_exec.get("state_updates", {}), indent=2)
        previous_step_output = f"**Previous Step ({last_exec['step_id']}) Output**:\nSuccessful. State Updated: {updates_str}\n"
    elif last_exec and not last_exec["success"]:
        previous_step_output = f"**Previous Step ({last_exec['step_id']}) Failed**:\nError: {last_exec['stderr']}\n"
    
    # Inject canonical stability temp dir so generated code can use it.
    runtime_context = {
        **state["context_variables"],
        "stability_temp_dir": _get_phase1_stability_temp_dir(),
    }
    runtime_context = _merge_phase1_protected_anchor_context(
        context_variables=runtime_context,
        dataset_columns=state.get("data_summary", {}).get("columns", []),
        target_column=state.get("data_summary", {}).get("target_column"),
    )
    prompt_context, token_to_path = _build_prompt_path_aliases(
        runtime_context,
        state["current_data_path"],
    )
    previous_step_output = _alias_paths_in_text(previous_step_output, token_to_path)
    memory_hints = _format_phase1_memory_hints_for_prompt(
        state.get("phase1_memory_pack", {})
    )
    memory_hints = _alias_paths_in_text(memory_hints, token_to_path)

    # Build the prompt for the LLM
    prompt = _build_code_generation_prompt(
        step_instruction=step_instruction,
        step_details=step_details,  # CRITICAL: Pass detailed instructions to LLM
        step_id=step_id,
        data_summary=state["data_summary"],
        context_variables=prompt_context,
        current_data_path=CURRENT_DATA_PATH_TOKEN,
        tool_docs=tool_docs,
        previous_step_output=previous_step_output,
        memory_hints=memory_hints,
        retry_level=retry_level,  # 传递重试级别
        last_error=state.get("last_error", ""),
        last_failed_code=_alias_paths_in_text(
            state.get("last_failed_code", ""),
            token_to_path,
        ),
    )
    
    # CRITICAL: For Code Interpreter mode, we don't need full conversation history
    # Each step is independent. Only keep the current prompt.
    # This dramatically reduces token usage (from 300k+ to ~20k per call)
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Get LLM from state
    llm = state.get("llm")
    if not llm:
        return {
            **state,
            "error": "LLM not found in state",
            "completed": False
        }
    
    # Call LLM to generate code
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Convert dict messages to LangChain message objects
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        
        # Invoke LLM
        response = llm.invoke(lc_messages)
        
        # Extract generated code
        generated_code = _extract_code_from_message(response.content)
        generated_code = _autofix_engineered_feature_tool_calls(generated_code)
        
        if not generated_code:
            return {
                **state,
                "error": "LLM did not generate code in expected format",
                "messages": state["messages"] + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content}
                ]
            }
        
        # === P0-3: AST 验证 ===
        is_valid, validation_error = validate_code_with_contracts(generated_code, TOOL_CONTRACTS, step_id)
        
        if not is_valid:
            # 验证失败，立即重试
            print(f"[AST Validation Failed] {validation_error}")
            
            # 构建验证错误的重试 prompt
            validation_retry_prompt = f"""
🔍 CODE VALIDATION FAILED (Pre-Execution Check)

Your generated code has an issue that was detected BEFORE execution:

**Validation Error:**
{validation_error}

**Your Previous Code:**
```python
{generated_code}
```

**Task:** {step_instruction}

**Requirements:**
1. Fix the validation error
2. Ensure all tool imports are correct
3. Use correct parameter names
4. Generate corrected code

**Generate the FIXED code now:**
"""
            
            # 增加重试计数
            new_retry_count = state.get("retry_count", 0) + 1
            
            # 如果重试次数超限，返回错误
            max_retries = state.get("max_retries", 3)
            if new_retry_count >= max_retries:
                return {
                    **state,
                    "error": f"AST validation failed after {max_retries} retries: {validation_error}",
                    "completed": False
                }
            
            # 递归调用自己进行重试（带上验证错误信息）
            return generate_code({
                **state,
                "retry_count": new_retry_count,
                "last_error": validation_error,
                "last_failed_code": generated_code,
                "messages": state["messages"] + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": validation_retry_prompt}
                ]
            })
        
        # 验证通过，继续执行
        # Store only the latest exchange in state (for debugging/logging)
        # We keep the full history for record-keeping, but don't send it to LLM
        messages_with_response = state["messages"] + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content}
        ]
        
        return {
            **state,
            "messages": messages_with_response,
            "context_variables": runtime_context,
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"LLM invocation failed: {str(e)}",
            "completed": False
        }


def _build_code_generation_prompt(
    step_instruction: str,
    step_details: str,
    step_id: str,
    data_summary: DataSummary,
    context_variables: Dict[str, Any],
    current_data_path: str,
    tool_docs: str = "",
    previous_step_output: str = "",
    memory_hints: str = "",
    retry_level: int = 0,
    last_error: str = "",
    last_failed_code: str = "",
) -> str:
    """
    Build the prompt for code generation.
    
    === P0-2 集成：根据 retry_level 调整 prompt 详细程度 ===
    
    This prompt engineering is critical for:
    1. Ensuring the LLM uses correct column names
    2. Following the Magic Output Protocol
    3. Using privacy-preserving tool signatures
    4. Dynamic tool documentation (Optimization 1)
    5. Chain of Memory - previous step context (Optimization 2)
    6. Including detailed instructions from SOP (CRITICAL FIX)
    7. Progressive disclosure based on retry level (P0-2)
    
    Args:
        step_instruction: Current step instruction from SOP (brief title)
        step_details: Detailed instructions from SOP (full workflow)
        step_id: Current step ID
        data_summary: Current dataset metadata
        context_variables: Current context variables
        current_data_path: Path to current data file
        tool_docs: Dynamic tool documentation (Optimization 1)
        previous_step_output: Previous step result summary (Optimization 2)
        memory_hints: Advisory memory hints from similar historical runs
        retry_level: Current retry level (0=first attempt, 1+=retries)
        last_error: Most recent validation/execution error to correct
        last_failed_code: Previous failed code snippet for targeted repair
    
    Returns:
        Formatted prompt string
    """
    prompt_data_summary = _build_prompt_data_summary(data_summary, context_variables)
    columns_str = ", ".join(prompt_data_summary["prompt_columns"])
    dtypes_str = json.dumps(
        prompt_data_summary["dtype_summary"],
        indent=2,
        ensure_ascii=False,
    )
    target_col = prompt_data_summary["target_column"]

    missing_values_str = ""
    if prompt_data_summary["missing_value_summary"]:
        missing_values_str = (
            "\n- Missing Value Summary:\n"
            f"{json.dumps(prompt_data_summary['missing_value_summary'], indent=2, ensure_ascii=False)}"
        )

    schema_notes_str = ""
    if prompt_data_summary["notes"]:
        schema_notes_str = "\n".join(
            f"- {note}" for note in prompt_data_summary["notes"]
        )

    # Format context variables
    context_str = json.dumps(
        _summarize_prompt_value(context_variables),
        indent=2,
        ensure_ascii=False,
    )

    preprocessing_context_section = ""
    preprocessing_keys = [
        "platform_source",
        "data_level",
        "candidate_branch",
        "zero_semantics_status",
        "zero_handling_strategy",
        "zero_handling_mode_used",
        "suspected_missingness_mechanism",
        "recommended_imputation_family",
        "preprocessing_context_report_path",
        "zero_pattern_report_path",
        "missingness_report_path",
        "preprocessing_report_path",
        "preprocessing_decision_pack_path",
    ]
    preprocessing_snapshot = {
        key: context_variables.get(key)
        for key in preprocessing_keys
        if context_variables.get(key) not in (None, "", [], {})
    }
    if preprocessing_snapshot:
        preprocessing_context_section = f"""
**Preprocessing Decision Context**
These facts were exported by the targeted preprocessing P0 chain and should be treated as authoritative unless a later step updates them:
```json
{json.dumps(preprocessing_snapshot, indent=2)}
```
"""

    json_safety_note = """
**CRITICAL: JSON Report Safety**
- If you write any JSON artifact, ensure all values are JSON-serializable native Python types.
- Convert numpy/pandas scalar values with `.item()` or `int()/float()/bool()` before writing.
- Safe helper pattern:
```python
def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)
```
- Then write reports with:
```python
json.dump(report_payload, f, indent=2, default=_json_default)
```
"""
    
    # === P0-2: 根据 retry_level 调整 prompt 复杂度 ===
    if retry_level == 0:
        # 首次尝试：简洁版
        complexity_note = "**Note:** This is your first attempt. Keep code simple and follow the tool signatures exactly."
    elif retry_level == 1:
        # 第一次重试：添加更多指导
        complexity_note = "**Note:** This is a retry. Review the tool parameter descriptions carefully and ensure you're using the correct names."
    else:
        # 后续重试：完整指导
        complexity_note = "**Note:** This is a subsequent retry. Study the complete examples provided and replicate the patterns exactly."

    engineered_tool_note = ""
    if step_id in {"1.5.1", "1.5.2", "1.5.3"}:
        engineered_tool_note = """
**CRITICAL: Engineered Feature Generator Call Style**
- `generate_reaction_ratios`, `generate_taxonomy_sums`, and `generate_pathway_scores` are DIRECT FUNCTIONS.
- Call them exactly like `result = generate_reaction_ratios(...)`.
- NEVER write `generate_reaction_ratios.get(...)`, `generate_taxonomy_sums.get(...)`, or `generate_pathway_scores.get(...)`.
- After the function call, inspect the returned dict with `result['status']` / `result.get('status')`.
"""

    scaling_tool_note = ""
    if step_id == "1.5.5":
        scaling_tool_note = """
**CRITICAL: Step 1.5.5 Input Policy**
- For Step `1.5.5`, `current_data_path` is already the prepared scaling input.
- Call `auto_scaling_tool(data_path=current_data_path, columns=None, method='standard')` directly.
- DO NOT define helper functions to rediscover engineered files or infer the input path.
- DO NOT gate execution on `engineered_features_merged`, `n_ratio_features`, `n_sum_features`, or `n_pathway_scores`.
- Even if some engineered feature families are empty, still scale the merged matrix when `current_data_path` already exists.
"""

    schema_detection_note = ""
    if step_id == "0.1":
        schema_detection_note = """
**CRITICAL: Sample-ID Detection Safety**
- Prefer explicit ID-like column names such as `Sample_ID`, `sample_id`, `subject_id`, `patient_id`, `row_id`.
- Only use a uniqueness-based fallback for `object` / string columns.
- NEVER choose a numeric metabolite or intensity column as `sample_id_column` only because its values are highly unique.
- If no safe ID-like column exists, set `sample_id_column = None`.
"""

    retry_feedback = ""
    if retry_level > 0 and last_error:
        retry_feedback = f"""
**CRITICAL: Previous Attempt Failed**
- Retry Count: {retry_level}
- Last Error: {last_error}
- Repair the previous snippet with the smallest safe change that resolves this error.
- Preserve the current SOP task, required artifacts, state-update keys, and local data boundary.
- Before returning code, check that every name used in the snippet is defined or imported; do not introduce implicit names such as `context_variables`, `cfg`, or `scenario`.
"""
        if last_failed_code:
            retry_feedback += f"""
**Previous Failed Code (Fix this, do not repeat it):**
```python
{last_failed_code}
```
"""
    
    # Build prompt with optimizations
    prompt = f"""You are a Python Expert Data Scientist working on a metabolomics analysis pipeline.

{previous_step_output}
**Current Task (Step {step_id}):**
{step_instruction}

**Detailed Instructions:**
{step_details}

**Current Dataset Information:**
- Local Path Token: {current_data_path}
- File Name: {data_summary.get('file_name', '<unknown>')}
- File Size (bytes): {int(data_summary.get('file_size_bytes', 0) or 0)}
- Number of Rows: {data_summary['n_rows']}
- Number of Columns: {data_summary['n_cols']}
- Columns: {columns_str}
- Target Column: {target_col}
- Prompt-visible Column Count: {len(prompt_data_summary['prompt_columns'])}
- Schema Notes:
{schema_notes_str or '- Full schema fits within prompt budget.'}
- Data Types:
{dtypes_str}
{missing_values_str}

**Privacy Boundary:** Row-level previews and local directory paths are intentionally unavailable.

**Context Variables:**
```json
{context_str}
```

{memory_hints}
{preprocessing_context_section}
{json_safety_note}

**CRITICAL: How to Use Context Variables in Your Code**
Context variables are provided above for your reference. To use them in your code:
1. DO NOT try to read from 'context_variables.json' file (it doesn't exist)
2. DIRECTLY use the values shown above in your code
3. Example: If you see `"is_balanced": true` above, use `is_balanced = True` in your code
4. Example: If you see `"protected_columns": ["Sample_ID", "Group"]`, use `protected_columns = ["Sample_ID", "Group"]`

{tool_docs}
{engineered_tool_note}
{scaling_tool_note}
{schema_detection_note}
{retry_feedback}

**CRITICAL: Privacy-Preserving Design**
All tools accept FILE PATHS, not DataFrame objects. Use the opaque local path token exactly as provided;
the runtime resolves it after validation and before local execution. Example:
```python
from src.tools.analysis.data_analysis_tools import analyze_dataset_quality
result = analyze_dataset_quality(file_path="{current_data_path}")
```

**CRITICAL: Tool Return Format**
All tools return JSON STRINGS. You MUST parse them:
```python
import json
result_json = some_tool(data_path="file.csv")
result = json.loads(result_json)  # Parse JSON string to dict

if result['success']:
    output_path = result['output_path']
```

**CRITICAL: Magic Output Protocol**
If your code creates a new file or calculates important metrics, you MUST print them using this format:
```python
import json
# After your calculations...
state_updates = {{
    "current_data_path": "/path/to/new/file.csv",  # If you created a new file
    "is_balanced": True,  # If you calculated class balance
    "feature_selection_method": "random_forest",  # If you chose a method
    # ... any other context variables to update
}}
# Convert numpy/pandas scalars to native Python types before printing.
safe_updates = json.loads(json.dumps(state_updates, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o))))
print(json.dumps({{"__METABO_UPDATE__": safe_updates}}))
```

**CRITICAL: Standardized Error Handling**
Wrap ALL tool imports and calls with proper error handling:
```python
import json

# Import phase
try:
    from src.tools.analysis.data_analysis_tools import delete_columns_tool
except ImportError as e:
    error_report = {{
        "__METABO_UPDATE__": {{
            "error": f"ImportError: {{str(e)}}",
            "error_type": "import_error",
            "failed_module": "src.tools.analysis.data_analysis_tools"
        }}
    }}
    print(json.dumps(error_report))
    raise

# Execution phase
try:
    result_json = delete_columns_tool(columns=['bad_col'], data_path='data.csv')
    result = json.loads(result_json)
    
    if not result.get('success'):
        error_report = {{
            "__METABO_UPDATE__": {{
                "error": f"Tool failed: {{result.get('error')}}",
                "error_type": "tool_execution_error",
                "tool_name": "delete_columns_tool"
            }}
        }}
        print(json.dumps(error_report))
        raise ValueError(result.get('error'))
        
except TypeError as e:
    error_report = {{
        "__METABO_UPDATE__": {{
            "error": f"TypeError: {{str(e)}}",
            "error_type": "signature_mismatch",
            "tool_name": "delete_columns_tool"
        }}
    }}
    print(json.dumps(error_report))
    raise
```

**CRITICAL: Atomic Execution Safety**
If your previous code crashed while writing a file, the file might be corrupted or incomplete.
In your fix, ensure you:
1. Recreate the file completely from the source input
2. Do not assume partial writes succeeded
3. Always read from the original source file, not the corrupted output

{complexity_note}

**CRITICAL: Code Format Requirements**
1. Generate ONLY executable Python code in a single code block
2. Do NOT include explanations, markdown, or text outside the code block
3. Keep code concise and focused (prefer < 150 lines)
4. Use existing tools when available instead of implementing from scratch
5. Avoid unnecessary print statements (only use for Magic Output Protocol)

**Instructions:**
1. Write Python code to accomplish the task described above
2. Use the correct column names from the data summary (e.g., target column is '{target_col}')
3. Use file paths, not DataFrame objects
4. If you create a new file or calculate metrics, use the Magic Output Protocol
5. Include error handling (try-except blocks)
6. Add comments explaining your logic
7. Keep code simple and maintainable

**Generate the Python code now:**
"""
    
    return prompt


def _format_phase1_memory_hints_for_prompt(phase1_memory_pack: Dict[str, Any]) -> str:
    """
    Format advisory Phase 1 memory hints for prompt injection.

    These hints are suggestions only. They must not override the current SOP step
    or any judgments derived from the current dataset metadata.
    """
    if not isinstance(phase1_memory_pack, dict) or not phase1_memory_pack:
        return ""

    preprocessing_hints = phase1_memory_pack.get("recommended_preprocessing_hints", []) or []
    provenance_cases = phase1_memory_pack.get("provenance_cases", []) or []
    semantic_entry_id = str(phase1_memory_pack.get("semantic_entry_id", "") or "").strip()
    semantic_strategy_key = str(phase1_memory_pack.get("semantic_strategy_key", "") or "").strip()
    semantic_confidence = phase1_memory_pack.get("semantic_confidence", 0.0)
    if not preprocessing_hints:
        return ""

    hint_lines: List[str] = []
    for idx, hint in enumerate(preprocessing_hints[:3], start=1):
        if isinstance(hint, dict):
            title = str(hint.get("title") or hint.get("name") or f"hint_{idx}").strip()
            detail = str(
                hint.get("detail")
                or hint.get("description")
                or hint.get("recommendation")
                or ""
            ).strip()
            if detail:
                hint_lines.append(f"- Hint {idx}: {title} | {detail}")
            else:
                hint_lines.append(f"- Hint {idx}: {title}")
        else:
            hint_lines.append(f"- Hint {idx}: {str(hint).strip()}")

    provenance_summary = ""
    if provenance_cases:
        provenance_summary = f"\n- Matched Memory Cases: {', '.join(str(case_id) for case_id in provenance_cases[:5])}"

    semantic_summary = ""
    if semantic_entry_id:
        semantic_summary = (
            "\n- Semantic Strategy Match: "
            f"{semantic_entry_id} (confidence={float(semantic_confidence or 0.0):.2f})"
        )
        if semantic_strategy_key:
            semantic_summary += f"\n- Semantic Strategy Key: {semantic_strategy_key}"

    return (
        "**Relevant Prior Experience (Advisory Only):**\n"
        "- The following hints come from similar historical runs.\n"
        "- When a semantic strategy match is present, it represents a distilled cross-run preference rather than one single case.\n"
        "- Use them only as suggestions for implementation style or preprocessing choices.\n"
        "- You MUST still follow the current SOP step and the current dataset metadata.\n"
        f"{chr(10).join(hint_lines)}"
        f"{semantic_summary}"
        f"{provenance_summary}\n"
    )


def _build_error_signature(error_message: str) -> str:
    """Normalize an error string into a compact signature for loose matching."""
    if not error_message:
        return ""

    normalized = str(error_message).lower()
    replacements = [
        ("business logic error", ""),
        ("traceback (most recent call last)", ""),
        ("code execution failed", ""),
        ("typeerror", "type error"),
        ("importerror", "import error"),
        ("valueerror", "value error"),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"[^a-z0-9_\-./ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _retrieve_phase1_fix_hints(
    step_id: str,
    error_message: str,
    phase1_memory_pack: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant advisory fix hints for the current failed step.

    Matching is intentionally lightweight for MVP:
    - prefer exact `step_id` matches when present in the hint payload
    - otherwise score by keyword overlap with the normalized error signature
    """
    if not isinstance(phase1_memory_pack, dict) or not phase1_memory_pack:
        return []

    error_signature = _build_error_signature(error_message)
    error_tokens = {
        token for token in error_signature.split(" ")
        if token and len(token) >= 4
    }

    scored_hints: List[tuple[float, Dict[str, Any]]] = []
    for hint in phase1_memory_pack.get("recommended_fix_hints", []) or []:
        if not isinstance(hint, dict):
            continue

        score = 0.0
        hint_step_id = str(hint.get("step_id") or "").strip()
        if hint_step_id and hint_step_id == str(step_id):
            score += 3.0

        hint_error = _build_error_signature(
            str(
                hint.get("error_signature")
                or hint.get("error")
                or hint.get("pattern")
                or ""
            )
        )
        if hint_error:
            hint_tokens = {
                token for token in hint_error.split(" ")
                if token and len(token) >= 4
            }
            if hint_error and hint_error in error_signature:
                score += 2.0
            score += 0.5 * len(error_tokens.intersection(hint_tokens))

        if score > 0:
            scored_hints.append((score, hint))

    scored_hints.sort(key=lambda item: item[0], reverse=True)
    return [hint for _, hint in scored_hints[:3]]


def _format_fix_hints_for_reflection(hints: List[Dict[str, Any]]) -> str:
    """Format advisory fix hints for reflection prompt injection."""
    if not hints:
        return ""

    lines: List[str] = []
    for idx, hint in enumerate(hints[:3], start=1):
        title = str(hint.get("title") or hint.get("name") or f"fix_hint_{idx}").strip()
        detail = str(
            hint.get("detail")
            or hint.get("description")
            or hint.get("recommendation")
            or hint.get("fix")
            or ""
        ).strip()
        if detail:
            lines.append(f"- Hint {idx}: {title} | {detail}")
        else:
            lines.append(f"- Hint {idx}: {title}")

    return (
        "**Relevant Prior Fix Experience (Advisory Only):**\n"
        "- The following fix hints come from similar historical failures.\n"
        "- Use them as repair suggestions only.\n"
        "- You MUST still analyze the current error and verify against the current dataset/context.\n"
        f"{chr(10).join(lines)}\n"
    )


def build_phase1_summary(state: Phase1State) -> Dict[str, Any]:
    """
    Build a minimal structured Phase 1 summary for later memory writeback.

    This function is intentionally lightweight for MVP. It summarizes execution
    outcomes without attempting to persist memory yet.
    """
    execution_history = state.get("execution_history", []) or []
    context_variables = state.get("context_variables", {}) or {}
    error_history = state.get("error_history", []) or []
    error_fix_hints = state.get("error_fix_hints", []) or []
    phase1_memory_pack = state.get("phase1_memory_pack", {}) or {}
    training_results = context_variables.get("training_results_json") or {}
    if isinstance(training_results, str):
        try:
            training_results = json.loads(training_results)
        except Exception:
            training_results = {}

    feature_selection_method = (
        context_variables.get("feature_selection_method")
        or context_variables.get("selection_strategy")
    )
    best_model = context_variables.get("best_model")
    best_model_score = context_variables.get("best_model_score")
    if isinstance(training_results, dict):
        best_model = best_model or training_results.get("best_model")
        if best_model_score is None:
            best_model_score = (
                training_results.get("best_model_holdout_score")
                if training_results.get("best_model_holdout_score") is not None
                else training_results.get("best_model_validation_score")
            )

    successful_steps = [record for record in execution_history if record.get("success")]
    failed_steps = [record for record in execution_history if not record.get("success")]
    failed_step_ids = [
        str(record.get("step_id"))
        for record in failed_steps
        if record.get("step_id")
    ]
    unique_failed_step_ids = list(dict.fromkeys(failed_step_ids))
    latest_step_status: Dict[str, bool] = {}
    for record in execution_history:
        step_id = str(record.get("step_id") or "").strip()
        if not step_id:
            continue
        latest_step_status[step_id] = bool(record.get("success"))
    unresolved_failed_step_ids = [
        step_id for step_id, success in latest_step_status.items() if not success
    ]
    recovered_step_ids = [
        step_id
        for step_id in unique_failed_step_ids
        if latest_step_status.get(step_id) is True
    ]

    recommended_fix_hints: List[Dict[str, Any]] = []
    for hint in error_fix_hints[:3]:
        if isinstance(hint, dict):
            recommended_fix_hints.append(dict(hint))

    terminal_error = state.get("error")
    timed_out = bool(state.get("phase1_timed_out", False))
    max_steps_reached = bool(state.get("phase1_max_steps_reached", False))
    completed_cleanly = bool(state.get("completed", False)) and not bool(terminal_error)
    completed_effectively = (
        not bool(terminal_error)
        and not timed_out
        and not max_steps_reached
        and bool(execution_history)
        and not unresolved_failed_step_ids
    )

    summary = {
        "completed": completed_cleanly,
        "completed_effectively": completed_effectively,
        "had_error": bool(terminal_error),
        "timed_out": timed_out,
        "max_steps_reached": max_steps_reached,
        "total_execution_attempts": int(len(execution_history)),
        "successful_execution_count": int(len(successful_steps)),
        "failed_execution_count": int(len(failed_steps)),
        "retry_count_final": int(state.get("retry_count", 0)),
        "failed_step_ids": unique_failed_step_ids,
        "recovered_step_ids": recovered_step_ids,
        "unresolved_failed_step_ids": unresolved_failed_step_ids,
        "last_error": state.get("last_error") or terminal_error,
        "recommended_fix_hints": recommended_fix_hints,
        "recommended_preprocessing_hints": list(
            phase1_memory_pack.get("recommended_preprocessing_hints", []) or []
        )[:3],
        "memory_provenance_cases": list(
            phase1_memory_pack.get("provenance_cases", []) or []
        )[:5],
        "feature_selection_method": feature_selection_method,
        "selection_strategy": context_variables.get("selection_strategy"),
        "consensus_strategy": context_variables.get("consensus_strategy"),
        "n_methods_used": context_variables.get("n_methods_used"),
        "best_model": best_model,
        "best_model_score": best_model_score,
        "final_selected_features": list(context_variables.get("final_selected_features", []) or []),
        "consensus_features": list(context_variables.get("consensus_features", []) or []),
        "protected_anchor_features": list(context_variables.get("protected_anchor_features", []) or []),
        "error_history_excerpt": list(error_history[-3:]) if error_history else [],
    }
    return summary


def execute_code(state: Phase1State) -> Phase1State:
    """
    Node B: The Executor - Execute generated code and update state.
    
    This node:
    1. Extracts the generated code from the last LLM message
    2. Saves it to a temporary .py file
    3. Executes it using subprocess
    4. Parses stdout for the Magic Output Protocol
    5. Updates context_variables and current_data_path
    6. Refreshes data_summary if the data path changed
    7. Records execution in history
    
    Args:
        state: Current Phase1State with generated code in messages
    
    Returns:
        Updated Phase1State with execution results and refreshed metadata
    
    Magic Output Protocol:
        The generated code can print JSON to stdout:
        {"__METABO_UPDATE__": {"is_balanced": True, "current_data_path": "/tmp/new.csv"}}
        
        This node parses this and updates the state accordingly.
    
    Metadata Synchronization:
        After successful execution, if current_data_path changed:
        1. Read the new CSV file
        2. Generate fresh data_summary
        3. Update state with new metadata
        
        This ensures the LLM always knows the current data shape and columns.
    """
    # Check if workflow is already completed or has fatal error
    if state.get("completed"):
        return state
    
    # Extract generated code from last message
    if not state["messages"]:
        return {
            **state,
            "error": "No messages found to extract code from",
            "completed": False
        }
    
    last_message = state["messages"][-1]
    if last_message.get("role") != "assistant":
        return {
            **state,
            "error": "Last message is not from assistant"
        }
    
    generated_code = _extract_code_from_message(last_message.get("content", ""))
    generated_code = _autofix_engineered_feature_tool_calls(generated_code)

    if str(state.get("current_step", {}).get("id", "")) == "6.2":
        context_vars = dict(state.get("context_variables", {}) or {})
        selected_holdout_path = str(context_vars.get("selected_holdout_data_path", "") or "").strip()
        if not selected_holdout_path:
            for candidate in (
                "output/phase1/final/selected_features_holdout.csv",
                "data/selected_features_holdout.csv",
            ):
                if os.path.exists(candidate):
                    selected_holdout_path = candidate
                    break
        if selected_holdout_path:
            context_vars["selected_holdout_data_path"] = selected_holdout_path
            context_vars.setdefault("holdout_data_path", selected_holdout_path)
            state = {**state, "context_variables": context_vars}
            # The LLM may hard-code a missing holdout even when Step 5.3 created it.
            generated_code = re.sub(
                r"(?m)^(\s*selected_holdout_data_path\s*=\s*)None\b.*$",
                rf"\1{selected_holdout_path!r}",
                generated_code,
            )
            generated_code = re.sub(
                r"(?m)^(\s*HOLDOUT_DATA_PATH\s*=\s*)None\b.*$",
                rf"\1{selected_holdout_path!r}",
                generated_code,
            )
    
    if not generated_code:
        return {
            **state,
            "error": "No code found in assistant message"
        }

    # Path aliases are exposed to the LLM; only the local executor receives real paths.
    _, token_to_path = _build_prompt_path_aliases(
        dict(state.get("context_variables", {}) or {}),
        str(state.get("current_data_path", "") or ""),
    )
    generated_code = _materialize_local_paths(generated_code, token_to_path)
    generated_code = _inject_phase1_runtime_bindings(generated_code, state)
    generated_code = _enforce_phase1_runtime_paths(generated_code)

    # Runtime safeguard: force stability temp logs into canonical output path.
    generated_code = _enforce_stability_temp_dir(generated_code)
    
    # Get current step for logging
    current_step = get_current_step(state)
    step_id = current_step.get("step_id", current_step.get("id", "unknown")) if current_step else "unknown"
    
    # Execute the code with tool enforcement
    execution_result = _execute_python_code(generated_code, state["current_data_path"], step_id=step_id)
    
    # Parse Magic Output Protocol from stdout
    state_updates = _parse_magic_output(execution_result["stdout"])
    state_updates = _normalize_phase1_state_paths(state_updates)
    
    # === THREE-LEVEL ERROR DETECTION ===
    # Level 1: Subprocess exit code (Python crash/exception) - checked below
    # Level 2: Magic Output Protocol explicit error
    # Level 3: Implicit error patterns in stdout (heuristic)
    
    business_error = None
    error_source = None  # Track where error was detected for debugging
    
    # Level 2: Check for explicit error via Magic Output Protocol
    if state_updates and "error" in state_updates and state_updates["error"]:
        business_error = state_updates["error"]
        error_source = "Magic Output Protocol"
    
    # Level 3: Implicit error detection - scan stdout for error patterns
    if not business_error and execution_result["success"]:
        stdout_text = execution_result["stdout"]
        stdout_lower = stdout_text.lower()
        
        # Define error indicators (ordered by specificity)
        error_indicators = [
            ("traceback (most recent call last)", "Python traceback"),
            ("an error occurred:", "Error message"),
            ("exception:", "Exception"),
            ("failed to", "Failure message"),
        ]
        
        # Check each indicator
        for indicator, description in error_indicators:
            if indicator in stdout_lower:
                # Extract the actual error message from stdout
                error_lines = []
                lines = stdout_text.split('\n')
                
                for i, line in enumerate(lines):
                    if indicator in line.lower():
                        error_lines.append(line.strip())
                        # Get next 2 lines for context
                        for j in range(1, min(3, len(lines) - i)):
                            next_line = lines[i + j].strip()
                            if next_line:
                                error_lines.append(next_line)
                        break
                
                if error_lines:
                    business_error = " | ".join(error_lines[:3])  # First 3 lines
                else:
                    business_error = f"{description} detected in output"
                
                error_source = f"Implicit ({indicator})"
                break
        
        # Additional check for "Error:" at start of line (common pattern)
        if not business_error:
            for line in stdout_text.split('\n'):
                line_stripped = line.strip()
                # Check if line starts with "Error:" (case-insensitive)
                if line_stripped.lower().startswith("error:"):
                    business_error = line_stripped
                    error_source = "Error line"
                    break
    
    # Determine if execution truly succeeded (all three levels pass)
    execution_truly_succeeded = execution_result["success"] and not business_error
    
    # If Python crashed OR Business Logic failed, treat as failure
    if not execution_result["success"] or business_error:
        # Determine the error message
        if business_error:
            final_error = f"Business logic error ({error_source}): {business_error}"
        elif execution_result["stderr"]:
            final_error = f"Code execution failed: {execution_result['stderr']}"
        else:
            final_error = "Unknown error occurred (execution failure detected but no message provided)"
        
        # === CRITICAL FIX: Extract and preserve critical state variables even on error ===
        # Some state variables are critical for workflow continuation (e.g., is_balanced)
        # We should preserve these even if the step fails
        critical_updates = {}
        if state_updates:
            # Define critical keys that should be preserved even on error
            critical_keys = [
                'is_balanced',           # Critical for feature selection branching
                'class_ratio',           # Important for understanding data
                'n_features_in_pool',    # Important for debugging
                'feature_pool_path',     # Path to feature pool file
                'class_counts',          # Class distribution information
                'mandatory_features',
                'n_mandatory_features',
                'protected_anchor_features',
                'n_protected_anchor_features',
            ]
            
            for key in critical_keys:
                if key in state_updates:
                    critical_updates[key] = state_updates[key]
        
        # Apply critical state variables to context
        updated_context = {**state["context_variables"], **critical_updates}
        updated_context = _merge_phase1_protected_anchor_context(
            context_variables=updated_context,
            dataset_columns=state.get("data_summary", {}).get("columns", []),
            target_column=state.get("data_summary", {}).get("target_column"),
        )
        _sync_phase1_consensus_artifacts(updated_context)
        # === END CRITICAL FIX ===
        
        # Create execution record marking this as FAILED
        execution_record = ExecutionRecord(
            step_id=step_id,
            generated_code=generated_code,
            stdout=execution_result["stdout"],
            stderr=execution_result["stderr"] if not business_error else f"Business Error: {business_error}",
            success=False,  # Mark as failed even if subprocess succeeded
            execution_time=execution_result["execution_time"],
            state_updates=critical_updates if critical_updates else None  # Preserve critical updates
        )
        
        # Update execution history
        execution_history = state["execution_history"] + [execution_record]
        
        # Return error state to trigger retry logic
        return {
            **state,
            "execution_history": execution_history,
            "context_variables": updated_context,  # ← Apply critical state variables
            "error": final_error,
            "last_error": final_error,
            "retry_count": state.get("retry_count", 0) + 1
        }
    # === FIX END ===
    
    # Execution succeeded (both Python and business logic)
    # Create successful execution record
    execution_record = ExecutionRecord(
        step_id=step_id,
        generated_code=generated_code,
        stdout=execution_result["stdout"],
        stderr=execution_result["stderr"],
        success=True,
        execution_time=execution_result["execution_time"],
        state_updates=state_updates
    )
    
    # Update execution history
    execution_history = state["execution_history"] + [execution_record]
    
    # Update context variables with parsed updates (excluding error field)
    updated_context = {**state["context_variables"]}
    if state_updates:
        for key, value in state_updates.items():
            if key != "current_data_path" and key != "error":  # Handle data path separately, exclude error
                updated_context[key] = value

    if state_updates and "training_results_json" in state_updates:
        training_results = state_updates["training_results_json"]
        if isinstance(training_results, str):
            try:
                training_results = json.loads(training_results)
            except Exception:
                training_results = {}
        persisted_paths = _persist_phase1_training_results_artifact(
            state_updates["training_results_json"]
        )
        if persisted_paths:
            updated_context["autogluon_results_paths"] = persisted_paths
            updated_context["autogluon_results_path"] = persisted_paths[0]
        if isinstance(training_results, dict):
            if training_results.get("best_model") is not None:
                updated_context["best_model"] = training_results.get("best_model")
            if training_results.get("primary_metric") is not None:
                updated_context["best_model_metric"] = training_results.get("primary_metric")
            best_model_score = training_results.get("best_model_holdout_score")
            if best_model_score is None:
                best_model_score = training_results.get("best_model_validation_score")
            if best_model_score is not None:
                updated_context["best_model_score"] = best_model_score

    if not updated_context.get("feature_selection_method") and updated_context.get("selection_strategy"):
        updated_context["feature_selection_method"] = updated_context.get("selection_strategy")
    
    # Check if data path changed
    new_data_path = state_updates.get("current_data_path") if state_updates else None
    current_data_path = new_data_path if new_data_path else state["current_data_path"]
    
    # Refresh data summary if path changed
    data_summary = state["data_summary"]
    if new_data_path and new_data_path != state["current_data_path"]:
        # Metadata Synchronization: Refresh summary from new file
        target_column = state["data_summary"].get("target_column")
        data_summary = get_data_summary(new_data_path, target_column)
    updated_context = _merge_phase1_protected_anchor_context(
        context_variables=updated_context,
        dataset_columns=data_summary.get("columns", []),
        target_column=data_summary.get("target_column"),
    )
    updated_context = _refresh_preprocessing_qa_after_legacy_step(
        step_id=str(step_id),
        context_variables=updated_context,
        current_data_path=current_data_path,
        data_summary=data_summary,
    )
    _sync_phase1_consensus_artifacts(updated_context)
    
    # Return updated state with retry_count reset on success
    return {
        **state,
        "execution_history": execution_history,
        "context_variables": updated_context,
        "current_data_path": current_data_path,
        "data_summary": data_summary,
        "error": None,
        "retry_count": 0  # Reset retry count on success
    }


def _extract_code_from_message(message_content: str) -> str:
    """
    Extract Python code from LLM message.
    
    Looks for code blocks marked with ```python or ```
    """
    # Try to find code block with python marker
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, message_content, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try to find any code block
    pattern = r"```\n(.*?)```"
    matches = re.findall(pattern, message_content, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # No code block found, return empty
    return ""


def _execute_python_code(code: str, working_dir: str, step_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute Python code in a subprocess.
    
    This function uses LocalPythonExecutor for secure code execution.
    
    Args:
        code: Python code to execute
        working_dir: Working directory for execution (typically a file path, will use project root)
        step_id: Step identifier for tool enforcement (e.g., "1.5.1")
    
    Returns:
        Dictionary with execution results:
        - success: bool
        - stdout: str
        - stderr: str
        - execution_time: float
    """
    # Use project root as working directory so relative paths work correctly
    # (e.g., "data/test_data_for_agent.csv" resolves from project root)
    import os
    project_root = os.getcwd()
    
    # Use LocalPythonExecutor for execution with tool enforcement
    executor = LocalPythonExecutor(timeout=1200)  # 20 minute timeout (increased for stability selection)
    return executor.execute(code, project_root, step_id=step_id)


def _parse_magic_output(stdout: str) -> Optional[Dict[str, Any]]:
    """
    Parse the Magic Output Protocol from stdout.
    
    Looks for JSON with "__METABO_UPDATE__" key:
    {"__METABO_UPDATE__": {"is_balanced": True, "current_data_path": "/tmp/new.csv"}}
    
    Args:
        stdout: Standard output from code execution
    
    Returns:
        Dictionary of state updates or None if not found
    """
    try:
        # Look for lines containing __METABO_UPDATE__
        for line in stdout.split('\n'):
            if "__METABO_UPDATE__" in line:
                # Try to parse as JSON
                try:
                    data = json.loads(line.strip())
                    if "__METABO_UPDATE__" in data:
                        return data["__METABO_UPDATE__"]
                except json.JSONDecodeError:
                    # Try to extract JSON from the line
                    json_match = re.search(r'\{.*"__METABO_UPDATE__".*\}', line)
                    if json_match:
                        data = json.loads(json_match.group())
                        if "__METABO_UPDATE__" in data:
                            return data["__METABO_UPDATE__"]
        
        return None
    
    except Exception:
        return None


def check_generate_code_result(state: Phase1State) -> str:
    """
    Check if code generation succeeded or if workflow should end.
    
    Args:
        state: Current Phase1State
    
    Returns:
        "execute" - Code generated successfully, proceed to execution
        "end" - Code generation failed or workflow completed
    """
    # Check if workflow is marked as completed
    if state.get("completed"):
        return "end"
    
    # A generation/validation error must never fall through to execute_code.
    # Otherwise execute_code can re-use the previous assistant message, after
    # which check_execution may route to reflect_and_fix without a newly failed
    # execution record. That creates an unpaired reflection trace and makes
    # the repair history impossible to audit. Generation errors are therefore
    # terminal at this graph boundary; execution failures are handled by
    # execute_code -> check_execution -> reflect_and_fix.
    if state.get("error"):
        return "end"

    # Do not execute a stale/non-code message if a caller invokes this router
    # with an incomplete message state.
    messages = state.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return "end"
    if not _extract_code_from_message(messages[-1].get("content", "")):
        return "end"
    
    # Otherwise, proceed to execution
    return "execute"


def should_continue(state: Phase1State) -> str:
    """
    Routing function to determine next node.
    
    Args:
        state: Current Phase1State
    
    Returns:
        "end" if workflow completed or error occurred
        "generate_code" to continue to next step
    """
    if state.get("completed"):
        # ====================================================================
        # 方案 A: Phase 1 完成时确保最终数据集列名已标准化
        # ====================================================================
        print("\n" + "="*80)
        print("Phase 1 Completed - Final Dataset Standardization Check")
        print("="*80)
        
        try:
            from src.utils.feature_name_standardizer import standardize_feature_name
            import pandas as pd

            cfg = get_config()
            io_policy = cfg.get_phase1_io_policy()
            dual_write_enabled = bool(io_policy.get("dual_write_enabled", False))
            selected_feature_paths = _resolve_phase1_selected_feature_paths(state)
            canonical_selected_features_path = selected_feature_paths["canonical"]
            legacy_selected_features_path = selected_feature_paths["legacy"]
            final_data_path = selected_feature_paths["active"]
            
            if os.path.exists(final_data_path):
                df = pd.read_csv(final_data_path)
                
                # 确定需要保护的列
                target_column = state.get("data_summary", {}).get("target_column", "Group")
                protected_columns = {'Sample_ID', 'sample_id', 'SampleID', 'sampleid', 'ROW_ID', 'row_id', '__row_id__', 'Group', target_column}
                
                # 检查是否需要标准化
                needs_standardization = False
                column_mapping = {}
                
                for col in df.columns:
                    if col in protected_columns:
                        column_mapping[col] = col
                    else:
                        standardized_name = standardize_feature_name(col)
                        column_mapping[col] = standardized_name
                        if col != standardized_name:
                            needs_standardization = True
                
                if needs_standardization:
                    print(f"[Final Standardization] Standardizing final dataset columns...")
                    
                    # 应用标准化
                    df.rename(columns=column_mapping, inplace=True)
                    df.to_csv(final_data_path, index=False)
                    
                    print(f"[Final Standardization] ✅ Final dataset standardized")
                    print(f"   Path: {final_data_path}")
                    print(f"   Shape: {df.shape}")
                else:
                    print(f"[Final Standardization] ✅ Final dataset already standardized")
                    print(f"   Path: {final_data_path}")
                    print(f"   Shape: {df.shape}")

                # Keep canonical and legacy final datasets synchronized.
                _mirror_phase1_final_dataset(
                    source_path=final_data_path,
                    canonical_path=canonical_selected_features_path,
                    legacy_path=legacy_selected_features_path,
                    dual_write_enabled=dual_write_enabled,
                )
        
        except Exception as e:
            print(f"[Final Standardization] ⚠️  Warning: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*80 + "\n")
        
        # ====================================================================
        # Phase 1 完成时导出 Artifact（Artifact-Based 架构）
        # ====================================================================
        print("\n" + "="*80)
        print("Phase 1 Completed - Exporting Artifacts")
        print("="*80)
        
        try:
            # 获取最终特征列表（使用标准化后的名称）
            context_vars = state.get("context_variables", {})
            cfg = get_config()
            selected_feature_paths = _resolve_phase1_selected_feature_paths(state)
            canonical_selected_features_path = selected_feature_paths["canonical"]
            canonical_artifacts_dir = cfg.get_phase1_path("artifacts_dir") or "output/phase1/artifacts"
            ag_results_path = os.path.join(
                canonical_artifacts_dir,
                "autogluon_training_results.json",
            )
            if not os.path.exists(ag_results_path):
                ag_results_path = (
                    cfg.get_phase1_path("autogluon_results")
                    or "data/autogluon_training_results.json"
                )
            target_column = (
                context_vars.get("target_column")
                or state.get("data_summary", {}).get("target_column")
                or "target"
            )
            holdout_eval_path = (
                context_vars.get("selected_holdout_data_path")
                or context_vars.get("holdout_data_path")
                or ""
            )
            
            # ✅ 优先使用 Step 5.3 的最终特征列表 (包含 mandatory features)
            # Fallback 到 Step 5.2.3 的 consensus features (如果 Step 5.3 未执行)
            final_features = context_vars.get("final_selected_features", 
                                              context_vars.get("consensus_features", []))
            protected_anchor_features = context_vars.get("protected_anchor_features", []) or []
            
            # 标准化特征列表中的名称
            if final_features:
                from src.utils.feature_name_standardizer import standardize_feature_name
                
                # 标准化特征名称
                standardized_features = [standardize_feature_name(f) for f in final_features]
                standardized_protected = [standardize_feature_name(f) for f in protected_anchor_features]
                standardized_features = list(dict.fromkeys(standardized_features + standardized_protected))
                if os.path.exists(canonical_selected_features_path):
                    selected_df = pd.read_csv(canonical_selected_features_path, nrows=0)
                    available_feature_names = {
                        standardize_feature_name(col)
                        for col in selected_df.columns
                    }
                    standardized_features = [
                        feature for feature in standardized_features if feature in available_feature_names
                    ]
                
                # 更新 context_variables 中的特征列表
                context_vars["consensus_features"] = standardized_features
                context_vars["final_selected_features"] = standardized_features
                state = {**state, "context_variables": context_vars}
                
                print(f"[Artifact Export] Standardized {len(standardized_features)} feature names")
            
            if final_features:
                # 导出 Phase1 panel scores artifact
                from src.tools.analysis.artifact_exporters import (
                    export_feature_provenance,
                    export_phase1_panel_scores,
                )
                
                artifact_path = export_phase1_panel_scores(
                    data_path=canonical_selected_features_path,
                    target_column=target_column,
                    selected_features=standardized_features,  # 使用标准化后的名称
                    ag_results_path=ag_results_path,
                    eval_data_path=holdout_eval_path,
                    output_dir=canonical_artifacts_dir,
                    artifact_filename='phase1_panel_scores.json'
                )
                
                print(f"\n✅ Phase1 panel scores artifact exported successfully")
                print(f"   Path: {artifact_path}")
                print(f"   Features: {len(standardized_features)}")

                phase0_output = context_vars.get("phase0_output")
                provenance_path = export_feature_provenance(
                    selected_features=standardized_features,
                    data_path=canonical_selected_features_path,
                    target_column=target_column,
                    phase0_output=phase0_output if isinstance(phase0_output, dict) else None,
                    output_dir=canonical_artifacts_dir,
                    artifact_filename="feature_provenance.json",
                )

                print(f"\n✅ Feature provenance artifact exported successfully")
                print(f"   Path: {provenance_path}")
            else:
                print(f"\n⚠️  Warning: No final features found in context_variables")
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to export Phase1 artifacts: {e}")
            import traceback
            traceback.print_exc()

        # ====================================================================
        # 阶段化迁移：镜像关键中间目录到 output/phase1/intermediate/latest
        # ====================================================================
        try:
            _ensure_phase1_intermediate_mirror()
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to mirror intermediate directories: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*80 + "\n")
        return "end"
    
    if state.get("error"):
        return "end"
    
    # Check if there are more steps
    current_step = get_current_step(state)
    if not current_step:
        # Some execution paths end here without setting completed=True.
        # Still force intermediate mirror materialization.
        try:
            _ensure_phase1_intermediate_mirror()
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to mirror intermediate directories on end: {e}")
        return "end"
    
    return "generate_code"


def check_execution(state: Phase1State) -> str:
    """
    Check execution result and decide next action.
    
    This routing function determines whether to:
    - Continue to next step (success)
    - Retry with reflection (failure but retries available)
    - End workflow (failure with no retries left)
    
    Args:
        state: Current Phase1State
    
    Returns:
        "success" - Execution succeeded, advance to next step
        "retry" - Execution failed but can retry
        "failed" - Execution failed and no retries left
    """
    # Check if there's an error
    if not state.get("error"):
        return "success"

    # Reflection is only valid after a failed execution. If an error was
    # produced during code generation (or a stale error leaked into this
    # router) and the latest execution record is successful/missing, terminate
    # rather than fabricating a repair episode from an unrelated record.
    execution_history = state.get("execution_history") or []
    if not execution_history or execution_history[-1].get("success", True):
        return "failed"
    
    # There's an error - check retry count
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if retry_count < max_retries:
        return "retry"  # Can retry
    else:
        return "failed"  # No more retries


def reflect_and_fix(state: Phase1State) -> Phase1State:
    """
    Error Reflection and Fix Node.
    
    When code execution fails, this node:
    1. Analyzes the error message
    2. Constructs a reflection prompt for the LLM
    3. Asks LLM to understand the error and generate fixed code
    4. Updates retry count and error history
    
    Privacy Guard: Error messages are truncated to 2000 chars max
    Context Management: Only keeps recent error history to avoid context bloat
    
    Args:
        state: Phase1State with error information
    
    Returns:
        Updated Phase1State with reflection prompt
    """
    # Check if workflow is already completed
    if state.get("completed"):
        return state
    
    # Get the last execution record (the failed one)
    execution_history = state.get("execution_history") or []
    if not execution_history:
        return {
            **state,
            "error": "Reflection skipped: no execution history found",
            "last_error": "Reflection skipped: no execution history found",
            "completed": False
        }

    last_execution = execution_history[-1]
    if last_execution.get("success", True):
        # Never turn a successful execution into a synthetic failure. This
        # guard protects the audit trail when a generation-time error or stale
        # state reaches the retry branch.
        message = (
            "Reflection skipped: latest execution record is successful; "
            "no failed execution is available for repair."
        )
        return {
            **state,
            "error": message,
            "last_error": message,
            "completed": False,
        }

    failed_code = last_execution["generated_code"]
    error_message = last_execution["stderr"]
    step_id = last_execution["step_id"]
    
    # Get current step instruction
    current_step = get_current_step(state)
    step_instruction = current_step.get("instruction", "") if current_step else ""
    
    # Privacy Guard: Truncate error message (tail 2000 chars for recent errors)
    if len(error_message) > 2000:
        error_message = f"[... truncated ...]\n{error_message[-2000:]}"
    
    # Context Management: Summarize error history if retry_count > 1
    error_context = ""
    retry_count = state.get("retry_count", 0)
    
    if retry_count > 1 and state.get("error_history"):
        # Summarize previous attempts to save context
        prev_attempts = len(state["error_history"])
        error_context = f"\n**Previous Attempts**: {prev_attempts} failed attempts before this.\n"
        
        # Only show the most recent previous error
        if state["error_history"]:
            last_prev_error = state["error_history"][-1]
            prev_error_msg = last_prev_error.get("error", "")
            if len(prev_error_msg) > 500:
                prev_error_msg = prev_error_msg[-500:]
            error_context += f"Last attempt error: {prev_error_msg}\n"

    fix_hints = _retrieve_phase1_fix_hints(
        step_id=step_id,
        error_message=error_message,
        phase1_memory_pack=state.get("phase1_memory_pack", {}),
    )
    fix_hints_text = _format_fix_hints_for_reflection(fix_hints)
    
    # Build reflection prompt
    reflection_prompt = _build_reflection_prompt(
        step_instruction=step_instruction,
        failed_code=failed_code,
        error_message=error_message,
        data_summary=state["data_summary"],
        context_variables=state["context_variables"],
        retry_count=retry_count,
        max_retries=state.get("max_retries", 3),
        error_context=error_context,
        fix_hints_text=fix_hints_text,
    )
    
    # Update error history
    error_history = state.get("error_history", []) + [{
        "step_id": step_id,
        "retry_attempt": retry_count,
        "error": error_message,
        "failed_code": failed_code[:500]  # Only store first 500 chars of code
    }]
    
    # Add reflection message to conversation
    messages = state["messages"] + [
        {"role": "user", "content": reflection_prompt}
    ]
    
    # Update reflection messages (for tracking)
    reflection_messages = state.get("reflection_messages", []) + [
        {"role": "user", "content": reflection_prompt}
    ]
    
    # Return updated state
    return {
        **state,
        "messages": messages,
        "reflection_messages": reflection_messages,
        "error_history": error_history,
        "error_fix_hints": fix_hints,
        "retry_count": retry_count,
        "last_error": error_message,
        # Preserve the complete failed snippet for the subsequent generation
        # call.  Without this field, the reflection node records the failure
        # but the next prompt receives only the error text and may rewrite the
        # whole step, introducing unrelated undefined names or changing the
        # analytical task.  Keeping the snippet enables a targeted, auditable
        # repair while retaining the existing privacy boundary.
        "last_failed_code": failed_code,
        "error": None  # Clear error to allow retry
    }


def _build_reflection_prompt(
    step_instruction: str,
    failed_code: str,
    error_message: str,
    data_summary: DataSummary,
    context_variables: Dict[str, Any],
    retry_count: int,
    max_retries: int,
    error_context: str = "",
    fix_hints_text: str = "",
) -> str:
    """
    Build error reflection prompt for LLM.
    
    This prompt helps the LLM:
    1. Understand what went wrong
    2. Analyze the error
    3. Generate corrected code
    
    Args:
        step_instruction: Original task instruction
        failed_code: Code that failed
        error_message: Error message from execution
        data_summary: Current dataset metadata
        context_variables: Current context
        retry_count: Current retry attempt number
        max_retries: Maximum retries allowed for the current step
        error_context: Summary of previous attempts
        fix_hints_text: Advisory fix hints from similar historical failures
    
    Returns:
        Reflection prompt string
    """
    columns_str = ", ".join(data_summary["columns"])
    target_col = data_summary.get("target_column", "unknown")
    context_str = json.dumps(context_variables, indent=2)
    
    prompt = f"""🔧 CODE EXECUTION FAILED - ERROR ANALYSIS AND FIX REQUIRED

**Retry Attempt**: {retry_count} / {max_retries}

{error_context}

**Original Task**:
{step_instruction}

**Current Dataset Information**:
- Columns: {columns_str}
- Rows: {data_summary['n_rows']}
- Target Column: {target_col}

**Context Variables**:
```json
{context_str}
```

{fix_hints_text}

**Failed Code**:
```python
{failed_code}
```

**Error Message**:
```
{error_message}
```

**Your Task**:
1. **Analyze the error**: Understand why the code failed
2. **Identify the root cause**: What went wrong?
3. **Generate fixed code**: Write corrected Python code that will succeed

**Common Error Patterns to Check**:
- ❌ Wrong column names (check against current columns: {columns_str})
- ❌ Missing imports
- ❌ File path errors
- ❌ Type mismatches
- ❌ Logic errors
- ❌ Corrupted file from previous failed write

**Requirements for Fixed Code**:
✅ Use correct column names from the data summary
✅ Include all necessary imports
✅ Use file paths, not DataFrame objects
✅ Add error handling (try-except)
✅ Include the Magic Output Protocol if needed
✅ Add comments explaining the fix
✅ Prefer a minimal patch to the failed code; preserve all valid task logic and required outputs
✅ Every referenced name must be locally defined or explicitly imported (do not assume `context_variables`, `cfg`, `scenario`, or other helper names exist at runtime)
✅ Do not replace a failed step with a placeholder, empty result, or unrelated reimplementation
✅ If the error is a Magic Output or output-contract failure, remove the failed/error payload and emit one valid success update only after the task has actually completed
✅ For file or directory errors, verify the resolved path type and create parent directories only for the intended file path; do not write text to a directory

**CRITICAL: Atomic Execution Safety**
If your previous code crashed while writing a file, the file might be corrupted.
In your fix:
1. Recreate the file completely from the source input
2. Do not assume partial writes succeeded
3. Always read from the original source file

**Generate the FIXED Python code now**:
"""
    
    return prompt
