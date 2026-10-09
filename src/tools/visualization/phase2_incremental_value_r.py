from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Evaluate.build_phase2_offline_repair import PROJECT_ROOT, _read_json, _resolve_existing_path
from src.tools.visualization.journal_theme import write_theme_json
from src.tools.analysis.nri_idi_tools import (
    align_binary_prediction_payloads,
    build_incremental_value_summary,
)


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _run_r_script(summary_json: Path, output_stem: Path, theme_json_path: str = "") -> None:
    r_script = PROJECT_ROOT / "src" / "tools" / "visualization" / "r" / "plot_phase2_incremental_value_summary.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(PROJECT_ROOT / ".r_libs")
    subprocess.run(
        ["Rscript", str(r_script), str(summary_json), str(output_stem.resolve()), theme_json_path],
        check=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _has_prediction_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    predictions = payload.get("predictions")
    return isinstance(predictions, list) and len(predictions) > 0


def _resolve_winner_scores_path(phase2_result_path: str, artifact_dir: str) -> Path:
    candidate_paths: List[Path] = []
    artifact_dir_path = Path(artifact_dir) if artifact_dir else None
    if artifact_dir_path is not None:
        candidate_paths.append(artifact_dir_path / "phase2_winner_scores.json")
    if phase2_result_path:
        candidate_paths.append(Path(phase2_result_path))

    seen: set[str] = set()
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if not candidate_str or candidate_str in seen:
            continue
        seen.add(candidate_str)
        try:
            resolved = _resolve_existing_path(candidate)
        except FileNotFoundError:
            continue
        try:
            payload = _read_json(resolved)
        except Exception:
            continue
        if _has_prediction_payload(payload.get("cv_predictions", {})):
            return resolved

    raise FileNotFoundError(
        "Unable to resolve a Phase 2 winner score payload with cv_predictions from "
        f"phase2_result_path={phase2_result_path!r}, artifact_dir={artifact_dir!r}"
    )


def _resolve_baseline_payload(winner_scores: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        winner_scores.get("phase1_panel_baseline_evaluation", {}).get("cv_predictions", {}),
        (
            winner_scores.get("prior_anchor_union_evaluation", {})
            .get("phase1_model_evaluation", {})
            .get("phase1_panel_baseline_evaluation", {})
            .get("cv_predictions", {})
        ),
        winner_scores.get("prior_anchor_union_evaluation", {}).get("phase1_model_evaluation", {}).get("cv_predictions", {}),
    ]
    for payload in candidates:
        if _has_prediction_payload(payload):
            return payload
    raise ValueError("No aligned Phase 1 baseline prediction payload found in phase2_winner_scores.json")


def _build_plot_summary(
    *,
    winner_scores: Dict[str, Any],
    bootstrap_iterations: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    winner_payload = winner_scores.get("cv_predictions", {})
    if not _has_prediction_payload(winner_payload):
        raise ValueError("Winner score payload is missing cv_predictions.")

    baseline_payload = _resolve_baseline_payload(winner_scores)
    aligned = align_binary_prediction_payloads(baseline_payload, winner_payload, positive_label=1)
    summary = build_incremental_value_summary(
        y_true=aligned.y_true.tolist(),
        p_old=aligned.p_old.tolist(),
        p_new=aligned.p_new.tolist(),
        baseline_name="Phase 1 panel",
        new_model_name="Phase 2 panel",
        comparison_scope="out_of_fold",
        positive_label=1,
        sample_ids=aligned.sample_ids,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )

    return {
        "figure_title": "Incremental Value of the Winner Panel",
        "x_label": "Improvement over the Phase 1 baseline",
        "summary": summary,
    }


def plot_phase2_incremental_value_summary(
    phase2_result_path: str,
    artifact_dir: str = "output/artifacts",
    save_path: str = "output/figures/fig4j_phase2_incremental_value_summary.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    winner_scores_path = _resolve_winner_scores_path(phase2_result_path=phase2_result_path, artifact_dir=artifact_dir)
    winner_scores = _read_json(winner_scores_path)

    plot_summary = _build_plot_summary(winner_scores=winner_scores)
    summary_json = asset_dir / f"{output_stem.name}_summary.json"
    summary_json.write_text(json.dumps(plot_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""

    _run_r_script(summary_json=summary_json, output_stem=output_stem, theme_json_path=theme_json_path)
    saved_paths = [path for path in _save_r_figure_bundle(save_path) if Path(path).exists()]

    metadata = {
        "figure_title": plot_summary["figure_title"],
        "winner_scores_path": str(winner_scores_path),
        "summary_json": str(summary_json),
        "theme_json_path": theme_json_path,
        "generated_paths": saved_paths,
        "incremental_value_summary": plot_summary["summary"],
    }
    return saved_paths[0], metadata
