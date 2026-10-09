from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .journal_theme import resolve_theme_color
from .viz_tools import _to_display_feature_names


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_existing_path(*candidates: Any) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        raw = Path(str(candidate))
        probe_paths = [raw]
        if not raw.is_absolute():
            probe_paths.append(PROJECT_ROOT / raw)
        for probe in probe_paths:
            if probe.exists():
                return probe.resolve()
    raise FileNotFoundError(f"Unable to resolve existing path from candidates: {candidates}")


def _read_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _read_csv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _ordered_unique(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _infer_phase1_root(data_path: Path) -> Optional[Path]:
    resolved = data_path.resolve()
    parts = resolved.parts
    if "phase1" not in parts:
        return None
    phase1_idx = len(parts) - 1 - parts[::-1].index("phase1")
    if phase1_idx < len(parts):
        return Path(*parts[: phase1_idx + 1])
    return None


def _resolve_unscaled_source_columns(
    *,
    features: Sequence[str],
    unscaled_matrix_path: Path,
    provenance_path: Path,
) -> List[str]:
    columns = _read_csv_header(unscaled_matrix_path)
    exact_map = {column: column for column in columns}
    lower_map = {column.lower(): column for column in columns}
    normalized_map = {_normalize_token(column): column for column in columns}

    provenance_payload = _read_json(provenance_path)
    provenance_lookup = {
        str(item.get("feature") or "").strip(): item
        for item in provenance_payload.get("features", [])
        if isinstance(item, dict) and str(item.get("feature") or "").strip()
    }

    resolved_columns: List[str] = []
    for feature in features:
        record = provenance_lookup.get(feature, {})
        candidate_names = _ordered_unique(
            [
                feature,
                str(record.get("standardized_name") or "").strip(),
                str(record.get("display_name") or "").strip(),
                *[str(item).strip() for item in record.get("mapped_hmdb_ids", []) or []],
                *[str(item).strip() for item in record.get("mapped_metabolites", []) or []],
            ]
        )

        source_column = None
        for candidate in candidate_names:
            if candidate in exact_map:
                source_column = exact_map[candidate]
                break
        if source_column is None:
            for candidate in candidate_names:
                lookup = lower_map.get(candidate.lower())
                if lookup:
                    source_column = lookup
                    break
        if source_column is None:
            for candidate in candidate_names:
                lookup = normalized_map.get(_normalize_token(candidate))
                if lookup:
                    source_column = lookup
                    break
        if source_column is None:
            raise KeyError(
                f"Unable to resolve unscaled source column for feature '{feature}' in {unscaled_matrix_path}"
            )
        resolved_columns.append(source_column)

    return resolved_columns


def _build_preprocessing_note(preprocessing_report_path: Path) -> str:
    report = _read_json(preprocessing_report_path)
    normalization_method = str(report.get("normalization_method") or "").strip()
    transformation_method = str(report.get("transformation_method") or "").strip()
    outlier_method = str(report.get("outlier_method") or "").strip()

    parts: List[str] = []
    if normalization_method:
        parts.append(f"{normalization_method}-normalized")
    if transformation_method:
        parts.append(f"{transformation_method}-transformed")
    if outlier_method:
        parts.append(f"{outlier_method}-capped")
    parts.append("non-z-score input matrix")
    return "Input scale: " + ", ".join(parts) + "."


def _resolve_rcs_inputs(
    data_path: str,
    features: Sequence[str],
    phase2_result_path: str = "",
) -> Dict[str, Any]:
    resolved_input_path = _resolve_existing_path(data_path)
    display_features = _to_display_feature_names(list(features))

    payload: Dict[str, Any] = {
        "data_path": str(resolved_input_path),
        "source_columns": list(features),
        "display_features": display_features,
        "preprocessing_note": "",
        "resolved_mode": "fallback_original_matrix",
    }

    phase1_root = _infer_phase1_root(resolved_input_path)
    if phase1_root is None and phase2_result_path:
        try:
            phase1_root = _infer_phase1_root(_resolve_existing_path(phase2_result_path))
        except Exception:
            phase1_root = None
    if phase1_root is None:
        return payload

    try:
        unscaled_matrix_path = _resolve_existing_path(
            phase1_root / "intermediate" / "latest" / "pre_engineering_matrix.csv",
        )
        provenance_path = _resolve_existing_path(
            phase1_root / "artifacts" / "feature_provenance.json",
            phase1_root / "legacy" / "artifacts" / "feature_provenance.json",
        )
        preprocessing_report_path = _resolve_existing_path(
            phase1_root / "intermediate" / "latest" / "preprocessing_report.json",
        )
        payload["data_path"] = str(unscaled_matrix_path)
        payload["source_columns"] = _resolve_unscaled_source_columns(
            features=features,
            unscaled_matrix_path=unscaled_matrix_path,
            provenance_path=provenance_path,
        )
        payload["preprocessing_note"] = _build_preprocessing_note(preprocessing_report_path)
        payload["resolved_mode"] = "unscaled_pre_engineering_matrix"
        payload["feature_provenance_path"] = str(provenance_path)
        payload["preprocessing_report_path"] = str(preprocessing_report_path)
    except Exception as exc:  # pragma: no cover - graceful fallback
        print(f"  ⚠️ Unable to resolve non-z-score RCS input bundle, falling back to original matrix: {exc}")

    return payload


def plot_rcs_curves_r(
    data_path: str,
    target_column: str,
    features: List[str],
    save_path: str = "output/figures/fig4d_rcs_panels.pdf",
    n_knots: int = 3,
    n_grid: int = 200,
    dpi: int = 300,
    figsize: Tuple[int, int] = (14, 10),
    backend: str = "r",
    layout_kwargs: Optional[Dict] = None,
    phase2_result_path: str = "",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    if not features:
        raise ValueError("RCS plotting requires at least one feature.")

    requested_backend = str(backend or "r").strip().lower()

    default_layout_kwargs = {
        "fontsize": 10,
        "title_fontsize": 17,
        "subtitle_fontsize": 10,
        "label_fontsize": 12,
        "line_color": resolve_theme_color(journal_theme, "winner", "#8A4F4A"),
        "line_fill": resolve_theme_color(journal_theme, "uncertainty_fill", "#D5DFDD"),
        "scatter_alpha": 0.18,
    }
    final_layout_kwargs = {**default_layout_kwargs, **(layout_kwargs or {})}

    input_bundle = _resolve_rcs_inputs(
        data_path=data_path,
        features=features,
        phase2_result_path=phase2_result_path,
    )

    def _python_fallback(reason: str):
        # plotRCS is an optional R package and is not available on every
        # deployment image.  The project already contains a statistically
        # equivalent patsy/statsmodels implementation; use it when the R
        # renderer is unavailable so Phase 3 still produces a downloadable
        # panel instead of dropping the complete figure task.
        print(f"  ⚠️ RCS R renderer unavailable; using Python fallback: {reason}")
        from .viz_tools import plot_rcs_curves as python_plot_rcs_curves

        fallback_features = input_bundle.get("source_columns") or list(features)
        return python_plot_rcs_curves(
            data_path=input_bundle["data_path"],
            target_column=target_column,
            features=fallback_features,
            save_path=save_path,
            n_knots=n_knots,
            n_grid=n_grid,
            dpi=dpi,
            figsize=figsize,
            backend="python",
            layout_kwargs=final_layout_kwargs,
        )

    if requested_backend != "r":
        if requested_backend == "python":
            return _python_fallback("backend='python'")
        raise ValueError(f"Unsupported RCS backend: {backend}")
    save_path_obj = Path(save_path)
    stem = save_path_obj.with_suffix("")
    output_pdf = stem.with_suffix(".pdf").resolve()
    output_png = stem.with_suffix(".png").resolve()
    output_svg = stem.with_suffix(".svg").resolve()
    summary_path = stem.with_suffix(".json").resolve()

    payload = {
        "data_path": input_bundle["data_path"],
        "target_column": target_column,
        "features": list(features),
        "source_columns": input_bundle.get("source_columns", list(features)),
        "display_features": input_bundle.get("display_features", list(features)),
        "preprocessing_note": input_bundle.get("preprocessing_note", ""),
        "output_pdf": str(output_pdf),
        "output_png": str(output_png),
        "output_svg": str(output_svg),
        "summary_path": str(summary_path),
        "line_color": final_layout_kwargs.get("line_color", "#8C1D18"),
        "line_fill": final_layout_kwargs.get("line_fill", "#D9A5A0"),
        "scatter_alpha": float(final_layout_kwargs.get("scatter_alpha", 0.18)),
        "fontsize": int(final_layout_kwargs.get("fontsize", 10)),
        "title_fontsize": int(final_layout_kwargs.get("title_fontsize", 17)),
        "subtitle_fontsize": int(final_layout_kwargs.get("subtitle_fontsize", 11)),
        "label_fontsize": int(final_layout_kwargs.get("label_fontsize", 12)),
        "font_family": journal_theme.get("font_family", "Times") if isinstance(journal_theme, dict) else "Times",
        "text_color": resolve_theme_color(journal_theme, "text_color", "#1F2937"),
        "axis_text_color": resolve_theme_color(journal_theme, "axis_text_color", "#374151"),
        "grid_color": resolve_theme_color(journal_theme, "grid_color", "#E9EDF2"),
        "panel_border_color": resolve_theme_color(journal_theme, "panel_border_color", "#D7DEE8"),
        "n_grid": int(n_grid),
        "dpi": int(dpi),
        "figsize": [float(figsize[0]), float(figsize[1])],
        "n_knots": int(n_knots),
    }

    r_script = PROJECT_ROOT / "src" / "tools" / "visualization" / "r" / "plot_rcs_singlepage_preview.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(PROJECT_ROOT / ".r_libs")
    with tempfile.TemporaryDirectory(prefix="phase3_rcs_r_") as tmpdir:
        payload_path = Path(tmpdir) / "payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            subprocess.run(
                ["Rscript", str(r_script), str(payload_path)],
                check=True,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        except Exception as exc:
            return _python_fallback(str(exc))

    primary_output = save_path_obj.resolve()
    if primary_output.suffix.lower() == ".png":
        primary_output = output_png
    elif primary_output.suffix.lower() == ".svg":
        primary_output = output_svg
    else:
        primary_output = output_pdf

    if not primary_output.exists():
        raise FileNotFoundError(f"RCS plotting completed but primary output is missing: {primary_output}")

    metadata = dict(final_layout_kwargs)
    metadata.update(
        {
            "rcs_summary_path": str(summary_path),
            "rcs_png_path": str(output_png),
            "rcs_svg_path": str(output_svg),
            "rcs_pdf_path": str(output_pdf),
            "rcs_backend": "r",
            "rcs_engine": "plotRCS",
            "rcs_input_data_path": str(input_bundle["data_path"]),
            "rcs_input_mode": input_bundle.get("resolved_mode", "fallback_original_matrix"),
            "feature_provenance_path": input_bundle.get("feature_provenance_path", ""),
            "preprocessing_report_path": input_bundle.get("preprocessing_report_path", ""),
        }
    )
    return str(primary_output), metadata
