from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.tools.visualization.journal_theme import write_theme_json

from .viz_tools import _build_stability_plot_dataframe, _format_feature_axis_label, _load_json_report


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _run_r_stability_landscape(csv_path: Path, output_stem: Path, title: str, theme_json_path: str = "") -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_phase1_stability_landscape.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(project_root / ".r_libs")
    command = [
        "Rscript",
        str(r_script),
        str(csv_path),
        str(output_stem.resolve()),
        title,
        theme_json_path,
    ]
    subprocess.run(command, check=True, cwd=str(project_root), env=env)


def plot_stability_landscape(
    stability_scores_path: str,
    stability_summary_path: str,
    feature_provenance_path: str = "",
    save_path: str = "output/figures/fig1d_stability_landscape.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    del dpi, figsize, layout_kwargs  # Rendering is delegated to the R script.

    stability_summary = _load_json_report(stability_summary_path)
    stability_scores_payload = _load_json_report(stability_scores_path)
    if not stability_summary or not stability_scores_payload:
        raise FileNotFoundError("Stability summary or stability score artifact is missing.")

    df = _build_stability_plot_dataframe(
        stability_summary=stability_summary,
        stability_scores_payload=stability_scores_payload,
        feature_provenance_path=feature_provenance_path,
    ).copy()

    df["display_label"] = df["display_name"].apply(lambda value: _format_feature_axis_label(value, max_len=28))

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    csv_path = output_path.parent / "phase1_stability_landscape_data.csv"
    export_cols = [
        "feature",
        "display_name",
        "display_label",
        "panel_score",
        "selection_frequency",
        "method_consensus",
        "status",
        "is_final_panel",
        "is_stable_core",
        "is_prior_protected",
    ]
    export_cols = [col for col in export_cols if col in df.columns]
    df.loc[:, export_cols].to_csv(csv_path, index=False)

    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    _run_r_stability_landscape(
        csv_path,
        output_stem,
        "Selection Stability of the Phase 1 Final Panel",
        theme_json_path,
    )
    saved_paths = _save_r_figure_bundle(save_path)
    metadata = {
        "figure_title": "Selection Stability of the Phase 1 Final Panel",
        "stability_scores_path": stability_scores_path,
        "stability_summary_path": stability_summary_path,
        "feature_provenance_path": feature_provenance_path,
        "data_csv": str(csv_path),
        "theme_json_path": theme_json_path,
        "auxiliary_outputs": saved_paths[1:],
        "n_features": int(len(df)),
        "n_final_panel": int(df["is_final_panel"].sum()),
        "n_stable_core": int(df["is_stable_core"].sum()),
    }
    return saved_paths[0], metadata
