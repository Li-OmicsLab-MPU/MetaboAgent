from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.tools.visualization.journal_theme import write_theme_json

from .viz_tools import (
    _STABILITY_METHOD_LABELS,
    _build_stability_plot_dataframe,
    _format_feature_axis_label,
    _load_json_report,
)


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _run_r_method_support_dot_matrix(long_csv: Path, output_stem: Path, title: str, theme_json_path: str = "", subtitle: str = "") -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_phase1_method_support_dot_matrix.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(project_root / ".r_libs")
    command = [
        "Rscript",
        str(r_script),
        str(long_csv),
        str(output_stem.resolve()),
        title,
        theme_json_path,
        subtitle,
    ]
    subprocess.run(command, check=True, cwd=str(project_root), env=env)


def plot_method_feature_heatmap(
    stability_scores_path: str,
    stability_summary_path: str,
    feature_provenance_path: str = "",
    save_path: str = "output/figures/fig1e_method_support_dot_matrix.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
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
    final_df = df[df["is_final_panel"]].copy().reset_index(drop=True)
    if final_df.empty:
        raise ValueError("No final panel features found in the Phase 1 stability outputs.")

    method_cols = [method for method in _STABILITY_METHOD_LABELS if method in final_df.columns]
    order_df = final_df[["feature", "display_name", "panel_score", "selection_frequency", "method_consensus"]].copy()
    order_df = order_df.sort_values(
        ["panel_score", "selection_frequency", "method_consensus", "display_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    order_map = {feature: idx for idx, feature in enumerate(order_df["feature"].tolist(), start=1)}

    long_rows: List[Dict] = []
    for _, row in order_df.iterrows():
        feature = row["feature"]
        source_row = final_df[final_df["feature"] == feature].iloc[0]
        display_label = _format_feature_axis_label(str(row["display_name"]), max_len=28).replace("…", "...")
        for method in method_cols:
            long_rows.append(
                {
                    "feature": feature,
                    "display_name": row["display_name"],
                    "display_label": display_label,
                    "feature_order": order_map[feature],
                    "selection_frequency": float(row["selection_frequency"]),
                    "panel_score": float(row["panel_score"]),
                    "method_key": method,
                    "method_label": _STABILITY_METHOD_LABELS[method],
                    "method_frequency": float(source_row.get(method, 0.0) or 0.0),
                }
            )

    long_df = pd.DataFrame(long_rows)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    long_csv = output_path.parent / "phase1_method_support_long.csv"
    long_df.to_csv(long_csv, index=False)

    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    _run_r_method_support_dot_matrix(
        long_csv,
        output_stem,
        "Method-Specific Support for the Phase 1 Final Panel",
        theme_json_path,
        "",
    )
    saved_paths = _save_r_figure_bundle(save_path)
    metadata = {
        "figure_title": "Method-Specific Support for the Phase 1 Final Panel",
        "stability_scores_path": stability_scores_path,
        "stability_summary_path": stability_summary_path,
        "feature_provenance_path": feature_provenance_path,
        "data_csv": str(long_csv),
        "theme_json_path": theme_json_path,
        "auxiliary_outputs": saved_paths[1:],
        "n_final_panel": int(len(order_df)),
        "n_methods": int(len(method_cols)),
    }
    return saved_paths[0], metadata
