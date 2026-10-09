from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr

from src.tools.visualization.journal_theme import write_theme_json
from src.tools.visualization.viz_tools import _to_phase3_display_feature_names


def _read_json(file_path: Path) -> Dict:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def _resolve_optional_path(path_value: str | None, search_anchor: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    search_roots = [search_anchor, search_anchor.parent, search_anchor.parent.parent, Path.cwd()]
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def _infer_feature_columns(df: pd.DataFrame) -> List[str]:
    protected = {
        "group",
        "Group",
        "target",
        "id",
        "ID",
        "Sample_ID",
        "sample_id",
        "SampleID",
        "sampleid",
        "ROW_ID",
        "row_id",
        "__row_id__",
    }
    cols: List[str] = []
    for column in df.columns:
        if column in protected:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            cols.append(column)
    return cols


def _flatten_numeric(df: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    values = df[list(columns)].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    return values


def _extract_sample_log2_matrix(df: pd.DataFrame, feature_columns: List[str], assume_log2: bool = False) -> np.ndarray:
    values = df[feature_columns].to_numpy(dtype=float)
    if not assume_log2:
        values = np.log2(np.clip(values, a_min=1e-12, a_max=None))
    return values


def _build_sample_kde_curves(sample_log2_matrix: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    curves: List[np.ndarray] = []
    for sample_values in sample_log2_matrix:
        finite_values = sample_values[np.isfinite(sample_values)]
        if finite_values.size < 3:
            continue
        if np.nanstd(finite_values) < 1e-8:
            finite_values = finite_values + np.linspace(-1e-6, 1e-6, finite_values.size)
        kde = gaussian_kde(finite_values)
        curves.append(kde(x_grid))
    if not curves:
        return np.empty((0, len(x_grid)))
    return np.vstack(curves)


def _format_feature_label(label: str, max_len: int = 22) -> str:
    label = str(label).replace("_", " ").strip()
    if len(label) <= max_len:
        return label
    return label[: max_len - 1] + "..."


def _load_feature_display_map(feature_provenance_path: str) -> Dict[str, str]:
    display_map: Dict[str, str] = {}
    if feature_provenance_path and Path(feature_provenance_path).exists():
        payload = _read_json(Path(feature_provenance_path))
        for record in payload.get("features", []):
            feature = str(record.get("feature", "")).strip()
            if not feature:
                continue
            mapped_metabolites = [str(item).strip() for item in (record.get("mapped_metabolites") or []) if str(item).strip()]
            original_name = str(record.get("original_name") or "").strip()
            if feature.startswith("RATIO_") and len(mapped_metabolites) >= 2:
                display_map[feature] = " / ".join(mapped_metabolites[:2])
            elif original_name:
                display_map[feature] = original_name
            else:
                display_map[feature] = str(record.get("display_name") or feature).strip()

    if display_map:
        resolved_names = _to_phase3_display_feature_names(list(display_map.keys()), feature_provenance_path=feature_provenance_path)
        for feature, resolved_name in zip(display_map.keys(), resolved_names):
            resolved_name = str(resolved_name).strip()
            if resolved_name:
                display_map[feature] = resolved_name
    return display_map


def _compute_retention_table(latest_dir: Path, raw_df: pd.DataFrame) -> pd.DataFrame:
    imputation_report = _read_json(latest_dir / "imputation_report.json")
    preprocessing_report = _read_json(latest_dir / "preprocessing_report.json")
    raw_features = len(_infer_feature_columns(raw_df))
    raw_samples = int(raw_df.shape[0])
    removed_features = int(imputation_report.get("removed_feature_count", 0) or 0)
    removed_rows = int(imputation_report.get("removed_row_count", 0) or 0)

    rows = [
        {"step": "Raw input", "sample_count": raw_samples, "feature_count": raw_features},
        {"step": "Zero handling", "sample_count": raw_samples, "feature_count": raw_features},
        {"step": "Feature filter", "sample_count": raw_samples, "feature_count": raw_features - removed_features},
        {"step": "Sample filter", "sample_count": raw_samples - removed_rows, "feature_count": raw_features - removed_features},
        {
            "step": "Post-imputation",
            "sample_count": int(imputation_report.get("final_shape", [raw_samples, 0])[0]),
            "feature_count": int(preprocessing_report.get("final_feature_count", raw_features - removed_features)),
        },
        {
            "step": "Final QC matrix",
            "sample_count": int(imputation_report.get("final_shape", [raw_samples, 0])[0]),
            "feature_count": int(preprocessing_report.get("final_feature_count", raw_features - removed_features)),
        },
    ]
    return pd.DataFrame(rows)


def _compute_iqr_capping_summary(pqn_df: pd.DataFrame, feature_columns: List[str]) -> List[Tuple[str, float]]:
    transformed = np.log2(np.clip(pqn_df[feature_columns].to_numpy(dtype=float), a_min=1e-12, a_max=None))
    cap_records: List[Tuple[str, float]] = []
    for idx, feature in enumerate(feature_columns):
        values = transformed[:, idx]
        q1 = float(np.nanpercentile(values, 25))
        q3 = float(np.nanpercentile(values, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clipped = np.clip(values, lower, upper)
        mean_abs_delta = float(np.mean(np.abs(clipped - values)))
        if mean_abs_delta > 0:
            cap_records.append((feature, mean_abs_delta))
    cap_records.sort(key=lambda item: item[1], reverse=True)
    return cap_records[:3]


def _save_figure_bundle(fig: plt.Figure, save_path: str) -> List[str]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    for ext in ordered_exts:
        current_path = stem.with_suffix(ext)
        fig.savefig(current_path, dpi=300, bbox_inches="tight", facecolor="white")
        saved_paths.append(str(current_path))
    return saved_paths


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _run_r_pheatmap(corr_csv: Path, stars_csv: Path, output_stem: Path, title: str, theme_json_path: str = "") -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_phase1_corr_heatmap.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(project_root / ".r_libs")
    command = [
        "Rscript",
        str(r_script),
        str(corr_csv),
        str(stars_csv),
        str(output_stem.resolve()),
        "complete",
        "6",
        title,
        theme_json_path,
    ]
    subprocess.run(command, check=True, cwd=str(project_root), env=env)


def _run_r_qc_overview(
    retention_csv: Path,
    sample_box_csv: Path,
    dispersion_csv: Path,
    band_csv: Path,
    summary_txt: Path,
    output_stem: Path,
    title: str,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_phase1_qc_overview.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(project_root / ".r_libs")
    command = [
        "Rscript",
        str(r_script),
        str(retention_csv),
        str(sample_box_csv),
        str(dispersion_csv),
        str(band_csv),
        str(summary_txt),
        str(output_stem.resolve()),
        title,
    ]
    subprocess.run(command, check=True, cwd=str(project_root), env=env)


def plot_phase1_qc_overview(
    phase1_latest_dir: str,
    feature_provenance_path: str = "",
    raw_data_path: str = "",
    save_path: str = "output/figures/fig1f_phase1_qc_overview.pdf",
    dpi: int = 300,
    figsize: Tuple[float, float] = (18.0, 6.1),
) -> Tuple[str, Dict]:
    latest_dir = Path(phase1_latest_dir)
    if not latest_dir.exists():
        raise FileNotFoundError(f"Phase 1 latest directory not found: {phase1_latest_dir}")

    preprocessing_context = _read_json(latest_dir / "preprocessing_context.json")
    normalization_report = _read_json(latest_dir / "normalization_decision_report.json")
    transformation_report = _read_json(latest_dir / "transformation_report.json")
    outlier_report = _read_json(latest_dir / "outlier_audit_report.json")
    missingness_report = _read_json(latest_dir / "missingness_report.json")

    context_dataset_path = _resolve_optional_path(preprocessing_context.get("dataset_path"), latest_dir)
    explicit_raw_path = Path(raw_data_path).resolve() if raw_data_path and Path(raw_data_path).exists() else None
    raw_source_path = explicit_raw_path or context_dataset_path or (latest_dir / "repeat_train_standardized.csv")

    raw_df = _read_csv(raw_source_path)
    imputed_df = _read_csv(latest_dir / "stage1_1_imputed.csv")
    pqn_snapshot_path = _resolve_optional_path(normalization_report.get("pqn_snapshot_path"), latest_dir)
    final_snapshot_path = _resolve_optional_path(outlier_report.get("final_qc_snapshot_path"), latest_dir)
    pqn_df = _read_csv(pqn_snapshot_path) if pqn_snapshot_path and pqn_snapshot_path.exists() else _read_csv(latest_dir / "stage1_1_imputed_preprocessed.csv")
    final_qc_df = _read_csv(final_snapshot_path) if final_snapshot_path and final_snapshot_path.exists() else _read_csv(latest_dir / "pre_engineering_matrix.csv")

    raw_features = _infer_feature_columns(raw_df)
    pqn_features = _infer_feature_columns(pqn_df)
    final_features = _infer_feature_columns(final_qc_df)
    common_features = [col for col in raw_features if col in pqn_features and col in final_features]
    if not common_features:
        raise ValueError("No common numeric features found across raw/PQN/final QC matrices.")

    display_map = _load_feature_display_map(feature_provenance_path)
    retention_df = _compute_retention_table(latest_dir, raw_df)
    raw_values = _flatten_numeric(raw_df, common_features)
    pqn_values = _flatten_numeric(pqn_df, common_features)
    final_values = _flatten_numeric(final_qc_df, common_features)

    raw_log_values = np.log2(np.clip(raw_values, a_min=1e-12, a_max=None))
    pqn_log_values = np.log2(np.clip(pqn_values, a_min=1e-12, a_max=None))
    final_log_values = final_values.copy()
    raw_sample_log2 = _extract_sample_log2_matrix(raw_df, common_features, assume_log2=False)
    pqn_sample_log2 = _extract_sample_log2_matrix(pqn_df, common_features, assume_log2=False)
    final_sample_log2 = _extract_sample_log2_matrix(final_qc_df, common_features, assume_log2=True)
    sample_intensity_df = pd.DataFrame(
        {
            "Total intensity": np.concatenate(
                [
                    raw_df[common_features].sum(axis=1).to_numpy(dtype=float),
                    pqn_df[common_features].sum(axis=1).to_numpy(dtype=float),
                ]
            ),
            "Stage": ["Raw"] * len(raw_df) + ["Post-PQN"] * len(pqn_df),
        }
    )

    x_min = float(np.nanpercentile(np.concatenate([raw_log_values, pqn_log_values, final_log_values]), 0.5))
    x_max = float(np.nanpercentile(np.concatenate([raw_log_values, pqn_log_values, final_log_values]), 99.5))
    x_grid = np.linspace(x_min, x_max, 240)
    raw_curves = _build_sample_kde_curves(raw_sample_log2, x_grid)
    pqn_curves = _build_sample_kde_curves(pqn_sample_log2, x_grid)
    final_curves = _build_sample_kde_curves(final_sample_log2, x_grid)
    all_curve_max = max(
        np.nanmax(raw_curves) if raw_curves.size else 0.0,
        np.nanmax(pqn_curves) if pqn_curves.size else 0.0,
        np.nanmax(final_curves) if final_curves.size else 0.0,
    )

    def _curve_band_summary(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if curves.size == 0:
            zeros = np.zeros_like(x_grid)
            return zeros, zeros, zeros, 0.0
        median_curve = np.median(curves, axis=0)
        q25_curve = np.percentile(curves, 25, axis=0)
        q75_curve = np.percentile(curves, 75, axis=0)
        band_width = float(np.mean(q75_curve - q25_curve))
        return median_curve, q25_curve, q75_curve, band_width

    raw_median_curve, raw_q25_curve, raw_q75_curve, raw_band_width = _curve_band_summary(raw_curves)
    pqn_median_curve, pqn_q25_curve, pqn_q75_curve, pqn_band_width = _curve_band_summary(pqn_curves)
    final_median_curve, final_q25_curve, final_q75_curve, final_band_width = _curve_band_summary(final_curves)
    raw_dispersion_curve = np.std(raw_curves, axis=0) if raw_curves.size else np.zeros_like(x_grid)
    pqn_dispersion_curve = np.std(pqn_curves, axis=0) if pqn_curves.size else np.zeros_like(x_grid)
    final_dispersion_curve = np.std(final_curves, axis=0) if final_curves.size else np.zeros_like(x_grid)
    raw_dispersion_mean = float(np.mean(raw_dispersion_curve))
    pqn_dispersion_mean = float(np.mean(pqn_dispersion_curve))
    final_dispersion_mean = float(np.mean(final_dispersion_curve))
    raw_dispersion_baseline = np.where(raw_dispersion_curve > 1e-10, raw_dispersion_curve, np.nan)
    raw_dispersion_norm = np.ones_like(x_grid)
    pqn_dispersion_norm = np.divide(pqn_dispersion_curve, raw_dispersion_baseline, out=np.full_like(x_grid, np.nan), where=np.isfinite(raw_dispersion_baseline))
    final_dispersion_norm = np.divide(final_dispersion_curve, raw_dispersion_baseline, out=np.full_like(x_grid, np.nan), where=np.isfinite(raw_dispersion_baseline))
    pqn_reduction_pct = float((1.0 - np.nanmean(pqn_dispersion_norm)) * 100.0)
    final_reduction_pct = float((1.0 - np.nanmean(final_dispersion_norm)) * 100.0)

    top_capped = _compute_iqr_capping_summary(pqn_df, common_features)
    top_capped_text = "; ".join(
        f"{_format_feature_label(display_map.get(feature, feature))} ({delta:.2f})"
        for feature, delta in top_capped
    ) or "No measurable IQR clipping was reconstructed."
    iqr_changed_mask = np.abs(final_log_values - pqn_log_values) > 1e-12
    iqr_changed_cells = int(iqr_changed_mask.sum())
    iqr_changed_pct = float(iqr_changed_cells / max(iqr_changed_mask.size, 1))

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")

    retention_long = pd.DataFrame(
        {
            "step": retention_df["step"].tolist() * 2,
            "series": ["Features"] * len(retention_df) + ["Samples"] * len(retention_df),
            "retained_count": retention_df["feature_count"].tolist() + retention_df["sample_count"].tolist(),
            "label": [str(int(value)) for value in retention_df["feature_count"].tolist()] + [str(int(value)) for value in retention_df["sample_count"].tolist()],
        }
    )
    retention_csv = output_path.parent / "phase1_qc_retention_source.csv"
    retention_long.to_csv(retention_csv, index=False)

    sample_box_df = sample_intensity_df.rename(columns={"Total intensity": "Total_intensity"}).copy()
    sample_box_csv = output_path.parent / "phase1_qc_sample_intensity_source.csv"
    sample_box_df.to_csv(sample_box_csv, index=False)

    dispersion_df = pd.DataFrame(
        {
            "x_log2": np.tile(x_grid, 3),
            "stage": ["Raw"] * len(x_grid) + ["Post-PQN"] * len(x_grid) + ["Final QC"] * len(x_grid),
            "normalized_dispersion": np.concatenate([raw_dispersion_norm, pqn_dispersion_norm, final_dispersion_norm]),
            "raw_baseline": np.concatenate([raw_dispersion_norm, raw_dispersion_norm, raw_dispersion_norm]),
            "post_pqn": np.concatenate([pqn_dispersion_norm, pqn_dispersion_norm, pqn_dispersion_norm]),
            "reduction_pct": np.concatenate(
                [
                    np.full(len(x_grid), 0.0),
                    np.full(len(x_grid), pqn_reduction_pct),
                    np.full(len(x_grid), final_reduction_pct),
                ]
            ),
        }
    )
    dispersion_csv = output_path.parent / "phase1_qc_dispersion_source.csv"
    dispersion_df.to_csv(dispersion_csv, index=False)

    band_df = pd.DataFrame(
        {
            "x_log2": np.tile(x_grid, 3),
            "stage": ["Raw"] * len(x_grid) + ["Post-PQN"] * len(x_grid) + ["Final QC"] * len(x_grid),
            "median_density": np.concatenate([raw_median_curve, pqn_median_curve, final_median_curve]),
            "q25_density": np.concatenate([raw_q25_curve, pqn_q25_curve, final_q25_curve]),
            "q75_density": np.concatenate([raw_q75_curve, pqn_q75_curve, final_q75_curve]),
            "band_width": np.concatenate(
                [
                    np.full(len(x_grid), raw_band_width),
                    np.full(len(x_grid), pqn_band_width),
                    np.full(len(x_grid), final_band_width),
                ]
            ),
        }
    )
    band_csv = output_path.parent / "phase1_qc_band_source.csv"
    band_df.to_csv(band_csv, index=False)

    distribution_source_df = pd.DataFrame(
        {
            "stage": ["raw_log2", "post_pqn_log2", "post_pqn_log2_iqr"],
            "n_values": [int(raw_log_values.size), int(pqn_log_values.size), int(final_log_values.size)],
            "mean": [float(np.mean(raw_log_values)), float(np.mean(pqn_log_values)), float(np.mean(final_log_values))],
            "std": [float(np.std(raw_log_values)), float(np.std(pqn_log_values)), float(np.std(final_log_values))],
            "mean_kde_dispersion": [raw_dispersion_mean, pqn_dispersion_mean, final_dispersion_mean],
            "mean_kde_dispersion_relative_to_raw": [1.0, float(np.nanmean(pqn_dispersion_norm)), float(np.nanmean(final_dispersion_norm))],
            "mean_kde_iqr_band_width": [raw_band_width, pqn_band_width, final_band_width],
            "source_path": [str(raw_source_path), str(pqn_snapshot_path) if pqn_snapshot_path else "", str(final_snapshot_path) if final_snapshot_path else ""],
        }
    )
    distribution_source_csv = output_path.parent / "phase1_qc_distribution_source.csv"
    distribution_source_df.to_csv(distribution_source_csv, index=False)

    summary_text = (
        "QC Summary  |  "
        f"Missingness: {missingness_report.get('missing_rate_global', 0.0):.2%}  |  "
        f"IQR capping: {iqr_changed_cells} cells ({iqr_changed_pct:.2%})  |  "
        f"Dispersion reduction: Raw->PQN ↓{pqn_reduction_pct:.1f}%, Raw->Final ↓{final_reduction_pct:.1f}%  |  "
        f"Top shifts: {top_capped_text}"
    )
    summary_txt = output_path.parent / "phase1_qc_summary.txt"
    summary_txt.write_text(summary_text, encoding="utf-8")

    _run_r_qc_overview(
        retention_csv=retention_csv,
        sample_box_csv=sample_box_csv,
        dispersion_csv=dispersion_csv,
        band_csv=band_csv,
        summary_txt=summary_txt,
        output_stem=output_stem,
        title="Phase 1 QC Overview",
    )
    saved_paths = _save_r_figure_bundle(save_path)
    metadata = {
        "figure_title": "Phase 1 QC Overview",
        "phase1_latest_dir": str(latest_dir),
        "raw_data_path": str(raw_source_path),
        "auxiliary_outputs": saved_paths[1:],
        "retention_source_csv": str(retention_csv),
        "sample_intensity_source_csv": str(sample_box_csv),
        "dispersion_source_csv": str(dispersion_csv),
        "band_source_csv": str(band_csv),
        "distribution_source_csv": str(distribution_source_csv),
        "summary_text_path": str(summary_txt),
        "missing_rate_global": float(missingness_report.get("missing_rate_global", 0.0) or 0.0),
        "iqr_changed_cells": iqr_changed_cells,
        "iqr_changed_pct": iqr_changed_pct,
        "mean_kde_dispersion": {
            "raw": raw_dispersion_mean,
            "post_pqn": pqn_dispersion_mean,
            "final_qc": final_dispersion_mean,
        },
        "mean_kde_dispersion_relative_to_raw": {
            "raw": 1.0,
            "post_pqn": float(np.nanmean(pqn_dispersion_norm)),
            "final_qc": float(np.nanmean(final_dispersion_norm)),
        },
        "mean_kde_iqr_band_width": {
            "raw": raw_band_width,
            "post_pqn": pqn_band_width,
            "final_qc": final_band_width,
        },
    }
    return saved_paths[0], metadata


def _plot_panel_correlation_heatmap(
    *,
    final_df: pd.DataFrame,
    selected_features: List[str],
    feature_provenance_path: str,
    save_path: str,
    figure_title: str,
    corr_prefix: str,
    source_metadata: Dict[str, str],
    journal_theme: Dict[str, Any] | None = None,
) -> Tuple[str, Dict]:
    if len(selected_features) < 2:
        raise ValueError("At least two selected features are required to build a correlation heatmap.")

    corr = final_df[selected_features].corr(method="pearson")
    pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns, dtype=float)
    for i, feature_i in enumerate(selected_features):
        for j, feature_j in enumerate(selected_features):
            if i == j:
                pvals.iat[i, j] = 0.0
                continue
            corr_value, p_value = pearsonr(final_df[feature_i], final_df[feature_j])
            corr.iat[i, j] = corr_value
            pvals.iat[i, j] = p_value

    display_map = _load_feature_display_map(feature_provenance_path)
    resolved_feature_names = _to_phase3_display_feature_names(
        selected_features,
        feature_provenance_path=feature_provenance_path,
    )
    labels = [
        _format_feature_label(display_map.get(feature) or resolved_name or feature, max_len=28)
        for feature, resolved_name in zip(selected_features, resolved_feature_names)
    ]
    corr.index = labels
    corr.columns = labels
    pvals.index = labels
    pvals.columns = labels

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    corr_csv = output_path.parent / f"{corr_prefix}_correlation_matrix.csv"
    pvals_csv = output_path.parent / f"{corr_prefix}_correlation_pvalues.csv"
    stars_csv = output_path.parent / f"{corr_prefix}_correlation_significance.csv"
    corr.to_csv(corr_csv)
    pvals.to_csv(pvals_csv)

    star_df = pvals.copy().astype(object)
    star_df.iloc[:, :] = np.where(
        pvals.to_numpy(dtype=float) < 0.001,
        "***",
        np.where(
            pvals.to_numpy(dtype=float) < 0.01,
            "**",
            np.where(pvals.to_numpy(dtype=float) < 0.05, "*", ""),
        ),
    )
    np.fill_diagonal(star_df.values, "")
    star_df.to_csv(stars_csv)

    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    _run_r_pheatmap(corr_csv, stars_csv, output_stem, figure_title, theme_json_path)
    saved_paths = _save_r_figure_bundle(save_path)
    abs_corr = np.abs(corr.to_numpy(dtype=float))
    upper_mask = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
    upper_values = abs_corr[upper_mask]
    metadata = {
        "figure_title": figure_title,
        "theme_json_path": theme_json_path,
        "auxiliary_outputs": saved_paths[1:],
        "n_features": len(selected_features),
        "max_abs_corr": float(np.nanmax(upper_values)) if upper_values.size else 0.0,
        "n_pairs_abs_corr_ge_05": int(np.sum(upper_values >= 0.5)) if upper_values.size else 0,
        "n_pairs_abs_corr_ge_09": int(np.sum(upper_values >= 0.9)) if upper_values.size else 0,
    }
    metadata.update(source_metadata)
    return saved_paths[0], metadata


def plot_phase1_final_panel_correlation_heatmap(
    phase1_selected_features_path: str,
    feature_provenance_path: str = "",
    save_path: str = "output/figures/fig1g_phase1_final_panel_correlation_heatmap.pdf",
    journal_theme: Dict[str, Any] | None = None,
) -> Tuple[str, Dict]:
    final_csv_path = Path(phase1_selected_features_path)
    if not final_csv_path.exists():
        raise FileNotFoundError(f"Phase 1 selected features file not found: {phase1_selected_features_path}")

    final_df = _read_csv(final_csv_path)
    selected_features = [
        col for col in final_df.columns
        if col not in {"group", "Group", "id", "ID", "sample_id", "Sample_ID", "sampleid", "SampleID"}
    ]
    return _plot_panel_correlation_heatmap(
        final_df=final_df,
        selected_features=selected_features,
        feature_provenance_path=feature_provenance_path,
        save_path=save_path,
        figure_title="Phase 1 Final Panel Correlation Heatmap",
        corr_prefix="phase1_final_panel",
        source_metadata={"phase1_selected_features_path": str(final_csv_path)},
        journal_theme=journal_theme,
    )


def plot_phase2_final_panel_correlation_heatmap(
    data_path: str,
    target_column: str,
    features: List[str],
    feature_provenance_path: str = "",
    save_path: str = "output/figures/fig4g_phase2_final_panel_correlation_heatmap.pdf",
) -> Tuple[str, Dict]:
    data_csv_path = Path(data_path)
    if not data_csv_path.exists():
        raise FileNotFoundError(f"Phase 2 winner data file not found: {data_path}")

    final_df = _read_csv(data_csv_path)
    selected_features = [feature for feature in features if feature in final_df.columns]
    if len(selected_features) < len(features):
        missing = sorted(set(features) - set(selected_features))
        if missing:
            print(f"Warning: {len(missing)} phase2 winner features missing in data matrix and will be skipped: {missing}")
    return _plot_panel_correlation_heatmap(
        final_df=final_df,
        selected_features=selected_features,
        feature_provenance_path=feature_provenance_path,
        save_path=save_path,
        figure_title="Phase 2 Winner Panel Correlation Heatmap",
        corr_prefix="phase2_final_panel",
        source_metadata={
            "phase2_data_path": str(data_csv_path),
            "target_column": str(target_column),
        },
    )
