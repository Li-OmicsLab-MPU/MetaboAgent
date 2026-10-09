"""
Dataset fingerprint utilities for long-term memory retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .types import DatasetFingerprint


def _read_tabular_data(data_path: str, **kwargs: Any) -> pd.DataFrame:
    """Read CSV or Excel files using the same format rules as the main pipeline."""
    lower_path = str(data_path).lower()
    if lower_path.endswith(".csv"):
        return pd.read_csv(data_path, **kwargs)
    if lower_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(data_path, **kwargs)
    raise ValueError(f"Unsupported file format for dataset fingerprint: {data_path}")


def _infer_column_naming_style(columns: list[str]) -> str:
    """Infer a coarse naming style from feature column patterns."""
    lowered = [str(col).lower() for col in columns]
    hmdb_hits = sum(1 for col in lowered if "hmdb" in col)
    kegg_hits = sum(1 for col in lowered if "c" in col and len(col) <= 10 and col[:1].isalpha())
    pathway_hits = sum(1 for col in lowered if "pathway" in col)
    ratio_hits = sum(1 for col in lowered if "ratio" in col or "/" in col)

    if hmdb_hits > 0 and hmdb_hits >= max(1, kegg_hits):
        return "hmdb_like"
    if kegg_hits > 0 and kegg_hits > hmdb_hits:
        return "kegg_like"
    if pathway_hits > 0:
        return "pathway_like"
    if ratio_hits > 0:
        return "ratio_like"
    return "mixed_or_unknown"


def _infer_feature_type_breakdown(columns: list[str], target_column: str) -> Dict[str, int]:
    """Count coarse feature families by column naming heuristics."""
    feature_columns = [str(col) for col in columns if str(col) != str(target_column)]
    lowered = [col.lower() for col in feature_columns]
    return {
        "pathway": sum(1 for col in lowered if "pathway" in col),
        "ratio": sum(1 for col in lowered if "ratio" in col or "/" in col),
        "taxonomy": sum(1 for col in lowered if "taxonomy" in col or "taxon" in col),
        "sum": sum(1 for col in lowered if col.endswith("_sum") or "sum_" in col),
    }


def build_dataset_fingerprint(
    data_path: str,
    target_column: str,
    data_summary: Optional[Dict[str, Any]] = None,
    context_variables: Optional[Dict[str, Any]] = None,
    disease_name: str = "",
    clinical_scenario: str = "",
) -> DatasetFingerprint:
    """
    Build a compact dataset fingerprint for memory retrieval.

    Prefer `data_summary` when available so callers can reuse existing metadata,
    but read the dataset when necessary for missing/zero statistics.
    """
    context_variables = context_variables or {}
    data_summary = data_summary or {}

    if data_summary and data_summary.get("columns"):
        columns = list(data_summary.get("columns", []))
        n_samples = int(data_summary.get("n_rows", 0))
        n_features = max(0, int(data_summary.get("n_cols", len(columns))) - (1 if target_column in columns else 0))
    else:
        df_meta = _read_tabular_data(data_path, nrows=5)
        columns = [str(col) for col in df_meta.columns]
        n_samples = 0
        n_features = max(0, len(columns) - (1 if target_column in columns else 0))

    df = _read_tabular_data(data_path)
    class_counts: Dict[str, int] = {}
    imbalance_ratio = 1.0
    if target_column in df.columns:
        counts = df[target_column].value_counts(dropna=False).to_dict()
        class_counts = {str(key): int(value) for key, value in counts.items()}
        if counts:
            max_count = max(counts.values())
            min_count = min(counts.values())
            imbalance_ratio = float(max_count / min_count) if min_count else float(max_count)

    feature_columns = [col for col in df.columns if str(col) != str(target_column)]
    feature_df = df[feature_columns] if feature_columns else df.iloc[:, 0:0]
    total_cells = int(feature_df.shape[0] * feature_df.shape[1]) if not feature_df.empty else 0
    missing_rate_global = float(feature_df.isna().sum().sum() / total_cells) if total_cells else 0.0

    numeric_df = feature_df.select_dtypes(include=["number"])
    numeric_cells = int(numeric_df.shape[0] * numeric_df.shape[1]) if not numeric_df.empty else 0
    zero_rate_global = float((numeric_df == 0).sum().sum() / numeric_cells) if numeric_cells else 0.0

    breakdown = _infer_feature_type_breakdown([str(col) for col in df.columns], target_column)
    protected_anchor_features = context_variables.get("protected_anchor_features", []) or []

    fingerprint: DatasetFingerprint = {
        "disease_name": str(disease_name or ""),
        "clinical_scenario": str(clinical_scenario or ""),
        "target_column": str(target_column or ""),
        "n_samples": int(len(df)) if len(df) else n_samples,
        "n_features": int(len(feature_columns)) if feature_columns else n_features,
        "class_counts": class_counts,
        "imbalance_ratio": float(imbalance_ratio),
        "missing_rate_global": float(missing_rate_global),
        "zero_rate_global": float(zero_rate_global),
        "n_pathway_features": int(breakdown["pathway"]),
        "n_ratio_features": int(breakdown["ratio"]),
        "n_taxonomy_features": int(breakdown["taxonomy"]),
        "n_sum_features": int(breakdown["sum"]),
        "n_protected_anchor_features": int(len(protected_anchor_features)),
        "column_naming_style": _infer_column_naming_style([str(col) for col in df.columns]),
        "has_hmdb_like_columns": any("hmdb" in str(col).lower() for col in df.columns),
        "has_kegg_like_columns": any(str(col).upper().startswith("C") and len(str(col)) <= 10 for col in df.columns),
    }
    return fingerprint
