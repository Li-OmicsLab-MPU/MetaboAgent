"""
Shared helpers for semantic memory keying and coarse strategy signatures.
"""

from __future__ import annotations

from typing import Any, Dict

from .types import DatasetFingerprint


def normalize_text(value: Any) -> str:
    """Normalize free-text values used in semantic keys."""
    return str(value or "").strip().lower()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bucket_numeric(value: Any, *, small_max: int, medium_max: int) -> str:
    numeric_value = _safe_int(value)
    if numeric_value <= 0:
        return "unknown"
    if numeric_value <= small_max:
        return "small"
    if numeric_value <= medium_max:
        return "medium"
    return "large"


def infer_dominant_feature_family(fingerprint: DatasetFingerprint | Dict[str, Any] | None) -> str:
    """Infer the dominant feature family from a dataset fingerprint."""
    payload = dict(fingerprint or {})
    family_counts = {
        "pathway": _safe_int(payload.get("n_pathway_features")),
        "ratio": _safe_int(payload.get("n_ratio_features")),
        "taxonomy": _safe_int(payload.get("n_taxonomy_features")),
        "sum": _safe_int(payload.get("n_sum_features")),
    }
    dominant_family = max(family_counts, key=family_counts.get)
    if family_counts[dominant_family] <= 0:
        return "mixed"
    return dominant_family


def build_strategy_semantic_key(fingerprint: DatasetFingerprint | Dict[str, Any] | None) -> str:
    """
    Build one coarse strategy semantic key for cross-run reuse.

    This key intentionally focuses on dataset structure and clinical scenario
    rather than disease identity, so strategy memory can transfer across
    related diseases with similar data characteristics.
    """
    payload = dict(fingerprint or {})
    scenario = normalize_text(payload.get("clinical_scenario")) or "unknown_scenario"
    column_style = normalize_text(payload.get("column_naming_style")) or "unknown_style"
    dominant_family = infer_dominant_feature_family(payload)
    sample_bucket = _bucket_numeric(payload.get("n_samples"), small_max=120, medium_max=400)
    feature_bucket = _bucket_numeric(payload.get("n_features"), small_max=150, medium_max=1200)
    anchor_state = "anchor_present" if _safe_int(payload.get("n_protected_anchor_features")) > 0 else "anchor_absent"
    return "|".join(
        [
            scenario,
            column_style,
            dominant_family,
            f"samples:{sample_bucket}",
            f"features:{feature_bucket}",
            anchor_state,
        ]
    )
