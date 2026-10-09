"""
Data Analysis Tools for Metabolomics

This package groups reusable analysis helpers for the MetaboAgent pipeline.
"""

from .f_bio_v2 import (
    BioContext,
    FeatureDescriptor,
    build_bio_context,
    build_empty_bio_context,
    calculate_f_bio_v2,
    compute_anchor_link,
    compute_coverage_gain,
    compute_disease_pathway_align,
    compute_direct_prior,
    compute_feature_support,
    detect_anchor_mode,
    merge_f_bio_v2_config,
    resolve_feature_descriptor,
)
from .clinical_utility_tools import (
    build_clinical_utility_payload,
    build_threshold_metrics_table,
    select_data_driven_companion_threshold,
    simulate_resource_impact,
    summarize_decision_curve_ranges,
    summarize_threshold_recommendation,
)
from .calibration_tools import (
    apply_prevalence_correction,
    build_adjusted_probability_calibration_summary,
    build_probability_calibration_summary,
    compute_integrated_calibration_index,
    estimate_calibration_intercept_slope,
)

__all__ = [
    "BioContext",
    "FeatureDescriptor",
    "apply_prevalence_correction",
    "build_adjusted_probability_calibration_summary",
    "build_clinical_utility_payload",
    "build_bio_context",
    "build_empty_bio_context",
    "build_probability_calibration_summary",
    "build_threshold_metrics_table",
    "calculate_f_bio_v2",
    "compute_anchor_link",
    "compute_integrated_calibration_index",
    "compute_coverage_gain",
    "compute_disease_pathway_align",
    "compute_direct_prior",
    "compute_feature_support",
    "detect_anchor_mode",
    "estimate_calibration_intercept_slope",
    "merge_f_bio_v2_config",
    "resolve_feature_descriptor",
    "select_data_driven_companion_threshold",
    "simulate_resource_impact",
    "summarize_decision_curve_ranges",
    "summarize_threshold_recommendation",
]
