"""
Clinical utility helpers for scenario-aware panel evaluation.

This module focuses on deterministic, report-ready calculations derived from
out-of-fold prediction payloads. It does not perform model training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from .calibration_tools import (
    apply_prevalence_correction,
    build_adjusted_probability_calibration_summary,
    build_probability_calibration_summary,
)
from .nri_idi_tools import align_binary_prediction_payloads, extract_binary_prediction_rows


def _as_probability_array(values: Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return arr


def _as_binary_array(values: Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(list(values)).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    unique = set(arr.tolist())
    if len(unique) != 2:
        raise ValueError(f"{name} must encode a binary outcome; observed labels={sorted(unique)}")
    return arr


def _validate_binary_probability_inputs(y_true: Sequence[Any], y_score: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    y_arr = _as_binary_array(y_true, name="y_true")
    score_arr = _as_probability_array(y_score, name="y_score")
    if y_arr.shape[0] != score_arr.shape[0]:
        raise ValueError("y_true and y_score must have identical lengths.")
    return y_arr, score_arr


def _extract_binary_arrays_from_prediction_payload(
    prediction_payload: Dict[str, Any],
    *,
    positive_label: Any = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = extract_binary_prediction_rows(prediction_payload)
    raw_labels = [row["y_true"] for row in rows]
    score_values = [row["prob"] for row in rows]
    label_array = _as_binary_array(raw_labels, name="prediction_payload.y_true")
    unique_labels = set(label_array.tolist())

    if positive_label not in unique_labels:
        if unique_labels == {0, 1}:
            positive_label = 1
        else:
            raise ValueError(
                "positive_label is not present in prediction payload labels; "
                f"observed labels={sorted(unique_labels)}"
            )

    y_true = np.asarray([1 if label == positive_label else 0 for label in raw_labels], dtype=int)
    y_score = _as_probability_array(score_values, name="prediction_payload.prob")
    return y_true, y_score


def _normalize_thresholds(thresholds: Sequence[Any]) -> List[float]:
    normalized: List[float] = []
    for value in thresholds:
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < threshold < 1.0:
            normalized.append(threshold)
    unique_sorted = sorted(set(normalized))
    if not unique_sorted:
        raise ValueError("thresholds must contain at least one valid probability in (0, 1).")
    return unique_sorted


def _resolve_boolean_ranges(thresholds: Sequence[float], flags: Sequence[bool]) -> List[List[float]]:
    if len(thresholds) != len(flags):
        raise ValueError("thresholds and flags must have identical lengths.")
    ranges: List[List[float]] = []
    start: Optional[float] = None
    end: Optional[float] = None

    for threshold, flag in zip(thresholds, flags):
        if flag:
            if start is None:
                start = float(threshold)
            end = float(threshold)
            continue
        if start is not None and end is not None:
            ranges.append([start, end])
            start, end = None, None

    if start is not None and end is not None:
        ranges.append([start, end])
    return ranges


def _calculate_net_benefit(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_pred = (y_score >= threshold).astype(int)
    n_samples = max(1, len(y_true))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return float((tp / n_samples) - (fp / n_samples) * (threshold / (1.0 - threshold)))


def _resolve_effective_target_prevalence(
    *,
    observed_prevalence: float,
    scenario_definition: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    scenario = dict(scenario_definition or {})
    use_observed_prevalence = bool(scenario.get("use_observed_prevalence", True))
    target_prevalence = scenario.get("target_prevalence")
    try:
        target_prevalence_float = float(target_prevalence)
    except (TypeError, ValueError):
        target_prevalence_float = None
    if target_prevalence_float is not None and not (0.0 < target_prevalence_float < 1.0):
        target_prevalence_float = None
    if target_prevalence_float is not None:
        return target_prevalence_float
    if use_observed_prevalence:
        return float(observed_prevalence)
    return None


def _build_discrimination_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    comprehensive_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = dict(comprehensive_metrics or {})
    if "roc_auc" not in summary:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_score))
    if "auprc" not in summary:
        summary["auprc"] = float(average_precision_score(y_true, y_score))
    if "optimal_threshold" not in summary:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_j = tpr - fpr
        optimal_idx = int(np.argmax(youden_j))
        summary["optimal_threshold"] = float(thresholds[optimal_idx])
        summary["optimal_youden_j"] = float(youden_j[optimal_idx])
    return {
        "roc_auc": float(summary.get("roc_auc", 0.0) or 0.0),
        "auprc": float(summary.get("auprc", 0.0) or 0.0),
        "optimal_threshold": float(summary.get("optimal_threshold", 0.0) or 0.0),
        "optimal_youden_j": float(summary.get("optimal_youden_j", 0.0) or 0.0),
    }


def _build_calibration_status(
    probability_calibration: Dict[str, Any],
    *,
    prior_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prior = dict(prior_status or {})
    slope = float(probability_calibration.get("calibration_slope", 1.0) or 1.0)
    intercept = float(probability_calibration.get("calibration_intercept", 0.0) or 0.0)
    ece = float(probability_calibration.get("expected_calibration_error", 0.0) or 0.0)
    recalibration_recommended = abs(intercept) > 0.10 or slope < 0.90 or slope > 1.10 or ece > 0.05

    warning = prior.get("warning") or (
        "Probability calibration is assessed from internal out-of-fold predictions only "
        "and still requires external validation before deployment."
    )
    external_validation_available = bool(prior.get("external_validation_available", False))
    requires_external_validation = bool(
        prior.get("requires_external_validation", not external_validation_available)
    )
    return {
        "available": bool(probability_calibration.get("available", False)),
        "assessment_scope": str(
            prior.get("assessment_scope")
            or probability_calibration.get("assessment_scope")
            or "internal_oof_probability_assessment"
        ),
        "requires_external_validation": requires_external_validation,
        "external_validation_available": external_validation_available,
        "probability_recalibration_recommended": bool(recalibration_recommended),
        "warning": str(warning),
        **{key: value for key, value in prior.items() if key not in {"available", "assessment_scope", "warning"}},
    }


def build_threshold_metrics_table(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    thresholds: Sequence[Any],
    *,
    positive_label: Any = 1,
) -> List[Dict[str, Any]]:
    """
    Compute scenario-ready operating characteristics across prespecified thresholds.
    """
    if positive_label != 1:
        y_true = [1 if value == positive_label else 0 for value in y_true]
    y_arr, score_arr = _validate_binary_probability_inputs(y_true, y_score)
    threshold_values = _normalize_thresholds(thresholds)

    rows: List[Dict[str, Any]] = []
    n_samples = int(len(y_arr))
    for threshold in threshold_values:
        y_pred = (score_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
        flagged_rate = (tp + fp) / n_samples if n_samples > 0 else 0.0

        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "ppv": float(ppv),
                "npv": float(npv),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "flagged_rate": float(flagged_rate),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "n_samples": n_samples,
            }
        )
    return rows


def summarize_threshold_recommendation(
    threshold_metrics: Sequence[Dict[str, Any]],
    default_action_threshold: float,
    *,
    scenario_name: str = "",
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """
    Select the scenario-default threshold row for report-friendly recommendation output.
    """
    rows = [dict(row) for row in threshold_metrics if isinstance(row, dict)]
    if not rows:
        raise ValueError("threshold_metrics must not be empty.")

    default_threshold = float(default_action_threshold)
    exact_match = next((row for row in rows if abs(float(row.get("threshold", 0.0)) - default_threshold) <= tolerance), None)
    if exact_match is not None:
        matched_row = exact_match
        rationale = "scenario_default"
    else:
        matched_row = min(rows, key=lambda row: abs(float(row.get("threshold", 0.0)) - default_threshold))
        rationale = "nearest_available_threshold"

    return {
        "scenario_name": str(scenario_name),
        "default_action_threshold": default_threshold,
        "selected_threshold": float(matched_row.get("threshold", 0.0) or 0.0),
        "rationale": rationale,
        "operating_characteristics": dict(matched_row),
    }


def _threshold_in_any_range(threshold: float, ranges: Sequence[Any], *, tolerance: float = 1e-9) -> bool:
    for item in ranges:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                left = float(item[0])
                right = float(item[1])
            except (TypeError, ValueError):
                continue
            if (left - tolerance) <= threshold <= (right + tolerance):
                return True
    return False


def select_data_driven_companion_threshold(
    threshold_metrics: Sequence[Dict[str, Any]],
    default_action_threshold: float,
    *,
    decision_curve_summary: Optional[Dict[str, Any]] = None,
    scenario_name: str = "",
    scenario_intended_use: str = "",
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """
    Select a data-driven companion threshold that complements the scenario default.

    Policy:
    1. Prefer thresholds in DCA-positive treat-none and treat-all ranges.
    2. Prefer thresholds at or above the default threshold to provide a stricter companion option.
    3. Preserve acceptable sensitivity while improving specificity, using a dynamic floor:
       max(0.80, default_sensitivity - 0.10).
    4. Break ties by Youden's J, then specificity, then proximity above the default.
    """
    rows = [dict(row) for row in threshold_metrics if isinstance(row, dict)]
    if not rows:
        raise ValueError("threshold_metrics must not be empty.")

    default_summary = summarize_threshold_recommendation(
        rows,
        default_action_threshold,
        scenario_name=scenario_name,
        tolerance=tolerance,
    )
    default_row = dict(default_summary.get("operating_characteristics", {}))
    default_threshold = float(default_summary.get("selected_threshold", default_action_threshold) or default_action_threshold)
    default_sensitivity = float(default_row.get("sensitivity", 0.0) or 0.0)
    target_sensitivity = max(0.80, default_sensitivity - 0.10)

    dca = dict(decision_curve_summary or {})
    treat_none_ranges = dca.get("winner_better_than_treat_none_ranges", [])
    treat_all_ranges = dca.get("winner_better_than_treat_all_ranges", [])
    baseline_ranges = dca.get("winner_better_than_baseline_ranges", [])
    baseline_available = bool(dca.get("baseline_comparison_available", False))

    def _score_row(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        threshold = float(row.get("threshold", 0.0) or 0.0)
        specificity = float(row.get("specificity", 0.0) or 0.0)
        sensitivity = float(row.get("sensitivity", 0.0) or 0.0)
        youden = sensitivity + specificity - 1.0
        distance_above_default = -(threshold - default_threshold)
        return (youden, specificity, sensitivity, distance_above_default)

    eligible_rows = [
        row for row in rows
        if float(row.get("threshold", 0.0) or 0.0) >= (default_threshold - tolerance)
    ]
    if not eligible_rows:
        eligible_rows = list(rows)

    dca_positive_rows = [
        row for row in eligible_rows
        if _threshold_in_any_range(float(row.get("threshold", 0.0) or 0.0), treat_none_ranges, tolerance=tolerance)
        and _threshold_in_any_range(float(row.get("threshold", 0.0) or 0.0), treat_all_ranges, tolerance=tolerance)
    ]
    if not dca_positive_rows:
        dca_positive_rows = [
            row for row in eligible_rows
            if _threshold_in_any_range(float(row.get("threshold", 0.0) or 0.0), treat_none_ranges, tolerance=tolerance)
        ]

    sensitivity_constrained_rows = [
        row for row in dca_positive_rows
        if float(row.get("sensitivity", 0.0) or 0.0) >= target_sensitivity
    ]
    candidate_rows = sensitivity_constrained_rows or dca_positive_rows or eligible_rows
    selected_row = max(candidate_rows, key=_score_row)

    selected_threshold = float(selected_row.get("threshold", 0.0) or 0.0)
    dca_supported = _threshold_in_any_range(selected_threshold, treat_none_ranges, tolerance=tolerance) and (
        not treat_all_ranges or _threshold_in_any_range(selected_threshold, treat_all_ranges, tolerance=tolerance)
    )
    baseline_supported = (
        _threshold_in_any_range(selected_threshold, baseline_ranges, tolerance=tolerance)
        if baseline_available and baseline_ranges else None
    )

    rationale_parts = [
        "data_driven_companion",
        f"sensitivity_floor>={target_sensitivity:.2f}",
    ]
    if dca_supported:
        rationale_parts.append("dca_supported")
    if baseline_supported:
        rationale_parts.append("baseline_supported")

    scenario_text = str(scenario_intended_use or scenario_name or "current scenario").strip()
    narrative = (
        f"For {scenario_text}, a data-driven companion threshold of {selected_threshold:.2f} "
        f"preserves sensitivity at {float(selected_row.get('sensitivity', 0.0)):.3f} "
        f"while improving specificity to {float(selected_row.get('specificity', 0.0)):.3f}."
    )
    if dca_supported:
        narrative += " This threshold also falls within a positive net-benefit range on decision-curve analysis."

    return {
        "scenario_name": str(scenario_name),
        "default_action_threshold": default_threshold,
        "target_sensitivity_floor": float(target_sensitivity),
        "selected_threshold": selected_threshold,
        "rationale": ";".join(rationale_parts),
        "dca_supported": bool(dca_supported),
        "baseline_supported": baseline_supported,
        "operating_characteristics": dict(selected_row),
        "narrative": narrative,
    }


def summarize_decision_curve_ranges(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    thresholds: Sequence[Any],
    *,
    baseline_scores: Optional[Sequence[Any]] = None,
    positive_label: Any = 1,
    target_prevalence: Optional[float] = None,
    summary_type: str = "legacy_absolute",
) -> Dict[str, Any]:
    """
    Summarize threshold ranges where the winner panel provides superior net benefit.
    """
    if positive_label != 1:
        y_true = [1 if value == positive_label else 0 for value in y_true]
    y_arr, score_arr = _validate_binary_probability_inputs(y_true, y_score)
    threshold_values = _normalize_thresholds(thresholds)

    prevalence = float(np.mean(y_arr))
    effective_prevalence = float(target_prevalence) if target_prevalence is not None else prevalence
    winner_better_than_treat_all: List[bool] = []
    winner_better_than_treat_none: List[bool] = []
    winner_net_benefits: List[float] = []
    baseline_net_benefits: List[float] = []

    baseline_arr: Optional[np.ndarray] = None
    if baseline_scores is not None:
        baseline_arr = _as_probability_array(baseline_scores, name="baseline_scores")
        if baseline_arr.shape[0] != score_arr.shape[0]:
            raise ValueError("baseline_scores must align with y_true and y_score.")

    for threshold in threshold_values:
        winner_nb = _calculate_net_benefit(y_arr, score_arr, threshold)
        treat_all_nb = float(effective_prevalence - (1.0 - effective_prevalence) * (threshold / (1.0 - threshold)))
        treat_none_nb = 0.0

        winner_net_benefits.append(winner_nb)
        winner_better_than_treat_all.append(winner_nb > treat_all_nb)
        winner_better_than_treat_none.append(winner_nb > treat_none_nb)

        if baseline_arr is not None:
            baseline_net_benefits.append(_calculate_net_benefit(y_arr, baseline_arr, threshold))

    peak_idx = int(np.argmax(winner_net_benefits))
    payload = {
        "summary_type": str(summary_type),
        "study_prevalence": prevalence,
        "target_prevalence": target_prevalence,
        "transport_applied": bool(target_prevalence is not None),
        "threshold_grid_min": float(threshold_values[0]),
        "threshold_grid_max": float(threshold_values[-1]),
        "peak_net_benefit_threshold": float(threshold_values[peak_idx]),
        "peak_net_benefit": float(winner_net_benefits[peak_idx]),
        "winner_better_than_treat_all_ranges": _resolve_boolean_ranges(threshold_values, winner_better_than_treat_all),
        "winner_better_than_treat_none_ranges": _resolve_boolean_ranges(threshold_values, winner_better_than_treat_none),
        "baseline_comparison_available": baseline_arr is not None,
    }

    if baseline_arr is not None:
        payload["winner_better_than_baseline_ranges"] = _resolve_boolean_ranges(
            threshold_values,
            [winner > baseline for winner, baseline in zip(winner_net_benefits, baseline_net_benefits)],
        )
    else:
        payload["winner_better_than_baseline_ranges"] = []

    return payload


def simulate_resource_impact(
    y_true: Sequence[Any],
    y_score: Sequence[Any],
    threshold: float,
    *,
    screen_population_size: int,
    panel_test_cost_per_person: float,
    confirmatory_test_cost_per_person: float = 0.0,
    baseline_scores: Optional[Sequence[Any]] = None,
    positive_label: Any = 1,
    baseline_name: str = "phase1_panel",
) -> Dict[str, Any]:
    """
    Simulate screening workload and rough resource impact per fixed population size.
    """
    if positive_label != 1:
        y_true = [1 if value == positive_label else 0 for value in y_true]
    y_arr, score_arr = _validate_binary_probability_inputs(y_true, y_score)
    threshold_value = float(threshold)
    if not (0.0 < threshold_value < 1.0):
        raise ValueError("threshold must be in (0, 1).")

    population_size = max(1, int(screen_population_size))
    panel_cost = float(panel_test_cost_per_person)
    confirmatory_cost = float(confirmatory_test_cost_per_person)

    y_pred = (score_arr >= threshold_value).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_arr == 1)))
    fp = int(np.sum((y_pred == 1) & (y_arr == 0)))
    flagged_rate = float(np.mean(y_pred))
    true_positive_rate = float(tp / len(y_arr))

    confirmatory_tests_triggered = int(round(flagged_rate * population_size))
    high_risk_identified = int(round(true_positive_rate * population_size))
    panel_test_cost_total = float(population_size * panel_cost)
    confirmatory_test_cost_total = float(confirmatory_tests_triggered * confirmatory_cost)
    total_screening_cost = panel_test_cost_total + confirmatory_test_cost_total

    payload: Dict[str, Any] = {
        "population_size": population_size,
        "threshold": threshold_value,
        "panel_test_cost_total": panel_test_cost_total,
        "confirmatory_tests_triggered": confirmatory_tests_triggered,
        "confirmatory_test_cost_total": confirmatory_test_cost_total,
        "total_screening_cost": total_screening_cost,
        "high_risk_identified": high_risk_identified,
        "observed_flagged_rate": flagged_rate,
        "observed_true_positive_rate": true_positive_rate,
        "baseline_name": str(baseline_name),
        "additional_high_risk_identified_vs_baseline": None,
        "additional_cost_vs_baseline": None,
        "cost_per_additional_high_risk_identified": None,
    }

    if baseline_scores is not None:
        baseline_arr = _as_probability_array(baseline_scores, name="baseline_scores")
        if baseline_arr.shape[0] != score_arr.shape[0]:
            raise ValueError("baseline_scores must align with y_true and y_score.")

        baseline_pred = (baseline_arr >= threshold_value).astype(int)
        baseline_tp = int(np.sum((baseline_pred == 1) & (y_arr == 1)))
        baseline_flagged_rate = float(np.mean(baseline_pred))
        baseline_true_positive_rate = float(baseline_tp / len(y_arr))
        baseline_confirmatory_count = int(round(baseline_flagged_rate * population_size))
        baseline_total_cost = panel_test_cost_total + float(baseline_confirmatory_count * confirmatory_cost)

        delta_identified = int(round((true_positive_rate - baseline_true_positive_rate) * population_size))
        delta_cost = float(total_screening_cost - baseline_total_cost)
        payload["additional_high_risk_identified_vs_baseline"] = delta_identified
        payload["additional_cost_vs_baseline"] = delta_cost
        if delta_identified > 0:
            payload["cost_per_additional_high_risk_identified"] = float(delta_cost / delta_identified)

    return payload


def build_clinical_utility_payload(
    *,
    winner_eval: Dict[str, Any],
    baseline_eval: Optional[Dict[str, Any]],
    scenario_definition: Dict[str, Any],
    calibration_status: Optional[Dict[str, Any]] = None,
    positive_label: Any = 1,
) -> Dict[str, Any]:
    """
    Build a canonical clinical utility payload from winner/baseline OOF predictions.
    """
    if not isinstance(winner_eval, dict) or not winner_eval.get("cv_predictions"):
        raise ValueError("winner_eval must contain a non-empty cv_predictions payload.")

    scenario = dict(scenario_definition or {})
    thresholds = scenario.get("risk_thresholds", [])
    if not thresholds:
        raise ValueError("scenario_definition must include non-empty risk_thresholds.")

    winner_y_true, winner_y_score = _extract_binary_arrays_from_prediction_payload(
        winner_eval["cv_predictions"],
        positive_label=positive_label,
    )

    aligned_baseline_scores: Optional[np.ndarray] = None
    baseline_name = "phase1_panel"
    if isinstance(baseline_eval, dict) and baseline_eval.get("cv_predictions"):
        aligned = align_binary_prediction_payloads(
            baseline_eval["cv_predictions"],
            winner_eval["cv_predictions"],
            positive_label=positive_label,
        )
        winner_y_true = aligned.y_true
        winner_y_score = aligned.p_new
        aligned_baseline_scores = aligned.p_old
        baseline_name = str(
            baseline_eval.get("panel_name")
            or baseline_eval.get("selected_model")
            or baseline_name
        )

    threshold_metrics_table = build_threshold_metrics_table(
        winner_y_true,
        winner_y_score,
        thresholds,
        positive_label=1,
    )
    default_threshold = float(
        scenario.get("default_action_threshold")
        if scenario.get("default_action_threshold") is not None
        else threshold_metrics_table[0]["threshold"]
    )
    discrimination_summary = _build_discrimination_summary(
        winner_y_true,
        winner_y_score,
        comprehensive_metrics=winner_eval.get("comprehensive_metrics"),
    )
    observed_prevalence = float(np.mean(winner_y_true))
    effective_target_prevalence = _resolve_effective_target_prevalence(
        observed_prevalence=observed_prevalence,
        scenario_definition=scenario,
    )
    decision_curve_relative = summarize_decision_curve_ranges(
        winner_y_true,
        winner_y_score,
        thresholds,
        baseline_scores=aligned_baseline_scores,
        positive_label=1,
        target_prevalence=None,
        summary_type="relative_case_control_comparison",
    )
    decision_curve_adjusted = summarize_decision_curve_ranges(
        winner_y_true,
        apply_prevalence_correction(
            winner_y_score,
            study_prevalence=observed_prevalence,
            target_prevalence=effective_target_prevalence,
        ) if effective_target_prevalence is not None and abs(effective_target_prevalence - observed_prevalence) > 1e-12 else winner_y_score,
        thresholds,
        baseline_scores=(
            apply_prevalence_correction(
                aligned_baseline_scores,
                study_prevalence=observed_prevalence,
                target_prevalence=effective_target_prevalence,
            )
            if aligned_baseline_scores is not None and effective_target_prevalence is not None and abs(effective_target_prevalence - observed_prevalence) > 1e-12
            else aligned_baseline_scores
        ),
        positive_label=1,
        target_prevalence=effective_target_prevalence if effective_target_prevalence is not None and abs(effective_target_prevalence - observed_prevalence) > 1e-12 else None,
        summary_type="absolute_prevalence_adjusted" if effective_target_prevalence is not None and abs(effective_target_prevalence - observed_prevalence) > 1e-12 else "observed_prevalence_absolute",
    )

    resource_assumptions = dict(scenario.get("resource_assumptions", {}))
    resource_impact = simulate_resource_impact(
        winner_y_true,
        winner_y_score,
        default_threshold,
        screen_population_size=int(resource_assumptions.get("screen_population_size", 1000)),
        panel_test_cost_per_person=float(resource_assumptions.get("panel_test_cost_per_person", 0.0)),
        confirmatory_test_cost_per_person=float(
            resource_assumptions.get("confirmatory_test_cost_per_person", 0.0)
        ),
        baseline_scores=aligned_baseline_scores,
        positive_label=1,
        baseline_name=str(resource_assumptions.get("baseline_strategy", baseline_name)),
    )

    probability_calibration = build_probability_calibration_summary(
        winner_y_true,
        winner_y_score,
        n_bins=10,
        strategy="quantile",
    )
    adjusted_probability_calibration = build_adjusted_probability_calibration_summary(
        winner_y_true,
        winner_y_score,
        study_prevalence=observed_prevalence,
        target_prevalence=effective_target_prevalence if effective_target_prevalence is not None and abs(effective_target_prevalence - observed_prevalence) > 1e-12 else None,
        n_bins=10,
        strategy="quantile",
    )
    calibration_status = _build_calibration_status(
        adjusted_probability_calibration if adjusted_probability_calibration.get("available") else probability_calibration,
        prior_status=calibration_status,
    )
    recommended_threshold_summary = summarize_threshold_recommendation(
        threshold_metrics_table,
        default_threshold,
        scenario_name=str(scenario.get("name", scenario.get("key", ""))),
    )
    data_driven_companion_threshold_summary = select_data_driven_companion_threshold(
        threshold_metrics_table,
        default_threshold,
        decision_curve_summary=decision_curve_relative,
        scenario_name=str(scenario.get("name", scenario.get("key", ""))),
        scenario_intended_use=str(scenario.get("intended_use", "")),
    )

    primary_calibration = adjusted_probability_calibration if adjusted_probability_calibration.get("available") else probability_calibration
    reporting_recommendation = {
        "primary_dca_field": "decision_curve_relative",
        "primary_calibration_field": (
            "probability_calibration_adjusted" if adjusted_probability_calibration.get("available") else "probability_calibration_raw"
        ),
        "suppress_ece_in_main_report": not bool(scenario.get("report_ece", False)),
    }

    return {
        "schema_version": "phase2.clinical_utility.v2",
        "scenario": {
            "key": str(scenario.get("key", "")),
            "name": str(scenario.get("name", scenario.get("key", ""))),
            "description": str(scenario.get("description", "")),
            "intended_use": str(scenario.get("intended_use", "")),
            "clinical_action": str(scenario.get("clinical_action", "")),
            "default_action_threshold": default_threshold,
            "risk_thresholds": [float(value) for value in _normalize_thresholds(thresholds)],
            "study_design": str(scenario.get("study_design", "case_control")),
            "use_observed_prevalence": bool(scenario.get("use_observed_prevalence", True)),
            "target_prevalence": effective_target_prevalence,
            "prevalence_source": str(scenario.get("prevalence_source", "")),
            "prevalence_reference_population": str(scenario.get("prevalence_reference_population", "")),
            "report_ece": bool(scenario.get("report_ece", False)),
        },
        "models": {
            "winner_panel": {
                "name": str(winner_eval.get("panel_name", "Phase2 Winner")),
                "selected_model": str(winner_eval.get("selected_model", "")),
                "feature_count": int(len(winner_eval.get("features", []) or [])),
                "features": list(winner_eval.get("features", []) or []),
            },
            "baseline_panel": {
                "name": baseline_name,
                "selected_model": str((baseline_eval or {}).get("selected_model", "")),
                "feature_count": int(len((baseline_eval or {}).get("features", []) or [])),
                "features": list((baseline_eval or {}).get("features", []) or []),
            } if baseline_eval else {},
        },
        "prevalence_context": {
            "study_design": str(scenario.get("study_design", "case_control")),
            "study_prevalence": observed_prevalence,
            "target_prevalence": effective_target_prevalence,
            "prevalence_source": str(scenario.get("prevalence_source", "")),
            "prevalence_reference_population": str(scenario.get("prevalence_reference_population", "")),
            "prevalence_transport_applied": bool(
                effective_target_prevalence is not None and abs(effective_target_prevalence - observed_prevalence) > 1e-12
            ),
        },
        "discrimination_summary": discrimination_summary,
        "threshold_metrics_table": threshold_metrics_table,
        "recommended_threshold_summary": recommended_threshold_summary,
        "data_driven_companion_threshold_summary": data_driven_companion_threshold_summary,
        "decision_curve_relative": decision_curve_relative,
        "decision_curve_adjusted": decision_curve_adjusted,
        "decision_curve_summary": decision_curve_relative,
        "resource_impact_per_1000": resource_impact,
        "probability_calibration_raw": probability_calibration,
        "probability_calibration_adjusted": adjusted_probability_calibration,
        "probability_calibration": primary_calibration,
        "calibration_status": dict(calibration_status),
        "reporting_recommendation": reporting_recommendation,
    }


__all__ = [
    "build_clinical_utility_payload",
    "build_threshold_metrics_table",
    "simulate_resource_impact",
    "select_data_driven_companion_threshold",
    "summarize_decision_curve_ranges",
    "summarize_threshold_recommendation",
]
