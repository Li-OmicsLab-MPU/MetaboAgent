"""
Probability calibration assessment helpers.

These utilities quantify calibration quality from out-of-fold probabilities
without retraining or post-hoc recalibration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from .nri_idi_tools import extract_binary_prediction_rows


def _as_binary_array(values: Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(list(values)).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    unique = set(arr.tolist())
    if len(unique) != 2:
        raise ValueError(f"{name} must encode a binary outcome; observed labels={sorted(unique)}")
    return arr


def _as_probability_array(values: Sequence[Any], *, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")
    return arr


def _validate_binary_probability_inputs(y_true: Sequence[Any], y_prob: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    y_arr = _as_binary_array(y_true, name="y_true")
    prob_arr = _as_probability_array(y_prob, name="y_prob")
    if y_arr.shape[0] != prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have identical lengths.")
    return y_arr.astype(int), prob_arr


def _clip_probabilities(y_prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)


def _probabilities_to_logit_feature(y_prob: Sequence[Any], eps: float = 1e-6) -> np.ndarray:
    clipped = _clip_probabilities(np.asarray(list(y_prob), dtype=float), eps=eps)
    logits = np.log(clipped / (1.0 - clipped))
    return logits.reshape(-1, 1)


def _quantile_nearest(values: np.ndarray, q: float) -> float:
    """
    Compute a nearest-neighbor quantile across NumPy versions.

    NumPy >= 1.22 uses the `method` keyword, while older releases expect
    `interpolation`. The project supports NumPy 1.21+, so keep both paths.
    """
    arr = np.asarray(values, dtype=float)
    try:
        return float(np.quantile(arr, q, method="nearest"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="nearest"))


def _validate_prediction_payload(prediction_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = extract_binary_prediction_rows(prediction_payload)
    seen_keys = set()
    for row in rows:
        key = (row["sample_id"], row["fold"])
        if key in seen_keys:
            raise ValueError(f"Duplicate prediction row detected for key={key}.")
        seen_keys.add(key)
    return rows


def _coerce_prediction_payload_metadata(
    prediction_payload: Dict[str, Any],
    *,
    n_samples: int,
) -> Dict[str, Any]:
    payload = dict(prediction_payload or {})
    return {
        "mean_score": payload.get("mean_score"),
        "label_names": list(payload.get("label_names", []) or []),
        "n_samples": int(payload.get("n_samples", n_samples) or n_samples),
        "k_folds": int(payload.get("k_folds", 0) or 0),
        "metric": str(payload.get("metric", "auprc") or "auprc"),
    }


def _row_probability_summary(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=int)
    y_prob = np.asarray([float(row["prob"]) for row in rows], dtype=float)
    return _validate_binary_probability_inputs(y_true, y_prob)


def fit_platt_recalibrator(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    *,
    clip_min: float = 1e-6,
) -> Dict[str, Any]:
    """
    Fit a Platt-style recalibrator on raw predicted probabilities.

    The model is fitted on logit(probability) using logistic regression and the
    learned intercept/slope pair is returned as a lightweight serializable dict.
    """
    y_arr, prob_arr = _validate_binary_probability_inputs(y_true, y_prob)
    features = _probabilities_to_logit_feature(prob_arr, eps=clip_min)

    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
        )
    except TypeError:
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
        )

    model.fit(features, y_arr)
    return {
        "method": "platt",
        "clip_min": float(clip_min),
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0][0]),
        "n_training_samples": int(len(y_arr)),
    }


def apply_platt_recalibrator(
    y_prob: Sequence[Any],
    recalibrator: Dict[str, Any],
    *,
    clip_min: Optional[float] = None,
) -> np.ndarray:
    """
    Apply a previously fitted Platt recalibrator to raw probabilities.
    """
    if not isinstance(recalibrator, dict):
        raise ValueError("recalibrator must be a dict produced by fit_platt_recalibrator.")
    if str(recalibrator.get("method", "")).strip().lower() != "platt":
        raise ValueError("Only 'platt' recalibrators are currently supported.")

    slope = float(recalibrator.get("slope", 1.0) or 1.0)
    intercept = float(recalibrator.get("intercept", 0.0) or 0.0)
    eps = float(clip_min if clip_min is not None else recalibrator.get("clip_min", 1e-6) or 1e-6)

    logits = _probabilities_to_logit_feature(y_prob, eps=eps).reshape(-1)
    calibrated = 1.0 / (1.0 + np.exp(-(intercept + slope * logits)))
    return _clip_probabilities(calibrated, eps=eps)


def build_recalibrated_prediction_payload(
    raw_prediction_payload: Dict[str, Any],
    calibrated_probabilities: Sequence[Any],
) -> Dict[str, Any]:
    """
    Build a cv-prediction-compatible payload using recalibrated class-1 probabilities.
    """
    rows = _validate_prediction_payload(raw_prediction_payload)
    calibrated = _as_probability_array(calibrated_probabilities, name="calibrated_probabilities")
    if len(rows) != len(calibrated):
        raise ValueError("calibrated_probabilities must align 1:1 with prediction rows.")

    metadata = _coerce_prediction_payload_metadata(raw_prediction_payload, n_samples=len(rows))
    payload_rows: List[Dict[str, Any]] = []
    for row, prob in zip(rows, calibrated):
        payload_rows.append(
            {
                "sample_id": row["sample_id"],
                "fold": row["fold"],
                "true_label": row["y_true"],
                "true_label_encoded": int(row["y_true"]),
                "pred_proba_class0": float(1.0 - prob),
                "pred_proba_class1": float(prob),
            }
        )

    return {
        "predictions": payload_rows,
        **metadata,
    }


def _build_threshold_migration_summary(
    raw_probabilities: Sequence[Any],
    calibrated_probabilities: Sequence[Any],
    *,
    source_thresholds: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    thresholds = []
    for value in source_thresholds or []:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < numeric < 1.0:
            thresholds.append(numeric)
    unique_thresholds = sorted(set(thresholds))

    raw_arr = _as_probability_array(raw_probabilities, name="raw_probabilities")
    calibrated_arr = _as_probability_array(calibrated_probabilities, name="calibrated_probabilities")
    if raw_arr.shape[0] != calibrated_arr.shape[0]:
        raise ValueError("raw_probabilities and calibrated_probabilities must have identical lengths.")

    mappings: List[Dict[str, float]] = []
    for threshold in unique_thresholds:
        quantile = float(np.mean(raw_arr <= threshold))
        calibrated_equivalent = _quantile_nearest(calibrated_arr, quantile)
        mappings.append(
            {
                "raw_threshold": float(threshold),
                "calibrated_equivalent_threshold": calibrated_equivalent,
            }
        )

    return {
        "source_thresholds": unique_thresholds,
        "mapping_rule": "nearest_predicted_risk_quantile",
        "mappings": mappings,
    }


def build_probability_recalibration_artifact(
    *,
    raw_prediction_payload: Dict[str, Any],
    calibrated_prediction_payload: Dict[str, Any],
    recalibration_status: Dict[str, Any],
    recalibration_config: Dict[str, Any],
    generated_at: str,
    panel_name: str,
    data_path: str,
    target_column: str,
    source_thresholds: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the canonical probability recalibration artifact.
    """
    raw_rows = _validate_prediction_payload(raw_prediction_payload)
    calibrated_rows = _validate_prediction_payload(calibrated_prediction_payload)
    if len(raw_rows) != len(calibrated_rows):
        raise ValueError("raw_prediction_payload and calibrated_prediction_payload must align 1:1.")

    raw_y, raw_prob = _row_probability_summary(raw_rows)
    calibrated_y, calibrated_prob = _row_probability_summary(calibrated_rows)
    if np.any(raw_y != calibrated_y):
        raise ValueError("raw and calibrated prediction payloads have mismatched labels.")

    raw_calibration = build_probability_calibration_summary(raw_y, raw_prob)
    calibrated_calibration = build_probability_calibration_summary(calibrated_y, calibrated_prob)
    raw_slope_distance = abs(float(raw_calibration.get("calibration_slope", 1.0)) - 1.0)
    calibrated_slope_distance = abs(float(calibrated_calibration.get("calibration_slope", 1.0)) - 1.0)

    improvement_summary = {
        "delta_brier_score": float(calibrated_calibration["brier_score"] - raw_calibration["brier_score"]),
        "delta_expected_calibration_error": float(
            calibrated_calibration["expected_calibration_error"] - raw_calibration["expected_calibration_error"]
        ),
        "delta_max_calibration_error": float(
            calibrated_calibration["max_calibration_error"] - raw_calibration["max_calibration_error"]
        ),
        "delta_abs_intercept": float(
            abs(calibrated_calibration["calibration_intercept"]) - abs(raw_calibration["calibration_intercept"])
        ),
        "delta_abs_slope_distance_to_1": float(calibrated_slope_distance - raw_slope_distance),
        "improved": bool(
            calibrated_calibration["brier_score"] <= raw_calibration["brier_score"]
            and calibrated_calibration["expected_calibration_error"] <= raw_calibration["expected_calibration_error"]
        ),
    }

    source_metadata = _coerce_prediction_payload_metadata(raw_prediction_payload, n_samples=len(raw_rows))
    threshold_summary = _build_threshold_migration_summary(
        raw_prob,
        calibrated_prob,
        source_thresholds=source_thresholds,
    )

    return {
        "schema_version": "phase2.probability_recalibration.v1",
        "generated_at": str(generated_at),
        "panel_name": str(panel_name),
        "data_path": str(data_path),
        "target_column": str(target_column),
        "recalibration_status": dict(recalibration_status or {}),
        "recalibration_config": dict(recalibration_config or {}),
        "source_payload_summary": {
            "source_payload_name": "winner_eval.cv_predictions",
            "source_payload_available": True,
            "n_samples": source_metadata["n_samples"],
            "k_folds": source_metadata["k_folds"],
            "metric": source_metadata["metric"],
            "label_names": source_metadata["label_names"],
        },
        "raw_prediction_payload": raw_prediction_payload,
        "calibrated_prediction_payload": calibrated_prediction_payload,
        "raw_probability_calibration": raw_calibration,
        "calibrated_probability_calibration": calibrated_calibration,
        "calibration_improvement_summary": improvement_summary,
        "threshold_migration_summary": threshold_summary,
    }


def run_cross_fitted_probability_recalibration(
    raw_prediction_payload: Dict[str, Any],
    *,
    method: str = "platt",
    clip_min: float = 1e-6,
) -> Dict[str, Any]:
    """
    Generate row-aligned recalibrated probabilities using fold-wise cross-fitting.

    Each outer validation fold is recalibrated using predictions from all other
    folds, preventing the calibrator from seeing the target fold during fitting.
    """
    normalized_method = str(method or "platt").strip().lower()
    if normalized_method != "platt":
        raise ValueError("Only 'platt' cross-fitted recalibration is currently supported.")

    rows = _validate_prediction_payload(raw_prediction_payload)
    metadata = _coerce_prediction_payload_metadata(raw_prediction_payload, n_samples=len(rows))
    unique_folds = sorted({row["fold"] for row in rows}, key=lambda value: (str(type(value)), str(value)))
    if len(unique_folds) < 2:
        raise ValueError("Cross-fitted recalibration requires predictions from at least 2 folds.")

    calibrated_by_key: Dict[Tuple[Any, Any], float] = {}
    fitted_calibrators: List[Dict[str, Any]] = []

    for target_fold in unique_folds:
        train_rows = [row for row in rows if row["fold"] != target_fold]
        test_rows = [row for row in rows if row["fold"] == target_fold]
        if not train_rows or not test_rows:
            raise ValueError(f"Fold {target_fold!r} does not have both train-side and test-side rows.")

        y_train, prob_train = _row_probability_summary(train_rows)
        recalibrator = fit_platt_recalibrator(y_train, prob_train, clip_min=clip_min)
        recalibrator["applied_to_fold"] = target_fold
        fitted_calibrators.append(recalibrator)

        prob_test = [row["prob"] for row in test_rows]
        calibrated_test = apply_platt_recalibrator(prob_test, recalibrator, clip_min=clip_min)
        for row, calibrated_prob in zip(test_rows, calibrated_test):
            calibrated_by_key[(row["sample_id"], row["fold"])] = float(calibrated_prob)

    calibrated_probabilities = [
        calibrated_by_key[(row["sample_id"], row["fold"])]
        for row in rows
    ]
    calibrated_payload = build_recalibrated_prediction_payload(raw_prediction_payload, calibrated_probabilities)

    return {
        "method": normalized_method,
        "strategy": "cross_fitted",
        "raw_prediction_payload": raw_prediction_payload,
        "calibrated_prediction_payload": calibrated_payload,
        "fitted_calibrators": fitted_calibrators,
        "n_folds_recalibrated": len(unique_folds),
        "n_samples": metadata["n_samples"],
        "clip_min": float(clip_min),
    }


def _compute_curve_bins(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int, strategy: str) -> List[Dict[str, Any]]:
    clipped = _clip_probabilities(y_prob)
    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(clipped, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.maximum.accumulate(bin_edges)
    else:
        strategy = "uniform"
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Keep only strictly increasing edges to avoid empty duplicate bins.
    unique_edges = [float(bin_edges[0])]
    for edge in bin_edges[1:]:
        edge_float = float(edge)
        if edge_float > unique_edges[-1]:
            unique_edges.append(edge_float)
    if unique_edges[-1] < 1.0:
        unique_edges[-1] = 1.0
    if len(unique_edges) < 2:
        unique_edges = [0.0, 1.0]

    rows: List[Dict[str, Any]] = []
    for idx in range(len(unique_edges) - 1):
        left = float(unique_edges[idx])
        right = float(unique_edges[idx + 1])
        if idx == len(unique_edges) - 2:
            mask = (clipped >= left) & (clipped <= right)
        else:
            mask = (clipped >= left) & (clipped < right)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_index": idx,
                    "lower_bound": left,
                    "upper_bound": right,
                    "sample_count": 0,
                    "mean_predicted_probability": None,
                    "observed_event_rate": None,
                    "absolute_calibration_error": None,
                }
            )
            continue

        mean_prob = float(np.mean(clipped[mask]))
        event_rate = float(np.mean(y_true[mask]))
        rows.append(
            {
                "bin_index": idx,
                "lower_bound": left,
                "upper_bound": right,
                "sample_count": count,
                "mean_predicted_probability": mean_prob,
                "observed_event_rate": event_rate,
                "absolute_calibration_error": float(abs(event_rate - mean_prob)),
            }
        )
    return rows


def estimate_calibration_intercept_slope(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
) -> Dict[str, float]:
    """
    Estimate calibration intercept and slope from logit(probability).

    A perfectly calibrated model is expected to have intercept ~= 0 and slope ~= 1.
    """
    y_arr, prob_arr = _validate_binary_probability_inputs(y_true, y_prob)
    clipped = _clip_probabilities(prob_arr)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
        )
    except TypeError:
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
        )

    model.fit(logits, y_arr)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def apply_prevalence_correction(
    y_prob: Sequence[Any],
    *,
    study_prevalence: float,
    target_prevalence: float,
    clip_min: float = 1e-6,
) -> np.ndarray:
    """
    Transport predicted probabilities from the study prevalence to a target prevalence.
    """
    study_prev = float(study_prevalence)
    target_prev = float(target_prevalence)
    if not (0.0 < study_prev < 1.0):
        raise ValueError("study_prevalence must be in (0, 1).")
    if not (0.0 < target_prev < 1.0):
        raise ValueError("target_prevalence must be in (0, 1).")

    clipped = _clip_probabilities(_as_probability_array(y_prob, name="y_prob"), eps=clip_min)
    raw_logits = np.log(clipped / (1.0 - clipped))
    prevalence_shift = np.log(target_prev / (1.0 - target_prev)) - np.log(study_prev / (1.0 - study_prev))
    corrected = 1.0 / (1.0 + np.exp(-(raw_logits + prevalence_shift)))
    return _clip_probabilities(corrected, eps=clip_min)


def compute_integrated_calibration_index(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
) -> float:
    """
    Approximate ICI using isotonic smoothing to avoid binning instability.
    """
    from sklearn.isotonic import IsotonicRegression

    y_arr, prob_arr = _validate_binary_probability_inputs(y_true, y_prob)
    order = np.argsort(prob_arr)
    sorted_prob = prob_arr[order]
    sorted_y = y_arr[order]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    smoothed = iso.fit_transform(sorted_prob, sorted_y)
    return float(np.mean(np.abs(smoothed - sorted_prob)))


def build_probability_calibration_summary(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Dict[str, Any]:
    """
    Build a report-ready probability calibration summary from OOF probabilities.
    """
    y_arr, prob_arr = _validate_binary_probability_inputs(y_true, y_prob)
    n_bins = max(2, int(n_bins))
    strategy = str(strategy or "quantile").strip().lower()
    if strategy not in {"uniform", "quantile"}:
        strategy = "quantile"

    prevalence = float(np.mean(y_arr))
    brier = float(brier_score_loss(y_arr, prob_arr))
    null_brier = float(np.mean((y_arr - prevalence) ** 2))
    brier_skill_score = float(1.0 - (brier / null_brier)) if null_brier > 0 else 0.0

    intercept_slope = estimate_calibration_intercept_slope(y_arr, prob_arr)
    curve_rows = _compute_curve_bins(y_arr, prob_arr, n_bins=n_bins, strategy=strategy)
    valid_curve_rows = [row for row in curve_rows if row.get("sample_count", 0) > 0]

    total_samples = max(1, len(y_arr))
    expected_calibration_error = float(
        sum(
            float(row["sample_count"]) / total_samples * float(row["absolute_calibration_error"])
            for row in valid_curve_rows
            if row.get("absolute_calibration_error") is not None
        )
    )
    max_calibration_error = float(
        max(
            [float(row["absolute_calibration_error"]) for row in valid_curve_rows if row.get("absolute_calibration_error") is not None]
            or [0.0]
        )
    )

    frac_pos, mean_pred = calibration_curve(y_arr, prob_arr, n_bins=n_bins, strategy=strategy)
    return {
        "available": True,
        "assessment_scope": "internal_oof_probability_assessment",
        "n_samples": int(len(y_arr)),
        "prevalence": prevalence,
        "n_bins": n_bins,
        "binning_strategy": strategy,
        "brier_score": brier,
        "brier_skill_score": brier_skill_score,
        "expected_calibration_error": expected_calibration_error,
        "max_calibration_error": max_calibration_error,
        "calibration_intercept": intercept_slope["calibration_intercept"],
        "calibration_slope": intercept_slope["calibration_slope"],
        "calibration_curve": [
            {
                "mean_predicted_probability": float(pred),
                "observed_event_rate": float(obs),
            }
            for pred, obs in zip(mean_pred, frac_pos)
        ],
        "calibration_bins": curve_rows,
    }


def build_adjusted_probability_calibration_summary(
    y_true: Sequence[Any],
    y_prob: Sequence[Any],
    *,
    study_prevalence: Optional[float],
    target_prevalence: Optional[float],
    n_bins: int = 10,
    strategy: str = "quantile",
    clip_min: float = 1e-6,
) -> Dict[str, Any]:
    """
    Build a target-prevalence-adjusted calibration summary when epidemiologic prevalence is known.
    """
    y_arr, prob_arr = _validate_binary_probability_inputs(y_true, y_prob)
    observed_prevalence = float(np.mean(y_arr))
    study_prev = float(study_prevalence) if study_prevalence is not None else observed_prevalence
    target_prev = float(target_prevalence) if target_prevalence is not None else None
    if target_prev is None or not (0.0 < target_prev < 1.0):
        return {
            "available": False,
            "assessment_scope": "target_population_probability_assessment",
            "prevalence_basis": "target_population_prevalence",
            "study_prevalence": study_prev,
            "target_prevalence": target_prev,
            "transport_applied": False,
            "warning": "Target prevalence is not available; prevalence-adjusted calibration is skipped.",
        }

    adjusted_prob = apply_prevalence_correction(
        prob_arr,
        study_prevalence=study_prev,
        target_prevalence=target_prev,
        clip_min=clip_min,
    )
    adjusted_summary = build_probability_calibration_summary(
        y_arr,
        adjusted_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    adjusted_summary.update(
        {
            "assessment_scope": "target_population_probability_assessment",
            "prevalence_basis": "target_population_prevalence",
            "study_prevalence": study_prev,
            "target_prevalence": target_prev,
            "transport_applied": True,
            "expected_calibration_error_legacy": adjusted_summary.get("expected_calibration_error"),
            "integrated_calibration_index": compute_integrated_calibration_index(y_arr, adjusted_prob),
            "adjusted_probabilities_available": True,
        }
    )
    return adjusted_summary


__all__ = [
    "apply_platt_recalibrator",
    "apply_prevalence_correction",
    "build_probability_calibration_summary",
    "build_adjusted_probability_calibration_summary",
    "build_probability_recalibration_artifact",
    "build_recalibrated_prediction_payload",
    "compute_integrated_calibration_index",
    "estimate_calibration_intercept_slope",
    "fit_platt_recalibrator",
    "run_cross_fitted_probability_recalibration",
]
