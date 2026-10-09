"""
Incremental value metrics for final panel evaluation.

This module provides core utilities for:
- continuous NRI
- IDI
- bootstrap confidence intervals
- alignment of binary prediction payloads produced by MetaboAgent evaluators
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[float]


@dataclass(frozen=True)
class AlignedBinaryPredictions:
    """Aligned binary prediction arrays for old-vs-new model comparison."""

    y_true: np.ndarray
    p_old: np.ndarray
    p_new: np.ndarray
    sample_ids: List[Any]


def _as_1d_array(values: Sequence[Any], *, name: str, dtype: Any = float) -> np.ndarray:
    arr = np.asarray(list(values), dtype=dtype).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _validate_probability_vector(probs: np.ndarray, *, name: str) -> None:
    if probs.ndim != 1:
        raise ValueError(f"{name} must be a 1D probability vector.")
    if np.any(~np.isfinite(probs)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1].")


def _validate_binary_inputs(
    y_true: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    *,
    positive_label: int = 1,
) -> None:
    if y_true.ndim != 1 or p_old.ndim != 1 or p_new.ndim != 1:
        raise ValueError("y_true, p_old, and p_new must all be 1D arrays.")
    if not (len(y_true) == len(p_old) == len(p_new)):
        raise ValueError("y_true, p_old, and p_new must have identical lengths.")
    if np.any(~np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    _validate_probability_vector(p_old, name="p_old")
    _validate_probability_vector(p_new, name="p_new")

    unique_labels = set(np.unique(y_true).tolist())
    allowed = {0, 1, positive_label}
    if len(unique_labels) > 2 or not unique_labels.issubset(allowed):
        raise ValueError(
            "y_true must represent a binary problem encoded with two labels; "
            f"observed labels={sorted(unique_labels)}"
        )
    if len(unique_labels) < 2:
        raise ValueError("y_true must contain both positive and negative samples.")


def extract_binary_prediction_rows(prediction_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize MetaboAgent prediction payloads into comparable row records.

    Expected input format:
    {
      "predictions": [
        {
          "sample_id": ...,
          "true_label_encoded": 0/1,   # preferred
          "true_label": ...,
          "pred_proba_class1": float,  # preferred
          "fold": ...
        },
        ...
      ]
    }
    """
    predictions = prediction_payload.get("predictions", []) if isinstance(prediction_payload, dict) else []
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("prediction_payload must contain a non-empty 'predictions' list.")

    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(predictions):
        if not isinstance(row, dict):
            raise ValueError(f"prediction row at index {idx} is not a dict.")
        if "pred_proba_class1" not in row:
            raise ValueError(f"prediction row at index {idx} missing 'pred_proba_class1'.")
        if "true_label_encoded" in row:
            y_val = row.get("true_label_encoded")
        elif "true_label" in row:
            y_val = row.get("true_label")
        else:
            raise ValueError(f"prediction row at index {idx} missing true label fields.")

        sample_id = row.get("sample_id", idx)
        fold = row.get("fold", -1)
        rows.append(
            {
                "sample_id": sample_id,
                "fold": fold,
                "y_true": y_val,
                "prob": row.get("pred_proba_class1"),
            }
        )
    return rows


def align_binary_prediction_payloads(
    baseline_payload: Dict[str, Any],
    new_payload: Dict[str, Any],
    *,
    positive_label: int = 1,
) -> AlignedBinaryPredictions:
    """
    Align two prediction payloads by (sample_id, fold) for fair NRI/IDI comparison.
    """
    baseline_rows = extract_binary_prediction_rows(baseline_payload)
    new_rows = extract_binary_prediction_rows(new_payload)

    def build_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
        index: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        for row in rows:
            key = (row["sample_id"], row["fold"])
            if key in index:
                raise ValueError(f"Duplicate prediction row detected for key={key}.")
            index[key] = row
        return index

    baseline_index = build_index(baseline_rows)
    new_index = build_index(new_rows)
    shared_keys = [key for key in baseline_index.keys() if key in new_index]
    if not shared_keys:
        raise ValueError("No overlapping (sample_id, fold) keys found between baseline and new payloads.")
    shared_keys = sorted(shared_keys, key=lambda item: (str(item[0]), int(item[1]) if str(item[1]).lstrip("-").isdigit() else str(item[1])))

    y_true: List[int] = []
    p_old: List[float] = []
    p_new: List[float] = []
    sample_ids: List[Any] = []

    for key in shared_keys:
        baseline_row = baseline_index[key]
        new_row = new_index[key]
        baseline_y = int(baseline_row["y_true"])
        new_y = int(new_row["y_true"])
        if baseline_y != new_y:
            raise ValueError(f"Mismatched true labels for key={key}: baseline={baseline_y}, new={new_y}")
        y_true.append(1 if baseline_y == positive_label else 0)
        p_old.append(float(baseline_row["prob"]))
        p_new.append(float(new_row["prob"]))
        sample_ids.append(key[0])

    y_true_arr = _as_1d_array(y_true, name="y_true", dtype=int)
    p_old_arr = _as_1d_array(p_old, name="p_old", dtype=float)
    p_new_arr = _as_1d_array(p_new, name="p_new", dtype=float)
    _validate_binary_inputs(y_true_arr, p_old_arr, p_new_arr, positive_label=1)
    return AlignedBinaryPredictions(y_true=y_true_arr, p_old=p_old_arr, p_new=p_new_arr, sample_ids=sample_ids)


def compute_continuous_nri(
    y_true: ArrayLike,
    p_old: ArrayLike,
    p_new: ArrayLike,
    *,
    positive_label: int = 1,
    tie_tolerance: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute continuous NRI for binary outcomes.
    """
    y_arr = _as_1d_array(y_true, name="y_true", dtype=int)
    p_old_arr = _as_1d_array(p_old, name="p_old", dtype=float)
    p_new_arr = _as_1d_array(p_new, name="p_new", dtype=float)
    _validate_binary_inputs(y_arr, p_old_arr, p_new_arr, positive_label=positive_label)

    delta = p_new_arr - p_old_arr
    event_mask = y_arr == positive_label
    nonevent_mask = ~event_mask

    n_events = int(np.sum(event_mask))
    n_nonevents = int(np.sum(nonevent_mask))
    if n_events == 0 or n_nonevents == 0:
        raise ValueError("continuous NRI requires both events and nonevents.")

    event_up = float(np.mean(delta[event_mask] > tie_tolerance))
    event_down = float(np.mean(delta[event_mask] < -tie_tolerance))
    nonevent_down = float(np.mean(delta[nonevent_mask] < -tie_tolerance))
    nonevent_up = float(np.mean(delta[nonevent_mask] > tie_tolerance))

    events_component = event_up - event_down
    nonevents_component = nonevent_down - nonevent_up
    nri_value = events_component + nonevents_component

    return {
        "type": "continuous",
        "value": float(nri_value),
        "events_up_rate": float(event_up),
        "events_down_rate": float(event_down),
        "nonevents_down_rate": float(nonevent_down),
        "nonevents_up_rate": float(nonevent_up),
        "events_component": float(events_component),
        "nonevents_component": float(nonevents_component),
        "n_events": float(n_events),
        "n_nonevents": float(n_nonevents),
    }


def compute_idi(
    y_true: ArrayLike,
    p_old: ArrayLike,
    p_new: ArrayLike,
    *,
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute IDI for binary outcomes using discrimination slope difference.
    """
    y_arr = _as_1d_array(y_true, name="y_true", dtype=int)
    p_old_arr = _as_1d_array(p_old, name="p_old", dtype=float)
    p_new_arr = _as_1d_array(p_new, name="p_new", dtype=float)
    _validate_binary_inputs(y_arr, p_old_arr, p_new_arr, positive_label=positive_label)

    event_mask = y_arr == positive_label
    nonevent_mask = ~event_mask
    if not np.any(event_mask) or not np.any(nonevent_mask):
        raise ValueError("IDI requires both events and nonevents.")

    old_event_mean = float(np.mean(p_old_arr[event_mask]))
    old_nonevent_mean = float(np.mean(p_old_arr[nonevent_mask]))
    new_event_mean = float(np.mean(p_new_arr[event_mask]))
    new_nonevent_mean = float(np.mean(p_new_arr[nonevent_mask]))

    old_slope = old_event_mean - old_nonevent_mean
    new_slope = new_event_mean - new_nonevent_mean
    idi_value = new_slope - old_slope

    return {
        "value": float(idi_value),
        "old_discrimination_slope": float(old_slope),
        "new_discrimination_slope": float(new_slope),
        "old_event_mean": float(old_event_mean),
        "old_nonevent_mean": float(old_nonevent_mean),
        "new_event_mean": float(new_event_mean),
        "new_nonevent_mean": float(new_nonevent_mean),
    }


def bootstrap_confidence_interval(
    y_true: ArrayLike,
    p_old: ArrayLike,
    p_new: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]],
    *,
    ci_level: float = 0.95,
    n_bootstrap: int = 200,
    random_state: int = 42,
    value_key: str = "value",
) -> Dict[str, Any]:
    """
    Estimate confidence interval and empirical p-value for one metric via stratified bootstrap resampling.
    """
    if n_bootstrap <= 1:
        return {
            "ci_95": None,
            "p_value_empirical": None,
            "n_bootstrap": int(n_bootstrap),
            "successful_bootstrap": 0,
            "failed_bootstrap": 0,
        }

    y_arr = _as_1d_array(y_true, name="y_true", dtype=int)
    p_old_arr = _as_1d_array(p_old, name="p_old", dtype=float)
    p_new_arr = _as_1d_array(p_new, name="p_new", dtype=float)
    _validate_binary_inputs(y_arr, p_old_arr, p_new_arr, positive_label=1)

    rng = np.random.default_rng(random_state)
    event_idx = np.where(y_arr == 1)[0]
    nonevent_idx = np.where(y_arr == 0)[0]
    if len(event_idx) == 0 or len(nonevent_idx) == 0:
        raise ValueError("Bootstrap CI requires both events and nonevents.")

    estimates: List[float] = []
    for _ in range(int(n_bootstrap)):
        sampled_events = rng.choice(event_idx, size=len(event_idx), replace=True)
        sampled_nonevents = rng.choice(nonevent_idx, size=len(nonevent_idx), replace=True)
        sampled_idx = np.concatenate([sampled_events, sampled_nonevents])
        try:
            metric_result = metric_fn(y_arr[sampled_idx], p_old_arr[sampled_idx], p_new_arr[sampled_idx])
            estimates.append(float(metric_result[value_key]))
        except Exception:
            continue

    alpha = 1.0 - float(ci_level)
    if not estimates:
        return {
            "ci_95": None,
            "p_value_empirical": None,
            "n_bootstrap": int(n_bootstrap),
            "successful_bootstrap": 0,
            "failed_bootstrap": int(n_bootstrap),
        }

    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    estimates_arr = np.asarray(estimates, dtype=float)
    p_left = (float(np.sum(estimates_arr <= 0.0)) + 1.0) / (len(estimates_arr) + 1.0)
    p_right = (float(np.sum(estimates_arr >= 0.0)) + 1.0) / (len(estimates_arr) + 1.0)
    p_empirical = float(min(1.0, 2.0 * min(p_left, p_right)))
    return {
        "ci_95": [lower, upper],
        "p_value_empirical": p_empirical,
        "n_bootstrap": int(n_bootstrap),
        "successful_bootstrap": int(len(estimates)),
        "failed_bootstrap": int(n_bootstrap - len(estimates)),
    }


def build_incremental_value_summary(
    *,
    y_true: ArrayLike,
    p_old: ArrayLike,
    p_new: ArrayLike,
    baseline_name: str,
    new_model_name: str,
    comparison_scope: str = "out_of_fold",
    positive_label: int = 1,
    sample_ids: Optional[Sequence[Any]] = None,
    include_nri: bool = True,
    include_idi: bool = True,
    bootstrap_iterations: int = 200,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Build one canonical summary payload for incremental value reporting.
    """
    y_arr = _as_1d_array(y_true, name="y_true", dtype=int)
    p_old_arr = _as_1d_array(p_old, name="p_old", dtype=float)
    p_new_arr = _as_1d_array(p_new, name="p_new", dtype=float)
    _validate_binary_inputs(y_arr, p_old_arr, p_new_arr, positive_label=positive_label)

    y_bin = np.where(y_arr == positive_label, 1, 0).astype(int)
    result: Dict[str, Any] = {
        "enabled": True,
        "baseline_name": str(baseline_name),
        "new_model_name": str(new_model_name),
        "comparison_scope": str(comparison_scope),
        "positive_label": int(positive_label),
        "n_samples": int(len(y_bin)),
        "n_events": int(np.sum(y_bin == 1)),
        "n_nonevents": int(np.sum(y_bin == 0)),
    }

    if sample_ids is not None:
        sample_id_list = list(sample_ids)
        if len(sample_id_list) != len(y_bin):
            raise ValueError("sample_ids length must match y_true length.")
        result["sample_ids"] = sample_id_list

    if include_nri:
        nri_result = compute_continuous_nri(y_bin, p_old_arr, p_new_arr, positive_label=1)
        nri_ci = bootstrap_confidence_interval(
            y_bin,
            p_old_arr,
            p_new_arr,
            lambda y, old, new: compute_continuous_nri(y, old, new, positive_label=1),
            ci_level=ci_level,
            n_bootstrap=bootstrap_iterations,
            random_state=random_state,
            value_key="value",
        )
        nri_result.update(nri_ci)
        result["nri"] = nri_result

    if include_idi:
        idi_result = compute_idi(y_bin, p_old_arr, p_new_arr, positive_label=1)
        idi_ci = bootstrap_confidence_interval(
            y_bin,
            p_old_arr,
            p_new_arr,
            lambda y, old, new: compute_idi(y, old, new, positive_label=1),
            ci_level=ci_level,
            n_bootstrap=bootstrap_iterations,
            random_state=random_state + 1,
            value_key="value",
        )
        idi_result.update(idi_ci)
        result["idi"] = idi_result

    result["delta_risk_summary"] = {
        "mean_delta_all": float(np.mean(p_new_arr - p_old_arr)),
        "mean_delta_events": float(np.mean((p_new_arr - p_old_arr)[y_bin == 1])),
        "mean_delta_nonevents": float(np.mean((p_new_arr - p_old_arr)[y_bin == 0])),
    }
    return result
