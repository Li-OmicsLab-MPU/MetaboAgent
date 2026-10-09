"""
Deterministic Phase 2 Engine

Phase 2 has been refactored from heuristic LLM-PToT into a deterministic,
math-driven Pareto-SFS engine with TOPSIS beam selection and dual-criteria
early stopping.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.tools.analysis import build_bio_context, build_clinical_utility_payload, calculate_f_bio_v2
from src.tools.analysis.f_bio_v2 import BioContext
from src.tools.analysis.calibration_tools import (
    build_probability_recalibration_artifact,
    run_cross_fitted_probability_recalibration,
)
from src.tools.analysis.nri_idi_tools import (
    align_binary_prediction_payloads,
    build_incremental_value_summary,
)
from src.tools.analysis.pareto_evaluator import (
    calculate_f_bio,
    calculate_f_corr,
    calculate_f_cost,
    calculate_f_perf,
)
from src.utils.config_manager import get_config
from src.utils.evaluation_audit import describe_path, write_audit_json, write_pipeline_lock
from src.utils.evaluation_artifacts import write_evaluation_artifacts


CandidateNode = Dict[str, Any]


def _normalize_features(features: Sequence[str]) -> List[str]:
    """Return a stable, deduplicated feature list."""
    return sorted(dict.fromkeys(features))


def _candidate_key(features: Sequence[str]) -> Tuple[str, ...]:
    return tuple(_normalize_features(features))


def _extract_binary_predictions(cv_predictions: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    predictions = cv_predictions.get("predictions", [])
    if not predictions:
        return np.array([]), np.array([])

    y_true = np.array([row["true_label_encoded"] for row in predictions], dtype=float)
    y_score = np.array([row["pred_proba_class1"] for row in predictions], dtype=float)
    return y_true, y_score


def _build_probability_column_diagnostics(cv_predictions: Dict[str, Any]) -> Dict[str, Any]:
    predictions = cv_predictions.get("predictions", []) if isinstance(cv_predictions, dict) else []
    label_names = cv_predictions.get("label_names", []) if isinstance(cv_predictions, dict) else []
    mapping = {
        "pred_proba_class0": label_names[0] if len(label_names) > 0 else None,
        "pred_proba_class1": label_names[1] if len(label_names) > 1 else None,
    }
    diagnostics: Dict[str, Any] = {
        "class_probability_mapping": mapping,
        "positive_label": 1,
        "positive_probability_column": "pred_proba_class1",
        "auc_using_class0": None,
        "auc_using_class1": None,
        "possible_direction_reversal_warning": False,
    }
    if not predictions:
        diagnostics["direction_reversal_reason"] = "no_predictions"
        return diagnostics

    try:
        y_true = np.array([row["true_label_encoded"] for row in predictions], dtype=float)
        proba0 = np.array([row["pred_proba_class0"] for row in predictions], dtype=float)
        proba1 = np.array([row["pred_proba_class1"] for row in predictions], dtype=float)
        if len(np.unique(y_true)) == 2:
            auc0 = float(roc_auc_score(y_true, proba0))
            auc1 = float(roc_auc_score(y_true, proba1))
            diagnostics["auc_using_class0"] = auc0
            diagnostics["auc_using_class1"] = auc1
            diagnostics["possible_direction_reversal_warning"] = bool(auc1 < 0.5 and auc0 > auc1)
            if diagnostics["possible_direction_reversal_warning"]:
                diagnostics["direction_reversal_reason"] = (
                    "pred_proba_class1 is mapped to label_names[1], but it ranks holdout positives below negatives. "
                    "Check external holdout distribution and label/probability semantics before interpreting performance."
                )
            else:
                diagnostics["direction_reversal_reason"] = "not_detected"
        else:
            diagnostics["direction_reversal_reason"] = "non_binary_labels"
    except Exception as exc:
        diagnostics["direction_reversal_reason"] = f"diagnostics_failed: {exc}"
    return diagnostics


def _infer_phase2_early_stopping_type(stop_reason: Optional[str], max_depth: int, physical_feature_limit: int) -> str:
    reason = str(stop_reason or "").strip().lower()
    if "delong" in reason:
        return "delong_patience"
    if "pool_exhausted" in reason:
        return "pool_exhausted"
    if "hard_prune" in reason:
        return "hard_prune_exhausted"
    if "pareto_empty" in reason:
        return "pareto_empty"
    if "beam_empty" in reason:
        return "beam_empty"
    if "fallback" in reason:
        return "fallback_to_global_best"
    if reason == "max_depth_reached":
        if int(max_depth) == int(physical_feature_limit):
            return "epv_limit_reached"
        return "max_depth_reached"
    return reason or "unknown"


def _build_incremental_value_payload(
    baseline_eval: Optional[CandidateNode],
    new_eval: Optional[CandidateNode],
    *,
    baseline_name: str = "phase1_champion_final_panel",
    comparison_scope: str = "out_of_fold",
    positive_label: int = 1,
    bootstrap_iterations: int = 200,
) -> Dict[str, Any]:
    """
    Build NRI/IDI summary between the final displayed panel and the Phase 1 baseline.
    """
    baseline_cv = baseline_eval.get("cv_predictions", {}) if isinstance(baseline_eval, dict) else {}
    new_cv = new_eval.get("cv_predictions", {}) if isinstance(new_eval, dict) else {}
    if not baseline_cv or not new_cv:
        return {
            "enabled": False,
            "reason": "missing_cv_predictions",
            "baseline_name": baseline_name,
            "comparison_scope": comparison_scope,
        }

    try:
        aligned = align_binary_prediction_payloads(
            baseline_cv,
            new_cv,
            positive_label=positive_label,
        )
        new_model_name = str(new_eval.get("selected_model") or "phase2_final_panel") if isinstance(new_eval, dict) else "phase2_final_panel"
        payload = build_incremental_value_summary(
            y_true=aligned.y_true,
            p_old=aligned.p_old,
            p_new=aligned.p_new,
            baseline_name=baseline_name,
            new_model_name=new_model_name,
            comparison_scope=comparison_scope,
            positive_label=1,
            sample_ids=aligned.sample_ids,
            bootstrap_iterations=bootstrap_iterations,
        )
        payload["enabled"] = True
        payload["baseline_feature_count"] = len(baseline_eval.get("features", []) or []) if isinstance(baseline_eval, dict) else 0
        payload["new_feature_count"] = len(new_eval.get("features", []) or []) if isinstance(new_eval, dict) else 0
        payload["baseline_features"] = copy.deepcopy(baseline_eval.get("features", [])) if isinstance(baseline_eval, dict) else []
        payload["new_features"] = copy.deepcopy(new_eval.get("features", [])) if isinstance(new_eval, dict) else []
        return payload
    except Exception as exc:
        return {
            "enabled": False,
            "reason": f"incremental_value_failed: {exc}",
            "baseline_name": baseline_name,
            "comparison_scope": comparison_scope,
        }


def _calculate_comprehensive_metrics(cv_predictions: Dict[str, Any], metric_name: str = "auprc") -> Dict[str, Any]:
    """Compute comprehensive binary clinical metrics from OOF predictions."""
    try:
        y_true, y_pred_proba = _extract_binary_predictions(cv_predictions)
        if y_true.size == 0:
            return {"roc_auc": 0.0, "auprc": 0.0, "brier_score": 0.0, "error": "empty_predictions"}

        roc_auc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        brier_score = brier_score_loss(y_true, y_pred_proba)

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_j = tpr - fpr
        optimal_idx = int(np.argmax(youden_j))
        optimal_threshold = float(thresholds[optimal_idx])

        y_pred_bin = (y_pred_proba >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

        sensitivity = recall_score(y_true, y_pred_bin)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred_bin)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        return {
            "roc_auc": float(roc_auc),
            "auprc": float(auprc),
            "brier_score": float(brier_score),
            "optimal_threshold": float(optimal_threshold),
            "optimal_youden_j": float(youden_j[optimal_idx]),
            "f1_score": float(f1),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "accuracy": float(accuracy),
            "ppv": float(precision),
            "npv": float(npv),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "primary_metric": metric_name,
            "positive_class": 1,
            "n_samples": int(len(y_true)),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(len(y_true) - np.sum(y_true)),
        }
    except Exception as exc:
        return {"roc_auc": 0.0, "auprc": 0.0, "brier_score": 0.0, "error": str(exc)}


def _auto_select_metric_from_df(
    df: pd.DataFrame,
    target_column: str,
    imbalance_threshold: float = 2.0,
) -> str:
    """
    Auto-select metric from class imbalance.
    NOTE: After 2026-05-07 update, this now always returns 'roc_auc'
    as per user requirement to unify performance evaluation.
    """
    return "roc_auc"


def _auto_select_metric(data_path: str, target_column: str, imbalance_threshold: float = 2.0) -> str:
    return "roc_auc"


def _is_dominated(candidate: CandidateNode, other: CandidateNode) -> bool:
    return (
        other["perf"] >= candidate["perf"]
        and other["bio"] >= candidate["bio"]
        and other["corr"] <= candidate["corr"]
        and other["cost"] <= candidate["cost"]
        and (
            other["perf"] > candidate["perf"]
            or other["bio"] > candidate["bio"]
            or other["corr"] < candidate["corr"]
            or other["cost"] < candidate["cost"]
        )
    )


def get_pareto_front(candidates: List[CandidateNode]) -> List[CandidateNode]:
    """Extract the non-dominated set over perf/bio/corr/cost."""
    pareto_front: List[CandidateNode] = []
    for idx, candidate in enumerate(candidates):
        dominated = False
        for jdx, other in enumerate(candidates):
            if idx == jdx:
                continue
            if _is_dominated(candidate, other):
                dominated = True
                break
        if not dominated:
            pareto_front.append(copy.deepcopy(candidate))
    return pareto_front


def _normalize_topsis_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Normalize TOPSIS weights into canonical perf/bio/corr/cost keys.

    Accepts both canonical keys and config-style keys:
    - perf / bio / corr / cost
    - f_perf / f_bio / f_corr / f_cost
    """
    canonical = {"perf": 1.0, "bio": 1.0, "corr": 1.0, "cost": 1.0}
    if not isinstance(weights, dict):
        return canonical

    key_aliases = {
        "perf": ("perf", "f_perf"),
        "bio": ("bio", "f_bio"),
        "corr": ("corr", "f_corr", "corr_benefit", "f_corr_benefit"),
        "cost": ("cost", "f_cost", "cost_benefit", "f_cost_benefit"),
    }
    for canonical_key, aliases in key_aliases.items():
        for alias in aliases:
            if alias not in weights:
                continue
            try:
                canonical[canonical_key] = float(weights.get(alias, canonical[canonical_key]))
            except (TypeError, ValueError):
                canonical[canonical_key] = canonical[canonical_key]
            break
    return canonical


def _resolve_phase2_scenario_key(
    clinical_scenario: str,
    cfg: Any,
) -> Tuple[str, str]:
    """
    Resolve a phase2 scenario key from explicit key, config metadata, or free text.

    Returns:
        (scenario_key, resolution_source)
    """
    active_scenario = str(cfg.get_active_scenario() or "primary_care").strip()
    scenario_text = str(clinical_scenario or "").strip()
    if not scenario_text:
        return active_scenario, "config_active_scenario"

    scenarios = ((cfg.config or {}).get("evaluation", {}) or {}).get("scenarios", {}) or {}
    scenario_text_lower = scenario_text.lower()

    if scenario_text in scenarios:
        return scenario_text, "explicit_key"
    if scenario_text_lower in scenarios:
        return scenario_text_lower, "explicit_key_lower"

    for key, meta in scenarios.items():
        name = str((meta or {}).get("name", "")).strip().lower()
        description = str((meta or {}).get("description", "")).strip().lower()
        key_lower = str(key).strip().lower()
        if scenario_text_lower == name or scenario_text_lower == description:
            return key_lower, "exact_metadata_match"
        if scenario_text_lower and (
            scenario_text_lower in name
            or name in scenario_text_lower
            or scenario_text_lower in description
            or description in scenario_text_lower
        ):
            return key_lower, "fuzzy_metadata_match"

    heuristic_rules = [
        ("primary_care", ("primary care", "screening", "基层", "早筛")),
        ("er_triage", ("er", "triage", "急诊", "分诊")),
        ("companion_diagnostics", ("companion", "创新药", "伴随诊断", "临床试验")),
    ]
    for key, tokens in heuristic_rules:
        if key not in scenarios:
            continue
        if any(token in scenario_text_lower for token in tokens):
            return key, "heuristic_text_match"

    return active_scenario, "fallback_active_scenario"


def _resolve_scenario_topsis_weights(
    cfg: Any,
    clinical_scenario: str,
    topsis_weights: Optional[Dict[str, float]],
) -> Tuple[Dict[str, float], str, str]:
    """
    Resolve TOPSIS weights with precedence:
    1. explicit `topsis_weights`
    2. scenario weights from config
    3. equal weights fallback
    """
    scenario_key, scenario_source = _resolve_phase2_scenario_key(clinical_scenario, cfg)
    if isinstance(topsis_weights, dict) and topsis_weights:
        return _normalize_topsis_weights(topsis_weights), scenario_key, "explicit_argument"

    try:
        scenario_weights = cfg.get_scenario_weights(scenario_key)
        return _normalize_topsis_weights(scenario_weights), scenario_key, f"config:{scenario_source}"
    except Exception:
        return _normalize_topsis_weights(None), scenario_key, "equal_weight_fallback"


def _resolve_soft_epsilon_perf_margin(
    phase2_config: Optional[Dict[str, Any]],
    scenario_key: str,
) -> Tuple[bool, float, str]:
    """Resolve soft epsilon-feasible preference config for the current scenario."""
    strategy_cfg = ((phase2_config or {}).get("soft_epsilon_feasible", {}) or {})
    enabled = bool(strategy_cfg.get("enabled", True))
    default_margin = float(strategy_cfg.get("default_perf_margin", 0.01) or 0.01)
    scenario_map = strategy_cfg.get("scenario_perf_margin", {}) or {}
    if scenario_key in scenario_map:
        try:
            return enabled, float(scenario_map.get(scenario_key, default_margin)), "scenario_specific"
        except (TypeError, ValueError):
            return enabled, default_margin, "scenario_specific_invalid_fallback"
    return enabled, default_margin, "default"


def calculate_topsis_scores(
    candidates: List[CandidateNode],
    weights: Optional[Dict[str, float]] = None,
) -> List[CandidateNode]:
    """Return candidates ranked by TOPSIS closeness coefficient."""
    if not candidates:
        return []

    weights = _normalize_topsis_weights(weights)
    criteria = ["perf", "bio", "corr_benefit", "cost_benefit"]

    # Align all dimensions into benefit direction:
    # perf/bio: larger is better
    # corr/cost: smaller is better -> convert to benefit scores first
    matrix = np.array(
        [
            [
                float(candidate["perf"]),
                float(candidate["bio"]),
                1.0 - float(candidate["corr"]),
                -float(candidate["cost"]),
            ]
            for candidate in candidates
        ],
        dtype=float,
    )

    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0.0] = 1.0
    normalized = matrix / norms

    weight_vector = np.array(
        [float(weights.get("perf", 1.0)), float(weights.get("bio", 1.0)), float(weights.get("corr", 1.0)), float(weights.get("cost", 1.0))],
        dtype=float,
    )
    weight_sum = float(np.sum(weight_vector))
    if weight_sum <= 0.0:
        weight_vector = np.ones_like(weight_vector) / len(weight_vector)
    else:
        weight_vector = weight_vector / weight_sum

    weighted = normalized * weight_vector
    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    distance_to_best = np.linalg.norm(weighted - ideal_best, axis=1)
    distance_to_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denominator = distance_to_best + distance_to_worst
    denominator[denominator == 0.0] = 1.0
    closeness = distance_to_worst / denominator

    ranked: List[CandidateNode] = []
    for idx, candidate in enumerate(candidates):
        enriched = copy.deepcopy(candidate)
        enriched["topsis_transformed"] = {
            "perf": float(matrix[idx, 0]),
            "bio": float(matrix[idx, 1]),
            "corr_benefit": float(matrix[idx, 2]),
            "cost_benefit": float(matrix[idx, 3]),
        }
        enriched["topsis_score"] = float(closeness[idx])
        enriched["distance_to_ideal_best"] = float(distance_to_best[idx])
        enriched["distance_to_ideal_worst"] = float(distance_to_worst[idx])
        ranked.append(enriched)

    ranked.sort(key=lambda row: row["topsis_score"], reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["topsis_rank"] = rank
    return ranked


def jaccard_similarity(features_a: Sequence[str], features_b: Sequence[str]) -> float:
    set_a = set(features_a)
    set_b = set(features_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def filter_diverse_top_k(
    ranked_candidates: List[CandidateNode],
    k: int,
    diversity_weight: float = 0.05,
    min_jaccard_distance: float = 1e-9,
) -> List[CandidateNode]:
    """Keep TOPSIS Top-1 and diversify the remaining beams by Jaccard distance."""
    if not ranked_candidates:
        return []

    top1 = copy.deepcopy(ranked_candidates[0])
    top1["jaccard_distance_to_top1"] = 1.0
    selected = [top1]

    if k <= 1:
        return selected

    remaining: List[CandidateNode] = []
    for candidate in ranked_candidates[1:]:
        enriched = copy.deepcopy(candidate)
        similarity = jaccard_similarity(top1["features"], enriched["features"])
        distance = 1.0 - similarity
        enriched["jaccard_similarity_to_top1"] = float(similarity)
        enriched["jaccard_distance_to_top1"] = float(distance)
        enriched["diversity_adjusted_score"] = float(enriched["topsis_score"] + diversity_weight * distance)
        remaining.append(enriched)

    diverse = [candidate for candidate in remaining if candidate["jaccard_distance_to_top1"] > min_jaccard_distance]
    diverse.sort(
        key=lambda row: (row["diversity_adjusted_score"], row["topsis_score"], row["jaccard_distance_to_top1"]),
        reverse=True,
    )

    for candidate in diverse:
        if len(selected) >= k:
            break
        selected.append(candidate)

    if len(selected) < k:
        fallback = [candidate for candidate in remaining if _candidate_key(candidate["features"]) not in {_candidate_key(row["features"]) for row in selected}]
        fallback.sort(key=lambda row: row["topsis_score"], reverse=True)
        for candidate in fallback:
            if len(selected) >= k:
                break
            selected.append(candidate)

    return selected[:k]


def _apply_memory_candidate_rerank(
    candidate_pool: List[str],
    sorted_base_pool: List[str],
    memory_search_prior: Optional[Dict[str, Any]],
    candidate_boost_cap: float = 0.15,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Reorder candidate exploration priority using advisory memory priors.

    The memory layer only changes exploration order. It never removes
    candidates and never changes the underlying evaluation functions.
    """
    normalized_candidate_pool = _normalize_features(candidate_pool)
    normalized_sorted_pool = _normalize_features(sorted_base_pool or candidate_pool)
    prior = dict(memory_search_prior or {})
    priority_scores = prior.get("candidate_priority_scores", {}) or {}
    anchor_hints = [str(feature) for feature in prior.get("anchor_feature_hints", []) or [] if feature]

    if not priority_scores and not anchor_hints:
        return normalized_candidate_pool, normalized_sorted_pool, {
            "memory_prior_enabled": False,
            "memory_reranked_candidate_count": 0,
            "memory_anchor_hints": [],
            "memory_applied_actions": [],
        }

    available_set = set(normalized_candidate_pool)
    filtered_anchor_hints = [feature for feature in anchor_hints if feature in available_set]

    bounded_scores: Dict[str, float] = {}
    for feature, value in priority_scores.items():
        feature_name = str(feature)
        if feature_name not in available_set:
            continue
        try:
            bounded_scores[feature_name] = max(0.0, min(float(value), float(candidate_boost_cap)))
        except (TypeError, ValueError):
            continue

    def _score(feature_name: str) -> Tuple[float, int, str]:
        return (
            bounded_scores.get(feature_name, 0.0),
            1 if feature_name in filtered_anchor_hints else 0,
            feature_name,
        )

    reranked_sorted_pool = sorted(normalized_sorted_pool, key=lambda feature_name: _score(feature_name), reverse=True)
    reranked_candidate_pool = sorted(normalized_candidate_pool, key=lambda feature_name: _score(feature_name), reverse=True)

    trace = {
        "memory_prior_enabled": True,
        "memory_reranked_candidate_count": len([feature for feature in reranked_sorted_pool if _score(feature)[0] > 0.0]),
        "memory_anchor_hints": filtered_anchor_hints,
        "memory_applied_actions": ["candidate_rerank", "anchor_priority_boost"],
    }
    return reranked_candidate_pool, reranked_sorted_pool, trace


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=float)

    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end

    midranks = np.empty(len(values), dtype=float)
    midranks[order] = ranks
    return midranks


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    if m <= 0 or n <= 0:
        raise ValueError("DeLong test requires both positive and negative samples")

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for idx in range(k):
        tx[idx, :] = _compute_midrank(positive_examples[idx, :])
        ty[idx, :] = _compute_midrank(negative_examples[idx, :])
        tz[idx, :] = _compute_midrank(predictions_sorted_transposed[idx, :])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true: Sequence[int], pred_one: Sequence[float], pred_two: Sequence[float]) -> float:
    """Return the two-sided p-value of correlated ROC AUCs via DeLong's test."""
    y_true_array = np.asarray(y_true, dtype=int)
    pred_one_array = np.asarray(pred_one, dtype=float)
    pred_two_array = np.asarray(pred_two, dtype=float)

    if y_true_array.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array")
    if len(np.unique(y_true_array)) != 2:
        return 1.0
    if pred_one_array.shape != pred_two_array.shape or pred_one_array.shape[0] != y_true_array.shape[0]:
        raise ValueError("Prediction arrays must align with y_true")

    order = np.argsort(-y_true_array)
    label_1_count = int(np.sum(y_true_array))
    predictions_sorted_transposed = np.vstack((pred_one_array, pred_two_array))[:, order]
    aucs, covariance = _fast_delong(predictions_sorted_transposed, label_1_count)

    variance = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    variance = max(float(variance), 1e-12)
    z_score = abs(float(aucs[0] - aucs[1])) / math.sqrt(variance)
    return float(math.erfc(z_score / math.sqrt(2.0)))


def calculate_global_icer(node: CandidateNode, reference_node: Optional[CandidateNode]) -> Optional[float]:
    """Compute the incremental complexity-effectiveness ratio."""
    if reference_node is None:
        perf = float(node.get("perf", 0.0))
        if abs(perf) < 1e-12:
            return None
        return float(node.get("cost", 0.0) / perf)

    delta_perf = float(node.get("perf", 0.0)) - float(reference_node.get("perf", 0.0))
    delta_cost = float(node.get("cost", 0.0)) - float(reference_node.get("cost", 0.0))
    if abs(delta_perf) < 1e-12:
        return None
    return float(delta_cost / delta_perf)


def _convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    if isinstance(obj, tuple):
        return [_convert_numpy_types(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json_payload(payload: Dict[str, Any], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(_convert_numpy_types(payload), handle, indent=2, ensure_ascii=False)


def _write_prediction_payload_csv(prediction_payload: Dict[str, Any], file_path: str) -> None:
    predictions = prediction_payload.get("predictions", []) if isinstance(prediction_payload, dict) else []
    if not predictions:
        return
    pd.DataFrame(predictions).to_csv(file_path, index=False)


def _summarize_numeric(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
        }
    arr = np.array([float(value) for value in values], dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def _safe_pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    series_a = pd.Series([float(value) for value in values_a], dtype=float)
    series_b = pd.Series([float(value) for value in values_b], dtype=float)
    corr = series_a.corr(series_b, method="pearson")
    return float(corr) if pd.notna(corr) else 0.0


def _safe_spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    series_a = pd.Series([float(value) for value in values_a], dtype=float)
    series_b = pd.Series([float(value) for value in values_b], dtype=float)
    corr = series_a.corr(series_b, method="spearman")
    return float(corr) if pd.notna(corr) else 0.0


def _topk_panel_overlap(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    topk_values: Sequence[int],
) -> Dict[str, Any]:
    overlaps: Dict[str, Any] = {}
    for topk in topk_values:
        k = max(1, int(topk))
        top_a = {_candidate_key(row.get("features", [])) for row in list(rows_a)[:k]}
        top_b = {_candidate_key(row.get("features", [])) for row in list(rows_b)[:k]}
        union = top_a | top_b
        overlaps[str(k)] = {
            "overlap_count": int(len(top_a & top_b)),
            "jaccard": float(len(top_a & top_b) / len(union)) if union else 1.0,
        }
    return overlaps


def _summarize_bio_component_means(bio_debug: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(bio_debug, dict):
        return {
            "direct_prior": 0.0,
            "disease_pathway_align": 0.0,
            "anchor_link": 0.0,
            "coverage_gain": 0.0,
        }

    p0_items = bio_debug.get("p0_items", []) or []
    component_rows: List[Dict[str, float]] = []
    for item in p0_items:
        member_supports = item.get("member_supports", {}) if isinstance(item, dict) else {}
        if not isinstance(member_supports, dict) or not member_supports:
            continue
        supports = list(member_supports.values())
        component_rows.append(
            {
                "direct_prior": float(np.mean([float(s.get("direct_prior", 0.0) or 0.0) for s in supports])),
                "disease_pathway_align": float(np.mean([float(s.get("disease_pathway_align", 0.0) or 0.0) for s in supports])),
                "anchor_link": float(np.mean([float(s.get("anchor_link", 0.0) or 0.0) for s in supports])),
                "coverage_gain": float(np.mean([float(s.get("coverage_gain", 0.0) or 0.0) for s in supports])),
            }
        )

    if not component_rows:
        return {
            "direct_prior": 0.0,
            "disease_pathway_align": 0.0,
            "anchor_link": 0.0,
            "coverage_gain": 0.0,
        }

    return {
        "direct_prior": float(np.mean([row["direct_prior"] for row in component_rows])),
        "disease_pathway_align": float(np.mean([row["disease_pathway_align"] for row in component_rows])),
        "anchor_link": float(np.mean([row["anchor_link"] for row in component_rows])),
        "coverage_gain": float(np.mean([row["coverage_gain"] for row in component_rows])),
    }


def generate_heuristic_candidates(*args: Any, **kwargs: Any) -> List[List[str]]:
    """Deprecated after deterministic Phase 2 refactor."""
    raise NotImplementedError(
        "generate_heuristic_candidates() has been removed from the execution path. "
        "Phase 2 now uses exhaustive deterministic expansion in ParetoSFS_Engine."
    )


def calculate_dynamic_max_depth(total_sample_count: int, absolute_max: int = 20, min_required_features: int = 5) -> int:
    """
    Calculate max search depth based on Dynamic EPV theory (Vittinghoff 2007).

    Strict 10-EPV is now derived from the total sample count rather than the
    minority-class count. The relaxed 5-EPV fallback and minimum threshold are
    preserved.
    """
    # 1. Calculate strict 10 EPV
    strict_max_features = total_sample_count // 10

    if strict_max_features < min_required_features:
        # Trigger Vittinghoff (2007) elastic 5-9 EPV rule
        relaxed_max_features = total_sample_count // 5
        print(f"[INFO] 触发动态 EPV 放宽: 严格 10 EPV 上限为 {strict_max_features}，存在欠拟合风险。")
        print(f"[INFO] 根据 Vittinghoff (2007) 弹性法则，已采用 5 EPV 标准，上限放宽至 {relaxed_max_features}。")
        max_depth = relaxed_max_features
    else:
        # Samples are sufficient, stick to strict 10 EPV
        print(f"[INFO] 样本量充足，基于总样本数严格遵守 10 EPV 标准，特征上限为 {strict_max_features}。")
        max_depth = strict_max_features

    # 2. Apply absolute physical upper limit
    final_max_depth = min(absolute_max, max_depth)
    return int(final_max_depth)


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class ParetoSFS_Engine:
    """Deterministic exhaustive Pareto-SFS engine with TOPSIS beam selection."""

    def __init__(
        self,
        data_path: str,
        target_column: str,
        candidate_pool: List[str],
        sorted_base_pool: Optional[List[str]] = None,
        taxonomy_map: Optional[Dict[str, Any]] = None,
        pathway_map: Optional[Dict[str, Any]] = None,
        priors_dict: Optional[Dict[str, float]] = None,
        ag_results_path: str = "data/autogluon_training_results.json",
        champion_model_family: Optional[str] = None,
        beam_width: int = 1,
        max_depth: Optional[int] = None,
        k_folds: int = 5,
        metric: Optional[str] = None,
        imbalance_threshold: float = 2.0,
        topsis_weights: Optional[Dict[str, float]] = None,
        improvement_delta: float = 0.005,
        patience: int = 3,
        use_phase1_config_in_perf: bool = False,
        verbose_search_trace: bool = False,
        trace_head_limit: int = 30,
        protected_features: Optional[List[str]] = None,
        protected_anchor_features: Optional[List[str]] = None,
        available_prior_anchor_features: Optional[List[str]] = None,
        disease_name: Optional[str] = None,
        clinical_scenario: str = "",
        memory_search_prior: Optional[Dict[str, Any]] = None,
        dataset_fingerprint: Optional[Dict[str, Any]] = None,
        phase0_output: Optional[Dict[str, Any]] = None,
        objective_mode: str = "four_objective",
        positive_class: Optional[str] = None,
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.taxonomy_map = taxonomy_map or {}
        self.pathway_map = pathway_map or {}
        self.priors_dict = priors_dict or {}
        self.ag_results_path = ag_results_path
        self.beam_width = max(1, int(beam_width))
        self.k_folds = k_folds
        self.imbalance_threshold = imbalance_threshold
        self.improvement_delta = improvement_delta
        self.patience = patience
        self.use_phase1_config_in_perf = use_phase1_config_in_perf
        self.verbose_search_trace = verbose_search_trace
        self.trace_head_limit = max(1, int(trace_head_limit))
        self.requested_clinical_scenario = str(clinical_scenario or "").strip()
        self.memory_search_prior = dict(memory_search_prior or {})
        self.dataset_fingerprint = dict(dataset_fingerprint or {})
        self.phase0_output = dict(phase0_output or {})
        self.positive_class = str(positive_class or "").strip() or None
        self.objective_mode = str(objective_mode or "four_objective").strip().lower()
        if self.objective_mode not in {"four_objective", "performance_only"}:
            raise ValueError(
                "objective_mode must be one of {'four_objective', 'performance_only'}"
            )
        self.cfg = get_config()
        self.resolved_clinical_scenario, self.clinical_scenario_resolution = _resolve_phase2_scenario_key(
            self.requested_clinical_scenario,
            self.cfg,
        )
        self.topsis_weights, _, self.topsis_weight_source = _resolve_scenario_topsis_weights(
            self.cfg,
            self.requested_clinical_scenario,
            topsis_weights,
        )
        self.phase2_config = self.cfg.get_phase2_config()
        (
            self.soft_epsilon_feasible_enabled,
            self.soft_epsilon_perf_margin,
            self.soft_epsilon_perf_margin_source,
        ) = _resolve_soft_epsilon_perf_margin(
            self.phase2_config,
            self.resolved_clinical_scenario,
        )
        delong_cfg = self.phase2_config.get("delong_early_stopping", {}) if isinstance(self.phase2_config, dict) else {}
        self.delong_p_value_threshold = float(delong_cfg.get("p_value_threshold", 0.05) or 0.05)
        self.practical_auc_override_enabled = bool(delong_cfg.get("practical_auc_override_enabled", True))
        self.practical_auc_override = float(delong_cfg.get("practical_auc_override", 0.02) or 0.02)
        self.disease_name = str(disease_name or get_config().get_test_disease_name() or "").strip()
        self.external_validation_data_path = self._resolve_external_validation_data_path()

        self.df = pd.read_csv(data_path)
        available_features = [col for col in self.df.columns if col != target_column]
        provided_protected = list(protected_features or []) + list(protected_anchor_features or [])
        normalized_protected = _normalize_features(provided_protected)
        available_feature_set = set(available_features)
        self.protected_features = [feature for feature in normalized_protected if feature in available_feature_set]
        self.protected_feature_set = set(self.protected_features)
        self.missing_protected_features = [
            feature for feature in normalized_protected if feature not in self.protected_feature_set
        ]
        normalized_available_prior = _normalize_features(
            list(available_prior_anchor_features or []) + list(self.protected_features)
        )
        self.available_prior_anchor_features = [
            feature for feature in normalized_available_prior if feature in available_feature_set
        ]
        self.available_prior_anchor_feature_set = set(self.available_prior_anchor_features)
        self.missing_available_prior_anchor_features = [
            feature for feature in normalized_available_prior if feature not in self.available_prior_anchor_feature_set
        ]

        requested_candidate_pool = candidate_pool or available_features
        self.phase1_panel_baseline_features = [
            feature for feature in _normalize_features(requested_candidate_pool) if feature in available_feature_set
        ]
        merged_candidate_pool = list(requested_candidate_pool) + self.protected_features
        initial_candidate_pool = _normalize_features(merged_candidate_pool)
        initial_candidate_pool_set = set(initial_candidate_pool)

        ranked_pool = sorted_base_pool or initial_candidate_pool
        initial_sorted_base_pool = [
            feature
            for feature in ranked_pool
            if feature in initial_candidate_pool_set and feature not in self.protected_feature_set
        ]
        if not initial_sorted_base_pool:
            initial_sorted_base_pool = [
                feature for feature in initial_candidate_pool if feature not in self.protected_feature_set
            ]

        phase2_memory_config = self.cfg.get_phase2_memory_config() if hasattr(self.cfg, "get_phase2_memory_config") else {}
        candidate_boost_cap = float(phase2_memory_config.get("candidate_boost_cap", 0.15))
        reranked_candidate_pool, reranked_sorted_pool, memory_rerank_trace = _apply_memory_candidate_rerank(
            candidate_pool=initial_candidate_pool,
            sorted_base_pool=initial_sorted_base_pool,
            memory_search_prior=self.memory_search_prior,
            candidate_boost_cap=candidate_boost_cap,
        )
        self.candidate_pool = reranked_candidate_pool
        self.candidate_pool_set = set(self.candidate_pool)
        self.sorted_base_pool = [
            feature for feature in reranked_sorted_pool
            if feature in self.candidate_pool_set and feature not in self.protected_feature_set
        ]
        if not self.sorted_base_pool:
            self.sorted_base_pool = [
                feature for feature in self.candidate_pool if feature not in self.protected_feature_set
            ]
        self.memory_rerank_trace = memory_rerank_trace

        if metric is None:
            # After 2026-05-07, always default to roc_auc as per user request
            self.metric = "roc_auc"
        else:
            self.metric = metric

        self.total_sample_count = int(len(self.df))
        (
            self.epv_reference_sample_count,
            self.epv_reference_source,
        ) = self._resolve_epv_reference_sample_count()
        self.minority_class_count = self._infer_minority_class_count()
        # Apply Vittinghoff (2007) elastic EPV rule
        self.physical_feature_limit = calculate_dynamic_max_depth(self.epv_reference_sample_count)
        self.expansion_feature_limit = max(0, self.physical_feature_limit - len(self.protected_features))

        if max_depth is None:
            self.max_depth = self.expansion_feature_limit
        else:
            self.max_depth = min(int(max_depth), self.expansion_feature_limit)

        self.f_bio_v2_config = self.cfg.get_phase2_f_bio_v2_config()
        self.use_f_bio_v2 = bool(self.f_bio_v2_config.get("enabled", False))
        self.f_bio_calibration_config = self.f_bio_v2_config.get("calibration", {}) if isinstance(self.f_bio_v2_config, dict) else {}
        probability_recalibration_cfg = self.phase2_config.get("probability_recalibration", {}) if isinstance(self.phase2_config, dict) else {}
        self.probability_recalibration_config = {
            "enabled": bool(probability_recalibration_cfg.get("enabled", True)),
            "method": str(probability_recalibration_cfg.get("method", "platt") or "platt").strip().lower(),
            "strategy": "cross_fitted",
            "clip_min": float(probability_recalibration_cfg.get("clip_min", 1e-6) or 1e-6),
            "selection_policy": str(probability_recalibration_cfg.get("selection_policy", "fixed_default_platt") or "fixed_default_platt"),
        }
        self.bio_context: Optional[BioContext] = None
        if self.use_f_bio_v2:
            try:
                self.bio_context = build_bio_context(
                    disease_name=self.disease_name,
                    feature_pool=self.candidate_pool,
                    priors_dict=self.priors_dict,
                    taxonomy_map=self.taxonomy_map,
                    pathway_map=self.pathway_map,
                    phase0_output=self.phase0_output,
                    config=self.f_bio_v2_config,
                )
            except Exception:
                self.bio_context = None

        self.champion_model_family = champion_model_family or self._load_champion_model_family()
        self.node_cache: Dict[Tuple[str, ...], CandidateNode] = {}
        self.search_details: Dict[str, Any] = {
            "engine": "ParetoSFS_Engine",
            "objective_mode": self.objective_mode,
            "objective_selection_contract": {
                "f_perf_used_for_selection": True,
                "f_bio_used_for_selection": self.objective_mode == "four_objective",
                "f_corr_used_for_selection": self.objective_mode == "four_objective",
                "f_cost_used_for_selection": self.objective_mode == "four_objective",
                "pareto_filter_used": self.objective_mode == "four_objective",
                "topsis_used": self.objective_mode == "four_objective",
                "diversity_beam_filter_used": self.objective_mode == "four_objective",
                "post_hoc_objective_scores_computed": True,
            },
            "metric": self.metric,
            "beam_width": self.beam_width,
            "clinical_scenario_input": self.requested_clinical_scenario,
            "clinical_scenario": self.resolved_clinical_scenario,
            "clinical_scenario_resolution": self.clinical_scenario_resolution,
            "topsis_weights": copy.deepcopy(self.topsis_weights),
            "topsis_weight_source": self.topsis_weight_source,
            "soft_epsilon_feasible_enabled": self.soft_epsilon_feasible_enabled,
            "soft_epsilon_perf_margin": self.soft_epsilon_perf_margin,
            "soft_epsilon_perf_margin_source": self.soft_epsilon_perf_margin_source,
            "delong_p_value_threshold": self.delong_p_value_threshold,
            "practical_auc_override_enabled": self.practical_auc_override_enabled,
            "practical_auc_override": self.practical_auc_override,
            "max_depth": self.max_depth,
            "physical_feature_limit": self.physical_feature_limit,
            "expansion_feature_limit": self.expansion_feature_limit,
            "total_sample_count": self.total_sample_count,
            "epv_reference_sample_count": self.epv_reference_sample_count,
            "epv_reference_source": self.epv_reference_source,
            "minority_class_count": self.minority_class_count,
            "disease_name": self.disease_name,
            "bio_score_version": "f_bio_v2" if self.use_f_bio_v2 else "f_bio_v1",
            "f_bio_v2_enabled": self.use_f_bio_v2,
            "protected_features": copy.deepcopy(self.protected_features),
            "n_protected_features": len(self.protected_features),
            "missing_protected_features": copy.deepcopy(self.missing_protected_features),
            "available_prior_anchor_features": copy.deepcopy(self.available_prior_anchor_features),
            "n_available_prior_anchor_features": len(self.available_prior_anchor_features),
            "missing_available_prior_anchor_features": copy.deepcopy(self.missing_available_prior_anchor_features),
            "memory_prior_enabled": bool(self.memory_rerank_trace.get("memory_prior_enabled", False)),
            "memory_prior_confidence": float(self.memory_search_prior.get("prior_confidence", 0.0) or 0.0),
            "memory_matched_cases": copy.deepcopy(self.memory_search_prior.get("provenance_cases", []) or []),
            "memory_reranked_candidate_count": int(self.memory_rerank_trace.get("memory_reranked_candidate_count", 0) or 0),
            "memory_anchor_hints": copy.deepcopy(self.memory_rerank_trace.get("memory_anchor_hints", []) or []),
            "memory_applied_actions": copy.deepcopy(self.memory_rerank_trace.get("memory_applied_actions", []) or []),
            "search_start_mode": "protected_anchor_seed" if self.protected_features else "empty_seed",
            "bio_context_summary": {
                "anchor_mode": self.bio_context.anchor_mode if self.bio_context else "disease_only",
                "protected_anchor_id_count": len(self.bio_context.protected_anchor_ids) if self.bio_context else 0,
                "external_prior_seed_count": len(self.bio_context.external_prior_seed_ids) if self.bio_context else 0,
                "disease_core_pathway_count": len(self.bio_context.disease_core_pathways) if self.bio_context else 0,
            },
            "bio_calibration_enabled": bool(self.f_bio_calibration_config.get("enabled", True)),
            "hard_prune_rescue_event_count": 0,
            "hard_prune_rescue_depths": [],
            "layers": [],
        }

        self.global_best_node: Optional[CandidateNode] = None
        self.global_best_depth: Optional[int] = None
        self.global_best_beams: List[CandidateNode] = []
        self.min_final_feature_count = 5
        self.best_min_feature_node: Optional[CandidateNode] = None
        self.best_min_feature_depth: Optional[int] = None
        self.best_min_feature_beams: List[CandidateNode] = []
        self.phase1_panel_baseline_eval_cache: Optional[CandidateNode] = None
        self.patience_counter = 0

    def _resolve_epv_reference_sample_count(self) -> Tuple[int, str]:
        """
        Prefer the original input dataset sample count for the EPV guard.

        Phase 2 often receives the Phase 1 train-only matrix after the leakage
        guard split. If upstream explicitly provides the original input sample
        size, use it as the EPV denominator; otherwise fall back to the current
        Phase 2 input matrix size.
        """
        fallback_count = max(1, int(self.total_sample_count))
        fingerprint = self.dataset_fingerprint if isinstance(self.dataset_fingerprint, dict) else {}
        explicit_original_n_samples = _coerce_positive_int(fingerprint.get("original_input_n_samples"))
        if explicit_original_n_samples is not None:
            return explicit_original_n_samples, "dataset_fingerprint.original_input_n_samples"

        explicit_epv_reference = _coerce_positive_int(fingerprint.get("epv_reference_sample_count"))
        if explicit_epv_reference is not None:
            return explicit_epv_reference, "dataset_fingerprint.epv_reference_sample_count"

        fingerprint_n_samples = _coerce_positive_int(fingerprint.get("n_samples"))
        if fingerprint_n_samples is not None:
            return fingerprint_n_samples, "dataset_fingerprint.n_samples"
        return fallback_count, "phase2_input_data"

    def _classify_auc_breakthrough(self, p_value: Optional[float], delta_auc: Optional[float]) -> Tuple[bool, str]:
        """Allow large AUC gains to update the global best even when DeLong is underpowered."""
        auc_gain = float(delta_auc or 0.0)
        p_val = float(p_value) if p_value is not None else math.inf

        statistically_significant = (
            p_val <= self.delong_p_value_threshold and auc_gain >= self.improvement_delta
        )
        if statistically_significant:
            return True, "significant_breakthrough"

        practical_override = self.practical_auc_override_enabled and auc_gain >= self.practical_auc_override
        if practical_override:
            return True, "practical_breakthrough_override"

        return False, "no_significant_breakthrough"

    def _calculate_legacy_bio_score(self, normalized_features: Sequence[str]) -> float:
        return float(
            calculate_f_bio(
                features_list=list(normalized_features),
                priors_dict=self.priors_dict,
                taxonomy_map=self.taxonomy_map,
                pathway_map=self.pathway_map,
                w_lit=0.2,
                w_mech=0.8,
                gamma=0.1,
                sigmoid_k=1.0,
            )
            or 0.0
        )

    def _calculate_bio_score(self, normalized_features: Sequence[str], return_debug: bool = False) -> Any:
        if self.use_f_bio_v2:
            try:
                return calculate_f_bio_v2(
                    features_list=list(normalized_features),
                    bio_context=self.bio_context,
                    disease_name=self.disease_name,
                    feature_pool=self.candidate_pool,
                    priors_dict=self.priors_dict,
                    taxonomy_map=self.taxonomy_map,
                    pathway_map=self.pathway_map,
                    return_debug=return_debug,
                )
            except Exception:
                legacy_score = self._calculate_legacy_bio_score(normalized_features)
                if return_debug:
                    return {
                        "f_bio_v2": legacy_score,
                        "legacy_f_bio": legacy_score,
                        "fallback_to_v1": True,
                    }
                return legacy_score

        legacy_score = self._calculate_legacy_bio_score(normalized_features)
        if return_debug:
            return {"f_bio_v1": float(legacy_score), "fallback_to_v1": False}
        return legacy_score

    def _calculate_v2_bio_debug(self, normalized_features: Sequence[str]) -> Dict[str, Any]:
        return calculate_f_bio_v2(
            features_list=list(normalized_features),
            bio_context=self.bio_context,
            disease_name=self.disease_name,
            feature_pool=self.candidate_pool,
            priors_dict=self.priors_dict,
            taxonomy_map=self.taxonomy_map,
            pathway_map=self.pathway_map,
            return_debug=True,
        )

    def _collect_calibration_candidates(self, final_node: CandidateNode) -> List[CandidateNode]:
        unique: Dict[Tuple[str, ...], CandidateNode] = {}

        def add_candidate(node: Optional[CandidateNode], priority: int = 3) -> None:
            if not isinstance(node, dict):
                return
            features = node.get("features", [])
            key = _candidate_key(features)
            if not key:
                return
            existing = unique.get(key)
            payload = copy.deepcopy(node)
            payload["_calibration_priority"] = min(priority, int(existing.get("_calibration_priority", priority))) if existing else int(priority)
            if existing is None:
                unique[key] = payload
                return
            for field in ["perf", "bio", "corr", "cost", "cv_predictions", "roc_auc", "topsis_score", "bio_debug", "bio_score_version"]:
                if existing.get(field) is None and payload.get(field) is not None:
                    existing[field] = payload.get(field)
            existing["_calibration_priority"] = min(int(existing.get("_calibration_priority", priority)), int(priority))

        add_candidate(final_node.get("search_model_evaluation"), priority=0)
        add_candidate(self.global_best_node, priority=1)
        for layer in self.search_details.get("layers", []):
            for node in layer.get("selected_beams", []) or []:
                add_candidate(node, priority=1)
        for node in self.node_cache.values():
            add_candidate(node, priority=3)

        ordered = sorted(
            unique.values(),
            key=lambda row: (
                int(row.get("_calibration_priority", 9)),
                -float(row.get("perf", 0.0) or 0.0),
                len(row.get("features", []) or []),
                list(row.get("features", []) or []),
            ),
        )
        max_candidates = max(1, int(self.f_bio_calibration_config.get("max_candidates", 200) or 200))
        return [copy.deepcopy(row) for row in ordered[:max_candidates]]

    def _build_phase2_bio_calibration_payload(self, final_node: CandidateNode) -> Dict[str, Any]:
        candidate_nodes = self._collect_calibration_candidates(final_node)
        topk_values = self.f_bio_calibration_config.get("topk_values", [5, 10, 20])
        topk_values = [max(1, int(value)) for value in topk_values if int(value) > 0]
        if not topk_values:
            topk_values = [5, 10, 20]

        candidate_rows: List[Dict[str, Any]] = []
        v1_candidates: List[CandidateNode] = []
        v2_candidates: List[CandidateNode] = []

        for node in candidate_nodes:
            features = _normalize_features(node.get("features", []))
            base_row = {
                "features": features,
                "n_features": len(features),
                "perf": float(node.get("perf", 0.0) or 0.0),
                "corr": float(node.get("corr", 0.0) or 0.0),
                "cost": float(node.get("cost", 0.0) or 0.0),
            }
            bio_v1 = self._calculate_legacy_bio_score(features)
            bio_v2_debug = self._calculate_v2_bio_debug(features)
            bio_v2 = float(bio_v2_debug.get("f_bio_v2", 0.0) or 0.0)

            row = {
                **base_row,
                "bio_v1": bio_v1,
                "bio_v2": bio_v2,
                "anchor_mode": bio_v2_debug.get("mode", self.bio_context.anchor_mode if self.bio_context else "disease_only"),
            }
            candidate_rows.append(row)
            v1_candidates.append({**base_row, "bio": bio_v1})
            v2_candidates.append({**base_row, "bio": bio_v2})

        ranked_v1 = calculate_topsis_scores(v1_candidates, weights=self.topsis_weights)
        ranked_v2 = calculate_topsis_scores(v2_candidates, weights=self.topsis_weights)
        rank_v1_map = {_candidate_key(row.get("features", [])): row for row in ranked_v1}
        rank_v2_map = {_candidate_key(row.get("features", [])): row for row in ranked_v2}

        for row in candidate_rows:
            key = _candidate_key(row.get("features", []))
            v1_ranked = rank_v1_map.get(key, {})
            v2_ranked = rank_v2_map.get(key, {})
            row["topsis_v1"] = float(v1_ranked.get("topsis_score", 0.0) or 0.0)
            row["topsis_v2"] = float(v2_ranked.get("topsis_score", 0.0) or 0.0)
            row["rank_v1"] = int(v1_ranked.get("topsis_rank", 0) or 0)
            row["rank_v2"] = int(v2_ranked.get("topsis_rank", 0) or 0)

        search_winner_features = _normalize_features(final_node.get("search_model_evaluation", {}).get("features", final_node.get("features", [])))
        search_winner_key = _candidate_key(search_winner_features)
        winner_v1 = ranked_v1[0] if ranked_v1 else {}
        winner_v2 = ranked_v2[0] if ranked_v2 else {}
        winner_v1_key = _candidate_key(winner_v1.get("features", []))
        winner_v2_key = _candidate_key(winner_v2.get("features", []))

        winner_v1_debug = self._calculate_v2_bio_debug(winner_v1.get("features", [])) if winner_v1 else {}
        winner_v2_debug = self._calculate_v2_bio_debug(winner_v2.get("features", [])) if winner_v2 else {}
        actual_winner_debug = self._calculate_v2_bio_debug(search_winner_features) if search_winner_features else {}

        calibration_payload = {
            "schema_version": "phase2.bio_calibration.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disease_name": self.disease_name,
            "f_bio_v2_enabled": self.use_f_bio_v2,
            "calibration_config": copy.deepcopy(self.f_bio_calibration_config),
            "sample_summary": {
                "explored_candidate_count": len(self.node_cache),
                "sampled_candidate_count": len(candidate_rows),
                "sampled_ratio": float(len(candidate_rows) / max(1, len(self.node_cache))),
                "topk_values": topk_values,
            },
            "score_distribution": {
                "bio_v1": _summarize_numeric([row["bio_v1"] for row in candidate_rows]),
                "bio_v2": _summarize_numeric([row["bio_v2"] for row in candidate_rows]),
                "topsis_v1": _summarize_numeric([row["topsis_v1"] for row in candidate_rows]),
                "topsis_v2": _summarize_numeric([row["topsis_v2"] for row in candidate_rows]),
            },
            "rank_consistency": {
                "bio_score_pearson": _safe_pearson(
                    [row["bio_v1"] for row in candidate_rows],
                    [row["bio_v2"] for row in candidate_rows],
                ),
                "bio_score_spearman": _safe_spearman(
                    [row["bio_v1"] for row in candidate_rows],
                    [row["bio_v2"] for row in candidate_rows],
                ),
                "topsis_pearson": _safe_pearson(
                    [row["topsis_v1"] for row in candidate_rows],
                    [row["topsis_v2"] for row in candidate_rows],
                ),
                "topsis_spearman": _safe_spearman(
                    [row["topsis_v1"] for row in candidate_rows],
                    [row["topsis_v2"] for row in candidate_rows],
                ),
                "topk_panel_overlap": _topk_panel_overlap(ranked_v1, ranked_v2, topk_values),
            },
            "winner_comparison": {
                "actual_search_winner_features": search_winner_features,
                "actual_search_winner_rank_under_v1": int(rank_v1_map.get(search_winner_key, {}).get("topsis_rank", 0) or 0),
                "actual_search_winner_rank_under_v2": int(rank_v2_map.get(search_winner_key, {}).get("topsis_rank", 0) or 0),
                "actual_display_winner_features": _normalize_features(final_node.get("features", [])),
                "winner_under_v1": copy.deepcopy(winner_v1),
                "winner_under_v2": copy.deepcopy(winner_v2),
                "winner_changed": bool(winner_v1_key != winner_v2_key),
                "winner_feature_jaccard": float(jaccard_similarity(winner_v1.get("features", []), winner_v2.get("features", []))),
            },
            "winner_debug": {
                "actual_search_winner_v2_debug": actual_winner_debug,
                "winner_under_v1_v2_debug": winner_v1_debug,
                "winner_under_v2_v2_debug": winner_v2_debug,
            },
            "candidate_rows": candidate_rows,
        }
        return _convert_numpy_types(calibration_payload)

    def _build_candidate_trace(
        self,
        expanded: List[CandidateNode],
        threshold: float,
        pareto_front: Optional[List[CandidateNode]] = None,
        epsilon_feasible_front: Optional[List[CandidateNode]] = None,
        selected_beams: Optional[List[CandidateNode]] = None,
    ) -> List[Dict[str, Any]]:
        pareto_keys = {_candidate_key(node["features"]) for node in (pareto_front or [])}
        epsilon_keys = {_candidate_key(node["features"]) for node in (epsilon_feasible_front or [])}
        selected_keys = {_candidate_key(node["features"]) for node in (selected_beams or [])}

        trace_rows: List[Dict[str, Any]] = []
        for node in expanded:
            key = _candidate_key(node["features"])
            passed_perf = float(node.get("perf", 0.0)) >= threshold
            status = (
                "selected_beam"
                if key in selected_keys
                else "epsilon_feasible"
                if key in epsilon_keys
                else "pareto_front"
                if key in pareto_keys
                else "pruned_by_perf"
                if not passed_perf
                else "survived_perf"
            )
            trace_row = self._node_trace_row(node, status=status)
            trace_row.update(
                {
                    "passed_perf_threshold": passed_perf,
                    "pruned_by_perf": not passed_perf,
                    "in_pareto_front": key in pareto_keys,
                    "in_epsilon_feasible_front": key in epsilon_keys,
                    "selected_beam": key in selected_keys,
                }
            )
            trace_rows.append(trace_row)
        return trace_rows

    def _node_trace_row(self, node: CandidateNode, status: str = "") -> Dict[str, Any]:
        bio_components = _summarize_bio_component_means(node.get("bio_debug", {}))
        return {
            "status": status,
            "features": copy.deepcopy(node.get("features", [])),
            "n_features": len(node.get("features", [])),
            "perf": float(node.get("perf", 0.0) or 0.0),
            "bio": float(node.get("bio", 0.0) or 0.0),
            "corr": float(node.get("corr", 0.0) or 0.0),
            "cost": float(node.get("cost", 0.0) or 0.0),
            "topsis": float(node.get("topsis_score", 0.0) or 0.0),
            "direct_prior": bio_components["direct_prior"],
            "disease_pathway_align": bio_components["disease_pathway_align"],
            "anchor_link": bio_components["anchor_link"],
            "coverage_gain": bio_components["coverage_gain"],
        }

    def _print_trace_rows(self, title: str, rows: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
        if not rows:
            print(f"{title}: []")
            return

        print(title)
        print(
            f"{'Status':<16} {'Perf':>7} {'Bio':>7} {'Corr':>7} {'Cost':>7} {'TOPSIS':>8} "
            f"{'Prior':>7} {'Path':>7} {'Anchor':>7} {'Cover':>7}  Features"
        )
        rows_to_print = rows[: limit] if limit else rows
        for row in rows_to_print:
            print(
                f"{str(row.get('status', '')):<16} "
                f"{float(row.get('perf', 0.0)):>7.4f} "
                f"{float(row.get('bio', 0.0)):>7.4f} "
                f"{float(row.get('corr', 0.0)):>7.4f} "
                f"{float(row.get('cost', 0.0)):>7.4f} "
                f"{float(row.get('topsis', 0.0)):>8.4f} "
                f"{float(row.get('direct_prior', 0.0)):>7.4f} "
                f"{float(row.get('disease_pathway_align', 0.0)):>7.4f} "
                f"{float(row.get('anchor_link', 0.0)):>7.4f} "
                f"{float(row.get('coverage_gain', 0.0)):>7.4f}  "
                f"{row.get('features', [])}"
            )
        if limit and len(rows) > len(rows_to_print):
            print(f"... truncated {len(rows) - len(rows_to_print)} more rows")

    def _print_panel_evaluation(self, title: str, node: Optional[CandidateNode]) -> None:
        if not isinstance(node, dict) or not node.get("features"):
            print(f"{title}: unavailable")
            return
        bio_components = _summarize_bio_component_means(node.get("bio_debug", {}))
        roc_auc = node.get("roc_auc")
        comprehensive_metrics = node.get("comprehensive_metrics") if isinstance(node.get("comprehensive_metrics"), dict) else {}
        if roc_auc is None:
            roc_auc = comprehensive_metrics.get("roc_auc")
        print(title)
        print(f"  features: {node.get('features', [])}")
        print(
            "  scores: "
            f"perf={float(node.get('perf', 0.0) or 0.0):.4f}, "
            f"bio={float(node.get('bio', 0.0) or 0.0):.4f}, "
            f"corr={float(node.get('corr', 0.0) or 0.0):.4f}, "
            f"cost={float(node.get('cost', 0.0) or 0.0):.4f}, "
            f"roc_auc={float(roc_auc or 0.0):.4f}"
        )
        print(
            "  bio components: "
            f"prior={bio_components['direct_prior']:.4f}, "
            f"path={bio_components['disease_pathway_align']:.4f}, "
            f"anchor={bio_components['anchor_link']:.4f}, "
            f"cover={bio_components['coverage_gain']:.4f}"
        )

    def _summarize_panel_evaluation(self, node: Optional[CandidateNode]) -> Dict[str, Any]:
        if not isinstance(node, dict) or not node.get("features"):
            return {}
        comprehensive_metrics = node.get("comprehensive_metrics") if isinstance(node.get("comprehensive_metrics"), dict) else {}
        roc_auc = node.get("roc_auc")
        if roc_auc is None:
            roc_auc = comprehensive_metrics.get("roc_auc")
        return {
            "features": copy.deepcopy(node.get("features", [])),
            "feature_count": len(node.get("features", []) or []),
            "selected_model": node.get("selected_model"),
            "perf": float(node.get("perf", 0.0) or 0.0),
            "bio": float(node.get("bio", 0.0) or 0.0),
            "corr": float(node.get("corr", 0.0) or 0.0),
            "cost": float(node.get("cost", 0.0) or 0.0),
            "roc_auc": float(roc_auc or 0.0),
            "auprc": float(comprehensive_metrics.get("auprc", 0.0) or 0.0),
            "brier_score": float(comprehensive_metrics.get("brier_score", 0.0) or 0.0),
            "f1_score": float(comprehensive_metrics.get("f1_score", 0.0) or 0.0),
            "sensitivity": float(comprehensive_metrics.get("sensitivity", 0.0) or 0.0),
            "specificity": float(comprehensive_metrics.get("specificity", 0.0) or 0.0),
            "topsis_score": float(node.get("topsis_score", 0.0) or 0.0),
            "bio_debug": copy.deepcopy(node.get("bio_debug", {})),
        }

    def _build_top_candidate_beams_summary(
        self,
        final_beams: List[CandidateNode],
        final_display_node: Optional[CandidateNode] = None,
        search_final_node: Optional[CandidateNode] = None,
        phase1_eval_node: Optional[CandidateNode] = None,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        top_k = max(1, int(self.beam_width))
        for idx, beam in enumerate((final_beams or [])[:top_k], start=1):
            search_eval = self._ensure_predictions(copy.deepcopy(beam))
            search_eval["selected_model"] = "RandomForest (deterministic search mode)"
            if search_eval.get("comprehensive_metrics") is None:
                search_eval["comprehensive_metrics"] = _calculate_comprehensive_metrics(search_eval["cv_predictions"], self.metric)

            if idx == 1 and isinstance(final_display_node, dict) and final_display_node.get("features"):
                display_eval = copy.deepcopy(final_display_node)
                phase1_eval = copy.deepcopy(phase1_eval_node) if phase1_eval_node else {}
                search_snapshot = copy.deepcopy(search_final_node) if search_final_node else copy.deepcopy(search_eval)
            else:
                display_eval, phase1_eval = self._resolve_final_display_node(search_eval)
                display_eval = copy.deepcopy(display_eval)
                search_snapshot = copy.deepcopy(search_eval)

            final_display_source = (
                "phase1_champion"
                if "Phase 1 champion config" in str(display_eval.get("selected_model", ""))
                else "randomforest_search"
            )
            summaries.append(
                {
                    "rank": idx,
                    "features": copy.deepcopy(search_eval.get("features", [])),
                    "feature_count": len(search_eval.get("features", []) or []),
                    "topsis_score": float(search_eval.get("topsis_score", 0.0) or 0.0),
                    "search_model_evaluation": self._summarize_panel_evaluation(search_snapshot),
                    "final_display_evaluation": self._summarize_panel_evaluation(display_eval),
                    "phase1_model_evaluation": self._summarize_panel_evaluation(phase1_eval),
                    "final_display_source": final_display_source,
                }
            )
        return summaries

    def _print_top_candidate_beams_summary(self, beam_summaries: List[Dict[str, Any]]) -> None:
        if not beam_summaries:
            return
        print("Top Candidate Beams:")
        for beam in beam_summaries:
            rank = beam.get("rank")
            feature_count = beam.get("feature_count", 0)
            features = beam.get("features", [])
            search_eval = beam.get("search_model_evaluation", {}) or {}
            final_eval = beam.get("final_display_evaluation", {}) or {}
            print(f"  Beam #{rank} ({feature_count} features): {features}")
            print(
                "    Search eval: "
                f"perf={float(search_eval.get('perf', 0.0)):.4f}, "
                f"bio={float(search_eval.get('bio', 0.0)):.4f}, "
                f"corr={float(search_eval.get('corr', 0.0)):.4f}, "
                f"cost={float(search_eval.get('cost', 0.0)):.4f}, "
                f"ROC-AUC={float(search_eval.get('roc_auc', 0.0)):.4f}, "
                f"TOPSIS={float(beam.get('topsis_score', 0.0)):.4f}"
            )
            print(
                "    Final display eval: "
                f"source={beam.get('final_display_source')}, "
                f"perf={float(final_eval.get('perf', 0.0)):.4f}, "
                f"bio={float(final_eval.get('bio', 0.0)):.4f}, "
                f"corr={float(final_eval.get('corr', 0.0)):.4f}, "
                f"cost={float(final_eval.get('cost', 0.0)):.4f}, "
                f"ROC-AUC={float(final_eval.get('roc_auc', 0.0)):.4f}"
            )

    def _print_layer_trace(
        self,
        depth: int,
        threshold: float,
        trace_rows: List[Dict[str, Any]],
        pareto_front: Optional[List[CandidateNode]] = None,
        epsilon_feasible_front: Optional[List[CandidateNode]] = None,
        selected_beams: Optional[List[CandidateNode]] = None,
        top1_features: Optional[List[str]] = None,
        p_value: Optional[float] = None,
        delta_auc: Optional[float] = None,
        improvement: Optional[str] = None,
        epsilon_best_perf: Optional[float] = None,
        epsilon_margin: Optional[float] = None,
        epsilon_fallback_to_pareto: Optional[bool] = None,
    ) -> None:
        if not self.verbose_search_trace:
            return

        print("\n" + "=" * 100)
        print(f"[Phase2 Trace] Depth {depth}")
        print(f"  Perf threshold: {threshold:.4f}")
        print(f"  Candidates expanded: {len(trace_rows)}")
        if epsilon_margin is not None:
            print(
                "  Soft epsilon-feasible: "
                f"best_perf={float(epsilon_best_perf or 0.0):.4f}, "
                f"margin={float(epsilon_margin or 0.0):.4f}, "
                f"fallback_to_pareto={bool(epsilon_fallback_to_pareto)}"
            )
        if top1_features:
            print(f"  Current Top-1: {top1_features}")
        if p_value is not None or delta_auc is not None or improvement:
            print(
                "  Decision: "
                f"improvement={improvement or 'n/a'}, "
                f"delta_auc={float(delta_auc or 0.0):.4f}, "
                f"delong_p={float(p_value or 0.0):.4f}"
            )
        print("-" * 100)
        self._print_trace_rows("Expanded Candidates", trace_rows, limit=self.trace_head_limit)
        print("-" * 100)
        pareto_rows = [self._node_trace_row(node, status="pareto_front") for node in (pareto_front or [])]
        self._print_trace_rows("Pareto Front", pareto_rows, limit=min(self.trace_head_limit, 10))
        print("-" * 100)
        epsilon_rows = [self._node_trace_row(node, status="epsilon_feasible") for node in (epsilon_feasible_front or [])]
        self._print_trace_rows("Epsilon-Feasible Front", epsilon_rows, limit=min(self.trace_head_limit, 10))
        print("-" * 100)
        beam_rows = [self._node_trace_row(node, status="selected_beam") for node in (selected_beams or [])]
        self._print_trace_rows("Selected Beams", beam_rows, limit=min(self.trace_head_limit, 10))
        print("=" * 100)

    def _infer_minority_class_count(self) -> int:
        if self.target_column not in self.df.columns:
            return max(1, len(self.df) // 10)
        counts = Counter(self.df[self.target_column].values)
        if not counts:
            return max(1, len(self.df) // 10)
        return int(min(counts.values()))

    def _load_champion_model_family(self) -> str:
        if not self.ag_results_path or not os.path.exists(self.ag_results_path):
            return "RandomForest"
        try:
            with open(self.ag_results_path, "r", encoding="utf-8") as handle:
                ag_results = json.load(handle)
            return ag_results.get("best_model", "RandomForest")
        except Exception:
            return "RandomForest"

    def _performance_threshold(self, depth: int) -> float:
        if depth == 0:
            return 0.50
        elif depth <= 2:
            return 0.55
        else:
            return 0.65

    def _maybe_disable_hard_prune_below_min_features(
        self,
        expanded: List[CandidateNode],
        survivors: List[CandidateNode],
        depth: int,
        threshold: float,
    ) -> List[CandidateNode]:
        """
        Disable hard performance pruning while the search has not yet reached the
        minimum final feature count. This keeps the growth phase feature-count
        driven instead of allowing an early cutoff to kill the search.
        """
        if not expanded:
            return survivors

        max_feature_count = max(len(node.get("features", []) or []) for node in expanded)
        if max_feature_count > self.min_final_feature_count:
            return survivors

        if len(survivors) >= len(expanded):
            return survivors

        self.search_details["hard_prune_rescue_event_count"] = int(
            self.search_details.get("hard_prune_rescue_event_count", 0)
        ) + 1
        rescue_depths = list(self.search_details.get("hard_prune_rescue_depths", []) or [])
        rescue_depths.append(
            {
                "depth": int(depth),
                "perf_threshold": float(threshold),
                "rescued_feature_count": int(max_feature_count),
                "original_survivor_count": int(len(survivors)),
                "expanded_candidate_count": int(len(expanded)),
                "best_rescued_perf": float(
                    max(float(node.get("perf", 0.0) or 0.0) for node in expanded)
                ),
            }
        )
        self.search_details["hard_prune_rescue_depths"] = rescue_depths
        return [copy.deepcopy(node) for node in expanded]

    def _build_soft_epsilon_feasible_front(
        self,
        pareto_front: List[CandidateNode],
    ) -> Tuple[List[CandidateNode], Dict[str, Any]]:
        """
        Build a performance-near-optimal subset before TOPSIS ranking.

        This is a soft preference, not a hard stop:
        - prefer candidates within `best_perf - margin`
        - if the subset is empty, fall back to the full pareto front
        """
        metadata = {
            "enabled": self.soft_epsilon_feasible_enabled,
            "perf_margin": self.soft_epsilon_perf_margin,
            "best_perf_in_front": 0.0,
            "epsilon_feasible_count": 0,
            "pareto_count": len(pareto_front),
            "fallback_to_pareto": False,
            "source": "disabled",
        }
        if not pareto_front:
            metadata["source"] = "empty_pareto_front"
            return [], metadata
        if not self.soft_epsilon_feasible_enabled:
            metadata["source"] = "disabled"
            metadata["best_perf_in_front"] = float(max(float(node.get("perf", 0.0) or 0.0) for node in pareto_front))
            metadata["epsilon_feasible_count"] = len(pareto_front)
            return [copy.deepcopy(node) for node in pareto_front], metadata

        best_perf = float(max(float(node.get("perf", 0.0) or 0.0) for node in pareto_front))
        margin = max(0.0, float(self.soft_epsilon_perf_margin))
        epsilon_subset = [
            copy.deepcopy(node)
            for node in pareto_front
            if float(node.get("perf", 0.0) or 0.0) >= (best_perf - margin)
        ]
        metadata.update(
            {
                "best_perf_in_front": best_perf,
                "perf_margin": margin,
                "epsilon_feasible_count": len(epsilon_subset),
                "source": "soft_epsilon_subset",
            }
        )
        if epsilon_subset:
            return epsilon_subset, metadata

        metadata["fallback_to_pareto"] = True
        metadata["source"] = "fallback_to_pareto"
        metadata["epsilon_feasible_count"] = len(pareto_front)
        return [copy.deepcopy(node) for node in pareto_front], metadata

    def _select_performance_only_beams(
        self,
        survivors: List[CandidateNode],
    ) -> Tuple[List[CandidateNode], List[CandidateNode], Dict[str, Any]]:
        """Select beams using f_perf only, with deterministic tie-breaking.

        The four objective scores are still computed by ``_build_node`` and
        retained for post-hoc evaluation, but none of f_bio, f_corr, f_cost,
        Pareto dominance, TOPSIS, or Jaccard diversity is used to choose the
        next search beams in this mode.
        """
        ranked = sorted(
            (copy.deepcopy(node) for node in survivors),
            key=lambda node: (
                -float(node.get("perf", 0.0) or 0.0),
                tuple(_normalize_features(node.get("features", []) or [])),
            ),
        )
        if not ranked:
            return [], [], {
                "enabled": True,
                "selection_rule": "f_perf_descending_then_feature_tuple",
                "best_perf": None,
                "performance_front_count": 0,
            }

        best_perf = float(ranked[0].get("perf", 0.0) or 0.0)
        performance_front = [
            copy.deepcopy(node)
            for node in ranked
            if math.isclose(float(node.get("perf", 0.0) or 0.0), best_perf, rel_tol=0.0, abs_tol=1e-12)
        ]
        selected_beams = [copy.deepcopy(node) for node in ranked[: self.beam_width]]
        for rank, node in enumerate(selected_beams, start=1):
            node["performance_only_rank"] = rank
            node["topsis_score"] = None
            node["jaccard_distance_to_top1"] = None

        metadata = {
            "enabled": True,
            "selection_rule": "f_perf_descending_then_feature_tuple",
            "best_perf": best_perf,
            "performance_front_count": len(performance_front),
            "survivor_count": len(ranked),
            "selected_beam_count": len(selected_beams),
            "topsis_used": False,
            "pareto_filter_used": False,
            "diversity_beam_filter_used": False,
        }
        return performance_front, selected_beams, metadata

    def _build_node(self, features: Sequence[str], include_predictions: bool = False) -> CandidateNode:
        normalized_features = _normalize_features(features)
        key = tuple(normalized_features)
        cached = self.node_cache.get(key)
        if cached is not None and (not include_predictions or cached.get("cv_predictions") is not None):
            return copy.deepcopy(cached)

        node: CandidateNode = copy.deepcopy(cached) if cached is not None else {"features": normalized_features}
        node["features"] = normalized_features

        if "perf" not in node:
            node["perf"] = calculate_f_perf(
                data_path=self.data_path,
                target_column=self.target_column,
                features_list=normalized_features,
                champion_model_family=self.champion_model_family,
                k_folds=self.k_folds,
                ag_results_path=self.ag_results_path,
                metric=self.metric,
                use_phase1_config=self.use_phase1_config_in_perf,
                positive_class=self.positive_class,
            )
            bio_score = self._calculate_bio_score(normalized_features, return_debug=self.use_f_bio_v2)
            if self.use_f_bio_v2 and isinstance(bio_score, dict):
                node["bio"] = float(bio_score.get("f_bio_v2", 0.0) or 0.0)
                node["bio_debug"] = copy.deepcopy(bio_score)
                node["bio_score_version"] = "f_bio_v1_fallback" if bio_score.get("fallback_to_v1") else "f_bio_v2"
            else:
                node["bio"] = float(bio_score or 0.0)
                node["bio_score_version"] = "f_bio_v1"
            node["corr"] = calculate_f_corr(
                data_path=self.data_path,
                features_list=normalized_features,
            )
            node["cost"] = calculate_f_cost(
                features_list=normalized_features,
                taxonomy_map=self.taxonomy_map,
                pathway_map=self.pathway_map,
                n_max=20,
                k=2.0,
                data_path=self.data_path,
                aggregation_discount=True,
            )

        if include_predictions and node.get("cv_predictions") is None:
            perf_score, predictions = calculate_f_perf(
                data_path=self.data_path,
                target_column=self.target_column,
                features_list=normalized_features,
                champion_model_family=self.champion_model_family,
                k_folds=self.k_folds,
                ag_results_path=self.ag_results_path,
                metric=self.metric,
                use_phase1_config=self.use_phase1_config_in_perf,
                return_predictions=True,
                positive_class=self.positive_class,
            )
            node["perf"] = perf_score
            node["cv_predictions"] = predictions
            y_true, y_score = _extract_binary_predictions(predictions)
            node["roc_auc"] = float(roc_auc_score(y_true, y_score)) if y_true.size and len(np.unique(y_true)) == 2 else 0.0

        self.node_cache[key] = copy.deepcopy(node)
        return copy.deepcopy(node)

    def _expand_current_beams(self, current_beams: List[Any]) -> List[CandidateNode]:
        expanded: List[CandidateNode] = []
        seen: set[Tuple[str, ...]] = set()

        for beam in current_beams:
            beam_features = beam.get("features", []) if isinstance(beam, dict) else list(beam)
            beam_feature_set = set(beam_features)
            for feature in self.sorted_base_pool:
                if feature in beam_feature_set:
                    continue
                candidate_features = _normalize_features(list(beam_features) + [feature])
                key = tuple(candidate_features)
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(self._build_node(candidate_features, include_predictions=False))

        return expanded

    def _ensure_predictions(self, node: CandidateNode) -> CandidateNode:
        return self._build_node(node["features"], include_predictions=True)

    def _record_layer(
        self,
        depth: int,
        expanded_count: int,
        threshold: float,
        survivors: List[CandidateNode],
        pareto_front: List[CandidateNode],
        epsilon_feasible_front: List[CandidateNode],
        selected_beams: List[CandidateNode],
        p_value: Optional[float],
        delta_auc: Optional[float],
        improvement: str,
        candidate_trace: Optional[List[Dict[str, Any]]] = None,
        epsilon_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        epsilon_metadata = epsilon_metadata or {}
        self.search_details["layers"].append(
            {
                "depth": depth,
                "expanded_count": expanded_count,
                "perf_threshold": threshold,
                "survivor_count": len(survivors),
                "pareto_count": len(pareto_front),
                "epsilon_feasible_count": len(epsilon_feasible_front),
                "soft_epsilon_enabled": bool(epsilon_metadata.get("enabled", False)),
                "soft_epsilon_perf_margin": epsilon_metadata.get("perf_margin"),
                "best_perf_in_front": epsilon_metadata.get("best_perf_in_front"),
                "epsilon_fallback_to_pareto": bool(epsilon_metadata.get("fallback_to_pareto", False)),
                "beam_count": len(selected_beams),
                "top1_features": selected_beams[0]["features"] if selected_beams else [],
                "top1_topsis_score": selected_beams[0].get("topsis_score", 0.0) if selected_beams else 0.0,
                "top1_perf": selected_beams[0].get("perf", 0.0) if selected_beams else 0.0,
                "top1_roc_auc": selected_beams[0].get("roc_auc") if selected_beams else None,
                "delong_p_value": p_value,
                "delta_auc_vs_global_best": delta_auc,
                "improvement_status": improvement,
                "pareto_front": [copy.deepcopy(node) for node in pareto_front],
                "epsilon_feasible_front": [copy.deepcopy(node) for node in epsilon_feasible_front],
                "selected_beams": [copy.deepcopy(node) for node in selected_beams],
                "candidate_trace": candidate_trace or [],
            }
        )

    def _update_global_best(self, top1: CandidateNode, selected_beams: List[CandidateNode], depth: int) -> None:
        previous_best = copy.deepcopy(self.global_best_node) if self.global_best_node is not None else None
        updated = copy.deepcopy(top1)
        updated["global_icer"] = calculate_global_icer(updated, previous_best)
        self.global_best_node = updated
        self.global_best_beams = [copy.deepcopy(node) for node in selected_beams]
        self.global_best_depth = depth
        self.patience_counter = 0

    def _update_best_min_feature_candidate(
        self,
        top1: CandidateNode,
        selected_beams: List[CandidateNode],
        depth: int,
    ) -> None:
        """Track the best candidate that satisfies the final minimum feature count."""
        if len(top1.get("features", [])) < self.min_final_feature_count:
            return

        should_update = self.best_min_feature_node is None or float(top1.get("perf", 0.0)) > float(
            self.best_min_feature_node.get("perf", 0.0)
        )
        if not should_update:
            return

        self.best_min_feature_node = copy.deepcopy(top1)
        self.best_min_feature_beams = [copy.deepcopy(node) for node in selected_beams]
        self.best_min_feature_depth = depth

    def _apply_min_feature_guard(
        self,
        final_beams: List[CandidateNode],
    ) -> Tuple[List[CandidateNode], bool]:
        """
        Ensure the final returned panel respects the minimum feature-count policy.

        Search is allowed to explore below the threshold early on, but the final
        returned winner should not roll back to a panel smaller than the minimum
        once an eligible >=5-feature solution has already been discovered.
        """
        if final_beams and len(final_beams[0].get("features", [])) >= self.min_final_feature_count:
            return final_beams, False

        if self.best_min_feature_beams:
            return [copy.deepcopy(node) for node in self.best_min_feature_beams], True

        return final_beams, False

    def _resolve_external_validation_data_path(self) -> str:
        """Best-effort resolve the aligned external holdout prepared by Phase 1."""
        candidate_paths: List[str] = []

        env_path = str(os.environ.get("METABOAGENT_EXTERNAL_EVAL_PATH", "") or "").strip()
        if env_path:
            candidate_paths.append(env_path)

        for payload in (self.dataset_fingerprint, self.phase0_output):
            if not isinstance(payload, dict):
                continue
            for key in (
                "external_validation_data_path",
                "external_eval_data_path",
                "selected_holdout_data_path",
                "holdout_data_path",
            ):
                value = str(payload.get(key, "") or "").strip()
                if value:
                    candidate_paths.append(value)

        # Phase 1 writes the aligned holdout next to the selected-features training matrix.
        data_dir = os.path.dirname(os.path.abspath(self.data_path))
        candidate_paths.append(os.path.join(data_dir, "selected_features_holdout.csv"))

        phase1_paths = self.cfg.get_phase1_paths() if hasattr(self.cfg, "get_phase1_paths") else {}
        for path_key in ("selected_features_csv", "legacy_selected_features"):
            base_path = str(phase1_paths.get(path_key, "") or "").strip()
            if base_path:
                candidate_paths.append(
                    os.path.join(os.path.dirname(os.path.abspath(base_path)), "selected_features_holdout.csv")
                )

        seen: set[str] = set()
        current_data_abs = os.path.abspath(self.data_path)
        for path in candidate_paths:
            resolved = os.path.abspath(os.path.expanduser(path))
            if not resolved or resolved in seen:
                continue
            seen.add(resolved)
            if resolved == current_data_abs:
                continue
            if os.path.exists(resolved):
                return resolved
        return ""

    def _evaluate_with_phase1_champion(self, features: Sequence[str]) -> CandidateNode:
        """Evaluate the final panel using Phase 1 champion model + resampling config."""
        normalized_features = _normalize_features(features)
        # The locked holdout belongs to the final Phase 2 panel.  Do not let
        # the Phase 1 AutoGluon winner determine this prediction path: its
        # model name and feature schema can change across AutoGluon releases,
        # while the final panel is a strict subset of the Phase 1 matrix.
        # The historical pipeline consequently fell back to this deterministic
        # RandomForest path; make that behavior explicit for reproducibility.
        perf_score, predictions = calculate_f_perf(
            data_path=self.data_path,
            target_column=self.target_column,
            features_list=normalized_features,
            champion_model_family="RandomForest",
            k_folds=self.k_folds,
            ag_results_path=self.ag_results_path,
            metric=self.metric,
            use_phase1_config=True,
            return_predictions=True,
            positive_class=self.positive_class,
        )
        y_true, y_score = _extract_binary_predictions(predictions)
        roc_auc = float(roc_auc_score(y_true, y_score)) if y_true.size and len(np.unique(y_true)) == 2 else 0.0
        bio_score = self._calculate_bio_score(normalized_features, return_debug=self.use_f_bio_v2)
        bio_value = float(
            bio_score.get("f_bio_v2", 0.0) if self.use_f_bio_v2 and isinstance(bio_score, dict) else bio_score or 0.0
        )
        bio_score_version = (
            "f_bio_v1_fallback"
            if self.use_f_bio_v2 and isinstance(bio_score, dict) and bio_score.get("fallback_to_v1")
            else "f_bio_v2"
            if self.use_f_bio_v2
            else "f_bio_v1"
        )
        return {
            "features": normalized_features,
            "perf": float(perf_score),
            "bio": bio_value,
            "corr": calculate_f_corr(
                data_path=self.data_path,
                features_list=normalized_features,
            ),
            "cost": calculate_f_cost(
                features_list=normalized_features,
                taxonomy_map=self.taxonomy_map,
                pathway_map=self.pathway_map,
                n_max=20,
                k=2.0,
                data_path=self.data_path,
                aggregation_discount=True,
            ),
            "cv_predictions": predictions,
            "roc_auc": roc_auc,
            "evaluation_scope": "development_cv",
            "evaluation_mode": "cv",
            "n_evaluated": int(predictions.get("n_samples", 0)) if isinstance(predictions, dict) else 0,
            "k_folds": int(predictions.get("k_folds", self.k_folds)) if isinstance(predictions, dict) else int(self.k_folds),
            "selected_model": f"{self.champion_model_family} (Phase 1 champion config)",
            "comprehensive_metrics": _calculate_comprehensive_metrics(predictions, self.metric),
            "bio_score_version": bio_score_version,
            "bio_debug": copy.deepcopy(bio_score) if isinstance(bio_score, dict) else {},
        }

    def _evaluate_on_external_holdout(
        self,
        features: Sequence[str],
        *,
        panel_name: str,
        selected_model: str,
    ) -> CandidateNode:
        """Evaluate a panel on the Phase 1-aligned external holdout cohort."""
        if not self.external_validation_data_path or not os.path.exists(self.external_validation_data_path):
            return {}

        normalized_features = _normalize_features(features)
        perf_score, predictions = calculate_f_perf(
            data_path=self.data_path,
            target_column=self.target_column,
            features_list=normalized_features,
            champion_model_family=self.champion_model_family,
            k_folds=self.k_folds,
            ag_results_path=self.ag_results_path,
            metric=self.metric,
            use_phase1_config=True,
            return_predictions=True,
            holdout_data_path=self.external_validation_data_path,
            evaluation_mode="holdout_only",
            positive_class=self.positive_class,
            use_phase1_autogluon_holdout=False,
        )
        y_true, y_score = _extract_binary_predictions(predictions)
        roc_auc = float(roc_auc_score(y_true, y_score)) if y_true.size and len(np.unique(y_true)) == 2 else 0.0
        bio_score = self._calculate_bio_score(normalized_features, return_debug=self.use_f_bio_v2)
        bio_value = float(
            bio_score.get("f_bio_v2", 0.0) if self.use_f_bio_v2 and isinstance(bio_score, dict) else bio_score or 0.0
        )
        bio_score_version = (
            "f_bio_v1_fallback"
            if self.use_f_bio_v2 and isinstance(bio_score, dict) and bio_score.get("fallback_to_v1")
            else "f_bio_v2"
            if self.use_f_bio_v2
            else "f_bio_v1"
        )
        return {
            "panel_name": panel_name,
            "features": normalized_features,
            "perf": float(perf_score),
            "bio": bio_value,
            "corr": calculate_f_corr(
                data_path=self.external_validation_data_path,
                features_list=normalized_features,
            ),
            "cost": calculate_f_cost(
                features_list=normalized_features,
                taxonomy_map=self.taxonomy_map,
                pathway_map=self.pathway_map,
                n_max=20,
                k=2.0,
                data_path=self.data_path,
                aggregation_discount=True,
            ),
            "cv_predictions": predictions,
            "roc_auc": roc_auc,
            "evaluation_scope": "locked_holdout",
            "evaluation_mode": "holdout_only",
            "n_evaluated": int(predictions.get("n_samples", 0)) if isinstance(predictions, dict) else 0,
            "k_folds": 1,
            "selected_model": selected_model,
            "comprehensive_metrics": _calculate_comprehensive_metrics(predictions, self.metric),
            "bio_score_version": bio_score_version,
            "bio_debug": copy.deepcopy(bio_score) if isinstance(bio_score, dict) else {},
            "external_data_path": self.external_validation_data_path,
        }

    def _build_external_validation_calibration_status(self) -> Dict[str, Any]:
        return {
            "available": False,
            "assessment_scope": "external_holdout_probability_assessment",
            "requires_external_validation": False,
            "external_validation_available": True,
            "requires_external_recalibration_review": True,
            "warning": (
                "Probability calibration is assessed on an external holdout cohort. "
                "Additional site-level transportability review and possible recalibration may still be required."
            ),
        }

    def _build_phase2_external_validation_payload(self, final_node: CandidateNode) -> Dict[str, Any]:
        if not self.external_validation_data_path or not os.path.exists(self.external_validation_data_path):
            return {}

        winner_eval = self._evaluate_on_external_holdout(
            final_node.get("features", []),
            panel_name="Phase2 Winner External Holdout",
            selected_model=str(final_node.get("selected_model", "")),
        )
        if not winner_eval or not winner_eval.get("cv_predictions"):
            return {}

        baseline_source = final_node.get("phase1_panel_baseline_evaluation", {})
        baseline_features = []
        baseline_model = "Phase1 Final Panel"
        if isinstance(baseline_source, dict):
            baseline_features = baseline_source.get("features", []) or []
            baseline_model = str(
                baseline_source.get("selected_model")
                or baseline_source.get("panel_name")
                or baseline_model
            )

        baseline_eval: Optional[Dict[str, Any]] = None
        if baseline_features:
            baseline_eval = self._evaluate_on_external_holdout(
                baseline_features,
                panel_name="Phase1 Final Panel External Holdout",
                selected_model=baseline_model,
            )

        scenario_definition = self._resolve_clinical_scenario_definition()
        try:
            payload = build_clinical_utility_payload(
                winner_eval=winner_eval,
                baseline_eval=baseline_eval,
                scenario_definition=scenario_definition,
                calibration_status=self._build_external_validation_calibration_status(),
                positive_label=1,
            )
        except Exception as exc:
            return {
                "schema_version": "phase2.external_validation.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "panel_name": "Phase 2 Pareto-SFS Winner",
                "data_path": self.data_path,
                "external_data_path": self.external_validation_data_path,
                "target_column": self.target_column,
                "error": f"external_validation_payload_failed: {exc}",
            }

        payload["schema_version"] = "phase2.external_validation.v1"
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["panel_name"] = "Phase 2 Pareto-SFS Winner"
        payload["data_path"] = self.data_path
        payload["external_data_path"] = self.external_validation_data_path
        payload["target_column"] = self.target_column
        payload["selected_model"] = final_node.get("selected_model")
        payload["selected_features"] = copy.deepcopy(final_node.get("features", []))
        payload["winner_scores"] = {
            "f_perf": winner_eval.get("roc_auc", winner_eval.get("perf", 0.0)),
            "f_bio": winner_eval.get("bio", 0.0),
            "f_corr": winner_eval.get("corr", 0.0),
            "f_cost": winner_eval.get("cost", 0.0),
            "roc_auc": winner_eval.get("roc_auc", winner_eval.get("perf", 0.0)),
            "search_mean_holdout_score": winner_eval.get("perf", 0.0),
        }
        if baseline_eval:
            payload["baseline_scores"] = {
                "f_perf": baseline_eval.get("roc_auc", baseline_eval.get("perf", 0.0)),
                "f_bio": baseline_eval.get("bio", 0.0),
                "f_corr": baseline_eval.get("corr", 0.0),
                "f_cost": baseline_eval.get("cost", 0.0),
                "roc_auc": baseline_eval.get("roc_auc", baseline_eval.get("perf", 0.0)),
                "search_mean_holdout_score": baseline_eval.get("perf", 0.0),
            }
        payload["incremental_value"] = _build_incremental_value_payload(
            baseline_eval,
            winner_eval,
            baseline_name="phase1_final_panel_external_holdout",
            comparison_scope="external_holdout",
            positive_label=1,
            bootstrap_iterations=200,
        )
        payload["winner_prediction_payload"] = copy.deepcopy(winner_eval.get("cv_predictions", {}))
        payload["baseline_prediction_payload"] = (
            copy.deepcopy(baseline_eval.get("cv_predictions", {}))
            if baseline_eval and baseline_eval.get("cv_predictions")
            else {}
        )
        winner_probability_diagnostics = _build_probability_column_diagnostics(payload["winner_prediction_payload"])
        payload["class_probability_mapping"] = winner_probability_diagnostics["class_probability_mapping"]
        payload["positive_label"] = winner_probability_diagnostics["positive_label"]
        payload["positive_probability_column"] = winner_probability_diagnostics["positive_probability_column"]
        payload["auc_using_class0"] = winner_probability_diagnostics["auc_using_class0"]
        payload["auc_using_class1"] = winner_probability_diagnostics["auc_using_class1"]
        payload["possible_direction_reversal_warning"] = winner_probability_diagnostics[
            "possible_direction_reversal_warning"
        ]
        payload["probability_column_diagnostics"] = {
            "winner": winner_probability_diagnostics,
            "baseline": _build_probability_column_diagnostics(payload["baseline_prediction_payload"])
            if payload["baseline_prediction_payload"]
            else {},
        }
        payload["winner_prediction_artifact_filename"] = "phase2_external_winner_predictions.csv"
        if payload["baseline_prediction_payload"]:
            payload["baseline_prediction_artifact_filename"] = "phase2_external_baseline_predictions.csv"
        payload["threshold_metrics_artifact_filename"] = "phase2_external_validation_table.csv"
        return _convert_numpy_types(payload)

    def _resolve_available_prior_anchor_features(self) -> List[str]:
        if self.available_prior_anchor_features:
            return copy.deepcopy(self.available_prior_anchor_features)
        anchor_ids = {
            str(anchor_id).strip().upper()
            for anchor_id in (self.bio_context.protected_anchor_ids if self.bio_context else [])
            if str(anchor_id).strip()
        }
        if not anchor_ids:
            return copy.deepcopy(self.protected_features)

        resolved = list(self.protected_features)
        for feature in self.sorted_base_pool:
            feature_upper = str(feature).strip().upper()
            hmdb_prefix = feature_upper.split("_")[0] if feature_upper.startswith("HMDB") else feature_upper
            if feature_upper in anchor_ids or hmdb_prefix in anchor_ids:
                resolved.append(feature)
        return _normalize_features(resolved)

    def _evaluate_phase1_panel_baseline(self) -> CandidateNode:
        """
        Evaluate the original Phase 1 final panel with the Phase 1 champion config.
        """
        if self.phase1_panel_baseline_eval_cache is not None:
            return copy.deepcopy(self.phase1_panel_baseline_eval_cache)

        baseline_features = _normalize_features(self.phase1_panel_baseline_features)
        if not baseline_features:
            self.phase1_panel_baseline_eval_cache = {}
            return {}

        try:
            baseline_eval = self._evaluate_with_phase1_champion(baseline_features)
        except Exception:
            baseline_eval = {}
        self.phase1_panel_baseline_eval_cache = copy.deepcopy(baseline_eval)
        return copy.deepcopy(baseline_eval)

    def _resolve_clinical_scenario_definition(self) -> Dict[str, Any]:
        """Resolve the normalized scenario definition for clinical utility reporting."""
        if hasattr(self.cfg, "get_scenario_definition"):
            try:
                scenario_definition = self.cfg.get_scenario_definition(self.resolved_clinical_scenario)
                if isinstance(scenario_definition, dict) and scenario_definition:
                    return copy.deepcopy(scenario_definition)
            except Exception:
                pass

        scenarios = ((getattr(self.cfg, "config", {}) or {}).get("evaluation", {}) or {}).get("scenarios", {}) or {}
        raw_definition = copy.deepcopy(scenarios.get(self.resolved_clinical_scenario, {}))
        return {
            "key": self.resolved_clinical_scenario,
            "name": str(raw_definition.get("name", self.resolved_clinical_scenario)),
            "description": str(raw_definition.get("description", "")),
            "intended_use": str(raw_definition.get("intended_use", "risk stratification")),
            "clinical_action": str(
                raw_definition.get("clinical_action", "flag high-risk individuals for confirmatory assessment")
            ),
            "weights": copy.deepcopy(raw_definition.get("weights", {})),
            "risk_thresholds": copy.deepcopy(raw_definition.get("risk_thresholds", [0.10, 0.15, 0.20])),
            "default_action_threshold": float(raw_definition.get("default_action_threshold", 0.10) or 0.10),
            "study_design": str(raw_definition.get("study_design", "case_control") or "case_control"),
            "use_observed_prevalence": bool(raw_definition.get("use_observed_prevalence", True)),
            "target_prevalence": raw_definition.get("target_prevalence"),
            "prevalence_source": str(raw_definition.get("prevalence_source", "")),
            "prevalence_reference_population": str(raw_definition.get("prevalence_reference_population", "")),
            "report_ece": bool(raw_definition.get("report_ece", False)),
            "resource_assumptions": copy.deepcopy(raw_definition.get("resource_assumptions", {})),
        }

    def _build_clinical_utility_calibration_status(self) -> Dict[str, Any]:
        """Provide a stable calibration-status placeholder for downstream artifact generation."""
        return {
            "available": False,
            "warning": (
                "Risk thresholds are derived from internal out-of-fold probabilities "
                "without formal probability calibration."
            ),
            "bio_calibration_enabled": bool(self.f_bio_calibration_config.get("enabled", True)),
            "bio_calibration_artifact_filename": "phase2_bio_calibration.json",
        }

    def _build_probability_recalibration_status(self) -> Dict[str, Any]:
        return {
            "formal_recalibration_applied": True,
            "recalibration_method": str(self.probability_recalibration_config.get("method", "platt") or "platt"),
            "assessment_scope": "internal_cross_fitted_recalibration",
            "recommended_for_current_internal_use": True,
            "requires_external_validation": True,
            "requires_external_recalibration_review": True,
            "warning": (
                "Current clinical thresholds should be interpreted on recalibrated probabilities only. "
                "External validation and possible site-specific recalibration remain required before deployment."
            ),
        }

    def _build_probability_recalibration_config_payload(self) -> Dict[str, Any]:
        method = str(self.probability_recalibration_config.get("method", "platt") or "platt")
        clip_min = float(self.probability_recalibration_config.get("clip_min", 1e-6) or 1e-6)
        return {
            "method": method,
            "strategy": "cross_fitted",
            "outer_cv_folds": int(self.k_folds),
            "inner_calibration_folds": max(2, int(self.k_folds)),
            "positive_label": 1,
            "probability_field": "pred_proba_class1",
            "selection_policy": str(
                self.probability_recalibration_config.get("selection_policy", "fixed_default_platt")
                or "fixed_default_platt"
            ),
            "clip_min": clip_min,
            "clip_max": float(1.0 - clip_min),
        }

    def _build_probability_recalibration_payload(
        self,
        winner_eval: Dict[str, Any],
        scenario_definition: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not bool(self.probability_recalibration_config.get("enabled", True)):
            return {}
        if not isinstance(winner_eval, dict):
            return {}
        raw_prediction_payload = copy.deepcopy(winner_eval.get("cv_predictions", {}))
        if not isinstance(raw_prediction_payload, dict) or not raw_prediction_payload.get("predictions"):
            return {}

        try:
            recalibration_result = run_cross_fitted_probability_recalibration(
                raw_prediction_payload,
                method=str(self.probability_recalibration_config.get("method", "platt") or "platt"),
                clip_min=float(self.probability_recalibration_config.get("clip_min", 1e-6) or 1e-6),
            )
            return build_probability_recalibration_artifact(
                raw_prediction_payload=recalibration_result["raw_prediction_payload"],
                calibrated_prediction_payload=recalibration_result["calibrated_prediction_payload"],
                recalibration_status=self._build_probability_recalibration_status(),
                recalibration_config=self._build_probability_recalibration_config_payload(),
                generated_at=datetime.now(timezone.utc).isoformat(),
                panel_name="Phase 2 Pareto-SFS Winner",
                data_path=self.data_path,
                target_column=self.target_column,
                source_thresholds=(scenario_definition or {}).get("risk_thresholds", []),
            )
        except Exception as exc:
            return {
                "schema_version": "phase2.probability_recalibration.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "panel_name": "Phase 2 Pareto-SFS Winner",
                "data_path": self.data_path,
                "target_column": self.target_column,
                "error": f"probability_recalibration_failed: {exc}",
            }

    def _normalize_clinical_utility_inputs(self, clinical_utility_inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fill in probability recalibration details when callers provide a partial payload.

        Some tests and older call sites build `clinical_utility_inputs` manually with
        only `winner_eval` and `scenario_definition`. Export should still emit the
        recalibration artifact and expose calibrated probabilities consistently.
        """
        if not isinstance(clinical_utility_inputs, dict) or not clinical_utility_inputs:
            return {}

        normalized_inputs = copy.deepcopy(clinical_utility_inputs)
        winner_eval = normalized_inputs.get("winner_eval")
        if not isinstance(winner_eval, dict):
            return normalized_inputs

        scenario_definition = normalized_inputs.get("scenario_definition")
        if not isinstance(scenario_definition, dict) or not scenario_definition:
            scenario_definition = self._resolve_clinical_scenario_definition()
            normalized_inputs["scenario_definition"] = scenario_definition

        probability_recalibration = normalized_inputs.get("probability_recalibration", {})
        if not isinstance(probability_recalibration, dict) or (
            not probability_recalibration and winner_eval.get("cv_predictions")
        ):
            probability_recalibration = self._build_probability_recalibration_payload(
                winner_eval,
                scenario_definition=scenario_definition,
            )
            normalized_inputs["probability_recalibration"] = probability_recalibration

        calibration_status = normalized_inputs.get("calibration_status")
        if not isinstance(calibration_status, dict):
            calibration_status = self._build_clinical_utility_calibration_status()

        if isinstance(probability_recalibration, dict) and probability_recalibration and not probability_recalibration.get("error"):
            raw_payload = copy.deepcopy(probability_recalibration.get("raw_prediction_payload", {}))
            calibrated_payload = copy.deepcopy(probability_recalibration.get("calibrated_prediction_payload", {}))
            if raw_payload and calibrated_payload:
                winner_eval = copy.deepcopy(winner_eval)
                winner_eval["raw_cv_predictions"] = raw_payload
                winner_eval["cv_predictions"] = calibrated_payload
                winner_eval["probability_source"] = {
                    "default_probability_payload": "calibrated_prediction_payload",
                    "raw_payload_available": True,
                    "calibrated_payload_available": True,
                    "formal_recalibration_applied": True,
                    "recalibration_method": probability_recalibration.get("recalibration_status", {}).get(
                        "recalibration_method",
                        "platt",
                    ),
                    "source_artifact_filename": "phase2_probability_recalibration.json",
                }
                normalized_inputs["winner_eval"] = winner_eval
                calibration_status = {
                    **self._build_clinical_utility_calibration_status(),
                    **calibration_status,
                    **copy.deepcopy(probability_recalibration.get("recalibration_status", {})),
                    "available": True,
                    "probability_recalibration_recommended": False,
                }

        normalized_inputs["calibration_status"] = calibration_status
        return normalized_inputs

    def _build_clinical_utility_inputs(
        self,
        final_node: CandidateNode,
        phase1_panel_baseline_eval: Optional[CandidateNode],
    ) -> Dict[str, Any]:
        """
        Package Phase 2 outputs required to build the later clinical utility artifact.
        """
        winner_eval = {
            "panel_name": "Phase2 Winner",
            "selected_model": final_node.get("selected_model"),
            "features": copy.deepcopy(final_node.get("features", [])),
            "cv_predictions": copy.deepcopy(final_node.get("cv_predictions", {})),
            "comprehensive_metrics": copy.deepcopy(final_node.get("comprehensive_metrics", {})),
        }
        baseline_eval: Optional[Dict[str, Any]] = None
        if isinstance(phase1_panel_baseline_eval, dict) and phase1_panel_baseline_eval.get("cv_predictions"):
            baseline_eval = {
                "panel_name": "Phase1 Final Panel",
                "selected_model": phase1_panel_baseline_eval.get("selected_model"),
                "features": copy.deepcopy(phase1_panel_baseline_eval.get("features", [])),
                "cv_predictions": copy.deepcopy(phase1_panel_baseline_eval.get("cv_predictions", {})),
                "comprehensive_metrics": copy.deepcopy(
                    phase1_panel_baseline_eval.get("comprehensive_metrics", {})
                ),
            }

        scenario_definition = self._resolve_clinical_scenario_definition()
        probability_recalibration = self._build_probability_recalibration_payload(
            winner_eval,
            scenario_definition=scenario_definition,
        )
        default_calibration_status = self._build_clinical_utility_calibration_status()

        if probability_recalibration and not probability_recalibration.get("error"):
            raw_payload = copy.deepcopy(probability_recalibration.get("raw_prediction_payload", {}))
            calibrated_payload = copy.deepcopy(probability_recalibration.get("calibrated_prediction_payload", {}))
            if raw_payload and calibrated_payload:
                winner_eval["raw_cv_predictions"] = raw_payload
                winner_eval["cv_predictions"] = calibrated_payload
                winner_eval["probability_source"] = {
                    "default_probability_payload": "calibrated_prediction_payload",
                    "raw_payload_available": True,
                    "calibrated_payload_available": True,
                    "formal_recalibration_applied": True,
                    "recalibration_method": probability_recalibration.get("recalibration_status", {}).get(
                        "recalibration_method",
                        "platt",
                    ),
                    "source_artifact_filename": "phase2_probability_recalibration.json",
                }
                default_calibration_status = {
                    **default_calibration_status,
                    **copy.deepcopy(probability_recalibration.get("recalibration_status", {})),
                    "available": True,
                    "probability_recalibration_recommended": False,
                }

        return {
            "winner_eval": winner_eval,
            "baseline_eval": baseline_eval,
            "scenario_definition": scenario_definition,
            "calibration_status": default_calibration_status,
            "probability_recalibration": probability_recalibration,
        }

    def _build_prior_anchor_union_evaluation(
        self,
        final_node: CandidateNode,
        search_final_node: CandidateNode,
    ) -> Dict[str, Any]:
        available_anchor_features = self._resolve_available_prior_anchor_features()
        union_features = _normalize_features(list(final_node.get("features", [])) + list(available_anchor_features))

        search_union_eval: CandidateNode
        if _candidate_key(union_features) == _candidate_key(search_final_node.get("features", [])):
            search_union_eval = copy.deepcopy(search_final_node)
        else:
            search_union_eval = self._ensure_predictions({"features": union_features})
            search_union_eval["selected_model"] = "RandomForest (deterministic search mode)"
            search_union_eval["comprehensive_metrics"] = _calculate_comprehensive_metrics(
                search_union_eval.get("cv_predictions", {}),
                self.metric,
            )

        phase1_union_eval: CandidateNode
        if _candidate_key(union_features) == _candidate_key(final_node.get("features", [])):
            phase1_union_eval = copy.deepcopy(final_node)
        else:
            phase1_union_eval = self._evaluate_with_phase1_champion(union_features)

        return {
            "available_prior_anchor_features": available_anchor_features,
            "n_available_prior_anchor_features": len(available_anchor_features),
            "union_features": union_features,
            "identical_to_final_panel": _candidate_key(union_features) == _candidate_key(final_node.get("features", [])),
            "search_model_evaluation": search_union_eval,
            "phase1_model_evaluation": phase1_union_eval,
        }

    def _print_final_trace_summary(self, final_node: CandidateNode) -> None:
        if not self.verbose_search_trace:
            return
        print("\n" + "#" * 100)
        print("[Phase2 Trace] Final Summary")
        self._print_panel_evaluation("Search Winner", final_node.get("search_model_evaluation"))
        self._print_panel_evaluation("Final Display Winner", final_node)
        self._print_top_candidate_beams_summary(final_node.get("top_candidate_beams", []))
        phase1_panel_baseline = final_node.get("phase1_panel_baseline_evaluation")
        if phase1_panel_baseline:
            self._print_panel_evaluation("Phase1 Panel Baseline", phase1_panel_baseline)
        phase1_eval = final_node.get("phase1_model_evaluation")
        if phase1_eval:
            self._print_panel_evaluation("Alternate Phase1 Champion Eval", phase1_eval)
        prior_anchor_union = final_node.get("prior_anchor_union_evaluation") if isinstance(final_node.get("prior_anchor_union_evaluation"), dict) else {}
        if prior_anchor_union:
            print(f"Available Prior Anchor Features: {prior_anchor_union.get('available_prior_anchor_features', [])}")
            self._print_panel_evaluation(
                "Final Panel + Prior Anchors (Search Eval)",
                prior_anchor_union.get("search_model_evaluation"),
            )
            self._print_panel_evaluation(
                "Final Panel + Prior Anchors (Phase1 Champion Eval)",
                prior_anchor_union.get("phase1_model_evaluation"),
            )
        incremental_value = final_node.get("incremental_value") if isinstance(final_node.get("incremental_value"), dict) else {}
        if incremental_value:
            print("Incremental Value vs Phase1 Baseline")
            print(f"  baseline: {incremental_value.get('baseline_name')}")
            print(f"  scope: {incremental_value.get('comparison_scope')}")
            if incremental_value.get("enabled"):
                nri = incremental_value.get("nri", {}) or {}
                idi = incremental_value.get("idi", {}) or {}
                print(
                    f"  continuous_nri={float(nri.get('value', 0.0) or 0.0):.4f}, "
                    f"idi={float(idi.get('value', 0.0) or 0.0):.4f}"
                )
                print(f"  nri_ci_95={nri.get('ci_95')}, idi_ci_95={idi.get('ci_95')}")
            else:
                print(f"  unavailable: {incremental_value.get('reason', 'unknown')}")
        print("#" * 100)

    def _resolve_final_display_node(self, search_node: CandidateNode) -> Tuple[CandidateNode, CandidateNode]:
        """
        Compare Phase 1 champion final evaluation vs search-stage RandomForest result.
        If champion underperforms on the primary metric, keep RandomForest for final display.
        """
        search_eval = copy.deepcopy(search_node)
        search_eval["selected_model"] = "RandomForest (deterministic search mode)"
        if search_eval.get("comprehensive_metrics") is None:
            search_eval["comprehensive_metrics"] = _calculate_comprehensive_metrics(search_eval["cv_predictions"], self.metric)

        try:
            champion_eval = self._evaluate_with_phase1_champion(search_eval["features"])
        except Exception:
            champion_eval = {}

        if champion_eval and float(champion_eval.get("perf", 0.0)) >= float(search_eval.get("perf", 0.0)):
            return champion_eval, search_eval
        return search_eval, champion_eval

    def run(self) -> CandidateNode:
        initial_panel_features = copy.deepcopy(self.protected_features)
        self.search_details["initial_panel_features"] = copy.deepcopy(initial_panel_features)

        current_beams: List[Any]
        final_beams: List[CandidateNode]
        if initial_panel_features:
            initial_node = self._build_node(initial_panel_features, include_predictions=False)
            current_beams = [copy.deepcopy(initial_node)]
            final_beams = [copy.deepcopy(initial_node)]
        else:
            current_beams = [[]]
            final_beams = []
        stop_reason = "max_depth_reached"
        rollback_triggered = False
        depth = 0

        while depth < self.max_depth:
            expanded = self._expand_current_beams(current_beams)
            if not expanded:
                stop_reason = "pool_exhausted"
                break

            threshold = self._performance_threshold(depth)
            survivors = [node for node in expanded if float(node["perf"]) >= threshold]
            survivors = self._maybe_disable_hard_prune_below_min_features(
                expanded,
                survivors,
                depth,
                threshold,
            )
            if not survivors:
                stop_reason = "hard_prune_exhausted"
                candidate_trace = self._build_candidate_trace(expanded, threshold)
                self._record_layer(
                    depth,
                    len(expanded),
                    threshold,
                    survivors,
                    [],
                    [],
                    [],
                    None,
                    None,
                    "hard_prune_exhausted",
                    candidate_trace,
                    {},
                )
                self._print_layer_trace(
                    depth,
                    threshold,
                    candidate_trace,
                    [],
                    [],
                    [],
                    None,
                    None,
                    None,
                    "hard_prune_exhausted",
                )
                break

            if self.objective_mode == "performance_only":
                pareto_front, selected_beams, epsilon_metadata = self._select_performance_only_beams(survivors)
                epsilon_feasible_front = copy.deepcopy(pareto_front)
            else:
                pareto_front = get_pareto_front(survivors)
                epsilon_feasible_front, epsilon_metadata = self._build_soft_epsilon_feasible_front(pareto_front)
                topsis_ranked = calculate_topsis_scores(epsilon_feasible_front, weights=self.topsis_weights)
                selected_beams = filter_diverse_top_k(topsis_ranked, self.beam_width)
            if not pareto_front:
                stop_reason = "pareto_empty"
                candidate_trace = self._build_candidate_trace(expanded, threshold)
                self._record_layer(
                    depth,
                    len(expanded),
                    threshold,
                    survivors,
                    pareto_front,
                    [],
                    [],
                    None,
                    None,
                    "pareto_empty",
                    candidate_trace,
                    {},
                )
                self._print_layer_trace(
                    depth,
                    threshold,
                    candidate_trace,
                    pareto_front,
                    [],
                    [],
                    None,
                    None,
                    None,
                    "pareto_empty",
                    None,
                    None,
                    None,
                )
                break

            if not selected_beams:
                stop_reason = "beam_empty"
                candidate_trace = self._build_candidate_trace(expanded, threshold, pareto_front, epsilon_feasible_front, [])
                self._record_layer(
                    depth,
                    len(expanded),
                    threshold,
                    survivors,
                    pareto_front,
                    epsilon_feasible_front,
                    selected_beams,
                    None,
                    None,
                    "beam_empty",
                    candidate_trace,
                    epsilon_metadata,
                )
                self._print_layer_trace(
                    depth,
                    threshold,
                    candidate_trace,
                    pareto_front,
                    epsilon_feasible_front,
                    selected_beams,
                    None,
                    None,
                    None,
                    "beam_empty",
                    epsilon_metadata.get("best_perf_in_front"),
                    epsilon_metadata.get("perf_margin"),
                    epsilon_metadata.get("fallback_to_pareto"),
                )
                break

            top1 = self._ensure_predictions(selected_beams[0])
            selected_beams[0] = copy.deepcopy(top1)
            self._update_best_min_feature_candidate(top1, selected_beams, depth)
            candidate_trace = self._build_candidate_trace(expanded, threshold, pareto_front, epsilon_feasible_front, selected_beams)
            final_beams = [copy.deepcopy(node) for node in selected_beams]
            current_beams = [copy.deepcopy(node) for node in selected_beams]

            p_value: Optional[float] = None
            delta_auc: Optional[float] = None
            improvement = "initialized"

            if self.global_best_node is None:
                self._update_global_best(top1, selected_beams, depth)
                improvement = "global_best_initialized"
            else:
                best_y_true, best_y_score = _extract_binary_predictions(self.global_best_node["cv_predictions"])
                top_y_true, top_y_score = _extract_binary_predictions(top1["cv_predictions"])
                if best_y_true.size and top_y_true.size and np.array_equal(best_y_true, top_y_true):
                    p_value = delong_roc_test(best_y_true, top_y_score, best_y_score)
                else:
                    p_value = 1.0

                delta_auc = float(top1.get("roc_auc", 0.0) - self.global_best_node.get("roc_auc", 0.0))
                is_breakthrough, improvement = self._classify_auc_breakthrough(p_value, delta_auc)
                
                # Rule: Do not trigger patience-based early stopping if we have fewer than 5 features
                # This ensures we reach a minimum complexity before allowing the search to plateau.
                current_feature_count = len(top1["features"])
                min_features_before_patience = 5
                
                if is_breakthrough:
                    self._update_global_best(top1, selected_beams, depth)
                elif current_feature_count < min_features_before_patience:
                    # Not significant, but we haven't reached the minimum feature count yet.
                    # We update global best anyway to keep the "best so far" moving forward,
                    # but we don't increment patience.
                    self._update_global_best(top1, selected_beams, depth)
                    improvement = "forced_growth_below_min_features"
                else:
                    self.patience_counter += 1
                    improvement = "no_significant_breakthrough"
                    if self.patience_counter >= self.patience:
                        rollback_triggered = True
                        stop_reason = "delong_patience_triggered"
                        final_beams = [copy.deepcopy(node) for node in self.global_best_beams]
                        self._record_layer(
                            depth,
                            len(expanded),
                            threshold,
                            survivors,
                            pareto_front,
                            epsilon_feasible_front,
                            selected_beams,
                            p_value,
                            delta_auc,
                            improvement,
                            candidate_trace,
                            epsilon_metadata,
                        )
                        self._print_layer_trace(
                            depth,
                            threshold,
                            candidate_trace,
                            pareto_front,
                            epsilon_feasible_front,
                            selected_beams,
                            top1["features"],
                            p_value,
                            delta_auc,
                            improvement,
                            epsilon_metadata.get("best_perf_in_front"),
                            epsilon_metadata.get("perf_margin"),
                            epsilon_metadata.get("fallback_to_pareto"),
                        )
                        break

            self._record_layer(
                depth,
                len(expanded),
                threshold,
                survivors,
                pareto_front,
                epsilon_feasible_front,
                selected_beams,
                p_value,
                delta_auc,
                improvement,
                candidate_trace,
                epsilon_metadata,
            )
            self._print_layer_trace(
                depth,
                threshold,
                candidate_trace,
                pareto_front,
                epsilon_feasible_front,
                selected_beams,
                top1["features"],
                p_value,
                delta_auc,
                improvement,
                epsilon_metadata.get("best_perf_in_front"),
                epsilon_metadata.get("perf_margin"),
                epsilon_metadata.get("fallback_to_pareto"),
            )
            depth += 1

        if not final_beams and self.global_best_beams:
            final_beams = [copy.deepcopy(node) for node in self.global_best_beams]
            stop_reason = "fallback_to_global_best"
            
        # Enforce rollback to global best if the search terminated but current node isn't the best
        if final_beams and self.global_best_beams and not rollback_triggered:
            final_node = final_beams[0]
            best_node = self.global_best_beams[0]
            if _candidate_key(final_node["features"]) != _candidate_key(best_node["features"]):
                final_beams = [copy.deepcopy(node) for node in self.global_best_beams]
                stop_reason = f"{stop_reason}_rollback_to_global_best"
                rollback_triggered = True

        final_beams, min_feature_guard_applied = self._apply_min_feature_guard(final_beams)
        self.search_details["min_final_feature_count"] = self.min_final_feature_count
        self.search_details["min_feature_guard_applied"] = min_feature_guard_applied

        if not final_beams:
            return {
                "features": [],
                "perf": 0.0,
                "bio": 0.0,
                "corr": 0.0,
                "cost": 0.0,
                "global_icer": None,
                "selected_model": None,
                "final_display_source": None,
                "search_model_evaluation": {},
                "phase1_panel_baseline_evaluation": {},
                "phase1_model_evaluation": {},
                "incremental_value": {},
                "clinical_scenario_definition": self._resolve_clinical_scenario_definition(),
                "clinical_utility_inputs": {},
                "search_details": self.search_details,
                "stop_reason": stop_reason,
                "rollback_triggered": rollback_triggered,
                "min_feature_guard_applied": min_feature_guard_applied,
                "clinical_scenario": self.resolved_clinical_scenario,
                "topsis_weights": copy.deepcopy(self.topsis_weights),
                "protected_anchor_features": copy.deepcopy(self.protected_features),
                "n_protected_anchor_features": len(self.protected_features),
                "missing_protected_anchor_features": copy.deepcopy(self.missing_protected_features),
                "bio_score_version": "f_bio_v2" if self.use_f_bio_v2 else "f_bio_v1",
            }

        search_final_node = self._ensure_predictions(final_beams[0])
        if self.global_best_node is not None and _candidate_key(search_final_node["features"]) == _candidate_key(self.global_best_node["features"]):
            search_final_node["global_icer"] = self.global_best_node.get("global_icer")
        else:
            search_final_node["global_icer"] = calculate_global_icer(search_final_node, self.global_best_node)
        search_final_node["selected_model"] = "RandomForest (deterministic search mode)"
        search_final_node["comprehensive_metrics"] = _calculate_comprehensive_metrics(search_final_node["cv_predictions"], self.metric)

        display_final_node, phase1_eval_node = self._resolve_final_display_node(search_final_node)
        display_final_node = copy.deepcopy(display_final_node)
        display_final_node["global_icer"] = search_final_node.get("global_icer")
        final_display_source = "phase1_champion" if "Phase 1 champion config" in str(display_final_node.get("selected_model", "")) else "randomforest_search"

        display_final_node["search_model_evaluation"] = copy.deepcopy(search_final_node)
        display_final_node["phase1_model_evaluation"] = copy.deepcopy(phase1_eval_node) if phase1_eval_node else {}
        display_final_node["final_display_source"] = final_display_source

        final_node = display_final_node
        phase1_panel_baseline_eval = self._evaluate_phase1_panel_baseline()
        final_node["clinical_scenario"] = self.resolved_clinical_scenario
        final_node["objective_mode"] = self.objective_mode
        final_node["clinical_scenario_definition"] = self._resolve_clinical_scenario_definition()
        final_node["topsis_weights"] = copy.deepcopy(self.topsis_weights)
        final_node["search_details"] = self.search_details
        final_node["stop_reason"] = stop_reason
        final_node["rollback_triggered"] = rollback_triggered
        final_node["min_feature_guard_applied"] = min_feature_guard_applied
        final_node["beam_width"] = self.beam_width
        final_node["metric"] = self.metric
        final_node["global_best_depth"] = self.global_best_depth
        final_node["protected_anchor_features"] = copy.deepcopy(self.protected_features)
        final_node["n_protected_anchor_features"] = len(self.protected_features)
        final_node["missing_protected_anchor_features"] = copy.deepcopy(self.missing_protected_features)
        final_node["bio_score_version"] = str(
            final_node.get("bio_score_version")
            or search_final_node.get("bio_score_version")
            or ("f_bio_v2" if self.use_f_bio_v2 else "f_bio_v1")
        )
        if final_node.get("bio_debug") is None and search_final_node.get("bio_debug") is not None:
            final_node["bio_debug"] = copy.deepcopy(search_final_node.get("bio_debug"))
        final_node["n_added_features_beyond_protected"] = max(
            0,
            len(final_node.get("features", []) or []) - len(self.protected_features),
        )
        final_node["phase1_panel_baseline_evaluation"] = copy.deepcopy(phase1_panel_baseline_eval) if phase1_panel_baseline_eval else {}
        final_node["available_prior_anchor_features"] = self._resolve_available_prior_anchor_features()
        final_node["prior_anchor_union_evaluation"] = self._build_prior_anchor_union_evaluation(final_node, search_final_node)
        final_node["incremental_value"] = _build_incremental_value_payload(
            phase1_panel_baseline_eval,
            final_node,
            baseline_name="phase1_champion_final_panel",
            comparison_scope="out_of_fold",
            positive_label=1,
            bootstrap_iterations=200,
        )
        final_node["clinical_utility_inputs"] = self._build_clinical_utility_inputs(
            final_node,
            phase1_panel_baseline_eval,
        )
        final_node["top_candidate_beams"] = self._build_top_candidate_beams_summary(
            final_beams,
            final_display_node=final_node,
            search_final_node=search_final_node,
            phase1_eval_node=phase1_eval_node,
        )

        self._print_final_trace_summary(final_node)
        self._export_artifact(final_node)
        return _convert_numpy_types(final_node)

    def _export_artifact(self, final_node: CandidateNode) -> None:
        if isinstance(final_node.get("clinical_utility_inputs"), dict):
            final_node["clinical_utility_inputs"] = self._normalize_clinical_utility_inputs(
                final_node.get("clinical_utility_inputs")
            )

        comprehensive_metrics = (
            final_node.get("comprehensive_metrics")
            if isinstance(final_node.get("comprehensive_metrics"), dict)
            else {}
        )
        display_roc_auc = final_node.get("roc_auc")
        if display_roc_auc is None:
            display_roc_auc = comprehensive_metrics.get("roc_auc")
        if display_roc_auc is None:
            display_roc_auc = final_node.get("perf", 0.0)

        cfg = get_config()
        phase1_paths = cfg.get_phase1_paths() if hasattr(cfg, "get_phase1_paths") else {}
        phase1_io_policy = cfg.get_phase1_io_policy() if hasattr(cfg, "get_phase1_io_policy") else {}
        artifact_dir = phase1_paths.get("legacy_artifacts_dir", "output/artifacts")
        canonical_artifact_dir = phase1_paths.get("artifacts_dir", artifact_dir)
        dual_write_enabled = bool(phase1_io_policy.get("dual_write_enabled", False))

        # The objective-ablation runner supplies a run-local artifact directory
        # so sequential branches cannot overwrite one another or the
        # completed Full comparator artifacts.  The override is opt-in and
        # leaves the production/default path unchanged.
        ablation_artifact_dir = str(
            os.environ.get("METABOAGENT_PHASE2_ARTIFACT_DIR", "") or ""
        ).strip()
        if ablation_artifact_dir:
            artifact_dir = ablation_artifact_dir
            canonical_artifact_dir = ablation_artifact_dir
            dual_write_enabled = False

        os.makedirs(artifact_dir, exist_ok=True)
        artifact_path = os.path.join(artifact_dir, "phase2_winner_scores.json")
        payload = {
            "panel_name": "Phase 2 Pareto-SFS Winner",
            "objective_mode": self.objective_mode,
            "objective_selection_contract": copy.deepcopy(
                self.search_details.get("objective_selection_contract", {})
            ),
            "data_path": self.data_path,
            "target_column": self.target_column,
            "selected_features": final_node.get("features", []),
            "bio_score_version": final_node.get("bio_score_version"),
            "protected_anchor_features": final_node.get("protected_anchor_features", []),
            "n_protected_anchor_features": final_node.get("n_protected_anchor_features", 0),
            "missing_protected_anchor_features": final_node.get("missing_protected_anchor_features", []),
            "n_added_features_beyond_protected": final_node.get("n_added_features_beyond_protected", 0),
            "scores": {
                "f_perf": display_roc_auc,
                "f_bio": final_node.get("bio", 0.0),
                "f_corr": final_node.get("corr", 0.0),
                "f_cost": final_node.get("cost", 0.0),
            },
            "roc_auc": display_roc_auc,
            "search_mean_cv_score": final_node.get("perf", 0.0),
            "metric": self.metric,
            "beam_width": self.beam_width,
            "max_depth_additional_features": self.max_depth,
            "global_icer": final_node.get("global_icer"),
            "selected_model": final_node.get("selected_model"),
            "bio_debug": final_node.get("bio_debug", {}),
            "cv_predictions": final_node.get("cv_predictions", {}),
            "comprehensive_metrics": final_node.get("comprehensive_metrics", {}),
            "incremental_value": final_node.get("incremental_value", {}),
            "stop_reason": final_node.get("stop_reason"),
            "rollback_triggered": final_node.get("rollback_triggered", False),
            "min_feature_guard_applied": final_node.get("min_feature_guard_applied", False),
            "available_prior_anchor_features": final_node.get("available_prior_anchor_features", []),
            "prior_anchor_union_evaluation": final_node.get("prior_anchor_union_evaluation", {}),
            "top_candidate_beams": final_node.get("top_candidate_beams", []),
        }
        payload["bio_debug_artifact_filename"] = "phase2_bio_debug.json"
        payload["clinical_utility_artifact_filename"] = "phase2_clinical_utility.json"
        if final_node.get("clinical_utility_inputs", {}).get("probability_recalibration"):
            payload["probability_recalibration_artifact_filename"] = "phase2_probability_recalibration.json"
        _write_json_payload(payload, artifact_path)

        # Sidecar audit records: generated after the winner is fixed and never
        # consumed by the Pareto search, calibration or model-selection logic.
        write_audit_json("model_selection_audit.json", {
            "assessment_domain": "model_selection",
            "status": "PASS",
            "selection_scope": "phase2_training_cross_validation",
            "target_column": self.target_column,
            "data_input": describe_path(self.data_path),
            "selected_model": final_node.get("selected_model"),
            "selection_metric": self.metric,
            "selected_features": final_node.get("features", []),
            "beam_width": self.beam_width,
            "max_depth_additional_features": self.max_depth,
            "search_summary": describe_path(os.path.join(artifact_dir, "phase2_search_summary.json")),
            "winner_scores": describe_path(artifact_path),
        })
        comprehensive = final_node.get("comprehensive_metrics", {}) or {}
        write_audit_json("performance_audit.json", {
            "assessment_domain": "performance",
            "status": "PASS" if comprehensive else "PARTIAL",
            "evaluation_scope": "cross_validated_training_predictions",
            "positive_label": 1,
            "metrics": comprehensive,
            "winner_scores": describe_path(artifact_path),
            "prediction_payload_available": bool(final_node.get("cv_predictions")),
        })
        write_audit_json("interpretability_audit.json", {
            "assessment_domain": "interpretability",
            "status": "PARTIAL",
            "feature_provenance_expected": "output/artifacts/feature_provenance.json",
            "winner_feature_contributions_expected": "output/artifacts/phase2_winner_feature_contributions.csv",
            "selected_features": final_node.get("features", []),
            "note": "Status is finalized by the Phase 3 figure/table manifest after exports complete.",
        })

        if dual_write_enabled:
            old_abs = os.path.abspath(artifact_dir)
            new_abs = os.path.abspath(canonical_artifact_dir)
            if old_abs != new_abs:
                os.makedirs(canonical_artifact_dir, exist_ok=True)
                canonical_artifact_path = os.path.join(canonical_artifact_dir, "phase2_winner_scores.json")
                _write_json_payload(payload, canonical_artifact_path)

        search_summary_payload = self._build_search_summary_payload(final_node)
        search_summary_path = os.path.join(artifact_dir, "phase2_search_summary.json")
        _write_json_payload(search_summary_payload, search_summary_path)

        if dual_write_enabled:
            old_abs = os.path.abspath(artifact_dir)
            new_abs = os.path.abspath(canonical_artifact_dir)
            if old_abs != new_abs:
                os.makedirs(canonical_artifact_dir, exist_ok=True)
                canonical_summary_path = os.path.join(canonical_artifact_dir, "phase2_search_summary.json")
                _write_json_payload(search_summary_payload, canonical_summary_path)

        try:
            from src.tools.analysis.artifact_exporters import export_phase2_winner_feature_contributions

            export_phase2_winner_feature_contributions(
                winner_scores=payload,
                search_summary=search_summary_payload,
                phase0_output=self.phase0_output,
                data_path=self.data_path,
                ag_results_path=self.ag_results_path,
                output_dir=artifact_dir,
            )
        except Exception as exc:
            print(f"⚠️  Warning: failed to export phase2 winner feature contributions artifact: {exc}")

        bio_debug_payload = self._build_phase2_bio_debug_payload(final_node)
        bio_debug_path = os.path.join(artifact_dir, "phase2_bio_debug.json")
        _write_json_payload(bio_debug_payload, bio_debug_path)

        if dual_write_enabled:
            old_abs = os.path.abspath(artifact_dir)
            new_abs = os.path.abspath(canonical_artifact_dir)
            if old_abs != new_abs:
                os.makedirs(canonical_artifact_dir, exist_ok=True)
                canonical_bio_debug_path = os.path.join(canonical_artifact_dir, "phase2_bio_debug.json")
                _write_json_payload(bio_debug_payload, canonical_bio_debug_path)

        clinical_utility_payload = self._build_phase2_clinical_utility_payload(final_node)
        if clinical_utility_payload:
            clinical_utility_path = os.path.join(artifact_dir, "phase2_clinical_utility.json")
            _write_json_payload(clinical_utility_payload, clinical_utility_path)

            threshold_rows = list(clinical_utility_payload.get("threshold_metrics_table", []) or [])
            if threshold_rows:
                threshold_table_path = os.path.join(artifact_dir, "phase2_clinical_utility_table.csv")
                pd.DataFrame(threshold_rows).to_csv(threshold_table_path, index=False)

            if dual_write_enabled:
                old_abs = os.path.abspath(artifact_dir)
                new_abs = os.path.abspath(canonical_artifact_dir)
                if old_abs != new_abs:
                    os.makedirs(canonical_artifact_dir, exist_ok=True)
                    canonical_clinical_utility_path = os.path.join(
                        canonical_artifact_dir,
                        "phase2_clinical_utility.json",
                    )
                    _write_json_payload(clinical_utility_payload, canonical_clinical_utility_path)
                    if threshold_rows:
                        canonical_threshold_table_path = os.path.join(
                            canonical_artifact_dir,
                            "phase2_clinical_utility_table.csv",
                        )
                        pd.DataFrame(threshold_rows).to_csv(canonical_threshold_table_path, index=False)

        probability_recalibration_payload = (
            final_node.get("clinical_utility_inputs", {}).get("probability_recalibration", {})
            if isinstance(final_node.get("clinical_utility_inputs", {}), dict)
            else {}
        )
        if probability_recalibration_payload and not probability_recalibration_payload.get("error"):
            recalibration_path = os.path.join(artifact_dir, "phase2_probability_recalibration.json")
            _write_json_payload(probability_recalibration_payload, recalibration_path)

            calibrated_rows = list(
                probability_recalibration_payload.get("calibrated_prediction_payload", {}).get("predictions", []) or []
            )
            raw_rows = list(
                probability_recalibration_payload.get("raw_prediction_payload", {}).get("predictions", []) or []
            )
            if raw_rows and calibrated_rows and len(raw_rows) == len(calibrated_rows):
                table_rows = []
                recalibration_method = str(
                    probability_recalibration_payload.get("recalibration_status", {}).get("recalibration_method", "platt")
                )
                for raw_row, calibrated_row in zip(raw_rows, calibrated_rows):
                    table_rows.append(
                        {
                            "sample_id": raw_row.get("sample_id"),
                            "fold": raw_row.get("fold"),
                            "true_label": raw_row.get("true_label"),
                            "true_label_encoded": raw_row.get("true_label_encoded"),
                            "pred_proba_class0_raw": raw_row.get("pred_proba_class0"),
                            "pred_proba_class1_raw": raw_row.get("pred_proba_class1"),
                            "pred_proba_class0_calibrated": calibrated_row.get("pred_proba_class0"),
                            "pred_proba_class1_calibrated": calibrated_row.get("pred_proba_class1"),
                            "recalibration_method": recalibration_method,
                            "calibration_role": "oof_calibrated",
                        }
                    )
                recalibration_table_path = os.path.join(artifact_dir, "phase2_probability_recalibration_table.csv")
                pd.DataFrame(table_rows).to_csv(recalibration_table_path, index=False)

            if dual_write_enabled:
                old_abs = os.path.abspath(artifact_dir)
                new_abs = os.path.abspath(canonical_artifact_dir)
                if old_abs != new_abs:
                    os.makedirs(canonical_artifact_dir, exist_ok=True)
                    canonical_recalibration_path = os.path.join(
                        canonical_artifact_dir,
                        "phase2_probability_recalibration.json",
                    )
                    _write_json_payload(probability_recalibration_payload, canonical_recalibration_path)
                    if raw_rows and calibrated_rows and len(raw_rows) == len(calibrated_rows):
                        canonical_recalibration_table_path = os.path.join(
                            canonical_artifact_dir,
                            "phase2_probability_recalibration_table.csv",
                        )
                        pd.DataFrame(table_rows).to_csv(canonical_recalibration_table_path, index=False)

        if bool(self.f_bio_calibration_config.get("enabled", True)):
            calibration_payload = self._build_phase2_bio_calibration_payload(final_node)
            calibration_path = os.path.join(artifact_dir, "phase2_bio_calibration.json")
            _write_json_payload(calibration_payload, calibration_path)

            if dual_write_enabled:
                old_abs = os.path.abspath(artifact_dir)
                new_abs = os.path.abspath(canonical_artifact_dir)
                if old_abs != new_abs:
                    os.makedirs(canonical_artifact_dir, exist_ok=True)
                    canonical_calibration_path = os.path.join(canonical_artifact_dir, "phase2_bio_calibration.json")
                    _write_json_payload(calibration_payload, canonical_calibration_path)

        # Lock the selected model/panel immediately before the first holdout access.
        # The lock is observational and is deliberately written before the
        # holdout evaluator is called, so the chronology can be audited later.
        lock_created_at = datetime.now(timezone.utc).isoformat()
        holdout_data_path = str(self.external_validation_data_path or "")
        write_pipeline_lock(
            run_id=str(os.environ.get("METABOAGENT_RUN_TAG", "") or ""),
            lock_created_at_utc=lock_created_at,
            status="LOCKED_BEFORE_HOLDOUT",
            selected_model=str(final_node.get("selected_model", "") or ""),
            selected_features=final_node.get("features", []) or [],
            train_data_path=str(self.data_path or ""),
            holdout_data_path=holdout_data_path,
            winner_artifact_path=artifact_path,
            evaluation_scope="locked_internal_holdout",
            lock_before_holdout=True,
        )
        holdout_started_at = datetime.now(timezone.utc).isoformat()
        external_validation_payload = self._build_phase2_external_validation_payload(final_node)
        if external_validation_payload:
            external_validation_path = os.path.join(artifact_dir, "phase2_external_validation.json")
            _write_json_payload(external_validation_payload, external_validation_path)

            external_threshold_rows = list(external_validation_payload.get("threshold_metrics_table", []) or [])
            if external_threshold_rows:
                external_threshold_table_path = os.path.join(artifact_dir, "phase2_external_validation_table.csv")
                pd.DataFrame(external_threshold_rows).to_csv(external_threshold_table_path, index=False)

            winner_prediction_payload = external_validation_payload.get("winner_prediction_payload", {})
            _write_prediction_payload_csv(
                winner_prediction_payload,
                os.path.join(artifact_dir, "phase2_external_winner_predictions.csv"),
            )

            baseline_prediction_payload = external_validation_payload.get("baseline_prediction_payload", {})
            if baseline_prediction_payload:
                _write_prediction_payload_csv(
                    baseline_prediction_payload,
                    os.path.join(artifact_dir, "phase2_external_baseline_predictions.csv"),
                )

            if dual_write_enabled:
                old_abs = os.path.abspath(artifact_dir)
                new_abs = os.path.abspath(canonical_artifact_dir)
                if old_abs != new_abs:
                    os.makedirs(canonical_artifact_dir, exist_ok=True)
                    canonical_external_validation_path = os.path.join(
                        canonical_artifact_dir,
                        "phase2_external_validation.json",
                    )
                    _write_json_payload(external_validation_payload, canonical_external_validation_path)

                    if external_threshold_rows:
                        canonical_external_threshold_table_path = os.path.join(
                            canonical_artifact_dir,
                            "phase2_external_validation_table.csv",
                        )
                        pd.DataFrame(external_threshold_rows).to_csv(
                            canonical_external_threshold_table_path,
                            index=False,
                        )

                    _write_prediction_payload_csv(
                        winner_prediction_payload,
                        os.path.join(canonical_artifact_dir, "phase2_external_winner_predictions.csv"),
                    )
                    if baseline_prediction_payload:
                        _write_prediction_payload_csv(
                            baseline_prediction_payload,
                            os.path.join(canonical_artifact_dir, "phase2_external_baseline_predictions.csv"),
                        )

        # Complete the chronology record after holdout predictions have been
        # materialized.  The payload remains explicitly internal holdout even
        # though legacy Phase 2 filenames use the word "external".
        holdout_completed_at = datetime.now(timezone.utc).isoformat()
        write_pipeline_lock(
            run_id=str(os.environ.get("METABOAGENT_RUN_TAG", "") or ""),
            lock_created_at_utc=lock_created_at,
            status="LOCKED_AND_HOLDOUT_EVALUATED" if external_validation_payload else "LOCKED_HOLDOUT_UNAVAILABLE",
            selected_model=str(final_node.get("selected_model", "") or ""),
            selected_features=final_node.get("features", []) or [],
            train_data_path=str(self.data_path or ""),
            holdout_data_path=holdout_data_path,
            winner_artifact_path=artifact_path,
            holdout_artifact_path=(
                os.path.join(artifact_dir, "phase2_external_validation.json")
                if external_validation_payload else ""
            ),
            evaluation_scope="locked_internal_holdout",
            holdout_evaluation_started_at_utc=holdout_started_at,
            holdout_evaluation_completed_at_utc=holdout_completed_at,
            lock_before_holdout=True,
        )

        # Run-scoped evaluation contract.  This is a read-only export of the
        # OOF/holdout predictions already produced above; it does not refit a
        # model or feed any new metric back into Phase 2 selection.
        try:
            evaluation_artifacts = write_evaluation_artifacts(
                winner_payload=payload,
                holdout_payload=external_validation_payload,
                phase2_artifact_path=artifact_path,
                holdout_artifact_path=(
                    os.path.join(artifact_dir, "phase2_external_validation.json")
                    if external_validation_payload else ""
                ),
                target_column=self.target_column,
                selected_model=str(final_node.get("selected_model") or ""),
                selected_features=final_node.get("features", []),
            )
            print(f"✓ Evaluation artifacts written: {evaluation_artifacts}")
            heldout_path = evaluation_artifacts.get("heldout_performance", "")
            write_audit_json("performance_audit.json", {
                "assessment_domain": "performance",
                "status": "PASS" if comprehensive else "PARTIAL",
                "evaluation_scope": "cross_validated_training_predictions",
                "evaluation_scopes": ["cross_validated_training_predictions", "locked_internal_holdout"],
                "positive_label": 1,
                "metrics": comprehensive,
                "winner_scores": describe_path(artifact_path),
                "prediction_payload_available": bool(final_node.get("cv_predictions")),
                "internal_holdout_performance": describe_path(heldout_path),
                "pipeline_lock": describe_path(os.path.join(os.environ.get("METABOAGENT_RUNTIME_ROOT", ""), "output", "audit", "pipeline_lock.json")),
            })
        except Exception as exc:
            # The observational export must never invalidate an otherwise
            # completed Phase 2 result.
            print(f"⚠️ Warning: evaluation artifact export failed: {exc}")

    def _build_phase2_clinical_utility_payload(self, final_node: CandidateNode) -> Dict[str, Any]:
        clinical_utility_inputs = final_node.get("clinical_utility_inputs", {})
        if not isinstance(clinical_utility_inputs, dict) or not clinical_utility_inputs:
            return {}

        winner_eval = clinical_utility_inputs.get("winner_eval")
        scenario_definition = clinical_utility_inputs.get("scenario_definition")
        if not isinstance(winner_eval, dict) or not winner_eval.get("cv_predictions"):
            return {}
        if not isinstance(scenario_definition, dict) or not scenario_definition.get("risk_thresholds"):
            return {}

        try:
            payload = build_clinical_utility_payload(
                winner_eval=winner_eval,
                baseline_eval=clinical_utility_inputs.get("baseline_eval"),
                scenario_definition=scenario_definition,
                calibration_status=clinical_utility_inputs.get("calibration_status"),
                positive_label=1,
            )
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
            payload["panel_name"] = "Phase 2 Pareto-SFS Winner"
            payload["data_path"] = self.data_path
            payload["target_column"] = self.target_column
            probability_recalibration = clinical_utility_inputs.get("probability_recalibration", {})
            if isinstance(probability_recalibration, dict) and probability_recalibration and not probability_recalibration.get("error"):
                raw_calibration = probability_recalibration.get("raw_probability_calibration", {})
                calibrated_calibration = probability_recalibration.get("calibrated_probability_calibration", {})
                payload["probability_source"] = {
                    "default_probability_payload": "calibrated_prediction_payload",
                    "raw_payload_available": bool(probability_recalibration.get("raw_prediction_payload")),
                    "calibrated_payload_available": bool(probability_recalibration.get("calibrated_prediction_payload")),
                    "formal_recalibration_applied": bool(
                        probability_recalibration.get("recalibration_status", {}).get("formal_recalibration_applied", False)
                    ),
                    "recalibration_method": str(
                        probability_recalibration.get("recalibration_status", {}).get("recalibration_method", "platt")
                    ),
                    "source_artifact_filename": "phase2_probability_recalibration.json",
                }
                payload["recalibration_summary"] = {
                    "raw_brier_score": raw_calibration.get("brier_score"),
                    "calibrated_brier_score": calibrated_calibration.get("brier_score"),
                    "raw_ece": raw_calibration.get("expected_calibration_error"),
                    "calibrated_ece": calibrated_calibration.get("expected_calibration_error"),
                    "raw_calibration_slope": raw_calibration.get("calibration_slope"),
                    "calibrated_calibration_slope": calibrated_calibration.get("calibration_slope"),
                    "raw_calibration_intercept": raw_calibration.get("calibration_intercept"),
                    "calibrated_calibration_intercept": calibrated_calibration.get("calibration_intercept"),
                }
                payload["probability_recalibration"] = copy.deepcopy(probability_recalibration)
            return _convert_numpy_types(payload)
        except Exception as exc:
            return {
                "schema_version": "phase2.clinical_utility.v2",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "panel_name": "Phase 2 Pareto-SFS Winner",
                "data_path": self.data_path,
                "target_column": self.target_column,
                "error": f"clinical_utility_payload_failed: {exc}",
            }

    def _build_phase2_bio_debug_payload(self, final_node: CandidateNode) -> Dict[str, Any]:
        bio_context_summary = {
            "disease_name": self.disease_name,
            "anchor_mode": self.bio_context.anchor_mode if self.bio_context else "disease_only",
            "protected_anchor_ids": copy.deepcopy(self.bio_context.protected_anchor_ids) if self.bio_context else [],
            "external_prior_seed_ids": copy.deepcopy(self.bio_context.external_prior_seed_ids) if self.bio_context else [],
            "disease_core_pathways": copy.deepcopy(self.bio_context.disease_core_pathways) if self.bio_context else [],
            "metadata": copy.deepcopy(self.bio_context.metadata) if self.bio_context else {},
            "config": copy.deepcopy(self.bio_context.config) if self.bio_context else copy.deepcopy(self.f_bio_v2_config),
        }

        search_model_eval = copy.deepcopy(final_node.get("search_model_evaluation", {}))
        phase1_model_eval = copy.deepcopy(final_node.get("phase1_model_evaluation", {}))
        payload = {
            "schema_version": "phase4.phase2_bio_debug.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "panel_name": "Phase 2 Pareto-SFS Winner",
            "disease_name": self.disease_name,
            "bio_score_version": final_node.get("bio_score_version"),
            "f_bio_v2_enabled": self.use_f_bio_v2,
            "selected_model": final_node.get("selected_model"),
            "selected_features": copy.deepcopy(final_node.get("features", [])),
            "winner_scores": {
                "f_perf": final_node.get("roc_auc", (final_node.get("comprehensive_metrics", {}) or {}).get("roc_auc", final_node.get("perf", 0.0))),
                "f_bio": final_node.get("bio", 0.0),
                "f_corr": final_node.get("corr", 0.0),
                "f_cost": final_node.get("cost", 0.0),
            },
            "roc_auc": final_node.get("roc_auc", (final_node.get("comprehensive_metrics", {}) or {}).get("roc_auc", final_node.get("perf", 0.0))),
            "search_mean_cv_score": final_node.get("perf", 0.0),
            "protected_anchor_features": copy.deepcopy(final_node.get("protected_anchor_features", [])),
            "n_protected_anchor_features": final_node.get("n_protected_anchor_features", 0),
            "missing_protected_anchor_features": copy.deepcopy(final_node.get("missing_protected_anchor_features", [])),
            "n_added_features_beyond_protected": final_node.get("n_added_features_beyond_protected", 0),
            "available_prior_anchor_features": copy.deepcopy(final_node.get("available_prior_anchor_features", [])),
            "prior_anchor_union_evaluation": copy.deepcopy(final_node.get("prior_anchor_union_evaluation", {})),
            "bio_context": bio_context_summary,
            "winner_bio_debug": copy.deepcopy(final_node.get("bio_debug", {})),
            "search_model_bio_debug": copy.deepcopy(search_model_eval.get("bio_debug", {})),
            "phase1_model_bio_debug": copy.deepcopy(phase1_model_eval.get("bio_debug", {})),
            "search_summary_snapshot": self._build_search_summary_payload(final_node),
            "search_details": copy.deepcopy(final_node.get("search_details", self.search_details)),
        }
        return _convert_numpy_types(payload)

    def _build_search_summary_payload(self, final_node: CandidateNode) -> Dict[str, Any]:
        layers = list(self.search_details.get("layers", []) or [])
        last_completed_depth = layers[-1].get("depth") if layers else None
        actual_depth = len(layers)
        stop_reason = final_node.get("stop_reason")
        early_stopping_type = _infer_phase2_early_stopping_type(
            stop_reason=stop_reason,
            max_depth=self.max_depth,
            physical_feature_limit=self.physical_feature_limit,
        )

        layer_summary: List[Dict[str, Any]] = []
        for layer in layers:
            layer_summary.append(
                {
                    "depth": layer.get("depth"),
                    "expanded_count": layer.get("expanded_count"),
                    "perf_threshold": layer.get("perf_threshold"),
                    "survivor_count": layer.get("survivor_count"),
                    "pareto_count": layer.get("pareto_count"),
                    "epsilon_feasible_count": layer.get("epsilon_feasible_count"),
                    "soft_epsilon_enabled": layer.get("soft_epsilon_enabled"),
                    "soft_epsilon_perf_margin": layer.get("soft_epsilon_perf_margin"),
                    "best_perf_in_front": layer.get("best_perf_in_front"),
                    "epsilon_fallback_to_pareto": layer.get("epsilon_fallback_to_pareto"),
                    "beam_count": layer.get("beam_count"),
                    "top1_features": layer.get("top1_features", []),
                    "top1_topsis_score": layer.get("top1_topsis_score"),
                    "top1_perf": layer.get("top1_perf"),
                    "top1_roc_auc": layer.get("top1_roc_auc"),
                    "delong_p_value": layer.get("delong_p_value"),
                    "delta_auc_vs_global_best": layer.get("delta_auc_vs_global_best"),
                    "improvement_status": layer.get("improvement_status"),
                }
            )

        top_layer = layers[-1] if layers else {}
        payload = {
            "schema_version": "phase4.phase2_search_summary.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "engine": self.search_details.get("engine", "ParetoSFS_Engine"),
            "metric": self.metric,
            "disease_name": self.disease_name,
            "bio_score_version": self.search_details.get("bio_score_version"),
            "f_bio_v2_enabled": self.search_details.get("f_bio_v2_enabled", False),
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "max_depth_additional_features": self.max_depth,
            "actual_depth": actual_depth,
            "last_completed_depth": last_completed_depth,
            "total_sample_count": self.total_sample_count,
            "minority_class_count": self.minority_class_count,
            "physical_feature_limit": self.physical_feature_limit,
            "expansion_feature_limit": self.expansion_feature_limit,
            "stop_reason": stop_reason,
            "early_stopping_type": early_stopping_type,
            "rollback_triggered": final_node.get("rollback_triggered", False),
            "min_feature_guard_applied": final_node.get("min_feature_guard_applied", False),
            "global_best_depth": final_node.get("global_best_depth"),
            "min_final_feature_count": self.search_details.get("min_final_feature_count"),
            "search_start_mode": self.search_details.get("search_start_mode"),
            "bio_context_summary": self.search_details.get("bio_context_summary", {}),
            "initial_panel_features": self.search_details.get("initial_panel_features", []),
            "protected_anchor_features": self.search_details.get("protected_features", []),
            "n_protected_anchor_features": self.search_details.get("n_protected_features", 0),
            "missing_protected_anchor_features": self.search_details.get("missing_protected_features", []),
            "available_prior_anchor_features": final_node.get("available_prior_anchor_features", []),
            "winner_feature_count": len(final_node.get("features", []) or []),
            "winner_features": final_node.get("features", []),
            "n_added_features_beyond_protected": final_node.get("n_added_features_beyond_protected", 0),
            "prior_anchor_union_evaluation": final_node.get("prior_anchor_union_evaluation", {}),
            "winner_scores": {
                "f_perf": final_node.get("roc_auc", (final_node.get("comprehensive_metrics", {}) or {}).get("roc_auc", final_node.get("perf", 0.0))),
                "f_bio": final_node.get("bio", 0.0),
                "f_corr": final_node.get("corr", 0.0),
                "f_cost": final_node.get("cost", 0.0),
            },
            "roc_auc": final_node.get("roc_auc", (final_node.get("comprehensive_metrics", {}) or {}).get("roc_auc", final_node.get("perf", 0.0))),
            "search_mean_cv_score": final_node.get("perf", 0.0),
            "incremental_value": final_node.get("incremental_value", {}),
            "top_candidate_beams": final_node.get("top_candidate_beams", []),
            "dual_criteria_summary": {
                "epv_guard_enabled": True,
                "delong_patience_enabled": True,
                "delong_patience": self.patience,
                "delong_p_value_threshold": self.delong_p_value_threshold,
                "delong_improvement_delta": self.improvement_delta,
                "practical_auc_override_enabled": self.practical_auc_override_enabled,
                "practical_auc_override": self.practical_auc_override,
                "epv_guard_triggered": early_stopping_type == "epv_limit_reached",
                "delong_guard_triggered": early_stopping_type == "delong_patience",
            },
            "top_layer_snapshot": {
                "depth": top_layer.get("depth"),
                "top1_features": top_layer.get("top1_features", []),
                "top1_perf": top_layer.get("top1_perf"),
                "top1_roc_auc": top_layer.get("top1_roc_auc"),
                "delong_p_value": top_layer.get("delong_p_value"),
                "delta_auc_vs_global_best": top_layer.get("delta_auc_vs_global_best"),
                "improvement_status": top_layer.get("improvement_status"),
            },
            "layer_summary": layer_summary,
        }
        return _convert_numpy_types(payload)


def run_ptot_search(
    data_path: str,
    target_column: str,
    candidate_pool: List[str],
    sorted_base_pool: List[str],
    taxonomy_map: Dict[str, Any],
    pathway_map: Dict[str, Any],
    priors_dict: Dict[str, float],
    checker_tool: Any = None,
    llm_caller: Any = None,
    clinical_scenario: str = "",
    max_depth: Optional[int] = None,
    ag_results_path: str = "data/autogluon_training_results.json",
    champion_model_family: Optional[str] = None,
    k_folds: int = 5,
    epsilon: float = 0.005,
    patience: int = 3,
    metric: Optional[str] = None,
    imbalance_threshold: float = 2.0,
    beam_width: int = 1,
    topsis_weights: Optional[Dict[str, float]] = None,
    protected_features: Optional[List[str]] = None,
    protected_anchor_features: Optional[List[str]] = None,
    available_prior_anchor_features: Optional[List[str]] = None,
    disease_name: Optional[str] = None,
    verbose_search_trace: bool = False,
    trace_head_limit: int = 30,
    memory_search_prior: Optional[Dict[str, Any]] = None,
    dataset_fingerprint: Optional[Dict[str, Any]] = None,
    phase0_output: Optional[Dict[str, Any]] = None,
    objective_mode: str = "four_objective",
    positive_class: Optional[str] = None,
) -> CandidateNode:
    """Backward-compatible functional entry for the deterministic engine."""
    engine = ParetoSFS_Engine(
        data_path=data_path,
        target_column=target_column,
        candidate_pool=candidate_pool,
        sorted_base_pool=sorted_base_pool,
        taxonomy_map=taxonomy_map,
        pathway_map=pathway_map,
        priors_dict=priors_dict,
        ag_results_path=ag_results_path,
        champion_model_family=champion_model_family,
        beam_width=beam_width,
        max_depth=max_depth,
        k_folds=k_folds,
        metric=metric,
        imbalance_threshold=imbalance_threshold,
        topsis_weights=topsis_weights,
        improvement_delta=epsilon,
        patience=patience,
        use_phase1_config_in_perf=False,
        protected_features=protected_features,
        protected_anchor_features=protected_anchor_features,
        available_prior_anchor_features=available_prior_anchor_features,
        disease_name=disease_name,
        clinical_scenario=clinical_scenario,
        verbose_search_trace=verbose_search_trace,
        trace_head_limit=trace_head_limit,
        memory_search_prior=memory_search_prior,
        dataset_fingerprint=dataset_fingerprint,
        phase0_output=phase0_output,
        objective_mode=objective_mode,
        positive_class=positive_class,
    )
    return engine.run()


def llm_ptot_search(*args: Any, **kwargs: Any) -> CandidateNode:
    """Compatibility alias retained for older scripts."""
    return run_ptot_search(*args, **kwargs)


__all__ = [
    "ParetoSFS_Engine",
    "calculate_topsis_scores",
    "calculate_global_icer",
    "delong_roc_test",
    "filter_diverse_top_k",
    "generate_heuristic_candidates",
    "get_pareto_front",
    "jaccard_similarity",
    "llm_ptot_search",
    "run_ptot_search",
]
