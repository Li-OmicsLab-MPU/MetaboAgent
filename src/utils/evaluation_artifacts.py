"""Run-scoped evaluation artifacts.

This module is observational: it consumes the prediction payloads already
created by Phase 2 and writes deterministic audit-side CSV/JSON files.  It
does not refit models, alter feature selection, or change the winner.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from src.utils.evaluation_audit import describe_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prediction_payload(payload: Any, *keys: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in keys:
        candidate = payload.get(key)
        if isinstance(candidate, dict) and candidate.get("predictions"):
            return candidate
    if payload.get("predictions"):
        return payload
    return {}


def _prediction_rows(payload: Dict[str, Any], model_name: str, scope: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(payload.get("predictions", []) or []):
        if not isinstance(item, dict):
            continue
        probability = item.get("pred_proba_class1")
        label = item.get("true_label_encoded", item.get("true_label"))
        try:
            y_true = int(float(label))
            y_score = float(probability)
        except (TypeError, ValueError):
            continue
        rows.append({
            "sample_id": item.get("sample_id", index),
            "fold": item.get("fold"),
            "y_true": y_true,
            "pred_probability": y_score,
            "pred_label": int(y_score >= 0.5),
            "model_name": model_name,
            "evaluation_scope": scope,
        })
    return rows


def _write_predictions_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_id", "fold", "y_true", "pred_probability", "pred_label",
        "model_name", "evaluation_scope",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(path)


def _classification_and_calibration(y_true: Sequence[int], y_score: Sequence[float]) -> Dict[str, Any]:
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return {"status": "NOT_ASSESSED", "reason": "both_classes_required"}
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    predicted = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    calibration = {"status": "NOT_ASSESSED", "slope": None, "intercept": None}
    try:
        clipped = np.clip(score, 1e-6, 1.0 - 1e-6)
        logit_score = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibrator.fit(logit_score, y)
        calibration = {
            "status": "ASSESSED",
            "slope": float(calibrator.coef_[0][0]),
            "intercept": float(calibrator.intercept_[0]),
            "fit_scope": "descriptive_locked_holdout_calibration",
        }
    except Exception as exc:
        calibration = {"status": "NOT_ASSESSED", "slope": None, "intercept": None, "reason": type(exc).__name__}
    return {
        "threshold": 0.5,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float(accuracy_score(y, predicted)),
        "sensitivity_recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "ppv_precision": float(precision_score(y, predicted, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "calibration": calibration,
    }


def _metric_summary(y_true: Sequence[int], y_score: Sequence[float]) -> Dict[str, Any]:
    if not y_true or len(set(y_true)) < 2:
        return {"status": "NOT_ASSESSED", "n_samples": len(y_true)}
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    return {
        "status": "ASSESSED",
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier_score": float(brier_score_loss(y, score)),
        "classification": _classification_and_calibration(y, score),
    }


def _bootstrap_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    seed: int,
    n_resamples: int,
) -> Dict[str, Any]:
    if not y_true or len(set(y_true)) < 2:
        return {"status": "NOT_ASSESSED", "reason": "both_classes_required"}
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    groups = [np.flatnonzero(y == 0), np.flatnonzero(y == 1)]
    rng = np.random.default_rng(seed)
    auc_values: List[float] = []
    pr_values: List[float] = []
    brier_values: List[float] = []
    for _ in range(int(n_resamples)):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        yy, ss = y[indices], score[indices]
        auc_values.append(float(roc_auc_score(yy, ss)))
        pr_values.append(float(average_precision_score(yy, ss)))
        brier_values.append(float(brier_score_loss(yy, ss)))

    def interval(values: Iterable[float]) -> List[float]:
        q = np.percentile(np.asarray(list(values), dtype=float), [2.5, 97.5])
        return [float(q[0]), float(q[1])]

    return {
        "status": "ASSESSED",
        "method": "stratified_percentile_bootstrap",
        "seed": int(seed),
        "n_resamples": int(n_resamples),
        "confidence_level": 0.95,
        "roc_auc_ci95": interval(auc_values),
        "pr_auc_ci95": interval(pr_values),
        "brier_score_ci95": interval(brier_values),
    }


def _fold_metrics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("fold"))
        grouped.setdefault(key, []).append(row)
    output = []
    for fold, fold_rows in sorted(grouped.items()):
        summary = _metric_summary(
            [int(row["y_true"]) for row in fold_rows],
            [float(row["pred_probability"]) for row in fold_rows],
        )
        summary["fold"] = None if fold in {"None", ""} else fold
        output.append(summary)
    return output


def write_evaluation_artifacts(
    *,
    winner_payload: Dict[str, Any],
    holdout_payload: Optional[Dict[str, Any]] = None,
    phase2_artifact_path: str = "",
    holdout_artifact_path: str = "",
    target_column: str = "",
    selected_model: str = "",
    selected_features: Optional[Sequence[str]] = None,
    bootstrap_seed: int = 4242,
    bootstrap_resamples: int = 2000,
) -> Dict[str, str]:
    """Write the four canonical evaluation artifacts under the current run."""
    runtime_root = Path(os.environ.get("METABOAGENT_RUNTIME_ROOT", "").strip() or ".")
    audit_dir = runtime_root / "output" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    oof_payload = _prediction_payload(winner_payload, "cv_predictions", "oof_predictions")
    holdout_payload = _prediction_payload(
        holdout_payload or {}, "winner_prediction_payload", "holdout_predictions", "cv_predictions"
    )
    oof_rows = _prediction_rows(oof_payload, selected_model or "phase2_winner", "development_oof_cv")
    holdout_rows = _prediction_rows(holdout_payload, selected_model or "phase2_winner", "locked_internal_holdout")
    oof_csv = audit_dir / "evaluation_predictions_oof.csv"
    holdout_csv = audit_dir / "evaluation_predictions_holdout.csv"
    _write_predictions_csv(oof_csv, oof_rows)
    _write_predictions_csv(holdout_csv, holdout_rows)

    oof_y = [int(row["y_true"]) for row in oof_rows]
    oof_score = [float(row["pred_probability"]) for row in oof_rows]
    holdout_y = [int(row["y_true"]) for row in holdout_rows]
    holdout_score = [float(row["pred_probability"]) for row in holdout_rows]
    protocol = {
        "schema_version": "metaboagent.evaluation_protocol.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("METABOAGENT_RUN_TAG", ""),
        "development_evaluation": {
            "protocol": oof_payload.get("evaluation_mode", "cross_validated_oof"),
            "k_folds": oof_payload.get("k_folds"),
            "prediction_scope": "out_of_fold",
            "preprocessing_fit_scope": "not_archived",
            "feature_selection_fit_scope": "not_archived",
            "hyperparameter_tuning_fit_scope": "not_archived",
        },
        "internal_holdout": {
            "available": bool(holdout_rows),
            "cohort_type": "internal_holdout",
            "external_validation_declared": False,
            "prediction_scope": "locked_internal_holdout",
            "lock_artifact": str(audit_dir / "pipeline_lock.json"),
        },
        "model": {
            "selected_model": selected_model,
            "selected_feature_count": len(list(selected_features or [])),
            "target_column": target_column,
        },
        "source_artifacts": {
            "phase2_winner_scores": describe_path(phase2_artifact_path),
            "phase2_external_validation": describe_path(holdout_artifact_path),
            "pipeline_lock": describe_path(audit_dir / "pipeline_lock.json"),
        },
        "interpretation": "Development performance is reported from out-of-fold predictions and an internal holdout; neither constitutes independent external validation.",
    }
    protocol_path = audit_dir / "evaluation_protocol.json"
    _write_json(protocol_path, protocol)

    performance = {
        "schema_version": "metaboagent.performance_uncertainty.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {
            "method": "stratified_percentile_bootstrap",
            "seed": int(bootstrap_seed),
            "n_resamples": int(bootstrap_resamples),
            "confidence_level": 0.95,
            "fit_repeated": False,
        },
        "development_oof": {
            "point_estimate": _metric_summary(oof_y, oof_score),
            "bootstrap_ci": _bootstrap_ci(oof_y, oof_score, seed=bootstrap_seed, n_resamples=bootstrap_resamples),
            "fold_metrics": _fold_metrics(oof_rows),
            "prediction_artifact": str(oof_csv),
        },
        "internal_holdout": {
            "point_estimate": _metric_summary(holdout_y, holdout_score),
            "bootstrap_ci": _bootstrap_ci(holdout_y, holdout_score, seed=bootstrap_seed + 1, n_resamples=bootstrap_resamples),
            "prediction_artifact": str(holdout_csv),
        },
        "policy": "Bootstrap intervals quantify prediction uncertainty conditional on archived predictions; they do not replace refitting the complete model-selection procedure.",
    }
    performance_path = audit_dir / "performance_uncertainty.json"
    _write_json(performance_path, performance)

    heldout_summary = _metric_summary(holdout_y, holdout_score)
    heldout_performance = {
        "schema_version": "metaboagent.heldout_performance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("METABOAGENT_RUN_TAG", ""),
        "status": "PASS" if heldout_summary.get("status") == "ASSESSED" else "PARTIAL",
        "evaluation_scope": "locked_internal_holdout",
        "cohort_type": "internal_holdout",
        "external_validation_declared": False,
        "target_column": target_column,
        "selected_model": selected_model,
        "selected_features": sorted(dict.fromkeys(str(feature) for feature in (selected_features or []))),
        "metrics": heldout_summary,
        "bootstrap_ci": performance["internal_holdout"]["bootstrap_ci"],
        "prediction_artifact": describe_path(holdout_csv),
        "source_artifact": describe_path(holdout_artifact_path),
        "lock_artifact": describe_path(audit_dir / "pipeline_lock.json"),
        "interpretation": (
            "Metrics are computed once from predictions made on the locked internal holdout. "
            "This is not independent external validation; no winner model is refit, and the calibration slope/intercept are descriptive diagnostics only."
        ),
    }
    heldout_path = audit_dir / "heldout_performance.json"
    _write_json(heldout_path, heldout_performance)
    return {
        "evaluation_protocol": str(protocol_path),
        "evaluation_predictions_oof": str(oof_csv),
        "evaluation_predictions_holdout": str(holdout_csv),
        "performance_uncertainty": str(performance_path),
        "heldout_performance": str(heldout_path),
    }
