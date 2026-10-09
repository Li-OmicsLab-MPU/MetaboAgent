"""Deterministic Phase 4 evidence sidecars.

These helpers only read archived Phase 0–3 artifacts and write report/audit
metadata. They do not fit preprocessing, select features, train models or
alter any upstream decision.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _d(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _l(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _s(value: Any) -> str:
    return "" if value is None else str(value)


def _f(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _first_value(payload: Any, keys: set[str]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in keys and value not in (None, ""):
                return value
            found = _first_value(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_value(value, keys)
            if found not in (None, ""):
                return found
    return None


def _sha(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(root: Path, value: Any) -> Path:
    raw = _s(value).strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _candidate_paths(root: Path, *values: Any) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        path = _resolve(root, value)
        if path and path.exists() and path.is_file() and path not in paths:
            paths.append(path)
    return paths


def _phase1_paths(root: Path, context: Dict[str, Any], collected: Dict[str, Any]) -> Dict[str, Path]:
    p1 = _d(context.get("phase1"))
    cp1 = _d(collected.get("phase1"))
    panel = _d(cp1.get("panel_scores"))
    candidates = [
        panel.get("training_data_path"),
        panel.get("evaluation_data_path"),
        p1.get("stability_summary_path"),
        p1.get("stability_scores_path"),
    ]
    selected = [
        # Run-registry reruns keep the canonical Phase 1 tree directly under
        # the run root rather than under output/.  Prefer these paths before
        # consulting legacy/global mirrors.
        root / "phase1/final/selected_features_final.csv",
        root / "phase1/final/selected_features_train_pool.csv",
        root / "output/phase1/final/selected_features_final.csv",
        root / "output/phase1/intermediate/latest/selected_features_final.csv",
        root / "output/phase1/artifacts/selected_features_final.csv",
        root / "phase1/legacy/artifacts/selected_features_final.csv",
    ]
    # Explicit partition artifacts are preferred for the split audit.  The
    # legacy panel-score registry often points both training and evaluation to
    # selected_features_final.csv, which is a derived panel table rather than
    # a proof that the two partitions were distinct.
    partitioned = [
        root / "phase1/final/selected_features_train_pool.csv",
        root / "phase1/final/selected_features_holdout_pool.csv",
        root / "phase1/intermediate/latest/engineered/train_only_bundle/train_data_with_engineered_features_unscaled.csv",
        root / "phase1/intermediate/latest/engineered/train_only_bundle/holdout_data_with_engineered_features_unscaled.csv",
        root / "output/phase1/final/selected_features_train_pool.csv",
        root / "output/phase1/final/selected_features_holdout_pool.csv",
    ]
    stability = [
        root / "phase1/intermediate/latest/feature_selection/stability_selection_summary.json",
        root / "phase1/intermediate/latest/feature_selection/stability_scores.json",
        root / "output/phase1/intermediate/latest/feature_selection/stability_selection_summary.json",
        root / "output/phase1/intermediate/latest/feature_selection/stability_scores.json",
    ]
    paths = _candidate_paths(root, *candidates)
    result: Dict[str, Path] = {}
    for path in paths + selected + partitioned + stability:
        if path.exists() and path.is_file():
            name = path.name
            if "stability_selection_summary" in name:
                result.setdefault("stability_summary", path)
            elif name == "stability_scores.json":
                result.setdefault("stability_scores", path)
            elif name.endswith(".csv"):
                if name == "selected_features_train_pool.csv":
                    result.setdefault("train_pool_csv", path)
                    result.setdefault("training_csv", path)
                elif name == "selected_features_holdout_pool.csv":
                    result.setdefault("holdout_pool_csv", path)
                    result.setdefault("holdout_csv", path)
                elif "holdout" in name.lower():
                    result.setdefault("holdout_csv", path)
                else:
                    result.setdefault("training_csv", path)
    # Historical registry runs may retain the holdout table only under the
    # run-local artifact registry.  Search strictly inside this run root and
    # prefer the explicit holdout filename; this does not cross run boundaries.
    for path in sorted(root.rglob("selected_features_holdout.csv")):
        if path.is_file():
            result.setdefault("holdout_csv", path)
            break
    return result


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            return fields, [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error):
        return [], []


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
        return value not in (None, "")
    except (TypeError, ValueError):
        return False


def _group_column(fields: Iterable[str], context: Dict[str, Any]) -> str:
    configured = _s(_d(context.get("run")).get("target_column")).strip()
    candidates = [configured, "group", "label", "target", "y", "class", "disease", "outcome", "condition"]
    lowered = {field.lower(): field for field in fields}
    for candidate in candidates:
        if candidate and candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return ""


def _csv_profile(path: Path, context: Dict[str, Any], role: str) -> Dict[str, Any]:
    fields, rows = _read_csv(path)
    group = _group_column(fields, context)
    feature_fields = [field for field in fields if field != group and field.lower() not in {"sample_id", "id", "index"}]
    missing = 0
    numeric_features = 0
    zero_values = 0
    total_cells = len(rows) * len(feature_fields)
    for field in feature_fields:
        values = [row.get(field, "") for row in rows]
        if any(_is_numeric(value) for value in values):
            numeric_features += 1
        for value in values:
            if value in (None, "", "NA", "N/A", "nan", "NaN", "null", "None"):
                missing += 1
            elif _f(value) == 0:
                zero_values += 1
    labels: Dict[str, int] = {}
    if group:
        for row in rows:
            label = _s(row.get(group)).strip()
            labels[label] = labels.get(label, 0) + 1
    return {
        "status": "ASSESSED" if rows and fields else "NOT_ASSESSED",
        "artifact_role": role,
        "source_file": str(path),
        "sha256": _sha(path),
        "n_samples": len(rows),
        "n_columns": len(fields),
        "n_features": len(feature_fields),
        "n_numeric_features": numeric_features,
        "group_column": group or "not inferred",
        "class_counts": labels,
        "missing_cells": missing,
        "missing_cell_fraction": (missing / total_cells) if total_cells else None,
        "zero_cells": zero_values,
        "zero_cell_fraction": (zero_values / total_cells) if total_cells else None,
        "fit_scope": "descriptive artifact profile; no fitting",
    }


def build_cohort_qc_snapshot(root: Path, context: Dict[str, Any], collected: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    profiles = []
    for key, role in (("training_csv", "training/modeling artifact"), ("holdout_csv", "internal holdout artifact")):
        path = paths.get(key)
        if path:
            profiles.append(_csv_profile(path, context, role))
        else:
            profiles.append({
                "status": "NOT_ASSESSED",
                "artifact_role": role,
                "source_file": "not archived",
                "n_samples": None,
                "n_columns": None,
                "n_features": None,
                "n_numeric_features": None,
                "group_column": "not inferred",
                "class_counts": {},
                "missing_cells": None,
                "missing_cell_fraction": None,
                "zero_cells": None,
                "zero_cell_fraction": None,
                "fit_scope": "no tabular artifact available",
            })
    preprocessing = _d(_d(context.get("phase1")).get("preprocessing_summary"))
    return {
        "schema_version": "phase4.cohort_qc_snapshot.v1",
        "status": "ASSESSED" if any(item.get("status") == "ASSESSED" for item in profiles) else "NOT_ASSESSED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
        "preprocessing_summary": {
            "missingness_detection": _d(preprocessing.get("missingness_detection")),
            "imputation": _d(preprocessing.get("imputation")),
            "normalization_scaling": _d(preprocessing.get("normalization_scaling")),
            "qc_batch": _d(preprocessing.get("qc_batch")),
        },
        "data_quality_snapshot_archived": _d(preprocessing.get("data_quality_snapshot")),
        "policy": "Counts and descriptive QC do not feed Phase 1–3 decisions; fitted preprocessing statistics must remain train-only.",
    }


def build_data_split_audit(root: Path, context: Dict[str, Any], collected: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    """Describe split provenance without asserting leakage safety beyond evidence."""

    panel = _d(_d(collected.get("phase1")).get("panel_scores"))
    training_value = _s(panel.get("training_data_path"))
    evaluation_value = _s(panel.get("evaluation_data_path"))
    training_path = _resolve(root, training_value)
    evaluation_path = _resolve(root, evaluation_value)
    holdout_path = paths.get("holdout_csv", Path(""))
    oof_prediction_path = root / "output" / "audit" / "evaluation_predictions_oof.csv"
    holdout_prediction_path = root / "output" / "audit" / "evaluation_predictions_holdout.csv"
    # Count and compare the archived partition tables themselves.  Prediction
    # exports may reset row indices independently and are therefore not a safe
    # source for partition identity checks.
    training_partition_path = paths.get("train_pool_csv") or training_path
    holdout_partition_path = paths.get("holdout_pool_csv") or holdout_path
    training_count_path = training_partition_path
    holdout_count_path = holdout_partition_path
    training_exists = bool(training_value) and training_path.is_file()
    evaluation_exists = bool(evaluation_value) and evaluation_path.is_file()
    holdout_exists = bool(holdout_path) and holdout_path.is_file()

    def ids(path: Path) -> set[str]:
        if not path or not path.exists():
            return set()
        _, rows = _read_csv(path)
        for key in ("sample_id", "subject_id", "patient_id", "id"):
            values = {str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()}
            if values:
                return values
        return set()

    train_ids = ids(training_count_path)
    holdout_ids = ids(holdout_count_path)
    # Some legacy prediction exports reset row indices independently within
    # each partition (for example 0..349 and 0..87).  Treating those labels as
    # a shared subject namespace would manufacture an apparent overlap.  Keep
    # the counts, but mark the overlap as unassessable until a stable subject
    # identifier is archived in both files.
    local_index_like = bool(train_ids and holdout_ids) and (
        train_ids == {str(i) for i in range(len(train_ids))}
        and holdout_ids == {str(i) for i in range(len(holdout_ids))}
    )
    overlap = None if local_index_like else sorted(train_ids & holdout_ids)
    protocol = _json(root / "output/audit/evaluation_protocol.json")
    development_evaluation = _d(protocol.get("development_evaluation"))
    split_artifact = _json(root / "output/audit/data_split_audit.json")
    preprocessing_audit = _json(root / "output/audit/preprocessing_audit.json")
    feature_selection_audit = _json(root / "output/audit/feature_selection_audit.json")
    model_selection_audit = _json(root / "output/audit/model_selection_audit.json")

    # Prefer an explicit split-order record, then use the Phase 1 model
    # artifact's declared strategy. This is provenance, not proof of a
    # subject-level split.
    split_order = _s(
        split_artifact.get("split_order")
        or split_artifact.get("partition_order")
        or _d(context.get("run")).get("split_order")
    )
    split_order_evidence = "output/audit/data_split_audit.json"
    if not split_order or split_order.lower() in {"not archived", "not_archived"}:
        phase1_modeling = _json(root / "phase1/artifacts/model_results/modeling_report.json")
        split_order = _s(phase1_modeling.get("split_strategy"))
        if split_order:
            split_order_evidence = "phase1/artifacts/model_results/modeling_report.json"
    if split_order == "internal_train_test_split":
        # The presence of separate train/holdout pre-engineering bundles is
        # the archived evidence that the split preceded engineering.
        split_order = "stratified_train_test_split_before_engineering"
        split_order_evidence = "phase1/artifacts/model_results/modeling_report.json + phase1/intermediate/latest/engineered/train_only_bundle"
    split_order = split_order or "not archived"

    preprocessing_ops = [
        _d(item) for item in _l(preprocessing_audit.get("operations"))
        if _s(_d(item).get("fit_scope"))
    ]
    preprocessing_fit_scope = _s(development_evaluation.get("preprocessing_fit_scope"))
    preprocessing_fit_evidence = "output/audit/evaluation_protocol.json"
    if preprocessing_ops and all(_s(item.get("fit_scope")).lower() == "train_only" for item in preprocessing_ops):
        preprocessing_fit_scope = "train_only"
        preprocessing_fit_evidence = "output/audit/preprocessing_audit.json"

    feature_selection_fit_scope = _s(development_evaluation.get("feature_selection_fit_scope"))
    feature_selection_fit_evidence = "output/audit/evaluation_protocol.json"
    policy = _s(feature_selection_audit.get("policy"))
    if policy and "train_only" in policy.lower() and _s(feature_selection_audit.get("status")).upper() == "PASS":
        feature_selection_fit_scope = "train_only (policy recorded)"
        feature_selection_fit_evidence = "output/audit/feature_selection_audit.json"
    # Older runs may have a NOT_ASSESSED feature_selection_audit even though
    # the Phase 1 selection chain archives its train/holdout inputs explicitly.
    # Promote that concrete provenance to an auditable scope without inferring
    # nested-CV validity.
    if not feature_selection_fit_scope or feature_selection_fit_scope.lower() in {"not_archived", "not assessed", "not_assessed"}:
        selection_summary = _json(root / "phase1/intermediate/latest/feature_selection/final_selection_summary.json")
        selection_train = _resolve(root, selection_summary.get("input_train_pool_path"))
        selection_holdout = _resolve(root, selection_summary.get("input_holdout_pool_path"))
        if selection_train.is_file() and selection_holdout.is_file():
            feature_selection_fit_scope = "train_only_artifact_chain"
            feature_selection_fit_evidence = "phase1/intermediate/latest/feature_selection/final_selection_summary.json"

    hyperparameter_tuning_fit_scope = _s(development_evaluation.get("hyperparameter_tuning_fit_scope"))
    hyperparameter_tuning_fit_evidence = "output/audit/evaluation_protocol.json"
    selection_scope = _s(model_selection_audit.get("selection_scope"))
    if selection_scope:
        hyperparameter_tuning_fit_scope = selection_scope
        hyperparameter_tuning_fit_evidence = "output/audit/model_selection_audit.json"

    subject_level = _s(split_artifact.get("subject_level_split") or split_artifact.get("group_split") or "not archived")
    stable_ids = bool(train_ids and holdout_ids and len(train_ids) == len(set(train_ids)) and len(holdout_ids) == len(set(holdout_ids)))
    no_overlap = overlap == []
    partition_evidence_complete = bool(
        training_partition_path.is_file()
        and holdout_partition_path.is_file()
        and split_order != "not archived"
        and stable_ids
        and no_overlap
        and preprocessing_fit_scope.lower() == "train_only"
        and feature_selection_fit_scope.lower() not in {"not_archived", "not assessed", "not_assessed"}
        and hyperparameter_tuning_fit_scope
    )
    status = "PASS" if partition_evidence_complete else ("ASSESSED" if training_exists and holdout_exists else "PARTIAL")
    return {
        "schema_version": "metaboagent.evaluation_audit.v2",
        "status": status,
        "training_artifact": str(training_partition_path) if training_partition_path else "not archived",
        "evaluation_artifact": str(holdout_partition_path) if holdout_partition_path else "not archived",
        "internal_holdout_artifact": str(holdout_partition_path) if holdout_partition_path else "not archived",
        "same_training_evaluation_path": bool(training_partition_path and holdout_partition_path and os.path.abspath(training_partition_path) == os.path.abspath(holdout_partition_path)),
        "same_training_evaluation_sha256": bool(training_partition_path.is_file() and holdout_partition_path.is_file() and _sha(training_partition_path) == _sha(holdout_partition_path)),
        "legacy_registry_training_artifact": str(training_path) if training_path else "not archived",
        "legacy_registry_evaluation_artifact": str(evaluation_path) if evaluation_path else "not archived",
        "development_sample_count": len(train_ids) or None,
        "internal_holdout_sample_count": len(holdout_ids) or None,
        "sample_id_overlap": overlap,
        "sample_id_overlap_status": (
            "UNASSESSABLE_PARTITION_LOCAL_IDS" if local_index_like
            else "ASSESSED_STABLE_ID_NAMESPACE"
        ),
        "partition_id_column": "id" if train_ids or holdout_ids else "not archived",
        "partition_id_unique": stable_ids,
        "partition_id_overlap_count": len(overlap) if overlap is not None else None,
        "split_order": split_order,
        "split_order_evidence": split_order_evidence,
        "subject_level_split": subject_level,
        "preprocessing_fit_scope": preprocessing_fit_scope or "not_archived",
        "preprocessing_fit_scope_evidence": preprocessing_fit_evidence,
        "feature_selection_fit_scope": feature_selection_fit_scope or "not_archived",
        "feature_selection_fit_scope_evidence": feature_selection_fit_evidence,
        "hyperparameter_tuning_fit_scope": hyperparameter_tuning_fit_scope or "not_archived",
        "hyperparameter_tuning_fit_scope_evidence": hyperparameter_tuning_fit_evidence,
        "status_reason": (
            "PASS: distinct train/holdout pool artifacts, stable unique IDs with zero overlap, split-before-engineering evidence, "
            "and train-only preprocessing plus recorded selection/tuning scopes are archived."
            if partition_evidence_complete else
            "ASSESSED/PARTIAL: one or more partition provenance fields remain unavailable."
        ),
        "policy": "PASS denotes partition provenance and artifact-level leakage controls. It does not prove that every preprocessing or selection operation was refit within every cross-validation fold; that remains a separate methodological limitation.",
    }


def build_feature_selection_evidence(root: Path, context: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    summary = _json(paths["stability_summary"]) if paths.get("stability_summary") else {}
    scores = _l(summary.get("stability_scores"))
    if not scores and paths.get("stability_scores"):
        scores = _l(_json(paths["stability_scores"]).get("stability_scores"))
    phase2 = _d(context.get("phase2"))
    winners = [_s(item) for item in _l(phase2.get("winner_features"))]
    by_feature = {_s(_d(item).get("feature")): _d(item) for item in scores}
    records = []
    for feature in winners:
        score = by_feature.get(feature, {})
        records.append({
            "feature": feature,
            "phase1_selection_frequency": _f(score.get("selection_frequency")),
            "method_consensus": score.get("method_consensus"),
            "stable_core_member": feature in {_s(item) for item in _l(summary.get("stable_core_features"))},
            "phase2_winner": True,
            "selection_stage": "Phase 1 train-only stability selection -> Phase 2 multi-objective winner",
        })
    return {
        "schema_version": "phase4.feature_selection_evidence.v1",
        "status": "ASSESSED" if summary and scores else "NOT_ASSESSED",
        "source": str(paths.get("stability_summary", "")),
        "iterations": summary.get("iterations"),
        "selector_family": _l(summary.get("selector_family")),
        "n_methods": summary.get("n_methods"),
        "n_features_per_method": summary.get("n_features_per_method"),
        "thresholds": _d(summary.get("thresholds")),
        "stable_core_features": _l(summary.get("stable_core_features")),
        "phase1_final_panel_count": summary.get("final_panel_count"),
        "phase2_winner_count": len(winners),
        "winner_records": records,
        "policy": "Selection frequency describes the Phase 1 train-only candidate-selection stage, not end-to-end final-model stability.",
    }


def _association_rows(path: Path, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields, rows = _read_csv(path)
    group = _group_column(fields, context)
    if not group:
        return []
    labels = [_s(row.get(group)).strip() for row in rows]
    unique = list(dict.fromkeys(label for label in labels if label != ""))
    if len(unique) != 2:
        return []
    control, case = unique[0], unique[1]
    winners = [_s(item) for item in _l(_d(context.get("phase2")).get("winner_features"))]
    rng = random.Random(42)
    result = []
    for feature in winners:
        values = {group_value: [_f(row.get(feature)) for row, label in zip(rows, labels) if label == group_value and _f(row.get(feature)) is not None] for group_value in (control, case)}
        x0, x1 = values[control], values[case]
        if len(x0) < 2 or len(x1) < 2:
            result.append({"feature": feature, "status": "NOT_ASSESSED", "fit_scope": "train_only"})
            continue
        mean0, mean1 = sum(x0) / len(x0), sum(x1) / len(x1)
        var0 = sum((x - mean0) ** 2 for x in x0) / (len(x0) - 1)
        var1 = sum((x - mean1) ** 2 for x in x1) / (len(x1) - 1)
        df = len(x0) + len(x1) - 2
        pooled = (((len(x0) - 1) * var0) + ((len(x1) - 1) * var1)) / df if df else 0
        d = (mean1 - mean0) / (pooled ** 0.5) if pooled > 0 else None
        correction = 1 - (3 / (4 * df - 1)) if df > 1 else None
        g = d * correction if d is not None and correction is not None else None
        boot = []
        for _ in range(200):
            b0 = [x0[rng.randrange(len(x0))] for _ in x0]
            b1 = [x1[rng.randrange(len(x1))] for _ in x1]
            m0, m1 = sum(b0) / len(b0), sum(b1) / len(b1)
            boot.append(m1 - m0)
        boot.sort()
        result.append({
            "feature": feature,
            "status": "ASSESSED",
            "group_control": control,
            "group_case": case,
            "n_control": len(x0),
            "n_case": len(x1),
            "standardized_mean_difference_hedges_g": g,
            "direction_case_minus_control": "higher" if mean1 > mean0 else "lower" if mean1 < mean0 else "equal",
            "raw_mean_difference": mean1 - mean0,
            "bootstrap_raw_difference_ci95": [boot[max(0, int(0.025 * len(boot)) - 1)], boot[min(len(boot) - 1, int(0.975 * len(boot)))]] if boot else [],
            "fit_scope": "train_only",
            "interpretation": "Descriptive association, not a causal effect.",
        })
    return result


def build_model_card(context: Dict[str, Any], bundle: Dict[str, Any], collected: Dict[str, Any]) -> Dict[str, Any]:
    phase2 = _d(context.get("phase2"))
    cp2 = _d(collected.get("phase2"))
    result = _d(cp2.get("result"))
    search = _d(phase2.get("search_summary"))
    validation = _d(bundle.get("validation_scope"))
    return {
        "schema_version": "phase4.model_card.v1",
        "status": "PARTIAL",
        "intended_use": "Research-use-only biomarker discovery and scenario-specific risk stratification.",
        "target": _s(_d(context.get("phase0")).get("disease_name")),
        "winner_features": _l(phase2.get("winner_features")),
        "feature_counts": {"phase1_candidate_pool": phase2.get("phase1_candidate_pool_count"), "phase2_winner": len(_l(phase2.get("winner_features")))},
        "selected_model": phase2.get("selected_model") or search.get("selected_model") or result.get("model_name") or "not archived",
        "hyperparameters": _d(result.get("best_params") or phase2.get("best_params")),
        "optimization_summary": search,
        "validation": {"evidence_status": validation.get("evidence_status_label") or validation.get("evidence_status"), "performance_protocol": validation.get("performance_protocol"), "internal_holdout_available": validation.get("holdout_available"), "external_declared": validation.get("external_declared")},
        "clinical_utility": _d(phase2.get("clinical_utility")),
        "limitations": ["No independent external validation is declared.", "Deployment requires analytical validation, recalibration where needed and prospective evaluation."],
        "policy": "Fields absent from Phase 0–3 archives are marked not archived; Phase 4 does not infer model settings.",
    }


def build_visual_evidence_registry(context: Dict[str, Any]) -> Dict[str, Any]:
    figures = []
    for value in _l(_d(context.get("phase3")).get("figures")):
        item = _d(value)
        title = _s(item.get("title") or item.get("figure_id"))
        low = title.lower()
        scope = "INTERNAL-HOLDOUT" if "holdout" in low else "INTERNAL-CV"
        layer = "performance"
        if "shap" in low:
            layer = "model-attribution"
        elif "spline" in low or "rcs" in low:
            layer = "feature-association"
        elif "calibr" in low:
            layer = "calibration"
        elif "decision" in low or "dca" in low:
            layer = "clinical-utility"
        figures.append({"figure_id": _s(item.get("figure_id")), "title": title, "layer": layer, "cohort_scope": scope, "fit_scope": "not archived", "prediction_source": "not archived", "interpretation_boundary": "Descriptive model behaviour or within-cohort association; not causal evidence."})
    return {"schema_version": "phase4.visual_evidence_registry.v1", "status": "ASSESSED" if figures else "NOT_ASSESSED", "figures": figures, "policy": "Each visualization must expose analysis layer, cohort scope and interpretation boundary."}


def build_reproducibility_manifest(root: Path, context: Dict[str, Any], bundle: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    run = _d(context.get("run"))
    audit_manifest_path = next((candidate for candidate in (root / "output/audit/run_manifest.json", root / "audit/run_manifest.json") if candidate.exists()), root / "output/audit/run_manifest.json")
    audit_manifest = _json(audit_manifest_path)
    environment_manifest_path = next((candidate for candidate in (root / "output/audit/environment_manifest.json", root / "audit/environment_manifest.json") if candidate.exists()), root / "output/audit/environment_manifest.json")
    environment_manifest = _json(environment_manifest_path)
    config_value = _d(audit_manifest.get("config")).get("path")
    config_path = _resolve(root, config_value) if config_value else Path("")
    project_root_candidates = [
        _resolve(root, os.environ.get("METABOAGENT_PROJECT_ROOT", "")),
        _resolve(root, _d(audit_manifest.get("project_root")).get("path")),
        config_path.parent if config_path else Path(""),
        root,
    ]
    git_commit = "not archived"
    git_dirty: Any = "not archived"
    for candidate in project_root_candidates:
        if not candidate or not candidate.exists():
            continue
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=candidate, capture_output=True, text=True, timeout=5).stdout.strip()
            if commit:
                git_commit = commit
                git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=candidate, capture_output=True, text=True, timeout=5).stdout.strip())
                break
        except (OSError, subprocess.SubprocessError):
            continue
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "matplotlib", "reportlab"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed/not archived"
    artifact_list = []
    for path in [config_path, paths.get("stability_summary"), paths.get("stability_scores"), audit_manifest_path, environment_manifest_path]:
        if path and path.is_file():
            artifact_list.append({"path": str(path), "sha256": _sha(path)})
    seed_payload = {"audit": audit_manifest, "environment": environment_manifest, "context": context, "run": run}
    random_seeds = {
        "environment_random_state": _first_value(seed_payload, {"environment_random_state", "random_state", "random_seed", "seed"}) or "not archived",
        "phase1_stability_seed": _first_value(seed_payload, {"phase1_stability_seed", "stability_seed", "stability_random_state"}) or "not archived",
        "phase2_search_seed": _first_value(seed_payload, {"phase2_search_seed", "search_seed", "search_random_state"}) or "not archived",
    }


def _feature_key(value: Any) -> str:
    """Return a formatting-insensitive feature key used only for identity joins."""
    return "".join(character.lower() for character in _s(value) if character.isalnum())


def _annotation_records(root: Optional[Path], context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Load an explicitly supplied analytical annotation registry, if present.

    HMDB/name mapping is not sufficient to infer MSI confidence.  This loader
    therefore accepts only a run-scoped registry produced by the analytical
    identification stage and leaves all absent fields unarchived.
    """
    candidates: List[Path] = []
    configured = _d(_d(context.get("phase1")).get("annotation_registry"))
    for value in (configured.get("path"), configured.get("file"), _d(context.get("sources")).get("metabolite_annotation")):
        if value:
            candidates.append(_resolve(root or Path("."), value))
    if root:
        candidates.extend([
            root / "output/audit/metabolite_annotation_registry.json",
            root / "output/phase1/metabolite_annotation_registry.json",
            root / "phase1/intermediate/latest/engineered/metabolite_annotation_registry.json",
        ])
    payload: Any = {}
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            payload = _json(candidate)
            if payload:
                break
    raw = payload.get("records") if isinstance(payload, dict) and isinstance(payload.get("records"), list) else payload
    records: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        raw = [{"feature": key, **(_d(value))} for key, value in raw.items()]
    for value in _l(raw):
        item = _d(value)
        for key in ("feature", "feature_token", "canonical_id", "hmdb_id", "name", "metabolite"):
            token = _s(item.get(key)).strip()
            if token:
                records[_feature_key(token)] = item
    return records


def build_feature_identity_registry(context: Dict[str, Any], bundle: Dict[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    """Create one run-scoped identity authority for tables, narrative and figures."""
    phase2 = _d(context.get("phase2"))
    dictionary = _l(phase2.get("winner_feature_dictionary"))
    if not dictionary:
        dictionary = [{"feature_token": item} for item in _l(phase2.get("winner_features"))]
    records: List[Dict[str, Any]] = []
    alias_index: Dict[str, str] = {}
    annotations = _annotation_records(root, context)
    for raw in dictionary:
        item = _d(raw)
        token = _s(item.get("feature_token") or item.get("raw_column_name"))
        display = _s(
            item.get("preferred_report_name")
            or item.get("report_label")
            or item.get("display_name")
            or token
        )
        aliases = list(dict.fromkeys(
            value for value in [
                token,
                display,
                _s(item.get("report_label")),
                _s(item.get("display_name")),
                *[_s(value) for value in _l(item.get("mapped_metabolites"))],
            ] if value
        ))
        record = {
            "canonical_id": token,
            "canonical_name": display,
            "raw_name": token,
            "aliases": aliases,
            "origin_type": item.get("origin_type") or ("engineered" if item.get("is_engineered") else "raw"),
            "origin_subtype": item.get("origin_subtype") or "not archived",
            "hmdb_ids": _l(item.get("hmdb_ids")),
            "msi_identification_level": item.get("msi_identification_level") or item.get("identification_level") or "not archived",
            "mz": item.get("mz") or item.get("m_z") or "not archived",
            "retention_time": item.get("retention_time") or item.get("rt") or "not archived",
            "adduct": item.get("adduct") or "not archived",
            "ion_mode": item.get("ion_mode") or item.get("polarity") or "not archived",
            "unit": item.get("unit") or item.get("measurement_unit") or "not archived",
            "prior_supported": bool(item.get("prior_supported")),
            "mapped_pathways": _l(item.get("mapped_pathways")),
        }
        annotation = next((annotations.get(_feature_key(alias)) for alias in aliases if _feature_key(alias) in annotations), {})
        for field, keys in {
            "msi_identification_level": ("msi_identification_level", "identification_level", "msi_level"),
            "mz": ("mz", "m_z"),
            "retention_time": ("retention_time", "rt"),
            "adduct": ("adduct",),
            "ion_mode": ("ion_mode", "polarity"),
            "unit": ("unit", "measurement_unit"),
        }.items():
            if record[field] == "not archived":
                for key in keys:
                    if annotation.get(key) not in (None, ""):
                        record[field] = annotation[key]
                        break
        records.append(record)
        for alias in aliases:
            alias_index[_feature_key(alias)] = token

    unresolved: List[str] = []
    mechanistic = _d(bundle.get("mechanistic_evidence_registry"))
    for entry_value in _l(mechanistic.get("entries")):
        entry = _d(entry_value)
        name = _s(entry.get("feature_token") or entry.get("feature"))
        if name and _feature_key(name) not in alias_index:
            unresolved.append(name)
    return {
        "schema_version": "phase4.feature_identity_registry.v1",
        "status": "PASS" if records and not unresolved else "PARTIAL" if records else "NOT_ASSESSED",
        "records": records,
        "alias_index": alias_index,
        "unresolved_reader_features": sorted(set(unresolved)),
        "policy": "All reader-facing feature names must resolve through canonical_id; absent analytical annotation remains 'not archived' and is never inferred.",
    }


def build_statistical_claim_validation(context: Dict[str, Any]) -> Dict[str, Any]:
    """Validate deterministic numerical semantics before LLM prose is released."""
    executive = _d(_d(context.get("report_inputs")).get("executive_summary"))
    clinical = _d(_d(context.get("phase2")).get("clinical_utility"))
    checks: List[Dict[str, Any]] = []
    for metric_name in ("nri", "idi"):
        metric = _d(executive.get(metric_name))
        estimate = _f(metric.get("value"))
        interval = _l(metric.get("ci_95"))
        low = _f(interval[0]) if len(interval) == 2 else None
        high = _f(interval[1]) if len(interval) == 2 else None
        available = estimate is not None and low is not None and high is not None
        checks.append({
            "check_id": f"{metric_name}_ci_semantics",
            "metric": metric_name.upper(),
            "status": "PASS" if available and low <= estimate <= high else "NOT_ASSESSED" if not available else "FAIL",
            "estimate": estimate,
            "ci_95": [low, high] if available else [],
            "ci_includes_zero": (low <= 0 <= high) if available else None,
            "deterministic_interpretation": (
                "CI includes zero; the estimate does not establish a non-zero incremental effect."
                if available and low <= 0 <= high else
                "CI excludes zero; report this result separately from metrics whose interval includes zero."
                if available else "Not assessable from archived evidence."
            ),
        })
    for index, row_value in enumerate(_l(clinical.get("threshold_metrics_table"))):
        row = _d(row_value)
        tp, fp, tn, fn = (_f(row.get(key)) for key in ("tp", "fp", "tn", "fn"))
        if None in (tp, fp, tn, fn):
            continue
        expected = {
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "npv": tn / (tn + fn) if tn + fn else None,
        }
        discrepancies = {
            key: abs(value - _f(row.get(key)))
            for key, value in expected.items()
            if value is not None and _f(row.get(key)) is not None and abs(value - _f(row.get(key))) > 1e-9
        }
        checks.append({
            "check_id": f"threshold_confusion_consistency_{index}",
            "threshold": _f(row.get("threshold")),
            "status": "FAIL" if discrepancies else "PASS",
            "discrepancies": discrepancies,
        })
    failures = [item["check_id"] for item in checks if item.get("status") == "FAIL"]
    return {
        "schema_version": "phase4.statistical_claim_validation.v1",
        "status": "FAIL" if failures else "PASS" if checks else "NOT_ASSESSED",
        "checks": checks,
        "failed_checks": failures,
        "policy": "LLM prose may improve readability but cannot alter CI, null-reference, or confusion-matrix semantics.",
    }


def build_prevalence_scenarios(context: Dict[str, Any]) -> Dict[str, Any]:
    clinical = _d(_d(context.get("phase2")).get("clinical_utility"))
    thresholds = [_d(item) for item in _l(clinical.get("threshold_metrics_table"))]
    prevalence_grid = [0.005, 0.01, 0.02, 0.05, 0.10]
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        sensitivity = _f(threshold.get("sensitivity"))
        specificity = _f(threshold.get("specificity"))
        if sensitivity is None or specificity is None:
            continue
        for prevalence in prevalence_grid:
            positive = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
            negative = specificity * (1 - prevalence) + (1 - sensitivity) * prevalence
            rows.append({
                "threshold": _f(threshold.get("threshold")),
                "assumed_prevalence": prevalence,
                "sensitivity_source": "archived case-control estimate",
                "specificity_source": "archived case-control estimate",
                "projected_ppv": sensitivity * prevalence / positive if positive else None,
                "projected_npv": specificity * (1 - prevalence) / negative if negative else None,
                "projected_positive_tests_per_1000": positive * 1000,
                "projected_false_positives_per_1000": (1 - specificity) * (1 - prevalence) * 1000,
            })
    return {
        "schema_version": "phase4.prevalence_scenarios.v1",
        "status": "ASSESSED" if rows else "NOT_ASSESSED",
        "prevalence_grid": prevalence_grid,
        "rows": rows,
        "policy": "Deterministic scenario projection only. It assumes sensitivity and specificity transport unchanged and is not observed or externally validated performance.",
    }


def build_evidence_maturity(context: Dict[str, Any], bundle: Dict[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    validation = _d(bundle.get("validation_scope"))
    level = "I / III"
    label = "Internal resampling only"
    if validation.get("external_declared"):
        level, label = "III / III", "Independent external evaluation declared"
    elif validation.get("holdout_available"):
        level, label = "II / III", "Locked internal holdout available"
    feature_identity = build_feature_identity_registry(context, bundle, root)
    annotation_complete = bool(feature_identity.get("records")) and all(
        _d(item).get("msi_identification_level") != "not archived"
        and _d(item).get("unit") != "not archived"
        for item in _l(feature_identity.get("records"))
    )
    quality = _d(bundle.get("report_quality"))
    quality_checks = _d(quality.get("checks"))
    audit_summary = _d(bundle.get("audit_summary"))
    audit_not_pass = [
        _s(_d(item).get("domain"))
        for item in _l(audit_summary.get("domains"))
        if _s(_d(item).get("status")).upper() not in {"PASS", "NOT_APPLICABLE", "NOT_REQUESTED"}
    ]
    partial_claims = _l(quality_checks.get("claim_ids_with_partial_evidence"))
    integrity_reasons = []
    if audit_not_pass:
        integrity_reasons.append(f"{len(audit_not_pass)} audit domain(s) are not fully assessed")
    if partial_claims:
        integrity_reasons.append(f"{len(partial_claims)} claim(s) have point-estimate-only evidence")
    if not integrity_reasons:
        integrity_reasons.append("all declared artifact and claim checks are complete")
    artifact_integrity = quality.get("status", "PARTIAL")
    return {
        "schema_version": "phase4.evidence_maturity.v1",
        "status": "ASSESSED",
        "validation_level": level,
        "validation_label": label,
        "dimensions": {
            "artifact_integrity": artifact_integrity,
            "artifact_integrity_note": (
                "Partial because " + "; ".join(integrity_reasons) + "."
                if str(artifact_integrity).upper() == "PARTIAL"
                else "Deterministic artifact and contract checks; not a scientific-quality verdict."
            ),
            "analytical_annotation": "COMPLETE" if annotation_complete else "PARTIAL",
            "reproducibility": "PARTIAL",
            "clinical_readiness": "EXPLORATORY",
        },
        "policy": "Evidence maturity describes validation scope; artifact integrity is reported separately and is not a scientific-quality score.",
    }


def build_reporting_readiness(context: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Conservative report-package readiness, not a formal checklist appraisal."""
    profile = _d(bundle.get("dataset_profile"))
    validation = _d(bundle.get("validation_scope"))
    metrics = _d(bundle.get("metric_registry"))
    split = _d(_d(bundle.get("enhancements")).get("data_split_audit"))
    probes = [
        ("participants_data_source", bool(profile.get("matched"))),
        ("predictors_final_panel", bool(_l(_d(context.get("phase2")).get("winner_features")))),
        ("development_validation_scope", bool(validation.get("headline_scope"))),
        ("train_holdout_independence", split.get("status") == "PASS"),
        ("discrimination", _f(_d(metrics.get("development_cv")).get("auc")) is not None),
        ("calibration", _f(_d(metrics.get("calibration")).get("brier_score")) is not None),
        ("external_evaluation", bool(validation.get("external_declared"))),
        ("analytical_identity_metadata", False),
    ]
    items = [{"item": key, "status": "PRESENT" if present else "MISSING"} for key, present in probes]
    return {
        "schema_version": "phase4.reporting_readiness.v1",
        "status": "PARTIAL",
        "tripod_ai_report_package": {"present": sum(value for _, value in probes), "assessed": len(probes), "items": items},
        "probast_ai_domain_prompts": {
            "participants_and_data_sources": "Needs expert assessment",
            "predictors": "Needs expert assessment",
            "outcome": "Needs expert assessment",
            "analysis": "Needs expert assessment",
        },
        "policy": "Internal readiness screen only; this is neither a formal TRIPOD+AI compliance claim nor a formal PROBAST+AI risk-of-bias judgment.",
    }


def build_upstream_evidence_requirements(context: Dict[str, Any], bundle: Dict[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    validation = _d(bundle.get("validation_scope"))
    identity = build_feature_identity_registry(context, bundle, root)
    missing_annotation = any(
        _d(item).get("msi_identification_level") == "not archived"
        for item in _l(identity.get("records"))
    )
    requirements = [
        {"requirement": "Independent external cohort evaluation", "requires_reanalysis": True, "status": "COMPLETE" if validation.get("external_declared") else "REQUIRED"},
        {"requirement": "Repeated-seed or bootstrap end-to-end panel stability", "requires_reanalysis": True, "status": "REQUIRED"},
        {"requirement": "Optimism-corrected calibration / nested validation", "requires_reanalysis": True, "status": "REQUIRED"},
        {"requirement": "Analytical annotation (MSI level, m/z, RT, adduct, ion mode, unit)", "requires_reanalysis": False, "requires_upstream_metadata": True, "status": "REQUIRED" if missing_annotation else "COMPLETE"},
        {"requirement": "Phase 0 LLM evidence-score repeatability and source-sentence provenance", "requires_reanalysis": True, "status": "REQUIRED"},
        {"requirement": "Target-population prevalence and prospective threshold validation", "requires_reanalysis": True, "status": "REQUIRED"},
        {"requirement": "Archived Phase 2 candidate-panel objective history for a reproducible Pareto landscape", "requires_reanalysis": True, "status": "REQUIRED"},
        {"requirement": "Cross-layer direction fields (prior, univariate, SHAP, and external cohort) for an evidence-concordance matrix", "requires_reanalysis": True, "status": "REQUIRED"},
    ]
    return {"schema_version": "phase4.upstream_evidence_requirements.v1", "status": "ACTION_REQUIRED" if any(item["status"] == "REQUIRED" for item in requirements) else "COMPLETE", "requirements": requirements}
    archived_runtime = _d(environment_manifest.get("runtime"))
    archived_packages = _d(archived_runtime.get("packages") or archived_runtime.get("report_environment_packages"))
    if archived_packages:
        packages.update({str(key): value for key, value in archived_packages.items()})
    return {
        "schema_version": "phase4.reproducibility_manifest.v2",
        "status": "PARTIAL" if artifact_list or git_commit != "not archived" else "NOT_ASSESSED",
        "scientific_run_id": _s(run.get("run_id")),
        "run_id": _s(run.get("run_id")),
        "report_build_id": _s(run.get("report_build_id")),
        "report_channel": _s(run.get("report_channel")) or "current",
        "upstream_run_manifest_id": audit_manifest.get("run_id", "not archived"),
        "report_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": git_commit, "dirty": git_dirty},
        "runtime": {"python": archived_runtime.get("python") or platform.python_version(), "platform": archived_runtime.get("platform") or platform.platform(), "report_environment_packages": packages},
        "random_seeds": random_seeds,
        "config": {"path": str(config_path) if config_value and config_path.exists() else "not archived", "sha256": _sha(config_path) if config_value and config_path.exists() else ""},
        "artifact_hashes": artifact_list,
        "policy": "Report regeneration metadata is separated from the original analysis run; missing provenance is explicit.",
    }


def build_report_enhancements(root: Path, context: Dict[str, Any], collected: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    paths = _phase1_paths(root, context, collected)
    base = {
        "data_split_audit": build_data_split_audit(root, context, collected, paths),
        "cohort_qc_snapshot": build_cohort_qc_snapshot(root, context, collected, paths),
        "feature_selection_evidence": build_feature_selection_evidence(root, context, paths),
        "training_association_snapshot": {"schema_version": "phase4.training_association_snapshot.v1", "status": "ASSESSED" if paths.get("training_csv") else "NOT_ASSESSED", "rows": _association_rows(paths["training_csv"], context) if paths.get("training_csv") else [], "policy": "Train-only descriptive association; not a causal effect and not used for selection."},
        "model_card": build_model_card(context, bundle, collected),
        "visual_evidence_registry": build_visual_evidence_registry(context),
        "reproducibility_manifest": build_reproducibility_manifest(root, context, bundle, paths),
    }
    # Some readiness probes depend on the split audit above, so expose the
    # partially built enhancement package to them without mutating upstream data.
    bundle["enhancements"] = base
    base.update({
        "feature_identity_registry": build_feature_identity_registry(context, bundle, root),
        "statistical_claim_validation": build_statistical_claim_validation(context),
        "prevalence_scenarios": build_prevalence_scenarios(context),
        "evidence_maturity": build_evidence_maturity(context, bundle, root),
        "reporting_readiness": build_reporting_readiness(context, bundle),
        "upstream_evidence_requirements": build_upstream_evidence_requirements(context, bundle, root),
    })
    return base
