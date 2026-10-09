"""Deterministic Evidence-to-Decision reporting for MetaboAgent Phase 4.

This module does not recompute scientific results.  It turns archived Phase 0-3
artifacts into an evidence-scoped report, explicit limitation registry, quality
gate and branded HTML/PDF delivery.
"""

from __future__ import annotations

import hashlib
import base64
import html
import json
import os
import shutil
import csv
import subprocess
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .report_enhancements import build_report_enhancements
from .contracts.integration import write_contract_package


BRAND = {
    "navy": "#18233A",
    "teal": "#138A86",
    "burgundy": "#8F2942",
    "ink": "#202735",
    "muted": "#667085",
    "line": "#D9E0E7",
    "paper": "#FFFFFF",
    "wash": "#F5F8FA",
}

# Curated from the user-provided Table1_metabolomics_benchmark_datasets.docx.
# It is study provenance, not a data-quality or evidence-quality ranking.
BENCHMARK_DATASETS = (
    {"label": "Lung cancer", "aliases": ["lung cancer"], "study_id": "ST000396", "specimen": "Plasma", "platform": "GC-MS", "data_level": "Untargeted metabolomics", "design": "CARET nested case-control, prediagnostic screening", "task": "100 cases vs 199 controls"},
    {"label": "Colorectal cancer", "aliases": ["colorectal cancer", "crc"], "study_id": "ST000284", "specimen": "Serum", "platform": "Targeted LC-MS/MS", "data_level": "Targeted metabolomics", "design": "CRC vs healthy controls", "task": "64 cases vs 84 controls"},
    {"label": "Hepatocellular carcinoma", "aliases": ["hepatocellular carcinoma", "hcc"], "study_id": "ST000865", "specimen": "Plasma", "platform": "Targeted GC-MS", "data_level": "Targeted metabolomics", "design": "HCC vs cirrhosis", "task": "40 cases vs 44 controls"},
    {"label": "Type 2 diabetes", "aliases": ["type 2 diabetes", "t2dm"], "study_id": "ST003390", "specimen": "Serum", "platform": "Integrated targeted LC-MS", "data_level": "Targeted metabolomics", "design": "T2DM vs controls", "task": "100 cases vs 200 controls"},
    {"label": "Myalgic encephalomyelitis/chronic fatigue syndrome (ME/CFS)", "aliases": ["cfs", "me/cfs", "chronic fatigue syndrome", "myalgic encephalomyelitis"], "study_id": "ST002001", "specimen": "Plasma", "platform": "Targeted + untargeted LC-MS", "data_level": "Hybrid targeted and untargeted metabolomics", "design": "Frequency-matched case-control", "task": "106 cases vs 91 controls"},
    {"label": "Lung adenocarcinoma", "aliases": ["lung adenocarcinoma", "luad"], "study_id": "ST000368", "specimen": "Serum", "platform": "Untargeted GC-TOFMS", "data_level": "Untargeted metabolomics", "design": "Selected serum cohort", "task": "43 cases vs 43 controls"},
    {"label": "Liver cancer risk", "aliases": ["liver cancer risk"], "study_id": "ST002764", "specimen": "Serum", "platform": "Untargeted lipidomics LC-MS", "data_level": "Untargeted lipidomics", "design": "ATBC nested case-control, future risk", "task": "219 cases vs 219 controls"},
    {"label": "Prostate cancer", "aliases": ["prostate cancer"], "study_id": "ST000783", "specimen": "Blood", "platform": "Biocrates p180 (FIA-MS)", "data_level": "Targeted metabolomics", "design": "Case-control", "task": "40 cases vs 50 controls"},
    {"label": "Pancreatic ductal adenocarcinoma", "aliases": ["pdac", "pancreatic ductal adenocarcinoma"], "study_id": "ST004073", "specimen": "Plasma", "platform": "Non-targeted + targeted lipidomics LC-MS", "data_level": "Hybrid targeted and untargeted lipidomics", "design": "Case-control", "task": "180 cases vs 164 controls"},
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase0_prior_counts(phase0: Dict[str, Any]) -> Dict[str, int]:
    """Return Phase 0 prior counts with compatibility for legacy schemas.

    Older phase0_output_latest.json files archive the authoritative records in
    ``confirmed_biomarkers`` and ``final_priors`` but do not materialize the
    convenience count fields consumed by Phase 4.  Counts must therefore be
    derived from those lists rather than defaulting to zero.  This keeps the
    report aligned with the Phase 0 atlas without inferring any new evidence.
    """

    def resolve_count(value: Any, keys: Iterable[str]) -> int:
        parsed = _safe_float(value)
        if parsed is not None:
            return max(int(parsed), 0)
        for key in keys:
            records = phase0.get(key)
            if isinstance(records, list):
                return len(records)
        return 0

    return {
        "confirmed_biomarkers": resolve_count(
            phase0.get("confirmed_biomarker_count"),
            ("confirmed_biomarkers",),
        ),
        "final_priors": resolve_count(
            phase0.get("final_priors_count"),
            ("final_priors",),
        ),
        "candidate_metabolites": resolve_count(
            phase0.get("candidate_metabolite_count"),
            ("candidate_metabolites",),
        ),
    }


def _dataset_profile(report_context: Dict[str, Any]) -> Dict[str, Any]:
    """Match run identity to the user-supplied nine-dataset benchmark catalogue."""

    phase0 = _safe_dict(report_context.get("phase0"))
    run = _safe_dict(report_context.get("run"))
    raw_tokens = " ".join([_safe_str(phase0.get("disease_name")), _safe_str(run.get("run_id"))]).strip()
    # Run identifiers commonly use underscores (for example
    # ``colorectal_cancer_full_metaboagent...``), while the benchmark catalogue
    # stores human-readable aliases.  Normalize separators before matching so
    # reader-facing titles do not fall back to the full technical run ID.
    tokens = re.sub(r"[_\-]+", " ", raw_tokens).lower()
    for item in BENCHMARK_DATASETS:
        normalized_aliases = [re.sub(r"[_\-]+", " ", str(alias)).lower() for alias in item["aliases"]]
        if any(alias in tokens for alias in normalized_aliases) or item["study_id"].lower() in tokens:
            matched = {"schema_version": "phase4.dataset_profile.v1", "matched": True, "metadata_source": "User-provided Table1_metabolomics_benchmark_datasets.docx", **item}
            task = _safe_str(item.get("task"))
            counts = re.search(r"(?P<cases>\d+)\s+cases?\s+vs\s+(?P<controls>\d+)\s+controls?", task, flags=re.IGNORECASE)
            if counts:
                matched["n_cases"] = int(counts.group("cases"))
                matched["n_controls"] = int(counts.group("controls"))
                matched["n_participants"] = int(counts.group("cases")) + int(counts.group("controls"))
            return matched
    # A run can be scientifically identifiable even when it is not present in
    # the optional benchmark catalogue.  Do not surface the implementation
    # state ("not matched") as the study title; derive a conservative label
    # from the archived disease field or run ID instead.  This keeps reports
    # readable without inventing catalogue metadata.
    fallback = _safe_str(phase0.get("disease_name") or run.get("disease") or run.get("run_id") or "Unspecified condition").strip()
    raw_tokens = re.split(r"[_\-\s]+", fallback)
    ignored_tokens = {"oldsplit", "audit", "schema", "preview", "report"}
    kept_tokens = [
        token for token in raw_tokens
        if token and token.lower() not in ignored_tokens
        and not re.fullmatch(r"seed\d+|retry\d+|\d{8}", token, flags=re.IGNORECASE)
    ]
    fallback = " ".join(kept_tokens).strip(" ._-") or "Unspecified condition"
    label_tokens = []
    for token in fallback.split():
        lower = token.lower()
        label_tokens.append({"hcc": "HCC", "crc": "CRC", "pdac": "PDAC", "t2dm": "T2DM", "me/cfs": "ME/CFS"}.get(lower, token.capitalize()))
    return {"schema_version": "phase4.dataset_profile.v1", "matched": False, "metadata_source": "Run identity fallback; benchmark catalogue metadata unavailable", "label": " ".join(label_tokens), "data_level": "Not catalogued"}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _safe_float(value)
    return "Not available" if numeric is None else f"{numeric:.{digits}f}"


def _format_ci(value: Any, digits: int = 3) -> str:
    points = value if isinstance(value, (list, tuple)) else []
    if len(points) != 2 or _safe_float(points[0]) is None or _safe_float(points[1]) is None:
        return "Not archived"
    return f"{float(points[0]):.{digits}f} to {float(points[1]):.{digits}f}"


def _display_evidence_scope(scope: Any) -> str:
    """Map machine-facing assessment scopes to the report vocabulary."""

    value = _safe_str(scope).strip().lower()
    if "external" in value:
        return "EXTERNAL"
    if "holdout" in value:
        return "INTERNAL-HOLDOUT"
    if any(token in value for token in ("oof", "cross", "cv", "development")):
        return "INTERNAL-CV"
    if "model" in value or "development" in value:
        return "MODEL-DEVELOPMENT"
    if value:
        return _safe_str(scope)
    return "NOT-ASSESSABLE"


def _metric_registry(
    root: Path,
    collected: Dict[str, Any],
    report_context: Dict[str, Any],
    audit_payloads: Dict[str, Dict[str, Any]],
    calibration: Dict[str, Any],
    calibration_source: str,
) -> Dict[str, Any]:
    """Create the single source of truth for report-facing performance metrics.

    Every downstream section must read this registry instead of independently
    traversing Phase 2, Phase 3 and audit payloads.  This prevents the common
    failure mode where Claim graph, performance tables and clinical utility
    prose report different scopes or different calibration versions.
    """

    uncertainty = _safe_dict(audit_payloads.get("performance_uncertainty.json"))
    oof = _safe_dict(uncertainty.get("development_oof"))
    holdout = _safe_dict(uncertainty.get("internal_holdout"))
    oof_point = _safe_dict(oof.get("point_estimate"))
    holdout_point = _safe_dict(holdout.get("point_estimate"))
    oof_ci = _safe_dict(oof.get("bootstrap_ci"))
    holdout_ci = _safe_dict(holdout.get("bootstrap_ci"))

    def fallback_auc(scope: str) -> Optional[float]:
        return _internal_auc(collected, report_context) if scope == "INTERNAL-CV" else _holdout_auc(report_context)

    def metric_block(
        point: Dict[str, Any],
        ci: Dict[str, Any],
        scope: str,
        prediction_name: str,
        fallback: Optional[float],
    ) -> Dict[str, Any]:
        auc = _first_numeric(point.get("roc_auc"), fallback)
        prediction_artifact = _artifact_ref(root, f"output/audit/{prediction_name}")
        point_source = _artifact_ref(
            root,
            _safe_dict(report_context.get("sources")).get("phase2_result")
            or "output/phase1/artifacts/phase2_winner_scores.json",
        )
        prediction_available = bool(prediction_artifact.get("exists"))
        return {
            "scope": scope,
            "n_samples": point.get("n_samples"),
            "n_positive": point.get("n_positive"),
            "n_negative": point.get("n_negative"),
            "auc": auc,
            "pr_auc": _safe_float(point.get("pr_auc")),
            "brier_score": _safe_float(point.get("brier_score")),
            "auc_ci_95": ci.get("roc_auc_ci95"),
            "pr_auc_ci_95": ci.get("pr_auc_ci95"),
            "brier_score_ci_95": ci.get("brier_score_ci95"),
            "estimation_method": (
                (
                    "pooled out-of-fold predictions with stratified percentile bootstrap"
                    if prediction_available
                    else "point estimate from archived Phase 2 winner-score artifact; prediction-level OOF output not archived"
                )
                if scope == "INTERNAL-CV"
                else (
                    "archived internal holdout predictions with stratified percentile bootstrap"
                    if prediction_available
                    else "point estimate from archived Phase 2 winner-score artifact; prediction-level holdout output not archived"
                )
            ),
            "prediction_artifact": prediction_artifact,
            "uncertainty_artifact": _artifact_ref(root, "output/audit/performance_uncertainty.json"),
            "point_estimate_source_artifact": point_source,
            "evidence_completeness": "COMPLETE" if prediction_available else "POINT-ESTIMATE-ONLY",
            "provenance_note": (
                "Prediction-level artifact and uncertainty payload are archived."
                if prediction_available
                else "The point estimate is available from the archived Phase 2 winner-score artifact; prediction-level evaluation output is not archived."
            ),
        }

    # The run-scoped uncertainty artifact is the canonical metric source for
    # all current evaluation predictions.  Only fall back to the legacy Phase
    # 2 clinical-utility payload when that audit artifact is unavailable.
    audit_calibration_brier = _safe_float(holdout_point.get("brier_score"))
    if audit_calibration_brier is not None:
        calibration_brier = audit_calibration_brier
        canonical_calibration_source = "performance_uncertainty"
        canonical_calibration_scope = "internal_holdout_probability_assessment"
        calibration_artifacts = [
            _artifact_ref(root, "output/audit/performance_uncertainty.json"),
            _artifact_ref(root, "output/audit/evaluation_predictions_holdout.csv"),
        ]
    else:
        calibration_brier = _safe_float(calibration.get("brier_score"))
        canonical_calibration_source = calibration_source
        canonical_calibration_scope = _safe_str(calibration.get("assessment_scope") or "INTERNAL-CV")
        calibration_artifacts = [
            _artifact_ref(root, "output/audit/performance_uncertainty.json"),
            _artifact_ref(root, _safe_dict(report_context.get("sources")).get("phase2_clinical_utility")),
        ]
    # A calibration summary can be archived without the row-level prediction
    # file needed to independently recompute it.  Keep those two evidence
    # levels distinct so the registry never implies stronger provenance than
    # the run package actually contains.
    calibration_prediction_artifact = _artifact_ref(
        root,
        "output/audit/evaluation_predictions_holdout.csv"
        if canonical_calibration_scope == "internal_holdout_probability_assessment"
        else "output/audit/evaluation_predictions_oof.csv",
    )
    calibration_prediction_available = bool(calibration_prediction_artifact.get("exists"))
    calibration_summary_available = any(bool(_safe_dict(item).get("exists")) for item in calibration_artifacts)
    if canonical_calibration_scope == "internal_holdout_probability_assessment":
        if calibration_prediction_available:
            calibration_estimation_method = "fixed internal holdout predictions; descriptive calibration metrics"
        else:
            calibration_estimation_method = "archived calibration summary; prediction-level holdout output not archived"
    elif calibration_prediction_available:
        calibration_estimation_method = "archived OOF prediction payload"
    else:
        calibration_estimation_method = "archived calibration summary; prediction-level OOF output not archived"
    calibration_block = {
        "scope": canonical_calibration_scope,
        "brier_score": calibration_brier,
        "source": canonical_calibration_source,
        "estimation_method": calibration_estimation_method,
        "source_artifacts": calibration_artifacts,
        "prediction_artifact": calibration_prediction_artifact,
        "evidence_completeness": (
            "PREDICTION-LEVEL" if calibration_prediction_available
            else "SUMMARY-ONLY" if calibration_summary_available
            else "NOT-AVAILABLE"
        ),
        "provenance_note": (
            "Row-level prediction artifact and calibration summary are archived."
            if calibration_prediction_available
            else "Calibration point estimates are archived, but the row-level prediction artifact is not available for independent recomputation."
            if calibration_summary_available
            else "No archived calibration payload was available."
        ),
    }
    return {
        "schema_version": "phase4.metric_registry.v1",
        "policy": "One canonical value per metric; scope, sample size, estimation method, uncertainty and evidence artifacts are mandatory fields.",
        "development_cv": metric_block(oof_point, oof_ci, "INTERNAL-CV", "evaluation_predictions_oof.csv", fallback_auc("INTERNAL-CV")),
        "internal_holdout": metric_block(holdout_point, holdout_ci, "INTERNAL-HOLDOUT", "evaluation_predictions_holdout.csv", fallback_auc("INTERNAL-HOLDOUT")),
        "calibration": calibration_block,
    }


def _first_numeric(*values: Any) -> Optional[float]:
    for value in values:
        numeric = _safe_float(value)
        if numeric is not None:
            return numeric
    return None


def _calibration_payload(clinical: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Choose the most decision-relevant calibration payload that has a Brier score.

    Some historical runs create an ``*_adjusted`` placeholder even where no
    prevalence adjustment was applied.  Presence of that empty object must not
    mask the archived primary or raw calibration result.
    """

    candidates = (
        ("prevalence-adjusted", _safe_dict(clinical.get("probability_calibration_adjusted"))),
        ("primary", _safe_dict(clinical.get("probability_calibration"))),
        ("raw", _safe_dict(clinical.get("probability_calibration_raw"))),
    )
    for label, payload in candidates:
        if _safe_float(payload.get("brier_score")) is not None:
            return payload, label
    for label, payload in candidates:
        if payload:
            return payload, label
    return {}, "not archived"


def _mechanistic_evidence_registry(report_context: Dict[str, Any]) -> Dict[str, Any]:
    """Separate model behaviour from biological knowledge and causal evidence."""

    discussion = _safe_dict(_safe_dict(report_context.get("report_inputs")).get("discussion"))
    feature_dictionary = _safe_list(discussion.get("winner_feature_dictionary"))
    pathways = [str(item).strip() for item in _safe_list(discussion.get("disease_core_pathways")) if str(item).strip()]
    figure_evidence = _safe_dict(discussion.get("figure_evidence_summary"))
    shap = _safe_dict(figure_evidence.get("shap"))
    rcs = _safe_dict(figure_evidence.get("rcs"))
    ranked = {
        _safe_str(item.get("feature") or item.get("display_feature") or item.get("name")).strip()
        for item in _safe_list(shap.get("top_features") or shap.get("ranked_features"))
        if isinstance(item, dict)
    }
    nonlinear = {
        _safe_str(item).strip() for item in _safe_list(rcs.get("nonlinear_features")) if _safe_str(item).strip()
    }
    entries: List[Dict[str, Any]] = []
    for record_value in feature_dictionary:
        record = _safe_dict(record_value)
        token = _safe_str(record.get("feature_token") or record.get("display_name")).strip()
        label = _safe_str(record.get("preferred_report_name") or record.get("display_name") or token).strip()
        aliases = {token, label, _safe_str(record.get("display_name")).strip()}
        entries.append({
            "feature": label or token,
            "feature_token": token,
            "model_association": {
                "shap_ranked": bool(aliases & ranked),
                "nonlinear_association": bool(aliases & nonlinear),
                "evidence_scope": "model attribution / within-cohort association",
            },
            "chemical_annotation": {
                "origin_subtype": _safe_str(record.get("origin_subtype") or "not recorded"),
                "hmdb_ids": _safe_list(record.get("hmdb_ids")),
                "annotation_available": bool(record.get("hmdb_ids") or record.get("mapped_metabolites")),
            },
            "pathway_mapping": {
                "mapped_pathways": _safe_list(record.get("mapped_pathways")),
                "matched_phase0_pathway": record.get("matched_pathway"),
                "available": bool(record.get("mapped_pathways") or record.get("matched_pathway")),
            },
            "prior_evidence": {
                "prior_supported": bool(record.get("prior_supported")),
                "matched_phase0_biomarker": record.get("matched_phase0_biomarker"),
            },
            "direct_mechanistic_validation": False,
        })
    pathway_supported = bool(pathways) or any(_safe_dict(item.get("pathway_mapping")).get("available") for item in entries)
    prior_supported = any(_safe_dict(item.get("prior_evidence")).get("prior_supported") for item in entries)
    interpretation_level = "PATHWAY-INFORMED" if pathway_supported or prior_supported else "MODEL-ASSOCIATION-ONLY"
    return {
        "schema_version": "phase4.mechanistic_evidence.v1",
        "interpretation_level": interpretation_level,
        "disease_core_pathways": pathways,
        "entries": entries,
        "policy": (
            "Model attribution, association and chemical annotation are not causal evidence. "
            "Disease-mechanistic language is permitted only when a run-local pathway or prior-evidence source is registered."
        ),
    }


def _narrative_block(
    claim_group: str,
    finding: str,
    interpretation: str,
    boundary: str,
    evidence_scope: str,
    source_artifacts: Iterable[str],
    *,
    short_version: Optional[str] = None,
    medium_version: Optional[str] = None,
    full_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Create one canonical Finding-Interpretation-Boundary narrative block.

    The same block is rendered at different lengths in the executive summary,
    PDF and HTML.  Narrative text is assembled only from archived registries;
    an LLM may polish it later, but it must not invent a new numerical claim.
    """

    finding = _safe_str(finding).strip()
    interpretation = _safe_str(interpretation).strip()
    boundary = _safe_str(boundary).strip()
    medium = _safe_str(medium_version or f"{finding} {interpretation} {boundary}").strip()
    full = _safe_str(full_version or medium).strip()
    short = _safe_str(short_version or finding).strip()
    return {
        "claim_group": claim_group,
        "finding": finding,
        "interpretation": interpretation,
        "boundary": boundary,
        "evidence_scope": evidence_scope,
        "source_artifacts": [str(item) for item in source_artifacts if str(item).strip()],
        "short_version": short,
        "medium_version": medium,
        "full_version": full,
    }


def _scientific_narrative_registry(
    report_context: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive a cross-format scientific narrative from validated registries.

    This is deliberately deterministic.  It addresses the common failure mode
    where HTML and PDF ask an LLM to independently explain the same results and
    consequently drift in feature names, thresholds or evidence scope.
    """

    phase0 = _safe_dict(report_context.get("phase0"))
    phase1 = _safe_dict(report_context.get("phase1"))
    phase2 = _safe_dict(report_context.get("phase2"))
    profile = _safe_dict(bundle.get("dataset_profile"))
    metric_registry = _safe_dict(bundle.get("metric_registry"))
    development = _safe_dict(metric_registry.get("development_cv"))
    holdout = _safe_dict(metric_registry.get("internal_holdout"))
    calibration = _safe_dict(metric_registry.get("calibration"))
    enhancements = _safe_dict(bundle.get("enhancements"))
    feature_evidence = _safe_dict(enhancements.get("feature_selection_evidence"))
    selection_summary = _safe_dict(_safe_dict(phase1.get("selection_strategy")).get("feature_selection_summary"))
    search_summary = _safe_dict(phase2.get("search_summary"))
    incremental = _safe_dict(phase2.get("incremental_value"))
    clinical = _safe_dict(phase2.get("clinical_utility"))
    recalibration = _safe_dict(clinical.get("recalibration_summary"))
    prevalence = _safe_dict(enhancements.get("prevalence_scenarios"))
    mechanistic = _safe_dict(bundle.get("mechanistic_evidence_registry"))

    phase1_count = phase1.get("selected_feature_count")
    winner_count = phase2.get("winner_feature_count") or len(_safe_list(phase2.get("winner_features")))
    stable_core_count = selection_summary.get("stable_core_count")
    protected_count = phase2.get("n_protected_anchor_features")
    added_count = phase2.get("n_added_features_beyond_protected")
    if added_count is None and _safe_float(protected_count) is not None:
        added_count = max(int(winner_count or 0) - int(protected_count or 0), 0)

    task = _safe_str(profile.get("task") or "the archived case-control cohort")
    study_id = _safe_str(profile.get("study_id") or "the archived study")
    specimen = _safe_str(profile.get("specimen") or "the archived biospecimen")
    platform = _safe_str(profile.get("platform") or "the archived metabolomics platform")
    design = _safe_str(profile.get("design") or "an archived case-control design")
    disease = _safe_str(profile.get("label") or phase0.get("disease_name") or "the target disease")
    cv_auc = _fmt(development.get("auc"))
    holdout_auc = _fmt(holdout.get("auc"))
    cv_num = _safe_float(development.get("auc"))
    holdout_num = _safe_float(holdout.get("auc"))
    auc_delta = abs(cv_num - holdout_num) if cv_num is not None and holdout_num is not None else None

    validation = _safe_dict(bundle.get("validation_scope"))
    if validation.get("external_declared"):
        validation_boundary = "An independent external cohort is declared in the run manifest; transportability still depends on the corresponding archived external estimates."
    else:
        validation_boundary = "The internal holdout comes from the same archived study and does not establish transportability to an independent population."

    study_finding = (
        f"The analysis used {study_id}, a {specimen.lower()} {platform} dataset with {task} under {design.lower()}."
    )
    study_interpretation = (
        "The archived workflow separates train-only candidate selection and model development from an internal held-out evaluation; Phase 0-4 counts describe evidence lineage rather than additional validation cohorts."
    )

    prior_counts = _phase0_prior_counts(phase0)
    prior_count = prior_counts["confirmed_biomarkers"]
    winner_records = [
        _safe_dict(item)
        for item in _safe_list(phase2.get("winner_feature_records"))
        if isinstance(item, dict)
    ]
    prior_supported_records = [
        item for item in winner_records
        if bool(item.get("is_prior_supported") or item.get("prior_supported"))
    ]
    final_panel_prior_overlap = len(prior_supported_records) if winner_records else None
    final_panel_size = int(winner_count or 0)
    prior_atlas_finding = (
        f"Phase 0 retained {prior_count} confirmed prior biomarker{'s' if int(prior_count or 0) != 1 else ''}."
    )
    if prior_count:
        prior_atlas_interpretation = (
            "The retained priors provide literature-based biological anchoring for the downstream panel search."
        )
        if final_panel_prior_overlap is not None:
            prior_atlas_interpretation += (
                f" {final_panel_prior_overlap}/{final_panel_size} final-panel members had Phase 0 prior support."
                if final_panel_prior_overlap
                else " None of the final-panel members overlapped the retained Phase 0 prior set."
            )
        prior_atlas_boundary = (
            "Prior support is literature-derived and does not establish causality, independent validation or target-population utility."
        )
    else:
        prior_atlas_interpretation = (
            "The prior-evidence atlas documents screened literature evidence and biological anchoring even though no candidate satisfied the confirmed-prior criterion."
        )
        prior_atlas_boundary = "Absence of a retained prior is a workflow result; it does not imply absence of biological association."

    low_stability: List[str] = []
    high_stability: List[str] = []
    name_by_token: Dict[str, str] = {}
    for item in _safe_list(phase2.get("winner_feature_records")):
        record = _safe_dict(item)
        token = _safe_str(record.get("feature") or record.get("feature_token") or record.get("standardized_name"))
        name_by_token[token] = _safe_str(record.get("display_name") or token)
    for item in _safe_list(_safe_dict(feature_evidence).get("winner_records")):
        record = _safe_dict(item)
        token = _safe_str(record.get("feature") or record.get("canonical_id"))
        frequency = _safe_float(record.get("phase1_selection_frequency"))
        label = name_by_token.get(token, token)
        if frequency is not None and frequency < 0.10:
            low_stability.append(label)
        if frequency is not None and frequency >= 0.75:
            high_stability.append(label)
    low_text = ", ".join(low_stability[:3])
    high_text = ", ".join(high_stability[:3])
    route_sentence = (
        f"The Phase 2 search began from {protected_count} protected anchor feature{'s' if int(protected_count or 0) != 1 else ''} and added {added_count} features through multi-objective panel search."
        if protected_count is not None and added_count is not None
        else "The Phase 2 panel was selected by the archived multi-objective search rather than by a simple frequency ranking."
    )
    if low_text:
        route_sentence += f" This explains why low-frequency Phase 1 candidates such as {low_text} may remain in the final panel: their inclusion requires a documented Phase 2 objective contribution, not high resampling frequency."
    discovery_finding = (
        f"The candidate pool was reduced from {phase1_count if phase1_count is not None else 'the archived Phase 1 set'} to {winner_count} final features."
    )
    discovery_interpretation = (
        "The final panel was selected after multi-objective refinement of the Phase 1 candidate set rather than by discrimination alone. "
        "The archived solution reflects the prespecified trade-off among predictive performance, biological support, redundancy control and analytical burden while preserving protected evidence anchors. "
        + route_sentence
        + (f" High-frequency stable candidates included {high_text}." if high_text else "")
    )
    discovery_boundary = "Selection frequency is train-only stability evidence; it is not equivalent to final-panel inclusion probability or independent validation."

    performance_finding = f"The final panel achieved an internal cross-validated AUROC of {cv_auc} and an internal holdout AUROC of {holdout_auc}."
    performance_interpretation = (
        f"The holdout point estimate was lower by {auc_delta:.3f}, consistent with modest attenuation beyond development resampling."
        if auc_delta is not None
        else "The two estimates provide complementary internal evidence for discrimination."
    )
    performance_boundary = validation_boundary

    slope = _first_numeric(recalibration.get("calibrated_calibration_slope"), calibration.get("calibration_slope"))
    intercept = _first_numeric(recalibration.get("calibrated_calibration_intercept"), calibration.get("calibration_intercept"))
    calibration_bits = [f"Brier score { _fmt(calibration.get('brier_score')) }"]
    if slope is not None:
        calibration_bits.append(f"calibration slope {slope:.3f}")
    if intercept is not None:
        calibration_bits.append(f"calibration intercept {intercept:.3f}")
    calibration_finding = "The archived probability assessment reported " + ", ".join(calibration_bits) + "."
    if slope is not None and intercept is not None:
        calibration_interpretation = (
            f"The internal calibration slope ({slope:.3f}) and intercept ({intercept:.3f}) were close to their ideal values of 1 and 0, respectively, while the Brier score summarized overall probability error within the development cohort."
        )
    else:
        calibration_interpretation = (
            "The Brier score summarizes overall probability error within the assessed cohort; calibration transportability remains unestablished."
        )
    calibration_boundary = "These calibration estimates are cohort-specific and do not establish target-population absolute-risk calibration."

    nri = _safe_dict(incremental.get("nri"))
    idi = _safe_dict(incremental.get("idi"))
    incremental_finding = (
        f"Against the Phase 1 reference, continuous NRI was {_fmt(nri.get('value'))} and IDI was {_fmt(idi.get('value'))}."
        if nri or idi else "No archived incremental-value estimates were available."
    )
    incremental_interpretation = "These are secondary out-of-fold reclassification measures and should be read alongside discrimination and panel size, not as independent validation."
    incremental_boundary = "Reclassification metrics do not establish clinical utility or transportability."

    threshold = _safe_dict(clinical.get("recommended_threshold_summary")).get("operating_characteristics")
    threshold = _safe_dict(threshold)
    resource = _safe_dict(clinical.get("resource_impact_per_1000"))
    threshold_value = _fmt(threshold.get("threshold"), 2)
    observed_prev = _safe_float(_safe_dict(clinical.get("prevalence_context")).get("study_prevalence"))
    clinical_finding = (
        f"At the illustrative threshold {threshold_value}, sensitivity was {_fmt(threshold.get('sensitivity'))}, specificity {_fmt(threshold.get('specificity'))}, and the observed-sample flagged rate {_fmt(threshold.get('flagged_rate'))}."
    ) if threshold else "No archived operating-point metrics were available."
    clinical_interpretation = (
        f"Under the archived sampling prevalence of {observed_prev * 100:.1f}% and a 1,000-person projection, the operating point corresponds to approximately {_fmt(resource.get('high_risk_identified'), 1)} true-positive-equivalent detections and {_fmt(resource.get('confirmatory_tests_triggered'), 1)} total flagged evaluations."
        if observed_prev is not None and resource else "The available threshold estimates are scenario analyses rather than deployment recommendations."
    )
    clinical_boundary = "Prevalence-adjusted PPV, NPV and false-positive burden must be recalculated and externally validated in the intended target population."

    shap = _safe_dict(_safe_dict(_safe_dict(report_context.get("report_inputs")).get("discussion")).get("figure_evidence_summary")).get("shap")
    shap = _safe_dict(shap)
    ranked = [_safe_dict(item) for item in _safe_list(shap.get("ranked_features"))]
    top_labels = [
        _safe_str(item.get("feature") or item.get("display_feature") or item.get("name"))
        for item in ranked[:4]
        if _safe_str(item.get("feature") or item.get("display_feature") or item.get("name")).strip()
    ]
    figure_evidence = _safe_dict(_safe_dict(_safe_dict(report_context.get("report_inputs")).get("discussion")).get("figure_evidence_summary"))
    nonlinear = _safe_list(_safe_dict(figure_evidence.get("rcs")).get("nonlinear_features"))
    top_text = ", ".join(top_labels) if top_labels else "the highest-ranked archived features"
    biology_finding = f"Global model attribution was concentrated in {top_text}."
    if nonlinear:
        biology_finding += f" Restricted cubic spline evidence flagged {', '.join(_safe_str(item) for item in nonlinear[:3])} as non-linear."
    biology_interpretation = "SHAP directions describe multivariable model behaviour, while spline curves describe within-cohort association structure; together they identify features for biological follow-up."
    biology_boundary = "Attribution, association and pathway annotations are hypothesis-generating and do not establish causality, protection or pathway activation."

    conclusion_finding = f"The archived run supports a {winner_count}-feature biomarker panel with internal discrimination evidence."
    conclusion_interpretation = "The most useful next step is independent external validation with complete analytical metadata and population-specific calibration review."
    conclusion_boundary = "The current evidence does not support clinical deployment or a validated population screening threshold."

    blocks = {
        "study_context": _narrative_block("study_context", study_finding, study_interpretation, validation_boundary, "DESCRIPTIVE-QC", ["dataset_profile.json", "phase_flow_manifest.json"]),
        "discovery_summary": _narrative_block("discovery_summary", discovery_finding, discovery_interpretation, discovery_boundary, "TRAIN-ONLY / MODEL-DEVELOPMENT", ["feature_selection_evidence.json", "phase2_search_summary.json"]),
        "performance_summary": _narrative_block("performance_summary", performance_finding, performance_interpretation, performance_boundary, "INTERNAL-CV + INTERNAL-HOLDOUT", ["metric_registry.json", "figure_manifest.json"]),
        "calibration_summary": _narrative_block("calibration_summary", calibration_finding, calibration_interpretation, calibration_boundary, _display_evidence_scope(calibration.get("scope")), ["metric_registry.json", "phase2_probability_recalibration.json"]),
        "incremental_value_summary": _narrative_block("incremental_value_summary", incremental_finding, incremental_interpretation, incremental_boundary, "INTERNAL-CV", ["phase2_result.json"]),
        "clinical_summary": _narrative_block("clinical_summary", clinical_finding, clinical_interpretation, clinical_boundary, "SCENARIO-PROJECTION", ["prevalence_scenarios.json", "metric_registry.json"]),
        "biology_summary": _narrative_block("biology_summary", biology_finding, biology_interpretation, biology_boundary, "MODEL-ASSOCIATION", ["mechanistic_evidence_registry.json", "figure_manifest.json"]),
        "prior_evidence_summary": _narrative_block("prior_evidence_summary", prior_atlas_finding, prior_atlas_interpretation, prior_atlas_boundary, "LITERATURE-CONTEXT", ["phase0_output_latest.json", "figure_manifest.json"]),
        "conclusion_summary": _narrative_block("conclusion_summary", conclusion_finding, conclusion_interpretation, conclusion_boundary, "EVIDENCE-TO-DECISION", ["claim_evidence_graph.json", "metric_registry.json"]),
    }
    return {
        "schema_version": "phase4.scientific_narrative_registry.v1",
        "policy": "Canonical deterministic narrative blocks are rendered at short, medium and full lengths; narrative text cannot override registry values or evidence scope.",
        "blocks": blocks,
    }


def _narrative_html_block(block: Dict[str, Any], *, version: str = "full") -> str:
    """Render a canonical narrative block for the HTML reader view."""

    if not block:
        return '<p class="empty">Narrative block not available.</p>'
    finding = _safe_str(block.get("finding")).strip()
    interpretation = _safe_str(block.get("interpretation")).strip()
    boundary = _safe_str(block.get("boundary")).strip()
    scope = _safe_str(block.get("evidence_scope") or "NOT_ASSESSABLE")
    return (
        '<div class="narrative-block">'
        f'<div class="narrative-scope">{html.escape(scope)}</div>'
        f'<p><strong>Finding.</strong> {html.escape(finding)}</p>'
        f'<p><strong>Interpretation.</strong> {html.escape(interpretation)}</p>'
        f'<p class="boundary"><strong>Boundary.</strong> {html.escape(boundary)}</p>'
        '</div>'
    )


def _resolve_path(run_root: Path, value: Any) -> Path:
    raw = _safe_str(value).strip()
    if not raw:
        return Path("")
    path = Path(raw)
    return path if path.is_absolute() else run_root / path


_FIGURE_OUTPUT_ALIASES = {
    "phase2_objective_shift_summary.pdf": "phase2_4d_radar_profile.pdf",
    "phase2_objective_shift_summary.png": "phase2_4d_radar_profile.png",
    "phase2_objective_shift_summary.svg": "phase2_4d_radar_profile.svg",
}


def _resolve_figure_path(run_root: Path, value: Any) -> Path:
    path = _resolve_path(run_root, value)
    if path.exists():
        return path
    alias = _FIGURE_OUTPUT_ALIASES.get(path.name)
    if alias:
        candidate = path.with_name(alias)
        if candidate.exists():
            return candidate
        candidate = run_root / "output" / "figures" / "single_panels" / alias
        if candidate.exists():
            return candidate
    return path


def _artifact_ref(run_root: Path, value: Any) -> Dict[str, Any]:
    path = _resolve_path(run_root, value)
    exists = bool(_safe_str(value).strip()) and path.exists()
    return {
        "path": _safe_str(value),
        "resolved_path": str(path) if _safe_str(value).strip() else "",
        "exists": exists,
        "sha256": _sha256(path) if exists and path.is_file() else "",
    }


def _portable_href(report_dir: Path, resolved: Path) -> str:
    """Stage an evidence file beneath the report so links survive packaging."""
    if not resolved.exists() or not resolved.is_file():
        return ""
    evidence_dir = report_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    digest = _sha256(resolved)[:12]
    target = evidence_dir / f"{digest}_{resolved.name}"
    try:
        if target.resolve() != resolved.resolve():
            shutil.copy2(resolved, target)
    except OSError:
        return ""
    return os.path.relpath(target, start=report_dir)


def _portable_pdf_preview(report_dir: Path, resolved: Path) -> str:
    """Render a first-page PNG preview for PDF-only figure outputs when possible."""
    if resolved.suffix.lower() != '.pdf' or not resolved.exists():
        return ''
    evidence_dir = report_dir / 'evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / f'{_sha256(resolved)[:12]}_{resolved.stem}_preview.png'
    if target.exists():
        return os.path.relpath(target, start=report_dir)
    try:
        executable = shutil.which('pdftoppm')
        if not executable:
            return ''
        prefix = target.with_suffix('')
        subprocess.run([executable, '-png', '-f', '1', '-singlefile', str(resolved), str(prefix)], check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ''
    return os.path.relpath(target, start=report_dir) if target.exists() else ''


def _figure_preview_candidates(run_root: Path, resolved: Path) -> List[Path]:
    """Yield image-first preview candidates for a figure artifact.

    Historical figure manifests frequently register only the PDF delivery
    asset, while the run workspace already contains sibling PNG/SVG outputs
    under ``single_panels`` or ``composite_panels``. Prefer those existing
    image artifacts so HTML reports can render inline figures even when local
    PDF preview tooling is unavailable.
    """
    candidates: List[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add(resolved)
    if resolved.suffix.lower() != ".pdf":
        return candidates

    base_name = _FIGURE_OUTPUT_ALIASES.get(resolved.name, resolved.name)
    base_stem = Path(base_name).stem
    search_roots = [
        resolved.parent,
        run_root / "output" / "figures" / "single_panels",
        run_root / "output" / "figures" / "composite_panels",
        run_root / "output" / "figures" / _safe_str(_safe_dict(_safe_dict(_read_json(run_root / "output" / "reports" / "current" / "figure_manifest.json")).get("run_id")) or ""),
    ]
    # The run-specific publication directory can be resolved directly from the
    # PDF parent without reading any additional manifest if it already follows
    # the standard output layout.
    if resolved.parent.name:
        search_roots.append(run_root / "output" / "figures" / resolved.parent.name)

    for root in search_roots:
        if not root:
            continue
        for suffix in (".svg", ".png", ".jpg", ".jpeg"):
            add(root / f"{base_stem}{suffix}")
    return candidates


def _ensure_phase0_prior_atlas(report_context: Dict[str, Any], run_root: Path) -> Optional[Path]:
    """Materialize a run-aligned Phase 0 atlas when Phase 3 omitted the task.

    Some legacy Phase 3 runs routed figures from ``phase0_state.json`` only;
    the state pointer could be absent while the authoritative Phase 0 output
    was still present under the run root.  This deterministic fallback uses
    only the archived candidate score table and never calls literature APIs or
    changes any biomarker decision.  A Phase 3-generated atlas is therefore
    replaced only with an equivalent run-aligned rendering, never with the
    shared/global latest figure.
    """
    root = Path(run_root).resolve()
    target = root / "output" / "figures" / "single_panels" / "phase0_prior_evidence_atlas.pdf"
    sources = _safe_dict(report_context.get("sources"))
    candidates: List[Path] = []
    source_value = sources.get("phase0_output")
    if isinstance(source_value, dict):
        source_value = source_value.get("selected_path") or source_value.get("path")
    if _safe_str(source_value).strip():
        candidates.append(_resolve_path(root, source_value))
    candidates.extend([
        root / "phase0" / "phase0_output_latest.json",
        root / "phase0" / "outputs" / "phase0_output_latest.json",
        root / "output" / "phase0" / "phase0_output_latest.json",
    ])
    payload: Dict[str, Any] = {}
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                payload = _safe_dict(json.loads(candidate.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload:
                break
    phase0_context = _safe_dict(report_context.get("phase0"))
    if not payload and phase0_context:
        payload = phase0_context
    records = [_safe_dict(item) for item in _safe_list(payload.get("candidate_scores_summary"))]
    if not records:
        records = [
            {
                **_safe_dict(item),
                "name": _safe_str(item.get("name") or item.get("id") or "Unlabelled prior"),
                "bio_prior_raw": item.get("confidence_score"),
                "selected_after_threshold": True,
            }
            for item in _safe_list(payload.get("confirmed_biomarkers"))
        ]
    if not records:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    def numeric(record: Dict[str, Any], key: str) -> Optional[float]:
        return _safe_float(record.get(key))

    records.sort(key=lambda item: numeric(item, "bio_prior_raw") if numeric(item, "bio_prior_raw") is not None else -1e9, reverse=True)
    names = [_safe_str(item.get("name") or item.get("metabolite_id") or "Unlabelled prior") for item in records]
    y = np.arange(len(records))
    threshold = numeric(_safe_dict(payload.get("selection_threshold")), "threshold")
    if threshold is None:
        threshold = numeric(_safe_dict(payload.get("normalization_stats")), "threshold")
    fig_height = max(4.8, min(12.0, 0.28 * len(records) + 2.4))
    fig, axes = plt.subplots(
        1, 5, figsize=(15.5, fig_height), sharey=True,
        gridspec_kw={"width_ratios": [4.8, 1.25, 1.25, 1.25, 1.25], "wspace": 0.08},
    )
    component_specs = [
        ("clinical_score", "Clinical evidence", "#E9B949"),
        ("specificity_score", "Disease specificity", "#F28E2B"),
        ("mechanistic_score", "Mechanistic plausibility", "#4DB6C2"),
        ("consistency", "Consistency", "#67BF7B"),
    ]
    selected = [bool(item.get("selected_after_threshold")) for item in records]
    total = [numeric(item, "bio_prior_raw") for item in records]
    total_values = [value if value is not None else 0.0 for value in total]
    axes[0].barh(y, total_values, color=["#2D9CDB" if flag else "#B8C0C8" for flag in selected], alpha=0.9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Total prior score", fontsize=8)
    axes[0].set_title("Prior score", fontsize=9, pad=8)
    if threshold is not None:
        axes[0].axvline(threshold, color="#8B5CF6", linestyle="--", linewidth=1)
        axes[0].text(threshold, 1.02, f"Threshold = {threshold:.2f}", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#6D4BC3")
    axes[0].grid(axis="x", color="#D9E0E7", linewidth=0.5)
    axes[0].set_axisbelow(True)
    for ax, (key, title, color) in zip(axes[1:], component_specs):
        values = [numeric(item, key) for item in records]
        ax.barh(y, [value if value is not None else 0 for value in values], color=color, alpha=0.9)
        for index, value in enumerate(values):
            if value is not None:
                ax.text(value + 0.03, index, f"{value:.1f}", va="center", fontsize=6, color="#536170")
        ax.set_title(title, fontsize=8, pad=8)
        ax.grid(axis="x", color="#D9E0E7", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=7)
    fig.suptitle("Prior Biomarker Evidence Atlas", fontsize=15, y=0.995)
    fig.text(0.01, 0.02, "Retained candidates are highlighted; blank component scores indicate that the archived Phase 0 record did not assess that domain.", fontsize=7, color="#667085")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="pdf", bbox_inches="tight", metadata={"Title": "Prior Biomarker Evidence Atlas", "Creator": "MetaboAgent Phase 4 fallback renderer"})
    plt.close(fig)
    return target if target.exists() else None


def _report_href(report_dir: Path, artifact: Dict[str, Any]) -> str:
    resolved = _safe_str(artifact.get("resolved_path")).strip()
    if not resolved or not bool(artifact.get("exists")):
        return ""
    return _portable_href(report_dir, Path(resolved))


def _claim_evidence_graph(bundle: Dict[str, Any], report_dir: Path) -> Dict[str, Any]:
    claims = []
    metric_registry = _safe_dict(bundle.get("metric_registry"))
    metric_artifact_requirements = {
        "internal-discrimination": "development_cv",
        "holdout-discrimination": "internal_holdout",
    }
    for claim_value in _safe_list(bundle.get("claim_registry")):
        claim = _safe_dict(claim_value)
        evidence = []
        for artifact_value in _safe_list(claim.get("evidence")):
            artifact = dict(_safe_dict(artifact_value))
            artifact["relative_href"] = _report_href(report_dir, artifact)
            artifact["artifact_name"] = Path(_safe_str(artifact.get("resolved_path") or artifact.get("path"))).name
            evidence.append(artifact)
        if not evidence:
            fallback = _claim_graph_fallback_artifact(report_dir, _safe_str(claim.get("claim_id")))
            if fallback:
                evidence.append(fallback)
        status = _safe_str(claim.get("status") or "NOT_AVAILABLE").upper()
        # A structural claim may be deterministically derived from a
        # report-local registry, but that is not the same as an upstream
        # archived artifact.  Never publish the contradictory pair
        # ``SUPPORTED + not archived``.
        if evidence and not any(_safe_dict(item).get("exists") for item in evidence):
            status = "NOT_AVAILABLE"
        elif evidence and status == "SUPPORTED" and not any(_safe_dict(item).get("path", "").startswith("output/") for item in evidence):
            status = "DERIVED"
        elif not evidence and status == "SUPPORTED":
            status = "NOT_AVAILABLE"
        scope = _safe_str(claim.get("evidence_scope")).upper()
        # AUC point estimates can be recovered from the archived Phase 2
        # winner-score artifact even when the prediction-level OOF/holdout
        # CSV was not packaged.  Mark that state as partial rather than
        # presenting it as fully direct-archived evaluation evidence.
        metric_key = metric_artifact_requirements.get(_safe_str(claim.get("claim_id")))
        metric = _safe_dict(metric_registry.get(metric_key)) if metric_key else {}
        prediction_artifact = _safe_dict(metric.get("prediction_artifact"))
        if metric_key and metric and not bool(prediction_artifact.get("exists")) and status == "SUPPORTED":
            status = "PARTIAL"
        if "SCENARIO" in scope or "UTILITY" in scope:
            strength = "SCENARIO-PROJECTION"
        elif "MODEL-ASSOCIATION" in scope or "LITERATURE" in scope:
            strength = "MODEL-INTERPRETIVE" if "MODEL" in scope else "LITERATURE-CONTEXT"
        elif metric_key and not bool(prediction_artifact.get("exists")):
            strength = "ARCHIVED-POINT-ESTIMATE"
        elif status == "SUPPORTED":
            strength = "DIRECT-ARCHIVED"
        elif status == "DERIVED":
            strength = "DERIVED-DETERMINISTIC"
        else:
            strength = "NOT-ASSESSABLE"
        claims.append({**claim, "status": status, "evidence_strength": strength, "evidence": evidence})
    return {"schema_version": "phase4.claim_evidence_graph.v1", "claims": claims, "policy": "Links resolve only to run-local archived artifacts; unavailable sources remain explicit."}


def _study_design_evidence(bundle: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    validation = _safe_dict(bundle.get("validation_scope"))
    return {
        "schema_version": "phase4.study_design_evidence.v1",
        "steps": [
            {"step": "Dataset provenance", "status": "SUPPORTED" if profile.get("matched") else "NOT_ASSESSED", "evidence": "dataset_profile.json", "summary": profile.get("study_id") or "run identity not matched to catalogue"},
            {"step": "Validation scope", "status": "SUPPORTED" if validation.get("performance_protocol") != "not recorded" else "NOT_ASSESSED", "evidence": "claim_evidence_graph.json", "summary": validation.get("evidence_status_label") or validation.get("evidence_status") or validation.get("headline_scope")},
            {"step": "Preprocessing", "status": "SUPPORTED", "evidence": "report_context_v2.json", "summary": "Archived Phase 1 preprocessing report"},
            {"step": "Selection and model development", "status": "SUPPORTED", "evidence": "claim_registry.json", "summary": "Archived Phase 1–2 feature/model selection"},
            {"step": "Validation and interpretation", "status": "SUPPORTED", "evidence": "mechanistic_evidence_registry.json", "summary": _safe_dict(bundle.get("mechanistic_evidence_registry")).get("interpretation_level")},
        ],
    }


def _feature_rows(phase2: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = [_safe_dict(item) for item in _safe_list(phase2.get("winner_feature_dictionary"))]
    by_token = {_safe_str(item.get("feature_token") or item.get("display_name")): item for item in records}
    rows = []
    for feature in _safe_list(phase2.get("winner_features")):
        token = _safe_str(feature)
        record = by_token.get(token, {})
        hmdb = _safe_list(record.get("hmdb_ids"))
        canonical = _safe_str(record.get("canonical_name") or record.get("preferred_report_name") or record.get("display_name") or token).strip() or token
        rows.append({
            "feature": canonical,
            "canonical_name": canonical,
            "feature_token": token,
            "raw_column_name": _safe_str(record.get("raw_column_name") or record.get("source_column") or token),
            "aliases": _safe_list(record.get("aliases") or record.get("synonyms")),
            "role": _safe_str(record.get("origin_subtype") or "Winner panel"),
            "hmdb_ids": hmdb,
            "chEBI_ids": _safe_list(record.get("chebi_ids") or record.get("chebi_id")),
            "inchikey": _safe_str(record.get("inchikey") or record.get("inchi_key")),
            "msi_identification_level": _safe_str(record.get("msi_identification_level") or record.get("identification_level")),
            "mz": record.get("mz") or record.get("m_z"),
            "adduct": _safe_str(record.get("adduct")),
            "retention_time": record.get("retention_time") or record.get("rt"),
            "ion_mode": _safe_str(record.get("ion_mode") or record.get("polarity")),
            "unit": _safe_str(record.get("unit") or record.get("measurement_unit")),
            "platform": _safe_str(record.get("platform") or record.get("assay_platform")),
            "hmdb_status": "Mapped" if hmdb else "Not mapped / not archived",
        })
    return rows


def _audit_payloads(run_root: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    audit_dir = run_root / "output" / "audit"
    summary = _read_json(audit_dir / "audit_summary.json")
    summary = _normalize_audit_summary(summary)
    payloads: Dict[str, Dict[str, Any]] = {}
    if audit_dir.exists():
        for path in sorted(audit_dir.glob("*.json")):
            payloads[path.name] = _read_json(path)
    return summary, payloads


def _normalize_audit_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Make audit non-results explicit instead of leaving ambiguous statuses.

    ``NOT_ASSESSED`` historically conflated an intentionally skipped module
    with a required audit artifact that was never produced.  We preserve the
    raw status and only promote it when the payload carries an explicit
    applicability/request flag; otherwise the reader-facing status is
    ``MISSING_REQUIRED`` with a deterministic explanation.  This does not
    turn a missing audit into PASS.
    """
    normalized = dict(_safe_dict(summary))
    domains = []
    for value in _safe_list(normalized.get("domains")):
        item = dict(_safe_dict(value))
        raw_status = _safe_str(item.get("status") or "NOT_ASSESSED").upper()
        item.setdefault("raw_status", raw_status)
        if raw_status == "NOT_ASSESSED":
            explicitly_not_applicable = (
                item.get("applicable") is False
                or item.get("not_applicable") is True
                or _safe_str(item.get("status_reason")).upper() in {"NOT_APPLICABLE", "NOT_APPLICABLE_TO_RUN"}
            )
            explicitly_not_requested = (
                item.get("requested") is False
                or item.get("enabled") is False
                or _safe_str(item.get("status_reason")).upper() in {"NOT_REQUESTED", "DISABLED"}
            )
            if explicitly_not_applicable:
                item["status"] = "NOT_APPLICABLE"
                item["status_reason"] = "This audit domain was explicitly marked not applicable to the run."
            elif explicitly_not_requested:
                item["status"] = "NOT_REQUESTED"
                item["status_reason"] = "This audit domain was not requested or was disabled for the run."
            else:
                item["status"] = "MISSING_REQUIRED"
                evidence_file = _safe_str(item.get("evidence_file")) or "the declared audit artifact"
                item["status_reason"] = (
                    f"No archived status was found for this required audit domain; {evidence_file} must be produced or explicitly waived."
                )
        domains.append(item)
    normalized["domains"] = domains
    normalized["status_semantics"] = {
        "PASS": "The audit domain ran and passed its deterministic checks.",
        "ASSESSED_WITH_WARNINGS": "The audit ran but reported warnings.",
        "NOT_APPLICABLE": "The domain was explicitly not applicable to this run.",
        "NOT_REQUESTED": "The domain was explicitly disabled or not requested.",
        "MISSING_REQUIRED": "The domain was expected but its archived audit result is missing.",
        "FAIL": "The audit ran and failed its deterministic checks.",
    }
    return normalized


def _validation_scope(
    collected: Dict[str, Any],
    report_context: Dict[str, Any],
    audit_payloads: Dict[str, Dict[str, Any]],
    run_root: Path,
) -> Dict[str, Any]:
    panel_scores = _safe_dict(_safe_dict(collected.get("phase1")).get("panel_scores"))
    protocol = _safe_str(panel_scores.get("performance_protocol")).strip()
    training_path = _safe_str(panel_scores.get("training_data_path")).strip()
    evaluation_path = _safe_str(panel_scores.get("evaluation_data_path")).strip()
    run_manifest = _safe_dict(audit_payloads.get("run_manifest.json"))
    declared = _safe_str(run_manifest.get("validation_cohort_type")).strip().lower()
    external_validation = _safe_dict(_safe_dict(collected.get("phase2")).get("external_validation"))
    external_path = _safe_str(external_validation.get("external_data_path")).strip()
    holdout_candidates = [
        run_root / "output" / "audit" / "evaluation_predictions_holdout.csv",
        run_root / "output" / "phase1" / "final" / "selected_features_holdout.csv",
        run_root / "output" / "phase1" / "final" / "selected_features_holdout_pool.csv",
    ]
    holdout_exists = any(path.exists() for path in holdout_candidates)
    oof_prediction_path = run_root / "output" / "audit" / "evaluation_predictions_oof.csv"
    holdout_prediction_path = run_root / "output" / "audit" / "evaluation_predictions_holdout.csv"
    training_partition_candidates = (
        run_root / "phase1" / "final" / "selected_features_train_pool.csv",
        run_root / "output" / "phase1" / "final" / "selected_features_train_pool.csv",
        run_root / "phase1" / "intermediate" / "latest" / "engineered" / "train_pool_pre_engineering.csv",
    )
    holdout_partition_candidates = (
        run_root / "phase1" / "final" / "selected_features_holdout_pool.csv",
        run_root / "output" / "phase1" / "final" / "selected_features_holdout_pool.csv",
        run_root / "phase1" / "intermediate" / "latest" / "engineered" / "holdout_pool_pre_engineering.csv",
    )
    training_partition_path = next((path for path in training_partition_candidates if path.is_file()), Path(""))
    holdout_partition_path = next((path for path in holdout_partition_candidates if path.is_file()), Path(""))
    oof_ids, oof_id_namespace, oof_raw_n, oof_id_column = _prediction_identity_set(
        oof_prediction_path, training_partition_path
    )
    holdout_ids, holdout_id_namespace, holdout_raw_n, holdout_id_column = _prediction_identity_set(
        holdout_prediction_path, holdout_partition_path
    )
    raw_oof_ids = _prediction_sample_ids(oof_prediction_path)
    raw_holdout_ids = _prediction_sample_ids(holdout_prediction_path)
    raw_sample_overlap = sorted(raw_oof_ids & raw_holdout_ids)
    sample_overlap = sorted(oof_ids & holdout_ids)
    protocol_audit = _safe_dict(audit_payloads.get("evaluation_protocol.json"))
    development_protocol = _safe_dict(protocol_audit.get("development_evaluation"))
    split_audit = _safe_dict(audit_payloads.get("data_split_audit.json"))
    split_order = _safe_str(split_audit.get("split_order") or split_audit.get("partition_order") or "not archived")
    subject_level = _safe_str(split_audit.get("subject_level_split") or split_audit.get("group_split") or "not archived")

    external_declared = declared in {"external", "independent_external", "multi_center_external"}
    if external_declared and external_path:
        headline = "EXTERNAL"
        evidence_status = "INDEPENDENT-EXTERNAL-VALIDATION"
        evidence_status_label = "Independent external validation declared"
        evidence_status_label_zh = "已声明独立外部验证"
    else:
        # A held-out subset is useful development evidence, but it is not an
        # independent external cohort and must never be promoted in the hero.
        headline = "DEVELOPMENT-ONLY"
        evidence_status = "DEVELOPMENT-ONLY-NO-INDEPENDENT-EXTERNAL-VALIDATION"
        evidence_status_label = "Development evaluation only; no independent external validation"
        evidence_status_label_zh = "仅开发期评估；无独立外部验证"

    same_training_evaluation = bool(training_path and evaluation_path) and os.path.abspath(training_path) == os.path.abspath(evaluation_path)
    warnings: List[str] = []
    if same_training_evaluation:
        warnings.append("Training and evaluation paths point to the same development modeling artifact, as expected for out-of-fold cross-validation; this does not by itself establish overlap with the internal holdout.")
    if external_validation and not external_declared:
        warnings.append("An external-validation-shaped artifact exists, but the run manifest does not declare an external cohort; it is not labelled EXTERNAL.")
    if not _safe_dict(audit_payloads.get("data_split_audit.json")):
        warnings.append("The deterministic data-split audit artifact is absent.")
    if holdout_exists and not external_declared:
        warnings.append("An internal holdout artifact is available; it is reported as internal development evidence, not as external validation.")
    if sample_overlap:
        warnings.append(f"The archived OOF and internal holdout prediction files share {len(sample_overlap)} sample IDs; this requires manual leakage review.")

    return {
        "headline_scope": headline,
        "evidence_status": evidence_status,
        "evidence_status_label": evidence_status_label,
        "evidence_status_label_zh": evidence_status_label_zh,
        "declared_validation_cohort_type": declared or "unspecified",
        "performance_protocol": protocol or "not recorded",
        "training_data": _artifact_ref(run_root, training_path),
        "evaluation_data": _artifact_ref(run_root, evaluation_path),
        "same_training_evaluation_path": same_training_evaluation,
        "same_training_evaluation_sha256": bool(
            _artifact_ref(run_root, training_path).get("sha256")
            and _artifact_ref(run_root, training_path).get("sha256") == _artifact_ref(run_root, evaluation_path).get("sha256")
        ),
        "development_oof_n": len(oof_ids) or None,
        "internal_holdout_n": len(holdout_ids) or None,
        "oof_holdout_sample_overlap": sample_overlap,
        "raw_prediction_sample_overlap_count": len(raw_sample_overlap),
        "prediction_id_namespace": {
            "oof": oof_id_namespace,
            "internal_holdout": holdout_id_namespace,
        },
        "prediction_id_column": {
            "oof": oof_id_column or "sample_id",
            "internal_holdout": holdout_id_column or "sample_id",
        },
        "prediction_ids_reconciled": (
            oof_id_namespace == "RECONCILED_PARTITION_ROW_INDEX"
            and holdout_id_namespace == "RECONCILED_PARTITION_ROW_INDEX"
        ),
        "prediction_id_reconciliation": {
            "oof_raw_n": oof_raw_n or None,
            "internal_holdout_raw_n": holdout_raw_n or None,
            "raw_overlap_n": len(raw_sample_overlap),
            "reconciled_overlap_n": len(sample_overlap),
            "policy": "Partition-local numeric prediction IDs are mapped to stable partition IDs only when row coverage and labels agree exactly; unresolved namespaces remain warnings.",
        },
        "split_order": split_order,
        "subject_level_split": subject_level,
        "preprocessing_fit_scope": _safe_str(development_protocol.get("preprocessing_fit_scope") or "not archived"),
        "feature_selection_fit_scope": _safe_str(development_protocol.get("feature_selection_fit_scope") or "not archived"),
        "hyperparameter_tuning_fit_scope": _safe_str(development_protocol.get("hyperparameter_tuning_fit_scope") or "not archived"),
        "holdout_available": holdout_exists,
        "external_artifact_available": bool(external_validation),
        "external_data": _artifact_ref(run_root, external_path),
        "external_declared": external_declared,
        "warnings": warnings,
    }


def _figure_status(report_context: Dict[str, Any], run_root: Path) -> Dict[str, Any]:
    phase3 = _safe_dict(report_context.get("phase3"))
    task_records = _safe_list(phase3.get("task_records"))
    completed = [item for item in task_records if _safe_dict(item).get("status") == "completed"]
    failed = [item for item in task_records if _safe_dict(item).get("status") == "failed"]
    skipped = [item for item in task_records if _safe_dict(item).get("status") == "skipped"]
    broken: List[Dict[str, str]] = []
    for item in task_records:
        record = _safe_dict(item)
        for output in _safe_list(record.get("outputs")):
            path = _resolve_figure_path(run_root, output)
            if _safe_str(output).strip() and not path.exists():
                broken.append({"task": _safe_str(record.get("task")), "path": _safe_str(output)})
    return {
        "required_task_count": len(task_records),
        "completed_task_count": len(completed),
        "failed_task_count": len(failed),
        "skipped_task_count": len(skipped),
        "failed_tasks": [
            {
                "task": _safe_str(_safe_dict(item).get("task")),
                "error": _safe_str(_safe_dict(item).get("error")),
                "skip_reason": _safe_str(_safe_dict(item).get("skip_reason")),
            }
            for item in failed + skipped
        ],
        "broken_declared_outputs": broken,
    }


def _holdout_auc(report_context: Dict[str, Any]) -> Optional[float]:
    for record in _safe_list(_safe_dict(report_context.get("phase3")).get("task_records")):
        item = _safe_dict(record)
        if item.get("task") == "plot_final_holdout_roc":
            return _safe_float(_safe_dict(item.get("renderer_metadata")).get("pooled_auc"))
    return None


def _prediction_sample_ids(path: Path) -> set[str]:
    if not path.exists() or not path.is_file():
        return set()
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            return {str(row.get("sample_id")).strip() for row in rows if str(row.get("sample_id", "")).strip()}
    except (OSError, UnicodeError, csv.Error):
        return set()


def _prediction_identity_set(prediction_path: Path, partition_path: Path) -> Tuple[set[str], str, int, str]:
    """Resolve prediction IDs into the stable partition namespace when possible.

    Some archived evaluation exports store ``sample_id`` as a partition-local
    row index (for example 0..349 for OOF and 0..87 for holdout). Comparing
    those two local namespaces directly manufactures an apparent overlap. If
    the corresponding partition table is archived, has a stable ID column,
    covers the complete row-index range, and its labels agree row-by-row with
    the prediction export, reconcile to the stable IDs before testing overlap.
    Otherwise retain the raw IDs and leave the warning active.
    """
    if not prediction_path.exists() or not partition_path.exists():
        return _prediction_sample_ids(prediction_path), "UNRESOLVED_ID_NAMESPACE", 0, ""
    try:
        with prediction_path.open(newline="", encoding="utf-8-sig") as handle:
            prediction_rows = list(csv.DictReader(handle))
        with partition_path.open(newline="", encoding="utf-8-sig") as handle:
            partition_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error):
        return _prediction_sample_ids(prediction_path), "UNRESOLVED_ID_NAMESPACE", 0, ""

    raw_ids = {
        _safe_str(row.get("sample_id")).strip()
        for row in prediction_rows
        if _safe_str(row.get("sample_id")).strip()
    }
    if not raw_ids:
        return set(), "NOT_ASSESSED", 0, ""

    id_column = ""
    for candidate in ("sample_id", "subject_id", "patient_id", "id"):
        if partition_rows and candidate in partition_rows[0]:
            values = [_safe_str(row.get(candidate)).strip() for row in partition_rows]
            if values and all(values) and len(values) == len(set(values)):
                id_column = candidate
                break
    if not id_column:
        return raw_ids, "UNRESOLVED_ID_NAMESPACE", len(raw_ids), ""

    stable_ids = [_safe_str(row.get(id_column)).strip() for row in partition_rows]
    stable_id_set = set(stable_ids)
    if raw_ids <= stable_id_set:
        return raw_ids, "STABLE_ID_NAMESPACE", len(raw_ids), id_column

    index_values: List[int] = []
    for row in prediction_rows:
        raw = _safe_str(row.get("sample_id")).strip()
        if not re.fullmatch(r"\d+", raw):
            return raw_ids, "UNRESOLVED_ID_NAMESPACE", len(raw_ids), id_column
        index_values.append(int(raw))
    expected_indices = set(range(len(partition_rows)))
    if len(prediction_rows) != len(partition_rows) or set(index_values) != expected_indices:
        return raw_ids, "UNRESOLVED_ID_NAMESPACE", len(raw_ids), id_column

    # Validate that the row-index mapping is not merely positional by chance.
    # The archived partition labels provide a lightweight, deterministic check.
    group_column = next(
        (candidate for candidate in ("group", "label", "target", "y", "class") if candidate in partition_rows[0]),
        "",
    )
    if group_column:
        for row in prediction_rows:
            index = int(_safe_str(row.get("sample_id")))
            observed = _safe_str(row.get("y_true")).strip()
            expected = _safe_str(partition_rows[index].get(group_column)).strip()
            if observed and expected and observed != expected:
                return raw_ids, "UNRESOLVED_ID_NAMESPACE", len(raw_ids), id_column

    return {stable_ids[index] for index in index_values}, "RECONCILED_PARTITION_ROW_INDEX", len(raw_ids), id_column


def _internal_auc(collected: Dict[str, Any], report_context: Dict[str, Any]) -> Optional[float]:
    phase2_result = _safe_dict(_safe_dict(collected.get("phase2")).get("result"))
    final_result = _safe_dict(phase2_result.get("final_result"))
    comprehensive = _safe_dict(phase2_result.get("comprehensive_metrics"))
    scores = _safe_dict(phase2_result.get("scores"))
    return _first_numeric(
        phase2_result.get("roc_auc"),
        comprehensive.get("roc_auc"),
        final_result.get("roc_auc"),
        final_result.get("perf"),
        scores.get("roc_auc"),
        scores.get("f_perf"),
        _safe_dict(_safe_dict(report_context.get("phase2")).get("winner_scores")).get("f_perf"),
    )


def _first_key_numeric(value: Any, keys: set[str]) -> Optional[float]:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                found = _safe_float(item)
                if found is not None:
                    return found
            found = _first_key_numeric(item, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_key_numeric(item, keys)
            if found is not None:
                return found
    return None


def _evaluation_metrics(
    collected: Dict[str, Any],
    report_context: Dict[str, Any],
    audit_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
    run_root: Optional[Path] = None,
) -> Dict[str, Any]:
    audit_payloads = audit_payloads or {}
    clinical = _safe_dict(_safe_dict(report_context.get("phase2")).get("clinical_utility"))
    calibration, calibration_source = _calibration_payload(clinical)
    root = (run_root or Path(_safe_str(report_context.get("run_root")) or ".")).resolve()
    registry = _metric_registry(root, collected, report_context, audit_payloads, calibration, calibration_source)
    return {
        "development_cv": {
            **_safe_dict(registry.get("development_cv")),
            "fold_metrics": _safe_list(_safe_dict(audit_payloads.get("performance_uncertainty.json")).get("development_oof", {}).get("fold_metrics")),
        },
        "internal_holdout": _safe_dict(registry.get("internal_holdout")),
        "calibration": _safe_dict(registry.get("calibration")),
    }


def _first_key_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                return item
            found = _first_key_value(item, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_key_value(item, keys)
            if found is not None:
                return found
    return None


def _claim(
    claim_id: str,
    statement: str,
    value: Any,
    scope: str,
    evidence: Iterable[Dict[str, Any]],
    status: str = "SUPPORTED",
    boundary: str = "",
) -> Dict[str, Any]:
    evidence_list = [item for item in evidence if isinstance(item, dict)]
    if not any(item.get("exists") for item in evidence_list):
        status = "NOT_ASSESSABLE"
    return {
        "claim_id": claim_id,
        "statement": statement,
        "value": value,
        "evidence_scope": scope,
        "status": status,
        "boundary": boundary,
        "evidence": evidence_list,
    }


def build_evidence_bundle(
    report_context: Dict[str, Any],
    collected: Dict[str, Any],
    run_root: str,
) -> Dict[str, Any]:
    """Build v2 evidence, terminology, limitation and quality registries."""

    root = Path(run_root).resolve()
    audit_summary, audits = _audit_payloads(root)
    validation = _validation_scope(collected, report_context, audits, root)
    figure_status = _figure_status(report_context, root)
    phase0 = _safe_dict(report_context.get("phase0"))
    phase1 = _safe_dict(report_context.get("phase1"))
    phase2 = _safe_dict(report_context.get("phase2"))
    clinical = _safe_dict(phase2.get("clinical_utility"))
    calibration, calibration_source = _calibration_payload(clinical)
    mechanistic_registry = _mechanistic_evidence_registry(report_context)
    dataset_profile = _dataset_profile(report_context)
    sources = _safe_dict(report_context.get("sources"))
    metric_registry = _metric_registry(root, collected, report_context, audits, calibration, calibration_source)
    internal_auc = _safe_float(_safe_dict(metric_registry.get("development_cv")).get("auc"))
    holdout_auc = _safe_float(_safe_dict(metric_registry.get("internal_holdout")).get("auc"))
    winner_features = _safe_list(phase2.get("winner_features"))
    prior_counts = _phase0_prior_counts(phase0)
    prior_count = prior_counts["confirmed_biomarkers"]
    prior_evidence_refs = [
        _artifact_ref(root, sources.get("phase0_output")),
        _artifact_ref(root, "phase0/phase0_output_latest.json"),
        _artifact_ref(root, "output/phase0/phase0_output_latest.json"),
    ]
    calibration_registry = _safe_dict(metric_registry.get("calibration"))
    calibration_scope = _display_evidence_scope(calibration_registry.get("scope"))

    claims = [
        _claim(
            "winner-panel-size",
            f"The selected panel contains {len(winner_features)} features.",
            len(winner_features),
            "MODEL-DEVELOPMENT",
            [_artifact_ref(root, sources.get("phase2_result"))],
        ),
        _claim(
            "internal-discrimination",
            f"The archived internal evaluation AUC is {_fmt(internal_auc)}.",
            internal_auc,
            "INTERNAL-CV",
            [_artifact_ref(root, "output/audit/performance_uncertainty.json"), _artifact_ref(root, "output/audit/evaluation_predictions_oof.csv"), _artifact_ref(root, sources.get("phase2_result"))],
            boundary="Internal resampling evidence does not establish transportability.",
        ),
        _claim(
            "holdout-discrimination",
            f"The archived holdout AUC is {_fmt(holdout_auc)}.",
            holdout_auc,
            "INTERNAL-HOLDOUT",
            [_artifact_ref(root, "output/audit/performance_uncertainty.json"), _artifact_ref(root, "output/audit/evaluation_predictions_holdout.csv"), _artifact_ref(root, "output/phase1/final/selected_features_holdout.csv")],
            boundary="This is an internal held-out subset from the development workflow, not independent external validation.",
        ),
        _claim(
            "calibration",
            (
                f"The canonical {calibration_registry.get('source')} calibration Brier score is {_fmt(calibration_registry.get('brier_score'))}."
                if _safe_float(calibration_registry.get('brier_score')) is not None
                else "No archived Brier score is available for the selected calibration scope."
            ),
            calibration_registry.get("brier_score"),
            calibration_scope,
            _safe_list(calibration_registry.get("source_artifacts")),
            boundary=(
                f"Assessment scope: {_safe_str(calibration_registry.get('scope') or 'not recorded')}. "
                "Calibration remains cohort-specific until independently validated and, where needed, recalibrated."
            ),
        ),
        _claim(
            "prior-evidence",
            (
                f"Phase 0 retained {prior_count} confirmed prior biomarkers."
                if prior_count
                else "No candidate passed the archived Phase 0 confirmed-biomarker criterion."
            ),
            prior_count,
            "LITERATURE-SUPPORTED",
            prior_evidence_refs,
            boundary=(
                "Prior support is literature-derived and does not establish causality, independent validation or target-population utility."
                if prior_count
                else "Absence of a confirmed prior is reported as a negative workflow result, not evidence that no biological association exists."
            ),
        ),
    ]

    limitations: List[Dict[str, Any]] = []
    for warning in validation.get("warnings", []):
        limitations.append({"category": "validation", "severity": "high", "message": warning})
    if not prior_count:
        limitations.append({
            "category": "prior_evidence",
            "severity": "information",
            "message": "No candidate satisfied the Phase 0 confirmed-prior criterion; the prior-evidence atlas documents screened evidence without a retained prior set.",
        })
    for task in figure_status.get("failed_tasks", []):
        limitations.append({
            "category": "figure_generation",
            "severity": "medium",
            "message": f"{task.get('task')}: {task.get('error') or task.get('skip_reason') or 'not completed'}",
        })
    for domain in _safe_list(audit_summary.get("domains")):
        item = _safe_dict(domain)
        if item.get("status") != "PASS":
            limitations.append({
                "category": "audit",
                "severity": "high" if item.get("status") == "FAIL" else "medium",
                "message": f"Audit domain {item.get('domain')} is {item.get('status', 'NOT_ASSESSED')}.",
            })
    preprocessing = _safe_dict(phase1.get("preprocessing_report"))
    for warning in _safe_list(preprocessing.get("warnings")):
        limitations.append({"category": "preprocessing", "severity": "medium", "message": _safe_str(warning)})

    terminology = {
        "MetaboAgent": "The complete agentic metabolomics analysis system.",
        "Evidence-to-Decision Report": "The read-only Phase 4 scientific reporting and audit layer.",
        "INTERNAL-CV": "Performance estimated using internal cross-validation or out-of-fold predictions.",
        "INTERNAL-HOLDOUT": "Performance on an archived held-out subset from the development workflow; it is not independent external validation.",
        "EXTERNAL": "Performance on an independently declared external cohort.",
        "confirmed biomarker": "A Phase 0 candidate satisfying the archived evidence-selection criterion.",
        "winner panel": "The feature panel selected by the Phase 2 multi-objective search.",
    }

    broken_claims = [item["claim_id"] for item in claims if item.get("status") != "SUPPORTED"]
    failed_figures = figure_status.get("failed_task_count", 0)
    audit_not_pass = [
        _safe_dict(item).get("domain")
        for item in _safe_list(audit_summary.get("domains"))
        if _safe_dict(item).get("status") != "PASS"
    ]
    audit_missing_required = [
        _safe_dict(item).get("domain")
        for item in _safe_list(audit_summary.get("domains"))
        if _safe_str(_safe_dict(item).get("status")).upper() == "MISSING_REQUIRED"
    ]
    quality_status = "PASS"
    if broken_claims or failed_figures or audit_not_pass:
        quality_status = "PARTIAL"
    if figure_status.get("broken_declared_outputs"):
        quality_status = "FAIL"
    if any(_safe_dict(item).get("status") == "FAIL" for item in _safe_list(audit_summary.get("domains"))):
        quality_status = "FAIL"

    quality = {
        "schema_version": "phase4.report_quality.v2",
        "status": quality_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "claim_evidence_complete": not broken_claims,
            "claim_ids_without_available_evidence": broken_claims,
            "figure_tasks_complete": not failed_figures,
            "failed_figure_task_count": failed_figures,
            "broken_declared_figure_outputs": figure_status.get("broken_declared_outputs", []),
            "all_audit_domains_pass": not audit_not_pass,
            "audit_domains_not_pass": audit_not_pass,
            "audit_domains_missing_required": audit_missing_required,
            "external_scope_guard_pass": not (
                validation.get("headline_scope") == "EXTERNAL" and not validation.get("external_declared")
            ),
        },
        "policy": "A PARTIAL report remains deliverable but must expose every unavailable or unassessed evidence domain.",
    }

    bundle = {
        "schema_version": "phase4.evidence_bundle.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "validation_scope": validation,
        "claim_registry": claims,
        "limitations_registry": limitations,
        "terminology_ledger": terminology,
        "figure_status": figure_status,
        "calibration": {"source": calibration_source, "metrics": calibration},
        "metric_registry": metric_registry,
        "evaluation_metrics": _evaluation_metrics(collected, report_context, audits, root),
        "mechanistic_evidence_registry": mechanistic_registry,
        "dataset_profile": dataset_profile,
        "audit_summary": audit_summary,
        "report_quality": quality,
    }
    bundle["enhancements"] = build_report_enhancements(root, report_context, collected, bundle)
    # Build one deterministic narrative registry after all enhancement
    # registries are available. HTML and PDF consume these same blocks at
    # different lengths; neither renderer asks an LLM to recompute facts.
    bundle["scientific_narrative_registry"] = _scientific_narrative_registry(report_context, bundle)
    narrative_blocks = _safe_dict(bundle["scientific_narrative_registry"]).get("blocks", {})
    narrative_required = {
        "study_context", "discovery_summary", "performance_summary",
        "clinical_summary", "biology_summary", "conclusion_summary",
    }
    narrative_missing = sorted(key for key in narrative_required if key not in narrative_blocks)
    claim_validation = _safe_dict(_safe_dict(bundle.get("enhancements")).get("statistical_claim_validation"))
    identity_registry = _safe_dict(_safe_dict(bundle.get("enhancements")).get("feature_identity_registry"))
    quality_checks = _safe_dict(_safe_dict(bundle.get("report_quality")).get("checks"))
    quality_checks["statistical_claim_validation"] = {
        "pass": claim_validation.get("status") == "PASS",
        "failed_checks": _safe_list(claim_validation.get("failed_checks")),
    }
    quality_checks["feature_identity_resolution"] = {
        "pass": not _safe_list(identity_registry.get("unresolved_reader_features")),
        "unresolved_reader_features": _safe_list(identity_registry.get("unresolved_reader_features")),
    }
    quality_checks["narrative_quality"] = {
        "pass": not narrative_missing,
        "missing_blocks": narrative_missing,
        "required_blocks": sorted(narrative_required),
        "policy": "Every reader-facing scientific section must expose a finding, interpretation and boundary from the canonical narrative registry.",
    }
    bundle["report_quality"]["checks"] = quality_checks
    if claim_validation.get("status") == "FAIL" or _safe_list(identity_registry.get("unresolved_reader_features")):
        bundle["report_quality"]["status"] = "FAIL"
    split_audit = _safe_dict(_safe_dict(bundle.get("enhancements")).get("data_split_audit"))
    if split_audit:
        bundle["validation_scope"]["warnings"] = [
            warning for warning in _safe_list(bundle["validation_scope"].get("warnings"))
            if warning != "The deterministic data-split audit artifact is absent."
        ]
        bundle["validation_scope"].update({
            "same_training_evaluation_path": split_audit.get("same_training_evaluation_path"),
            "same_training_evaluation_sha256": split_audit.get("same_training_evaluation_sha256"),
            "development_oof_n": split_audit.get("development_sample_count"),
            "internal_holdout_n": split_audit.get("internal_holdout_sample_count"),
            "oof_holdout_sample_overlap": split_audit.get("sample_id_overlap", []),
            "split_order": split_audit.get("split_order"),
            "subject_level_split": split_audit.get("subject_level_split"),
            "preprocessing_fit_scope": split_audit.get("preprocessing_fit_scope"),
            "feature_selection_fit_scope": split_audit.get("feature_selection_fit_scope"),
            "hyperparameter_tuning_fit_scope": split_audit.get("hyperparameter_tuning_fit_scope"),
        })
        if split_audit.get("sample_id_overlap"):
            bundle["validation_scope"].setdefault("warnings", []).append(
                f"The archived development and internal holdout artifacts share {len(split_audit.get('sample_id_overlap'))} sample IDs."
            )
    return bundle


def write_evidence_bundle(bundle: Dict[str, Any], report_context: Dict[str, Any], report_dir: str) -> Dict[str, str]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    run_root = Path(_safe_str(bundle.get("run_root"))) if _safe_str(bundle.get("run_root")) else None
    contract_package = None
    if run_root:
        _ensure_phase0_prior_atlas(report_context, run_root)
        contract_package = write_contract_package(report_context, bundle, str(run_root), str(report_path))
        bundle["contract_manifests"] = contract_package
        contract_quality = _safe_dict(bundle.get("report_quality"))
        contract_validation = _safe_dict(contract_package.get("validation"))
        contract_quality.setdefault("checks", {})["contract_manifests"] = contract_validation
        contract_quality["contract_manifest_paths"] = contract_package.get("paths", {})
        if not bool(contract_validation.get("valid")):
            contract_quality["status"] = "FAIL"
        bundle["report_quality"] = contract_quality
    split_audit_payload = _safe_dict(_safe_dict(bundle.get("enhancements")).get("data_split_audit"))
    if run_root and split_audit_payload:
        _write_json(run_root / "output" / "audit" / "data_split_audit.json", split_audit_payload)
    context_v2 = dict(report_context)
    context_v2["schema_version"] = "phase4.report_context.v2"
    context_v2["evidence"] = {
        "validation_scope": bundle.get("validation_scope", {}),
        "metric_registry": bundle.get("metric_registry", {}),
        "evaluation_metrics": bundle.get("evaluation_metrics", {}),
        "figure_status": bundle.get("figure_status", {}),
        "report_quality": bundle.get("report_quality", {}),
        "contract_manifests": {
            "validation": _safe_dict(_safe_dict(bundle.get("contract_manifests")).get("validation")),
            "paths": _safe_dict(_safe_dict(bundle.get("contract_manifests")).get("paths")),
        },
        "enhancements": {
            key: {"status": _safe_dict(value).get("status", "NOT_ASSESSED")}
            for key, value in _safe_dict(bundle.get("enhancements")).items()
        },
        "scientific_narrative_registry": bundle.get("scientific_narrative_registry", {}),
    }
    graph = _claim_evidence_graph(bundle, report_path)
    def _sync_quality_with_graph(graph_payload: Dict[str, Any]) -> None:
        """Make report_quality reflect the final claim graph, not its pre-graph seed."""
        quality = _safe_dict(bundle.get("report_quality"))
        claims = _safe_list(graph_payload.get("claims"))
        partial_claims = [
            _safe_str(item.get("claim_id"))
            for item in claims
            if _safe_str(item.get("status")).upper() == "PARTIAL"
        ]
        unavailable_claims = [
            _safe_str(item.get("claim_id"))
            for item in claims
            if _safe_str(item.get("status")).upper() == "NOT_AVAILABLE"
        ]
        derived_claims = [
            _safe_str(item.get("claim_id"))
            for item in claims
            if _safe_str(item.get("status")).upper() == "DERIVED"
        ]
        checks = _safe_dict(quality.get("checks"))
        checks["claim_evidence_complete"] = not partial_claims and not unavailable_claims
        checks["claim_ids_without_available_evidence"] = unavailable_claims
        checks["claim_ids_with_partial_evidence"] = partial_claims
        checks["claim_ids_with_derived_evidence"] = derived_claims
        quality["checks"] = checks
        if (partial_claims or unavailable_claims) and quality.get("status") == "PASS":
            quality["status"] = "PARTIAL"
        bundle["report_quality"] = quality
        context_v2["evidence"]["report_quality"] = quality
    _sync_quality_with_graph(graph)
    ladder = _study_design_evidence(bundle, _safe_dict(bundle.get("dataset_profile")))
    enhancements = _safe_dict(bundle.get("enhancements"))
    output = {
        "report_context_v2": _write_json(report_path / "report_context_v2.json", context_v2),
        "claim_registry": _write_json(report_path / "claim_registry.json", {
            "schema_version": "phase4.claim_registry.v2",
            "claims": bundle.get("claim_registry", []),
        }),
        "claim_evidence_graph": _write_json(report_path / "claim_evidence_graph.json", graph),
        "metric_registry": _write_json(report_path / "metric_registry.json", _safe_dict(bundle.get("metric_registry"))),
        "data_split_audit": _write_json(report_path / "data_split_audit.json", _safe_dict(_safe_dict(bundle.get("enhancements")).get("data_split_audit"))),
        "study_design_evidence": _write_json(report_path / "study_design_evidence.json", ladder),
        "dataset_profile": _write_json(report_path / "dataset_profile.json", _safe_dict(bundle.get("dataset_profile"))),
        "executive_fact_card": _write_json(report_path / "executive_fact_card.json", {
            "schema_version": "phase4.executive_fact_card.v1",
            "validation_scope": _safe_dict(bundle.get("validation_scope")).get("headline_scope"),
            "claims": bundle.get("claim_registry", []),
            "winner_features": _safe_list(_safe_dict(report_context.get("phase2")).get("winner_features")),
            "policy": "Default executive prose is deterministic. Optional LLM use may polish language but must not change these archived facts.",
        }),
        "terminology_ledger": _write_json(report_path / "terminology_ledger.json", {
            "schema_version": "phase4.terminology_ledger.v1",
            "terms": bundle.get("terminology_ledger", {}),
        }),
        "limitations_registry": _write_json(report_path / "limitations_registry.json", {
            "schema_version": "phase4.limitations_registry.v1",
            "limitations": bundle.get("limitations_registry", []),
        }),
        "report_quality": _write_json(report_path / "report_quality.json", bundle.get("report_quality", {})),
        "phase_flow_manifest": _write_json(report_path / "phase_flow_manifest.json", _safe_dict(_safe_dict(bundle.get("contract_manifests")).get("phase_flow_manifest"))),
        "figure_manifest": _write_json(report_path / "figure_manifest.json", _safe_dict(_safe_dict(bundle.get("contract_manifests")).get("figure_manifest"))),
        "report_template": _write_json(report_path / "report_template.json", _safe_dict(_safe_dict(bundle.get("contract_manifests")).get("report_template"))),
        "phase_flow_figure": _safe_str(_safe_dict(bundle.get("contract_manifests")).get("paths", {}).get("phase_flow_figure")),
        "mechanistic_evidence_registry": _write_json(
            report_path / "mechanistic_evidence_registry.json",
            bundle.get("mechanistic_evidence_registry", {}),
        ),
        "cohort_qc_snapshot": _write_json(report_path / "cohort_qc_snapshot.json", _safe_dict(_safe_dict(bundle.get("enhancements")).get("cohort_qc_snapshot"))),
        "feature_selection_evidence": _write_json(report_path / "feature_selection_evidence.json", _safe_dict(_safe_dict(bundle.get("enhancements")).get("feature_selection_evidence"))),
        "training_association_snapshot": _write_json(report_path / "training_association_snapshot.json", _safe_dict(_safe_dict(bundle.get("enhancements")).get("training_association_snapshot"))),
        "model_card": _write_json(report_path / "model_card.json", _safe_dict(_safe_dict(bundle.get("enhancements")).get("model_card"))),
        "visual_evidence_registry": _write_json(report_path / "visual_evidence_registry.json", _safe_dict(_safe_dict(bundle.get("enhancements")).get("visual_evidence_registry"))),
        "reproducibility_manifest": _write_json(report_path / "reproducibility_manifest.json", _safe_dict(enhancements.get("reproducibility_manifest"))),
        "feature_identity_registry": _write_json(report_path / "feature_identity_registry.json", _safe_dict(enhancements.get("feature_identity_registry"))),
        "statistical_claim_validation": _write_json(report_path / "statistical_claim_validation.json", _safe_dict(enhancements.get("statistical_claim_validation"))),
        "prevalence_scenarios": _write_json(report_path / "prevalence_scenarios.json", _safe_dict(enhancements.get("prevalence_scenarios"))),
        "evidence_maturity": _write_json(report_path / "evidence_maturity.json", _safe_dict(enhancements.get("evidence_maturity"))),
        "reporting_readiness": _write_json(report_path / "reporting_readiness.json", _safe_dict(enhancements.get("reporting_readiness"))),
        "upstream_evidence_requirements": _write_json(report_path / "upstream_evidence_requirements.json", _safe_dict(enhancements.get("upstream_evidence_requirements"))),
        "scientific_narrative_registry": _write_json(
            report_path / "scientific_narrative_registry.json",
            _safe_dict(bundle.get("scientific_narrative_registry")),
        ),
    }
    # The first graph pass occurs before the enhancement registries above are
    # written.  Rebuild it once all report-local registries exist so structural
    # contract claims carry the same portable evidence links as the HTML graph.
    graph = _claim_evidence_graph(bundle, report_path)
    _sync_quality_with_graph(graph)
    output["claim_evidence_graph"] = _write_json(
        report_path / "claim_evidence_graph.json",
        graph,
    )
    # Keep the machine-readable claim registry synchronized with the graph;
    # this prevents structural contract claims from appearing evidence-less
    # in one registry while being linked in the reader-facing appendix.
    bundle["claim_registry"] = graph.get("claims", [])
    output["claim_registry"] = _write_json(
        report_path / "claim_registry.json",
        {"schema_version": "phase4.claim_registry.v2", "claims": bundle.get("claim_registry", [])},
    )
    # The second graph pass can discover portable links written by the first
    # pass; persist the synchronized quality/context sidecars after it.
    output["report_quality"] = _write_json(
        report_path / "report_quality.json", bundle.get("report_quality", {})
    )
    output["report_context_v2"] = _write_json(
        report_path / "report_context_v2.json", context_v2
    )
    # CSV sidecars keep the principal audit tables usable outside the HTML report.
    qc_profiles = _safe_list(_safe_dict(enhancements.get("cohort_qc_snapshot")).get("profiles"))
    output["cohort_qc_snapshot_csv"] = _write_csv(report_path / "cohort_qc_snapshot.csv", qc_profiles)
    selection_records = _safe_list(_safe_dict(enhancements.get("feature_selection_evidence")).get("winner_records"))
    output["feature_selection_evidence_csv"] = _write_csv(report_path / "feature_selection_evidence.csv", selection_records)
    association_rows = _safe_list(_safe_dict(enhancements.get("training_association_snapshot")).get("rows"))
    output["training_association_snapshot_csv"] = _write_csv(report_path / "training_association_snapshot.csv", association_rows)
    prevalence_rows = _safe_list(_safe_dict(enhancements.get("prevalence_scenarios")).get("rows"))
    output["prevalence_scenarios_csv"] = _write_csv(report_path / "prevalence_scenarios.csv", prevalence_rows)
    return output


def _write_csv(path: Path, rows: List[Any]) -> str:
    """Write a deterministic, flat CSV sidecar for an audit table."""
    normalized = [_safe_dict(row) for row in rows]
    columns: List[str] = []
    for row in normalized:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["status"])
        writer.writeheader()
        for row in normalized:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return str(path)


def _narrative(llm_sections: Dict[str, Any], key: str) -> str:
    payload = _safe_dict(_safe_dict(llm_sections.get("sections")).get(key))
    return _safe_str(payload.get("content")).strip()


def _canonicalize_reader_text(text: str, metric_registry: Dict[str, Any], feature_rows: Optional[List[Dict[str, Any]]] = None) -> str:
    """Remove reader-facing placeholders and normalize feature aliases."""

    value = _safe_str(text)
    value = re.sub(r"\b\d{4,}-fold\b", "out-of-fold", value, flags=re.IGNORECASE)
    value = re.sub(r"\bNone[- ]fold\b", "out-of-fold", value, flags=re.IGNORECASE)
    value = re.sub(r"\bFig\.\s+and\s+Fig\.", "Fig.", value, flags=re.IGNORECASE)
    value = re.sub(r"\bFig\.\s*(?=[,.;:)])", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bN/A\b", "Not available", value, flags=re.IGNORECASE)
    value = value.replace("Caption evidence was not registered.", "Figure caption not archived.")
    value = re.sub(
        r"\bsupports?\s+(?:a\s+)?default\s+(?:decision\s+)?threshold\b",
        "shows an illustrative operating threshold",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\brecommended clinical threshold\b", "illustrative operating threshold", value, flags=re.IGNORECASE)
    value = re.sub(r"\bdefault clinical threshold\b", "illustrative operating threshold", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+;", ";", value)

    calibration = _safe_dict(metric_registry.get("calibration"))
    brier = _safe_float(calibration.get("brier_score"))
    if brier is not None and re.search(r"\bbrier\s+score\b", value, flags=re.IGNORECASE):
        calibration_label = _safe_str(calibration.get("source") or "canonical")
        value = re.sub(
            r"[^.!?]*\bbrier\s+score\b[^.!?]*[.!?]",
            f"The canonical {calibration_label} calibration Brier score was {brier:.3f}. ",
            value,
            flags=re.IGNORECASE,
        )
    # Some legacy LLM sections use the shorter form "Brier not archived".
    # Replace only that token so slope/intercept availability remains visible
    # while every Brier mention agrees with the canonical metric registry.
    if brier is not None:
        value = re.sub(
            r"\bbrier(?:\s+score)?\s+not\s+archived\b",
            f"Brier score {brier:.3f}",
            value,
            flags=re.IGNORECASE,
        )

    for row in sorted(feature_rows or [], key=lambda item: len(_safe_str(item.get("canonical_name"))), reverse=True):
        canonical = _safe_str(row.get("canonical_name")).strip()
        aliases = [_safe_str(row.get("feature_token")), _safe_str(row.get("raw_column_name")), *_safe_list(row.get("aliases"))]
        if not canonical:
            continue
        for alias in sorted({item.strip() for item in aliases if item and item.strip() and item.strip() != canonical}, key=len, reverse=True):
            value = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", canonical, value)
    return value


def _canonicalize_llm_sections(llm_sections: Dict[str, Any], metric_registry: Dict[str, Any], feature_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned = dict(llm_sections or {})
    sections = {}
    for key, payload_value in _safe_dict(cleaned.get("sections")).items():
        payload = dict(_safe_dict(payload_value))
        content = _canonicalize_reader_text(payload.get("content"), metric_registry, feature_rows)
        # Older archived llm_sections may contain a speculative Intended Use
        # subsection.  Intended use is a study-design decision, not something
        # Phase 4 may infer from performance artifacts, so suppress it during
        # deterministic re-rendering as well as in new writer output.
        if key == "clinical_utility_decision_support":
            content = re.sub(
                r"(?ms)^### Intended Use\s*.*?(?=^### |\Z)", "", content
            ).strip()
        payload["content"] = content
        sections[key] = payload
    cleaned["sections"] = sections
    return cleaned


def _paragraphs(text: str) -> str:
    if not text:
        return '<p class="empty">No evidence-grounded narrative was generated for this section.</p>'
    blocks: List[str] = []
    inserted: set[str] = set()
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            blocks.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            blocks.append(f"<h3>{html.escape(line[3:])}</h3>")
        elif line.startswith("- "):
            blocks.append(f"<p class=\"bullet\">• {html.escape(line[2:])}</p>")
        else:
            blocks.append(f"<p>{html.escape(line)}</p>")
    return "".join(blocks)


def _language_blocks(english: str, chinese: str) -> str:
    """Render parallel, deterministic language blocks without translating evidence IDs."""

    return (
        f'<div class="lang-block lang-en">{_paragraphs(english)}</div>'
        f'<div class="lang-block lang-zh">{_paragraphs(chinese)}</div>'
    )


def _paragraphs_with_inline_figures(text: str, figure_after: Dict[str, str]) -> str:
    """Render narrative paragraphs and place matching figure cards inline."""
    if not text:
        return '<p class="empty">No evidence-grounded narrative was generated for this section.</p>'
    blocks: List[str] = []
    inserted: set[str] = set()
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            block = f"<h3>{html.escape(line[4:])}</h3>"
        elif line.startswith("## "):
            block = f"<h3>{html.escape(line[3:])}</h3>"
        elif line.startswith("- "):
            block = f'<p class="bullet">• {html.escape(line[2:])}</p>'
        else:
            block = f"<p>{html.escape(line)}</p>"
        blocks.append(block)
        for marker, card in figure_after.items():
            if marker not in inserted and marker.lower() in line.lower():
                blocks.append(f'<div class="inline-figure-evidence">{card}</div>')
                inserted.add(marker)
    return "".join(blocks)


def _language_blocks_with_inline_figures(
    english: str,
    chinese: str,
    figure_after: Dict[str, str],
) -> str:
    return (
        f'<div class="lang-block lang-en">{_paragraphs_with_inline_figures(english, figure_after)}</div>'
        f'<div class="lang-block lang-zh">{_paragraphs(chinese)}</div>'
    )


def _zh_narrative(report_context: Dict[str, Any], bundle: Dict[str, Any], key: str) -> str:
    """Produce a factual Chinese companion narrative from the same archived inputs."""

    phase2 = _safe_dict(report_context.get("phase2"))
    clinical = _safe_dict(phase2.get("clinical_utility"))
    features = _safe_list(phase2.get("winner_features"))
    feature_rows = _feature_rows(phase2)
    canonical_by_token = {_safe_str(item.get("feature_token")): _safe_str(item.get("canonical_name")) for item in feature_rows}
    display_features = [canonical_by_token.get(_safe_str(item), _safe_str(item)) for item in features]
    calibration = _safe_dict(_safe_dict(bundle.get("calibration")).get("metrics"))
    metric_registry = _safe_dict(bundle.get("metric_registry"))
    canonical_calibration = _safe_dict(metric_registry.get("calibration")) or calibration
    validation = _safe_dict(bundle.get("validation_scope"))
    mechanism = _safe_dict(bundle.get("mechanistic_evidence_registry"))
    scope = _safe_str(validation.get("evidence_status") or validation.get("headline_scope") or "未记录")
    if key == "executive_summary":
        auc = _internal_auc({}, report_context)
        feature_labels = "、".join(display_features)
        return (
            "### 主要证据摘要\n"
            f"本次归档的 Phase 2 工作流选择了 {len(features)} 个最终特征（{feature_labels or '未归档'}），内部交叉验证 AUC 为 {_fmt(auc)}。\n"
            f"本报告的主要验证范围标记为 {scope}；所有性能结论均保留其各自的验证范围。除非明确声明独立外部队列，不应解释为外部验证。报告质量状态反映证据工件是否齐全，并不替代方法学同行评议。"
        )
    if key == "methodology_workflow":
        return (
            "### 分析路径\n"
            "MetaboAgent 按归档的 Phase 0 至 Phase 3 工件整理先验证据、预处理、特征选择、模型搜索、性能评估和可视化。Phase 4 仅消费既有结果，不重新拟合模型或改变分析决策。"
        )
    if key == "results":
        return (
            "### 结果与不确定性\n"
            f"最终模型包含 {len(features)} 个特征。区分度、校准和增量价值仅可在各自归档的队列与重采样范围内解释。"
        )
    if key == "clinical_utility_decision_support":
        brier = _fmt(canonical_calibration.get("brier_score"))
        return (
            "### 阈值与临床应用边界\n"
            f"归档校准评估的 Brier score 为 {brier}。阈值下的敏感度、特异度、PPV、NPV 和阳性标记率仅描述当前评估队列，未经独立外部验证和必要再校准时不构成可迁移的绝对风险阈值。"
        )
    if key == "discussion_mechanistic_insights":
        level = _safe_str(mechanism.get("interpretation_level"))
        if level == "MODEL-ASSOCIATION-ONLY":
            return (
                "### 证据范围内的生物学解释\n"
                "本次运行未归档最终特征的 Phase 0 先验证据、HMDB 关联通路或疾病核心通路。因此，本节仅报告模型归因和队列内关联，不提出疾病机制、通路激活或因果效应。\n"
                "SHAP 描述多变量模型中各特征对预测的贡献，限制性立方样条描述观察范围内的关联形状。两者均不能单独证明生物学保护效应或致病机制。"
            )
        return (
            "### 证据范围内的生物学解释\n"
            "已登记的通路或先验证据可用于为模型关联提供生物学背景，但并不证明通路激活或疾病因果关系。SHAP 与样条结果仍应解释为模型行为和队列内关联。"
        )
    return "未生成对应的中文叙事。"


def _scope_class(scope: str) -> str:
    scope = scope.lower()
    if "external" in scope:
        return "scope-external"
    if "holdout" in scope:
        return "scope-holdout"
    if "not" in scope:
        return "scope-missing"
    return "scope-internal"


def _inline_image_data_uri(path: Path) -> str:
    """Return a self-contained image URI for report-local previews."""
    if not path.exists() or not path.is_file():
        return ""
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(path.suffix.lower())
    if not mime:
        return ""
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:{mime};base64,{encoded}"


def _figure_cards(
    report_context: Dict[str, Any],
    run_root: Path,
    report_dir: Path,
    include_ids: Optional[set[str]] = None,
    exclude_ids: Optional[set[str]] = None,
) -> str:
    cards: List[str] = []
    for figure in _safe_list(_safe_dict(report_context.get("phase3")).get("figures")):
        item = _safe_dict(figure)
        title = _safe_str(item.get("title") or item.get("figure_id") or "Figure")
        figure_id = _safe_str(item.get("figure_id") or "figure")
        if include_ids is not None and figure_id not in include_ids:
            continue
        if exclude_ids is not None and figure_id in exclude_ids:
            continue
        outputs = [_safe_dict(value) for value in _safe_list(item.get("all_outputs"))]
        primary = _safe_dict(item.get("primary_output"))
        if primary:
            outputs.insert(0, primary)
        preview = ""
        seen: set[str] = set()
        for output in outputs:
            raw_path = _safe_str(output.get("path")).strip()
            if not raw_path or raw_path in seen:
                continue
            seen.add(raw_path)
            resolved = _resolve_figure_path(run_root, raw_path)
            for candidate in _figure_preview_candidates(run_root, resolved):
                if not candidate.exists():
                    continue
                suffix = candidate.suffix.lower()
                if not preview and suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                    inline_src = _inline_image_data_uri(candidate)
                    # Figures are rendered as embedded previews.  Keep the source
                    # artifact in the evidence bundle, but do not expose it as a
                    # hyperlink from the reader-facing figure card.
                    src = inline_src
                    preview = f'<img src="{html.escape(src)}" alt="{html.escape(title)}" data-inline="true">'
                    break
                if not preview and suffix == '.pdf':
                    preview_relative = _portable_pdf_preview(report_dir, candidate)
                    if preview_relative:
                        preview_path = report_dir / Path(preview_relative)
                        inline_src = _inline_image_data_uri(preview_path)
                        src = inline_src
                        preview = f'<img src="{html.escape(src)}" alt="{html.escape(title)} (PDF first-page preview)" data-inline="true">'
                        break
        caption = _safe_str(item.get("caption_seed")).strip()
        if "external holdout" in caption.lower():
            caption = "Internal holdout ROC curve of the final winner panel. This held-out subset is development evidence, not independent external validation."
        if preview:
            caption_html = f'<p class="caption">{html.escape(caption)}</p>' if caption else '<p class="caption">Figure metadata only; caption unavailable.</p>'
            cards.append(
                '<article class="figure-card">'
                f'<div class="figure-kicker">Figure {html.escape(figure_id)}</div>'
                f'<h3>{html.escape(title)}</h3>{preview}'
                f'{caption_html}'
                '</article>'
            )
    return "".join(cards) or '<p class="empty">No registered figure assets are available.</p>'


def _contract_figure_cards(
    report_context: Dict[str, Any],
    contract_manifest: Dict[str, Any],
    run_root: Path,
    report_dir: Path,
    *,
    include_ids: Optional[set[str]] = None,
    section_id: Optional[str] = None,
    displays: Optional[set[str]] = None,
) -> str:
    """Render figure cards from the canonical manifest rather than task IDs.

    Legacy Phase 3 records are used only to locate the source asset.  Titles,
    placement, scope and captions come from ``figure_manifest``.  This makes
    the report renderer insensitive to the historical ``fig2a``/``fig4a``
    naming scheme.
    """
    legacy_by_id = {
        _safe_str(_safe_dict(item).get("figure_id")): _safe_dict(item)
        for item in _safe_list(_safe_dict(report_context.get("phase3")).get("figures"))
    }
    cards: List[str] = []
    for value in _safe_list(contract_manifest.get("figures")):
        figure = _safe_dict(value)
        figure_id = _safe_str(figure.get("figure_id"))
        if figure_id in {"objective_shift", "objective_shift_radar"}:
            continue
        placement = _safe_dict(figure.get("placement"))
        display = _safe_str(placement.get("display"))
        if include_ids is not None and figure_id not in include_ids:
            continue
        if section_id is not None and _safe_str(placement.get("section_id")) != section_id:
            continue
        if displays is not None and display not in displays:
            continue
        title = _safe_str(figure.get("title") or figure_id or "Figure")
        preview = ""
        raw_outputs: List[str] = []
        if figure_id == "phase_flow_overview":
            raw_outputs = [str(report_dir / "figures" / "phase_flow_overview.svg")]
        else:
            legacy = legacy_by_id.get(_safe_str(figure.get("legacy_figure_id")), {})
            for output in _safe_list(legacy.get("all_outputs")):
                path = _safe_str(_safe_dict(output).get("path")).strip()
                if path:
                    raw_outputs.append(path)
            primary = _safe_str(_safe_dict(legacy.get("primary_output")).get("path")).strip()
            if primary:
                raw_outputs.insert(0, primary)
            # A Phase 0 atlas may be discovered directly from the run-scoped
            # output even when the historical Phase 3 figure inventory omitted
            # its legacy ``fig1c`` record.
            if not raw_outputs:
                for output in _safe_list(figure.get("outputs")):
                    path = _safe_str(_safe_dict(output).get("path")).strip()
                    if path:
                        raw_outputs.append(path)
        seen: set[str] = set()
        for raw_path in raw_outputs:
            if raw_path in seen:
                continue
            seen.add(raw_path)
            resolved = Path(raw_path)
            if not resolved.is_absolute():
                candidate_report = report_dir / resolved
                resolved = candidate_report if candidate_report.exists() else run_root / resolved
            resolved = resolved.resolve()
            for candidate in _figure_preview_candidates(run_root, resolved):
                if not candidate.exists() or not candidate.is_file():
                    continue
                suffix = candidate.suffix.lower()
                if not preview and suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                    inline_src = _inline_image_data_uri(candidate)
                    src = inline_src
                    preview = f'<img src="{html.escape(src)}" alt="{html.escape(title)}" data-inline="true">'
                    break
                if not preview and suffix == ".pdf":
                    preview_relative = _portable_pdf_preview(report_dir, candidate)
                    if preview_relative:
                        preview_path = report_dir / Path(preview_relative)
                        inline_src = _inline_image_data_uri(preview_path)
                        src = inline_src
                        preview = f'<img src="{html.escape(src)}" alt="{html.escape(title)} (PDF first-page preview)" data-inline="true">'
                        break
        caption_contract = _safe_dict(figure.get("caption_contract"))
        caption = _safe_str(caption_contract.get("text_template")).strip()
        if not caption or caption.startswith("Historical figure inventory fallback"):
            caption = (
                f"{title}. Scope: {_safe_str(figure.get('cohort_scope') or 'not archived')}; "
                f"fit scope: {_safe_str(figure.get('fit_scope') or 'not archived')}. "
                f"{_safe_str(figure.get('interpretation_boundary') or 'Interpretation is restricted to the registered evidence scope.') }"
            )
        if preview or figure_id == "phase_flow_overview":
            cards.append(
                '<article class="figure-card">'
                f'<div class="figure-kicker">Figure {html.escape(figure_id)}</div>'
                f'<h3>{html.escape(title)}</h3>{preview}'
                f'<p class="caption">{html.escape(caption)}</p>'
                '</article>'
            )
    return "".join(cards) or '<p class="empty">No manifest-registered figure assets are available.</p>'


def _metric_cards(bundle: Dict[str, Any], report_dir: Optional[Path] = None) -> str:
    cards: List[str] = []
    for claim in _safe_list(bundle.get("claim_registry")):
        item = _safe_dict(claim)
        value = item.get("value")
        # Structural claims (phase flow, QC, selection, utility and
        # interpretation) are already represented by their dedicated
        # sections.  They deliberately have no scalar value and should not
        # occupy scarce Executive Summary metric-card space as “Not available”.
        if value is None:
            continue
        if isinstance(value, float):
            value_text = f"{value:.3f}"
        elif value is None:
            value_text = "N/A"
        else:
            value_text = _safe_str(value)
        scope = _safe_str(item.get("evidence_scope") or "NOT-ASSESSABLE")
        claim_id = _safe_str(item.get("claim_id"))
        chinese_statement = {
            "winner-panel-size": "最终入选特征数量。",
            "internal-discrimination": "归档的内部评估区分度。",
            "holdout-discrimination": "归档留出集的区分度。",
            "calibration": "归档队列内的概率校准表现。",
            "prior-evidence": "Phase 0 先验证据筛选结果。",
        }.get(claim_id, "归档证据摘要。")
        links = []
        for evidence in _safe_list(item.get("evidence")):
            href = _report_href(report_dir, _safe_dict(evidence)) if report_dir else ""
            if href:
                links.append(f'<a href="{html.escape(href)}">{html.escape(Path(_safe_str(_safe_dict(evidence).get("resolved_path"))).name)}</a>')
        cards.append(
            '<article class="metric-card">'
            f'<span class="scope {_scope_class(scope)}">{html.escape(scope)}</span>'
            f'<div class="metric-value">{html.escape(value_text)}</div>'
            f'<p class="lang-en">{html.escape(_safe_str(item.get("statement")))}</p>'
            f'<p class="lang-zh">{html.escape(chinese_statement)}</p>'
            f'<small class="lang-en">{html.escape(_safe_str(item.get("boundary")))}</small>'
            '<small class="lang-zh">所有指标均受其归档验证范围限制。</small>'
            + (f'<div class="asset-links">{" · ".join(links)}</div>' if links else '') +
            '</article>'
        )
    return "".join(cards)


_CLAIM_GRAPH_FALLBACKS = {
    # These claims are added by the contract/template layer.  They describe
    # report structure or interpretation boundaries rather than a single
    # model metric, so their evidence is the corresponding deterministic
    # registry rather than a raw prediction artifact.
    "phase-flow": ("phase_flow_manifest.json", "Phase-flow manifest"),
    "cohort-qc": ("cohort_qc_snapshot.json", "Cohort/QC snapshot"),
    "selection-stability": ("feature_selection_evidence.json", "Feature-selection evidence"),
    "clinical-utility": ("metric_registry.json", "Metric registry"),
    "biological-context": ("mechanistic_evidence_registry.json", "Mechanistic evidence registry"),
}


def _claim_graph_fallback_artifact(report_dir: Path, claim_id: str) -> Dict[str, Any]:
    candidate = _CLAIM_GRAPH_FALLBACKS.get(claim_id)
    if not candidate:
        return {}
    relative_name, _ = candidate
    path = report_dir / relative_name
    if not path.exists():
        return {}
    href = _portable_href(report_dir, path)
    return {
        "path": relative_name,
        "resolved_path": str(path),
        "exists": True,
        "sha256": _sha256(path),
        "relative_href": href,
        "artifact_name": path.name,
    }


def _claim_graph_fallback(report_dir: Path, claim_id: str) -> str:
    artifact = _claim_graph_fallback_artifact(report_dir, claim_id)
    if not artifact:
        return ""
    label = _CLAIM_GRAPH_FALLBACKS[claim_id][1]
    href = _safe_str(artifact.get("relative_href"))
    return f'<a href="{html.escape(href)}">{html.escape(label)}</a>' if href else ""


def _claim_graph_html(bundle: Dict[str, Any], report_dir: Path) -> str:
    rows = []
    for claim_value in _safe_list(bundle.get("claim_registry")):
        claim = _safe_dict(claim_value)
        links = []
        for artifact_value in _safe_list(claim.get("evidence")):
            artifact = _safe_dict(artifact_value)
            href = _report_href(report_dir, artifact)
            if href:
                links.append(f'<a href="{html.escape(href)}">{html.escape(Path(_safe_str(artifact.get("resolved_path"))).name)}</a>')
        if not links:
            fallback = _claim_graph_fallback(report_dir, _safe_str(claim.get("claim_id")))
            links = [fallback] if fallback else []
        if links:
            # Keep each evidence artifact on its own line.  A single claim can
            # be backed by several long, run-scoped filenames; joining them
            # inline makes the compact two-column graph overflow and overlap.
            evidence_label = '<div class="claim-evidence-links">' + ''.join(
                f'<span>{link}</span>' for link in links
            ) + '</div>'
        else:
            evidence_label = "Artifact not archived in this run"
        rows.append(f'<article class="claim-row"><div><strong>{html.escape(_safe_str(claim.get("statement")))}</strong><br><span class="scope {_scope_class(_safe_str(claim.get("evidence_scope")))}">{html.escape(_safe_str(claim.get("evidence_scope")))}</span> <span class="status status-{html.escape(_safe_str(claim.get("status") or "not_available").lower())}">{html.escape(_safe_str(claim.get("status") or "NOT_AVAILABLE"))}</span> <small>{html.escape(_safe_str(claim.get("evidence_strength") or "NOT_ASSESSABLE"))}</small></div><div class="asset-links">{evidence_label}</div></article>')
    return "".join(rows) or '<p class="empty">No claims were registered.</p>'


def _evaluation_evidence_table(bundle: Dict[str, Any]) -> str:
    """Present internal resampling and internal holdout side by side without externalising either."""

    claims = {
        _safe_str(_safe_dict(item).get("claim_id")): _safe_dict(item)
        for item in _safe_list(bundle.get("claim_registry"))
    }
    validation = _safe_dict(bundle.get("validation_scope"))
    metrics = _safe_dict(bundle.get("metric_registry"))
    internal = _safe_dict(metrics.get("development_cv"))
    holdout = _safe_dict(metrics.get("internal_holdout"))
    external_status = "Declared" if validation.get("external_declared") else "Not declared"
    cv_metrics = internal
    holdout_metrics = holdout
    def metric_line(auc: Any, pr_auc: Any, ci: Any, pr_ci: Any) -> str:
        auc_text = _fmt(auc)
        pr_text = _fmt(pr_auc) if _safe_float(pr_auc) is not None else "Not archived"
        ci_text = _format_ci(ci) if isinstance(ci, (list, tuple)) and len(ci) == 2 else "Not archived"
        pr_ci_text = _format_ci(pr_ci) if isinstance(pr_ci, (list, tuple)) and len(pr_ci) == 2 else "Not archived"
        return f"AUC {auc_text} (95% CI {ci_text}); PR-AUC {pr_text} (95% CI {pr_ci_text})"
    rows = [
        (
            "Development resampling (CV)", "开发集重采样（CV）", metric_line(cv_metrics.get("auc"), cv_metrics.get("pr_auc"), cv_metrics.get("auc_ci_95"), cv_metrics.get("pr_auc_ci_95")), "INTERNAL-CV",
            "Model-development evidence; not external validation.", "模型开发证据；不是外部验证。",
        ),
        (
            "Internal holdout", "内部留出集", metric_line(holdout_metrics.get("auc"), holdout_metrics.get("pr_auc"), holdout_metrics.get("auc_ci_95"), holdout_metrics.get("pr_auc_ci_95")) if validation.get("holdout_available") or holdout_metrics.get("auc") is not None else "Not archived", "INTERNAL-HOLDOUT",
            "Held-out subset of the development workflow; not independent external validation.", "开发流程的预留子集；不是独立外部验证。",
        ),
        (
            "Independent external cohort", "独立外部队列", external_status, "EXTERNAL" if validation.get("external_declared") else "NOT-AVAILABLE",
            "Only an independently declared cohort may support external-validation claims.", "只有独立声明的队列才可支持外部验证主张。",
        ),
    ]
    return "".join(
        "<tr>"
        f'<td><span class="lang-en">{html.escape(en_label)}</span><span class="lang-zh">{html.escape(zh_label)}</span></td>'
        f"<td>{html.escape(value)}</td><td><span class=\"scope {_scope_class(scope)}\">{html.escape(scope)}</span></td>"
        f'<td><span class="lang-en">{html.escape(en_boundary)}</span><span class="lang-zh">{html.escape(zh_boundary)}</span></td>'
        "</tr>"
        for en_label, zh_label, value, scope, en_boundary, zh_boundary in rows
    )


def _study_ladder_html(bundle: Dict[str, Any]) -> str:
    """Render one complete, reader-facing experiment flow diagram."""
    nodes = [
        ("01", "Dataset provenance", "Source cohort and assay metadata"),
        ("02", "Cohort and QC", "Samples, classes, missingness and feature snapshot"),
        ("03", "Preprocessing and selection", "Training-fitted preprocessing and candidate-to-panel filtering"),
        ("04", "Model development", "5-fold out-of-fold evaluation and winner specification"),
        ("05", "Internal validation and interpretation", "Internal holdout, calibration, DCA and model interpretation"),
    ]
    cards = []
    for index, (number, title, summary) in enumerate(nodes):
        arrow = '<span class="study-flow-arrow" aria-hidden="true">→</span>' if index < len(nodes) - 1 else ''
        cards.append(
            f'<div class="study-flow-node"><span class="study-flow-number">{number}</span>'
            f'<strong>{html.escape(title)}</strong><small>{html.escape(summary)}</small></div>{arrow}'
        )
    return '<div class="study-flow" role="img" aria-label="MetaboAgent experiment flow">' + "".join(cards) + '</div>'


def _cohort_flow_html(bundle: Dict[str, Any], winner_count: int) -> str:
    qc = _safe_dict(_safe_dict(bundle.get("enhancements")).get("cohort_qc_snapshot"))
    profiles = {_safe_str(_safe_dict(item).get("artifact_role")): _safe_dict(item) for item in _safe_list(qc.get("profiles"))}
    train = profiles.get("training/modeling artifact", {})
    holdout = profiles.get("internal holdout artifact", {})
    train_n = _safe_float(train.get("n_samples"))
    holdout_n = _safe_float(holdout.get("n_samples"))
    features = _safe_float(train.get("n_features"))
    total = int(train_n + holdout_n) if train_n is not None and holdout_n is not None else None
    sample_flow = f"{total} samples → {int(train_n)} development + {int(holdout_n)} internal holdout" if total is not None else "Sample flow not fully archived"
    feature_flow = f"{int(features)} measured features → {winner_count} final panel features" if features is not None else f"Measured feature count not archived → {winner_count} final panel features"
    return (
        f'<div class="evidence-box"><strong>Sample flow:</strong> {html.escape(sample_flow)}<br>'
        f'<strong>Feature flow:</strong> {html.escape(feature_flow)}<br>'
        f'<span class="boundary">The final panel count is distinct from the Phase 1 handoff/candidate count.</span></div>'
    )


def _placeholder_hits(llm_sections: Dict[str, Any]) -> List[str]:
    sections = _safe_dict(llm_sections.get("sections"))
    text = "\n".join(_safe_str(_safe_dict(payload).get("content")) for payload in sections.values())
    hits: List[str] = []
    for label, pattern in (
        ('None-fold', r'\bNone[- ]fold\b'),
        ('malformed-fold', r'\b\d{4,}-fold\b'),
        ('N/A', r'\bN/A\b'),
        ('Fig.', r'\bFig\.\s*(?:None|N/A)?(?=[,.;:)])'),
        ('unregistered-caption', r'Caption evidence was not registered'),
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(label)
    return hits


def _enhancement_tables(bundle: Dict[str, Any]) -> Tuple[str, str, str, str]:
    def display(value: Any) -> str:
        return "Not archived" if value in (None, "") else _safe_str(value)

    enhancements = _safe_dict(bundle.get("enhancements"))
    qc = _safe_dict(enhancements.get("cohort_qc_snapshot"))
    qc_rows = []
    for profile_value in _safe_list(qc.get("profiles")):
        profile = _safe_dict(profile_value)
        counts = json.dumps(_safe_dict(profile.get("class_counts")), ensure_ascii=False) if profile.get("class_counts") else "Not inferred"
        qc_rows.append(
            f"<tr><td>{html.escape(_safe_str(profile.get('artifact_role')))}</td><td>{html.escape(_safe_str(profile.get('n_samples')))}</td><td>{html.escape(_safe_str(profile.get('n_features')))}</td><td>{html.escape(_safe_str(profile.get('missing_cell_fraction') if profile.get('missing_cell_fraction') is not None else 'Not assessed'))}</td><td>{html.escape(counts)}</td></tr>"
        )
    qc_html = "".join(qc_rows) or '<tr><td colspan="5">Cohort QC snapshot was not assessable from archived tabular artifacts.</td></tr>'

    selection = _safe_dict(enhancements.get("feature_selection_evidence"))
    selection_rows = []
    for record_value in _safe_list(selection.get("winner_records")):
        record = _safe_dict(record_value)
        frequency = _safe_float(record.get("phase1_selection_frequency"))
        selection_rows.append(
            f"<tr><td>{html.escape(_safe_str(record.get('feature')))}</td><td>{'Not archived' if frequency is None else f'{frequency:.3f}'}</td><td>{html.escape(_safe_str(record.get('method_consensus') or 'Not archived'))}</td><td>{'Yes' if record.get('stable_core_member') else 'No'}</td><td>{html.escape(_safe_str(record.get('selection_stage')))}</td></tr>"
        )
    selection_html = "".join(selection_rows) or '<tr><td colspan="5">Phase 1 stability-selection artifact was not assessed.</td></tr>'

    model = _safe_dict(enhancements.get("model_card"))
    model_rows = []
    for label, value in (
        ("Target", model.get("target")),
        ("Selected model", model.get("selected_model")),
        ("Winner features", len(_safe_list(model.get("winner_features")))),
        ("Internal holdout", _safe_dict(model.get("validation")).get("internal_holdout_available")),
        ("External validation", "Declared" if _safe_dict(model.get("validation")).get("external_declared") else "Not declared"),
    ):
        model_rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(display(value))}</td></tr>")
    model_html = "".join(model_rows) or '<tr><td colspan="2">Model card was not assessed.</td></tr>'

    repro = _safe_dict(enhancements.get("reproducibility_manifest"))
    repro_rows = []
    runtime = _safe_dict(repro.get("runtime"))
    git = _safe_dict(repro.get("git"))
    for label, value in (("Git commit", git.get("commit")), ("Git dirty", git.get("dirty")), ("Python", runtime.get("python")), ("Environment random state", _safe_dict(repro.get("random_seeds")).get("environment_random_state")), ("Stability seed", _safe_dict(repro.get("random_seeds")).get("phase1_stability_seed"))):
        repro_rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(display(value))}</td></tr>")
    return qc_html, selection_html, model_html, "".join(repro_rows)


def _association_table(bundle: Dict[str, Any]) -> str:
    rows = []
    payload = _safe_dict(_safe_dict(bundle.get("enhancements")).get("training_association_snapshot"))
    for record_value in _safe_list(payload.get("rows")):
        record = _safe_dict(record_value)
        effect = _safe_float(record.get("standardized_mean_difference_hedges_g"))
        rows.append(
            f"<tr><td>{html.escape(_safe_str(record.get('feature')))}</td><td>{'Not assessed' if effect is None else f'{effect:.3f}'}</td><td>{html.escape(_safe_str(record.get('direction_case_minus_control') or 'Not assessed'))}</td><td>{html.escape(_safe_str(record.get('fit_scope') or 'train_only'))}</td><td>{html.escape(_safe_str(record.get('interpretation') or 'Descriptive association, not a causal effect.'))}</td></tr>"
        )
    return "".join(rows) or '<tr><td colspan="5">Train-only association snapshot was not assessable.</td></tr>'


def _biomarker_evidence_table(
    bundle: Dict[str, Any], feature_rows: List[Dict[str, Any]]
) -> str:
    """Join identity, selection and association evidence without inventing it."""

    enhancements = _safe_dict(bundle.get("enhancements"))
    selection_records = {
        _safe_str(_safe_dict(item).get("feature")): _safe_dict(item)
        for item in _safe_list(_safe_dict(enhancements.get("feature_selection_evidence")).get("winner_records"))
    }
    association_records = {
        _safe_str(_safe_dict(item).get("feature")): _safe_dict(item)
        for item in _safe_list(_safe_dict(enhancements.get("training_association_snapshot")).get("rows"))
    }
    rows: List[str] = []
    for feature in feature_rows:
        name = _safe_str(feature.get("canonical_name"))
        raw_name = _safe_str(feature.get("raw_column_name"))
        selection = selection_records.get(name) or selection_records.get(raw_name) or {}
        association = association_records.get(name) or association_records.get(raw_name) or {}
        frequency = _safe_float(selection.get("phase1_selection_frequency"))
        effect = _safe_float(association.get("standardized_mean_difference_hedges_g"))
        hmdb = ", ".join(_safe_list(feature.get("hmdb_ids"))) or "Not archived"
        rows.append(
            "<tr>"
            f"<td>{html.escape(name or raw_name)}</td>"
            f"<td>{html.escape(_safe_str(feature.get('role')) or 'Raw feature')}</td>"
            f"<td>{html.escape(hmdb)}</td>"
            f"<td>{'Not archived' if frequency is None else f'{frequency:.3f}'}</td>"
            f"<td>{html.escape(_safe_str(selection.get('method_consensus')) or 'Not archived')}</td>"
            f"<td>{'Not archived' if effect is None else f'{effect:.3f}'}</td>"
            f"<td>{html.escape(_safe_str(association.get('direction_case_minus_control')) or 'Not archived')}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="7">Winner-panel evidence was not assessable.</td></tr>'


def _compact_report_path(value: Any) -> str:
    """Keep report-facing audit references portable and free of server roots."""
    text = _safe_str(value).strip()
    if not text:
        return "Not archived"
    normalized = text.replace("\\", "/")
    return Path(normalized).name if "/" in normalized else normalized


def _compact_report_text(value: Any) -> str:
    """Remove absolute filesystem prefixes from explanatory audit text."""
    text = _safe_str(value).strip()
    if not text:
        return ""
    return re.sub(r"(?<![A-Za-z0-9])(?:/[^\s<]+)+/([^/\s<]+)", r"\1", text)


def _audit_table(bundle: Dict[str, Any]) -> str:
    rows: List[str] = []
    for domain in _safe_list(_safe_dict(bundle.get("audit_summary")).get("domains")):
        item = _safe_dict(domain)
        status = _safe_str(item.get("status") or "NOT_ASSESSED")
        rows.append(
            "<tr>"
            f"<td>{html.escape(_safe_str(item.get('domain')))}</td>"
            f"<td><span class=\"status status-{html.escape(status.lower())}\">{html.escape(status)}</span></td>"
            f"<td><code>{html.escape(_compact_report_path(item.get('evidence_file')))}</code>"
            f"<br><small>{html.escape(_compact_report_text(item.get('status_reason')))}</small></td>"
            "</tr>"
        )
    enhancement_labels = {
        "reproducibility_manifest": "Reproducibility manifest",
        "visual_evidence_registry": "Visual evidence registry",
        "cohort_qc_snapshot": "Cohort/QC snapshot",
    }
    for key, label in enhancement_labels.items():
        payload = _safe_dict(_safe_dict(bundle.get("enhancements")).get(key))
        if payload:
            status = _safe_str(payload.get("status") or "NOT_ASSESSED")
            rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td><span class=\"status status-{html.escape(status.lower())}\">{html.escape(status)}</span></td>"
                f"<td><code>{html.escape(key + '.json')}</code></td>"
                "</tr>"
            )
    return "".join(rows) or '<tr><td colspan="3">Audit summary unavailable.</td></tr>'


def _threshold_explorer(report_context: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    clinical = _safe_dict(_safe_dict(report_context.get("phase2")).get("clinical_utility"))
    rows = [_safe_dict(item) for item in _safe_list(clinical.get("threshold_metrics_table")) if isinstance(item, dict)]
    rows = [item for item in rows if _safe_float(item.get("threshold")) is not None]
    if not rows:
        return '<p class="empty">Threshold-level operating characteristics were not archived.</p>', "[]"
    rows.sort(key=lambda item: float(item.get("threshold")))
    minimum = float(rows[0]["threshold"])
    maximum = float(rows[-1]["threshold"])
    default = _safe_float(_safe_dict(clinical.get("recommended_threshold_summary")).get("selected_threshold"))
    if default is None:
        default = minimum
    prevalence_context = _safe_dict(clinical.get("prevalence_context"))
    observed_prevalence = _first_numeric(
        prevalence_context.get("observed_sampling_prevalence"),
        prevalence_context.get("observed_prevalence"),
        clinical.get("observed_prevalence"),
    )
    if observed_prevalence is None and bundle:
        profiles = _safe_list(_safe_dict(_safe_dict(bundle.get("enhancements")).get("cohort_qc_snapshot")).get("profiles"))
        positive = total = 0
        for profile_value in profiles:
            counts = _safe_dict(_safe_dict(profile_value).get("class_counts"))
            for label, count in counts.items():
                total += int(count or 0)
                if str(label).lower() in {"1", "case", "positive", "disease"}:
                    positive += int(count or 0)
        if total:
            observed_prevalence = positive / total
    threshold_rows_html = []
    default_row = min(rows, key=lambda item: abs(float(item.get("threshold")) - default))
    for item in rows:
        sensitivity = _safe_float(item.get("sensitivity"))
        specificity = _safe_float(item.get("specificity"))
        flagged = _safe_float(item.get("flagged_rate"))
        tp1000 = _safe_float(item.get("tp_per_1000"))
        fp1000 = _safe_float(item.get("fp_per_1000"))
        if tp1000 is None and observed_prevalence is not None and sensitivity is not None:
            tp1000 = sensitivity * observed_prevalence * 1000
        if fp1000 is None and observed_prevalence is not None and specificity is not None:
            fp1000 = (1 - specificity) * (1 - observed_prevalence) * 1000
        flagged_text = "Not available" if flagged is None else f"{flagged * 1000:.1f}"
        tp_text = "Not available" if tp1000 is None else f"{tp1000:.1f}"
        fp_text = "Not available" if fp1000 is None else f"{fp1000:.1f}"
        ppv = _safe_float(item.get("ppv"))
        npv = _safe_float(item.get("npv"))
        sensitivity_text = "Not available" if sensitivity is None else f"{sensitivity:.3f}"
        specificity_text = "Not available" if specificity is None else f"{specificity:.3f}"
        ppv_text = "Not available" if ppv is None else f"{ppv:.3f}"
        npv_text = "Not available" if npv is None else f"{npv:.3f}"
        threshold_rows_html.append(
            f'<tr><td>{_safe_float(item.get("threshold")):.2f}</td><td>{sensitivity_text}</td><td>{specificity_text}</td><td>{ppv_text}</td><td>{npv_text}</td><td>{flagged_text}</td><td>{tp_text}</td><td>{fp_text}</td></tr>'
        )
    prevalence_note = (
        f"Observed sampling prevalence used for descriptive TP/FP rates: {observed_prevalence:.3f}."
        if observed_prevalence is not None else
        "Observed sampling prevalence was not archived; TP/FP per 1,000 are therefore not inferred."
    )
    component = f'''
    <div class="threshold-explorer" data-threshold-explorer>
      <label for="threshold-slider"><span class="lang-en inline">Illustrative operating threshold</span><span class="lang-zh inline">示例运行阈值</span> <strong id="threshold-value">{default:.2f}</strong></label>
      <input id="threshold-slider" type="range" min="{minimum}" max="{maximum}" value="{default}" step="0.01">
      <div class="threshold-grid">
        <div><span class="lang-en">Sensitivity</span><span class="lang-zh">敏感度</span><strong id="threshold-sensitivity">{html.escape(_safe_str(default_row.get("sensitivity") if default_row.get("sensitivity") is not None else "Not available"))}</strong></div>
        <div><span class="lang-en">Specificity</span><span class="lang-zh">特异度</span><strong id="threshold-specificity">{html.escape(_safe_str(default_row.get("specificity") if default_row.get("specificity") is not None else "Not available"))}</strong></div>
        <div><span>PPV</span><strong id="threshold-ppv">{html.escape(_safe_str(default_row.get("ppv") if default_row.get("ppv") is not None else "Not available"))}</strong></div>
        <div><span>NPV</span><strong id="threshold-npv">{html.escape(_safe_str(default_row.get("npv") if default_row.get("npv") is not None else "Not available"))}</strong></div>
        <div><span class="lang-en">Flagged rate</span><span class="lang-zh">阳性标记率</span><strong id="threshold-flagged">{html.escape(_safe_str(default_row.get("flagged_rate") if default_row.get("flagged_rate") is not None else "Not available"))}</strong></div>
      </div>
      <h3><span class="lang-en">Threshold operating characteristics per 1,000 screened</span><span class="lang-zh">每 1,000 人的阈值运行特征</span></h3>
      <div class="table-scroll"><table><thead><tr><th>Threshold</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>NPV</th><th>Flagged/1000</th><th>TP/1000</th><th>FP/1000</th></tr></thead><tbody>{''.join(threshold_rows_html)}</tbody></table></div>
      <p class="boundary lang-en">{html.escape(prevalence_note)} These are descriptive case-control operating characteristics; PPV/NPV and referral burden change with target-population prevalence.</p><p class="boundary lang-zh">{html.escape(prevalence_note)} 这些是病例-对照队列中的描述性运行特征；PPV/NPV 和转诊负担会随目标人群患病率变化。</p>
      <p class="boundary lang-en">Values are not portable absolute-risk thresholds without independent external validation and recalibration.</p><p class="boundary lang-zh">未经独立外部验证和再校准，这些数值不能作为可迁移的绝对风险阈值。</p>
    </div>'''
    return component, json.dumps(rows, ensure_ascii=False)


def _prevalence_scenario_table(bundle: Dict[str, Any]) -> str:
    payload = _safe_dict(_safe_dict(bundle.get("enhancements")).get("prevalence_scenarios"))
    rows = _safe_list(payload.get("rows"))
    if not rows:
        return '<p class="empty">Prevalence scenarios were not assessable from archived sensitivity and specificity.</p>'
    thresholds = sorted({_safe_float(_safe_dict(item).get("threshold")) for item in rows if _safe_float(_safe_dict(item).get("threshold")) is not None})
    selected = thresholds[0] if thresholds else None
    clinical = _safe_dict(_safe_dict(bundle.get("enhancements")).get("model_card"))
    utility = _safe_dict(clinical.get("clinical_utility"))
    intended = _safe_float(_safe_dict(utility.get("recommended_threshold_summary")).get("selected_threshold"))
    if intended in thresholds:
        selected = intended
    body = []
    for item_value in rows:
        item = _safe_dict(item_value)
        if selected is not None and _safe_float(item.get("threshold")) != selected:
            continue
        body.append(
            "<tr>"
            f"<td>{_safe_float(item.get('assumed_prevalence')) * 100:.1f}%</td>"
            f"<td>{_safe_float(item.get('projected_ppv')):.3f}</td>"
            f"<td>{_safe_float(item.get('projected_npv')):.3f}</td>"
            f"<td>{_safe_float(item.get('projected_positive_tests_per_1000')):.1f}</td>"
            f"<td>{_safe_float(item.get('projected_false_positives_per_1000')):.1f}</td>"
            "</tr>"
        )
    return (
        f'<h3>Prevalence-adjusted scenario projection (threshold {selected:.2f})</h3>'
        '<div class="table-scroll"><table><thead><tr><th>Assumed prevalence</th><th>Projected PPV</th><th>Projected NPV</th><th>Positive tests/1000</th><th>False positives/1000</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        '<p class="boundary">Scenario projection only: sensitivity and specificity are transported unchanged from the archived case-control sample. These are not observed population results.</p>'
    )


def _maturity_panel(bundle: Dict[str, Any]) -> str:
    payload = _safe_dict(_safe_dict(bundle.get("enhancements")).get("evidence_maturity"))
    label = _safe_str(payload.get("validation_label") or "Internal model development and holdout evaluation")
    return f'<div class="evidence-box"><strong>Validation scope:</strong> {html.escape(label)}. Independent external validation is the recommended next evidence step.</div>'


def _identity_cards(bundle: Dict[str, Any]) -> str:
    registry = _safe_dict(_safe_dict(bundle.get("enhancements")).get("feature_identity_registry"))
    selection = {
        _safe_str(_safe_dict(item).get("feature")): _safe_dict(item)
        for item in _safe_list(_safe_dict(_safe_dict(bundle.get("enhancements")).get("feature_selection_evidence")).get("winner_records"))
    }
    association = {
        _safe_str(_safe_dict(item).get("feature")): _safe_dict(item)
        for item in _safe_list(_safe_dict(_safe_dict(bundle.get("enhancements")).get("training_association_snapshot")).get("rows"))
    }
    cards = []
    for item_value in _safe_list(registry.get("records")):
        item = _safe_dict(item_value)
        token = _safe_str(item.get("canonical_id"))
        sel = selection.get(token, {})
        assoc = association.get(token, {})
        frequency = _safe_float(sel.get("phase1_selection_frequency"))
        effect = _safe_float(assoc.get("standardized_mean_difference_hedges_g"))
        cards.append(
            '<article class="identity-card">'
            f'<h3>{html.escape(_safe_str(item.get("canonical_name")))}</h3>'
            f'<p><strong>HMDB:</strong> {html.escape(", ".join(_safe_list(item.get("hmdb_ids"))) or "Not archived")}</p>'
            f'<p><strong>MSI level:</strong> {html.escape(_safe_str(item.get("msi_identification_level")))}</p>'
            f'<p><strong>Assay:</strong> m/z {html.escape(_safe_str(item.get("mz")))} · RT {html.escape(_safe_str(item.get("retention_time")))} · {html.escape(_safe_str(item.get("ion_mode")))}</p>'
            f'<p><strong>Selection frequency:</strong> {"Not archived" if frequency is None else f"{frequency:.3f}"}</p>'
            f'<p><strong>Train-only Hedges g:</strong> {"Not archived" if effect is None else f"{effect:.3f}"} ({html.escape(_safe_str(assoc.get("direction_case_minus_control")) or "direction not archived")})</p>'
            '</article>'
        )
    return '<div class="identity-grid">' + ''.join(cards) + '</div>' if cards else '<p class="empty">Biomarker identity cards were not assessable.</p>'


def _reporting_readiness_table(bundle: Dict[str, Any]) -> str:
    payload = _safe_dict(_safe_dict(bundle.get("enhancements")).get("reporting_readiness"))
    tripod = _safe_dict(payload.get("tripod_ai_report_package"))
    rows = ''.join(
        f'<tr><td>{html.escape(_safe_str(_safe_dict(item).get("item")).replace("_", " ").title())}</td><td>{html.escape(_safe_str(_safe_dict(item).get("status")))}</td></tr>'
        for item in _safe_list(tripod.get("items"))
    )
    return (
        f'<p><strong>TRIPOD+AI report-package coverage:</strong> {tripod.get("present", 0)}/{tripod.get("assessed", 0)} locally assessable probes.</p>'
        f'<table><thead><tr><th>Readiness probe</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>'
        '<p class="boundary">Internal readiness screen only. It is not a formal TRIPOD+AI compliance statement or a formal PROBAST+AI risk-of-bias assessment; all four PROBAST+AI domains still require expert appraisal.</p>'
    )


def _english_only_html(document: str) -> str:
    """Remove legacy bilingual markup and keep the public report English-only."""
    document = re.sub(r'<div class="language-toggle"[^>]*>.*?</div>', '', document, flags=re.DOTALL)
    for tag in ("span", "p", "div"):
        document = re.sub(
            rf'<{tag} class="[^"]*lang-zh[^"]*">.*?</{tag}>',
            '',
            document,
            flags=re.DOTALL,
        )
    for tag in ("span", "p", "div"):
        document = re.sub(
            rf'<{tag} class="[^"]*lang-en[^"]*">(.*?)</{tag}>',
            r'\1',
            document,
            flags=re.DOTALL,
        )
    document = re.sub(
        r'const zhLabels=.*?setLanguage\(["\']en["\']\);',
        '',
        document,
        flags=re.DOTALL,
    )
    # Guard against Chinese fragments coming from cached bilingual narratives.
    document = re.sub(r'[\u3400-\u9fff]+', '', document)
    return document


def render_evidence_html(
    report_context: Dict[str, Any],
    llm_sections: Dict[str, Any],
    bundle: Dict[str, Any],
    report_dir: str,
    run_root: str,
) -> str:
    """Render the branded, self-contained HTML shell with linked local assets."""

    raw_llm_sections = llm_sections
    report_path = Path(report_dir)
    root = Path(run_root).resolve()
    report_path.mkdir(parents=True, exist_ok=True)
    if not _safe_dict(bundle.get("contract_manifests")):
        _ensure_phase0_prior_atlas(report_context, root)
        bundle["contract_manifests"] = write_contract_package(report_context, bundle, str(root), str(report_path))
        contract_validation = _safe_dict(bundle["contract_manifests"].get("validation"))
        quality = _safe_dict(bundle.get("report_quality"))
        quality.setdefault("checks", {})["contract_manifests"] = contract_validation
        quality["contract_manifest_paths"] = bundle["contract_manifests"].get("paths", {})
        if not bool(contract_validation.get("valid")):
            quality["status"] = "FAIL"
        bundle["report_quality"] = quality
    output_path = report_path / "report.html"
    run = _safe_dict(report_context.get("run"))
    phase0 = _safe_dict(report_context.get("phase0"))
    phase0_counts = _phase0_prior_counts(phase0)
    phase1 = _safe_dict(report_context.get("phase1"))
    phase2 = _safe_dict(report_context.get("phase2"))
    validation = _safe_dict(bundle.get("validation_scope"))
    quality = _safe_dict(bundle.get("report_quality"))
    preprocessing_summary = _safe_dict(phase1.get("preprocessing_summary"))
    preprocessing_report = _safe_dict(phase1.get("preprocessing_report"))
    preprocessing = {**preprocessing_summary, **preprocessing_report}
    zero_block = _safe_dict(preprocessing_summary.get("zero_handling"))
    scaling_block = _safe_dict(preprocessing_summary.get("normalization_scaling"))
    imputation_block = _safe_dict(preprocessing_summary.get("imputation"))
    zero_handling_display = _safe_str(preprocessing.get("zero_handling_strategy") or zero_block.get("strategy") or "not recorded")
    imputation_display = _safe_str(preprocessing.get("imputation_method") or imputation_block.get("method") or "not recorded")
    normalization_display = _safe_str(preprocessing.get("normalization_method") or scaling_block.get("normalization_method") or "not recorded")
    transformation_display = _safe_str(preprocessing.get("transformation_method") or scaling_block.get("transformation_method") or "not recorded")
    outlier_display = _safe_str(preprocessing.get("outlier_method") or scaling_block.get("outlier_method") or "not recorded")
    threshold_component, threshold_json = _threshold_explorer(report_context, bundle)
    run_id = _safe_str(run.get("run_id") or "unresolved-run")
    winner_features = _safe_list(phase2.get("winner_features"))
    profile = _safe_dict(bundle.get("dataset_profile"))
    disease = _safe_str(profile.get("label") or phase0.get("disease_name") or _safe_dict(phase2.get("search_summary")).get("disease_name") or "Unspecified condition")
    feature_rows_data = _feature_rows(phase2)
    metric_registry = _safe_dict(bundle.get("metric_registry"))
    llm_sections = _canonicalize_llm_sections(llm_sections, metric_registry, feature_rows_data)
    narrative_registry = _safe_dict(bundle.get("scientific_narrative_registry"))
    narrative_blocks = _safe_dict(narrative_registry.get("blocks"))

    def narrative_html(key: str) -> str:
        return _narrative_html_block(_safe_dict(narrative_blocks.get(key)), version="full")

    limitations_html = "".join(
        f'<li><span class="severity severity-{html.escape(_safe_str(_safe_dict(item).get("severity")))}">{html.escape(_safe_str(_safe_dict(item).get("severity")))}</span>{html.escape(_safe_str(_safe_dict(item).get("message")))}</li>'
        for item in _safe_list(bundle.get("limitations_registry"))
    ) or "<li>No automatically registered limitation was found.</li>"
    feature_rows = "".join(
        f"<tr><td>{index}</td><td>{html.escape(_safe_str(row.get('canonical_name')))}<br><small>Raw: <code>{html.escape(_safe_str(row.get('raw_column_name')) or 'Not recorded')}</code></small></td><td>{html.escape(_safe_str(row.get('role')))}<br><small>MSI: {html.escape(_safe_str(row.get('msi_identification_level')) or 'Not recorded')} · Ion: {html.escape(_safe_str(row.get('ion_mode')) or 'Not recorded')} · Unit: {html.escape(_safe_str(row.get('unit')) or 'Not recorded')}</small></td><td>{html.escape(', '.join(_safe_list(row.get('hmdb_ids'))) or 'Not recorded')}</td></tr>"
        for index, row in enumerate(feature_rows_data, 1)
    ) or '<tr><td colspan="4">Winner feature list unavailable.</td></tr>'
    audit_rows = _audit_table(bundle)
    contract_manifest = _safe_dict(_safe_dict(bundle.get("contract_manifests")).get("figure_manifest"))
    study_flow_figure = _contract_figure_cards(
        report_context, contract_manifest, root, report_path, include_ids={"phase_flow_overview"}
    )
    prior_atlas_figure = _contract_figure_cards(
        report_context, contract_manifest, root, report_path, include_ids={"phase0_prior_evidence_atlas"}
    )
    selection_inline_figures = _contract_figure_cards(
        report_context, contract_manifest, root, report_path, include_ids={"selection_stability"}
    )
    performance_figures = _contract_figure_cards(
        report_context, contract_manifest, root, report_path, section_id="model_and_validation", displays={"inline_main_text"}
    )
    clinical_figures = _contract_figure_cards(
        report_context, contract_manifest, root, report_path, section_id="clinical_utility", displays={"inline_main_text"}
    )
    biological_figures = _contract_figure_cards(
        report_context, contract_manifest, root, report_path, section_id="biological_interpretation", displays={"inline_main_text"}
    )
    supplementary_figures = _contract_figure_cards(
        report_context, contract_manifest, root, report_path, displays={"supplementary", "appendix", "gallery_only"}
    )
    results_narrative = narrative_html("performance_summary")
    calibration_narrative = narrative_html("calibration_summary")
    incremental_narrative = narrative_html("incremental_value_summary")
    clinical_narrative = narrative_html("clinical_summary")
    biology_narrative = narrative_html("biology_summary")
    discovery_narrative = narrative_html("discovery_summary")
    study_narrative = narrative_html("study_context")
    conclusion_narrative = narrative_html("conclusion_summary")
    prior_count = phase0_counts["confirmed_biomarkers"]
    final_prior_count = phase0_counts["final_priors"]
    prior_atlas_note = (
        "The atlas shows the retained Phase 0 evidence scores used for biological anchoring."
        if prior_count
        else "The atlas documents screened literature evidence even though no candidate satisfied the confirmed-prior criterion."
    )
    methodology_narrative = _language_blocks(
        _narrative(llm_sections, "methodology_workflow"),
        _zh_narrative(report_context, bundle, "methodology_workflow"),
    )
    claim_graph = _claim_graph_html(bundle, report_path)
    metric_cards = _metric_cards(bundle)
    evaluation_table = _evaluation_evidence_table(bundle)
    qc_table, selection_table, model_table, _repro_table = _enhancement_tables(bundle)
    association_table = _association_table(bundle)
    biomarker_evidence_table = _biomarker_evidence_table(bundle, feature_rows_data)
    maturity_panel = _maturity_panel(bundle)
    identity_cards = _identity_cards(bundle)
    prevalence_scenario_table = _prevalence_scenario_table(bundle)
    reporting_readiness_table = _reporting_readiness_table(bundle)
    maturity = _safe_dict(_safe_dict(bundle.get("enhancements")).get("evidence_maturity"))
    profiles = _safe_list(_safe_dict(_safe_dict(bundle.get("enhancements")).get("cohort_qc_snapshot")).get("profiles"))
    assessed_profiles = [_safe_dict(item) for item in profiles if _safe_dict(item).get("status") == "ASSESSED"]
    cohort_n = sum(int(_safe_dict(item).get("n_samples") or 0) for item in assessed_profiles) or "Not archived"
    class_counts: Dict[str, int] = {}
    for profile_item in assessed_profiles:
        for label, count in _safe_dict(profile_item.get("class_counts")).items():
            class_counts[str(label)] = class_counts.get(str(label), 0) + int(count or 0)
    class_count_display = " / ".join(f"{key}: {value}" for key, value in class_counts.items()) or "Not archived"
    quality_class = html.escape(_safe_str(quality.get("status", "PARTIAL")).lower())
    source_placeholder_hits = _placeholder_hits(raw_llm_sections)
    placeholder_hits = _placeholder_hits(llm_sections)
    if placeholder_hits:
        quality["status"] = "FAIL"
        quality.setdefault("checks", {})["reader_facing_placeholder_scan"] = {"pass": False, "hits": placeholder_hits}
        quality_class = "fail"
    else:
        quality.setdefault("checks", {})["reader_facing_placeholder_scan"] = {"pass": True, "hits": []}
    quality.setdefault("checks", {})["source_placeholder_scan"] = {
        "pass": not source_placeholder_hits,
        "hits": source_placeholder_hits,
        "policy": "Source narrative placeholders may be deterministically sanitized; only reader-facing residual placeholders fail the report gate.",
    }
    enhancement_statuses = {
        key: _safe_str(_safe_dict(value).get("status") or "NOT_ASSESSED")
        for key, value in _safe_dict(bundle.get("enhancements")).items()
    }
    quality.setdefault("checks", {})["enhancement_statuses"] = enhancement_statuses
    css = f'''
    :root{{--navy:{BRAND['navy']};--teal:{BRAND['teal']};--burgundy:{BRAND['burgundy']};--ink:{BRAND['ink']};--muted:{BRAND['muted']};--line:{BRAND['line']};--wash:{BRAND['wash']};}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} section{{scroll-margin-top:110px}} body{{margin:0;color:var(--ink);background:#eef3f5;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.65}}
    .site-header{{position:sticky;top:0;z-index:20;background:rgba(24,35,58,.97);color:white;border-bottom:3px solid var(--teal);box-shadow:0 8px 30px rgba(10,25,42,.18)}}
    .header-inner{{max-width:1480px;margin:auto;padding:12px 28px;display:flex;align-items:center;gap:16px}} .brand-mark{{width:48px;height:48px;flex:none}} .brand-name{{font-family:Georgia,serif;font-size:20px;letter-spacing:.02em}} .brand-sub{{font-size:12px;color:#cbd5df}}
    .header-meta{{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}} .chip{{border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:4px 10px;font-size:11px;letter-spacing:.04em}}
    .layout{{max-width:1480px;margin:auto;display:grid;grid-template-columns:250px minmax(0,1fr);gap:28px;padding:30px 28px 80px}} nav{{position:sticky;top:92px;align-self:start;background:white;border:1px solid var(--line);border-radius:14px;padding:18px;max-height:calc(100vh - 120px);overflow:auto}} nav a{{display:block;color:#445166;text-decoration:none;padding:7px 10px;border-left:2px solid transparent;font-size:13px}} nav a:hover{{color:var(--teal);border-left-color:var(--teal);background:var(--wash)}}
    main{{min-width:0}} .hero{{background:linear-gradient(135deg,var(--navy),#263d5e);color:white;border-radius:20px;padding:48px;margin-bottom:24px;position:relative;overflow:hidden}} .hero:after{{content:"";position:absolute;width:340px;height:340px;border:1px solid rgba(255,255,255,.12);border-radius:50%;right:-100px;top:-140px;box-shadow:0 0 0 60px rgba(19,138,134,.08),0 0 0 120px rgba(19,138,134,.04)}} .eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:#9be0dc}} h1,h2,h3{{font-family:Georgia,"Times New Roman",serif}} h1{{font-size:44px;line-height:1.12;margin:10px 0 18px;max-width:880px}} .hero p{{max-width:800px;color:#dce6ef}}
    .hero-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:28px;position:relative;z-index:1}} .hero-grid div{{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);border-radius:10px;padding:12px}} .hero-grid span{{display:block;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#a9bbc9}} .hero-grid strong{{font-size:15px}}
    .maturity-grid,.identity-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--line);border-left:1px solid var(--line);margin:18px 0}} .maturity-grid>div,.identity-card{{padding:14px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);background:white}} .maturity-grid span,.maturity-grid small{{display:block;color:var(--muted);font-size:11px}} .maturity-grid strong{{display:block;color:var(--navy);font-size:18px;margin:3px 0}} .identity-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}} .identity-card h3{{margin:0 0 8px;font-size:16px}} .identity-card p{{font-size:11px;margin:3px 0}}
    section{{background:white;border:1px solid var(--line);border-radius:16px;padding:32px;margin:20px 0;box-shadow:0 4px 20px rgba(40,60,80,.05)}} section>h2{{font-size:28px;color:var(--navy);margin:0 0 8px;padding-bottom:12px;border-bottom:1px solid var(--line)}} .section-deck{{color:var(--muted);max-width:900px;margin-top:0}}
    .metric-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:20px}} .metric-card{{border:1px solid var(--line);border-radius:12px;padding:18px;background:linear-gradient(180deg,#fff,var(--wash))}} .metric-value{{font-family:Georgia,serif;font-size:34px;color:var(--navy);margin:10px 0 4px}} .metric-card p{{margin:0;font-size:14px}} .metric-card small{{display:block;color:var(--muted);margin-top:10px}}
    .scope,.status{{display:inline-block;border-radius:999px;padding:3px 8px;font-size:10px;font-weight:700;letter-spacing:.06em}} .scope-internal{{background:#e8f1fb;color:#235a8f}} .scope-holdout{{background:#e5f7f5;color:#087b74}} .scope-external{{background:#eaf6e8;color:#357a31}} .scope-missing{{background:#f7e9ed;color:var(--burgundy)}}
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} .evidence-box{{background:var(--wash);border-left:4px solid var(--teal);padding:16px 18px;border-radius:4px 10px 10px 4px}} .narrative-block{{background:#fbfcfd;border-left:3px solid var(--teal);padding:14px 18px;margin:18px 0 22px}} .narrative-block p{{margin:5px 0;font-size:14px}} .narrative-scope{{color:var(--teal);font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px}} .boundary{{color:var(--burgundy);font-size:13px}}
    table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}} th{{text-align:left;background:var(--navy);color:white;padding:10px}} td{{padding:10px;border-bottom:1px solid var(--line);vertical-align:top}} .table-scroll{{overflow-x:auto}} code{{font-family:"SFMono-Regular",Consolas,monospace;font-size:11px;word-break:break-all}}
    .figure-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .figure-card{{border:0;border-radius:0;padding:0;overflow:visible;background:transparent}} .figure-card img{{width:100%;height:auto;display:block;background:white;border:0}} .figure-kicker{{font-size:10px;color:var(--teal);font-weight:700;text-transform:uppercase;letter-spacing:.1em}} .figure-card h3{{margin:4px 0 12px}} .caption{{font-family:Georgia,serif;font-size:13px;color:#536170}} .asset-links a{{font-size:11px;color:var(--teal)}}
    .threshold-explorer{{min-width:0;overflow:hidden;background:var(--wash);padding:20px;border-radius:12px}} input[type=range]{{width:100%;accent-color:var(--teal)}} .threshold-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-top:16px}} .threshold-grid div{{min-width:0;background:white;border:1px solid var(--line);padding:12px;border-radius:8px}} .threshold-grid span{{display:block;color:var(--muted);font-size:11px;overflow-wrap:anywhere}} .threshold-grid strong{{font-size:20px;color:var(--navy);overflow-wrap:anywhere}} .threshold-explorer .table-scroll{{max-width:100%;overflow-x:auto}} .threshold-explorer table{{min-width:760px;table-layout:fixed}} .threshold-explorer th,.threshold-explorer td{{white-space:nowrap;font-size:11px;padding:8px 7px}}
    .limitations{{padding-left:0;list-style:none}} .limitations li{{padding:10px 0;border-bottom:1px solid var(--line)}} .severity{{display:inline-block;min-width:74px;margin-right:10px;text-transform:uppercase;font-size:9px;font-weight:700;color:white;padding:3px 7px;border-radius:999px;text-align:center}} .severity-high{{background:var(--burgundy)}} .severity-medium{{background:#a46616}} .severity-information{{background:var(--teal)}}
    .status-pass{{background:#eaf6e8;color:#357a31}} .status-partial,.status-not_assessed,.status-missing_required{{background:#fff1dc;color:#8a5510}} .status-not_applicable,.status-not_requested{{background:#eef1f4;color:#536170}} .status-fail{{background:#f7e9ed;color:var(--burgundy)}} .quality-{quality_class}{{border-left:0}}
    .report-footer{{background:var(--navy);color:#cbd5df;padding:24px 28px;font-size:11px}} .footer-inner{{max-width:1480px;margin:auto;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}} .empty{{color:var(--muted);font-style:italic}} .bullet{{margin:4px 0}}
    @media(max-width:980px){{.layout{{grid-template-columns:1fr}}nav{{display:none}}.metric-grid,.figure-grid,.two-col,.claim-grid{{grid-template-columns:1fr}}.hero-grid,.threshold-grid{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:34px}}}}
    @media print{{body{{background:white}}.site-header,nav{{display:none}}.layout{{display:block;padding:0}}section,.hero{{break-inside:avoid;box-shadow:none}}.asset-links{{display:none}}}}
    /* v2.1: restrained academic layout and deterministic bilingual view */
    body{{background:#fbfcfd;font-family:Arial,"Helvetica Neue",sans-serif;line-height:1.62}} .site-header{{position:sticky;background:rgba(255,255,255,.98);color:var(--navy);border-bottom:1px solid var(--line);box-shadow:none}} .header-inner{{max-width:1320px;padding:10px 28px}} .brand-mark{{width:38px;height:38px}} .brand-name{{font-family:Arial,"Helvetica Neue",sans-serif;font-size:17px;font-weight:700;letter-spacing:0}} .brand-sub{{color:var(--muted);font-size:11px}} .header-meta{{margin-left:auto}} .language-toggle{{display:inline-flex;border:1px solid var(--line);border-radius:6px;padding:2px;background:white}} .language-toggle button{{border:0;background:transparent;color:var(--muted);padding:5px 9px;font-size:11px;cursor:pointer;border-radius:4px}} .language-toggle button[aria-pressed="true"]{{background:var(--navy);color:white}}
    .layout{{max-width:1320px;grid-template-columns:205px minmax(0,1fr);gap:34px;padding-top:26px}} nav{{top:72px;border:0;border-left:1px solid var(--line);border-radius:0;padding:8px 0;background:transparent;box-shadow:none}} nav strong{{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);padding:0 12px}} nav a{{font-size:12px;padding:6px 12px}} .hero{{background:white;color:var(--ink);border:1px solid var(--line);border-left:4px solid var(--teal);border-radius:4px;padding:38px 42px;box-shadow:none}} .hero:after{{display:none}} .eyebrow{{color:var(--teal);font-weight:700}} h1,h2,h3{{font-family:Georgia,"Times New Roman",serif}} h1{{font-size:38px;color:var(--navy);max-width:760px}} .hero p{{color:var(--muted)}} .hero-grid{{grid-template-columns:repeat(4,1fr);gap:0;border-top:1px solid var(--line);margin-top:24px}} .hero-grid div{{background:transparent;border:0;border-right:1px solid var(--line);border-radius:0;padding:14px 14px 0 0;margin-right:14px}} .hero-grid div:last-child{{border-right:0}} .hero-grid span{{color:var(--muted)}} .hero-grid strong{{font-size:13px;color:var(--ink)}} .format-note{{font-size:11px;color:var(--muted);margin:14px 0 0;max-width:900px}}
    section{{border:0;border-top:1px solid var(--line);border-radius:0;padding:30px 0;margin:0;box-shadow:none;background:transparent}} section>h2{{font-size:25px;border:0;padding:0;margin-bottom:8px}} .section-deck{{font-size:14px}} .metric-grid{{grid-template-columns:repeat(3,minmax(0,1fr));gap:0;border-top:1px solid var(--line);border-left:1px solid var(--line)}} .metric-card{{border:0;border-right:1px solid var(--line);border-bottom:1px solid var(--line);border-radius:0;padding:16px;background:white}} .metric-value{{font-size:28px}} .metric-card p{{font-size:12px;min-height:38px}} .metric-card small{{font-size:11px}} table{{font-size:12px;border-top:1px solid var(--line)}} th{{background:#f1f4f6;color:var(--navy);font-weight:700}} td{{padding:9px}} .evidence-box{{background:#f7faf9;border-left-width:3px;border-radius:0}} .figure-card{{border:0;border-radius:0;background:transparent}} .figure-grid{{gap:14px}} .threshold-explorer{{border:1px solid var(--line);border-radius:0;background:#f7faf9}} .threshold-grid div{{border-radius:0}} .report-footer{{background:#fff;color:var(--muted);border-top:1px solid var(--line)}} .quality-partial{{border-left:0}} .lang-zh{{display:none}} body[data-language="zh"] .lang-en{{display:none}} body[data-language="zh"] .lang-zh{{display:block}} body[data-language="zh"] .lang-zh.inline{{display:inline}} @media(max-width:980px){{.hero-grid{{grid-template-columns:repeat(2,1fr)}}.metric-grid{{grid-template-columns:1fr}}.claim-grid{{grid-template-columns:1fr}}}} @media(max-width:640px){{.claim-row{{grid-template-columns:1fr;gap:4px}}.claim-row .asset-links{{margin-top:2px}}}}
    .study-flow{{display:flex;align-items:stretch;gap:10px;padding:24px 8px;border:0;background:transparent;overflow-x:auto}} .study-flow-node{{flex:1 1 0;min-width:150px;padding:16px;border:1px solid #c8dfdc;background:white;display:flex;flex-direction:column;gap:7px}} .study-flow-number{{font-family:Georgia,serif;font-size:20px;color:var(--teal)}} .study-flow-node strong{{font-size:13px;color:var(--navy)}} .study-flow-node small{{font-size:11px;line-height:1.45;color:var(--muted)}} .study-flow-arrow{{align-self:center;color:var(--teal);font-size:22px;font-weight:700}} .ladder-grid{{display:block}} .ladder-step{{display:none}} #claim-graph{{padding-top:18px;padding-bottom:18px}} #claim-graph .section-deck{{font-size:12px;margin-bottom:8px}} .claim-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:28px}} .claim-row{{display:grid;grid-template-columns:minmax(0,1fr) minmax(180px,240px);gap:12px;align-items:start;padding:7px 0;border-bottom:1px solid var(--line);min-width:0}} .claim-row strong{{font-size:11px;line-height:1.35}} .claim-row .scope{{font-size:9px;padding:2px 6px}} .claim-row .asset-links{{font-size:10px;white-space:normal;overflow-wrap:anywhere;word-break:break-word;min-width:0}} .claim-evidence-links{{display:flex;flex-direction:column;align-items:flex-start;gap:3px;line-height:1.3}} .claim-evidence-links span{{display:block;max-width:100%}} .claim-evidence-links a{{display:block;overflow-wrap:anywhere;word-break:break-word}}
    .inline-evidence{{grid-template-columns:1fr}} .single-phase-flow .figure-card .figure-kicker,.single-phase-flow .figure-card h3,.single-phase-flow .figure-card .caption{{display:none}} details{{border-top:1px solid var(--line);padding:12px 0}} details summary{{cursor:pointer;color:var(--navy);font-weight:700;font-size:13px;list-style-position:outside}} details[open] summary{{margin-bottom:14px}} .technical-stack>details:first-child{{border-top:0}} .supplementary-grid{{margin-top:16px}} .report-note{{border-left:3px solid var(--teal);padding:10px 14px;background:#f7faf9;color:#475467;font-size:13px}} @media(max-width:800px){{.maturity-grid,.identity-grid{{grid-template-columns:1fr}}}}
    /* v2.3: keep figures readable without letting one raster dominate the page. */
    .figure-card img{{max-height:480px;width:auto;max-width:100%;object-fit:contain;margin:0 auto}} .single-phase-flow .figure-card img{{max-height:300px;width:100%}} .supplementary-grid .figure-card img{{max-height:380px}} .figure-card .caption{{max-width:900px;margin:8px auto 0}} @media(max-width:640px){{.figure-card img,.single-phase-flow .figure-card img{{max-height:300px}}}}
    section p,section li,section td,section th{{text-align:justify;text-justify:inter-word}} .figure-card p, .figure-card h3, .section-deck{{text-align:justify}}
    '''

    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MetaboAgent Evidence-to-Decision Report | {html.escape(disease)}</title><style>{css}</style></head>
<body>
<header class="site-header"><div class="header-inner">
<svg class="brand-mark" viewBox="0 0 64 64" role="img" aria-label="MetaboAgent mark"><circle cx="18" cy="20" r="6" fill="#55c7bf"/><circle cx="45" cy="14" r="5" fill="#fff"/><circle cx="47" cy="43" r="7" fill="#55c7bf"/><circle cx="17" cy="46" r="4" fill="#fff"/><path d="M23 19L40 15M21 25L43 39M21 43L41 43" stroke="#fff" stroke-width="2.5"/><path d="M42 32l7 7 9-14" fill="none" stroke="#55c7bf" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>
<div><div class="brand-name">MetaboAgent</div><div class="brand-sub">Evidence-to-Decision Report</div></div>
<div class="header-meta"><div class="language-toggle" aria-label="Language selector"><button type="button" data-language="en" aria-pressed="true">English</button><button type="button" data-language="zh" aria-pressed="false">中文</button></div></div>
</div></header>
<div class="layout"><nav><div class="lang-en"><strong>Report contents</strong>
<a href="#executive">Executive summary</a><a href="#study-design">Study flow</a><a href="#cohort">Cohort and QC</a><a href="#prior">Biomarker discovery</a><a href="#model">Model development</a><a href="#performance">Validation</a><a href="#clinical">Clinical utility</a><a href="#interpretability">Interpretation</a><a href="#conclusion">Conclusion</a><a href="#technical">Technical appendix</a></div><div class="lang-zh"><strong>报告目录</strong>
<a href="#executive">主要摘要</a><a href="#study-design">研究流程</a><a href="#cohort">队列与质控</a><a href="#prior">标志物发现</a><a href="#model">模型开发</a><a href="#performance">验证</a><a href="#clinical">临床效用</a><a href="#interpretability">结果解释</a><a href="#conclusion">结论</a><a href="#technical">技术附录</a></div></nav>
<main>
<div class="hero"><div class="eyebrow"><span class="lang-en inline">MetaboAgent scientific delivery</span><span class="lang-zh inline">MetaboAgent 科学交付报告</span></div><h1><span class="lang-en">{html.escape(disease)}<br>Evidence-to-Decision Report</span><span class="lang-zh">{html.escape(disease)}<br>证据到决策报告</span></h1><p class="lang-en">A traceable account of data handling, biomarker prioritisation, model development, validation scope, clinical operating characteristics and the boundaries of interpretation.</p><p class="lang-zh">对数据处理、生物标志物优选、模型开发、验证范围、阈值表现与解释边界的可追溯总结。</p>
<div class="hero-grid"><div><span>Cohort</span><strong>{html.escape(_safe_str(cohort_n))} samples</strong></div><div><span>Classes</span><strong>{html.escape(class_count_display)}</strong></div><div><span>Winner panel</span><strong>{len(winner_features)} features</strong></div><div><span>Validation scope</span><strong>Internal development and holdout</strong></div></div><p class="format-note">HTML is the interactive evidence dossier; the companion PDF is the curated scientific summary. Both are generated from the same canonical evidence bundle. The JSON/CSV/manifests form the machine-auditable archive.</p></div>

<section id="executive" class="quality-{quality_class}"><h2><span class="lang-en">Executive evidence summary</span><span class="lang-zh">主要证据摘要</span></h2><p class="section-deck lang-en">The central findings and their validation scope, constrained to archived evidence.</p><p class="section-deck lang-zh">基于归档证据概括核心结果及其验证范围。</p>{maturity_panel}<div class="metric-grid">{metric_cards}</div>{conclusion_narrative}<p class="boundary lang-en">Artifact integrity is not a scientific-quality verdict. All performance estimates are development or internal-holdout evidence unless an independent external cohort is explicitly declared.</p><p class="boundary lang-zh">工件完整性并不等同于科学质量结论。除非明确声明独立外部队列，否则所有性能估计均属于开发期或内部留出证据。</p></section>
<section id="study-design"><h2><span class="lang-en">Study design and Phase 0–4 flow</span><span class="lang-zh">研究设计与 Phase 0–4 流程</span></h2><p class="section-deck lang-en">The complete evidence path from prior-biomarker discovery to the final report; counts are resolved from the current run rather than hard-coded figure slots.</p><p class="section-deck lang-zh">从先验标志物发现到最终报告的完整证据路径；各阶段数量由当前运行自动解析，而非写死图位。</p>{study_narrative}<div class="inline-figure-evidence single-phase-flow">{study_flow_figure}</div></section>

<section id="cohort"><h2><span class="lang-en">Cohort and data profile</span><span class="lang-zh">队列与数据概况</span></h2><table><tbody><tr><th>Study ID</th><td>{html.escape(_safe_str(profile.get('study_id') or 'not catalogued'))}</td></tr><tr><th>Specimen</th><td>{html.escape(_safe_str(profile.get('specimen') or 'not catalogued'))}</td></tr><tr><th>Platform/source</th><td>{html.escape(_safe_str(profile.get('platform') or preprocessing.get('platform_source') or 'not recorded'))}</td></tr><tr><th>Data level</th><td>{html.escape(_safe_str(profile.get('data_level') or preprocessing.get('data_level') or 'not recorded'))}</td></tr><tr><th>Study design/task</th><td>{html.escape(_safe_str(profile.get('design')))}; {html.escape(_safe_str(profile.get('task')))}</td></tr><tr><th>Dataset fingerprint</th><td><code>{html.escape(json.dumps(_safe_dict(phase2.get('dataset_fingerprint')), ensure_ascii=False))}</code></td></tr></tbody></table><h3><span class="lang-en">Cohort/QC snapshot</span><span class="lang-zh">队列/QC 快照</span></h3><table><thead><tr><th>Artifact</th><th>Samples</th><th>Features</th><th>Missing-cell fraction</th><th>Class counts</th></tr></thead><tbody>{qc_table}</tbody></table><p class="boundary lang-en">QC values are descriptive snapshots from archived tabular artifacts; preprocessing fit statistics must remain training-only.</p><p class="boundary lang-zh">QC 数值是对归档表格工件的描述性快照；预处理拟合统计量必须仅来自训练数据。</p><p class="boundary lang-en">Catalogue metadata source: user-provided Table 1; see <a href="dataset_profile.json">dataset_profile.json</a>.</p><p class="boundary lang-zh">目录元数据来源：用户提供的 Table 1；见 <a href="dataset_profile.json">dataset_profile.json</a>。</p></section>

<section id="preprocessing"><h2>Preprocessing and quality control</h2><table><tbody><tr><th>Zero handling</th><td>{html.escape(zero_handling_display)}</td></tr><tr><th>Imputation</th><td>{html.escape(imputation_display)}</td></tr><tr><th>Normalisation</th><td>{html.escape(normalization_display)}</td></tr><tr><th>Transformation</th><td>{html.escape(transformation_display)}</td></tr><tr><th>Outlier handling</th><td>{html.escape(outlier_display)}</td></tr></tbody></table><p class="boundary">Values are taken from the archived Phase 1 preprocessing report; no preprocessing step is refit during report generation.</p></section>

<section id="prior"><h2><span class="lang-en">Biomarker discovery and evidence consolidation</span><span class="lang-zh">生物标志物发现与证据整合</span></h2><p class="section-deck lang-en">Prior knowledge is kept distinct from data-driven selection, then joined with train-only stability and association evidence for the final panel.</p><p class="section-deck lang-zh">先验知识与数据驱动筛选分开处理，随后与仅训练集的稳定性及关联证据共同形成最终面板。</p><div class="report-note"><strong>{html.escape(_safe_str(prior_count))}</strong> confirmed prior biomarkers and <strong>{html.escape(_safe_str(final_prior_count))}</strong> final priors were archived. {html.escape(prior_atlas_note)}</div>{discovery_narrative}<div class="figure-grid inline-evidence">{prior_atlas_figure}</div><h3><span class="lang-en">Biomarker identity and evidence cards</span><span class="lang-zh">生物标志物身份与证据卡</span></h3>{identity_cards}<h3><span class="lang-en">Final-panel evidence table</span><span class="lang-zh">最终面板证据表</span></h3><div class="table-scroll"><table><thead><tr><th>Feature</th><th>Role</th><th>HMDB ID</th><th>Selection frequency</th><th>Method consensus</th><th>Hedges g</th><th>Direction</th></tr></thead><tbody>{biomarker_evidence_table}</tbody></table></div><p class="boundary lang-en">Missing identities or statistics remain “Not archived” and are never inferred. Selection frequency reflects Phase 1 candidate stability, while Hedges g is a descriptive train-only association.</p><p class="boundary lang-zh">缺失的标识或统计量保留为“未归档”，不会推断补齐。选择频率代表 Phase 1 候选稳定性；Hedges g 为仅训练集的描述性关联。</p><div class="figure-grid inline-evidence">{selection_inline_figures}</div></section>

<section id="model"><h2><span class="lang-en">Model development and selection</span><span class="lang-zh">模型开发与选择</span></h2><p class="section-deck lang-en">The winning specification is reported as a balance of discrimination, stability and parsimony; the report does not refit or retune the archived model.</p><p class="section-deck lang-zh">最终模型以区分度、稳定性与简约性的平衡进行呈现；报告阶段不会重新拟合或调参。</p><div class="evidence-box"><span class="lang-en"><strong>Selected model:</strong> {html.escape(_safe_str(phase2.get('selected_model') or 'not recorded'))}<br><strong>Search engine:</strong> {html.escape(_safe_str(_safe_dict(phase2.get('search_summary')).get('engine') or 'not recorded'))}</span><span class="lang-zh"><strong>入选模型：</strong>{html.escape(_safe_str(phase2.get('selected_model') or '未记录'))}<br><strong>搜索引擎：</strong>{html.escape(_safe_str(_safe_dict(phase2.get('search_summary')).get('engine') or '未记录'))}</span></div><h3><span class="lang-en">Compact model specification</span><span class="lang-zh">紧凑模型规格</span></h3><table><tbody>{model_table}</tbody></table><details><summary><span class="lang-en inline">Analysis rationale and archived workflow narrative</span><span class="lang-zh inline">分析依据与归档流程叙述</span></summary>{methodology_narrative}</details></section>

<section id="performance"><h2><span class="lang-en">Performance and uncertainty</span><span class="lang-zh">性能与不确定性</span></h2><h3><span class="lang-en">Evaluation evidence</span><span class="lang-zh">评估证据</span></h3><table><thead><tr><th>Evaluation set</th><th>AUC / status</th><th>Scope</th><th>Interpretation boundary</th></tr></thead><tbody>{evaluation_table}</tbody></table>{results_narrative}<div class="figure-grid inline-evidence">{performance_figures}</div>{calibration_narrative}{incremental_narrative}<p class="boundary lang-en">Development CV and an internal holdout are complementary development evidence. Neither is a substitute for independently declared external validation.</p><p class="boundary lang-zh">开发集 CV 与内部留出集是互补的开发期证据，二者均不能替代独立声明的外部验证。</p></section>

<section id="clinical"><h2><span class="lang-en">Exploratory operating characteristics and decision analysis</span><span class="lang-zh">探索性运行特征与决策分析</span></h2>{threshold_component}{prevalence_scenario_table}{clinical_narrative}<div class="figure-grid inline-evidence">{clinical_figures}</div></section>

<section id="interpretability"><h2><span class="lang-en">Model interpretability and biological context</span><span class="lang-zh">模型可解释性与生物学背景</span></h2>{biology_narrative}<div class="evidence-box"><span class="lang-en"><strong>Interpretation level:</strong> {html.escape(_safe_str(_safe_dict(bundle.get('mechanistic_evidence_registry')).get('interpretation_level')))}. See <code>mechanistic_evidence_registry.json</code> for feature-level provenance.</span><span class="lang-zh"><strong>解释层级：</strong>{html.escape(_safe_str(_safe_dict(bundle.get('mechanistic_evidence_registry')).get('interpretation_level')))}。特征层面的证据来源见 <code>mechanistic_evidence_registry.json</code>。</span></div><p class="boundary lang-en">SHAP and spline analyses describe model behaviour or association structure. They do not establish causal, protective or mechanistic effects.</p><p class="boundary lang-zh">SHAP 与样条分析描述模型行为或关联结构，不能证明因果、保护效应或疾病机制。</p><div class="figure-grid inline-evidence">{biological_figures}</div></section>

<section id="conclusion"><h2><span class="lang-en">Integrated conclusion and next evidence step</span><span class="lang-zh">综合结论与下一步证据</span></h2>{conclusion_narrative}<p class="section-deck lang-en">Recommended next evidence step: independent external validation with complete analytical metadata and population-specific recalibration review.</p><p class="section-deck lang-zh">建议的下一步证据：在完整分析元数据基础上进行独立外部验证，并评估目标人群特异性再校准。</p></section>

<section id="technical"><h2><span class="lang-en">Technical appendix</span><span class="lang-zh">技术附录</span></h2><p class="section-deck lang-en">Traceability and reproducibility material is retained in full but collapsed by default so that it does not interrupt the scientific narrative.</p><p class="section-deck lang-zh">追溯与可复现材料完整保留，但默认折叠，避免打断科学叙事。</p><div class="technical-stack"><details><summary>TRIPOD+AI / PROBAST+AI readiness</summary>{reporting_readiness_table}</details><details id="claim-graph"><summary><span class="lang-en inline">Claim-to-evidence graph</span><span class="lang-zh inline">结论到证据图谱</span></summary><div class="claim-grid">{claim_graph}</div></details><details><summary><span class="lang-en inline">Supplementary figures</span><span class="lang-zh inline">补充图</span></summary><div class="figure-grid supplementary-grid">{supplementary_figures}</div></details><details><summary><span class="lang-en inline">Reproducibility and integrity audit</span><span class="lang-zh inline">可复现性与完整性审计</span></summary><table><thead><tr><th>Domain</th><th>Status</th><th>Evidence artifact</th></tr></thead><tbody>{audit_rows}</tbody></table><p class="boundary">Audit status measures deterministic artifact availability and declared process evidence. It does not replace expert methodological review.</p></details><details><summary><span class="lang-en inline">Machine-readable artifact index</span><span class="lang-zh inline">机器可读工件索引</span></summary><table><thead><tr><th>Artifact</th><th>Purpose</th></tr></thead><tbody><tr><td><code>report_context_v2.json</code></td><td>Machine-readable report facts</td></tr><tr><td><code>phase_flow_manifest.json</code></td><td>Phase 0–4 data lineage contract</td></tr><tr><td><code>figure_manifest.json</code></td><td>Figure provenance and inline-placement contract</td></tr><tr><td><code>feature_identity_registry.json</code></td><td>Canonical feature identity and analytical metadata</td></tr><tr><td><code>statistical_claim_validation.json</code></td><td>Deterministic numerical-claim checks</td></tr><tr><td><code>prevalence_scenarios.json</code></td><td>Prevalence-adjusted scenario projections</td></tr><tr><td><code>reporting_readiness.json</code></td><td>Internal TRIPOD+AI/PROBAST+AI readiness screen</td></tr><tr><td><code>upstream_evidence_requirements.json</code></td><td>Gaps requiring upstream metadata or reanalysis</td></tr><tr><td><code>report_quality.json</code></td><td>Deterministic artifact publication gate</td></tr></tbody></table></details></div></section>
</main></div>
<footer class="report-footer"><div class="footer-inner"><span>MetaboAgent Evidence-to-Decision Report · Research Use Only</span><span>Run {html.escape(run_id)} · Generated {html.escape(datetime.now(timezone.utc).isoformat())}</span></div></footer>
<script>const thresholdRows={threshold_json};const slider=document.querySelector('#threshold-slider');function f(v){{return v===null||v===undefined?'Not available':Number(v).toFixed(3)}}function updateThreshold(){{if(!slider||!thresholdRows.length)return;const value=Number(slider.value);const row=thresholdRows.reduce((a,b)=>Math.abs(Number(b.threshold)-value)<Math.abs(Number(a.threshold)-value)?b:a);document.querySelector('#threshold-value').textContent=Number(row.threshold).toFixed(2);document.querySelector('#threshold-sensitivity').textContent=f(row.sensitivity);document.querySelector('#threshold-specificity').textContent=f(row.specificity);document.querySelector('#threshold-ppv').textContent=f(row.ppv);document.querySelector('#threshold-npv').textContent=f(row.npv);document.querySelector('#threshold-flagged').textContent=f(row.flagged_rate)}}if(slider){{slider.addEventListener('input',updateThreshold)}}const zhLabels={{"Cohort and data profile":"队列与数据概况","Data partition and leakage control":"数据划分与泄漏控制","Preprocessing and quality control":"预处理与质量控制","Prior evidence and biological anchoring":"先验证据与生物学锚定","Feature engineering and selection":"特征工程与选择","Figures and source assets":"图表与源工件","Limitations and failure registry":"局限性与失败登记","Reproducibility and audit":"可复现性与审计","Artifact appendix":"工件附录","Dataset fingerprint":"数据指纹","Selected Phase 1 features":"Phase 1 入选特征","Winner features":"最终入选特征","Data level":"数据层级","Platform/source":"平台/来源","Development resampling protocol":"开发期重采样流程","Internal holdout artifact":"内部留出集工件","Independent external cohort":"独立外部队列","Training data provenance":"训练数据来源","Evaluation data provenance":"评估数据来源","Same archived artifact":"归档工件是否相同","Boundary warnings":"边界警示","Zero handling":"零值处理","Imputation":"缺失值填补","Normalisation":"标准化","Transformation":"变换","Outlier handling":"离群值处理","Feature":"特征","Role":"角色","Domain":"领域","Status":"状态","Evidence artifact":"证据工件","Artifact":"工件","Purpose":"用途"}};function setLanguage(language){{document.body.dataset.language=language;document.documentElement.lang=language==='zh'?'zh-CN':'en';document.querySelectorAll('h2,h3,th').forEach(node=>{{if(!node.dataset.enText)node.dataset.enText=node.textContent.trim();const english=node.dataset.enText;const chinese=zhLabels[english];if(chinese)node.textContent=language==='zh'?chinese:english;}});document.querySelectorAll('[data-language]').forEach(item=>item.setAttribute('aria-pressed',String(item.dataset.language===language)));updateThreshold();}}document.querySelectorAll('[data-language]').forEach(button=>button.addEventListener('click',()=>setLanguage(button.dataset.language)));setLanguage('en');</script>
</body></html>'''
    if not profile.get("matched"):
        document = re.sub(r'<section id="cohort">.*?</section>', '', document, count=1, flags=re.DOTALL)
        document = document.replace('<a href="#cohort">Cohort and data</a>', '')
        document = document.replace('<a href="#cohort">队列与数据</a>', '')
    document = document.replace(
        '<p class="boundary lang-en">Development CV and an internal holdout are complementary development evidence.',
        '<p class="boundary lang-en">The canonical metric registry is archived in <a href="metric_registry.json">metric_registry.json</a>. Development CV and an internal holdout are complementary development evidence.',
    )
    document = document.replace("Same archived artifact", "Same training/evaluation artifact")
    document = document.replace("归档工件是否相同", "训练/评估工件是否相同")
    # The detailed partition modules are intentionally omitted from the
    # reader-facing report; remove their legacy language-toggle labels too.
    document = document.replace('"Data partition and leakage control":"数据划分与泄漏控制",', '')
    document = document.replace("<code>{}</code>", "<code>Not archived</code>")
    # Never perform a global replacement on the serialized HTML here: figure
    # PNG/SVG assets are embedded as base64 and may legitimately contain the
    # byte sequence ``N/A``.  Reader-facing placeholders are normalized while
    # narrative/table text is generated, before assets are serialized.
    document = _english_only_html(document)
    output_path.write_text(document, encoding="utf-8")
    # The HTML-level quality checks are only knowable after all text and
    # portable assets have been materialized; refresh the sidecar written
    # before rendering so the delivery package exposes the final gate.
    _write_json(report_path / "report_quality.json", quality)
    return str(output_path)


def render_evidence_pdf(
    report_context: Dict[str, Any],
    bundle: Dict[str, Any],
    report_dir: str,
) -> Optional[str]:
    """Render the static, publication-style companion to the interactive HTML.

    The PDF intentionally uses the same evidence bundle and contract manifests
    as HTML.  Interactions (language toggle, threshold slider, collapsible
    appendix) are flattened into their archived default state, while the core
    claims, phase flow, panel identity, validation metrics, limitations, audit
    status and registered main figures remain available in the same order.
    """

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import (
            Image as RLImage,
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return None

    report_path = Path(report_dir)
    output_path = report_path / "report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run = _safe_dict(report_context.get("run"))
    phase0 = _safe_dict(report_context.get("phase0"))
    phase1 = _safe_dict(report_context.get("phase1"))
    phase2 = _safe_dict(report_context.get("phase2"))
    profile = _safe_dict(bundle.get("dataset_profile")) or _dataset_profile(report_context)
    disease = _safe_str(profile.get("label") or phase0.get("disease_name") or "Unspecified condition")
    run_id = _safe_str(run.get("run_id") or "unresolved-run")
    enhancements = _safe_dict(bundle.get("enhancements"))
    maturity = _safe_dict(enhancements.get("evidence_maturity"))
    identity_registry = _safe_dict(enhancements.get("feature_identity_registry"))
    readiness = _safe_dict(enhancements.get("reporting_readiness"))
    report_quality = _safe_dict(bundle.get("report_quality"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MA_Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=colors.HexColor(BRAND["navy"]), spaceAfter=10))
    styles.add(ParagraphStyle(name="MA_Subtitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor(BRAND["teal"]), spaceBefore=2, spaceAfter=7))
    styles.add(ParagraphStyle(name="MA_H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor(BRAND["navy"]), spaceBefore=14, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name="MA_H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor(BRAND["navy"]), spaceBefore=10, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle(name="MA_Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3, leading=14, textColor=colors.HexColor(BRAND["ink"]), alignment=TA_LEFT, spaceAfter=5))
    styles.add(ParagraphStyle(name="MA_Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=11, textColor=colors.HexColor(BRAND["muted"]), spaceAfter=3))
    styles.add(ParagraphStyle(name="MA_Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=7.7, leading=10, textColor=colors.HexColor(BRAND["muted"]), spaceBefore=4, spaceAfter=8))
    styles.add(ParagraphStyle(name="MA_Table", parent=styles["BodyText"], fontName="Helvetica", fontSize=6.8, leading=9, textColor=colors.HexColor(BRAND["ink"]), wordWrap="CJK"))
    styles.add(ParagraphStyle(name="MA_TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=colors.white, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="MA_FlowPhase", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=colors.HexColor(BRAND["teal"]), spaceAfter=3))
    styles.add(ParagraphStyle(name="MA_FlowTitle", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=colors.HexColor(BRAND["navy"]), spaceAfter=3))
    styles.add(ParagraphStyle(name="MA_Kicker", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.5, leading=9, textColor=colors.HexColor(BRAND["teal"]), uppercase=True, spaceAfter=2))
    styles.add(ParagraphStyle(name="MA_SnapshotLabel", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7, leading=8, textColor=colors.HexColor(BRAND["muted"]), alignment=TA_LEFT, spaceAfter=2))
    styles.add(ParagraphStyle(name="MA_SnapshotValue", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=colors.HexColor(BRAND["navy"]), alignment=TA_LEFT, spaceAfter=0))

    def table_cell(value: Any, *, heading: bool = False) -> Any:
        style = styles["MA_TableHead"] if heading else styles["MA_Table"]
        text = _safe_str(value).replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
        return Paragraph(html.escape(text).replace("\n", "<br/>"), style)

    def body(value: Any) -> Any:
        text = _safe_str(value).replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
        return Paragraph(html.escape(text).replace("\n", "<br/>"), styles["MA_Body"])

    narrative_registry = _safe_dict(bundle.get("scientific_narrative_registry"))
    narrative_blocks = _safe_dict(narrative_registry.get("blocks"))

    def narrative_pdf(key: str, *, version: str = "medium") -> List[Any]:
        block = _safe_dict(narrative_blocks.get(key))
        if not block:
            return []
        scope = _safe_str(block.get("evidence_scope") or "NOT_ASSESSABLE")
        finding = _safe_str(block.get("finding"))
        interpretation = _safe_str(block.get("interpretation"))
        boundary = _safe_str(block.get("boundary"))
        return [
            Paragraph(html.escape(scope), styles["MA_Kicker"]),
            body(f"Finding. {finding}"),
            body(f"Interpretation. {interpretation}"),
            body(f"Boundary. {boundary}"),
        ]

    def image_flowable(path: Optional[Path], max_width: float = 165 * mm, max_height: float = 78 * mm) -> Optional[Any]:
        if path is None or not path.exists() or not path.is_file():
            return None
        try:
            width, height = ImageReader(str(path)).getSize()
            if not width or not height:
                return None
            scale = min(max_width / float(width), max_height / float(height), 1.0)
            return RLImage(str(path), width=width * scale, height=height * scale, hAlign="CENTER")
        except Exception:
            return None

    def compact_run_label(value: str) -> str:
        match = re.search(r"(20\d{6})[_-](\d{6})", value)
        if match:
            date, clock = match.groups()
            return f"Run {date[:4]}-{date[4:6]}-{date[6:]} {clock[:2]}:{clock[2:4]}:{clock[4:]}"
        tail = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
        return f"Run {tail[-24:]}" if tail else "Run current"

    def candidate_run_roots() -> List[Path]:
        roots: List[Path] = []
        for raw in (
            _safe_str(bundle.get("run_root")),
            _safe_str(_safe_dict(report_context.get("run")).get("run_root")),
        ):
            if raw:
                roots.append(Path(raw).expanduser().resolve())
        # The report may be rendered in ``output/reports/.staging/<build>``.
        # Infer the real run root from that stable directory boundary so
        # assets remain resolvable even when report_context contains a mounted
        # /nas path and the current process runs under /home.
        for parent in (report_path, *report_path.parents):
            if parent.name == "output":
                roots.append(parent.parent.resolve())
        unique: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique

    def resolved_figure_preview(figure_id: str, title: str, legacy_id: str = "") -> Optional[Path]:
        """Locate a report-local preview, falling back to a portable PDF preview."""
        evidence_dir = report_path / "evidence"
        if evidence_dir.exists():
            direct = sorted(evidence_dir.glob(f"*_{figure_id}_preview.png"))
            if direct:
                return direct[0]
        legacy_by_id = {
            _safe_str(_safe_dict(item).get("figure_id")): _safe_dict(item)
            for item in _safe_list(_safe_dict(report_context.get("phase3")).get("figures"))
        }
        legacy = legacy_by_id.get(legacy_id, {})
        outputs: List[str] = []
        primary = _safe_str(_safe_dict(legacy.get("primary_output")).get("path")).strip()
        if primary:
            outputs.append(primary)
        outputs.extend(_safe_str(_safe_dict(item).get("path")).strip() for item in _safe_list(legacy.get("all_outputs")))
        manifest_item = next((
            _safe_dict(item) for item in _safe_list(_safe_dict(_read_json(report_path / "figure_manifest.json")).get("figures"))
            if _safe_str(_safe_dict(item).get("figure_id")) == figure_id
        ), {})
        outputs.extend(_safe_str(_safe_dict(item).get("path")).strip() for item in _safe_list(manifest_item.get("outputs")))
        seen: set[str] = set()
        for raw in outputs:
            if not raw or raw in seen:
                continue
            seen.add(raw)
            for root in candidate_run_roots():
                resolved = _resolve_figure_path(root, raw)
                if not resolved.exists():
                    candidate = report_path / raw
                    resolved = candidate if candidate.exists() else resolved
                if not resolved.exists():
                    continue
                # Reuse the HTML resolver's image-first sibling search.  This
                # is the key portability fix: the server may not have
                # pdftoppm, while Phase 3 already archived PNG/SVG siblings.
                vector_candidate: Optional[Path] = None
                for candidate_path in _figure_preview_candidates(root, resolved):
                    if not candidate_path.exists():
                        continue
                    if candidate_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        return candidate_path
                    if candidate_path.suffix.lower() == ".svg":
                        vector_candidate = vector_candidate or candidate_path
                        continue
                    if candidate_path.suffix.lower() == ".pdf":
                        preview_relative = _portable_pdf_preview(report_path, candidate_path)
                        if preview_relative:
                            preview = report_path / Path(preview_relative)
                            if preview.exists():
                                return preview
                # ReportLab's Image flowable cannot consume SVG directly.  Do
                # not return a vector-only candidate here; the release gate
                # should fail rather than create a misleading placeholder.
        return None

    missing_main_figures: List[str] = []

    def figure_block(figure_id: str, title: str, caption: str, legacy_id: str = "") -> List[Any]:
        image_path = resolved_figure_preview(figure_id, title, legacy_id)
        image = image_flowable(image_path)
        if image is None:
            missing_main_figures.append(figure_id)
            return [Paragraph(html.escape(title), styles["MA_H2"]), body("Figure omitted from the static PDF because no portable preview was available; the interactive HTML retains the registered artifact.")]
        return [
            Paragraph(html.escape(title), styles["MA_H2"]),
            image,
            Paragraph(html.escape(caption or "Interpretation is restricted to the registered evidence scope."), styles["MA_Caption"]),
        ]

    def phase_flow_block() -> List[Any]:
        flow = _read_json(report_path / "phase_flow_manifest.json")
        nodes = [_safe_dict(item) for item in _safe_list(flow.get("nodes"))]
        if not nodes:
            return [body("Phase 0-4 flow manifest unavailable.")]
        # Keep the machine-readable manifest granular, but collapse internal
        # substeps (for example Phase 1 preprocessing and stability selection)
        # into one reader-facing card per Phase 0-4 stage. This keeps the PDF
        # aligned with the HTML flow graphic while retaining all counts.
        phase_order = ["phase0", "phase1", "phase2", "phase3", "phase4"]
        phase_labels = {
            "phase0": "Literature, disease knowledge and prior evidence",
            "phase1": "QC, preprocessing and train-only feature selection",
            "phase2": "Multi-objective panel search and model development",
            "phase3": "Prediction, uncertainty, clinical utility and interpretation figures",
            "phase4": "Evidence contract, narrative, inline figures and delivery package",
        }
        grouped: Dict[str, List[Dict[str, Any]]] = {phase: [] for phase in phase_order}
        for node in nodes:
            phase = _safe_str(node.get("phase"))
            if phase in grouped:
                grouped[phase].append(node)
        display_nodes: List[Dict[str, Any]] = []
        for phase in phase_order:
            candidates = grouped.get(phase) or []
            if not candidates:
                continue
            headlines = [
                _safe_str(_safe_dict(item.get("stage_summary")).get("headline")).strip()
                for item in candidates
                if _safe_str(_safe_dict(item.get("stage_summary")).get("headline")).strip()
            ]
            item_rows: List[str] = []
            seen_items = set()
            for candidate in candidates:
                for item in _safe_list(_safe_dict(candidate.get("stage_summary")).get("items")):
                    label = _safe_str(_safe_dict(item).get("label")).strip()
                    value = _safe_str(_safe_dict(item).get("value")).strip()
                    text = f"{label}: {value}" if label else value
                    if text and text not in seen_items:
                        seen_items.add(text)
                        item_rows.append(text)
            display_nodes.append({
                "phase": phase,
                "label": phase_labels.get(phase, phase),
                "headline": headlines[-1] if headlines else _safe_str(candidates[-1].get("label")),
                "items": item_rows,
            })
        rows: List[List[Any]] = []
        for index, node in enumerate(display_nodes):
            items = "; ".join(_safe_list(node.get("items")))
            card = Table([[Paragraph(html.escape(_safe_str(node.get("phase")).upper()), styles["MA_FlowPhase"])], [Paragraph(html.escape(_safe_str(node.get("label"))), styles["MA_FlowTitle"])], [table_cell(node.get("headline") or "Not archived")], [table_cell(items or "No stage counts archived")]], colWidths=[165 * mm])
            card.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BRAND["wash"] if index % 2 == 0 else "#F9FCFB")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8DFDC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            rows.append([card])
            if index < len(display_nodes) - 1:
                rows.append([Paragraph("↓", ParagraphStyle(name=f"MA_Arrow{index}", parent=styles["MA_Body"], alignment=1, fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor(BRAND["teal"])))])
        return [Table(rows, colWidths=[165 * mm], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))]

    def header_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(colors.HexColor(BRAND["navy"]))
        canvas.rect(0, height - 18 * mm, width, 18 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(18 * mm, height - 11 * mm, f"MetaboAgent | {disease}")
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(width - 18 * mm, height - 11 * mm, f"{compact_run_label(run_id)} | Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor(BRAND["teal"]))
        canvas.setLineWidth(2)
        canvas.line(0, height - 18.5 * mm, width, height - 18.5 * mm)
        canvas.setFillColor(colors.HexColor(BRAND["muted"]))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(18 * mm, 11 * mm, "MetaboAgent - Research Use Only")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=27 * mm, bottomMargin=20 * mm,
        title=f"MetaboAgent Evidence-to-Decision Report: {disease}", author="MetaboAgent",
    )
    metric_registry = _safe_dict(bundle.get("metric_registry"))
    development_metrics = _safe_dict(metric_registry.get("development_cv"))
    holdout_metrics = _safe_dict(metric_registry.get("internal_holdout"))
    winner_count = len(_safe_list(phase2.get("winner_features")))
    phase1_count = _safe_dict(phase1).get("selected_feature_count")
    external_declared = bool(_safe_dict(bundle.get("validation_scope")).get("external_declared"))
    task_counts = re.search(
        r"(?P<cases>\d+)\s+cases?\s+vs\s+(?P<controls>\d+)\s+controls?",
        _safe_str(profile.get("task")),
        flags=re.IGNORECASE,
    )
    n_participants = profile.get("n_participants")
    if n_participants is None and task_counts:
        n_participants = int(task_counts.group("cases")) + int(task_counts.group("controls"))
    participant_text = str(n_participants) if n_participants is not None else "Not archived"
    cv_auc = _fmt(development_metrics.get("auc"))
    holdout_auc = _fmt(holdout_metrics.get("auc"))
    snapshot_values = [
        ("Participants", participant_text),
        ("Phase 1 candidates", str(phase1_count) if phase1_count is not None else "Not archived"),
        ("Final panel", str(winner_count)),
        ("CV AUROC", cv_auc),
        ("Holdout AUROC", holdout_auc),
    ]
    snapshot_cells = []
    for label, value in snapshot_values:
        snapshot_cells.append([
            Paragraph(html.escape(label), styles["MA_SnapshotLabel"]),
            Paragraph(html.escape(value), styles["MA_SnapshotValue"]),
        ])
    snapshot_table = Table([snapshot_cells], colWidths=[33 * mm] * len(snapshot_cells))
    snapshot_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(BRAND["wash"])),
        ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor(BRAND["line"])),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor(BRAND["line"])),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    if external_declared:
        external_sentence = "Independent external validation was declared in the archived run."
    else:
        external_sentence = "No independent external cohort was declared; transportability and clinical threshold performance therefore remain unestablished."
    executive_narrative = (
        f"MetaboAgent identified a {winner_count}-feature panel after train-only candidate screening and multi-objective refinement. "
        f"The panel achieved an internal cross-validated AUROC of {cv_auc} and an internal holdout AUROC of {holdout_auc}. "
        f"{external_sentence} Clinical utility and biological interpretation should therefore be read within the archived evidence scope."
    )
    story: List[Any] = [
        Paragraph("MetaboAgent Evidence-to-Decision Report", styles["MA_Title"]),
        Paragraph(html.escape(disease), styles["MA_H1"]),
        Paragraph(
            f"{html.escape(compact_run_label(run_id))}<br/>Validation scope: internal model development and holdout evaluation.<br/>"
            "Format policy: HTML is the interactive evidence dossier; the companion PDF is the curated scientific summary. Both are generated from the same canonical evidence bundle. The JSON/CSV/manifests form the machine-auditable archive.",
            styles["MA_Body"],
        ),
        Paragraph("Executive scientific summary", styles["MA_H1"]),
        snapshot_table,
        Spacer(1, 3 * mm),
        *narrative_pdf("conclusion_summary"),
        Spacer(1, 4 * mm),
    ]
    story.append(Paragraph("Archived quantitative findings", styles["MA_H2"]))
    claim_rows = [[table_cell("Scope", heading=True), table_cell("Finding", heading=True), table_cell("Boundary", heading=True)]]
    for claim in _safe_list(bundle.get("claim_registry")):
        item = _safe_dict(claim)
        if item.get("value") is None:
            continue
        claim_rows.append([
            table_cell(item.get("evidence_scope")),
            table_cell(item.get("statement")),
            table_cell(item.get("boundary")),
        ])
    claim_table = Table(claim_rows, colWidths=[38 * mm, 72 * mm, 58 * mm], repeatRows=1)
    claim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND["navy"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(BRAND["line"])),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(BRAND["wash"])]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([claim_table, Paragraph("Winner panel", styles["MA_H1"])])
    features = _safe_list(phase2.get("winner_features"))
    story.append(body(", ".join(_safe_str(item) for item in features) or "Not available"))
    story.append(Paragraph("Study design and Phase 0-4 flow", styles["MA_H1"]))
    story.extend(narrative_pdf("study_context"))
    story.extend(phase_flow_block())
    story.append(Paragraph("Biomarker discovery and identity", styles["MA_H1"]))
    story.append(body("Prior literature, train-only selection stability and multi-objective panel optimization are retained as separate evidence layers. Missing analytical identity fields are not inferred from HMDB identifiers."))
    story.extend(narrative_pdf("prior_evidence_summary"))
    story.extend(narrative_pdf("discovery_summary"))
    identity_records = [_safe_dict(item) for item in _safe_list(identity_registry.get("records"))]
    selection_registry = _safe_dict(_safe_dict(bundle.get("enhancements")).get("feature_selection_evidence"))
    selection_by_feature = {
        _safe_str(_safe_dict(item).get("feature")): _safe_dict(item)
        for item in _safe_list(selection_registry.get("winner_records"))
    }
    # Analytical assay metadata are optional in the single-table upload
    # contract.  Do not print the same ``not archived`` string in every row;
    # report coverage once and reserve the reader-facing table for identity and
    # selection evidence.  The full fields remain in the machine-readable
    # feature_identity_registry.json artifact.
    annotation_fields = ("msi_identification_level", "mz", "retention_time", "adduct", "ion_mode", "unit")
    annotation_available = any(
        _safe_str(record.get(field)).strip() and _safe_str(record.get(field)).strip().lower() != "not archived"
        for record in identity_records for field in annotation_fields
    )
    if annotation_available:
        identity_intro = "Feature identity is reported using the uploaded metabolite names, HMDB identifiers and assay metadata supplied with the analysis."
        identity_rows = [[table_cell("Feature", heading=True), table_cell("HMDB", heading=True), table_cell("MSI", heading=True), table_cell("Assay metadata", heading=True), table_cell("Phase 1 selection", heading=True)]]
    else:
        identity_intro = "Feature identity is reported using the uploaded metabolite names and available HMDB identifiers."
        identity_rows = [[table_cell("Feature", heading=True), table_cell("HMDB", heading=True), table_cell("Phase 1 selection", heading=True)]]
    for item in identity_records:
        record = _safe_dict(item)
        token = _safe_str(record.get("canonical_id") or record.get("raw_name") or record.get("canonical_name"))
        selection_record = selection_by_feature.get(token, {})
        frequency = selection_record.get("phase1_selection_frequency")
        consensus = selection_record.get("method_consensus")
        if frequency is not None and consensus is not None:
            selection = f"frequency {_fmt(frequency)}; consensus {consensus}"
        else:
            selection = "Not archived"
        base = [
            table_cell(record.get("canonical_name")),
            table_cell(", ".join(_safe_list(record.get("hmdb_ids"))) or "Not archived"),
        ]
        if annotation_available:
            assay = "m/z: {0}; RT: {1}; adduct: {2}; ion: {3}; unit: {4}".format(
                _safe_str(record.get("mz") or "not archived"), _safe_str(record.get("retention_time") or "not archived"),
                _safe_str(record.get("adduct") or "not archived"), _safe_str(record.get("ion_mode") or "not archived"),
                _safe_str(record.get("unit") or "not archived"),
            )
            base.extend([table_cell(record.get("msi_identification_level") or "Not archived"), table_cell(assay)])
        base.append(table_cell(selection))
        identity_rows.append(base)
    if len(identity_rows) == 1:
        identity_rows.append([table_cell("No winner features archived")] + [table_cell("Not archived") for _ in range(len(identity_rows[0]) - 1)])
    if annotation_available:
        identity_col_widths = [39 * mm, 27 * mm, 18 * mm, 65 * mm, 36 * mm]
    else:
        identity_col_widths = [65 * mm, 39 * mm, 81 * mm]
    identity_table = Table(identity_rows, colWidths=identity_col_widths, repeatRows=1)
    identity_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND["navy"])), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(BRAND["line"])), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(BRAND["wash"])]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(body(identity_intro))
    story.append(identity_table)
    story.append(Paragraph("Main registered figures", styles["MA_H1"]))
    figure_manifest = _read_json(report_path / "figure_manifest.json")
    figure_lookup = {_safe_str(_safe_dict(item).get("figure_id")): _safe_dict(item) for item in _safe_list(figure_manifest.get("figures"))}
    # The PDF is a curated scientific summary, not a dump of every Phase 3
    # output.  Choose six decision-driving figures when available.  A Phase 0
    # atlas is included only when the run retained prior biomarkers; the SHAP
    # composite is preferred because it has portable PNG/SVG siblings on
    # installations without Poppler.
    main_figures = ["selection_stability", "model_performance_cv", "model_performance_holdout", "model_calibration", "clinical_decision_curve", "shap_interpretation_composite"]
    if any(_phase0_prior_counts(phase0).values()):
        main_figures.insert(0, "phase0_prior_evidence_atlas")
    legacy_ids = {"phase0_prior_evidence_atlas": "fig1c", "selection_stability": "fig1d", "model_performance_cv": "fig4a", "model_performance_holdout": "fig4a_holdout", "model_calibration": "fig4e", "clinical_decision_curve": "fig4b", "shap_interpretation_composite": "fig4i"}
    # A minimal/unit report may legitimately have no registered figures.  In
    # that case there are no planned PDF figure slots to gate; real runs with
    # declared main figures are still blocked if an asset cannot be resolved.
    main_figures = [figure_id for figure_id in main_figures if figure_id in figure_lookup]
    for figure_id in main_figures:
        item = figure_lookup.get(figure_id, {})
        title = _safe_str(item.get("title") or figure_id)
        caption = _safe_str(_safe_dict(item.get("caption_contract")).get("text_template") or item.get("interpretation_boundary"))
        if figure_id == "model_performance_cv":
            story.extend(narrative_pdf("performance_summary"))
        elif figure_id == "model_calibration":
            story.extend(narrative_pdf("calibration_summary"))
        elif figure_id == "shap_interpretation_composite":
            story.extend(narrative_pdf("biology_summary"))
        story.extend(figure_block(figure_id, title, caption, legacy_ids.get(figure_id, "")))
    if missing_main_figures:
        # A reader-facing PDF with missing core figures is not a valid static
        # release.  Preserve the HTML/audit bundle, but stop PDF publication
        # and make the failure explicit in the deterministic quality gate.
        report_quality.setdefault("checks", {})["pdf_core_figures"] = {
            "pass": False,
            "missing": sorted(set(missing_main_figures)),
        }
        report_quality["status"] = "FAIL"
        _write_json(report_path / "report_quality.json", report_quality)
        return None
    story.append(Paragraph("Exploratory clinical utility and prevalence scenarios", styles["MA_H1"]))
    story.extend(narrative_pdf("clinical_summary"))
    story.append(body("Operating points are illustrative case-control characteristics. They are not validated clinical thresholds. Prevalence-adjusted projections are scenario analyses and require target-population validation."))
    prevalence = _safe_dict(enhancements.get("prevalence_scenarios"))
    prevalence_rows = [[table_cell("Threshold", heading=True), table_cell("Prevalence", heading=True), table_cell("Projected PPV", heading=True), table_cell("Projected NPV", heading=True), table_cell("Positive / 1000", heading=True), table_cell("False positive / 1000", heading=True)]]
    for row in _safe_list(prevalence.get("rows")):
        item = _safe_dict(row)
        if abs((_safe_float(item.get("threshold")) or 0) - 0.15) > 1e-9:
            continue
        prevalence_rows.append([table_cell(_fmt(item.get("threshold"), 2)), table_cell(f"{(_safe_float(item.get('assumed_prevalence')) or 0) * 100:.1f}%"), table_cell(f"{(_safe_float(item.get('projected_ppv')) or 0) * 100:.2f}%"), table_cell(f"{(_safe_float(item.get('projected_npv')) or 0) * 100:.2f}%"), table_cell(f"{_safe_float(item.get('projected_positive_tests_per_1000')) or 0:.1f}"), table_cell(f"{_safe_float(item.get('projected_false_positives_per_1000')) or 0:.1f}")])
    if len(prevalence_rows) > 1:
        prevalence_table = Table(prevalence_rows, colWidths=[22 * mm, 23 * mm, 25 * mm, 25 * mm, 32 * mm, 38 * mm], repeatRows=1)
        prevalence_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND["navy"])), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(BRAND["line"])), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(BRAND["wash"])]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(prevalence_table)
    # The reader-facing report carries interpretation boundaries inside each
    # Finding/Interpretation/Boundary block.  Do not add a separate
    # "Limitations and evidence gaps" chapter; the machine-readable registry
    # remains archived for audit and reproducibility.
    story.append(Paragraph("Integrated conclusion and next evidence step", styles["MA_H1"]))
    story.extend(narrative_pdf("conclusion_summary"))
    story.append(body("Technical audit, claim-to-evidence graph, supplementary figure index and machine-readable provenance remain available in the HTML evidence dossier and the accompanying JSON/CSV evidence package."))
    _write_json(report_path / "pdf_content_manifest.json", {
        "schema_version": "phase4.pdf_content_manifest.v1",
        "run_id": run_id,
        "source_contracts": ["report_context_v2.json", "phase_flow_manifest.json", "figure_manifest.json", "report_quality.json"],
        "core_sections": ["executive", "study_context_phase_flow", "biomarker_discovery", "performance", "clinical_utility", "biological_interpretation", "conclusion"],
        "embedded_main_figures": main_figures,
        "supplementary_figures_indexed": [
            _safe_str(_safe_dict(item).get("figure_id")) for item in _safe_list(figure_manifest.get("figures"))
            if _safe_str(_safe_dict(item).get("figure_id")) not in main_figures and _safe_str(_safe_dict(item).get("figure_id")) != "phase_flow_overview"
        ],
        "interactive_features_flattened": ["language_toggle", "threshold_explorer", "collapsible_technical_appendix", "local_artifact_navigation"],
        "technical_audit_location": "HTML_and_evidence_package",
        "policy": "HTML and PDF share the same evidence bundle, claims and scientific_narrative_registry; PDF is a curated static scientific summary and does not recompute scientific results. Technical audit remains machine-auditable in HTML and the evidence package.",
    })
    report_quality.setdefault("checks", {})["pdf_core_figures"] = {
        "pass": True,
        "missing": [],
        "embedded_main_figures": main_figures,
    }
    _write_json(report_path / "report_quality.json", report_quality)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return str(output_path)


def refresh_final_delivery(
    run_root: str,
    run_id: str,
    report_outputs: Dict[str, str],
    report_context: Dict[str, Any],
    delivery_id: str = "current",
    report_build_id: str = "",
) -> Dict[str, str]:
    """Atomically refresh the canonical Phase 4 delivery.

    ``run_id`` identifies the scientific analysis. ``delivery_id`` identifies
    the report release channel (``current`` by default).  Keeping these two
    concepts separate prevents report-only rebuilds from masquerading as new
    analysis runs.
    """

    root = Path(run_root).resolve()
    delivery_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", _safe_str(delivery_id)).strip(".-") or "current"
    delivery_root = root / "output" / "final_delivery"
    delivery = delivery_root / delivery_name
    staging_delivery = delivery_root / ".staging" / (_safe_str(report_build_id) or f"report-{uuid.uuid4().hex[:12]}")
    if staging_delivery.exists():
        shutil.rmtree(staging_delivery)
    staging_delivery.mkdir(parents=True, exist_ok=True)
    publish_target = delivery
    delivery = staging_delivery
    report_dest = delivery / "report"
    audit_dest = delivery / "audit"
    figure_dest = delivery / "figures"
    source_dest = delivery / "source_data"
    for path in (report_dest, audit_dest, figure_dest, source_dest):
        path.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, str]] = []

    def copy_file(source: Path, destination: Path) -> None:
        if not source.exists() or not source.is_file():
            return
        # Historical-run fallback manifests may already point at the final
        # delivery asset.  Staging it again would raise SameFileError and
        # abort an otherwise read-only Phase 4 report rebuild.
        try:
            if source.resolve() == destination.resolve():
                return
        except OSError:
            pass
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"source": str(source), "destination": str(destination), "sha256": _sha256(destination)})

    for path_value in report_outputs.values():
        source = Path(path_value)
        if source.exists() and source.is_file():
            copy_file(source, report_dest / source.name)

    # Preserve nested portable evidence staged below the report directory.
    # This makes links work from the final delivery without requiring access
    # to the original run root or its absolute mount point.
    report_dirs = {Path(path_value).parent for path_value in report_outputs.values() if Path(path_value).exists()}
    for report_source in report_dirs:
        for source in report_source.rglob("*"):
            if source.is_file() and source.parent != report_source:
                copy_file(source, report_dest / source.relative_to(report_source))

    audit_source = root / "output" / "audit"
    if audit_source.exists():
        shutil.copytree(audit_source, audit_dest, dirs_exist_ok=True)

    for figure in _safe_list(_safe_dict(report_context.get("phase3")).get("figures")):
        item = _safe_dict(figure)
        candidates = [_safe_dict(item.get("primary_output"))]
        candidates.extend(_safe_dict(value) for value in _safe_list(item.get("all_outputs")))
        candidates.extend(_safe_dict(value) for value in _safe_list(item.get("auxiliary_outputs")))
        for output in candidates:
            raw = _safe_str(output.get("path")).strip()
            source = _resolve_figure_path(root, raw)
            if source.exists() and source.is_file():
                copy_file(source, figure_dest / source.name)

    for pattern in (
        "output/artifacts/*.csv",
        "output/phase1/artifacts/*.csv",
        "output/figures/assets/**/*.csv",
    ):
        for source in root.glob(pattern):
            relative = source.relative_to(root / "output")
            copy_file(source, source_dest / relative)

    manifest_path = delivery / "delivery_manifest.json"
    _write_json(manifest_path, {
        "schema_version": "phase4.final_delivery.v3",
        "scientific_run_id": run_id,
        "report_build_id": _safe_str(report_build_id),
        "release_channel": delivery_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_root": str(delivery),
        "copied_files": copied,
    })
    inventory_rows = []
    inventory_path = delivery / "delivery_file_inventory.json"
    for source in sorted(delivery.rglob("*")):
        if not source.is_file() or source == inventory_path:
            continue
        inventory_rows.append({"path": str(source.relative_to(delivery)), "size_bytes": source.stat().st_size, "sha256": _sha256(source)})
    _write_json(inventory_path, {
        "schema_version": "phase4.delivery_file_inventory.v2",
        "scientific_run_id": run_id,
        "report_build_id": _safe_str(report_build_id),
        "release_channel": delivery_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": inventory_rows,
    })
    previous_delivery = delivery_root / ".previous-current"
    if previous_delivery.exists():
        shutil.rmtree(previous_delivery)
    if publish_target.exists():
        publish_target.replace(previous_delivery)
    staging_delivery.replace(publish_target)
    if previous_delivery.exists():
        shutil.rmtree(previous_delivery)

    latest_path = delivery_root / "latest.json"
    _write_json(latest_path, {
        "schema_version": "phase4.report_release_pointer.v1",
        "scientific_run_id": run_id,
        "report_build_id": _safe_str(report_build_id),
        "release_channel": delivery_name,
        "report": str(publish_target / "report" / "report.html"),
        "delivery_root": str(publish_target),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return {
        "delivery_root": str(publish_target),
        "delivery_manifest": str(publish_target / "delivery_manifest.json"),
        "delivery_file_inventory": str(publish_target / "delivery_file_inventory.json"),
        "latest_pointer": str(latest_path),
    }
