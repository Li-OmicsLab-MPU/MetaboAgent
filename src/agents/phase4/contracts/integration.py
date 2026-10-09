"""Build and persist the run-scoped Phase 4 report contracts.

The integration layer deliberately keeps facts in ``report_context`` and uses
manifests only for lineage, placement and publication rules.  It is therefore
safe to run after Phase 0-3: it does not refit a model or alter any analysis
decision.
"""

from __future__ import annotations

import html
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .validate_manifests import validate_figures, validate_flow, validate_template


PHASES = ("phase0", "phase1", "phase2", "phase3", "phase4")


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _first_int(value: Any, keys: Iterable[str]) -> Optional[int]:
    wanted = {str(item).lower() for item in keys}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in wanted:
                try:
                    number = int(item)
                    if number >= 0:
                        return number
                except (TypeError, ValueError):
                    pass
            found = _first_int(item, wanted)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_int(item, wanted)
            if found is not None:
                return found
    return None


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> Optional[str]:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _source_path(report_context: Dict[str, Any], key: str) -> str:
    sources = _safe_dict(report_context.get("sources"))
    value = sources.get(key)
    if isinstance(value, dict):
        value = value.get("selected_path") or value.get("path")
    if _safe_str(value).strip():
        return _safe_str(value).strip()
    registry = _safe_dict(report_context.get("source_registry"))
    meta = _safe_dict(registry.get(key))
    return _safe_str(meta.get("selected_path") or meta.get("path")).strip()


def _artifact(
    root: Path,
    artifact_id: str,
    kind: str,
    raw_path: str,
    scope: str,
    produced_by: Optional[str],
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    raw = _safe_str(raw_path).strip()
    if not raw:
        return {
            "id": artifact_id,
            "kind": kind,
            "path": None,
            "sha256": None,
            "status": "not_archived",
            "scope": scope,
            "produced_by": produced_by,
            **({"notes": notes} if notes else {}),
        }
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    relative = _relative_path(root, candidate)
    if relative is None:
        return {
            "id": artifact_id,
            "kind": kind,
            "path": None,
            "sha256": None,
            "status": "not_archived",
            "scope": scope,
            "produced_by": produced_by,
            "notes": "Source is outside the explicit run root and was not copied into the report package.",
        }
    exists = candidate.exists() and candidate.is_file()
    return {
        "id": artifact_id,
        "kind": kind,
        "path": relative,
        "sha256": _sha256(candidate) if exists else None,
        "status": "complete" if exists else "not_archived",
        "scope": scope,
        "produced_by": produced_by,
        **({"notes": notes} if notes else {}),
    }


def _node_status(artifacts: Dict[str, Dict[str, Any]], outputs: List[str], default: str = "planned") -> str:
    if not outputs:
        return default
    states = [artifacts.get(item, {}).get("status") for item in outputs]
    if all(state == "complete" for state in states):
        return "complete"
    if any(state == "complete" for state in states):
        return "partial"
    return "planned" if all(state == "planned" for state in states) else "not_archived"


def build_phase_flow_manifest(
    report_context: Dict[str, Any],
    bundle: Dict[str, Any],
    run_root: str,
    report_dir: str,
) -> Dict[str, Any]:
    """Build a Phase 0-4 lineage manifest from normalized report facts."""

    root = Path(run_root).resolve()
    report_path = Path(report_dir).resolve()
    phase0 = _safe_dict(report_context.get("phase0"))
    phase1 = _safe_dict(report_context.get("phase1"))
    phase2 = _safe_dict(report_context.get("phase2"))
    phase3 = _safe_dict(report_context.get("phase3"))
    source_map = {
        "raw_matrix": _safe_str(phase0.get("data_path") or phase1.get("data_path")),
        "split_manifest": _source_path(report_context, "phase0_phase1_test_result") or _source_path(report_context, "data_split_audit"),
        "phase0_literature": _source_path(report_context, "phase0_output"),
        "phase0_priors": _source_path(report_context, "phase0_output"),
        "phase1_preprocessor": _source_path(report_context, "preprocessing_report"),
        "candidate_matrix": _source_path(report_context, "data_analysis_for_modeling"),
        "stability_summary": _source_path(report_context, "feature_selection_summary"),
        "winner_panel": _source_path(report_context, "phase2_result"),
        "model_spec": _source_path(report_context, "phase2_result"),
        "oof_predictions": str(root / "output" / "audit" / "evaluation_predictions_oof.csv"),
        "holdout_predictions": str(root / "output" / "audit" / "evaluation_predictions_holdout.csv"),
        "performance_metrics": str(root / "output" / "audit" / "performance_uncertainty.json"),
        "figure_registry": _source_path(report_context, "figure_manifest"),
        "phase_flow_manifest": str(report_path / "phase_flow_manifest.json"),
        "report_context": str(report_path / "report_context_v2.json"),
        "report_html": str(report_path / "report.html"),
        "report_pdf": str(report_path / "report.pdf"),
    }
    artifact_specs = [
        ("raw_matrix", "raw_matrix", "raw", None, None),
        ("split_manifest", "split_manifest", "mixed", None, None),
        ("phase0_literature", "json", "raw", "phase0_literature", None),
        ("phase0_priors", "json", "raw", "phase0_literature", None),
        ("phase1_preprocessor", "json", "train_only", "phase1_preprocessing", None),
        ("candidate_matrix", "table", "mixed", "phase1_preprocessing", None),
        ("stability_summary", "json", "train_only", "phase1_selection", None),
        ("winner_panel", "table", "development_cv", "phase2_model", None),
        ("model_spec", "model", "development_cv", "phase2_model", "Some historical runs may not archive all hyperparameters."),
        ("oof_predictions", "prediction", "development_cv", "phase3_validation", None),
        ("holdout_predictions", "prediction", "internal_holdout", "phase3_validation", None),
        ("performance_metrics", "metric", "mixed", "phase3_validation", None),
        ("figure_registry", "json", "mixed", "phase3_validation", None),
        ("phase_flow_manifest", "json", "mixed", "phase4_report", None),
        ("report_context", "json", "mixed", "phase4_report", None),
        ("report_html", "report", "mixed", "phase4_report", None),
        ("report_pdf", "report", "mixed", "phase4_report", None),
    ]
    artifacts_list = [
        _artifact(root, artifact_id, kind, source_map.get(artifact_id, ""), scope, produced_by, notes)
        for artifact_id, kind, scope, produced_by, notes in artifact_specs
    ]
    artifacts = {item["id"]: item for item in artifacts_list}
    n_samples = _first_int({"phase0": phase0, "phase1": phase1, "phase2": phase2, "phase3": phase3}, {"n_samples", "sample_count", "total_samples"})
    n_features = _first_int({"phase1": phase1, "phase2": phase2}, {"n_features", "feature_count", "selected_feature_count"})
    winner_count = len(_safe_list(phase2.get("winner_features"))) or _safe_int(phase2.get("winner_feature_count")) or None
    stage_summaries = _stage_summaries(report_context)

    nodes = [
        {
            "id": "phase0_literature", "phase": "phase0", "label": "Literature, disease knowledge and prior evidence",
            "inputs": [], "outputs": ["phase0_literature", "phase0_priors"], "sample_scope": "raw", "fit_scope": "none",
            "status": _node_status(artifacts, ["phase0_literature", "phase0_priors"]), "n_samples_in": None, "n_samples_out": None,
            "n_features_in": None, "n_features_out": None, "decision_effect": "changes_candidates",
            "evidence_boundary": "Literature-supported priors are not independent biomarker validation.",
            "stage_summary": stage_summaries["phase0"],
        },
        {
            "id": "phase1_preprocessing", "phase": "phase1", "label": "QC, preprocessing and candidate matrix construction",
            "inputs": ["raw_matrix", "split_manifest"], "outputs": ["phase1_preprocessor", "candidate_matrix"], "sample_scope": "mixed", "fit_scope": "within_training_fold",
            "status": _node_status(artifacts, ["phase1_preprocessor", "candidate_matrix"]), "n_samples_in": n_samples, "n_samples_out": n_samples,
            "n_features_in": n_features, "n_features_out": n_features, "decision_effect": "changes_candidates",
            "evidence_boundary": "Fit statistics must be estimated from training data only.",
            "stage_summary": stage_summaries["phase1"],
        },
        {
            "id": "phase1_selection", "phase": "phase1", "label": "Train-only stability selection",
            "inputs": ["candidate_matrix", "phase0_priors"], "outputs": ["stability_summary"], "sample_scope": "train_only", "fit_scope": "within_training_fold",
            "status": _node_status(artifacts, ["stability_summary"]), "n_samples_in": n_samples, "n_samples_out": n_samples,
            "n_features_in": n_features, "n_features_out": _first_int(phase1, {"candidate_pool_count", "selected_feature_count"}), "decision_effect": "changes_candidates",
            "evidence_boundary": "Selection frequency is candidate-selection stability, not final-model stability.",
            "stage_summary": stage_summaries["phase1"],
        },
        {
            "id": "phase2_model", "phase": "phase2", "label": "Multi-objective panel search and model development",
            "inputs": ["candidate_matrix", "stability_summary", "split_manifest"], "outputs": ["winner_panel", "model_spec"], "sample_scope": "development_cv", "fit_scope": "within_training_fold",
            "status": _node_status(artifacts, ["winner_panel", "model_spec"]), "n_samples_in": n_samples, "n_samples_out": n_samples,
            "n_features_in": _first_int(phase2, {"phase1_candidate_pool_count", "candidate_pool_count"}), "n_features_out": winner_count,
            "decision_effect": "changes_model", "evidence_boundary": "The selected panel remains development evidence until independently validated.",
            "stage_summary": stage_summaries["phase2"],
        },
        {
            "id": "phase3_validation", "phase": "phase3", "label": "Prediction, uncertainty, clinical utility and interpretation figures",
            "inputs": ["winner_panel", "model_spec", "split_manifest"], "outputs": ["oof_predictions", "holdout_predictions", "performance_metrics", "figure_registry"], "sample_scope": "mixed", "fit_scope": "mixed",
            "status": _node_status(artifacts, ["oof_predictions", "holdout_predictions", "performance_metrics", "figure_registry"]), "n_samples_in": n_samples, "n_samples_out": n_samples,
            "n_features_in": winner_count, "n_features_out": winner_count, "decision_effect": "descriptive_only",
            "evidence_boundary": "Internal CV and internal holdout do not establish external transportability.",
            "stage_summary": stage_summaries["phase3"],
        },
        {
            "id": "phase4_report", "phase": "phase4", "label": "Evidence contract, narrative, inline figures and delivery package",
            "inputs": ["phase0_literature", "phase0_priors", "phase1_preprocessor", "candidate_matrix", "stability_summary", "winner_panel", "model_spec", "oof_predictions", "holdout_predictions", "performance_metrics", "figure_registry", "phase_flow_manifest"],
            "outputs": ["report_context", "report_html", "report_pdf"], "sample_scope": "mixed", "fit_scope": "none", "status": "planned",
            "n_samples_in": n_samples, "n_samples_out": n_samples, "n_features_in": winner_count, "n_features_out": winner_count,
            "decision_effect": "creates_report", "evidence_boundary": "Phase 4 must not recompute or alter scientific decisions.",
            "stage_summary": stage_summaries["phase4"],
        },
    ]
    edges = [
        {"from": "phase0_literature", "to": "phase1_selection", "artifact_id": "phase0_priors", "label": "biological priors"},
        {"from": "phase1_preprocessing", "to": "phase1_selection", "artifact_id": "candidate_matrix", "label": "candidate matrix"},
        {"from": "phase1_selection", "to": "phase2_model", "artifact_id": "stability_summary", "label": "stable candidate evidence"},
        {"from": "phase2_model", "to": "phase3_validation", "artifact_id": "winner_panel", "label": "final panel"},
        {"from": "phase2_model", "to": "phase3_validation", "artifact_id": "model_spec", "label": "selected model"},
        {"from": "phase3_validation", "to": "phase4_report", "artifact_id": "performance_metrics", "label": "canonical metrics"},
        {"from": "phase3_validation", "to": "phase4_report", "artifact_id": "figure_registry", "label": "registered figures"},
        {"from": "phase4_report", "to": "phase4_report", "artifact_id": "phase_flow_manifest", "label": "self-describing flow contract"},
    ]
    return {
        "schema_version": "metaboagent.phase_flow.v1",
        "run_id": _safe_str(_safe_dict(report_context.get("run")).get("run_id") or "phase4-report"),
        "study": {
            "disease": _safe_str(_safe_dict(bundle.get("dataset_profile")).get("label") or phase0.get("disease_name") or "Unspecified condition"),
            "study_id": _safe_str(_safe_dict(bundle.get("dataset_profile")).get("study_id")) or None,
            "intended_use": "Research-use-only biomarker panel development",
            "outcome": _safe_str(_safe_dict(bundle.get("dataset_profile")).get("task") or "Classification outcome not archived"),
            "specimen": _safe_str(_safe_dict(bundle.get("dataset_profile")).get("specimen")) or None,
            "platform": _safe_str(_safe_dict(bundle.get("dataset_profile")).get("platform")) or None,
        },
        "artifacts": artifacts_list,
        "nodes": nodes,
        "edges": edges,
        "policy": {
            "phase_order": list(PHASES),
            "external_validation_rule": "Only an explicitly declared external cohort may be labelled external.",
            "missing_artifact_rule": "Missing evidence is rendered as not_archived and cannot be inferred.",
        },
    }


def _safe_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def _stage_summaries(report_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build concise, reader-facing outputs for the five-stage flow figure.

    These summaries intentionally contain only stage products and counts.  The
    audit-oriented status, fit-scope and evidence-boundary fields remain in the
    machine-readable lineage nodes but are not repeated in the reader-facing
    workflow graphic.
    """

    phase0 = _safe_dict(report_context.get("phase0"))
    phase1 = _safe_dict(report_context.get("phase1"))
    phase2 = _safe_dict(report_context.get("phase2"))
    phase3 = _safe_dict(report_context.get("phase3"))

    prior_count = _safe_int(phase0.get("confirmed_biomarker_count"))
    if prior_count is None:
        prior_count = _safe_int(phase0.get("final_priors_count"))
    prior_count = prior_count or 0
    final_prior_count = _safe_int(phase0.get("final_priors_count"))
    final_prior_count = final_prior_count if final_prior_count is not None else prior_count

    engineering = _safe_dict(phase1.get("engineering_profile"))
    records = _safe_list(phase1.get("feature_records"))
    raw_count = _safe_int(engineering.get("raw_count"))
    engineered_count = sum(
        _safe_int(engineering.get(key)) or 0
        for key in ("engineered_ratio_count", "engineered_sum_count", "engineered_pathway_count", "taxonomy_sum_count", "group_sum_count")
    )
    if not records and raw_count is None:
        raw_count = _safe_int(phase1.get("selected_feature_count")) or 0
    if records:
        raw_from_records = sum(
            1 for item in records
            if not _safe_dict(item).get("is_engineered")
            and _safe_str(_safe_dict(item).get("origin_type")).lower() not in {"engineered", "engineered_feature"}
        )
        engineered_from_records = len(records) - raw_from_records
        if raw_count is None:
            raw_count = raw_from_records
        if engineered_count == 0 and engineered_from_records:
            engineered_count = engineered_from_records
    raw_count = raw_count or 0
    selected_count = _safe_int(phase1.get("selected_feature_count"))
    selected_count = selected_count if selected_count is not None else raw_count + engineered_count

    winner_count = len(_safe_list(phase2.get("winner_features")))
    if not winner_count:
        winner_count = _safe_int(phase2.get("winner_feature_count")) or 0
    figure_count = _safe_int(phase3.get("figure_count"))
    if figure_count is None:
        figure_count = len(_safe_list(phase3.get("figures")))

    return {
        "phase0": {
            "headline": f"{prior_count} prior biomarkers retained",
            "items": [
                {"label": "Confirmed priors", "value": str(prior_count)},
                {"label": "Final prior set", "value": str(final_prior_count)},
            ],
        },
        "phase1": {
            "headline": f"{selected_count} features selected",
            "items": [
                {"label": "Raw features", "value": str(raw_count)},
                {"label": "Combination / engineered", "value": str(engineered_count)},
                {"label": "Total Phase 1 panel", "value": str(selected_count)},
            ],
        },
        "phase2": {
            "headline": f"{winner_count} features after multi-objective optimization",
            "items": [
                {"label": "Winner-panel features", "value": str(winner_count)},
            ],
        },
        "phase3": {
            "headline": f"{figure_count} result figures generated",
            "items": [
                {"label": "Registered result figures", "value": str(figure_count)},
                {"label": "Performance, utility, interpretation", "value": "included"},
            ],
        },
        "phase4": {
            "headline": "1 HTML and 1 static PDF evidence report assembled",
            "items": [
                {"label": "Reader-facing report", "value": "HTML + PDF"},
                {"label": "Contracts", "value": "3 manifests"},
            ],
        },
    }


_FIGURE_MAP = {
    "fig1d": ("selection_stability", "feature_selection", "discovery_and_selection", "winner-panel-size", "main_figure"),
    "fig1e": ("selection_method_support", "feature_selection", "discovery_and_selection", "winner-panel-size", "supplementary_figure"),
    "fig1g": ("panel_correlation", "association", "discovery_and_selection", "winner-panel-size", "supplementary_figure"),
    "fig2a": ("baseline_model_performance", "model_performance", "model_and_validation", "internal-discrimination", "supplementary_figure"),
    "fig2b": ("selection_baseline_comparison", "model_performance", "model_and_validation", "internal-discrimination", "supplementary_figure"),
    "fig4a": ("model_performance_cv", "model_performance", "model_and_validation", "internal-discrimination", "main_figure"),
    "fig4a_holdout": ("model_performance_holdout", "model_performance", "model_and_validation", "holdout-discrimination", "main_figure"),
    "fig4b": ("clinical_decision_curve", "clinical_utility", "clinical_utility", "clinical-utility", "main_figure"),
    "fig4c": ("model_shap", "interpretability", "biological_interpretation", "biological-context", "main_figure"),
    "fig4d": ("model_rcs", "association", "biological_interpretation", "biological-context", "supplementary_figure"),
    "fig4e": ("model_calibration", "calibration", "model_and_validation", "calibration", "main_figure"),
    "fig4f": ("threshold_performance", "clinical_utility", "clinical_utility", "clinical-utility", "supplementary_figure"),
    "fig4h": ("objective_shift", "other", "discovery_and_selection", "winner-panel-size", "supplementary_figure"),
    "fig4h_alt": ("objective_shift_radar", "other", "discovery_and_selection", "winner-panel-size", "audit_only"),
    "fig4i": ("shap_interpretation_composite", "interpretability", "biological_interpretation", "biological-context", "supplementary_figure"),
    "fig4j": ("incremental_value", "model_performance", "model_and_validation", "internal-discrimination", "supplementary_figure"),
}


def _figure_type_for_id(legacy_id: str, title: str) -> Tuple[str, str, str, str, str]:
    if legacy_id in _FIGURE_MAP:
        return _FIGURE_MAP[legacy_id]
    lowered = f"{legacy_id} {title}".lower()
    if "calibr" in lowered:
        return ("model_calibration", "calibration", "model_and_validation", "calibration", "supplementary_figure")
    if "decision" in lowered or "threshold" in lowered:
        return ("clinical_utility", "clinical_utility", "clinical_utility", "clinical-utility", "supplementary_figure")
    if "shap" in lowered or "rcs" in lowered:
        return ("biological_interpretation", "interpretability", "biological_interpretation", "biological-context", "supplementary_figure")
    return (legacy_id.replace("-", "_").lower() or "figure", "other", "technical_appendix", "winner-panel-size", "audit_only")


def _source_artifacts_for_type(analysis_type: str) -> List[str]:
    if analysis_type in {"feature_selection", "association"}:
        return ["stability_summary", "winner_panel"]
    if analysis_type in {"model_performance", "calibration"}:
        return ["oof_predictions", "holdout_predictions", "performance_metrics"]
    if analysis_type == "clinical_utility":
        return ["performance_metrics", "holdout_predictions"]
    if analysis_type in {"biology", "interpretability"}:
        return ["winner_panel", "phase0_priors"]
    if analysis_type == "qc":
        return ["raw_matrix", "candidate_matrix", "split_manifest"]
    return ["phase_flow_manifest"]


def _caption_fields(analysis_type: str) -> List[str]:
    fields = ["cohort_scope", "fit_scope", "analysis_method", "interpretation_boundary"]
    if analysis_type in {"model_performance", "calibration", "clinical_utility"}:
        fields = ["n_samples", "outcome_definition", *fields, "uncertainty"]
    if analysis_type in {"feature_selection", "association", "biology", "interpretability"}:
        fields.append("panel_or_feature_definition")
    return list(dict.fromkeys(fields))


def _output_descriptor(root: Path, report_dir: Path, raw_path: str) -> Dict[str, Any]:
    candidate = Path(_safe_str(raw_path))
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    relative = _relative_path(root, candidate)
    if relative is None:
        return {"path": f"external://{candidate.name}", "format": candidate.suffix.lstrip(".") or "other", "role": "primary", "status": "not_archived", "sha256": None}
    return {"path": relative, "format": candidate.suffix.lstrip(".") or "other", "role": "primary", "status": "available" if candidate.exists() else "not_archived", "sha256": _sha256(candidate) if candidate.exists() else None}


def build_figure_manifest(report_context: Dict[str, Any], bundle: Dict[str, Any], flow: Dict[str, Any], run_root: str, report_dir: str) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    report_path = Path(report_dir).resolve()
    run_id = _safe_str(_safe_dict(report_context.get("run")).get("run_id") or "phase4-report")
    figures: List[Dict[str, Any]] = []
    figures.append({
        "figure_id": "phase_flow_overview", "legacy_figure_id": None, "title": "Phase 0-4 evidence and data flow", "analysis_type": "workflow",
        "source_artifacts": ["phase_flow_manifest"], "claim_ids": ["phase-flow"], "cohort_scope": "mixed", "prediction_source": None, "fit_scope": "none",
        "placement": {"section_id": "study_overview", "display": "inline_main_text", "anchor_after_claim": "phase-flow"},
        "caption_contract": {"required_fields": _caption_fields("workflow"), "text_template": "Deterministic rendering of the run-scoped Phase 0-4 artifact lineage."},
        "outputs": [{"path": "figures/phase_flow_overview.svg", "format": "svg", "role": "primary", "status": "planned", "sha256": None}],
        "publication_role": "main_figure", "status": "planned", "interpretation_boundary": "This is a provenance diagram, not a performance result."
    })
    # Phase 0 is a first-class evidence stage.  Older Phase 3 runs could
    # omit the atlas task from the router even when the Phase 0 output was
    # valid, so Phase 4 also discovers a run-scoped atlas deterministically.
    phase0_candidates = [
        root / "output" / "figures" / "single_panels" / "phase0_prior_evidence_atlas.pdf",
        root / "output" / "figures" / "single_panels" / "phase0_prior_evidence_atlas.png",
        root / "output" / "figures" / "single_panels" / "phase0_prior_evidence_atlas.svg",
    ]
    phase0_atlas = next((path for path in phase0_candidates if path.exists()), None)
    if phase0_atlas is not None:
        phase0_output = _output_descriptor(root, report_path, str(phase0_atlas))
        figures.append({
            "figure_id": "phase0_prior_evidence_atlas",
            "legacy_figure_id": "fig1c",
            "title": "Prior Biomarker Evidence Atlas",
            "analysis_type": "prior_evidence",
            "source_artifacts": ["phase0_priors"],
            "claim_ids": ["prior-evidence"],
            "cohort_scope": "literature",
            "prediction_source": None,
            "fit_scope": "none",
            "placement": {"section_id": "discovery_and_selection", "display": "inline_main_text", "anchor_after_claim": "prior-evidence"},
            "caption_contract": {"required_fields": _caption_fields("prior_evidence"), "text_template": "Phase 0 ranked prior-biomarker evidence atlas. Retained candidates are shown with their archived literature and evidence scores; this figure documents biological anchoring and does not establish causality."},
            "outputs": [phase0_output],
            "publication_role": "main_figure",
            "status": "ready" if phase0_output.get("status") == "available" else "not_archived",
            "interpretation_boundary": "The atlas summarizes archived literature evidence and prior scoring; it is not a clinical performance result."
        })
    for legacy in _safe_list(_safe_dict(report_context.get("phase3")).get("figures")):
        item = _safe_dict(legacy)
        legacy_id = _safe_str(item.get("figure_id") or item.get("id") or "figure")
        canonical, analysis_type, section_id, claim_id, role = _figure_type_for_id(legacy_id, _safe_str(item.get("title")))
        if any(existing.get("figure_id") == canonical for existing in figures):
            canonical = f"{canonical}_{legacy_id.replace('-', '_')}"
        primary = _safe_dict(item.get("primary_output"))
        raw_output = _safe_str(primary.get("path"))
        if not raw_output:
            outputs = _safe_list(item.get("all_outputs"))
            raw_output = _safe_str(_safe_dict(outputs[0]).get("path")) if outputs else ""
        source_artifacts = _source_artifacts_for_type(analysis_type)
        output_descriptor = _output_descriptor(root, report_path, raw_output) if raw_output else {"path": "", "format": "other", "role": "primary", "status": "not_archived", "sha256": None}
        figures.append({
            "figure_id": canonical, "legacy_figure_id": legacy_id, "title": _safe_str(item.get("title") or canonical), "analysis_type": analysis_type,
            "source_artifacts": source_artifacts, "claim_ids": [claim_id], "cohort_scope": "internal_holdout" if "holdout" in legacy_id else ("development_cv" if analysis_type in {"model_performance", "calibration"} else "mixed"),
            "prediction_source": "Archived internal holdout predictions" if "holdout" in legacy_id else None,
            "fit_scope": "within_training_fold" if analysis_type in {"feature_selection", "model_performance", "calibration"} else "full_development",
            "placement": {"section_id": section_id, "display": "inline_main_text" if role == "main_figure" else ("supplementary" if role == "supplementary_figure" else "appendix"), "anchor_after_claim": claim_id},
            "caption_contract": {"required_fields": _caption_fields(analysis_type), "text_template": _safe_str(item.get("caption_seed")) or None},
            "outputs": [output_descriptor],
            "publication_role": role, "status": "ready" if output_descriptor.get("status") == "available" else "not_archived",
            "interpretation_boundary": "Figure scope and interpretation are constrained by the archived source artifact."
        })
    return {
        "schema_version": "metaboagent.figure_manifest.v1",
        "run_id": run_id,
        "figures": figures,
        "policy": {
            "external_label_rule": "A figure may use external scope only when its source artifact is explicitly external.",
            "main_text_rule": "Every main-text figure must be anchored to a claim and a report section.",
            "caption_rule": "A publication figure cannot be ready without all required caption fields.",
        },
    }


def _ensure_claim(bundle: Dict[str, Any], claim_id: str, statement: str, scope: str, boundary: str) -> None:
    claims = bundle.setdefault("claim_registry", [])
    if any(_safe_str(_safe_dict(item).get("claim_id")) == claim_id for item in claims):
        return
    claims.append({"claim_id": claim_id, "statement": statement, "value": None, "evidence_scope": scope, "status": "SUPPORTED", "boundary": boundary, "evidence": []})


def build_report_template_manifest(report_context: Dict[str, Any], bundle: Dict[str, Any], figures: Dict[str, Any]) -> Dict[str, Any]:
    """Create a run-compatible template instance with figure slots from the manifest."""
    for claim_id, statement, scope, boundary in [
        ("phase-flow", "The report exposes a deterministic Phase 0-4 evidence flow.", "REPORT-STRUCTURE", "The flow describes provenance and boundaries; it is not a scientific performance result."),
        ("cohort-qc", "Cohort and QC evidence are displayed with their declared scope.", "DESCRIPTIVE-QC", "QC summaries are descriptive and do not establish analytical validity."),
        ("selection-stability", "Feature selection stability is reported separately from final-model stability.", "TRAIN-ONLY", "Selection frequency is not an independent validation result."),
        ("clinical-utility", "Clinical utility analyses are exploratory and prevalence-dependent.", "INTERNAL-HOLDOUT", "Threshold performance requires external validation and recalibration."),
        ("biological-context", "Biological interpretation is evidence-scoped and hypothesis-generating.", "MODEL-ASSOCIATION", "Pathway annotations and attribution do not establish causality."),
    ]:
        _ensure_claim(bundle, claim_id, statement, scope, boundary)
    all_claims = [
        _safe_str(_safe_dict(item).get("claim_id"))
        for item in _safe_list(bundle.get("claim_registry"))
        if _safe_str(_safe_dict(item).get("claim_id"))
    ]
    sections = [
        ("executive_summary", 1, "Executive evidence summary", "hybrid", True, "always", "llm_allowed", ["internal-discrimination", "winner-panel-size"], [], "main_text"),
        ("study_overview", 2, "Study design and Phase 0-4 evidence flow", "hybrid", True, "always", "deterministic_only", ["phase-flow"], [], "main_text"),
        ("cohort_and_qc", 3, "Cohort, data profile and QC", "hybrid", True, "always", "llm_allowed", ["cohort-qc"], [], "main_text"),
        ("discovery_and_selection", 4, "Prior evidence and feature selection", "hybrid", True, "always", "llm_allowed", ["selection-stability", "winner-panel-size"], [], "main_text"),
        ("model_and_validation", 5, "Model development and validation", "hybrid", True, "always", "llm_allowed", ["prior-evidence", "internal-discrimination", "holdout-discrimination", "calibration"], [], "main_text"),
        ("clinical_utility", 6, "Exploratory clinical utility", "hybrid", True, "if_available", "llm_allowed", ["clinical-utility"], [], "main_text"),
        ("biological_interpretation", 7, "Evidence-scoped biological interpretation", "hybrid", True, "if_available", "llm_allowed", ["biological-context"], [], "main_text"),
        ("limitations", 8, "Limitations and next validation study", "narrative", True, "always", "deterministic_only", [], [], "main_text"),
        ("technical_appendix", 9, "Claim graph, audit and artifact appendix", "appendix", True, "always", "deterministic_only", [], [], "collapsible_appendix"),
    ]
    figure_slots_by_section: Dict[str, List[Dict[str, Any]]] = {}
    for item in _safe_list(figures.get("figures")):
        figure = _safe_dict(item)
        section_id = _safe_str(_safe_dict(figure.get("placement")).get("section_id"))
        if not section_id:
            continue
        role = _safe_str(figure.get("publication_role"))
        figure_slots_by_section.setdefault(section_id, []).append({
            "slot_id": f"slot_{figure.get('figure_id')}",
            "figure_id": figure.get("figure_id"),
            "required": role == "main_figure" and section_id in {"study_overview", "model_and_validation"},
            "display": _safe_dict(figure.get("placement")).get("display") or "appendix",
            "fallback": "show_boundary" if role == "main_figure" else "move_to_appendix",
        })
    section_records = []
    for section_id, order, title, kind, required, visibility, llm_policy, required_claim_ids, _, render_mode in sections:
        valid_claims = [claim for claim in required_claim_ids if claim in all_claims]
        section_records.append({
            "section_id": section_id, "order": order, "title": title, "kind": kind, "required": required,
            "visibility": visibility, "llm_policy": llm_policy, "required_claim_ids": valid_claims,
            "figure_slots": figure_slots_by_section.get(section_id, []), "render_mode": render_mode,
        })
    return {
        "schema_version": "metaboagent.report_template.v1",
        "template_id": "metaboagent_evidence_to_decision",
        "template_version": "v1.2",
        "audience": "journal_reviewers",
        "language_policy": {"primary": "en", "supported": ["en", "zh"]},
        "sections": section_records,
        "release_gates": [
            {"gate_id": "manifest_contracts", "description": "All run-scoped report contracts validate together.", "severity": "blocker", "rule": "phase flow, figure manifest and template must pass cross-file validation."},
            {"gate_id": "single_source_metrics", "description": "All reader-facing numeric claims resolve to one metric registry.", "severity": "blocker", "rule": "No conflicting values for the same metric and scope."},
            {"gate_id": "no_unresolved_placeholders", "description": "Reader-facing output contains no programmer placeholders.", "severity": "blocker", "rule": "Reject malformed figure citations, empty fields and unresolved model names."},
            {"gate_id": "external_scope_guard", "description": "External validation labels require explicit external provenance.", "severity": "blocker", "rule": "No external scope without an explicitly declared external cohort artifact."},
            {"gate_id": "caption_completeness", "description": "Main figures have complete publication captions.", "severity": "warning", "rule": "All caption contract fields must be populated before main-text publication."},
        ],
    }


def render_phase_flow_svg(flow: Dict[str, Any], output_path: Path) -> str:
    """Render one reader-facing Phase 0-4 arrow flow with stage outputs.

    The manifest retains detailed lineage nodes (including two Phase 1
    substeps), but the publication-facing graphic intentionally collapses them
    into five phase boxes.  Audit metadata such as status and fit scope stays
    machine-readable and is not repeated in this visual summary.
    """

    phase_labels = {
        "phase0": "Prior evidence",
        "phase1": "Preprocessing and feature selection",
        "phase2": "Multi-objective optimization",
        "phase3": "Validation and result figures",
        "phase4": "Evidence-to-decision report",
    }
    fills = ["#eef8f7", "#f3f6fb", "#f1f9ef", "#fff7ea", "#f8f1f5"]

    def wrap(value: Any, limit: int) -> List[str]:
        words = _safe_str(value).split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > limit:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [""]

    grouped: Dict[str, List[Dict[str, Any]]] = {phase: [] for phase in PHASES}
    for node_value in _safe_list(flow.get("nodes")):
        node = _safe_dict(node_value)
        phase = _safe_str(node.get("phase"))
        if phase in grouped:
            grouped[phase].append(node)

    display_nodes: List[Dict[str, Any]] = []
    for phase in PHASES:
        candidates = grouped.get(phase, [])
        if not candidates:
            continue
        summary = next(
            (_safe_dict(item.get("stage_summary")) for item in candidates if _safe_dict(item.get("stage_summary"))),
            {"headline": _safe_str(candidates[0].get("label")), "items": []},
        )
        display_nodes.append({"phase": phase, "label": phase_labels.get(phase, phase), "summary": summary})

    box_width = 238
    box_height = 218
    gap = 54
    margin = 24
    width = max(1180, margin * 2 + len(display_nodes) * box_width + max(0, len(display_nodes) - 1) * gap)
    height = 286
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Phase 0-4 data flow and stage outputs</title>',
        '<desc id="desc">A five-stage arrow flow from prior biomarkers through feature selection, optimization, validation figures and the final evidence report.</desc>',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#138a86"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for index, display in enumerate(display_nodes):
        box_x = margin + index * (box_width + gap)
        box_y = 26
        phase = html.escape(_safe_str(display.get("phase")).upper())
        label_lines = wrap(display.get("label"), 26)[:2]
        summary = _safe_dict(display.get("summary"))
        headline_lines = wrap(summary.get("headline"), 27)[:3]
        item_lines: List[str] = []
        for item_value in _safe_list(summary.get("items"))[:3]:
            item = _safe_dict(item_value)
            item_lines.extend(wrap(f"{item.get('label')}: {item.get('value')}", 31)[:2])
        fill = fills[index % len(fills)]
        parts.append(f'<rect x="{box_x}" y="{box_y}" width="{box_width}" height="{box_height}" rx="4" fill="{fill}" stroke="#c9dedd"/>')
        parts.append(f'<text x="{box_x + 16}" y="{box_y + 27}" font-family="Arial" font-size="12" font-weight="700" fill="#138a86">{phase}</text>')
        y = box_y + 56
        for line in label_lines:
            parts.append(f'<text x="{box_x + 16}" y="{y}" font-family="Arial" font-size="15" font-weight="700" fill="#18233a">{html.escape(line)}</text>')
            y += 20
        y += 8
        for line_index, line in enumerate(headline_lines):
            parts.append(f'<text x="{box_x + 16}" y="{y}" font-family="Arial" font-size="13" font-weight="700" fill="#202735">{html.escape(line)}</text>')
            y += 18
        y += 8
        for line in item_lines:
            parts.append(f'<text x="{box_x + 16}" y="{y}" font-family="Arial" font-size="11" fill="#536170">{html.escape(line)}</text>')
            y += 16
        if index < len(display_nodes) - 1:
            arrow_y = box_y + box_height / 2
            arrow_x = box_x + box_width + 8
            parts.append(f'<line x1="{arrow_x}" y1="{arrow_y}" x2="{arrow_x + gap - 16}" y2="{arrow_y}" stroke="#138a86" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(parts), encoding="utf-8")
    return str(output_path)


def validate_contracts(flow: Dict[str, Any], figures: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    errors.extend(validate_flow(flow))
    errors.extend(validate_template(template))
    errors.extend(validate_figures(figures, flow, template))
    return {"valid": not errors, "errors": errors, "phase_count": len({item.get("phase") for item in flow.get("nodes", [])}), "figure_count": len(figures.get("figures", [])), "section_count": len(template.get("sections", []))}


def write_contract_package(report_context: Dict[str, Any], bundle: Dict[str, Any], run_root: str, report_dir: str) -> Dict[str, Any]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    flow = build_phase_flow_manifest(report_context, bundle, run_root, report_dir)
    figures = build_figure_manifest(report_context, bundle, flow, run_root, report_dir)
    template = build_report_template_manifest(report_context, bundle, figures)
    flow_path = report_path / "phase_flow_manifest.json"
    figure_path = report_path / "figure_manifest.json"
    template_path = report_path / "report_template.json"
    svg_path = report_path / "figures" / "phase_flow_overview.svg"
    render_phase_flow_svg(flow, svg_path)
    flow["artifacts"] = [
        {**item, "status": "complete", "path": _relative_path(Path(run_root).resolve(), flow_path) or item.get("path"), "sha256": _sha256(flow_path) if flow_path.exists() else item.get("sha256")}
        if item.get("id") == "phase_flow_manifest" else item
        for item in flow.get("artifacts", [])
    ]
    for figure in figures.get("figures", []):
        if figure.get("figure_id") == "phase_flow_overview":
            figure["status"] = "ready"
            figure["outputs"][0].update({"status": "available", "sha256": _sha256(svg_path)})
    validation = validate_contracts(flow, figures, template)
    flow_path.write_text(json.dumps(flow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figure_path.write_text(json.dumps(figures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    template_path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Rewrite the flow after its own file exists so the self-reference hash is
    # not allowed to become a circular hash dependency.
    flow["artifacts"] = [
        {**item, "sha256": None} if item.get("id") == "phase_flow_manifest" else item
        for item in flow.get("artifacts", [])
    ]
    flow_path.write_text(json.dumps(flow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "phase_flow_manifest": flow,
        "figure_manifest": figures,
        "report_template": template,
        "validation": validation,
        "paths": {
            # Release metadata must remain valid after the staging directory
            # is atomically promoted to ``reports/current``.
            "phase_flow_manifest": "phase_flow_manifest.json",
            "figure_manifest": "figure_manifest.json",
            "report_template": "report_template.json",
            "phase_flow_figure": "figures/phase_flow_overview.svg",
        },
    }
