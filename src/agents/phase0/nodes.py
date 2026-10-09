"""
Phase 0 Node Implementations.

This module upgrades the Phase 0 execution chain from:
candidate -> evidence text -> LogProb -> adaptive filtering

to:
candidate -> Stage 1 coarse screen -> EvidencePack -> structured scoring ->
ranking / thresholding -> feature rule definition

Legacy function names are retained as compatibility aliases so other parts of
the codebase can migrate incrementally.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.agents.phase0.disease_aliases import (
    canonicalize_phase0_disease_name,
    get_phase0_disease_aliases,
)
from src.agents.phase0.evidence_collector import EvidenceCollector
from src.agents.phase0.disease_pathway_retriever import DiseasePathwayRetriever
from src.agents.phase0.evidence_models import CandidateMetabolite, EvidencePack, Stage1ScreenRecord
from src.agents.phase0.scorer import Phase0Scorer
from src.agents.phase0.state import Phase0State
from src.tools.domain.pubmed_retriever import PubMedRetriever
from src.utils.config_manager import get_config

logger = logging.getLogger(__name__)


def _get_phase0_defaults() -> Dict[str, Any]:
    """Load Phase 0 default parameters from the global config."""
    return get_config().get_phase0_defaults()


def _resolve_project_path(path_value: str) -> Path:
    """Resolve project-relative paths against the repository root."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[3] / candidate


def _load_json_resource(path_value: str) -> Dict[str, Any]:
    """Load a JSON file if present, otherwise return an empty dict."""
    if not path_value:
        return {}
    resolved = _resolve_project_path(path_value)
    if not resolved.exists():
        return {}
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load JSON resource %s: %s", resolved, exc)
        return {}


def _sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation with safe fallback."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _phase0_evidence_zone(hit_count: int) -> str:
    """Map literature hit count to the scoring evidence zone."""
    if int(hit_count) <= 20:
        return "low_evidence"
    if int(hit_count) < 100:
        return "mid_evidence"
    return "high_evidence"


def _get_phase0_memory_pack(state: Phase0State) -> Dict[str, Any]:
    """Return the advisory Phase 0 memory pack, if available."""
    return dict(state.get("phase0_memory_pack", {}) or {})


def _extract_memory_seed_names(phase0_memory_pack: Dict[str, Any]) -> List[str]:
    """Extract normalized seed biomarker names from the Phase 0 memory pack."""
    names: List[str] = []
    for item in phase0_memory_pack.get("biomarker_seed_hints", []) or []:
        if isinstance(item, dict):
            candidate_name = str(item.get("name", "") or "").strip()
        else:
            candidate_name = str(item or "").strip()
        if candidate_name:
            names.append(candidate_name)
    return list(dict.fromkeys(names))


def _merge_phase0_disease_synonyms(existing: Sequence[str], phase0_memory_pack: Dict[str, Any]) -> List[str]:
    """Merge runtime disease synonyms with advisory disease-memory hints."""
    synonyms = [str(value).strip() for value in existing if str(value).strip()]
    memory_synonyms = [
        str(value).strip()
        for value in phase0_memory_pack.get("disease_synonym_hints", []) or []
        if str(value).strip()
    ]
    return list(dict.fromkeys(synonyms + memory_synonyms))


def _canonicalize_candidate(candidate: Dict[str, Any]) -> CandidateMetabolite:
    """Normalize one candidate record into the current CandidateMetabolite shape."""
    return {
        "id": str(candidate.get("id", "") or ""),
        "name": str(candidate.get("name", "") or ""),
        "synonyms": [str(x) for x in candidate.get("synonyms", []) if x],
        "pathways": [str(x) for x in candidate.get("pathways", []) if x],
    }


def _build_legacy_evidence_map(evidence_packs: Dict[str, EvidencePack]) -> Dict[str, str]:
    """
    Build a text evidence map for backward compatibility with cache/export code.

    The legacy cache manager counts `Title:` delimiters. We synthesize lightweight
    blocks from study-level evidence so old consumers continue to work until the
    cache schema is upgraded.
    """
    evidence_map: Dict[str, str] = {}
    for metabolite, pack in evidence_packs.items():
        studies = pack.get("study_level_evidence", [])
        if not studies:
            evidence_map[metabolite] = "No evidence available."
            continue

        lines: List[str] = []
        for idx, study in enumerate(studies, start=1):
            lines.extend(
                [
                    f"[Evidence {idx}]",
                    f"ID: {study.get('study_id', 'Unknown')}",
                    f"Title: Study {study.get('study_id', 'Unknown')}",
                    f"Abstract: {study.get('key_quote') or study.get('mechanistic_notes') or 'No abstract available.'}",
                    "",
                ]
            )
        evidence_map[metabolite] = "\n".join(lines).strip()
    return evidence_map


def _build_stage1_summary(
    candidates: Sequence[CandidateMetabolite],
    stage1_candidates: Sequence[CandidateMetabolite],
    coarse_screen_map: Dict[str, Stage1ScreenRecord],
) -> Dict[str, Any]:
    """Summarize Stage 1 retention / drop behavior for auditability."""
    records = list(coarse_screen_map.values())
    hard_drop_count = sum(1 for item in records if item.get("hard_drop"))
    zero_hit_records = [item for item in records if int(item.get("total_hit_count", 0)) == 0]
    zero_hit_retained = [item for item in zero_hit_records if not item.get("hard_drop")]
    zero_hit_dropped = [item for item in zero_hit_records if item.get("hard_drop")]
    low_hit_records = [
        item for item in records
        if 0 < int(item.get("total_hit_count", 0)) <= 3
    ]

    return {
        "candidate_count": len(candidates),
        "stage1_candidate_count": len(stage1_candidates),
        "hard_drop_count": hard_drop_count,
        "survival_rate": (len(stage1_candidates) / len(candidates)) if candidates else 0.0,
        "retained_by_pubmed_count": sum(
            1 for item in records if not item.get("hard_drop") and int(item.get("pubmed_hit_count", 0)) > 0
        ),
        "retained_by_literature_count": sum(
            1 for item in records if not item.get("hard_drop") and int(item.get("total_hit_count", 0)) > 0
        ),
        "retained_by_pathway_overlap_count": sum(
            1 for item in records if not item.get("hard_drop") and int(item.get("total_hit_count", 0)) == 0
            and int(item.get("pathway_overlap_count", 0)) > 0
        ),
        "zero_hit_count": len(zero_hit_records),
        "zero_hit_retained_count": len(zero_hit_retained),
        "zero_hit_dropped_count": len(zero_hit_dropped),
        "low_hit_count": len(low_hit_records),
        "low_hit_retained_count": sum(1 for item in low_hit_records if not item.get("hard_drop")),
        "low_hit_dropped_count": sum(1 for item in low_hit_records if item.get("hard_drop")),
        "missing_disease_core_pathway_count": sum(
            1 for item in records if bool(item.get("pathway_relevance_unknown"))
        ),
    }


# ============================================================================
# Node 1: Generate Candidates
# ============================================================================
def generate_candidates(state: Phase0State) -> Phase0State:
    """Load disease-linked candidate metabolites from the static disease map."""
    phase0_defaults = _get_phase0_defaults()
    phase0_stage1 = get_config().get_phase0_stage1_config()
    disease_name = state.get("disease_name", "")
    resolved_disease_name = canonicalize_phase0_disease_name(disease_name)
    max_candidates = int(state.get("max_candidates", phase0_defaults.get("max_candidates", 50)))
    max_candidates_from_kg = int(phase0_stage1.get("max_candidates_from_kg", max_candidates))
    effective_limit = min(max_candidates, max_candidates_from_kg)
    disease_map_path = get_config().get_phase0_path("disease_map") or "storage/disease_map.json"
    fuzzy_cutoff = float(phase0_defaults.get("disease_fuzzy_cutoff", 0.6))

    logger.info("Generating candidates for disease: %s", disease_name)
    if resolved_disease_name and resolved_disease_name.lower() != disease_name.strip().lower():
        logger.info("Resolved Phase 0 disease alias '%s' -> '%s'", disease_name, resolved_disease_name)

    try:
        phase0_memory_pack = _get_phase0_memory_pack(state)
        memory_seed_biomarkers = {
            name.lower(): name
            for name in _extract_memory_seed_names(phase0_memory_pack)
        }
        resolved_path = _resolve_project_path(disease_map_path)
        if not resolved_path.exists():
            error_msg = f"Disease map not found at {resolved_path}"
            logger.error(error_msg)
            return {**state, "candidates": [], "error": error_msg}

        with open(resolved_path, "r", encoding="utf-8") as handle:
            disease_map = json.load(handle)

        matched_key = resolved_disease_name.lower() or disease_name.lower()
        candidates = disease_map.get(matched_key)

        if candidates is None:
            disease_keys = list(disease_map.keys())
            matches = []
            for lookup_name in [resolved_disease_name, disease_name]:
                if not lookup_name:
                    continue
                matches = get_close_matches(
                    lookup_name.lower(),
                    disease_keys,
                    n=1,
                    cutoff=fuzzy_cutoff,
                )
                if matches:
                    matched_key = matches[0]
                    logger.info("Fuzzy matched '%s' to '%s'", lookup_name, matched_key)
                    candidates = disease_map.get(matched_key, [])
                    break
            if not matches:
                logger.warning("No disease-map match found for %s", disease_name)
                candidates = []

        canonical_candidates = [_canonicalize_candidate(candidate) for candidate in candidates]

        if memory_seed_biomarkers:
            canonical_candidates = sorted(
                canonical_candidates,
                key=lambda candidate: (
                    0 if str(candidate.get("name", "") or "").lower() in memory_seed_biomarkers else 1,
                    -len(candidate.get("pathways", []) or []),
                    str(candidate.get("name", "") or ""),
                ),
            )
        canonical_candidates = canonical_candidates[:effective_limit]

        alias_hints = get_phase0_disease_aliases(disease_name)
        if matched_key:
            alias_hints.append(matched_key)
        disease_synonyms = _merge_phase0_disease_synonyms(alias_hints, phase0_memory_pack)

        return {
            **state,
            "disease_synonyms": disease_synonyms,
            "candidates": canonical_candidates,
            "memory_seed_biomarkers": list(memory_seed_biomarkers.values()),
            "error": None,
        }
    except Exception as exc:
        error_msg = f"Error generating candidates: {exc}"
        logger.error(error_msg, exc_info=True)
        return {**state, "candidates": [], "error": error_msg}


# ============================================================================
# Node 2: Prepare Disease Pathways
# ============================================================================
def prepare_disease_pathways(state: Phase0State) -> Phase0State:
    """Retrieve runtime disease pathways for the current disease before Stage 1."""
    disease_name = state.get("disease_name", "")
    disease_synonyms = state.get("disease_synonyms", [])
    force_refresh = bool(state.get("force_refresh", False))
    phase0_memory_pack = _get_phase0_memory_pack(state)
    memory_pathway_hints = [
        str(pathway).strip()
        for pathway in phase0_memory_pack.get("pathway_family_hints", []) or []
        if str(pathway).strip()
    ]

    if not disease_name:
        return {
            **state,
            "disease_pathway_pack": {},
            "disease_core_pathways": [],
            "external_disease_pathways_broad": [],
            "top_pathways": [],
        }

    retriever = DiseasePathwayRetriever()
    pathway_pack = retriever.get_disease_pathway_pack(
        disease_name=disease_name,
        disease_synonyms=disease_synonyms,
        force_refresh=force_refresh,
    )
    return {
        **state,
        "disease_pathway_pack": pathway_pack,
        "disease_core_pathways": list(pathway_pack.get("disease_core_pathways", []) or []),
        "external_disease_pathways_broad": list(pathway_pack.get("external_disease_pathways_broad", []) or []),
        "top_pathways": list(dict.fromkeys(list(pathway_pack.get("top_pathways", []) or []) + memory_pathway_hints)),
        "memory_pathway_hints": memory_pathway_hints,
        "error": None,
    }


# ============================================================================
# Node 3: Stage 1 Coarse Screen
# ============================================================================
def stage1_coarse_screen(state: Phase0State) -> Phase0State:
    """
    Apply fast hit-count screening before expensive EvidencePack collection.

    Hard-drop rule:
    - Drop if hit count == 0 AND no usable pathway relevance signal is found.
    """
    candidates = state.get("candidates", [])
    disease_name = state.get("disease_name", "")
    disease_synonyms = state.get("disease_synonyms", [])

    if not candidates:
        logger.warning("No candidates available for Stage 1 coarse screening")
        return {
            **state,
            "stage1_candidates": [],
            "coarse_screen_map": {},
            "screening_summary": _build_stage1_summary([], [], {}),
            "error": None,
        }

    phase0_stage1 = get_config().get_phase0_stage1_config()
    pubmed_config = get_config().get_phase0_pubmed_config()
    hit_count_sources = [str(x).lower() for x in phase0_stage1.get("hit_count_sources", ["pubmed"])]
    hit_count_intent = str(pubmed_config.get("hit_count_intent", "general"))
    require_pathway_if_zero_hit = bool(phase0_stage1.get("hard_drop_require_pathway_if_zero_hit", True))
    pathway_relevance_min_score = float(phase0_stage1.get("pathway_relevance_min_score", 1.0))
    retain_low_hit_with_strong_pathway_only = bool(
        phase0_stage1.get("retain_low_hit_with_strong_pathway_only", False)
    )
    low_hit_max_hits = int(phase0_stage1.get("low_hit_max_hits", 0))
    low_hit_pathway_relevance_min_score = float(
        phase0_stage1.get("low_hit_pathway_relevance_min_score", pathway_relevance_min_score)
    )
    candidate_sort_by = str(phase0_stage1.get("candidate_sort_by", "kg_then_hits"))

    retriever = PubMedRetriever(email=pubmed_config.get("email", "metaboagent@example.com"))
    collector = EvidenceCollector(
        retriever=retriever,
        disease_pathway_pack=state.get("disease_pathway_pack", {}),
    )

    coarse_screen_map: Dict[str, Stage1ScreenRecord] = {}
    kept_candidates: List[CandidateMetabolite] = []

    for candidate in candidates:
        metabolite_name = candidate.get("name", "")
        metabolite_terms = [metabolite_name, *(candidate.get("synonyms", []) or [])]
        disease_terms = [disease_name, *(disease_synonyms or [])]
        query = retriever.build_metabolite_disease_query(metabolite_terms, disease_terms, field="tiab")

        pubmed_hit_count = 0
        biorxiv_hit_count = 0
        if "pubmed" in hit_count_sources:
            pubmed_hit_count = retriever.count_hits(query, source="pubmed", intent=hit_count_intent)
        if "biorxiv" in hit_count_sources:
            biorxiv_hit_count = retriever.count_hits(query, source="biorxiv", intent=hit_count_intent)

        context = collector.build_pathway_mechanism_context(candidate, disease_name=disease_name)
        metabolite_pathways = list(context.get("metabolite_pathways", []) or [])
        disease_core_pathways = list(context.get("disease_core_pathways", []) or [])
        disease_related_pathways = list(context.get("overlapping_pathways", []) or [])
        pathway_overlap_count = len(disease_related_pathways)
        pathway_relevance_score = float(context.get("pathway_relevance_score", pathway_overlap_count) or 0.0)
        pathway_relevance_unknown = not bool(disease_core_pathways)

        total_hit_count = int(pubmed_hit_count + biorxiv_hit_count)
        hard_drop = False
        retain_reason = None
        drop_reason = None
        low_hit_requires_pathway = (
            retain_low_hit_with_strong_pathway_only
            and low_hit_max_hits > 0
            and 0 < total_hit_count <= low_hit_max_hits
        )

        if total_hit_count > 0 and not low_hit_requires_pathway:
            retain_reason = (
                "retained_due_to_pubmed_hits"
                if pubmed_hit_count > 0
                else "retained_due_to_biorxiv_hits"
            )
        elif total_hit_count > 0 and low_hit_requires_pathway:
            if pathway_relevance_score >= low_hit_pathway_relevance_min_score:
                retain_reason = "retained_low_hit_strong_pathway_support"
            else:
                hard_drop = True
                drop_reason = "dropped_low_hit_weak_pathway_support"
        elif require_pathway_if_zero_hit:
            if pathway_relevance_score >= pathway_relevance_min_score:
                retain_reason = "retained_pathway_overlap_with_zero_hit"
            else:
                hard_drop = True
                drop_reason = (
                    "dropped_zero_hit_missing_disease_core_pathways"
                    if pathway_relevance_unknown
                    else "dropped_zero_hit_no_core_pathway"
                )
        else:
            retain_reason = "retained_zero_hit_pathway_requirement_disabled"

        record: Stage1ScreenRecord = {
            "metabolite_id": candidate.get("id", ""),
            "metabolite_name": metabolite_name,
            "disease_name": disease_name,
            "synonyms": list(candidate.get("synonyms", []) or []),
            "pubmed_hit_count": pubmed_hit_count,
            "stage1_pubmed_hit_count": pubmed_hit_count,
            "biorxiv_hit_count": biorxiv_hit_count,
            "stage1_biorxiv_hit_count": biorxiv_hit_count,
            "total_hit_count": total_hit_count,
            "metabolite_pathways": metabolite_pathways,
            "disease_core_pathways": disease_core_pathways,
            "disease_related_pathways": disease_related_pathways,
            "pathway_overlap_count": pathway_overlap_count,
            "metabolite_pathway_count": len(metabolite_pathways),
            "disease_core_pathway_count": len(disease_core_pathways),
            "pathway_relevance_score": pathway_relevance_score,
            "pathway_relevance_unknown": pathway_relevance_unknown,
            "hard_drop": hard_drop,
            "retain_reason": retain_reason,
            "drop_reason": drop_reason,
        }
        coarse_screen_map[metabolite_name] = record

        if not hard_drop:
            kept_candidates.append(candidate)

    if candidate_sort_by == "kg_then_hits":
        kept_candidates = sorted(
            kept_candidates,
            key=lambda candidate: (
                -int(candidate.get("pathways") is not None and len(candidate.get("pathways", [])) > 0),
                -float(coarse_screen_map.get(candidate.get("name", ""), {}).get("pathway_relevance_score", 0.0)),
                -int(coarse_screen_map.get(candidate.get("name", ""), {}).get("total_hit_count", 0)),
                candidate.get("name", ""),
            ),
        )
    else:
        kept_candidates = sorted(
            kept_candidates,
            key=lambda candidate: (
                -float(coarse_screen_map.get(candidate.get("name", ""), {}).get("pathway_relevance_score", 0.0)),
                -int(coarse_screen_map.get(candidate.get("name", ""), {}).get("total_hit_count", 0)),
                candidate.get("name", ""),
            ),
        )

    logger.info(
        "Stage 1 coarse screen kept %d/%d candidates",
        len(kept_candidates),
        len(candidates),
    )
    screening_summary = _build_stage1_summary(candidates, kept_candidates, coarse_screen_map)

    return {
        **state,
        "stage1_candidates": kept_candidates,
        "coarse_screen_map": coarse_screen_map,
        "screening_summary": screening_summary,
        "error": None,
    }


# ============================================================================
# Node 4: Collect EvidencePack
# ============================================================================
def collect_evidence_pack(state: Phase0State) -> Phase0State:
    """Collect structured EvidencePack placeholders for Stage 1 survivors."""
    disease_name = state.get("disease_name", "")
    disease_synonyms = state.get("disease_synonyms", [])
    candidates = state.get("stage1_candidates", []) or state.get("candidates", [])

    logger.info("Collecting EvidencePacks for %d candidates", len(candidates))

    if not candidates:
        return {
            **state,
            "evidence_packs": {},
            "evidence_map": {},
            "error": None,
        }

    try:
        collector = EvidenceCollector()
        evidence_packs = collector.collect_for_candidates(
            candidates=candidates,
            disease_name=disease_name,
            disease_synonyms=disease_synonyms,
        )
        evidence_map = _build_legacy_evidence_map(evidence_packs)

        logger.info("Collected %d EvidencePacks", len(evidence_packs))
        return {
            **state,
            "evidence_packs": evidence_packs,
            "evidence_map": evidence_map,
            "error": None,
        }
    except Exception as exc:
        error_msg = f"Error collecting EvidencePack: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "evidence_packs": {},
            "evidence_map": {},
            "error": error_msg,
        }


# ============================================================================
# Node 4: Score EvidencePack
# ============================================================================
def score_evidence_pack(state: Phase0State) -> Phase0State:
    """Score each EvidencePack and compute raw / normalized biological priors."""
    evidence_packs = state.get("evidence_packs", {})

    logger.info("Scoring %d EvidencePacks", len(evidence_packs))

    if not evidence_packs:
        return {
            **state,
            "scoring_results": {},
            "logprob_scores": {},
            "error": None,
        }

    try:
        scorer = Phase0Scorer()
        scoring_results = scorer.score_evidence_packs(evidence_packs)
        normalization_stats = scorer.summarize_normalization_stats(scoring_results)
        logprob_scores = {
            metabolite: float(result.get("bio_prior_raw", 0.0))
            for metabolite, result in scoring_results.items()
        }

        return {
            **state,
            "scoring_results": scoring_results,
            "logprob_scores": logprob_scores,
            "normalization_stats": normalization_stats,
            "error": None,
        }
    except Exception as exc:
        error_msg = f"Error scoring EvidencePack objects: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "scoring_results": {},
            "logprob_scores": {},
            "error": error_msg,
        }


# ============================================================================
# Node 5: Rank and Threshold
# ============================================================================
def rank_and_threshold(state: Phase0State) -> Phase0State:
    """
    Rank candidates by `bio_prior_raw` and apply the new Phase 0 threshold:
    threshold = mean + z * std, where z defaults to 0.5.
    """
    scoring_results = state.get("scoring_results", {})

    logger.info("Ranking and thresholding %d scored candidates", len(scoring_results))

    if not scoring_results:
        return {
            **state,
            "final_priors": [],
            "selection_threshold": {
                "rule": "mu_plus_0.5sigma",
                "zscore": None,
                "threshold": None,
                "mean": None,
                "std": None,
            },
            "error": None,
        }

    try:
        phase0_defaults = _get_phase0_defaults()
        scoring_config = get_config().get_phase0_scoring_config()
        threshold_zscore = float(
            scoring_config.get(
                "threshold_zscore",
                phase0_defaults.get("threshold_zscore", 0.5),
            )
        )

        raw_scores = {
            metabolite: float(result.get("bio_prior_raw", 0.0))
            for metabolite, result in scoring_results.items()
        }
        values = list(raw_scores.values())
        mean = sum(values) / len(values)
        std = _sample_std(values)
        threshold = mean + threshold_zscore * std

        phase0_memory_pack = _get_phase0_memory_pack(state)
        seed_names = {name.lower() for name in _extract_memory_seed_names(phase0_memory_pack)}

        ranked = sorted(
            raw_scores.items(),
            key=lambda item: (
                -item[1],
                0 if item[0].lower() in seed_names else 1,
                item[0],
            ),
        )
        final_priors = [metabolite for metabolite, score in ranked if score >= threshold]

        normalization_stats = dict(state.get("normalization_stats", {}) or {})
        normalization_stats.update(
            {
                "raw_min": min(values),
                "raw_max": max(values),
                "mean": mean,
                "std": std,
                "threshold": threshold,
            }
        )

        logger.info(
            "Thresholding complete: %d/%d metabolites retained (threshold=%.3f)",
            len(final_priors),
            len(ranked),
            threshold,
        )
        selection_threshold = {
            "rule": scoring_config.get("threshold_rule", "mu_plus_0.5sigma"),
            "zscore": threshold_zscore,
            "threshold": threshold,
            "mean": mean,
            "std": std,
        }
        candidate_scores_summary: List[Dict[str, Any]] = []
        for metabolite, score in ranked:
            detailed_result = scoring_results.get(metabolite, {}) or {}
            screen_record = (state.get("coarse_screen_map", {}) or {}).get(metabolite, {}) or {}
            evidence_pack = (state.get("evidence_packs", {}) or {}).get(metabolite, {}) or {}
            weights = detailed_result.get("weights", {}) or {}
            literature_hit_count = int(
                evidence_pack.get("literature_hit_count")
                or screen_record.get("total_hit_count")
                or 0
            )
            w_clin = float(weights.get("w_clin", 1.0))
            w_spec = float(weights.get("w_spec", 1.0))
            w_mech = float(weights.get("w_mech", 1.0))
            lambda_consistency = float(scoring_config.get("lambda_consistency", 0.5))
            clinical_score = detailed_result.get("clinical_evidence")
            specificity_score = detailed_result.get("disease_specificity")
            mechanistic_score = detailed_result.get("mechanistic_plausibility")
            consistency_score = detailed_result.get("consistency")

            row = {
                "name": metabolite,
                "bio_prior_raw": score,
                "memory_seed_hit": metabolite.lower() in seed_names,
                "selected_after_threshold": score >= threshold,
                "metabolite_id": detailed_result.get("metabolite_id") or screen_record.get("metabolite_id"),
            }
            row.update(
                {
                    "bio_prior_norm": detailed_result.get("bio_prior_norm"),
                    "clinical_score": clinical_score,
                    "specificity_score": specificity_score,
                    "mechanistic_score": mechanistic_score,
                    "consistency": consistency_score,
                    "w_clin": w_clin,
                    "w_spec": w_spec,
                    "w_mech": w_mech,
                    "lambda_consistency": lambda_consistency,
                    "clinical_contribution": None if clinical_score is None else round(w_clin * float(clinical_score), 6),
                    "specificity_contribution": None if specificity_score is None else round(w_spec * float(specificity_score), 6),
                    "mechanistic_contribution": None if mechanistic_score is None else round(w_mech * float(mechanistic_score), 6),
                    "consistency_contribution": None if consistency_score is None else round(lambda_consistency * float(consistency_score), 6),
                    "evidence_zone": _phase0_evidence_zone(literature_hit_count),
                    "literature_hit_count": literature_hit_count,
                    "pubmed_hit_count": int(
                        screen_record.get("stage1_pubmed_hit_count")
                        or screen_record.get("pubmed_hit_count")
                        or evidence_pack.get("literature_hit_count", 0)
                        or 0
                    ),
                    "stage1_pubmed_hit_count": int(
                        screen_record.get("stage1_pubmed_hit_count")
                        or screen_record.get("pubmed_hit_count")
                        or evidence_pack.get("literature_hit_count", 0)
                        or 0
                    ),
                    "stage1_total_hit_count": int(screen_record.get("total_hit_count", 0) or 0),
                    "evidence_query_hit_count": int(
                        (
                            evidence_pack.get("query_metadata", {}) or {}
                        ).get("stage2_retrieval_hit_count", evidence_pack.get("literature_hit_count", 0))
                        or 0
                    ),
                    "pathway_overlap_count": int(screen_record.get("pathway_overlap_count", 0) or 0),
                    "pathway_relevance_score": float(screen_record.get("pathway_relevance_score", 0.0) or 0.0),
                    "disease_core_pathway_count": int(screen_record.get("disease_core_pathway_count", 0) or 0),
                    "pathway_relevance_unknown": bool(screen_record.get("pathway_relevance_unknown", False)),
                    "hard_drop": bool(screen_record.get("hard_drop", False)),
                    "retain_reason": screen_record.get("retain_reason"),
                    "drop_reason": screen_record.get("drop_reason"),
                    "score_confidence": detailed_result.get("score_confidence"),
                    "key_pmids": list(detailed_result.get("key_pmids", []) or []),
                }
            )
            candidate_scores_summary.append(row)

        return {
            **state,
            "final_priors": final_priors,
            "normalization_stats": normalization_stats,
            "selection_threshold": selection_threshold,
            "candidate_scores_summary": candidate_scores_summary,
            "error": None,
        }
    except Exception as exc:
        error_msg = f"Error ranking / thresholding candidates: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "final_priors": [],
            "selection_threshold": {
                "rule": scoring_config.get("threshold_rule", "mu_plus_0.5sigma") if "scoring_config" in locals() else "mu_plus_0.5sigma",
                "zscore": None,
                "threshold": None,
                "mean": None,
                "std": None,
            },
            "error": error_msg,
        }


# ============================================================================
# Node 6: Define Feature Rules
# ============================================================================
def define_feature_rules(state: Phase0State) -> Phase0State:
    """
    Define Phase 1 feature rules from retained biomarkers and pathway context.

    Priority:
    1. Final priors
    2. Stage 1 survivors
    3. Original candidates
    """
    candidates = state.get("candidates", [])
    stage1_candidates = state.get("stage1_candidates", []) or candidates
    final_priors = set(state.get("final_priors", []))

    if not candidates:
        logger.warning("No candidates available for feature rule definition")
        return {
            **state,
            "feature_definitions": {"target_pathways": [], "target_metabolites": []},
            "error": None,
        }

    try:
        phase0_defaults = _get_phase0_defaults()
        pathway_map_path = get_config().get_phase0_path("pathway_map") or "storage/pathbank_pathway_map.json"
        pathway_top_k = int(phase0_defaults.get("pathway_top_k", 20))
        pathway_map = _load_json_resource(pathway_map_path)

        candidate_pool = [candidate for candidate in stage1_candidates if candidate.get("name") in final_priors]
        if not candidate_pool:
            candidate_pool = list(stage1_candidates)

        all_pathways: List[str] = []
        for candidate in candidate_pool:
            candidate_pathways = list(candidate.get("pathways", []) or [])
            if not candidate_pathways:
                metabolite_id = candidate.get("id", "")
                candidate_pathways = list(pathway_map.get(metabolite_id, []) or [])
            all_pathways.extend([str(pathway) for pathway in candidate_pathways if pathway])

        pathway_counter = Counter(all_pathways)
        top_pathways = [pathway for pathway, _ in pathway_counter.most_common(pathway_top_k)]
        if len(top_pathways) < pathway_top_k:
            for hint in state.get("memory_pathway_hints", []) or []:
                if hint and hint not in top_pathways:
                    top_pathways.append(str(hint))
                if len(top_pathways) >= pathway_top_k:
                    break

        return {
            **state,
            "feature_definitions": {
                "target_pathways": top_pathways,
                "target_metabolites": list(state.get("final_priors", [])),
            },
            "error": None,
        }
    except Exception as exc:
        error_msg = f"Error defining feature rules: {exc}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "feature_definitions": {"target_pathways": [], "target_metabolites": []},
            "error": error_msg,
        }


# ============================================================================
# Legacy Compatibility Aliases
# ============================================================================
def retrieve_evidence(state: Phase0State) -> Phase0State:
    """Legacy alias for the new EvidencePack collection node."""
    return collect_evidence_pack(state)


def logprob_judge(state: Phase0State) -> Phase0State:
    """Legacy alias for the new structured scoring node."""
    return score_evidence_pack(state)


def adaptive_filtering(state: Phase0State) -> Phase0State:
    """Legacy alias for the new ranking / thresholding node."""
    return rank_and_threshold(state)
