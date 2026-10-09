"""
Phase 0 state definitions.

This module upgrades Phase 0 from a Yes/No + LogProb filter into a structured,
multi-stage evidence pipeline while keeping legacy fields for compatibility with
the current nodes and downstream consumers during migration.
"""

from typing import Dict, List, Optional, TypedDict

from src.agents.phase0.evidence_models import (
    CandidateMetabolite,
    EvidencePack,
    ScoringResult,
    Stage1ScreenRecord,
)
from src.agents.phase0.disease_pathway_models import DiseasePathwayPack, DiseasePathwayCandidate
from src.utils.config_manager import get_config


class Phase0FeatureDefinitions(TypedDict, total=False):
    """Feature rules exported to downstream phases."""

    target_pathways: List[str]
    target_metabolites: List[str]


class Phase0NormalizationStats(TypedDict, total=False):
    """Statistics used for score normalization and thresholding."""

    raw_min: float
    raw_max: float
    upper_bound: float
    mean: float
    std: float
    threshold: float


class Phase0State(TypedDict):
    """
    State for the upgraded Phase 0 funnel.

    New fields support:
    - Stage 1 coarse screening
    - EvidencePack construction
    - Structured rubric scoring
    - Score normalization and thresholding

    Legacy fields (`evidence_map`, `logprob_scores`) are intentionally retained
    so the codebase can migrate incrementally without breaking existing nodes.
    """

    disease_name: str
    disease_synonyms: List[str]
    candidates: List[CandidateMetabolite]
    stage1_candidates: List[CandidateMetabolite]
    coarse_screen_map: Dict[str, Stage1ScreenRecord]
    disease_pathway_pack: DiseasePathwayPack
    disease_core_pathways: List[str]
    external_disease_pathways_broad: List[DiseasePathwayCandidate]
    top_pathways: List[str]
    evidence_packs: Dict[str, EvidencePack]
    scoring_results: Dict[str, ScoringResult]
    final_priors: List[str]
    feature_definitions: Phase0FeatureDefinitions
    normalization_stats: Phase0NormalizationStats
    confirmed_biomarkers: List[Dict[str, object]]
    candidate_scores_summary: List[Dict[str, object]]
    selection_threshold: Dict[str, object]
    screening_summary: Dict[str, object]
    scoring_summary: Dict[str, object]
    phase0_memory_pack: Dict[str, object]
    memory_seed_biomarkers: List[str]
    memory_pathway_hints: List[str]
    evidence_dir: Optional[str]
    full_result_path: Optional[str]

    # Legacy transitional fields
    evidence_map: Dict[str, str]
    logprob_scores: Dict[str, float]

    error: Optional[str]
    cache_hit: bool
    use_cache: bool
    force_refresh: bool
    execution_time: float
    max_candidates: int


def create_initial_phase0_state(
    disease_name: str,
    use_cache: Optional[bool] = None,
    force_refresh: bool = False,
    max_candidates: Optional[int] = None,
    phase0_memory_pack: Optional[Dict[str, object]] = None,
) -> Phase0State:
    """
    Create the initial upgraded Phase 0 state with compatibility defaults.

    Args:
        disease_name: Disease name to analyze
        use_cache: Whether to use caching (default: read from config)
        force_refresh: Whether to bypass cache and force full rerun
        max_candidates: Maximum number of candidates to process (default: read from config)

    Returns:
        Initialized Phase0State
    """
    config = get_config()
    phase0_defaults = config.get_phase0_defaults()
    phase0_config = config.get_phase0_config()
    if use_cache is None:
        use_cache = bool(phase0_defaults.get("use_cache", True))

    if max_candidates is None:
        max_candidates = int(phase0_defaults.get("max_candidates", 50))

    return {
        "disease_name": disease_name,
        "disease_synonyms": [],
        "candidates": [],
        "stage1_candidates": [],
        "coarse_screen_map": {},
        "disease_pathway_pack": {},
        "disease_core_pathways": [],
        "external_disease_pathways_broad": [],
        "top_pathways": [],
        "evidence_packs": {},
        "scoring_results": {},
        "evidence_map": {},
        "logprob_scores": {},
        "final_priors": [],
        "feature_definitions": {
            "target_pathways": [],
            "target_metabolites": [],
        },
        "normalization_stats": {
            "upper_bound": float(phase0_defaults.get("normalize_upper_quantile", 0.95)),
            "threshold": float(phase0_defaults.get("threshold_zscore", 0.5)),
        },
        "confirmed_biomarkers": [],
        "candidate_scores_summary": [],
        "selection_threshold": {},
        "screening_summary": {},
        "scoring_summary": {},
        "phase0_memory_pack": dict(phase0_memory_pack or {}),
        "memory_seed_biomarkers": [],
        "memory_pathway_hints": [],
        "evidence_dir": None,
        "full_result_path": None,
        "error": None,
        "cache_hit": False,
        "use_cache": use_cache,
        "force_refresh": force_refresh,
        "execution_time": 0.0,
        "max_candidates": min(
            max_candidates,
            int(phase0_config.get("stage1", {}).get("max_candidates_from_kg", max_candidates)),
        ),
    }
