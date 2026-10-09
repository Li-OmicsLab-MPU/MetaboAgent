"""
Memory type definitions for MetaboAgent.

These TypedDicts define the minimal schema used by the long-term memory MVP.
The current implementation covers Phase 0 warm start, Phase 1 fix hints, and
Phase 2 search priors.
"""

from typing import Any, Dict, List, TypedDict


class DatasetFingerprint(TypedDict, total=False):
    """Compact summary of the current dataset for similarity retrieval."""

    disease_name: str
    clinical_scenario: str
    target_column: str
    n_samples: int
    n_features: int
    class_counts: Dict[str, int]
    imbalance_ratio: float
    missing_rate_global: float
    zero_rate_global: float
    n_pathway_features: int
    n_ratio_features: int
    n_taxonomy_features: int
    n_sum_features: int
    n_protected_anchor_features: int
    column_naming_style: str
    has_hmdb_like_columns: bool
    has_kegg_like_columns: bool


class Phase1MemoryPack(TypedDict, total=False):
    """Advisory memory hints used by Phase 1 prompt and reflection logic."""

    recommended_preprocessing_hints: List[Dict[str, Any]]
    known_failure_patterns: List[Dict[str, Any]]
    recommended_fix_hints: List[Dict[str, Any]]
    strategy_confidence: float
    provenance_cases: List[str]
    semantic_entry_id: str
    semantic_confidence: float
    semantic_strategy_key: str


class Phase0MemoryPack(TypedDict, total=False):
    """Advisory disease memory used to warm start Phase 0 retrieval and ranking."""

    matched_diseases: List[str]
    disease_synonym_hints: List[str]
    pathway_family_hints: List[str]
    biomarker_seed_hints: List[Dict[str, Any]]
    confidence: float
    provenance_cases: List[str]
    semantic_entry_id: str
    semantic_confidence: float


class Phase2SearchPrior(TypedDict, total=False):
    """Advisory search prior used to warm start Phase 2 exploration."""

    candidate_priority_scores: Dict[str, float]
    anchor_feature_hints: List[str]
    suggested_beam_width: int
    suggested_max_depth: int
    expected_panel_size_range: List[int]
    known_good_feature_groups: List[List[str]]
    known_bad_feature_groups: List[List[str]]
    prior_confidence: float
    provenance_cases: List[str]
    semantic_entry_id: str
    semantic_confidence: float
    semantic_strategy_key: str


class MemoryCase(TypedDict, total=False):
    """Single episodic memory case written after a workflow run."""

    case_id: str
    created_at: str
    disease_name: str
    clinical_scenario: str
    dataset_fingerprint: DatasetFingerprint
    run_summary: Dict[str, Any]
    phase0_summary: Dict[str, Any]
    phase1_summary: Dict[str, Any]
    phase2_summary: Dict[str, Any]
    quality_score: float
    tags: List[str]


class SemanticMemoryEntry(TypedDict, total=False):
    """Generic semantic memory entry distilled from multiple episodic cases."""

    entry_id: str
    entry_type: str
    semantic_key: str
    disease_name: str
    clinical_scenario: str
    created_at: str
    updated_at: str
    confidence: float
    support_count: int
    supporting_case_ids: List[str]
    payload: Dict[str, Any]


class RunMemoryContext(TypedDict, total=False):
    """Combined memory context injected into one workflow run."""

    memory_enabled: bool
    dataset_fingerprint: DatasetFingerprint
    phase0_memory_pack: Phase0MemoryPack
    phase1_memory_pack: Phase1MemoryPack
    phase2_search_prior: Phase2SearchPrior
    retrieval_trace: Dict[str, Any]
    writeback_policy: Dict[str, Any]
