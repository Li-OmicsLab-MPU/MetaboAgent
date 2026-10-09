"""
Phase 0 evidence model skeletons.

These TypedDict definitions provide a stable JSON-oriented schema for the new
funnel-style Phase 0 pipeline:
1. Stage 1 coarse screening
2. EvidencePack construction
3. Structured rubric scoring
"""

from typing import Dict, List, Literal, Optional, TypedDict


class CandidateMetabolite(TypedDict, total=False):
    """Canonical candidate metabolite record used across Phase 0."""

    id: str
    name: str
    synonyms: List[str]
    pathways: List[str]


class Stage1ScreenRecord(TypedDict, total=False):
    """Fast screening output before expensive abstract collection."""

    metabolite_id: str
    metabolite_name: str
    disease_name: str
    synonyms: List[str]
    pubmed_hit_count: int
    stage1_pubmed_hit_count: int
    biorxiv_hit_count: int
    stage1_biorxiv_hit_count: int
    total_hit_count: int
    metabolite_pathways: List[str]
    disease_core_pathways: List[str]
    disease_related_pathways: List[str]
    pathway_overlap_count: int
    metabolite_pathway_count: int
    disease_core_pathway_count: int
    pathway_relevance_score: float
    pathway_relevance_unknown: bool
    hard_drop: bool
    retain_reason: Optional[str]
    drop_reason: Optional[str]


class LiteratureQueryMetadata(TypedDict, total=False):
    """Traceable metadata for literature retrieval and filtering."""

    disease_query: str
    metabolite_query: str
    combined_query: str
    query_field: str
    years_back: Optional[int]
    top_k_requested: int
    retrieved_count: int
    stage2_retrieval_hit_count: int
    included_count: int
    excluded_count: int


class StudyLevelEvidence(TypedDict, total=False):
    """Structured information extracted from a single abstract."""

    study_id: str
    year: Optional[int]
    study_type: Literal[
        "human_cohort",
        "human_trial",
        "animal",
        "cell",
        "review_meta",
        "other",
    ]
    sample_size_category: Optional[Literal["<50", "50-200", ">200"]]
    main_direction: Optional[
        Literal[
            "increase_risk_or_severity",
            "decrease_risk_or_severity",
            "no_clear_association",
            "mixed_or_unclear",
        ]
    ]
    significance: Literal["significant", "non_significant", "not_reported"]
    metabolite_role: Literal[
        "primary_outcome",
        "secondary_or_exploratory",
        "incidental_or_background",
    ]
    population_notes: Optional[str]
    key_quote: Optional[str]
    mechanistic_notes: Optional[str]


class AggregateEvidenceSummary(TypedDict, total=False):
    """Aggregate summary merged from all included studies."""

    total_studies: int
    study_type_counts: Dict[str, int]
    sample_size_counts: Dict[str, int]
    direction_counts: Dict[str, int]
    any_systematic_review_or_meta: bool
    overall_mechanistic_summary: List[str]
    pathway_summary: List[str]
    comments: Optional[str]


class PathwayMechanismContext(TypedDict, total=False):
    """Static-db and LLM-ready mechanism context for a metabolite."""

    metabolite_pathways: List[str]
    disease_core_pathways: List[str]
    overlapping_pathways: List[str]
    pathway_relevance_score: float
    reaction_neighbors: List[str]
    causal_evidence_flags: List[str]
    summary: Optional[str]


class EvidencePack(TypedDict, total=False):
    """Machine-readable evidence bundle for one metabolite-disease pair."""

    metabolite: str
    metabolite_id: Optional[str]
    disease: str
    literature_hit_count: int
    systematic_review_flag: bool
    meta_analysis_flag: bool
    query_metadata: LiteratureQueryMetadata
    study_level_evidence: List[StudyLevelEvidence]
    aggregate_summary: AggregateEvidenceSummary
    pathway_mechanism_context: PathwayMechanismContext
    included_pmids: List[str]
    excluded_pmids: List[str]


class ScoringWeights(TypedDict):
    """Dynamic scoring weights selected from the evidence zone."""

    w_clin: float
    w_spec: float
    w_mech: float


class ScoringResult(TypedDict, total=False):
    """Structured scoring output for a metabolite EvidencePack."""

    metabolite: str
    metabolite_id: Optional[str]
    disease: str
    clinical_evidence: int
    disease_specificity: int
    mechanistic_plausibility: int
    consistency: int
    weights: ScoringWeights
    s_base: float
    s_bio: float
    bio_prior_raw: float
    bio_prior_norm: Optional[float]
    legacy_confidence_score: Optional[float]
    score_confidence: Optional[float]
    rationale: Dict[str, str]
    key_pmids: List[str]
