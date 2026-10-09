"""
Runtime disease-pathway retrieval models for Phase 0.

These models describe the per-disease pathway pack produced during a single
MetaboAgent run and consumed by Stage 1 coarse screening.
"""

from typing import Any, Dict, List, Literal, TypedDict


class DiseasePathwayCandidate(TypedDict, total=False):
    """Rule-based candidate pathway before LLM reranking."""

    pathway_id: str
    pathway_name: str
    db: Literal["Reactome", "KEGG"]
    source_url: str
    class_labels: List[str]
    rule_score: float
    interpretability_score: float
    has_small_molecule_support: bool
    match_reasons: Dict[str, Any]


class DiseasePathwayRerankItem(TypedDict, total=False):
    """LLM adjudication result for one candidate pathway."""

    pathway_name: str
    db: Literal["Reactome", "KEGG"]
    relevance: Literal["high", "moderate", "low"]
    specificity: Literal["disease_specific", "family_level", "generic"]
    metabolite_interpretable: bool
    keep_for_core: bool
    confidence: Literal["high", "medium", "low"]
    rationale: str


class DiseasePathwayPack(TypedDict, total=False):
    """Runtime disease-level pathway pack for the current Phase 0 execution."""

    disease: str
    disease_synonyms: List[str]
    query_terms: Dict[str, Any]
    external_disease_pathways_broad: List[DiseasePathwayCandidate]
    disease_core_pathways: List[str]
    top_pathways: List[str]
    llm_rerank_used: bool
    llm_rerank_items: List[DiseasePathwayRerankItem]
    retrieval_metadata: Dict[str, Any]
    cache_key: str
    generated_at: str
