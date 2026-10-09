"""
Phase 0 Subgraph: Researcher Agent

This module implements the Phase 0 subgraph for the MetaboAgent hierarchical architecture.
Phase 0 is being upgraded from a LogProb filter into a structured evidence
funnel while preserving migration compatibility for downstream phases.

Components:
- state.py: Phase0State definition
- evidence_models.py: EvidencePack / ScoringResult schema skeletons
- nodes.py: Node implementations (legacy + upcoming funnel nodes)
- graph.py: Subgraph construction
- utils.py: Helper functions
- prompts.py: LLM prompt templates

Author: MetaboAgent Team
Date: 2025-01-06
"""

from src.agents.phase0.graph import create_phase0_workflow
from src.agents.phase0.disease_pathway_models import (
    DiseasePathwayCandidate,
    DiseasePathwayPack,
    DiseasePathwayRerankItem,
)
from src.agents.phase0.disease_pathway_retriever import DiseasePathwayRetriever
from src.agents.phase0.evidence_collector import EvidenceCollector
from src.agents.phase0.evidence_models import (
    CandidateMetabolite,
    EvidencePack,
    ScoringResult,
    Stage1ScreenRecord,
)
from src.agents.phase0.scorer import (
    Phase0Scorer,
    score_evidence_pack,
    score_evidence_packs,
)
from src.agents.phase0.state import Phase0State, create_initial_phase0_state

__all__ = [
    "CandidateMetabolite",
    "DiseasePathwayCandidate",
    "DiseasePathwayPack",
    "DiseasePathwayRetriever",
    "DiseasePathwayRerankItem",
    "EvidenceCollector",
    "EvidencePack",
    "Phase0Scorer",
    "Phase0State",
    "ScoringResult",
    "Stage1ScreenRecord",
    "create_initial_phase0_state",
    "create_phase0_workflow",
    "score_evidence_pack",
    "score_evidence_packs",
]
