"""
Phase 0 Evidence Collector.

This module implements the first usable EvidencePack collector for the upgraded
Phase 0 pipeline. The collector is intentionally conservative and migration-
friendly:

1. Retrieve top abstracts with metadata using PubMedRetriever
2. Apply lightweight rule-based relevance filtering
3. Build static pathway / reaction context from existing project databases
4. Assemble an EvidencePack skeleton that can later be enriched by LLM-based
   structured extraction
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.phase0.evidence_models import (
    AggregateEvidenceSummary,
    CandidateMetabolite,
    EvidencePack,
    LiteratureQueryMetadata,
    PathwayMechanismContext,
    StudyLevelEvidence,
)
from src.agents.phase0.disease_pathway_models import DiseasePathwayPack
from src.agents.phase0.prompts import (
    create_evidence_extraction_prompt,
    create_evidence_relevance_filter_prompt,
)
from src.tools.domain.pubmed_retriever import PubMedRetriever
from src.utils.config_manager import get_config

logger = logging.getLogger(__name__)

_PATHWAY_GENERIC_TOKENS = {
    "and",
    "by",
    "for",
    "of",
    "the",
    "in",
    "to",
    "via",
    "with",
    "from",
    "or",
    "on",
    "at",
    "pathway",
    "pathways",
    "metabolism",
    "metabolic",
    "signaling",
    "signalling",
    "disease",
    "diseases",
    "defective",
    "loss",
    "function",
    "functions",
    "mutants",
    "mutant",
    "causes",
    "cause",
    "mediated",
    "events",
    "response",
    "production",
    "processing",
}

_PATHWAY_CONCEPT_KEYWORDS = {
    "central_carbon": [
        "central carbon",
        "glycolysis",
        "gluconeogenesis",
        "warburg",
        "pyruvate",
        "tricarboxylic",
        "tca",
        "citric acid cycle",
        "citrate cycle",
        "lactate",
        "pentose phosphate",
        "oxidative phosphorylation",
    ],
    "choline_lipid": [
        "choline",
        "phosphocholine",
        "phosphorylcholine",
        "phosphatidylcholine",
        "glycerophospholipid",
        "glycerophosphocholine",
        "phospholipid",
        "sphingolipid",
        "sphingomyelin",
        "sphingosine",
    ],
    "fatty_acid": [
        "fatty acid",
        "beta oxidation",
        "acylcarnitine",
        "carnitine",
        "linoleic",
        "linolenic",
        "arachidonic",
        "lipid metabolism",
    ],
    "one_carbon_amino_acid": [
        "glycine",
        "serine",
        "methionine",
        "betaine",
        "sarcosine",
        "homocysteine",
        "folate",
        "one carbon",
        "dimethylglycine",
        "s-adenosyl",
        "methyl",
    ],
    "nucleotide_purine": [
        "purine",
        "pyrimidine",
        "hypoxanthine",
        "xanthine",
        "adenosine",
        "inosine",
        "nucleotide",
    ],
    "immune_inflammation": [
        "immune",
        "inflammatory",
        "inflammation",
        "interleukin",
        "cytokine",
        "antigen",
        "mhc",
        "proteasome",
        "ubiquitination",
        "b cell",
        "t cell",
    ],
    "gut_barrier": [
        "gut",
        "intestinal",
        "bowel",
        "epithelial",
        "mucosal",
        "microbiome",
    ],
    "neurotransmitter": [
        "dopamine",
        "serotonin",
        "gaba",
        "glutamate",
        "glutamine",
        "neurotransmitter",
        "nmda",
        "synaptic",
        "neuronal",
        "melatonin",
        "acetylcholine",
    ],
    "oxidative_stress": [
        "oxidative stress",
        "glutathione",
        "redox",
        "reactive oxygen",
    ],
    "steroid_hormone": [
        "steroid",
        "androgen",
        "estrogen",
        "cortisol",
        "adrenal",
        "dehydroepiandrosterone",
        "dhea",
    ],
}


class EvidenceCollector:
    """Collect EvidencePack objects for metabolite-disease candidate pairs."""

    def __init__(
        self,
        retriever: Optional[PubMedRetriever] = None,
        disease_pathway_pack: Optional[DiseasePathwayPack] = None,
    ) -> None:
        self.config = get_config()
        self.phase0_config = self.config.get_phase0_config()
        self.phase0_paths = self.config.get_phase0_paths()
        self.literature_config = self.config.get_phase0_literature_config()
        self.pubmed_config = self.config.get_phase0_pubmed_config()
        self.llm_config = self.config.get_phase0_llm_config()
        self.retriever = retriever or PubMedRetriever(
            email=self.pubmed_config.get("email", "metaboagent@example.com")
        )
        self.max_batch_size = int(self.literature_config.get("max_batch_size", 8))
        self._extraction_llm: Optional[ChatOpenAI] = None
        self._filter_llm: Optional[ChatOpenAI] = None
        self._runtime_disease_pathway_pack = disease_pathway_pack or {}

        self._pathway_map = self._load_json_optional(self.phase0_paths.get("pathway_map", ""))
        self._pathway_map_reverse = self._load_json_optional(
            self.phase0_paths.get("pathway_map_reverse", "")
        )
        self._reaction_graph = self._load_json_optional(self.phase0_paths.get("reaction_graph", ""))
        self._disease_pathway_map = self._load_json_optional(
            self.phase0_paths.get("disease_pathway_map", "")
        )
        self._metabolite_context_map = self._load_json_optional(
            self.phase0_paths.get("metabolite_context_map", "")
        )

    def collect_evidence_pack(
        self,
        candidate: CandidateMetabolite,
        disease_name: str,
        disease_synonyms: Optional[Sequence[str]] = None,
    ) -> EvidencePack:
        """
        Collect a first-pass EvidencePack for one metabolite-disease pair.

        This implementation uses a conservative funnel:
        1. rule-based relevance prefilter
        2. optional LLM relevance filtering
        3. batched LLM extraction into study-level evidence
        4. heuristic fallback whenever the LLM fails
        """
        metabolite_name = candidate.get("name", "")
        metabolite_id = candidate.get("id")
        metabolite_terms = self._unique_terms([metabolite_name, *(candidate.get("synonyms", []) or [])])
        disease_terms = self._unique_terms([disease_name, *((disease_synonyms or []) or [])])

        top_k = int(self.literature_config.get("top_k_abstracts", 12))
        years_back = self.literature_config.get("years_back")
        intent = str(self.pubmed_config.get("abstract_intent", self.pubmed_config.get("intent", "association")))

        search_result = self.retriever.search_tiab_metabolite_disease(
            metabolite_terms=metabolite_terms,
            disease_terms=disease_terms,
            max_results=top_k,
            intent=intent,
            years_back=years_back,
            sort="relevance",
        )

        raw_papers = list(search_result.get("papers", []))
        included_papers, excluded_papers = self.filter_relevant_papers(
            papers=raw_papers,
            metabolite_terms=metabolite_terms,
            disease_terms=disease_terms,
            enabled=bool(self.literature_config.get("relevance_filter_enabled", True)),
        )

        mechanism_context = self.build_pathway_mechanism_context(
            candidate=candidate,
            disease_name=disease_name,
        )

        llm_extraction = self.extract_structured_evidence_with_llm(
            metabolite_name=metabolite_name,
            disease_name=disease_name,
            papers=included_papers,
            mechanism_context=mechanism_context,
        )
        study_level_evidence = llm_extraction.get("study_level_evidence") or self.build_study_level_evidence(included_papers)
        study_level_evidence = self.reconcile_study_level_evidence(
            extracted_studies=study_level_evidence,
            included_papers=included_papers,
        )
        aggregate_summary = self.build_aggregate_summary(
            study_level_evidence=study_level_evidence,
            pathway_context=mechanism_context,
            total_retrieved=len(raw_papers),
            total_included=len(study_level_evidence),
            llm_mechanistic_summary=llm_extraction.get("overall_mechanistic_summary"),
            llm_pathway_summary=llm_extraction.get("pathway_summary"),
            llm_comments=llm_extraction.get("comments"),
        )

        any_review_meta = bool(aggregate_summary.get("any_systematic_review_or_meta", False))
        meta_analysis_flag = any(
            "meta" in " ".join(p.get("publication_types", [])).lower()
            or "meta-analysis" in (p.get("title", "") or "").lower()
            for p in included_papers
        )

        query_metadata: LiteratureQueryMetadata = {
            "disease_query": self.retriever.build_or_query(disease_terms, field="tiab"),
            "metabolite_query": self.retriever.build_or_query(metabolite_terms, field="tiab"),
            "combined_query": search_result.get("query", ""),
            "query_field": "tiab",
            "years_back": years_back,
            "top_k_requested": top_k,
            "retrieved_count": len(raw_papers),
            "stage2_retrieval_hit_count": int(search_result.get("pubmed_hit_count", 0)),
            "included_count": len(study_level_evidence),
            "excluded_count": len(excluded_papers),
        }

        return {
            "metabolite": metabolite_name,
            "metabolite_id": metabolite_id,
            "disease": disease_name,
            "literature_hit_count": int(search_result.get("pubmed_hit_count", 0)),
            "systematic_review_flag": any_review_meta,
            "meta_analysis_flag": meta_analysis_flag,
            "query_metadata": query_metadata,
            "study_level_evidence": study_level_evidence,
            "aggregate_summary": aggregate_summary,
            "pathway_mechanism_context": mechanism_context,
            "included_pmids": [str(row.get("study_id", "")) for row in study_level_evidence if row.get("study_id")],
            "excluded_pmids": [str(p.get("id", "")) for p in excluded_papers if p.get("id")],
        }

    def collect_for_candidates(
        self,
        candidates: Sequence[CandidateMetabolite],
        disease_name: str,
        disease_synonyms: Optional[Sequence[str]] = None,
    ) -> Dict[str, EvidencePack]:
        """Collect EvidencePacks for multiple candidates."""
        results: Dict[str, EvidencePack] = {}
        for candidate in candidates:
            metabolite_name = candidate.get("name", "")
            if not metabolite_name:
                continue
            try:
                results[metabolite_name] = self.collect_evidence_pack(
                    candidate=candidate,
                    disease_name=disease_name,
                    disease_synonyms=disease_synonyms,
                )
            except Exception as exc:
                logger.error("Failed to collect EvidencePack for %s: %s", metabolite_name, exc, exc_info=True)
        return results

    def filter_relevant_papers(
        self,
        papers: Sequence[Dict[str, Any]],
        metabolite_terms: Sequence[str],
        disease_terms: Sequence[str],
        enabled: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Lightweight relevance filter.

        Rules:
        - Keep all papers when filtering is disabled.
        - Otherwise require at least one metabolite term and one disease term to
          appear in title or abstract.
        """
        if not enabled:
            return list(papers), []

        included: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []

        normalized_metabolite_terms = [term.lower() for term in metabolite_terms if term]
        normalized_disease_terms = [term.lower() for term in disease_terms if term]

        for paper in papers:
            haystack = " ".join(
                [
                    str(paper.get("title", "") or "").lower(),
                    str(paper.get("abstract", "") or "").lower(),
                ]
            )
            has_metabolite = any(term in haystack for term in normalized_metabolite_terms)
            has_disease = any(term in haystack for term in normalized_disease_terms)

            if has_metabolite and has_disease:
                included.append(paper)
            else:
                excluded.append(paper)

        min_relevant = int(self.literature_config.get("min_relevant_abstracts", 1))
        if len(included) < min_relevant:
            logger.info(
                "Relevance filter kept only %d papers (< %d); falling back to raw retrieval set",
                len(included),
                min_relevant,
            )
            included, excluded = list(papers), []

        llm_filtered = self._llm_filter_relevant_papers(
            papers=included,
            metabolite_name=metabolite_terms[0] if metabolite_terms else "",
            disease_name=disease_terms[0] if disease_terms else "",
        )
        if llm_filtered is None:
            return included, excluded

        llm_included, llm_excluded = llm_filtered
        if len(llm_included) < min_relevant:
            logger.info(
                "LLM relevance filter kept only %d papers (< %d); using rule-filtered set instead",
                len(llm_included),
                min_relevant,
            )
            return included, excluded
        return llm_included, excluded + llm_excluded

    def build_pathway_mechanism_context(
        self,
        candidate: CandidateMetabolite,
        disease_name: str,
    ) -> PathwayMechanismContext:
        """Build static mechanism context from current project JSON resources."""
        metabolite_id = candidate.get("id", "") or ""
        metabolite_name = candidate.get("name", "") or ""

        context_from_index = self._lookup_metabolite_context(metabolite_id, metabolite_name)
        metabolite_pathways = self._extract_metabolite_pathways(candidate, context_from_index)
        disease_core_pathways = self._extract_disease_pathways(disease_name)
        overlapping_pathways, pathway_relevance_score, pathway_match_flags = self._match_relevant_pathways(
            metabolite_pathways=metabolite_pathways,
            disease_core_pathways=disease_core_pathways,
        )
        reaction_neighbors = self._extract_reaction_neighbors(metabolite_id, metabolite_name)

        summary_parts: List[str] = []
        if overlapping_pathways:
            overlap_preview = ", ".join(overlapping_pathways[:3])
            summary_parts.append(f"Disease-relevant pathway matches: {overlap_preview}")
        if reaction_neighbors:
            neighbor_preview = ", ".join(reaction_neighbors[:3])
            summary_parts.append(f"Reaction-network neighbors: {neighbor_preview}")

        return {
            "metabolite_pathways": metabolite_pathways,
            "disease_core_pathways": disease_core_pathways,
            "overlapping_pathways": overlapping_pathways,
            "pathway_relevance_score": pathway_relevance_score,
            "reaction_neighbors": reaction_neighbors,
            "causal_evidence_flags": pathway_match_flags,
            "summary": "; ".join(summary_parts) if summary_parts else None,
        }

    def build_study_level_evidence(
        self,
        papers: Sequence[Dict[str, Any]],
    ) -> List[StudyLevelEvidence]:
        """
        Build heuristic study-level placeholders.

        These records intentionally leave uncertain fields as null / conservative
        defaults, so the same schema can later be replaced by batched LLM
        extraction without changing downstream code.
        """
        evidence_rows: List[StudyLevelEvidence] = []
        for paper in papers:
            title = str(paper.get("title", "") or "")
            abstract = str(paper.get("abstract", "") or "")
            publication_types = [str(x) for x in paper.get("publication_types", [])]
            study_type = self._infer_study_type(title=title, abstract=abstract, publication_types=publication_types)

            evidence_rows.append(
                {
                    "study_id": str(paper.get("id", "Unknown")),
                    "year": int(paper.get("year", 0)) if paper.get("year") else None,
                    "study_type": study_type,
                    "sample_size_category": None,
                    "main_direction": None,
                    "significance": "not_reported",
                    "metabolite_role": "secondary_or_exploratory",
                    "population_notes": self._infer_population_notes(title, abstract),
                    "key_quote": self._extract_key_quote(abstract),
                    "mechanistic_notes": self._extract_mechanistic_notes(abstract),
                }
            )
        return evidence_rows

    def build_aggregate_summary(
        self,
        study_level_evidence: Sequence[StudyLevelEvidence],
        pathway_context: PathwayMechanismContext,
        total_retrieved: int,
        total_included: int,
        llm_mechanistic_summary: Optional[Sequence[str]] = None,
        llm_pathway_summary: Optional[Sequence[str]] = None,
        llm_comments: Optional[str] = None,
    ) -> AggregateEvidenceSummary:
        """Aggregate study placeholders into a valid EvidencePack summary block."""
        study_type_counts = {
            "human_cohort": 0,
            "human_trial": 0,
            "animal": 0,
            "cell": 0,
            "review_meta": 0,
            "other": 0,
        }
        sample_size_counts = {"<50": 0, "50-200": 0, ">200": 0, "unknown": 0}
        direction_counts = {
            "increase_risk_or_severity": 0,
            "decrease_risk_or_severity": 0,
            "no_clear_association": 0,
            "mixed_or_unclear": 0,
        }

        for row in study_level_evidence:
            study_type = row.get("study_type", "other")
            if study_type in study_type_counts:
                study_type_counts[study_type] += 1
            else:
                study_type_counts["other"] += 1

            size_category = row.get("sample_size_category")
            if size_category in {"<50", "50-200", ">200"}:
                sample_size_counts[str(size_category)] += 1
            else:
                sample_size_counts["unknown"] += 1

            direction = row.get("main_direction")
            if direction in direction_counts:
                direction_counts[str(direction)] += 1

        pathway_summary: List[str] = []
        if llm_pathway_summary:
            pathway_summary.extend([str(x) for x in llm_pathway_summary if x])
        if not pathway_summary and pathway_context.get("overlapping_pathways"):
            pathway_summary.append(
                "Overlapping pathways: "
                + ", ".join(pathway_context.get("overlapping_pathways", [])[:4])
            )
        elif not pathway_summary and pathway_context.get("metabolite_pathways"):
            pathway_summary.append(
                "Metabolite pathways: "
                + ", ".join(pathway_context.get("metabolite_pathways", [])[:4])
            )

        overall_mechanistic_summary: List[str] = []
        if llm_mechanistic_summary:
            overall_mechanistic_summary.extend([str(x) for x in llm_mechanistic_summary if x])
        if not overall_mechanistic_summary and pathway_context.get("reaction_neighbors"):
            overall_mechanistic_summary.append(
                "Reaction neighbors: "
                + ", ".join(pathway_context.get("reaction_neighbors", [])[:4])
            )
        if pathway_context.get("summary"):
            overall_mechanistic_summary.append(str(pathway_context["summary"]))

        comments: Optional[str] = llm_comments
        if total_retrieved == 0:
            comments = "No PubMed abstracts retrieved."
        elif comments is None and total_included < total_retrieved:
            comments = (
                f"Relevance filtering kept {total_included}/{total_retrieved} abstracts."
            )
        elif comments is None and total_included > 0:
            comments = "Structured evidence extracted from batch abstracts."

        return {
            "total_studies": len(study_level_evidence),
            "study_type_counts": study_type_counts,
            "sample_size_counts": sample_size_counts,
            "direction_counts": direction_counts,
            "any_systematic_review_or_meta": study_type_counts["review_meta"] > 0,
            "overall_mechanistic_summary": overall_mechanistic_summary,
            "pathway_summary": pathway_summary,
            "comments": comments,
        }

    def extract_structured_evidence_with_llm(
        self,
        metabolite_name: str,
        disease_name: str,
        papers: Sequence[Dict[str, Any]],
        mechanism_context: PathwayMechanismContext,
    ) -> Dict[str, Any]:
        """
        Run batched LLM extraction and merge multiple extraction batches.

        Returns an empty dict on any non-recoverable issue so the caller can
        safely fall back to heuristic study-level evidence.
        """
        if not papers:
            return {}

        llm = self._get_extraction_llm()
        if llm is None:
            return {}

        pathway_context = self.format_pathway_context_for_llm(mechanism_context)
        merged_studies: List[StudyLevelEvidence] = []
        merged_mechanistic_summary: List[str] = []
        merged_pathway_summary: List[str] = []
        comments: List[str] = []

        for batch in self._chunk_papers(list(papers), self.max_batch_size):
            try:
                abstract_list = self.format_abstracts_for_llm(batch)
                system_prompt, user_prompt = create_evidence_extraction_prompt(
                    metabolite=metabolite_name,
                    disease=disease_name,
                    abstract_list=abstract_list,
                    pathway_context=pathway_context,
                )
                payload = self._invoke_json_llm(llm, system_prompt, user_prompt)
                if not isinstance(payload, dict):
                    continue

                batch_studies = payload.get("study_level_evidence", [])
                if isinstance(batch_studies, list):
                    merged_studies.extend(self._normalize_study_level_evidence(batch_studies))

                aggregate = payload.get("aggregate_summary", {})
                if isinstance(aggregate, dict):
                    merged_mechanistic_summary.extend(
                        [str(x) for x in aggregate.get("overall_mechanistic_summary", []) if x]
                    )
                    merged_pathway_summary.extend(
                        [str(x) for x in aggregate.get("pathway_summary", []) if x]
                    )
                    if aggregate.get("comments"):
                        comments.append(str(aggregate.get("comments")))
            except Exception as exc:
                logger.warning(
                    "LLM evidence extraction failed for %s/%s batch: %s",
                    metabolite_name,
                    disease_name,
                    exc,
                )

        if not merged_studies:
            return {}

        return {
            "study_level_evidence": self._deduplicate_studies(merged_studies),
            "overall_mechanistic_summary": self._unique_list(merged_mechanistic_summary),
            "pathway_summary": self._unique_list(merged_pathway_summary),
            "comments": " | ".join(self._unique_list(comments)) if comments else None,
        }

    def reconcile_study_level_evidence(
        self,
        extracted_studies: Sequence[StudyLevelEvidence],
        included_papers: Sequence[Dict[str, Any]],
    ) -> List[StudyLevelEvidence]:
        """
        Reconcile extracted study rows against the included PMID list.

        Guarantees:
        - every included paper has one study row (heuristic backfill if needed)
        - `study_type` is post-processed with source metadata when LLM outputs `other`
        - output order follows included paper order for stable downstream diffs
        """
        included_lookup = {
            str(paper.get("id", "")): paper
            for paper in included_papers
            if str(paper.get("id", "")).strip()
        }
        heuristic_lookup = {
            str(row.get("study_id", "")): row
            for row in self.build_study_level_evidence(included_papers)
            if str(row.get("study_id", "")).strip()
        }
        extracted_lookup = {
            str(row.get("study_id", "")): row
            for row in self._deduplicate_studies(extracted_studies)
            if str(row.get("study_id", "")).strip()
        }

        reconciled: List[StudyLevelEvidence] = []
        for paper in included_papers:
            study_id = str(paper.get("id", "")).strip()
            if not study_id:
                continue

            extracted_row = extracted_lookup.get(study_id)
            heuristic_row = heuristic_lookup.get(study_id)

            if extracted_row is None and heuristic_row is not None:
                reconciled.append(heuristic_row)
                continue
            if extracted_row is None:
                continue

            merged_row = {
                **heuristic_row,
                **extracted_row,
            } if heuristic_row else dict(extracted_row)
            reconciled.append(self._postprocess_study_row(merged_row, paper))

        return reconciled

    def format_abstracts_for_llm(self, papers: Sequence[Dict[str, Any]]) -> str:
        """Proxy to the retriever formatter for downstream prompt construction."""
        return self.retriever.format_batch_abstracts_for_llm(list(papers))

    def format_pathway_context_for_llm(self, context: PathwayMechanismContext) -> str:
        """Serialize static mechanism context for future extraction prompts."""
        return json.dumps(context, ensure_ascii=False, indent=2)

    def _extract_metabolite_pathways(
        self,
        candidate: CandidateMetabolite,
        indexed_context: Dict[str, Any],
    ) -> List[str]:
        metabolite_id = candidate.get("id", "") or ""
        metabolite_name = candidate.get("name", "") or ""
        candidate_pathways = list(candidate.get("pathways", []) or [])
        if candidate_pathways:
            return self._normalize_pathway_list(candidate_pathways)

        if indexed_context.get("pathways"):
            return self._normalize_pathway_list(indexed_context.get("pathways", []))

        lookup_keys = [metabolite_id, metabolite_name, metabolite_name.lower()]
        pathways: List[str] = []

        for key in lookup_keys:
            if not key:
                continue
            value = self._pathway_map.get(key)
            if isinstance(value, list):
                pathways.extend([str(x) for x in value if x])

        if not pathways and self._pathway_map_reverse:
            for pathway_name, members in self._pathway_map_reverse.items():
                if not isinstance(members, list):
                    continue
                member_set = {str(x).lower() for x in members}
                if metabolite_id.lower() in member_set or metabolite_name.lower() in member_set:
                    pathways.append(str(pathway_name))

        return self._normalize_pathway_list(pathways)

    def _extract_disease_pathways(self, disease_name: str) -> List[str]:
        runtime_core = self._runtime_disease_pathway_pack.get("disease_core_pathways", [])
        if isinstance(runtime_core, list) and runtime_core:
            return self._normalize_pathway_list(runtime_core)

        runtime_top = self._runtime_disease_pathway_pack.get("top_pathways", [])
        if isinstance(runtime_top, list) and runtime_top:
            return self._normalize_pathway_list(runtime_top)

        disease_key = disease_name.lower().strip()
        if self._disease_pathway_map:
            value = self._disease_pathway_map.get(disease_key) or self._disease_pathway_map.get(disease_name)
            if isinstance(value, list):
                return self._normalize_pathway_list(value)
            if isinstance(value, dict):
                for field_name in (
                    "disease_core_pathways",
                    "top_pathways",
                    "core_pathways",
                    "pathways",
                    "all_pathways",
                ):
                    field_value = value.get(field_name)
                    if isinstance(field_value, list) and field_value:
                        return self._normalize_pathway_list(field_value)
        return []

    def _extract_reaction_neighbors(self, metabolite_id: str, metabolite_name: str) -> List[str]:
        if not self._reaction_graph:
            return []

        lookup_keys = [metabolite_id, metabolite_name, metabolite_name.lower()]
        for key in lookup_keys:
            if not key:
                continue
            value = self._reaction_graph.get(key)
            if isinstance(value, dict):
                neighbors = value.get("neighbors") or value.get("connected_metabolites") or value.get("products") or value.get("substrates")
                if isinstance(neighbors, list):
                    return sorted(dict.fromkeys([str(x) for x in neighbors if x]))[:10]
            if isinstance(value, list):
                return sorted(dict.fromkeys([str(x) for x in value if x]))[:10]
        return []

    def _lookup_metabolite_context(self, metabolite_id: str, metabolite_name: str) -> Dict[str, Any]:
        if not self._metabolite_context_map:
            return {}
        for key in [metabolite_id, metabolite_name, metabolite_name.lower()]:
            if key and isinstance(self._metabolite_context_map.get(key), dict):
                return dict(self._metabolite_context_map[key])
        return {}

    def _normalize_pathway_list(self, values: Sequence[Any]) -> List[str]:
        """Return a stable, de-duplicated list of non-empty pathway names."""
        normalized: List[str] = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return sorted(normalized)

    def _normalize_pathway_text(self, pathway_name: str) -> str:
        """Normalize a pathway string for robust comparison."""
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(pathway_name or "").lower())).strip()

    def _tokenize_pathway_name(self, pathway_name: str) -> List[str]:
        """Tokenize pathway names while dropping low-information generic words."""
        tokens = self._normalize_pathway_text(pathway_name).split()
        return [
            token
            for token in tokens
            if (
                token not in _PATHWAY_GENERIC_TOKENS
                and (len(token) >= 4 or token in {"tca", "gaba", "nmda", "mhc"})
            )
        ]

    def _extract_pathway_concepts(self, pathway_name: str) -> set[str]:
        """Map pathway names onto coarse mechanistic concepts for Stage 1 bridging."""
        normalized_name = self._normalize_pathway_text(pathway_name)
        concepts: set[str] = set()
        for concept, keywords in _PATHWAY_CONCEPT_KEYWORDS.items():
            for keyword in keywords:
                normalized_keyword = self._normalize_pathway_text(keyword)
                if normalized_keyword and normalized_keyword in normalized_name:
                    concepts.add(concept)
                    break
        return concepts

    def _score_pathway_pair(
        self,
        metabolite_pathway: str,
        disease_pathway: str,
    ) -> Tuple[float, Optional[str]]:
        """Score whether a metabolite pathway is biologically close to a disease pathway."""
        metabolite_norm = self._normalize_pathway_text(metabolite_pathway)
        disease_norm = self._normalize_pathway_text(disease_pathway)
        if not metabolite_norm or not disease_norm:
            return 0.0, None

        metabolite_tokens = set(self._tokenize_pathway_name(metabolite_pathway))
        disease_tokens = set(self._tokenize_pathway_name(disease_pathway))
        shared_tokens = metabolite_tokens & disease_tokens

        if metabolite_norm == disease_norm:
            return 3.0, f"exact:{metabolite_pathway}->{disease_pathway}"

        if metabolite_norm in disease_norm or disease_norm in metabolite_norm:
            if len(shared_tokens) >= 2:
                return 2.0, f"phrase:{metabolite_pathway}->{disease_pathway}"

        shared_concepts = self._extract_pathway_concepts(metabolite_pathway) & self._extract_pathway_concepts(disease_pathway)
        if shared_concepts:
            concept_list = ",".join(sorted(shared_concepts))
            score = 1.5 if len(shared_concepts) == 1 else 1.75
            return score, f"concept[{concept_list}]:{metabolite_pathway}->{disease_pathway}"

        if len(shared_tokens) >= 2:
            token_list = ",".join(sorted(shared_tokens))
            return 1.0, f"tokens[{token_list}]:{metabolite_pathway}->{disease_pathway}"

        return 0.0, None

    def _match_relevant_pathways(
        self,
        metabolite_pathways: Sequence[str],
        disease_core_pathways: Sequence[str],
    ) -> Tuple[List[str], float, List[str]]:
        """Bridge metabolite pathways to disease pathways using exact and concept-level matching."""
        if not metabolite_pathways or not disease_core_pathways:
            return [], 0.0, []

        scored_matches: List[Tuple[str, float, str]] = []
        for disease_pathway in disease_core_pathways:
            best_score = 0.0
            best_reason: Optional[str] = None
            for metabolite_pathway in metabolite_pathways:
                score, reason = self._score_pathway_pair(metabolite_pathway, disease_pathway)
                if score > best_score:
                    best_score = score
                    best_reason = reason
            if best_score > 0.0 and best_reason:
                scored_matches.append((disease_pathway, best_score, best_reason))

        scored_matches.sort(key=lambda item: (-item[1], item[0].lower()))
        matched_pathways = [pathway for pathway, _, _ in scored_matches]
        pathway_relevance_score = round(sum(score for _, score, _ in scored_matches[:4]), 3)
        match_flags = [reason for _, _, reason in scored_matches[:6]]
        return matched_pathways, pathway_relevance_score, match_flags

    def _infer_study_type(
        self,
        title: str,
        abstract: str,
        publication_types: Sequence[str],
    ) -> str:
        text = " ".join([title, abstract, " ".join(publication_types)]).lower()
        publication_blob = " ".join(publication_types).lower()

        if any(term in text for term in ["meta-analysis", "systematic review"]) or "meta-analysis" in publication_blob:
            return "review_meta"
        if any(term in text for term in ["randomized", "randomised", "clinical trial", "intervention"]):
            return "human_trial"
        if any(term in text for term in ["mouse", "mice", "murine", "rat", "animal model"]):
            return "animal"
        if any(term in text for term in ["cell line", "in vitro", "cells", "organoid"]):
            return "cell"
        if any(term in text for term in ["cohort", "case-control", "cross-sectional", "patients", "subjects"]):
            return "human_cohort"
        return "other"

    def _get_extraction_llm(self) -> Optional[ChatOpenAI]:
        """Lazily create the Phase 0 extraction model."""
        if self._extraction_llm is None:
            model_name = self.llm_config.get("extraction_model") or self.llm_config.get("model")
            if not model_name:
                return None
            self._extraction_llm = ChatOpenAI(
                model=model_name,
                temperature=float(self.llm_config.get("temperature", 0.0)),
            )
        return self._extraction_llm

    def _get_filter_llm(self) -> Optional[ChatOpenAI]:
        """Lazily create the Phase 0 relevance filter model."""
        if self._filter_llm is None:
            model_name = self.llm_config.get("extraction_model") or self.llm_config.get("model")
            if not model_name:
                return None
            self._filter_llm = ChatOpenAI(
                model=model_name,
                temperature=float(self.llm_config.get("temperature", 0.0)),
            )
        return self._filter_llm

    def _llm_filter_relevant_papers(
        self,
        papers: Sequence[Dict[str, Any]],
        metabolite_name: str,
        disease_name: str,
    ) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """
        Use the LLM to further refine relevance after rule-based prefiltering.

        Returns:
        - `(included, excluded)` on success
        - `None` on failure, so callers can fall back to rule filtering
        """
        if len(papers) <= 1:
            return list(papers), []

        llm = self._get_filter_llm()
        if llm is None:
            return None

        try:
            abstract_list = self.format_abstracts_for_llm(papers)
            system_prompt, user_prompt = create_evidence_relevance_filter_prompt(
                metabolite=metabolite_name,
                disease=disease_name,
                abstract_list=abstract_list,
            )
            payload = self._invoke_json_llm(llm, system_prompt, user_prompt)
            if not isinstance(payload, dict):
                return None

            included_ids = {str(x) for x in payload.get("included_ids", []) if x}
            excluded_ids = {str(x) for x in payload.get("excluded_ids", []) if x}
            if not included_ids and not excluded_ids:
                return None

            included: List[Dict[str, Any]] = []
            excluded: List[Dict[str, Any]] = []
            for paper in papers:
                paper_id = str(paper.get("id", ""))
                if paper_id in included_ids:
                    included.append(paper)
                elif paper_id in excluded_ids:
                    excluded.append(paper)
                else:
                    included.append(paper)
            return included, excluded
        except Exception as exc:
            logger.warning("LLM relevance filtering failed: %s", exc)
            return None

    def _infer_population_notes(self, title: str, abstract: str) -> Optional[str]:
        text = " ".join([title, abstract])
        lowered = text.lower()
        notes: List[str] = []
        if "single-center" in lowered or "single center" in lowered:
            notes.append("single-center")
        if "multicenter" in lowered or "multi-center" in lowered or "multi center" in lowered:
            notes.append("multicenter")
        if "patients" in lowered:
            notes.append("patients")
        if "healthy controls" in lowered or "control subjects" in lowered:
            notes.append("case-control context")
        return ", ".join(notes) if notes else None

    def _extract_key_quote(self, abstract: str) -> Optional[str]:
        abstract = (abstract or "").strip()
        if not abstract:
            return None
        parts = re.split(r"(?<=[.!?])\s+", abstract)
        return parts[0][:300] if parts else abstract[:300]

    def _extract_mechanistic_notes(self, abstract: str) -> Optional[str]:
        abstract = (abstract or "").strip()
        if not abstract:
            return None
        lowered = abstract.lower()
        if any(term in lowered for term in ["pathway", "mechanism", "inflammation", "oxidative stress", "signaling"]):
            return self._extract_key_quote(abstract)
        return None

    def _unique_terms(self, terms: Sequence[str]) -> List[str]:
        """Normalize, deduplicate, and keep a stable order of terms."""
        seen = set()
        output: List[str] = []
        for term in terms:
            normalized = re.sub(r"\s+", " ", str(term or "").strip())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(normalized)
        return output

    def _chunk_papers(
        self,
        papers: Sequence[Dict[str, Any]],
        chunk_size: int,
    ) -> List[List[Dict[str, Any]]]:
        """Split a paper list into stable batches."""
        if chunk_size <= 0:
            chunk_size = len(papers) or 1
        return [list(papers[i:i + chunk_size]) for i in range(0, len(papers), chunk_size)]

    def _invoke_json_llm(
        self,
        llm: ChatOpenAI,
        system_prompt: str,
        user_prompt: str,
    ) -> Any:
        """Invoke the model and parse a JSON object from the response."""
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        text = self._coerce_response_text(content)
        return self._parse_json_object(text)

    def _coerce_response_text(self, content: Any) -> str:
        """Convert LangChain response content variants into plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
            return "\n".join(parts)
        return str(content)

    def _parse_json_object(self, text: str) -> Any:
        """Parse a JSON object/array from a possibly fenced model response."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty LLM response")

        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fenced_match:
            text = fenced_match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            object_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
            if not object_match:
                raise
            return json.loads(object_match.group(1))

    def _normalize_study_level_evidence(
        self,
        studies: Sequence[Dict[str, Any]],
    ) -> List[StudyLevelEvidence]:
        """Coerce arbitrary extraction rows into the local StudyLevelEvidence schema."""
        normalized: List[StudyLevelEvidence] = []
        valid_study_types = {"human_cohort", "human_trial", "animal", "cell", "review_meta", "other"}
        valid_sample_sizes = {"<50", "50-200", ">200"}
        valid_directions = {
            "increase_risk_or_severity",
            "decrease_risk_or_severity",
            "no_clear_association",
            "mixed_or_unclear",
        }
        valid_significance = {"significant", "non_significant", "not_reported"}
        valid_roles = {"primary_outcome", "secondary_or_exploratory", "incidental_or_background"}

        for study in studies:
            if not isinstance(study, dict):
                continue
            study_type = str(study.get("study_type", "other"))
            study_type = self._normalize_study_type_label(study_type)
            if study_type not in valid_study_types:
                study_type = "other"

            sample_size = study.get("sample_size_category")
            if sample_size not in valid_sample_sizes:
                sample_size = None

            main_direction = study.get("main_direction")
            if main_direction not in valid_directions:
                main_direction = None

            significance = str(study.get("significance", "not_reported"))
            if significance not in valid_significance:
                significance = "not_reported"

            metabolite_role = str(study.get("metabolite_role", "secondary_or_exploratory"))
            if metabolite_role not in valid_roles:
                metabolite_role = "secondary_or_exploratory"

            year = study.get("year")
            if not isinstance(year, int):
                try:
                    year = int(year) if year is not None else None
                except (TypeError, ValueError):
                    year = None

            normalized.append(
                {
                    "study_id": str(study.get("study_id", "Unknown")),
                    "year": year,
                    "study_type": study_type,
                    "sample_size_category": sample_size,
                    "main_direction": main_direction,
                    "significance": significance,
                    "metabolite_role": metabolite_role,
                    "population_notes": self._coerce_optional_str(study.get("population_notes")),
                    "key_quote": self._coerce_optional_str(study.get("key_quote")),
                    "mechanistic_notes": self._coerce_optional_str(study.get("mechanistic_notes")),
                }
            )
        return normalized

    def _postprocess_study_row(
        self,
        study: Dict[str, Any],
        source_paper: Dict[str, Any],
    ) -> StudyLevelEvidence:
        """
        Refine a normalized study row with source-paper metadata.

        This especially targets overuse of `other` from the extraction model.
        """
        title = str(source_paper.get("title", "") or "")
        abstract = str(source_paper.get("abstract", "") or "")
        publication_types = [str(x) for x in source_paper.get("publication_types", [])]

        heuristic_type = self._infer_study_type(
            title=title,
            abstract=abstract,
            publication_types=publication_types,
        )
        current_type = self._normalize_study_type_label(str(study.get("study_type", "other")))
        if current_type == "other":
            current_type = heuristic_type
        elif current_type in {"human_cohort", "human_trial"} and heuristic_type == "review_meta":
            current_type = "review_meta"

        year = study.get("year")
        if year is None and source_paper.get("year") is not None:
            try:
                year = int(source_paper.get("year"))
            except (TypeError, ValueError):
                year = None

        sample_size = study.get("sample_size_category")
        if sample_size not in {"<50", "50-200", ">200"}:
            sample_size = None

        main_direction = study.get("main_direction")
        if main_direction not in {
            "increase_risk_or_severity",
            "decrease_risk_or_severity",
            "no_clear_association",
            "mixed_or_unclear",
        }:
            main_direction = None

        significance = str(study.get("significance", "not_reported"))
        if significance not in {"significant", "non_significant", "not_reported"}:
            significance = "not_reported"

        metabolite_role = str(study.get("metabolite_role", "secondary_or_exploratory"))
        if metabolite_role not in {"primary_outcome", "secondary_or_exploratory", "incidental_or_background"}:
            metabolite_role = "secondary_or_exploratory"

        return {
            "study_id": str(study.get("study_id", source_paper.get("id", "Unknown"))),
            "year": year,
            "study_type": current_type,
            "sample_size_category": sample_size,
            "main_direction": main_direction,
            "significance": significance,
            "metabolite_role": metabolite_role,
            "population_notes": self._coerce_optional_str(
                study.get("population_notes") or self._infer_population_notes(title, abstract)
            ),
            "key_quote": self._coerce_optional_str(
                study.get("key_quote") or self._extract_key_quote(abstract)
            ),
            "mechanistic_notes": self._coerce_optional_str(
                study.get("mechanistic_notes") or self._extract_mechanistic_notes(abstract)
            ),
        }

    def _normalize_study_type_label(self, raw_label: str) -> str:
        """Map free-form or slightly off-schema study type labels into canonical values."""
        text = str(raw_label or "").strip().lower()
        if not text:
            return "other"

        alias_map = {
            "human cohort": "human_cohort",
            "cohort": "human_cohort",
            "case-control": "human_cohort",
            "case control": "human_cohort",
            "cross-sectional": "human_cohort",
            "cross sectional": "human_cohort",
            "observational": "human_cohort",
            "human observational": "human_cohort",
            "trial": "human_trial",
            "clinical trial": "human_trial",
            "rct": "human_trial",
            "intervention": "human_trial",
            "review": "review_meta",
            "systematic review": "review_meta",
            "meta-analysis": "review_meta",
            "meta analysis": "review_meta",
            "animal model": "animal",
            "in vivo": "animal",
            "in vitro": "cell",
            "cell line": "cell",
            "organoid": "cell",
        }
        if text in alias_map:
            return alias_map[text]

        if "meta" in text or "systematic review" in text or text == "review":
            return "review_meta"
        if any(term in text for term in ["trial", "randomized", "randomised", "intervention"]):
            return "human_trial"
        if any(term in text for term in ["cohort", "case-control", "case control", "cross-sectional", "cross sectional", "observational"]):
            return "human_cohort"
        if any(term in text for term in ["mouse", "mice", "murine", "rat", "animal"]):
            return "animal"
        if any(term in text for term in ["cell", "in vitro", "organoid"]):
            return "cell"
        if text in {"human_cohort", "human_trial", "animal", "cell", "review_meta", "other"}:
            return text
        return "other"

    def _deduplicate_studies(
        self,
        studies: Sequence[StudyLevelEvidence],
    ) -> List[StudyLevelEvidence]:
        """Deduplicate study rows by `study_id` while preserving order."""
        seen = set()
        deduped: List[StudyLevelEvidence] = []
        for study in studies:
            study_id = str(study.get("study_id", "Unknown"))
            if study_id in seen:
                continue
            seen.add(study_id)
            deduped.append(study)
        return deduped

    def _unique_list(self, items: Sequence[str]) -> List[str]:
        """Deduplicate a string list while preserving order."""
        seen = set()
        output: List[str] = []
        for item in items:
            clean = str(item).strip()
            if not clean:
                continue
            if clean in seen:
                continue
            seen.add(clean)
            output.append(clean)
        return output

    def _coerce_optional_str(self, value: Any) -> Optional[str]:
        """Normalize nullable free-text values."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _load_json_optional(self, path_value: str) -> Dict[str, Any]:
        """Load a JSON file if present; otherwise return an empty dict."""
        if not path_value:
            return {}
        resolved_path = self._resolve_path(path_value)
        if not resolved_path.exists():
            return {}
        try:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
            return {}
        except Exception as exc:
            logger.warning("Failed to load JSON resource %s: %s", resolved_path, exc)
            return {}

    def _resolve_path(self, path_value: str) -> Path:
        """Resolve project-relative storage paths against the repository root."""
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        project_root = Path(__file__).resolve().parents[3]
        return project_root / candidate
