"""
Phase 0 Cache Manager

Manages caching of Phase 0 results to avoid redundant computations.

Author: MetaboAgent Team
Date: 2026-03-27
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from difflib import get_close_matches

from src.agents.phase0.disease_aliases import canonicalize_phase0_disease_name
from src.utils.config_manager import get_config

logger = logging.getLogger(__name__)


class Phase0CacheManager:
    """
    Manages Phase 0 result caching.
    
    Features:
    - Load cached results for known diseases
    - Save new Phase 0 results
    - Check cache validity
    - Update existing cache entries
    - Backup management
    """
    
    CACHE_VERSION = "2.0.0"
    
    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_path: Path to the cache file
        """
        if cache_path is None:
            cache_path = get_config().get_phase0_path("cache_path") or "storage/phase0_cache.json"

        output_dir = get_config().get_phase0_path("output_dir") or "storage/phase0_outputs"
        output_latest = get_config().get_phase0_path("output_latest") or "storage/phase0_output_latest.json"
        evidence_dir = get_config().get_phase0_path("evidence_dir") or "storage/phase0_evidence"
        full_result_dir = get_config().get_phase0_path("full_result_dir") or evidence_dir

        self.cache_path = Path(cache_path)
        self.cache_dir = self.cache_path.parent
        self.backup_dir = self.cache_dir / "phase0_cache_backup"
        self.output_dir = Path(output_dir)
        self.output_latest_path = Path(output_latest)
        self.evidence_dir = Path(evidence_dir)
        self.full_result_dir = Path(full_result_dir)
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.full_result_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logger.info(f"Loaded cache with {len(cache_data.get('diseases', {}))} diseases")
                return cache_data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return self._create_empty_cache()
        else:
            logger.info("No existing cache found, creating new cache")
            return self._create_empty_cache()
    
    def _create_empty_cache(self) -> Dict[str, Any]:
        """Create empty cache structure."""
        return {
            "cache_version": self.CACHE_VERSION,
            "last_updated": datetime.now().isoformat(),
            "diseases": {}
        }
    
    def _save_cache(self):
        """Save cache to file."""
        self.cache["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.info("Cache saved successfully")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _normalize_disease_name(self, disease_name: str) -> str:
        """Normalize disease name for consistent matching."""
        return canonicalize_phase0_disease_name(disease_name).strip().lower()
    
    def has_cache(self, disease_name: str, fuzzy_match: bool = True) -> bool:
        """
        Check if cache exists for a disease.
        
        Args:
            disease_name: Disease name to check
            fuzzy_match: Whether to use fuzzy matching
        
        Returns:
            True if cache exists, False otherwise
        """
        normalized_name = self._normalize_disease_name(disease_name)
        
        # Exact match
        if normalized_name in self.cache["diseases"]:
            return True
        
        # Fuzzy match
        if fuzzy_match:
            fuzzy_cutoff = float(
                get_config().get_phase0_defaults().get("cache_fuzzy_cutoff", 0.8)
            )
            disease_keys = list(self.cache["diseases"].keys())
            matches = get_close_matches(
                normalized_name,
                disease_keys,
                n=1,
                cutoff=fuzzy_cutoff,
            )
            return len(matches) > 0
        
        return False
    
    def get_cache(self, disease_name: str, fuzzy_match: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get cached Phase 0 results for a disease.
        
        Args:
            disease_name: Disease name
            fuzzy_match: Whether to use fuzzy matching
        
        Returns:
            Cached results or None if not found
        """
        normalized_name = self._normalize_disease_name(disease_name)
        
        # Exact match
        if normalized_name in self.cache["diseases"]:
            cached_data = self._hydrate_cached_result(self.cache["diseases"][normalized_name])
            logger.info(f"✓ Cache hit for '{disease_name}'")
            logger.info(f"  - Cached at: {cached_data['cached_at']}")
            logger.info(f"  - Confirmed biomarkers: {len(cached_data.get('confirmed_biomarkers', []))}")
            logger.info(f"  - Enriched pathways: {len(cached_data.get('enriched_pathways', []))}")
            return cached_data
        
        # Fuzzy match
        if fuzzy_match:
            fuzzy_cutoff = float(
                get_config().get_phase0_defaults().get("cache_fuzzy_cutoff", 0.8)
            )
            disease_keys = list(self.cache["diseases"].keys())
            matches = get_close_matches(
                normalized_name,
                disease_keys,
                n=1,
                cutoff=fuzzy_cutoff,
            )
            
            if matches:
                matched_name = matches[0]
                cached_data = self._hydrate_cached_result(self.cache["diseases"][matched_name])
                logger.info(f"✓ Cache hit for '{disease_name}' (matched to '{matched_name}')")
                logger.info(f"  - Cached at: {cached_data['cached_at']}")
                logger.info(f"  - Confirmed biomarkers: {len(cached_data.get('confirmed_biomarkers', []))}")
                logger.info(f"  - Enriched pathways: {len(cached_data.get('enriched_pathways', []))}")
                return cached_data
        
        logger.info(f"✗ Cache miss for '{disease_name}'")
        return None

    def _hydrate_cached_result(self, cached_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rehydrate cached compact results with detailed artifacts when available.

        Cache entries intentionally store a compact Phase 0 summary, but downstream
        export code expects `coarse_screen_map`, `evidence_packs`, and
        `scoring_results` to be present when generating `phase0_full_result.json`.
        When a cache hit occurs we restore those details from the persisted
        `full_result_path`, if available.
        """
        hydrated = dict(cached_data or {})
        full_result_path = str(hydrated.get("full_result_path") or "").strip()
        if not full_result_path:
            return hydrated

        resolved_path = Path(full_result_path)
        if not resolved_path.exists():
            logger.warning("Cached Phase 0 full result path is missing: %s", resolved_path)
            return hydrated

        try:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                full_result_payload = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load cached Phase 0 full result %s: %s", resolved_path, exc)
            return hydrated

        if not isinstance(full_result_payload, dict):
            return hydrated

        for key in (
            "coarse_screen_map",
            "evidence_packs",
            "scoring_results",
            "candidate_scores_summary",
            "selection_threshold",
            "screening_summary",
            "scoring_summary",
            "normalization_stats",
            "runtime_pathway_summary",
            "feature_definitions",
            "final_priors",
        ):
            value = full_result_payload.get(key)
            if value not in (None, {}, []):
                hydrated[key] = value

        return hydrated
    
    def save_cache(
        self,
        disease_name: str,
        phase0_result: Dict[str, Any],
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save Phase 0 results to cache.
        
        Args:
            disease_name: Disease name
            phase0_result: Complete Phase 0 result
            execution_time: Execution time in seconds
            metadata: Additional metadata
        """
        normalized_name = self._normalize_disease_name(disease_name)
        detailed_paths = self._export_evidence_details(disease_name, phase0_result)
        runtime_pathway_summary = self._extract_runtime_pathway_summary(phase0_result)
        
        # Create cache entry
        cache_entry = {
            "disease_name": disease_name,
            "cached_at": datetime.now().isoformat(),
            "phase0_version": self.CACHE_VERSION,
            "execution_time": execution_time,
            "candidates_count": len(phase0_result.get("candidates", [])),
            "stage1_candidates_count": len(phase0_result.get("stage1_candidates", [])),
            "coarse_screen_map": phase0_result.get("coarse_screen_map", {}) or {},
            "evidence_packs": phase0_result.get("evidence_packs", {}) or {},
            "scoring_results": phase0_result.get("scoring_results", {}) or {},
            "confirmed_biomarkers": self._extract_biomarkers(phase0_result),
            "enriched_pathways": self._extract_pathways(phase0_result),
            "feature_definitions": phase0_result.get("feature_definitions", {}),
            "candidate_scores_summary": self._extract_candidate_scores_summary(phase0_result),
            "selection_threshold": self._extract_selection_threshold(phase0_result),
            "screening_summary": self._extract_screening_summary(phase0_result),
            "scoring_summary": self._extract_scoring_summary(phase0_result),
            "normalization_stats": phase0_result.get("normalization_stats", {}),
            "runtime_pathway_summary": runtime_pathway_summary,
            "evidence_dir": detailed_paths.get("evidence_dir"),
            "full_result_path": detailed_paths.get("full_result_path"),
            "metadata": metadata or {}
        }
        
        # Backup old cache if exists
        if normalized_name in self.cache["diseases"]:
            self._backup_cache()
        
        # Save to cache
        self.cache["diseases"][normalized_name] = cache_entry
        self._save_cache()
        
        logger.info(f"✓ Cached Phase 0 results for '{disease_name}'")
        logger.info(f"  - Confirmed biomarkers: {len(cache_entry['confirmed_biomarkers'])}")
        logger.info(f"  - Enriched pathways: {len(cache_entry['enriched_pathways'])}")

    def export_phase0_result(
        self,
        disease_name: str,
        phase0_result: Dict[str, Any],
        cache_hit: bool,
    ) -> Dict[str, str]:
        """
        Export Phase 0 outputs to deterministic paths.

        Writes two stable files:
        1) disease-specific file: storage/phase0_outputs/phase0_<disease>.json
        2) latest file: storage/phase0_output_latest.json
        """
        normalized_name = self._normalize_disease_name(disease_name)
        safe_name = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_") or "unknown_disease"
        disease_output_path = self.output_dir / f"phase0_{safe_name}.json"
        detailed_paths = self._export_evidence_details(disease_name, phase0_result)
        confirmed_biomarkers = phase0_result.get("confirmed_biomarkers") or self._extract_biomarkers(phase0_result)
        final_priors = phase0_result.get("final_priors") or self._extract_biomarker_names(phase0_result)
        candidate_scores_summary = self._extract_candidate_scores_summary(phase0_result)
        selection_threshold = phase0_result.get("selection_threshold") or self._extract_selection_threshold(phase0_result)
        screening_summary = phase0_result.get("screening_summary") or self._extract_screening_summary(phase0_result)
        scoring_summary = phase0_result.get("scoring_summary") or self._extract_scoring_summary(phase0_result)
        runtime_pathway_summary = self._extract_runtime_pathway_summary(phase0_result)

        payload = {
            "disease_name": disease_name,
            "normalized_disease_name": normalized_name,
            "generated_at": datetime.now().isoformat(),
            "cache_hit": cache_hit,
            "final_priors": final_priors,
            "confirmed_biomarkers": confirmed_biomarkers,
            "feature_definitions": phase0_result.get("feature_definitions", {}),
            "candidate_scores_summary": candidate_scores_summary,
            "selection_threshold": selection_threshold,
            "screening_summary": screening_summary,
            "scoring_summary": scoring_summary,
            "normalization_stats": phase0_result.get("normalization_stats", {}),
            "runtime_pathway_summary": runtime_pathway_summary,
            "evidence_dir": detailed_paths.get("evidence_dir"),
            "full_result_path": detailed_paths.get("full_result_path"),
            "execution_time": phase0_result.get("execution_time", 0.0),
            "error": phase0_result.get("error"),
        }

        with open(disease_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(self.output_latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Exported Phase 0 output to {disease_output_path}")
        logger.info(f"✓ Updated latest Phase 0 output at {self.output_latest_path}")

        return {
            "disease_output_path": str(disease_output_path),
            "latest_output_path": str(self.output_latest_path),
        }

    @staticmethod
    def _phase0_evidence_zone(hit_count: int) -> str:
        if int(hit_count) <= 20:
            return "low_evidence"
        if int(hit_count) < 100:
            return "mid_evidence"
        return "high_evidence"

    def _compose_candidate_score_row(
        self,
        *,
        metabolite: str,
        result: Dict[str, Any] | None,
        screen: Dict[str, Any] | None,
        evidence_pack: Dict[str, Any] | None,
        existing_row: Dict[str, Any] | None = None,
        selected_after_threshold: Optional[bool] = None,
    ) -> Dict[str, Any]:
        result = dict(result or {})
        screen = dict(screen or {})
        evidence_pack = dict(evidence_pack or {})
        existing_row = dict(existing_row or {})

        weights = dict(result.get("weights", {}) or {})
        scoring_config = get_config().get_phase0_scoring_config()
        lambda_consistency = float(scoring_config.get("lambda_consistency", 0.5))

        clinical_score = result.get("clinical_evidence", existing_row.get("clinical_score"))
        specificity_score = result.get("disease_specificity", existing_row.get("specificity_score"))
        mechanistic_score = result.get("mechanistic_plausibility", existing_row.get("mechanistic_score"))
        consistency_score = result.get("consistency", existing_row.get("consistency"))
        literature_hit_count = int(
            evidence_pack.get("literature_hit_count")
            or existing_row.get("literature_hit_count")
            or screen.get("total_hit_count")
            or 0
        )

        w_clin = float(weights.get("w_clin", existing_row.get("w_clin", 1.0) or 1.0))
        w_spec = float(weights.get("w_spec", existing_row.get("w_spec", 1.0) or 1.0))
        w_mech = float(weights.get("w_mech", existing_row.get("w_mech", 1.0) or 1.0))

        row = {
            "name": metabolite,
            "metabolite_id": result.get("metabolite_id", screen.get("metabolite_id", existing_row.get("metabolite_id"))),
            "bio_prior_raw": result.get("bio_prior_raw", existing_row.get("bio_prior_raw")),
            "bio_prior_norm": result.get("bio_prior_norm", existing_row.get("bio_prior_norm")),
            "clinical_score": clinical_score,
            "specificity_score": specificity_score,
            "mechanistic_score": mechanistic_score,
            "consistency": consistency_score,
            "memory_seed_hit": bool(existing_row.get("memory_seed_hit", False)),
            "selected_after_threshold": bool(
                selected_after_threshold
                if selected_after_threshold is not None
                else existing_row.get("selected_after_threshold", False)
            ),
            "w_clin": w_clin,
            "w_spec": w_spec,
            "w_mech": w_mech,
            "lambda_consistency": lambda_consistency,
            "clinical_contribution": None if clinical_score is None else round(w_clin * float(clinical_score), 6),
            "specificity_contribution": None if specificity_score is None else round(w_spec * float(specificity_score), 6),
            "mechanistic_contribution": None if mechanistic_score is None else round(w_mech * float(mechanistic_score), 6),
            "consistency_contribution": None if consistency_score is None else round(lambda_consistency * float(consistency_score), 6),
            "evidence_zone": self._phase0_evidence_zone(literature_hit_count),
            "literature_hit_count": literature_hit_count,
            "pubmed_hit_count": int(
                screen.get("stage1_pubmed_hit_count")
                or screen.get("pubmed_hit_count")
                or existing_row.get("pubmed_hit_count")
                or evidence_pack.get("literature_hit_count", 0)
                or 0
            ),
            "stage1_pubmed_hit_count": int(
                screen.get("stage1_pubmed_hit_count")
                or screen.get("pubmed_hit_count")
                or existing_row.get("stage1_pubmed_hit_count")
                or evidence_pack.get("literature_hit_count", 0)
                or 0
            ),
            "stage1_total_hit_count": int(
                screen.get("total_hit_count")
                or existing_row.get("stage1_total_hit_count")
                or existing_row.get("total_hit_count")
                or 0
            ),
            "total_hit_count": int(
                screen.get("total_hit_count")
                or existing_row.get("total_hit_count")
                or existing_row.get("stage1_total_hit_count")
                or 0
            ),
            "evidence_query_hit_count": int(
                (evidence_pack.get("query_metadata", {}) or {}).get(
                    "stage2_retrieval_hit_count",
                    existing_row.get("evidence_query_hit_count", evidence_pack.get("literature_hit_count", 0)),
                )
                or 0
            ),
            "pathway_overlap_count": int(screen.get("pathway_overlap_count", existing_row.get("pathway_overlap_count", 0)) or 0),
            "pathway_relevance_score": float(screen.get("pathway_relevance_score", existing_row.get("pathway_relevance_score", 0.0)) or 0.0),
            "disease_core_pathway_count": int(screen.get("disease_core_pathway_count", existing_row.get("disease_core_pathway_count", 0)) or 0),
            "pathway_relevance_unknown": bool(screen.get("pathway_relevance_unknown", existing_row.get("pathway_relevance_unknown", False))),
            "hard_drop": bool(screen.get("hard_drop", existing_row.get("hard_drop", False))),
            "retain_reason": screen.get("retain_reason", existing_row.get("retain_reason")),
            "drop_reason": screen.get("drop_reason", existing_row.get("drop_reason")),
            "score_confidence": result.get("score_confidence", existing_row.get("score_confidence")),
            "key_pmids": list(result.get("key_pmids", existing_row.get("key_pmids", [])) or []),
        }
        return row

    def _extract_biomarker_names(self, phase0_result: Dict[str, Any]) -> List[str]:
        """
        Resolve biomarker names robustly from multiple Phase 0 result shapes.

        Priority:
        1. Non-empty confirmed_biomarkers
        2. Non-empty final_priors
        3. feature_definitions.target_metabolites
        """
        confirmed_biomarkers = phase0_result.get("confirmed_biomarkers", [])
        if isinstance(confirmed_biomarkers, list) and confirmed_biomarkers:
            return [
                item.get("name")
                for item in confirmed_biomarkers
                if isinstance(item, dict) and item.get("name")
            ]

        final_priors = phase0_result.get("final_priors", [])
        if isinstance(final_priors, list) and final_priors:
            return [str(name) for name in final_priors if name]

        feature_defs = phase0_result.get("feature_definitions", {}) or {}
        target_metabolites = feature_defs.get("target_metabolites", [])
        if isinstance(target_metabolites, list) and target_metabolites:
            return [str(name) for name in target_metabolites if name]

        return []
    
    def _extract_biomarkers(self, phase0_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract confirmed biomarkers from Phase 0 result with v2-compatible fields."""
        if isinstance(phase0_result.get("confirmed_biomarkers"), list) and phase0_result.get("confirmed_biomarkers"):
            return list(phase0_result.get("confirmed_biomarkers", []))

        final_priors = self._extract_biomarker_names(phase0_result)
        candidates = phase0_result.get("candidates", [])
        logprob_scores = phase0_result.get("logprob_scores", {})
        evidence_map = phase0_result.get("evidence_map", {})
        evidence_packs = phase0_result.get("evidence_packs", {})
        scoring_results = phase0_result.get("scoring_results", {})
        coarse_screen_map = phase0_result.get("coarse_screen_map", {})
        
        biomarkers = []
        for prior_name in final_priors:
            # Find candidate info
            candidate_info = next(
                (c for c in candidates if c.get("name") == prior_name),
                None
            )
            
            if candidate_info:
                evidence = evidence_map.get(prior_name, "")
                evidence_count = 0
                if evidence and evidence != "No evidence available.":
                    # Count number of papers (rough estimate)
                    evidence_count = evidence.count("Title:")
                pack = scoring_results.get(prior_name, {})
                evidence_pack = evidence_packs.get(prior_name, {})
                screen_record = coarse_screen_map.get(prior_name, {})
                
                biomarkers.append({
                    "id": candidate_info.get("id"),
                    "name": prior_name,
                    "confidence_score": pack.get("legacy_confidence_score", logprob_scores.get(prior_name, 0.0)),
                    "evidence_count": evidence_pack.get("aggregate_summary", {}).get("total_studies", evidence_count),
                    "pubmed_hit_count": screen_record.get(
                        "stage1_pubmed_hit_count",
                        screen_record.get(
                            "pubmed_hit_count",
                            evidence_pack.get("literature_hit_count", 0),
                        ),
                    ),
                    "stage1_pubmed_hit_count": screen_record.get(
                        "stage1_pubmed_hit_count",
                        evidence_pack.get("literature_hit_count", 0),
                    ),
                    "stage1_total_hit_count": screen_record.get("total_hit_count", 0),
                    "evidence_query_hit_count": evidence_pack.get(
                        "query_metadata",
                        {},
                    ).get("stage2_retrieval_hit_count", evidence_pack.get("literature_hit_count", 0)),
                    "pathway_overlap_count": screen_record.get("pathway_overlap_count", 0),
                    "retain_reason": screen_record.get("retain_reason"),
                    "drop_reason": screen_record.get("drop_reason"),
                    "clinical_score": pack.get("clinical_evidence"),
                    "specificity_score": pack.get("disease_specificity"),
                    "mechanistic_score": pack.get("mechanistic_plausibility"),
                    "consistency": pack.get("consistency"),
                    "bio_prior_raw": pack.get("bio_prior_raw"),
                    "bio_prior_norm": pack.get("bio_prior_norm"),
                    "key_pmids": pack.get("key_pmids", []),
                })
            else:
                biomarkers.append({
                    "id": None,
                    "name": prior_name,
                    "confidence_score": logprob_scores.get(prior_name, 0.0),
                    "evidence_count": 0,
                })
        
        return biomarkers

    def _extract_candidate_scores_summary(self, phase0_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build a concise score table for all candidates with available scoring results."""
        existing_summary = phase0_result.get("candidate_scores_summary", [])
        scoring_results = phase0_result.get("scoring_results", {}) or {}
        coarse_screen_map = phase0_result.get("coarse_screen_map", {}) or {}
        evidence_packs = phase0_result.get("evidence_packs", {}) or {}
        selection_threshold = phase0_result.get("selection_threshold", {}) or {}
        threshold_value = selection_threshold.get("threshold")
        existing_rows_by_name = {
            str(row.get("name")): dict(row)
            for row in (existing_summary or [])
            if isinstance(row, dict) and row.get("name")
        }
        confirmed_by_name = {
            str(row.get("name")): dict(row)
            for row in (phase0_result.get("confirmed_biomarkers", []) or [])
            if isinstance(row, dict) and row.get("name")
        }

        if not scoring_results and not coarse_screen_map and isinstance(existing_summary, list):
            normalized_rows: List[Dict[str, Any]] = []
            for row in existing_summary:
                if not isinstance(row, dict) or not row.get("name"):
                    continue
                biomarker = confirmed_by_name.get(str(row.get("name")), {})
                normalized = dict(row)
                normalized.setdefault("metabolite_id", biomarker.get("id"))
                normalized.setdefault("w_clin", None)
                normalized.setdefault("w_spec", None)
                normalized.setdefault("w_mech", None)
                normalized.setdefault("lambda_consistency", float(get_config().get_phase0_scoring_config().get("lambda_consistency", 0.5)))
                normalized.setdefault("clinical_contribution", None)
                normalized.setdefault("specificity_contribution", None)
                normalized.setdefault("mechanistic_contribution", None)
                normalized.setdefault("consistency_contribution", None)
                normalized.setdefault("evidence_zone", None)
                normalized.setdefault("literature_hit_count", biomarker.get("evidence_query_hit_count"))
                normalized.setdefault("pubmed_hit_count", biomarker.get("pubmed_hit_count"))
                normalized.setdefault("stage1_pubmed_hit_count", biomarker.get("stage1_pubmed_hit_count"))
                normalized.setdefault("stage1_total_hit_count", biomarker.get("stage1_total_hit_count"))
                normalized.setdefault("total_hit_count", biomarker.get("stage1_total_hit_count"))
                normalized.setdefault("evidence_query_hit_count", biomarker.get("evidence_query_hit_count"))
                normalized.setdefault("pathway_overlap_count", biomarker.get("pathway_overlap_count"))
                normalized.setdefault("pathway_relevance_score", None)
                normalized.setdefault("disease_core_pathway_count", None)
                normalized.setdefault("pathway_relevance_unknown", None)
                normalized.setdefault("hard_drop", False)
                normalized.setdefault("retain_reason", biomarker.get("retain_reason"))
                normalized.setdefault("drop_reason", biomarker.get("drop_reason"))
                normalized.setdefault("score_confidence", None)
                normalized.setdefault("key_pmids", biomarker.get("key_pmids", []))
                normalized_rows.append(normalized)
            return normalized_rows

        summary_rows: List[Dict[str, Any]] = []
        for metabolite, result in scoring_results.items():
            screen = coarse_screen_map.get(metabolite, {})
            evidence_pack = evidence_packs.get(metabolite, {})
            selected_after_threshold = None
            bio_prior_raw = result.get("bio_prior_raw")
            if isinstance(bio_prior_raw, (int, float)) and isinstance(threshold_value, (int, float)):
                selected_after_threshold = float(bio_prior_raw) >= float(threshold_value)
            summary_rows.append(
                self._compose_candidate_score_row(
                    metabolite=metabolite,
                    result=result,
                    screen=screen,
                    evidence_pack=evidence_pack,
                    existing_row=existing_rows_by_name.get(metabolite),
                    selected_after_threshold=selected_after_threshold,
                )
            )

        for metabolite, screen in coarse_screen_map.items():
            if metabolite in scoring_results:
                continue
            summary_rows.append(
                self._compose_candidate_score_row(
                    metabolite=metabolite,
                    result=None,
                    screen=screen,
                    evidence_pack=evidence_packs.get(metabolite, {}),
                    existing_row=existing_rows_by_name.get(metabolite),
                )
            )

        for metabolite, row in existing_rows_by_name.items():
            if metabolite in scoring_results or metabolite in coarse_screen_map:
                continue
            summary_rows.append(dict(row))

        return sorted(
            summary_rows,
            key=lambda row: (
                -(
                    row.get("bio_prior_raw")
                    if isinstance(row.get("bio_prior_raw"), (int, float))
                    else float("-inf")
                ),
                row.get("name", ""),
            ),
        )

    def _extract_selection_threshold(self, phase0_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract threshold metadata for downstream inspection."""
        if isinstance(phase0_result.get("selection_threshold"), dict):
            return dict(phase0_result.get("selection_threshold", {}))

        normalization_stats = phase0_result.get("normalization_stats", {}) or {}
        scoring_config = get_config().get_phase0_scoring_config()
        defaults = get_config().get_phase0_defaults()
        threshold_zscore = float(
            scoring_config.get("threshold_zscore", defaults.get("threshold_zscore", 0.5))
        )
        return {
            "rule": scoring_config.get("threshold_rule", "mu_plus_0.5sigma"),
            "zscore": threshold_zscore,
            "threshold": normalization_stats.get("threshold"),
            "mean": normalization_stats.get("mean"),
            "std": normalization_stats.get("std"),
        }

    def _extract_screening_summary(self, phase0_result: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize Stage 1 screening outcomes."""
        if isinstance(phase0_result.get("screening_summary"), dict):
            return dict(phase0_result.get("screening_summary", {}))

        candidates = phase0_result.get("candidates", []) or []
        stage1_candidates = phase0_result.get("stage1_candidates", []) or []
        coarse_screen_map = phase0_result.get("coarse_screen_map", {}) or {}
        records = list(coarse_screen_map.values())
        hard_drop_count = sum(1 for item in records if item.get("hard_drop"))
        zero_hit_records = [item for item in records if int(item.get("total_hit_count", 0)) == 0]
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
            "zero_hit_retained_count": sum(1 for item in zero_hit_records if not item.get("hard_drop")),
            "zero_hit_dropped_count": sum(1 for item in zero_hit_records if item.get("hard_drop")),
            "missing_disease_core_pathway_count": sum(
                1 for item in records if bool(item.get("pathway_relevance_unknown"))
            ),
        }

    def _extract_scoring_summary(self, phase0_result: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize scoring-layer outcomes."""
        if isinstance(phase0_result.get("scoring_summary"), dict):
            return dict(phase0_result.get("scoring_summary", {}))

        scoring_results = phase0_result.get("scoring_results", {}) or {}
        final_priors = phase0_result.get("final_priors", []) or []
        bio_prior_values = [
            result.get("bio_prior_raw")
            for result in scoring_results.values()
            if isinstance(result.get("bio_prior_raw"), (int, float))
        ]
        return {
            "scored_candidate_count": len(scoring_results),
            "confirmed_count": len(final_priors),
            "raw_score_min": min(bio_prior_values) if bio_prior_values else None,
            "raw_score_max": max(bio_prior_values) if bio_prior_values else None,
        }
    
    def _extract_pathways(self, phase0_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract enriched pathways from Phase 0 result."""
        feature_defs = phase0_result.get("feature_definitions", {})
        target_pathways = feature_defs.get("target_pathways", [])
        
        # For now, just return pathway names
        # In the future, could include p-values and metabolite counts
        return [{"name": pathway} for pathway in target_pathways]

    def _extract_runtime_pathway_summary(self, phase0_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a compact export-friendly summary of runtime disease pathways."""
        disease_pathway_pack = phase0_result.get("disease_pathway_pack", {}) or {}
        retrieval_metadata = disease_pathway_pack.get("retrieval_metadata", {}) or {}
        broad_candidates = disease_pathway_pack.get("external_disease_pathways_broad", []) or []

        broad_candidate_preview: List[Dict[str, Any]] = []
        for item in broad_candidates[:20]:
            if not isinstance(item, dict):
                continue
            broad_candidate_preview.append({
                "pathway_id": item.get("pathway_id"),
                "pathway_name": item.get("pathway_name"),
                "db": item.get("db"),
                "rule_score": item.get("rule_score"),
                "interpretability_score": item.get("interpretability_score"),
                "has_small_molecule_support": item.get("has_small_molecule_support"),
            })

        return {
            "cache_key": disease_pathway_pack.get("cache_key"),
            "generated_at": disease_pathway_pack.get("generated_at"),
            "disease_synonyms": disease_pathway_pack.get("disease_synonyms", []),
            "query_terms": disease_pathway_pack.get("query_terms", {}),
            "disease_core_pathways": phase0_result.get("disease_core_pathways")
            or disease_pathway_pack.get("disease_core_pathways", []),
            "top_pathways": phase0_result.get("top_pathways")
            or disease_pathway_pack.get("top_pathways", []),
            "llm_rerank_used": disease_pathway_pack.get("llm_rerank_used", False),
            "llm_trigger_reason": retrieval_metadata.get("llm_trigger_reason"),
            "llm_model": retrieval_metadata.get("llm_model"),
            "llm_rerank_items": disease_pathway_pack.get("llm_rerank_items", []),
            "broad_candidate_count": len(broad_candidates),
            "broad_candidate_preview": broad_candidate_preview,
            "retrieval_metadata": retrieval_metadata,
        }

    def _export_evidence_details(
        self,
        disease_name: str,
        phase0_result: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Export detailed per-candidate Phase 0 evidence artifacts.

        Layout:
        - storage/phase0_evidence/<disease>/candidate_index.json
        - storage/phase0_evidence/<disease>/<metabolite>__evidence_pack.json
        - storage/phase0_evidence/<disease>/<metabolite>__scoring.json
        - storage/phase0_evidence/<disease>/phase0_full_result.json
        """
        normalized_name = self._normalize_disease_name(disease_name)
        safe_disease = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_") or "unknown_disease"
        disease_dir = self.evidence_dir / safe_disease
        disease_dir.mkdir(parents=True, exist_ok=True)

        evidence_packs = phase0_result.get("evidence_packs", {}) or {}
        scoring_results = phase0_result.get("scoring_results", {}) or {}
        coarse_screen_map = phase0_result.get("coarse_screen_map", {}) or {}
        runtime_pathway_summary = self._extract_runtime_pathway_summary(phase0_result)

        candidate_index: List[Dict[str, Any]] = []
        for metabolite in sorted(set(list(evidence_packs.keys()) + list(scoring_results.keys()) + list(coarse_screen_map.keys()))):
            safe_metabolite = re.sub(r"[^a-zA-Z0-9]+", "_", metabolite).strip("_") or "unknown_metabolite"
            evidence_pack_path = disease_dir / f"{safe_metabolite}__evidence_pack.json"
            scoring_path = disease_dir / f"{safe_metabolite}__scoring.json"

            if metabolite in evidence_packs:
                with open(evidence_pack_path, "w", encoding="utf-8") as handle:
                    json.dump(evidence_packs[metabolite], handle, indent=2, ensure_ascii=False)
            if metabolite in scoring_results:
                with open(scoring_path, "w", encoding="utf-8") as handle:
                    json.dump(scoring_results[metabolite], handle, indent=2, ensure_ascii=False)

            candidate_index.append({
                "name": metabolite,
                "evidence_pack_path": str(evidence_pack_path) if metabolite in evidence_packs else None,
                "scoring_path": str(scoring_path) if metabolite in scoring_results else None,
                "hard_drop": coarse_screen_map.get(metabolite, {}).get("hard_drop"),
                "retain_reason": coarse_screen_map.get(metabolite, {}).get("retain_reason"),
                "drop_reason": coarse_screen_map.get(metabolite, {}).get("drop_reason"),
                "stage1_pubmed_hit_count": coarse_screen_map.get(metabolite, {}).get(
                    "stage1_pubmed_hit_count",
                    coarse_screen_map.get(metabolite, {}).get("pubmed_hit_count"),
                ),
                "total_hit_count": coarse_screen_map.get(metabolite, {}).get("total_hit_count"),
                "pathway_overlap_count": coarse_screen_map.get(metabolite, {}).get("pathway_overlap_count"),
                "disease_core_pathway_count": coarse_screen_map.get(metabolite, {}).get("disease_core_pathway_count"),
                "pathway_relevance_unknown": coarse_screen_map.get(metabolite, {}).get("pathway_relevance_unknown"),
            })

        candidate_index_path = disease_dir / "candidate_index.json"
        with open(candidate_index_path, "w", encoding="utf-8") as handle:
            json.dump(candidate_index, handle, indent=2, ensure_ascii=False)

        full_result_payload = {
            "disease_name": disease_name,
            "generated_at": datetime.now().isoformat(),
            "coarse_screen_map": coarse_screen_map,
            "evidence_packs": evidence_packs,
            "scoring_results": scoring_results,
            "final_priors": phase0_result.get("final_priors", []),
            "feature_definitions": phase0_result.get("feature_definitions", {}),
            "candidate_scores_summary": phase0_result.get("candidate_scores_summary") or self._extract_candidate_scores_summary(phase0_result),
            "selection_threshold": phase0_result.get("selection_threshold") or self._extract_selection_threshold(phase0_result),
            "screening_summary": phase0_result.get("screening_summary") or self._extract_screening_summary(phase0_result),
            "scoring_summary": phase0_result.get("scoring_summary") or self._extract_scoring_summary(phase0_result),
            "normalization_stats": phase0_result.get("normalization_stats", {}),
            "runtime_pathway_summary": runtime_pathway_summary,
        }
        full_result_path = disease_dir / "phase0_full_result.json"
        with open(full_result_path, "w", encoding="utf-8") as handle:
            json.dump(full_result_payload, handle, indent=2, ensure_ascii=False)

        return {
            "evidence_dir": str(disease_dir),
            "candidate_index_path": str(candidate_index_path),
            "full_result_path": str(full_result_path),
        }
    
    def _backup_cache(self):
        """Backup current cache file."""
        if self.cache_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"phase0_cache_{timestamp}.json"
            
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Backed up cache to {backup_path}")
            except Exception as e:
                logger.error(f"Error backing up cache: {e}")
    
    def invalidate_cache(self, disease_name: str):
        """
        Invalidate (delete) cache for a disease.
        
        Args:
            disease_name: Disease name
        """
        normalized_name = self._normalize_disease_name(disease_name)
        
        if normalized_name in self.cache["diseases"]:
            # Backup before deletion
            self._backup_cache()
            
            del self.cache["diseases"][normalized_name]
            self._save_cache()
            
            logger.info(f"✓ Invalidated cache for '{disease_name}'")
        else:
            logger.warning(f"✗ No cache found for '{disease_name}'")
    
    def list_cached_diseases(self) -> List[str]:
        """List all cached diseases."""
        return list(self.cache["diseases"].keys())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        diseases = self.cache["diseases"]
        
        return {
            "total_diseases": len(diseases),
            "cache_version": self.cache["cache_version"],
            "last_updated": self.cache["last_updated"],
            "total_biomarkers": sum(
                len(d.get("confirmed_biomarkers", []))
                for d in diseases.values()
            ),
            "total_pathways": sum(
                len(d.get("enriched_pathways", []))
                for d in diseases.values()
            ),
            "diseases": list(diseases.keys())
        }
