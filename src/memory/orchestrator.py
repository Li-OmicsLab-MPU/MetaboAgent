"""
Top-level orchestration helpers for long-term memory MVP.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.utils.config_manager import get_config

from .fingerprint import build_dataset_fingerprint
from .retriever import MemoryRetriever
from .semantic_utils import build_strategy_semantic_key, infer_dominant_feature_family
from .store import MemoryStore
from .types import (
    DatasetFingerprint,
    MemoryCase,
    Phase0MemoryPack,
    Phase1MemoryPack,
    Phase2SearchPrior,
    RunMemoryContext,
    SemanticMemoryEntry,
)


class MetaboMemoryOrchestrator:
    """Coordinate fingerprint building, retrieval, and writeback."""

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        retriever: Optional[MemoryRetriever] = None,
    ) -> None:
        self.cfg = get_config()
        self.memory_config = self.cfg.get_memory_config()
        self.store = store or MemoryStore()
        self.retriever = retriever or MemoryRetriever(self.store, self.memory_config)
        self._last_retrieval_trace: Dict[str, Dict[str, Any]] = {}

    def is_enabled(self) -> bool:
        """Return whether memory is globally enabled."""
        return bool(self.memory_config.get("enabled", False))

    def _quality_bucket(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"

    def _normalize_text(self, value: Any) -> str:
        return str(value or "").strip().lower()

    def _build_hint_signature(self, hint: Dict[str, Any]) -> str:
        """Build a stable signature for one structured hint payload."""
        try:
            return json.dumps(hint or {}, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return self._normalize_text(str(hint or ""))

    def _accumulate_weighted_hint(
        self,
        hint_scores: Dict[str, float],
        hint_examples: Dict[str, Dict[str, Any]],
        hint: Any,
        weight: float,
    ) -> None:
        """Accumulate one structured hint example with a quality-derived weight."""
        if not isinstance(hint, dict) or not hint:
            return
        signature = self._build_hint_signature(hint)
        if not signature:
            return
        hint_scores[signature] = hint_scores.get(signature, 0.0) + weight
        hint_examples.setdefault(signature, dict(hint))

    def _collect_strategy_keys(self, disease_name: Optional[str] = None) -> List[str]:
        """Collect strategy semantic keys from episodic cases."""
        normalized_disease = self._normalize_text(disease_name)
        strategy_keys: List[str] = []
        for case in self.store.load_cases():
            if normalized_disease and self._normalize_text(case.get("disease_name")) != normalized_disease:
                continue
            fingerprint = dict(case.get("dataset_fingerprint", {}) or {})
            if not fingerprint:
                continue
            has_strategy_data = bool(case.get("phase1_summary")) or bool(case.get("phase2_summary"))
            if not has_strategy_data:
                continue
            strategy_key = build_strategy_semantic_key(fingerprint)
            if strategy_key:
                strategy_keys.append(strategy_key)
        return list(dict.fromkeys(strategy_keys))

    def get_last_retrieval_trace(self, phase_name: Optional[str] = None) -> Dict[str, Any]:
        """Return the latest memory retrieval trace for one phase or all phases."""
        if phase_name:
            return dict(self._last_retrieval_trace.get(phase_name, {}) or {})
        return {key: dict(value or {}) for key, value in self._last_retrieval_trace.items()}

    def summarize_retrieval_trace(self, trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a compact debug summary from one detailed retrieval trace."""
        payload = dict(trace or {})
        selected_case_ids = list(payload.get("selected_case_ids", []) or [])
        cases = list(payload.get("cases", []) or [])
        filtered_examples = []
        for item in cases:
            if item.get("included"):
                continue
            filtered_examples.append(
                {
                    "case_id": item.get("case_id", ""),
                    "reasons": list(item.get("reasons", []) or [])[:2],
                }
            )
            if len(filtered_examples) >= 3:
                break
        return {
            "phase": payload.get("phase", ""),
            "considered_case_count": int(payload.get("considered_case_count", 0) or 0),
            "eligible_case_count": int(payload.get("eligible_case_count", 0) or 0),
            "filtered_case_count": int(payload.get("filtered_case_count", 0) or 0),
            "selected_case_count": int(payload.get("selected_case_count", 0) or 0),
            "selected_case_ids": selected_case_ids,
            "filtered_examples": filtered_examples,
        }

    def build_dataset_fingerprint(
        self,
        data_path: str,
        target_column: str,
        data_summary: Optional[Dict[str, Any]] = None,
        context_variables: Optional[Dict[str, Any]] = None,
        disease_name: str = "",
        clinical_scenario: str = "",
    ) -> DatasetFingerprint:
        """Build a dataset fingerprint if memory is enabled."""
        if not self.is_enabled():
            return {}
        return build_dataset_fingerprint(
            data_path=data_path,
            target_column=target_column,
            data_summary=data_summary,
            context_variables=context_variables,
            disease_name=disease_name,
            clinical_scenario=clinical_scenario,
        )

    def prepare_phase1_memory_pack(
        self,
        fingerprint: DatasetFingerprint,
    ) -> Phase1MemoryPack:
        """Retrieve advisory Phase 1 hints for the current dataset."""
        if not self.is_enabled() or not bool(self.memory_config.get("read_enabled", True)):
            return {}
        top_k = int((self.memory_config.get("top_k", {}) or {}).get("phase1", 5))
        pack = self.retriever.retrieve_phase1_memory(fingerprint, top_k=top_k)
        self._last_retrieval_trace["phase1"] = self.retriever.get_last_debug_trace("phase1")
        return pack

    def prepare_phase0_memory_pack(self, disease_name: str) -> Phase0MemoryPack:
        """Retrieve advisory Phase 0 disease memory for the current disease."""
        if not self.is_enabled() or not bool(self.memory_config.get("read_enabled", True)):
            return {}
        top_k = int((self.memory_config.get("top_k", {}) or {}).get("phase0", 5))
        pack = self.retriever.retrieve_phase0_memory(disease_name=disease_name, top_k=top_k)
        self._last_retrieval_trace["phase0"] = self.retriever.get_last_debug_trace("phase0")
        return pack

    def prepare_phase2_search_prior(
        self,
        fingerprint: DatasetFingerprint,
        candidate_pool: List[str],
    ) -> Phase2SearchPrior:
        """Retrieve advisory Phase 2 search priors for the current dataset."""
        if not self.is_enabled() or not bool(self.memory_config.get("read_enabled", True)):
            return {}
        top_k = int((self.memory_config.get("top_k", {}) or {}).get("phase2", 8))
        pack = self.retriever.retrieve_phase2_prior(
            fingerprint=fingerprint,
            candidate_pool=candidate_pool,
            top_k=top_k,
        )
        self._last_retrieval_trace["phase2"] = self.retriever.get_last_debug_trace("phase2")
        return pack

    def prepare_run_memory_context(
        self,
        fingerprint: DatasetFingerprint,
        candidate_pool: Optional[List[str]] = None,
    ) -> RunMemoryContext:
        """Build a unified memory context for one workflow run."""
        candidate_pool = candidate_pool or []
        return {
            "memory_enabled": self.is_enabled(),
            "dataset_fingerprint": fingerprint,
            "phase0_memory_pack": self.prepare_phase0_memory_pack(str(fingerprint.get("disease_name", "") or "")),
            "phase1_memory_pack": self.prepare_phase1_memory_pack(fingerprint),
            "phase2_search_prior": self.prepare_phase2_search_prior(fingerprint, candidate_pool),
            "retrieval_trace": self.get_last_retrieval_trace(),
            "writeback_policy": self.memory_config.get("writeback", {}),
        }

    def run_phase2_search_with_memory(
        self,
        *,
        data_path: str,
        target_column: str,
        candidate_pool: List[str],
        sorted_base_pool: List[str],
        taxonomy_map: Dict[str, Any],
        pathway_map: Dict[str, Any],
        priors_dict: Dict[str, float],
        checker_tool: Any = None,
        llm_caller: Any = None,
        clinical_scenario: str = "",
        disease_name: str = "",
        data_summary: Optional[Dict[str, Any]] = None,
        context_variables: Optional[Dict[str, Any]] = None,
        dataset_fingerprint: Optional[DatasetFingerprint] = None,
        phase0_summary: Optional[Dict[str, Any]] = None,
        writeback: bool = True,
        **search_kwargs: Any,
    ) -> Any:
        """
        Unified Phase 2 entry that injects dataset fingerprint and memory priors.

        This helper keeps the existing `run_ptot_search()` interface intact while
        centralizing the memory-aware warm-start logic in one place.
        """
        from src.agents.phase2.ptot_engine import run_ptot_search

        fingerprint = dict(dataset_fingerprint or {})
        if not fingerprint:
            fingerprint = self.build_dataset_fingerprint(
                data_path=data_path,
                target_column=target_column,
                data_summary=data_summary,
                context_variables=context_variables,
                disease_name=disease_name,
                clinical_scenario=clinical_scenario,
            )

        memory_search_prior = self.prepare_phase2_search_prior(
            fingerprint=fingerprint,
            candidate_pool=candidate_pool,
        )
        phase2_retrieval_trace = self.get_last_retrieval_trace("phase2")
        resolved_context_variables = dict(context_variables or {})
        phase0_output = resolved_context_variables.get("phase0_output")
        if not isinstance(phase0_output, dict):
            phase0_output = None

        # Re-resolve Phase 0 priors against the actual Phase 2 feature pool.
        # This closes the hand-off even when an older/partial Phase 1 result did
        # not persist its protected-anchor context.
        if phase0_output:
            try:
                from src.tools.domain.metabolite_name_mapper_enhanced import (
                    extract_phase0_biomarker_terms,
                    resolve_prior_biomarkers_to_columns,
                )

                prior_terms = extract_phase0_biomarker_terms(phase0_output)
                resolution = resolve_prior_biomarkers_to_columns(prior_terms, candidate_pool)
                matched = list(resolution.get("protected_features", []) or [])
                for key in ("protected_anchor_features", "available_prior_anchor_features"):
                    existing = list(resolved_context_variables.get(key, []) or [])
                    resolved_context_variables[key] = list(dict.fromkeys(existing + matched))
                resolved_context_variables["phase0_prior_resolution_report"] = dict(
                    resolution.get("resolution_report", {}) or {}
                )
            except Exception as exc:
                resolved_context_variables["phase0_prior_resolution_report"] = {
                    "error": str(exc),
                    "matched_columns": [],
                    "match_rate": 0.0,
                }
        if "protected_anchor_features" not in search_kwargs:
            search_kwargs["protected_anchor_features"] = list(
                resolved_context_variables.get("protected_anchor_features", []) or []
            )
        if "available_prior_anchor_features" not in search_kwargs:
            search_kwargs["available_prior_anchor_features"] = list(
                resolved_context_variables.get("available_prior_anchor_features", []) or []
            )

        result = run_ptot_search(
            data_path=data_path,
            target_column=target_column,
            candidate_pool=candidate_pool,
            sorted_base_pool=sorted_base_pool,
            taxonomy_map=taxonomy_map,
            pathway_map=pathway_map,
            priors_dict=priors_dict,
            checker_tool=checker_tool,
            llm_caller=llm_caller,
            clinical_scenario=clinical_scenario,
            disease_name=disease_name,
            memory_search_prior=memory_search_prior,
            dataset_fingerprint=fingerprint,
            phase0_output=phase0_output,
            **search_kwargs,
        )
        if isinstance(result, dict):
            phase2_summary = self.build_phase2_summary(
                phase2_result=result,
                memory_search_prior=memory_search_prior,
            )
            result["phase2_summary"] = phase2_summary
            result["memory_debug"] = {
                "phase2": self.summarize_retrieval_trace(phase2_retrieval_trace),
            }
            if writeback:
                memory_case = self.build_phase2_memory_case(
                    disease_name=disease_name,
                    clinical_scenario=clinical_scenario,
                    dataset_fingerprint=fingerprint,
                    phase0_summary=phase0_summary,
                    phase2_summary=phase2_summary,
                    tags=["phase2", "episodic"],
                )
                result["memory_case_id"] = self.write_memory_case(memory_case)
        return result

    def build_phase0_summary(self, phase0_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a compact Phase 0 summary suitable for later disease-memory retrieval."""
        result = dict(phase0_result or {})
        return {
            "disease_name": str(result.get("disease_name", "") or ""),
            "disease_synonyms": list(result.get("disease_synonyms", []) or []),
            "final_priors": list(result.get("final_priors", []) or []),
            "target_pathways": list((result.get("feature_definitions", {}) or {}).get("target_pathways", []) or []),
            "confirmed_biomarkers": list(result.get("confirmed_biomarkers", []) or []),
            "candidate_scores_summary": list(result.get("candidate_scores_summary", []) or []),
            "screening_summary": dict(result.get("screening_summary", {}) or {}),
            "selection_threshold": dict(result.get("selection_threshold", {}) or {}),
        }

    def _estimate_phase0_quality_score(self, phase0_summary: Optional[Dict[str, Any]]) -> float:
        """Estimate a simple writeback quality score for Phase 0 episodic memory."""
        summary = phase0_summary or {}
        score = 0.0

        final_priors = list(summary.get("final_priors", []) or [])
        target_pathways = list(summary.get("target_pathways", []) or [])
        screening_summary = dict(summary.get("screening_summary", {}) or {})

        if final_priors:
            score += 0.35
        if target_pathways:
            score += 0.20

        candidate_count = int(screening_summary.get("candidate_count", 0) or 0)
        stage1_count = int(screening_summary.get("stage1_candidate_count", 0) or 0)
        if candidate_count > 0:
            survival_rate = max(0.0, min(stage1_count / max(candidate_count, 1), 1.0))
            score += 0.15 * survival_rate

        threshold = dict(summary.get("selection_threshold", {}) or {})
        if threshold.get("threshold") is not None:
            score += 0.10
        if summary.get("candidate_scores_summary"):
            score += 0.10
        if summary.get("confirmed_biomarkers"):
            score += 0.10

        return max(0.0, min(1.0, score))

    def _estimate_phase1_quality_score(self, phase1_summary: Optional[Dict[str, Any]]) -> float:
        """Estimate a simple writeback quality score for Phase 1 episodic memory."""
        summary = phase1_summary or {}
        score = 0.0

        if summary.get("completed"):
            score += 0.4
        if not summary.get("had_error"):
            score += 0.2

        total_attempts = int(summary.get("total_execution_attempts", 0) or 0)
        failed_attempts = int(summary.get("failed_execution_count", 0) or 0)
        if total_attempts > 0:
            success_ratio = max(0.0, 1.0 - (failed_attempts / max(total_attempts, 1)))
            score += 0.25 * success_ratio

        retry_count_final = int(summary.get("retry_count_final", 0) or 0)
        retry_penalty = min(0.15, 0.03 * retry_count_final)
        score -= retry_penalty

        if summary.get("best_model"):
            score += 0.1
        if summary.get("final_selected_features"):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _estimate_phase2_quality_score(self, phase2_summary: Optional[Dict[str, Any]]) -> float:
        """Estimate a simple writeback quality score for Phase 2 episodic memory."""
        summary = phase2_summary or {}
        score = 0.0

        winner_features = list(summary.get("winner_features", []) or [])
        winner_scores = summary.get("winner_scores", {}) or {}
        memory_prior_summary = summary.get("memory_prior_summary", {}) or {}

        if winner_features:
            score += 0.35
        if summary.get("stop_reason"):
            score += 0.15
        if summary.get("actual_depth", 0):
            score += 0.10

        try:
            score += 0.20 * max(0.0, min(float(winner_scores.get("f_perf", 0.0) or 0.0), 1.0))
        except (TypeError, ValueError):
            pass
        try:
            score += 0.10 * max(0.0, min(float(winner_scores.get("f_bio", 0.0) or 0.0), 1.0))
        except (TypeError, ValueError):
            pass

        if memory_prior_summary.get("memory_prior_enabled"):
            score += 0.05
        if summary.get("rollback_triggered"):
            score -= 0.05

        return max(0.0, min(1.0, score))

    def build_phase2_summary(
        self,
        *,
        phase2_result: Optional[Dict[str, Any]],
        memory_search_prior: Optional[Phase2SearchPrior] = None,
    ) -> Dict[str, Any]:
        """Build a minimal structured Phase 2 summary from the final search result."""
        result = dict(phase2_result or {})
        search_details = dict(result.get("search_details", {}) or {})
        memory_prior = dict(memory_search_prior or {})

        winner_features = list(result.get("features", []) or [])
        winner_scores = {
            "f_perf": result.get("perf", 0.0),
            "f_bio": result.get("bio", 0.0),
            "f_corr": result.get("corr", 0.0),
            "f_cost": result.get("cost", 0.0),
        }

        return {
            "winner_features": winner_features,
            "winner_feature_count": len(winner_features),
            "winner_scores": winner_scores,
            "selected_model": result.get("selected_model"),
            "metric": result.get("metric"),
            "stop_reason": result.get("stop_reason"),
            "rollback_triggered": bool(result.get("rollback_triggered", False)),
            "min_feature_guard_applied": bool(result.get("min_feature_guard_applied", False)),
            "actual_depth": len(list(search_details.get("layers", []) or [])),
            "global_best_depth": result.get("global_best_depth"),
            "top_candidate_beams": list(result.get("top_candidate_beams", []) or [])[:5],
            "incremental_value": dict(result.get("incremental_value", {}) or {}),
            "protected_anchor_features": list(result.get("protected_anchor_features", []) or []),
            "available_prior_anchor_features": list(result.get("available_prior_anchor_features", []) or []),
            "prior_anchor_union_evaluation": dict(result.get("prior_anchor_union_evaluation", {}) or {}),
            "memory_prior_summary": {
                "memory_prior_enabled": bool(search_details.get("memory_prior_enabled", False)),
                "memory_prior_confidence": float(search_details.get("memory_prior_confidence", 0.0) or 0.0),
                "memory_matched_cases": list(search_details.get("memory_matched_cases", []) or []),
                "memory_reranked_candidate_count": int(search_details.get("memory_reranked_candidate_count", 0) or 0),
                "memory_anchor_hints": list(search_details.get("memory_anchor_hints", []) or []),
                "memory_applied_actions": list(search_details.get("memory_applied_actions", []) or []),
                "retrieved_prior_confidence": float(memory_prior.get("prior_confidence", 0.0) or 0.0),
                "retrieved_anchor_feature_hints": list(memory_prior.get("anchor_feature_hints", []) or []),
                "retrieved_semantic_entry_id": str(memory_prior.get("semantic_entry_id", "") or ""),
                "retrieved_semantic_confidence": float(memory_prior.get("semantic_confidence", 0.0) or 0.0),
                "retrieved_semantic_strategy_key": str(memory_prior.get("semantic_strategy_key", "") or ""),
            },
        }

    def build_phase1_memory_case(
        self,
        *,
        disease_name: str,
        clinical_scenario: str,
        dataset_fingerprint: Optional[DatasetFingerprint] = None,
        phase0_summary: Optional[Dict[str, Any]] = None,
        phase1_summary: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> MemoryCase:
        """Build a minimal episodic memory case from a completed Phase 1 run."""
        fingerprint = dict(dataset_fingerprint or {})
        summary = dict(phase1_summary or {})
        tag_values = list(tags or [])

        if disease_name:
            tag_values.append(f"disease:{disease_name}")
        if clinical_scenario:
            tag_values.append(f"scenario:{clinical_scenario}")
        if summary.get("best_model"):
            tag_values.append(f"model:{summary.get('best_model')}")
        if summary.get("completed") and not summary.get("had_error"):
            tag_values.append("status:success")
        else:
            tag_values.append("status:failed")

        quality_score = self._estimate_phase1_quality_score(summary)
        tag_values.append(f"quality:{self._quality_bucket(quality_score)}")

        return {
            "case_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disease_name": str(disease_name or ""),
            "clinical_scenario": str(clinical_scenario or ""),
            "dataset_fingerprint": fingerprint,
            "phase0_summary": dict(phase0_summary or {}),
            "phase1_summary": summary,
            "quality_score": quality_score,
            "tags": list(dict.fromkeys(tag_values)),
        }

    def build_phase0_memory_case(
        self,
        *,
        disease_name: str,
        clinical_scenario: str = "",
        phase0_summary: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> MemoryCase:
        """Build a minimal episodic memory case from a completed Phase 0 run."""
        summary = dict(phase0_summary or {})
        tag_values = list(tags or [])

        if disease_name:
            tag_values.append(f"disease:{disease_name}")
        if clinical_scenario:
            tag_values.append(f"scenario:{clinical_scenario}")

        for biomarker in list(summary.get("final_priors", []) or [])[:3]:
            tag_values.append(f"prior:{biomarker}")

        if summary.get("final_priors"):
            tag_values.append("status:success")
        else:
            tag_values.append("status:empty")

        quality_score = self._estimate_phase0_quality_score(summary)
        tag_values.append(f"quality:{self._quality_bucket(quality_score)}")

        return {
            "case_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disease_name": str(disease_name or ""),
            "clinical_scenario": str(clinical_scenario or ""),
            "phase0_summary": summary,
            "quality_score": quality_score,
            "tags": list(dict.fromkeys(tag_values)),
        }

    def build_phase2_memory_case(
        self,
        *,
        disease_name: str,
        clinical_scenario: str,
        dataset_fingerprint: Optional[DatasetFingerprint] = None,
        phase0_summary: Optional[Dict[str, Any]] = None,
        phase2_summary: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> MemoryCase:
        """Build a minimal episodic memory case from a completed Phase 2 run."""
        fingerprint = dict(dataset_fingerprint or {})
        summary = dict(phase2_summary or {})
        tag_values = list(tags or [])

        if disease_name:
            tag_values.append(f"disease:{disease_name}")
        if clinical_scenario:
            tag_values.append(f"scenario:{clinical_scenario}")

        winner_features = list(summary.get("winner_features", []) or [])
        for feature in winner_features[:3]:
            tag_values.append(f"winner:{feature}")

        if summary.get("rollback_triggered"):
            tag_values.append("status:rollback")
        elif winner_features:
            tag_values.append("status:success")
        else:
            tag_values.append("status:empty")

        quality_score = self._estimate_phase2_quality_score(summary)
        tag_values.append(f"quality:{self._quality_bucket(quality_score)}")

        return {
            "case_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disease_name": str(disease_name or ""),
            "clinical_scenario": str(clinical_scenario or ""),
            "dataset_fingerprint": fingerprint,
            "phase0_summary": dict(phase0_summary or {}),
            "phase2_summary": summary,
            "quality_score": quality_score,
            "tags": list(dict.fromkeys(tag_values)),
        }

    def _estimate_run_quality_score(
        self,
        *,
        phase0_summary: Optional[Dict[str, Any]] = None,
        phase1_summary: Optional[Dict[str, Any]] = None,
        phase2_summary: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Estimate one overall quality score for the full workflow run."""
        phase0_score = self._estimate_phase0_quality_score(phase0_summary)
        phase1_score = self._estimate_phase1_quality_score(phase1_summary)
        phase2_score = self._estimate_phase2_quality_score(phase2_summary)
        combined = (0.25 * phase0_score) + (0.30 * phase1_score) + (0.45 * phase2_score)
        return max(0.0, min(1.0, combined))

    def build_run_summary(
        self,
        *,
        disease_name: str,
        clinical_scenario: str,
        dataset_fingerprint: Optional[DatasetFingerprint] = None,
        phase0_result: Optional[Dict[str, Any]] = None,
        phase1_result: Optional[Dict[str, Any]] = None,
        phase2_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build one compact summary that represents the entire workflow run."""
        phase0_summary = dict(
            (phase0_result or {}).get("phase0_summary")
            or self.build_phase0_summary(phase0_result)
        )
        phase1_summary = dict((phase1_result or {}).get("phase1_summary", {}) or {})
        phase2_summary = dict((phase2_result or {}).get("phase2_summary", {}) or {})
        fingerprint = dict(dataset_fingerprint or (phase1_result or {}).get("dataset_fingerprint", {}) or {})

        phase_case_ids = {
            "phase0": str((phase0_result or {}).get("memory_case_id", "") or ""),
            "phase1": str((phase1_result or {}).get("memory_case_id", "") or ""),
            "phase2": str((phase2_result or {}).get("memory_case_id", "") or ""),
        }

        return {
            "disease_name": str(disease_name or ""),
            "clinical_scenario": str(clinical_scenario or ""),
            "dataset_fingerprint_available": bool(fingerprint),
            "phase_completion": {
                "phase0": bool(phase0_summary),
                "phase1": bool(phase1_summary),
                "phase2": bool(phase2_summary),
            },
            "phase_case_ids": phase_case_ids,
            "final_priors_count": len(list(phase0_summary.get("final_priors", []) or [])),
            "phase1_selected_feature_count": len(list(phase1_summary.get("final_selected_features", []) or [])),
            "phase2_winner_feature_count": len(list(phase2_summary.get("winner_features", []) or [])),
            "phase1_best_model": phase1_summary.get("best_model"),
            "phase2_stop_reason": phase2_summary.get("stop_reason"),
            "phase2_metric": phase2_summary.get("metric"),
        }

    def build_run_memory_case(
        self,
        *,
        disease_name: str,
        clinical_scenario: str,
        dataset_fingerprint: Optional[DatasetFingerprint] = None,
        phase0_result: Optional[Dict[str, Any]] = None,
        phase1_result: Optional[Dict[str, Any]] = None,
        phase2_result: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> MemoryCase:
        """Build one run-level memory case that contains the full workflow trace."""
        phase0_summary = dict(
            (phase0_result or {}).get("phase0_summary")
            or self.build_phase0_summary(phase0_result)
        )
        phase1_summary = dict((phase1_result or {}).get("phase1_summary", {}) or {})
        phase2_summary = dict((phase2_result or {}).get("phase2_summary", {}) or {})
        fingerprint = dict(dataset_fingerprint or (phase1_result or {}).get("dataset_fingerprint", {}) or {})
        run_summary = self.build_run_summary(
            disease_name=disease_name,
            clinical_scenario=clinical_scenario,
            dataset_fingerprint=fingerprint,
            phase0_result=phase0_result,
            phase1_result=phase1_result,
            phase2_result=phase2_result,
        )
        quality_score = self._estimate_run_quality_score(
            phase0_summary=phase0_summary,
            phase1_summary=phase1_summary,
            phase2_summary=phase2_summary,
        )

        tag_values = list(tags or [])
        if disease_name:
            tag_values.append(f"disease:{disease_name}")
        if clinical_scenario:
            tag_values.append(f"scenario:{clinical_scenario}")
        tag_values.append("case:run")
        tag_values.append(f"quality:{self._quality_bucket(quality_score)}")

        phase_completion = run_summary.get("phase_completion", {}) or {}
        if all(bool(phase_completion.get(name)) for name in ("phase0", "phase1", "phase2")):
            tag_values.append("status:success")
        else:
            tag_values.append("status:partial")

        for feature in list(phase2_summary.get("winner_features", []) or [])[:3]:
            tag_values.append(f"winner:{feature}")

        return {
            "case_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disease_name": str(disease_name or ""),
            "clinical_scenario": str(clinical_scenario or ""),
            "dataset_fingerprint": fingerprint,
            "run_summary": run_summary,
            "phase0_summary": phase0_summary,
            "phase1_summary": phase1_summary,
            "phase2_summary": phase2_summary,
            "quality_score": quality_score,
            "tags": list(dict.fromkeys(tag_values)),
        }

    def _build_disease_semantic_profile(self, disease_name: str) -> Optional[SemanticMemoryEntry]:
        """Distill one disease-level semantic profile from high-quality episodic cases."""
        semantic_config = dict(self.memory_config.get("semantic", {}) or {})
        if not bool(semantic_config.get("enabled", True)):
            return None

        normalized_disease = self._normalize_text(disease_name)
        if not normalized_disease:
            return None

        min_case_quality = float(semantic_config.get("min_case_quality", 0.55) or 0.55)
        max_supporting_cases = int(semantic_config.get("max_supporting_cases", 20) or 20)
        disease_cases = [
            case
            for case in self.store.load_cases()
            if self._normalize_text(case.get("disease_name")) == normalized_disease
            and float(case.get("quality_score", 0.0) or 0.0) >= min_case_quality
        ]
        if not disease_cases:
            return None

        disease_cases = sorted(
            disease_cases,
            key=lambda case: float(case.get("quality_score", 0.0) or 0.0),
            reverse=True,
        )[:max_supporting_cases]

        disease_profile_cfg = dict(semantic_config.get("disease_profile", {}) or {})
        top_biomarkers = int(disease_profile_cfg.get("top_biomarkers", 15) or 15)
        top_pathways = int(disease_profile_cfg.get("top_pathways", 12) or 12)
        top_synonyms = int(disease_profile_cfg.get("top_synonyms", 8) or 8)

        biomarker_scores: Dict[str, float] = {}
        pathway_scores: Dict[str, float] = {}
        synonym_scores: Dict[str, float] = {}
        scenario_scores: Dict[str, float] = {}
        supporting_case_ids: List[str] = []
        max_quality = 0.0

        for case in disease_cases:
            case_id = str(case.get("case_id", "") or "")
            if case_id:
                supporting_case_ids.append(case_id)
            quality = float(case.get("quality_score", 0.0) or 0.0)
            max_quality = max(max_quality, quality)
            weight = max(0.1, quality)

            clinical_scenario = str(case.get("clinical_scenario", "") or "").strip()
            if clinical_scenario:
                scenario_scores[clinical_scenario] = scenario_scores.get(clinical_scenario, 0.0) + weight

            phase0_summary = dict(case.get("phase0_summary", {}) or {})
            phase1_summary = dict(case.get("phase1_summary", {}) or {})
            phase2_summary = dict(case.get("phase2_summary", {}) or {})

            for synonym in phase0_summary.get("disease_synonyms", []) or []:
                synonym_name = str(synonym or "").strip()
                if synonym_name:
                    synonym_scores[synonym_name] = synonym_scores.get(synonym_name, 0.0) + weight

            for pathway in phase0_summary.get("target_pathways", []) or []:
                pathway_name = str(pathway or "").strip()
                if pathway_name:
                    pathway_scores[pathway_name] = pathway_scores.get(pathway_name, 0.0) + weight

            biomarker_sources: List[str] = []
            biomarker_sources.extend(phase0_summary.get("final_priors", []) or [])
            biomarker_sources.extend(phase2_summary.get("winner_features", []) or [])
            biomarker_sources.extend(phase1_summary.get("final_selected_features", []) or [])
            for biomarker in biomarker_sources:
                biomarker_name = str(biomarker or "").strip()
                if biomarker_name:
                    biomarker_scores[biomarker_name] = biomarker_scores.get(biomarker_name, 0.0) + weight

        top_synonym_names = [
            name for name, _ in sorted(synonym_scores.items(), key=lambda item: (-item[1], item[0]))[:top_synonyms]
        ]
        top_pathway_names = [
            name for name, _ in sorted(pathway_scores.items(), key=lambda item: (-item[1], item[0]))[:top_pathways]
        ]
        top_biomarker_items = [
            {"name": name, "score": round(score, 4)}
            for name, score in sorted(biomarker_scores.items(), key=lambda item: (-item[1], item[0]))[:top_biomarkers]
        ]
        preferred_scenarios = [
            name for name, _ in sorted(scenario_scores.items(), key=lambda item: (-item[1], item[0]))[:3]
        ]
        confidence = max(0.0, min(1.0, (0.65 * max_quality) + (0.35 * min(1.0, len(disease_cases) / 5.0))))

        return {
            "entry_type": "disease_profile",
            "semantic_key": normalized_disease,
            "disease_name": disease_name,
            "clinical_scenario": preferred_scenarios[0] if preferred_scenarios else "",
            "confidence": round(confidence, 4),
            "support_count": len(disease_cases),
            "supporting_case_ids": supporting_case_ids,
            "payload": {
                "disease_synonym_hints": top_synonym_names,
                "pathway_family_hints": top_pathway_names,
                "biomarker_seed_hints": top_biomarker_items,
                "preferred_scenarios": preferred_scenarios,
            },
        }

    def _build_strategy_semantic_profile(self, semantic_key: str) -> Optional[SemanticMemoryEntry]:
        """Distill one strategy-level semantic profile from high-quality episodic cases."""
        semantic_config = dict(self.memory_config.get("semantic", {}) or {})
        if not bool(semantic_config.get("enabled", True)):
            return None

        normalized_key = self._normalize_text(semantic_key)
        if not normalized_key:
            return None

        min_case_quality = float(semantic_config.get("min_case_quality", 0.55) or 0.55)
        max_supporting_cases = int(semantic_config.get("max_supporting_cases", 20) or 20)
        strategy_cases = []
        for case in self.store.load_cases():
            quality = float(case.get("quality_score", 0.0) or 0.0)
            if quality < min_case_quality:
                continue
            fingerprint = dict(case.get("dataset_fingerprint", {}) or {})
            if not fingerprint:
                continue
            has_strategy_data = bool(case.get("phase1_summary")) or bool(case.get("phase2_summary"))
            if not has_strategy_data:
                continue
            if build_strategy_semantic_key(fingerprint) != normalized_key:
                continue
            strategy_cases.append(case)

        if not strategy_cases:
            return None

        strategy_cases = sorted(
            strategy_cases,
            key=lambda case: float(case.get("quality_score", 0.0) or 0.0),
            reverse=True,
        )[:max_supporting_cases]

        strategy_profile_cfg = dict(semantic_config.get("strategy_profile", {}) or {})
        top_preprocessing_hints = int(strategy_profile_cfg.get("top_preprocessing_hints", 8) or 8)
        top_fix_hints = int(strategy_profile_cfg.get("top_fix_hints", 6) or 6)
        top_models = int(strategy_profile_cfg.get("top_models", 4) or 4)
        top_feature_methods = int(strategy_profile_cfg.get("top_feature_methods", 4) or 4)
        top_anchor_hints = int(strategy_profile_cfg.get("top_anchor_hints", 12) or 12)
        top_good_groups = int(strategy_profile_cfg.get("top_good_groups", 6) or 6)

        preprocessing_hint_scores: Dict[str, float] = {}
        preprocessing_hint_examples: Dict[str, Dict[str, Any]] = {}
        fix_hint_scores: Dict[str, float] = {}
        fix_hint_examples: Dict[str, Dict[str, Any]] = {}
        model_scores: Dict[str, float] = {}
        feature_method_scores: Dict[str, float] = {}
        anchor_feature_scores: Dict[str, float] = {}
        scenario_scores: Dict[str, float] = {}
        good_group_scores: Dict[str, float] = {}
        good_group_examples: Dict[str, List[str]] = {}
        panel_sizes: List[int] = []
        supporting_case_ids: List[str] = []
        preferred_disease_scores: Dict[str, float] = {}
        signature_snapshot: Dict[str, Any] = {}
        max_quality = 0.0

        for case in strategy_cases:
            case_id = str(case.get("case_id", "") or "")
            if case_id:
                supporting_case_ids.append(case_id)
            quality = float(case.get("quality_score", 0.0) or 0.0)
            max_quality = max(max_quality, quality)
            weight = max(0.1, quality)

            disease_name = str(case.get("disease_name", "") or "").strip()
            if disease_name:
                preferred_disease_scores[disease_name] = preferred_disease_scores.get(disease_name, 0.0) + weight

            clinical_scenario = str(case.get("clinical_scenario", "") or "").strip()
            if clinical_scenario:
                scenario_scores[clinical_scenario] = scenario_scores.get(clinical_scenario, 0.0) + weight

            fingerprint = dict(case.get("dataset_fingerprint", {}) or {})
            if not signature_snapshot:
                signature_snapshot = {
                    "clinical_scenario": str(fingerprint.get("clinical_scenario", "") or ""),
                    "column_naming_style": str(fingerprint.get("column_naming_style", "") or ""),
                    "dominant_feature_family": infer_dominant_feature_family(fingerprint),
                    "n_samples_bucket": build_strategy_semantic_key(fingerprint).split("|")[3].split(":", 1)[-1],
                    "n_features_bucket": build_strategy_semantic_key(fingerprint).split("|")[4].split(":", 1)[-1],
                    "anchor_state": build_strategy_semantic_key(fingerprint).split("|")[5],
                }

            phase1_summary = dict(case.get("phase1_summary", {}) or {})
            phase2_summary = dict(case.get("phase2_summary", {}) or {})

            for hint in phase1_summary.get("recommended_preprocessing_hints", []) or []:
                self._accumulate_weighted_hint(
                    preprocessing_hint_scores,
                    preprocessing_hint_examples,
                    hint,
                    weight,
                )
            for hint in phase1_summary.get("recommended_fix_hints", []) or []:
                self._accumulate_weighted_hint(
                    fix_hint_scores,
                    fix_hint_examples,
                    hint,
                    weight,
                )

            best_model = str(phase1_summary.get("best_model", "") or "").strip()
            if best_model:
                model_scores[best_model] = model_scores.get(best_model, 0.0) + weight

            feature_method = str(phase1_summary.get("feature_selection_method", "") or "").strip()
            if feature_method:
                feature_method_scores[feature_method] = feature_method_scores.get(feature_method, 0.0) + weight

            phase2_anchor_sources: List[str] = []
            phase2_anchor_sources.extend(phase2_summary.get("winner_features", []) or [])
            phase2_anchor_sources.extend(phase2_summary.get("protected_anchor_features", []) or [])
            phase2_anchor_sources.extend(phase2_summary.get("available_prior_anchor_features", []) or [])
            for feature in phase2_anchor_sources:
                feature_name = str(feature or "").strip()
                if feature_name:
                    anchor_feature_scores[feature_name] = anchor_feature_scores.get(feature_name, 0.0) + weight

            winner_features = [
                str(feature or "").strip()
                for feature in phase2_summary.get("winner_features", []) or []
                if str(feature or "").strip()
            ]
            if winner_features:
                panel_sizes.append(len(winner_features))
                group_signature = "|".join(winner_features[:5])
                good_group_scores[group_signature] = good_group_scores.get(group_signature, 0.0) + weight
                good_group_examples.setdefault(group_signature, winner_features[:5])

        preferred_scenarios = [
            name for name, _ in sorted(scenario_scores.items(), key=lambda item: (-item[1], item[0]))[:3]
        ]
        preferred_diseases = [
            name for name, _ in sorted(preferred_disease_scores.items(), key=lambda item: (-item[1], item[0]))[:3]
        ]
        recommended_preprocessing_hints = [
            {
                **dict(preprocessing_hint_examples[signature]),
                "score": round(score, 4),
            }
            for signature, score in sorted(
                preprocessing_hint_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:top_preprocessing_hints]
        ]
        recommended_fix_hints = [
            {
                **dict(fix_hint_examples[signature]),
                "score": round(score, 4),
            }
            for signature, score in sorted(
                fix_hint_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:top_fix_hints]
        ]
        preferred_models = [
            {"name": name, "score": round(score, 4)}
            for name, score in sorted(model_scores.items(), key=lambda item: (-item[1], item[0]))[:top_models]
        ]
        preferred_feature_selection_methods = [
            {"name": name, "score": round(score, 4)}
            for name, score in sorted(
                feature_method_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:top_feature_methods]
        ]
        anchor_feature_hints = [
            name for name, _ in sorted(anchor_feature_scores.items(), key=lambda item: (-item[1], item[0]))[:top_anchor_hints]
        ]
        known_good_feature_groups = [
            list(good_group_examples[signature])
            for signature, _ in sorted(good_group_scores.items(), key=lambda item: (-item[1], item[0]))[:top_good_groups]
        ]
        expected_panel_size_range = [min(panel_sizes), max(panel_sizes)] if panel_sizes else []
        confidence = max(0.0, min(1.0, (0.65 * max_quality) + (0.35 * min(1.0, len(strategy_cases) / 5.0))))

        return {
            "entry_type": "strategy_profile",
            "semantic_key": normalized_key,
            "disease_name": preferred_diseases[0] if preferred_diseases else "",
            "clinical_scenario": preferred_scenarios[0] if preferred_scenarios else "",
            "confidence": round(confidence, 4),
            "support_count": len(strategy_cases),
            "supporting_case_ids": supporting_case_ids,
            "payload": {
                "strategy_signature": signature_snapshot,
                "preferred_scenarios": preferred_scenarios,
                "preferred_diseases": preferred_diseases,
                "recommended_preprocessing_hints": recommended_preprocessing_hints,
                "recommended_fix_hints": recommended_fix_hints,
                "preferred_models": preferred_models,
                "preferred_feature_selection_methods": preferred_feature_selection_methods,
                "anchor_feature_hints": anchor_feature_hints,
                "expected_panel_size_range": expected_panel_size_range,
                "known_good_feature_groups": known_good_feature_groups,
                "known_bad_feature_groups": [],
            },
        }

    def refresh_semantic_memory(self, disease_name: Optional[str] = None) -> Dict[str, Any]:
        """Refresh semantic memory entries from high-quality episodic cases."""
        semantic_config = dict(self.memory_config.get("semantic", {}) or {})
        if not self.is_enabled() or not bool(semantic_config.get("enabled", True)):
            return {"updated_entry_ids": [], "updated_count": 0}

        disease_names: List[str]
        if disease_name:
            disease_names = [str(disease_name or "").strip()]
        else:
            disease_names = sorted(
                {
                    str(case.get("disease_name", "") or "").strip()
                    for case in self.store.load_cases()
                    if str(case.get("disease_name", "") or "").strip()
                }
            )
        strategy_keys = self._collect_strategy_keys(disease_name=disease_name)

        updated_entry_ids: List[str] = []
        for name in disease_names:
            entry = self._build_disease_semantic_profile(name)
            if entry is None:
                continue
            entry_id = self.store.upsert_semantic_entry(entry)
            if entry_id:
                updated_entry_ids.append(entry_id)
        for strategy_key in strategy_keys:
            entry = self._build_strategy_semantic_profile(strategy_key)
            if entry is None:
                continue
            entry_id = self.store.upsert_semantic_entry(entry)
            if entry_id:
                updated_entry_ids.append(entry_id)

        return {
            "updated_entry_ids": list(dict.fromkeys(updated_entry_ids)),
            "updated_count": len(list(dict.fromkeys(updated_entry_ids))),
        }

    def write_memory_case(self, case: MemoryCase) -> str:
        """Persist one episodic memory case if writeback is enabled."""
        if not self.is_enabled() or not bool(self.memory_config.get("write_enabled", True)):
            return ""
        payload = dict(case)
        payload.setdefault("case_id", str(uuid4()))
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        case_id = self.store.append_case(payload)

        semantic_config = dict(self.memory_config.get("semantic", {}) or {})
        if bool(semantic_config.get("enabled", True)) and bool(semantic_config.get("auto_refresh_on_write", True)):
            disease_name = str(payload.get("disease_name", "") or "").strip()
            if disease_name:
                self.refresh_semantic_memory(disease_name)
        return case_id

    def compact_memory_cases(self) -> Dict[str, Any]:
        """Trigger one dedup/compaction pass over the episodic memory store."""
        if not self.is_enabled():
            return {"before_count": 0, "after_count": 0, "removed_count": 0}
        return self.store.compact_cases()
