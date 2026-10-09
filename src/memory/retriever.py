"""
Minimal retrieval layer for long-term memory MVP.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.utils.config_manager import get_config

from .semantic_utils import build_strategy_semantic_key
from .store import MemoryStore
from .types import DatasetFingerprint, MemoryCase, Phase0MemoryPack, Phase1MemoryPack, Phase2SearchPrior


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MemoryRetriever:
    """Retrieve simple Phase 1 and Phase 2 hints from stored memory cases."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        memory_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.memory_config = memory_config or get_config().get_memory_config()
        self._last_debug_trace: Dict[str, Dict[str, Any]] = {}

    def _get_retrieval_config(self) -> Dict[str, Any]:
        return dict(self.memory_config.get("retrieval", {}) or {})

    def _get_phase_retrieval_config(self, phase_name: str) -> Dict[str, Any]:
        retrieval = self._get_retrieval_config()
        return dict(retrieval.get(phase_name, {}) or {})

    def _get_recent_case_limit(self) -> Optional[int]:
        limit = int(self._get_retrieval_config().get("recent_case_limit", 0) or 0)
        return limit if limit > 0 else None

    def _build_hint_signature(self, hint: Dict[str, Any]) -> str:
        try:
            return json.dumps(hint or {}, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(hint or "")

    def _dedupe_structured_hints(self, hints: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        unique_hints: List[Dict[str, Any]] = []
        seen_signatures = set()
        for hint in hints:
            if not isinstance(hint, dict) or not hint:
                continue
            signature = self._build_hint_signature(hint)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_hints.append(dict(hint))
            if len(unique_hints) >= limit:
                break
        return unique_hints

    def _case_has_phase_data(self, case: MemoryCase, phase_name: str) -> bool:
        return bool(case.get(f"{phase_name}_summary"))

    def _is_failed_phase1_case(self, case: MemoryCase) -> bool:
        summary = dict(case.get("phase1_summary", {}) or {})
        return bool(summary) and (not bool(summary.get("completed", False)) or bool(summary.get("had_error", False)))

    def _is_rollback_phase2_case(self, case: MemoryCase) -> bool:
        summary = dict(case.get("phase2_summary", {}) or {})
        return bool(summary) and bool(summary.get("rollback_triggered", False))

    def get_last_debug_trace(self, phase_name: Optional[str] = None) -> Dict[str, Any]:
        """Return the latest retrieval debug trace for one phase or all phases."""
        if phase_name:
            return dict(self._last_debug_trace.get(phase_name, {}) or {})
        return {key: dict(value or {}) for key, value in self._last_debug_trace.items()}

    def _build_case_debug_entry(
        self,
        *,
        case: MemoryCase,
        phase_name: str,
        base_similarity: float,
        included: bool,
        reasons: List[str],
    ) -> Dict[str, Any]:
        return {
            "case_id": str(case.get("case_id", "") or ""),
            "disease_name": str(case.get("disease_name", "") or ""),
            "clinical_scenario": str(case.get("clinical_scenario", "") or ""),
            "quality_score": _safe_float(case.get("quality_score"), 0.0),
            "base_similarity": round(base_similarity, 4),
            "included": bool(included),
            "reasons": list(reasons),
            "has_phase_data": bool(case.get(f"{phase_name}_summary")),
            "tags": list(case.get("tags", []) or []),
        }

    def _passes_phase_quality_gate(
        self,
        case: MemoryCase,
        phase_name: str,
        base_similarity: float = 1.0,
    ) -> tuple[bool, List[str]]:
        phase_config = self._get_phase_retrieval_config(phase_name)
        min_quality = _safe_float(phase_config.get("min_quality_score"), 0.0)
        quality_score = _safe_float(case.get("quality_score"), 0.0)
        reasons: List[str] = []
        if quality_score < min_quality:
            reasons.append(f"quality_below_min:{quality_score:.3f}<{min_quality:.3f}")
            return False, reasons

        if phase_name == "phase0":
            disease_similarity_min = _safe_float(phase_config.get("disease_similarity_min"), 0.0)
            if base_similarity < disease_similarity_min:
                reasons.append(f"disease_similarity_below_min:{base_similarity:.3f}<{disease_similarity_min:.3f}")
                return False, reasons
        elif phase_name == "phase1" and bool(phase_config.get("exclude_failed_cases", True)):
            if self._is_failed_phase1_case(case):
                reasons.append("failed_phase1_case")
                return False, reasons
        elif phase_name == "phase2" and bool(phase_config.get("exclude_rollback_cases", True)):
            if self._is_rollback_phase2_case(case):
                reasons.append("rollback_phase2_case")
                return False, reasons

        reasons.append("passed_quality_gate")
        return True, reasons

    def _score_case_similarity(self, fingerprint: DatasetFingerprint, case: MemoryCase) -> float:
        case_fp = case.get("dataset_fingerprint", {}) or {}
        score = 0.0

        if fingerprint.get("disease_name") and fingerprint.get("disease_name") == case.get("disease_name"):
            score += 3.0
        if fingerprint.get("clinical_scenario") and fingerprint.get("clinical_scenario") == case.get("clinical_scenario"):
            score += 1.0
        if fingerprint.get("column_naming_style") and fingerprint.get("column_naming_style") == case_fp.get("column_naming_style"):
            score += 0.5

        for key in ("n_samples", "n_features", "imbalance_ratio"):
            current_value = _safe_float(fingerprint.get(key))
            case_value = _safe_float(case_fp.get(key))
            if current_value <= 0 or case_value <= 0:
                continue
            relative_gap = abs(current_value - case_value) / max(current_value, case_value, 1.0)
            score += max(0.0, 1.0 - relative_gap)

        score += _safe_float(case.get("quality_score"), 0.0)
        return score

    def _get_top_cases(self, fingerprint: DatasetFingerprint, top_k: int, phase_name: str) -> List[MemoryCase]:
        cases = self.store.load_cases(limit=self._get_recent_case_limit())
        eligible_cases: List[MemoryCase] = []
        debug_cases: List[Dict[str, Any]] = []
        for case in cases:
            base_similarity = self._score_case_similarity(fingerprint, case)
            reasons: List[str] = []
            has_phase_data = self._case_has_phase_data(case, phase_name)
            if not has_phase_data:
                reasons.append("missing_phase_summary")
                debug_cases.append(
                    self._build_case_debug_entry(
                        case=case,
                        phase_name=phase_name,
                        base_similarity=base_similarity,
                        included=False,
                        reasons=reasons,
                    )
                )
                continue

            passed_gate, gate_reasons = self._passes_phase_quality_gate(
                case,
                phase_name=phase_name,
                base_similarity=base_similarity,
            )
            reasons.extend(gate_reasons)
            if passed_gate:
                eligible_cases.append(case)
            debug_cases.append(
                self._build_case_debug_entry(
                    case=case,
                    phase_name=phase_name,
                    base_similarity=base_similarity,
                    included=passed_gate,
                    reasons=reasons,
                )
            )
        ranked = sorted(
            eligible_cases,
            key=lambda case: self._score_case_similarity(fingerprint, case),
            reverse=True,
        )
        selected = ranked[: max(0, top_k)]
        selected_ids = [str(case.get("case_id", "") or "") for case in selected if case.get("case_id")]
        self._last_debug_trace[phase_name] = {
            "phase": phase_name,
            "top_k": top_k,
            "considered_case_count": len(cases),
            "eligible_case_count": len(eligible_cases),
            "selected_case_ids": selected_ids,
            "selected_case_count": len(selected_ids),
            "filtered_case_count": max(0, len(cases) - len(eligible_cases)),
            "cases": debug_cases,
        }
        return selected

    def _score_phase0_case_similarity(self, disease_name: str, case: MemoryCase) -> float:
        current = str(disease_name or "").strip().lower()
        case_disease = str(case.get("disease_name", "") or "").strip().lower()
        if not current or not case_disease:
            return 0.0

        if current == case_disease:
            score = 5.0
        elif current in case_disease or case_disease in current:
            score = 3.0
        else:
            score = 2.0 * SequenceMatcher(None, current, case_disease).ratio()

        phase0_summary = case.get("phase0_summary", {}) or {}
        if phase0_summary:
            score += 0.5
        score += _safe_float(case.get("quality_score"), 0.0)
        return score

    def retrieve_phase0_memory(self, disease_name: str, top_k: int = 5) -> Phase0MemoryPack:
        """Build a simple advisory Phase 0 disease memory pack from similar cases."""
        normalized_disease = str(disease_name or "").strip().lower()
        if not normalized_disease:
            return {}

        cases = self.store.load_cases(limit=self._get_recent_case_limit())
        eligible_cases: List[MemoryCase] = []
        debug_cases: List[Dict[str, Any]] = []
        for case in cases:
            base_similarity = self._score_phase0_case_similarity(normalized_disease, case)
            reasons: List[str] = []
            has_any_phase_data = (
                self._case_has_phase_data(case, "phase0")
                or self._case_has_phase_data(case, "phase1")
                or self._case_has_phase_data(case, "phase2")
            )
            if not has_any_phase_data:
                reasons.append("missing_any_phase_summary")
                debug_cases.append(
                    self._build_case_debug_entry(
                        case=case,
                        phase_name="phase0",
                        base_similarity=base_similarity,
                        included=False,
                        reasons=reasons,
                    )
                )
                continue

            passed_gate, gate_reasons = self._passes_phase_quality_gate(
                case,
                phase_name="phase0",
                base_similarity=base_similarity,
            )
            reasons.extend(gate_reasons)
            if passed_gate:
                eligible_cases.append(case)
            debug_cases.append(
                self._build_case_debug_entry(
                    case=case,
                    phase_name="phase0",
                    base_similarity=base_similarity,
                    included=passed_gate,
                    reasons=reasons,
                )
            )
        ranked = sorted(
            eligible_cases,
            key=lambda case: self._score_phase0_case_similarity(normalized_disease, case),
            reverse=True,
        )
        top_cases = ranked[: max(0, top_k)]

        biomarker_counts: Dict[str, float] = {}
        pathway_counts: Dict[str, float] = {}
        synonym_counts: Dict[str, float] = {}
        matched_diseases: List[str] = []
        provenance_cases: List[str] = []
        semantic_entry = None
        if hasattr(self.store, "get_semantic_entry"):
            semantic_entry = self.store.get_semantic_entry("disease_profile", normalized_disease)

        if semantic_entry:
            payload = dict(semantic_entry.get("payload", {}) or {})
            semantic_weight = max(0.1, _safe_float(semantic_entry.get("confidence"), 0.0))
            semantic_disease_name = str(semantic_entry.get("disease_name", "") or "").strip()
            if semantic_disease_name:
                matched_diseases.append(semantic_disease_name)

            for synonym in payload.get("disease_synonym_hints", []) or []:
                synonym_name = str(synonym or "").strip()
                if synonym_name:
                    synonym_counts[synonym_name] = synonym_counts.get(synonym_name, 0.0) + semantic_weight

            for pathway in payload.get("pathway_family_hints", []) or []:
                pathway_name = str(pathway or "").strip()
                if pathway_name:
                    pathway_counts[pathway_name] = pathway_counts.get(pathway_name, 0.0) + semantic_weight

            for biomarker in payload.get("biomarker_seed_hints", []) or []:
                if isinstance(biomarker, dict):
                    biomarker_name = str(biomarker.get("name", "") or "").strip()
                    biomarker_score = _safe_float(biomarker.get("score"), 0.0)
                else:
                    biomarker_name = str(biomarker or "").strip()
                    biomarker_score = 0.0
                if biomarker_name:
                    biomarker_counts[biomarker_name] = biomarker_counts.get(biomarker_name, 0.0) + max(
                        semantic_weight,
                        biomarker_score,
                    )
            for case_id in semantic_entry.get("supporting_case_ids", []) or []:
                case_name = str(case_id or "").strip()
                if case_name:
                    provenance_cases.append(case_name)

        for case in top_cases:
            case_id = str(case.get("case_id", "") or "")
            if case_id:
                provenance_cases.append(case_id)

            case_disease = str(case.get("disease_name", "") or "")
            if case_disease:
                matched_diseases.append(case_disease)

            case_weight = max(0.1, _safe_float(case.get("quality_score"), 0.0))
            phase0_summary = case.get("phase0_summary", {}) or {}

            for synonym in phase0_summary.get("disease_synonyms", []) or []:
                synonym_name = str(synonym or "").strip()
                if synonym_name:
                    synonym_counts[synonym_name] = synonym_counts.get(synonym_name, 0.0) + case_weight

            for pathway in phase0_summary.get("target_pathways", []) or []:
                pathway_name = str(pathway or "").strip()
                if pathway_name:
                    pathway_counts[pathway_name] = pathway_counts.get(pathway_name, 0.0) + case_weight

            biomarker_sources: List[str] = []
            biomarker_sources.extend(phase0_summary.get("final_priors", []) or [])
            if not biomarker_sources:
                biomarker_sources.extend((case.get("phase2_summary", {}) or {}).get("winner_features", []) or [])
            if not biomarker_sources:
                biomarker_sources.extend((case.get("phase1_summary", {}) or {}).get("final_selected_features", []) or [])

            for biomarker in biomarker_sources:
                biomarker_name = str(biomarker or "").strip()
                if biomarker_name:
                    biomarker_counts[biomarker_name] = biomarker_counts.get(biomarker_name, 0.0) + case_weight

        biomarker_seed_hints = [
            {"name": name, "score": score}
            for name, score in sorted(biomarker_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        ]
        pathway_family_hints = [
            name for name, _ in sorted(pathway_counts.items(), key=lambda item: (-item[1], item[0]))[:15]
        ]
        disease_synonym_hints = [
            name for name, _ in sorted(synonym_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ]

        selected_ids = [str(case.get("case_id", "") or "") for case in top_cases if case.get("case_id")]
        self._last_debug_trace["phase0"] = {
            "phase": "phase0",
            "top_k": top_k,
            "considered_case_count": len(cases),
            "eligible_case_count": len(eligible_cases),
            "selected_case_ids": selected_ids,
            "selected_case_count": len(selected_ids),
            "filtered_case_count": max(0, len(cases) - len(eligible_cases)),
            "cases": debug_cases,
            "semantic_entry_id": str((semantic_entry or {}).get("entry_id", "") or ""),
            "semantic_support_count": int((semantic_entry or {}).get("support_count", 0) or 0),
        }

        episodic_confidence = 0.0 if not top_cases else min(1.0, len(top_cases) / max(1, top_k))
        semantic_confidence = _safe_float((semantic_entry or {}).get("confidence"), 0.0)

        return {
            "matched_diseases": list(dict.fromkeys(matched_diseases)),
            "disease_synonym_hints": disease_synonym_hints,
            "pathway_family_hints": pathway_family_hints,
            "biomarker_seed_hints": biomarker_seed_hints,
            "confidence": max(episodic_confidence, semantic_confidence),
            "provenance_cases": list(dict.fromkeys(provenance_cases)),
            "semantic_entry_id": str((semantic_entry or {}).get("entry_id", "") or ""),
            "semantic_confidence": semantic_confidence,
        }

    def retrieve_phase1_memory(self, fingerprint: DatasetFingerprint, top_k: int = 5) -> Phase1MemoryPack:
        """Build a simple advisory Phase 1 memory pack from similar cases."""
        cases = self._get_top_cases(fingerprint, top_k, phase_name="phase1")
        provenance_cases = [str(case.get("case_id")) for case in cases if case.get("case_id")]
        fix_hints: List[Dict[str, Any]] = []
        preprocessing_hints: List[Dict[str, Any]] = []
        semantic_entry = None
        semantic_strategy_key = build_strategy_semantic_key(fingerprint)
        if hasattr(self.store, "get_semantic_entry") and semantic_strategy_key:
            semantic_entry = self.store.get_semantic_entry("strategy_profile", semantic_strategy_key)

        if semantic_entry:
            payload = dict(semantic_entry.get("payload", {}) or {})
            for hint in payload.get("recommended_preprocessing_hints", []) or []:
                if isinstance(hint, dict):
                    preprocessing_hints.append(hint)
            for hint in payload.get("recommended_fix_hints", []) or []:
                if isinstance(hint, dict):
                    fix_hints.append(hint)
            for case_id in semantic_entry.get("supporting_case_ids", []) or []:
                case_name = str(case_id or "").strip()
                if case_name:
                    provenance_cases.append(case_name)

        for case in cases:
            phase1_summary = case.get("phase1_summary", {}) or {}
            for hint in phase1_summary.get("recommended_fix_hints", []) or []:
                if isinstance(hint, dict):
                    fix_hints.append(hint)
            for hint in phase1_summary.get("recommended_preprocessing_hints", []) or []:
                if isinstance(hint, dict):
                    preprocessing_hints.append(hint)

        preprocessing_hints = self._dedupe_structured_hints(preprocessing_hints, limit=5)
        fix_hints = self._dedupe_structured_hints(fix_hints, limit=5)
        episodic_confidence = 0.0 if not cases else min(1.0, len(cases) / max(1, top_k))
        semantic_confidence = _safe_float((semantic_entry or {}).get("confidence"), 0.0)
        trace = dict(self._last_debug_trace.get("phase1", {}) or {})
        trace["semantic_entry_id"] = str((semantic_entry or {}).get("entry_id", "") or "")
        trace["semantic_support_count"] = int((semantic_entry or {}).get("support_count", 0) or 0)
        trace["semantic_strategy_key"] = semantic_strategy_key
        self._last_debug_trace["phase1"] = trace

        return {
            "recommended_preprocessing_hints": preprocessing_hints,
            "known_failure_patterns": fix_hints,
            "recommended_fix_hints": fix_hints,
            "strategy_confidence": max(episodic_confidence, semantic_confidence),
            "provenance_cases": list(dict.fromkeys(provenance_cases)),
            "semantic_entry_id": str((semantic_entry or {}).get("entry_id", "") or ""),
            "semantic_confidence": semantic_confidence,
            "semantic_strategy_key": semantic_strategy_key,
        }

    def retrieve_phase2_prior(
        self,
        fingerprint: DatasetFingerprint,
        candidate_pool: List[str],
        top_k: int = 8,
    ) -> Phase2SearchPrior:
        """Build a simple advisory Phase 2 search prior from similar cases."""
        cases = self._get_top_cases(fingerprint, top_k, phase_name="phase2")
        candidate_priority_scores: Dict[str, float] = {}
        anchor_feature_hints: List[str] = []
        known_good_feature_groups: List[List[str]] = []
        expected_panel_size_range: List[int] = []
        semantic_entry = None
        semantic_strategy_key = build_strategy_semantic_key(fingerprint)
        if hasattr(self.store, "get_semantic_entry") and semantic_strategy_key:
            semantic_entry = self.store.get_semantic_entry("strategy_profile", semantic_strategy_key)

        if semantic_entry:
            payload = dict(semantic_entry.get("payload", {}) or {})
            semantic_weight = max(0.1, _safe_float(semantic_entry.get("confidence"), 0.0))
            expected_panel_size_range = [
                int(value)
                for value in (payload.get("expected_panel_size_range", []) or [])
                if isinstance(value, (int, float))
            ][:2]
            for feature in payload.get("anchor_feature_hints", []) or []:
                feature_name = str(feature or "").strip()
                if feature_name in candidate_pool:
                    candidate_priority_scores[feature_name] = candidate_priority_scores.get(feature_name, 0.0) + semantic_weight
                    anchor_feature_hints.append(feature_name)
            for group in payload.get("known_good_feature_groups", []) or []:
                if isinstance(group, list):
                    normalized_group = [str(item or "").strip() for item in group if str(item or "").strip()]
                    if normalized_group:
                        known_good_feature_groups.append(normalized_group)

        for case in cases:
            phase2_summary = case.get("phase2_summary", {}) or {}
            winner_features = phase2_summary.get("winner_features", []) or []
            for feature in winner_features:
                feature_name = str(feature)
                if feature_name in candidate_pool:
                    candidate_priority_scores[feature_name] = candidate_priority_scores.get(feature_name, 0.0) + 1.0
                    anchor_feature_hints.append(feature_name)

        provenance_cases = [str(case.get("case_id")) for case in cases if case.get("case_id")]
        if semantic_entry:
            for case_id in semantic_entry.get("supporting_case_ids", []) or []:
                case_name = str(case_id or "").strip()
                if case_name:
                    provenance_cases.append(case_name)
        unique_hints = list(dict.fromkeys(anchor_feature_hints))
        unique_groups: List[List[str]] = []
        seen_group_signatures = set()
        for group in known_good_feature_groups:
            signature = tuple(group)
            if signature in seen_group_signatures:
                continue
            seen_group_signatures.add(signature)
            unique_groups.append(group)
            if len(unique_groups) >= 5:
                break
        episodic_confidence = 0.0 if not cases else min(1.0, len(cases) / max(1, top_k))
        semantic_confidence = _safe_float((semantic_entry or {}).get("confidence"), 0.0)
        trace = dict(self._last_debug_trace.get("phase2", {}) or {})
        trace["semantic_entry_id"] = str((semantic_entry or {}).get("entry_id", "") or "")
        trace["semantic_support_count"] = int((semantic_entry or {}).get("support_count", 0) or 0)
        trace["semantic_strategy_key"] = semantic_strategy_key
        self._last_debug_trace["phase2"] = trace
        return {
            "candidate_priority_scores": candidate_priority_scores,
            "anchor_feature_hints": unique_hints[:10],
            "expected_panel_size_range": expected_panel_size_range,
            "known_good_feature_groups": unique_groups,
            "known_bad_feature_groups": [],
            "prior_confidence": max(episodic_confidence, semantic_confidence),
            "provenance_cases": list(dict.fromkeys(provenance_cases)),
            "semantic_entry_id": str((semantic_entry or {}).get("entry_id", "") or ""),
            "semantic_confidence": semantic_confidence,
            "semantic_strategy_key": semantic_strategy_key,
        }
