"""
Phase 0 rubric scorer.

This module provides the first scoring-layer implementation for the upgraded
Phase 0 funnel. It keeps the public interface stable while remaining
conservative:

1. Score an EvidencePack on 4 rubric dimensions
2. Apply dynamic evidence-zone weights
3. Compute S_base / S_bio / bio_prior_raw
4. Normalize scores into bio_prior_norm for downstream f_bio migration

The current implementation uses an LLM scoring prompt when available, with a
deterministic heuristic scorer as a safe fallback.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.phase0.evidence_models import EvidencePack, ScoringResult, ScoringWeights
from src.agents.phase0.prompts import create_scoring_prompt
from src.utils.config_manager import get_config

logger = logging.getLogger(__name__)


class Phase0Scorer:
    """Score EvidencePack objects into rubric-driven biological priors."""

    def __init__(self) -> None:
        self.config = get_config()
        self.phase0_config = self.config.get_phase0_config()
        self.scoring_config = self.config.get_phase0_scoring_config()
        self.weights_config = self.config.get_phase0_weights_config()
        self.defaults = self.config.get_phase0_defaults()
        self.llm_config = self.config.get_phase0_llm_config()

        self.lambda_consistency = float(self.scoring_config.get("lambda_consistency", 0.5))
        self.normalize_method = str(self.scoring_config.get("normalize_method", "quantile95"))
        self.normalize_upper_quantile = float(
            self.scoring_config.get(
                "normalize_upper_quantile",
                self.defaults.get("normalize_upper_quantile", 0.95),
            )
        )
        self._scoring_llm: Optional[ChatOpenAI] = None

    def select_dynamic_weights(self, literature_hit_count: int) -> ScoringWeights:
        """Select dynamic weights based on the evidence count zone."""
        if literature_hit_count <= 20:
            weight_block = self.weights_config.get("low_evidence", {})
        elif literature_hit_count < 100:
            weight_block = self.weights_config.get("mid_evidence", {})
        else:
            weight_block = self.weights_config.get("high_evidence", {})

        return {
            "w_clin": float(weight_block.get("w_clin", 1.0)),
            "w_spec": float(weight_block.get("w_spec", 1.0)),
            "w_mech": float(weight_block.get("w_mech", 1.0)),
        }

    def score_evidence_pack(self, evidence_pack: EvidencePack) -> ScoringResult:
        """Score a single EvidencePack using the LLM agent with heuristic fallback."""
        metabolite = str(evidence_pack.get("metabolite", ""))
        disease = str(evidence_pack.get("disease", ""))
        metabolite_id = evidence_pack.get("metabolite_id")
        hit_count = int(evidence_pack.get("literature_hit_count", 0))

        llm_scores = self._score_with_llm(evidence_pack)
        if llm_scores is not None:
            clinical_score = llm_scores["clinical_evidence"]
            specificity_score = llm_scores["disease_specificity"]
            mechanism_score = llm_scores["mechanistic_plausibility"]
            consistency_score = llm_scores["consistency"]
            rationale = llm_scores["rationale"]
            key_pmids = llm_scores["key_pmids"]
            fallback_used = False
        else:
            clinical_score, clinical_rationale = self._score_clinical_evidence(evidence_pack)
            specificity_score, specificity_rationale = self._score_disease_specificity(evidence_pack)
            mechanism_score, mechanism_rationale = self._score_mechanistic_plausibility(evidence_pack)
            consistency_score, consistency_rationale = self._score_consistency(evidence_pack)
            key_pmids = list(evidence_pack.get("included_pmids", [])[:5])
            rationale = {
                "clinical_evidence": clinical_rationale,
                "disease_specificity": specificity_rationale,
                "mechanistic_plausibility": mechanism_rationale,
                "consistency": consistency_rationale,
                "overall": "Heuristic fallback used because LLM scoring was unavailable or invalid.",
            }
            fallback_used = True

        weights = self.select_dynamic_weights(hit_count)
        s_base = (
            weights["w_clin"] * clinical_score
            + weights["w_spec"] * specificity_score
            + weights["w_mech"] * mechanism_score
        )
        s_bio = s_base + self.lambda_consistency * consistency_score

        overall_suffix = (
            f" Hit count={hit_count}; weights=[clin={weights['w_clin']}, "
            f"spec={weights['w_spec']}, mech={weights['w_mech']}]."
        )
        if fallback_used:
            rationale["overall"] = (rationale.get("overall", "") + overall_suffix).strip()
        else:
            rationale["overall"] = (rationale.get("overall", "") + overall_suffix).strip()

        score_confidence = self._estimate_score_confidence(evidence_pack, consistency_score)

        return {
            "metabolite": metabolite,
            "metabolite_id": metabolite_id,
            "disease": disease,
            "clinical_evidence": clinical_score,
            "disease_specificity": specificity_score,
            "mechanistic_plausibility": mechanism_score,
            "consistency": consistency_score,
            "weights": weights,
            "s_base": float(s_base),
            "s_bio": float(s_bio),
            "bio_prior_raw": float(s_bio),
            "bio_prior_norm": None,
            "legacy_confidence_score": float(s_bio),
            "score_confidence": float(score_confidence),
            "rationale": rationale,
            "key_pmids": key_pmids,
        }

    def score_evidence_packs(
        self,
        evidence_packs: Dict[str, EvidencePack],
    ) -> Dict[str, ScoringResult]:
        """
        Score a batch of EvidencePacks and normalize `bio_prior_norm` jointly.
        """
        scored: Dict[str, ScoringResult] = {}
        for metabolite, evidence_pack in evidence_packs.items():
            try:
                scored[metabolite] = self.score_evidence_pack(evidence_pack)
            except Exception as exc:
                logger.error("Failed to score EvidencePack for %s: %s", metabolite, exc, exc_info=True)

        return self.normalize_scoring_results(scored)

    def normalize_scoring_results(
        self,
        scoring_results: Dict[str, ScoringResult],
        method: Optional[str] = None,
    ) -> Dict[str, ScoringResult]:
        """
        Normalize raw biological prior scores into `[0, 1]`.

        Methods:
        - `theoretical`: normalize by the maximum achievable score under the
          candidate-specific weights.
        - `quantile95`: normalize by the empirical 95th percentile of the batch,
          with theoretical fallback for tiny batches.
        """
        if not scoring_results:
            return {}

        chosen_method = str(method or self.normalize_method or "quantile95")
        raw_scores = [float(result.get("bio_prior_raw", 0.0)) for result in scoring_results.values()]

        quantile_upper = self._percentile(raw_scores, self.normalize_upper_quantile)
        theoretical_upper = max(
            self.compute_theoretical_max(result.get("weights", {})) for result in scoring_results.values()
        )

        if chosen_method == "theoretical":
            normalization_upper = max(theoretical_upper, 1e-8)
        else:
            if len(raw_scores) < 5 or quantile_upper <= 0:
                normalization_upper = max(theoretical_upper, 1e-8)
            else:
                normalization_upper = max(quantile_upper, 1e-8)

        normalized: Dict[str, ScoringResult] = {}
        for metabolite, result in scoring_results.items():
            bio_prior_raw = float(result.get("bio_prior_raw", 0.0))
            norm_value = min(max(bio_prior_raw / normalization_upper, 0.0), 1.0)
            normalized[metabolite] = {
                **result,
                "bio_prior_norm": float(norm_value),
            }

        return normalized

    def compute_theoretical_max(self, weights: Dict[str, float]) -> float:
        """Compute the theoretical rubric maximum under a specific weight set."""
        w_clin = float(weights.get("w_clin", 1.0))
        w_spec = float(weights.get("w_spec", 1.0))
        w_mech = float(weights.get("w_mech", 1.0))
        return 3.0 * w_clin + 3.0 * w_spec + 3.0 * w_mech + self.lambda_consistency

    def summarize_normalization_stats(
        self,
        scoring_results: Dict[str, ScoringResult],
    ) -> Dict[str, float]:
        """Return batch-level summary stats for reporting / state storage."""
        raw_scores = [float(result.get("bio_prior_raw", 0.0)) for result in scoring_results.values()]
        if not raw_scores:
            return {
                "raw_min": 0.0,
                "raw_max": 0.0,
                "upper_bound": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "threshold": 0.0,
            }

        mean = sum(raw_scores) / len(raw_scores)
        std = self._sample_std(raw_scores)
        upper_bound = self._percentile(raw_scores, self.normalize_upper_quantile)
        threshold = mean + float(self.scoring_config.get("threshold_zscore", 0.5)) * std

        return {
            "raw_min": min(raw_scores),
            "raw_max": max(raw_scores),
            "upper_bound": upper_bound,
            "mean": mean,
            "std": std,
            "threshold": threshold,
        }

    def _score_with_llm(self, evidence_pack: EvidencePack) -> Optional[Dict[str, Any]]:
        """
        Use the Scoring Agent prompt to assign rubric scores.

        Returns a normalized dict on success, otherwise `None` so the caller can
        fall back to deterministic heuristics.
        """
        llm = self._get_scoring_llm()
        if llm is None:
            return None

        try:
            metabolite = str(evidence_pack.get("metabolite", ""))
            disease = str(evidence_pack.get("disease", ""))
            serialized_pack = serialize_evidence_pack_for_prompt(evidence_pack)
            system_prompt, user_prompt = create_scoring_prompt(
                metabolite=metabolite,
                disease=disease,
                evidence_pack=serialized_pack,
            )
            payload = self._invoke_json_llm(llm, system_prompt, user_prompt)
            if not isinstance(payload, dict):
                return None
            return self._normalize_llm_scores(payload, evidence_pack)
        except Exception as exc:
            logger.warning(
                "LLM scoring failed for %s/%s: %s",
                evidence_pack.get("metabolite", ""),
                evidence_pack.get("disease", ""),
                exc,
            )
            return None

    def _get_scoring_llm(self) -> Optional[ChatOpenAI]:
        """Lazily initialize the scoring LLM."""
        if self._scoring_llm is None:
            model_name = self.llm_config.get("scoring_model") or self.llm_config.get("model")
            if not model_name:
                return None
            self._scoring_llm = ChatOpenAI(
                model=model_name,
                temperature=float(self.llm_config.get("temperature", 0.0)),
            )
        return self._scoring_llm

    def _invoke_json_llm(
        self,
        llm: ChatOpenAI,
        system_prompt: str,
        user_prompt: str,
    ) -> Any:
        """Invoke the LLM and parse a JSON object from the response body."""
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
        """Parse a JSON object/array from a raw model response."""
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty LLM scoring response")

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

    def _normalize_llm_scores(
        self,
        payload: Dict[str, Any],
        evidence_pack: EvidencePack,
    ) -> Optional[Dict[str, Any]]:
        """Validate and normalize the LLM scoring payload."""
        clinical = self._coerce_bounded_int(payload.get("clinical_evidence"), min_value=0, max_value=3)
        specificity = self._coerce_bounded_int(payload.get("disease_specificity"), min_value=0, max_value=3)
        mechanism = self._coerce_bounded_int(payload.get("mechanistic_plausibility"), min_value=0, max_value=3)
        consistency = self._coerce_bounded_int(payload.get("consistency"), min_value=-1, max_value=1)

        if any(value is None for value in [clinical, specificity, mechanism, consistency]):
            return None

        rationale_payload = payload.get("rationale", {})
        if not isinstance(rationale_payload, dict):
            rationale_payload = {}

        included_pmids = {str(x) for x in evidence_pack.get("included_pmids", []) if x}
        key_pmids: List[str] = []
        for item in payload.get("key_pmids", []) or []:
            candidate = str(item).strip()
            if not candidate:
                continue
            if included_pmids and candidate not in included_pmids:
                continue
            if candidate not in key_pmids:
                key_pmids.append(candidate)

        if not key_pmids:
            key_pmids = list(evidence_pack.get("included_pmids", [])[:5])

        return {
            "clinical_evidence": clinical,
            "disease_specificity": specificity,
            "mechanistic_plausibility": mechanism,
            "consistency": consistency,
            "rationale": {
                "clinical_evidence": self._coerce_text(rationale_payload.get("clinical_evidence")),
                "disease_specificity": self._coerce_text(rationale_payload.get("disease_specificity")),
                "mechanistic_plausibility": self._coerce_text(rationale_payload.get("mechanistic_plausibility")),
                "consistency": self._coerce_text(rationale_payload.get("consistency")),
                "overall": self._coerce_text(rationale_payload.get("overall")),
            },
            "key_pmids": key_pmids,
        }

    def _coerce_bounded_int(
        self,
        value: Any,
        min_value: int,
        max_value: int,
    ) -> Optional[int]:
        """Convert a scalar into a bounded integer, else return None."""
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        if numeric < min_value or numeric > max_value:
            return None
        return numeric

    def _coerce_text(self, value: Any) -> str:
        """Normalize rationale text fields."""
        if value is None:
            return ""
        return str(value).strip()

    def _score_clinical_evidence(self, evidence_pack: EvidencePack) -> Tuple[int, str]:
        summary = evidence_pack.get("aggregate_summary", {})
        study_counts = summary.get("study_type_counts", {})
        sample_counts = summary.get("sample_size_counts", {})
        human_count = int(study_counts.get("human_cohort", 0)) + int(study_counts.get("human_trial", 0))
        trial_count = int(study_counts.get("human_trial", 0))
        large_count = int(sample_counts.get(">200", 0))
        review_flag = bool(summary.get("any_systematic_review_or_meta", False))

        if human_count == 0:
            return 0, "No human studies detected in the EvidencePack."
        if review_flag or (human_count >= 3 and (large_count > 0 or trial_count > 0)):
            return 3, "Human evidence includes review/meta or multiple stronger human studies."
        if human_count >= 2 or trial_count > 0:
            return 2, "Multiple human studies are present, but strength remains moderate."
        return 1, "Only limited human evidence is present."

    def _score_disease_specificity(self, evidence_pack: EvidencePack) -> Tuple[int, str]:
        summary = evidence_pack.get("aggregate_summary", {})
        pathway_context = evidence_pack.get("pathway_mechanism_context", {})
        comments = str(summary.get("comments", "") or "").lower()
        human_count = int(summary.get("study_type_counts", {}).get("human_cohort", 0)) + int(
            summary.get("study_type_counts", {}).get("human_trial", 0)
        )
        overlap_count = len(pathway_context.get("overlapping_pathways", []) or [])
        review_flag = bool(summary.get("any_systematic_review_or_meta", False))

        if human_count == 0:
            return 0, "Specificity cannot be supported without human disease evidence."
        if review_flag and human_count >= 2:
            return 3, "Review/meta-level evidence plus repeated human studies supports stronger disease specificity."
        if overlap_count >= 2 and human_count >= 2:
            return 2, "Repeated disease-linked context and pathway overlap suggest moderate specificity."
        if "generic" in comments or "placeholder" in comments:
            return 1, "Evidence is disease-relevant but remains too generic for higher specificity."
        return 1, "Some disease-linked evidence exists, but specificity remains limited."

    def _score_mechanistic_plausibility(self, evidence_pack: EvidencePack) -> Tuple[int, str]:
        pathway_context = evidence_pack.get("pathway_mechanism_context", {})
        summary = evidence_pack.get("aggregate_summary", {})
        overlap_count = len(pathway_context.get("overlapping_pathways", []) or [])
        neighbor_count = len(pathway_context.get("reaction_neighbors", []) or [])
        causal_flags = len(pathway_context.get("causal_evidence_flags", []) or [])
        mechanism_notes = len(summary.get("overall_mechanistic_summary", []) or [])

        if causal_flags > 0:
            return 3, "Causal or intervention-style mechanism flags are present."
        if overlap_count >= 2 or (overlap_count >= 1 and neighbor_count >= 1):
            return 2, "The metabolite maps onto disease-linked pathways with nontrivial network context."
        if mechanism_notes > 0 or neighbor_count > 0:
            return 1, "Only hypothesis-level or indirect mechanistic support is available."
        return 0, "Mechanistic linkage is weak or absent in the current EvidencePack."

    def _score_consistency(self, evidence_pack: EvidencePack) -> Tuple[int, str]:
        summary = evidence_pack.get("aggregate_summary", {})
        direction_counts = summary.get("direction_counts", {})
        pos = int(direction_counts.get("increase_risk_or_severity", 0))
        neg = int(direction_counts.get("decrease_risk_or_severity", 0))
        nulls = int(direction_counts.get("no_clear_association", 0))
        mixed = int(direction_counts.get("mixed_or_unclear", 0))
        total = int(summary.get("total_studies", 0))

        if total <= 2:
            return 0, "Evidence remains too sparse to judge consistency."
        if pos > 0 and neg > 0:
            return -1, "Directionality is conflicting across included studies."
        dominant = max(pos, neg)
        if dominant >= 2 and nulls == 0 and mixed <= 1:
            return 1, "Most studies point in the same direction."
        return 0, "The direction of association is not yet stable enough."

    def _estimate_score_confidence(self, evidence_pack: EvidencePack, consistency: int) -> float:
        """Estimate confidence in the assigned score, scaled to `[0, 1]`."""
        summary = evidence_pack.get("aggregate_summary", {})
        total_studies = int(summary.get("total_studies", 0))
        human_count = int(summary.get("study_type_counts", {}).get("human_cohort", 0)) + int(
            summary.get("study_type_counts", {}).get("human_trial", 0)
        )
        review_flag = 1.0 if summary.get("any_systematic_review_or_meta", False) else 0.0

        confidence = 0.15
        confidence += min(total_studies / 10.0, 0.35)
        confidence += min(human_count / 5.0, 0.30)
        confidence += 0.10 * review_flag
        if consistency == 1:
            confidence += 0.10
        elif consistency == -1:
            confidence -= 0.10
        return min(max(confidence, 0.0), 1.0)

    def _percentile(self, values: Sequence[float], q: float) -> float:
        """Compute a simple linear-interpolated percentile."""
        if not values:
            return 0.0
        clamped_q = min(max(float(q), 0.0), 1.0)
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        pos = clamped_q * (len(ordered) - 1)
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return float(ordered[lower])
        fraction = pos - lower
        return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)

    def _sample_std(self, values: Sequence[float]) -> float:
        """Sample standard deviation with safe fallback for tiny batches."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(max(variance, 0.0))


def score_evidence_pack(evidence_pack: EvidencePack) -> ScoringResult:
    """Convenience wrapper for scoring a single EvidencePack."""
    return Phase0Scorer().score_evidence_pack(evidence_pack)


def score_evidence_packs(evidence_packs: Dict[str, EvidencePack]) -> Dict[str, ScoringResult]:
    """Convenience wrapper for scoring a batch of EvidencePacks."""
    return Phase0Scorer().score_evidence_packs(evidence_packs)


def serialize_evidence_pack_for_prompt(evidence_pack: EvidencePack) -> str:
    """Serialize an EvidencePack into deterministic JSON for the scoring prompt."""
    return json.dumps(evidence_pack, ensure_ascii=False, indent=2)
