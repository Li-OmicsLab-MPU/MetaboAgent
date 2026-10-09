"""
Runtime disease-pathway retrieval and optional LLM reranking for Phase 0.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.phase0.disease_aliases import (
    canonicalize_phase0_disease_name,
    get_phase0_disease_aliases,
)
from src.agents.phase0.disease_pathway_models import (
    DiseasePathwayCandidate,
    DiseasePathwayPack,
    DiseasePathwayRerankItem,
)
from src.agents.phase0.prompts import create_disease_pathway_rerank_prompt
from src.data_ops.build_disease_pathway_map import (
    KEGG_CANDIDATES_JSON,
    REACTOME_HUMAN_JSON,
    REACTOME_SMALL_MOLECULE_JSON,
    build_disease_record,
    build_query_terms,
    infer_disease_families,
)
from src.utils.config_manager import get_config

logger = logging.getLogger(__name__)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return slug or "unknown_disease"


def _json_default_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[3] / candidate


class DiseasePathwayRetriever:
    """Retrieve and rerank disease pathways for the current Phase 0 run."""

    def __init__(self) -> None:
        self.config = get_config()
        self.phase0_llm_config = self.config.get_phase0_llm_config()
        self.pathway_retrieval_config = self.config.get_phase0_pathway_retrieval_config()
        self.phase0_paths = self.config.get_phase0_paths()

        self.reactome_pathways = self._load_json(REACTOME_HUMAN_JSON)
        self.reactome_small_molecule = self._load_json(REACTOME_SMALL_MOLECULE_JSON)
        self.kegg_candidates = self._load_json(KEGG_CANDIDATES_JSON)

        cache_dir_value = self.pathway_retrieval_config.get(
            "runtime_cache_dir",
            "storage/external_pathways/runtime_disease_pathways",
        )
        self.runtime_cache_dir = _json_default_path(cache_dir_value)
        self.runtime_cache_dir.mkdir(parents=True, exist_ok=True)
        self._rerank_llm: Optional[ChatOpenAI] = None

    def get_disease_pathway_pack(
        self,
        disease_name: str,
        disease_synonyms: Optional[Sequence[str]] = None,
        force_refresh: bool = False,
    ) -> DiseasePathwayPack:
        resolved_disease_name = canonicalize_phase0_disease_name(disease_name)
        merged_synonyms = self._merge_runtime_synonyms(
            disease_name=disease_name,
            disease_synonyms=disease_synonyms or [],
        )
        cache_key = self._build_cache_key(resolved_disease_name)
        cache_path = self.runtime_cache_dir / f"{cache_key}.json"

        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                payload.setdefault("retrieval_metadata", {})
                payload["retrieval_metadata"]["cache_hit"] = True
                return payload
            except Exception as exc:
                logger.warning("Failed to load runtime disease pathway cache %s: %s", cache_path, exc)

        pack = self._build_runtime_pack(resolved_disease_name, merged_synonyms)
        pack["cache_key"] = cache_key
        pack.setdefault("retrieval_metadata", {})
        pack["retrieval_metadata"]["cache_hit"] = False
        pack["retrieval_metadata"]["requested_disease_name"] = disease_name

        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(pack, handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to save runtime disease pathway cache %s: %s", cache_path, exc)

        return pack

    @staticmethod
    def _merge_runtime_synonyms(
        disease_name: str,
        disease_synonyms: Sequence[str],
    ) -> List[str]:
        merged: List[str] = []
        for alias in get_phase0_disease_aliases(disease_name):
            if alias:
                merged.append(alias)
        for alias in disease_synonyms:
            alias_text = str(alias or "").strip()
            if alias_text:
                merged.append(alias_text)

        deduped: List[str] = []
        seen = set()
        for alias in merged:
            normalized = alias.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(alias)
        return deduped

    def _build_runtime_pack(
        self,
        disease_name: str,
        disease_synonyms: Sequence[str],
    ) -> DiseasePathwayPack:
        baseline = build_disease_record(
            disease_name=disease_name,
            reactome_pathways=self.reactome_pathways,
            reactome_small_molecule=self.reactome_small_molecule,
            kegg_candidates=self.kegg_candidates,
        )

        query_terms = build_query_terms(disease_name)
        families = infer_disease_families(query_terms)
        broad_records = self._normalize_broad_candidates(baseline.get("all_pathways_ranked", []))

        use_llm, trigger_reason = self._should_use_llm_rerank(
            families=families,
            broad_count=len(broad_records),
            core_count=len(baseline.get("disease_core_pathways", [])),
        )

        llm_rerank_items: List[DiseasePathwayRerankItem] = []
        disease_core_pathways = list(baseline.get("disease_core_pathways", []))
        top_pathways = list(baseline.get("top_pathways", []))

        if use_llm:
            llm_rerank_items = self._rerank_with_llm(
                disease_name=disease_name,
                disease_synonyms=list(disease_synonyms),
                disease_families=families,
                broad_candidates=broad_records,
            )
            kept_names = [
                item["pathway_name"]
                for item in llm_rerank_items
                if item.get("keep_for_core")
            ]
            if kept_names:
                disease_core_pathways = kept_names
                top_pathways = kept_names[:]

        return {
            "disease": disease_name,
            "disease_synonyms": list(disease_synonyms),
            "query_terms": query_terms,
            "external_disease_pathways_broad": broad_records,
            "disease_core_pathways": disease_core_pathways,
            "top_pathways": top_pathways,
            "llm_rerank_used": bool(use_llm and llm_rerank_items),
            "llm_rerank_items": llm_rerank_items,
            "retrieval_metadata": {
                **baseline.get("retrieval_metadata", {}),
                "disease_families": families,
                "llm_trigger_reason": trigger_reason,
                "llm_rerank_input_count": min(
                    len(broad_records),
                    int(self.pathway_retrieval_config.get("llm_rerank_top_k", 30)),
                ),
                "llm_model": self.pathway_retrieval_config.get("llm_rerank_model")
                or self.phase0_llm_config.get("model"),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _normalize_broad_candidates(self, records: Sequence[Dict[str, Any]]) -> List[DiseasePathwayCandidate]:
        normalized: List[DiseasePathwayCandidate] = []
        for item in records:
            db = str(item.get("db", "") or "")
            pathway_id = str(item.get("pathway_id", "") or "")
            has_small_molecule_support = False
            if db == "Reactome":
                support = self.reactome_small_molecule.get(pathway_id, {})
                has_small_molecule_support = bool(support.get("has_small_molecule_support"))

            normalized.append(
                {
                    "pathway_id": pathway_id,
                    "pathway_name": str(item.get("pathway_name", "") or ""),
                    "db": db or "Reactome",
                    "source_url": str(item.get("source_url", "") or ""),
                    "class_labels": list(item.get("class_labels", []) or []),
                    "rule_score": float(item.get("score", 0.0) or 0.0),
                    "interpretability_score": float(item.get("interpretability_score", 0.0) or 0.0),
                    "has_small_molecule_support": has_small_molecule_support,
                    "match_reasons": dict(item.get("match_reasons", {}) or {}),
                }
            )
        return normalized

    def _should_use_llm_rerank(
        self,
        families: Sequence[str],
        broad_count: int,
        core_count: int,
    ) -> tuple[bool, str]:
        if not bool(self.pathway_retrieval_config.get("use_runtime_disease_pathway_retrieval", True)):
            return False, "runtime_retrieval_disabled"
        if not bool(self.pathway_retrieval_config.get("use_llm_rerank", True)):
            return False, "llm_rerank_disabled"

        trigger_config = self.pathway_retrieval_config.get("trigger", {}) or {}
        min_core_count = int(trigger_config.get("min_core_count", 5))
        min_broad_count = int(trigger_config.get("min_broad_count", 20))
        family_triggers = {str(item) for item in trigger_config.get("families", [])}

        if core_count < min_core_count and broad_count >= min_broad_count:
            return True, "low_core_high_broad"
        if any(family in family_triggers for family in families):
            return True, "family_trigger"
        return False, "rule_only"

    def _rerank_with_llm(
        self,
        disease_name: str,
        disease_synonyms: Sequence[str],
        disease_families: Sequence[str],
        broad_candidates: Sequence[DiseasePathwayCandidate],
    ) -> List[DiseasePathwayRerankItem]:
        llm = self._get_rerank_llm()
        if llm is None:
            return []

        top_k = int(self.pathway_retrieval_config.get("llm_rerank_top_k", 30))
        candidate_payload = json.dumps(list(broad_candidates[:top_k]), ensure_ascii=False, indent=2)
        disease_synonyms_text = json.dumps(list(disease_synonyms), ensure_ascii=False)
        disease_families_text = json.dumps(list(disease_families), ensure_ascii=False)
        system_prompt, user_prompt = create_disease_pathway_rerank_prompt(
            disease=disease_name,
            disease_synonyms=disease_synonyms_text,
            disease_families=disease_families_text,
            candidate_pathways=candidate_payload,
        )

        try:
            payload = self._invoke_json_llm(llm, system_prompt, user_prompt)
            items = payload.get("pathway_rankings", []) if isinstance(payload, dict) else []
            return self._normalize_rerank_items(items, broad_candidates[:top_k])
        except Exception as exc:
            logger.warning("Disease pathway LLM rerank failed for %s: %s", disease_name, exc)
            return []

    def _normalize_rerank_items(
        self,
        items: Sequence[Dict[str, Any]],
        broad_candidates: Sequence[DiseasePathwayCandidate],
    ) -> List[DiseasePathwayRerankItem]:
        broad_lookup = {
            (str(item.get("db", "") or ""), str(item.get("pathway_name", "") or "")): item
            for item in broad_candidates
        }
        normalized: List[DiseasePathwayRerankItem] = []
        seen = set()
        for item in items:
            pathway_name = str(item.get("pathway_name", "") or "").strip()
            db = str(item.get("db", "") or "").strip()
            key = (db, pathway_name)
            if not pathway_name or key in seen or key not in broad_lookup:
                continue
            seen.add(key)
            normalized.append(
                {
                    "pathway_name": pathway_name,
                    "db": db or "Reactome",
                    "relevance": self._coerce_enum(str(item.get("relevance", "moderate")), {"high", "moderate", "low"}, "moderate"),
                    "specificity": self._coerce_enum(
                        str(item.get("specificity", "generic")),
                        {"disease_specific", "family_level", "generic"},
                        "generic",
                    ),
                    "metabolite_interpretable": bool(item.get("metabolite_interpretable", False)),
                    "keep_for_core": bool(item.get("keep_for_core", False)),
                    "confidence": self._coerce_enum(
                        str(item.get("confidence", "medium")),
                        {"high", "medium", "low"},
                        "medium",
                    ),
                    "rationale": str(item.get("rationale", "") or "").strip(),
                }
            )
        return normalized

    def _get_rerank_llm(self) -> Optional[ChatOpenAI]:
        if self._rerank_llm is None:
            model_name = (
                self.pathway_retrieval_config.get("llm_rerank_model")
                or self.phase0_llm_config.get("model")
            )
            if not model_name:
                return None

            api_key = str(
                self.phase0_llm_config.get("api_key")
                or os.getenv("OPENAI_API_KEY")
                or ""
            ).strip()
            api_base = str(
                self.phase0_llm_config.get("api_base")
                or os.getenv("OPENAI_API_BASE")
                or ""
            ).strip()
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not found; skipping Phase 0 disease-pathway LLM rerank for this run"
                )
                return None

            try:
                self._rerank_llm = ChatOpenAI(
                    model=model_name,
                    temperature=float(self.pathway_retrieval_config.get("llm_rerank_temperature", 0.0)),
                    openai_api_key=api_key,
                    openai_api_base=api_base if api_base else None,
                )
            except Exception as exc:
                logger.warning("Failed to initialize Phase 0 disease-pathway rerank LLM: %s", exc)
                return None
        return self._rerank_llm

    def _invoke_json_llm(self, llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        text = self._coerce_response_text(content)
        return self._parse_json_object(text)

    @staticmethod
    def _coerce_response_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("Expected top-level JSON object from disease pathway reranker")
        return parsed

    @staticmethod
    def _coerce_enum(value: str, allowed: set[str], default: str) -> str:
        return value if value in allowed else default

    @staticmethod
    def _load_json(path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_cache_key(self, disease_name: str) -> str:
        normalized = _safe_slug(disease_name)
        version_bits = "|".join(
            [
                normalized,
                str(self.pathway_retrieval_config.get("llm_rerank_model") or self.phase0_llm_config.get("model")),
                str(self.pathway_retrieval_config.get("llm_rerank_top_k", 30)),
            ]
        )
        digest = hashlib.sha256(version_bits.encode("utf-8")).hexdigest()[:12]
        return f"{normalized}_{digest}"
