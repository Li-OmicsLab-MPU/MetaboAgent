"""
Minimal storage layer for long-term memory MVP.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from src.utils.config_manager import get_config

from .types import MemoryCase, SemanticMemoryEntry


class MemoryStore:
    """Persist and load episodic memory cases from JSONL."""

    def __init__(
        self,
        memory_config: Optional[Dict[str, Any]] = None,
        case_store_path: Optional[str | Path] = None,
        semantic_store_path: Optional[str | Path] = None,
    ) -> None:
        cfg = get_config()
        self.memory_config = memory_config or cfg.get_memory_config()
        self.storage_dir = Path(self.memory_config.get("storage_dir", "storage/memory"))
        self.case_store_path = Path(
            case_store_path or self.memory_config.get("case_store_path", str(self.storage_dir / "memory_cases.jsonl"))
        )
        self.semantic_store_path = Path(
            semantic_store_path or self.memory_config.get("semantic_store_path", str(self.storage_dir / "semantic_memory.json"))
        )
        self.ensure_storage()

    def ensure_storage(self) -> None:
        """Create memory directories and placeholder files if they do not exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.case_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.semantic_store_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.case_store_path.exists():
            self.case_store_path.touch()
        if not self.semantic_store_path.exists():
            with open(self.semantic_store_path, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": "memory.semantic.v1", "entries": []}, handle, indent=2)

    def append_case(self, case: MemoryCase) -> str:
        """Append one episodic memory case to the JSONL case store."""
        case_id = str(case.get("case_id") or uuid4())
        payload = dict(case)
        payload["case_id"] = case_id
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())

        compaction_config = dict(self.memory_config.get("compaction", {}) or {})
        if bool(compaction_config.get("enabled", True)) and bool(compaction_config.get("dedup_on_write", True)):
            recent_window = int(compaction_config.get("recent_window", 200) or 200)
            recent_cases = self.load_cases(limit=recent_window)
            existing_case = self._find_duplicate_case(payload, recent_cases)
            if existing_case is not None:
                return str(existing_case.get("case_id", "") or case_id)

        with open(self.case_store_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return case_id

    def load_cases(self, limit: Optional[int] = None) -> List[MemoryCase]:
        """Load cases from JSONL store, newest last in file order."""
        if not self.case_store_path.exists():
            return []

        cases: List[MemoryCase] = []
        with open(self.case_store_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    cases.append(payload)

        if limit is not None and limit > 0:
            return cases[-limit:]
        return cases

    def load_semantic_store(self) -> Dict[str, Any]:
        """Load the semantic memory JSON payload."""
        self.ensure_storage()
        if not self.semantic_store_path.exists():
            return {"schema_version": "memory.semantic.v1", "entries": []}
        try:
            with open(self.semantic_store_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            payload = {"schema_version": "memory.semantic.v1", "entries": []}
        if not isinstance(payload, dict):
            payload = {"schema_version": "memory.semantic.v1", "entries": []}
        payload.setdefault("schema_version", "memory.semantic.v1")
        payload.setdefault("entries", [])
        return payload

    def write_semantic_store(self, payload: Dict[str, Any]) -> None:
        """Persist the semantic memory JSON payload."""
        self.ensure_storage()
        normalized = dict(payload or {})
        normalized.setdefault("schema_version", "memory.semantic.v1")
        normalized.setdefault("entries", [])
        with open(self.semantic_store_path, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, ensure_ascii=False)

    def load_semantic_entries(self, entry_type: Optional[str] = None) -> List[SemanticMemoryEntry]:
        """Return semantic memory entries, optionally filtered by entry type."""
        payload = self.load_semantic_store()
        entries = payload.get("entries", []) or []
        if not isinstance(entries, list):
            return []
        normalized_entries = [entry for entry in entries if isinstance(entry, dict)]
        if entry_type:
            return [entry for entry in normalized_entries if str(entry.get("entry_type", "") or "") == entry_type]
        return normalized_entries

    def get_semantic_entry(self, entry_type: str, semantic_key: str) -> Optional[SemanticMemoryEntry]:
        """Return one semantic entry by type and normalized key."""
        normalized_key = self._normalize_text(semantic_key)
        for entry in self.load_semantic_entries(entry_type=entry_type):
            if self._normalize_text(entry.get("semantic_key")) == normalized_key:
                return entry
        return None

    def upsert_semantic_entry(self, entry: SemanticMemoryEntry) -> str:
        """Insert or replace one semantic entry by `(entry_type, semantic_key)`."""
        payload = self.load_semantic_store()
        entries = [item for item in (payload.get("entries", []) or []) if isinstance(item, dict)]
        normalized_entry = dict(entry or {})
        normalized_entry.setdefault("entry_id", str(uuid4()))
        normalized_entry.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        normalized_entry["updated_at"] = datetime.now(timezone.utc).isoformat()

        target_type = self._normalize_text(normalized_entry.get("entry_type"))
        target_key = self._normalize_text(normalized_entry.get("semantic_key"))
        replaced = False
        for idx, item in enumerate(entries):
            if self._normalize_text(item.get("entry_type")) != target_type:
                continue
            if self._normalize_text(item.get("semantic_key")) != target_key:
                continue
            normalized_entry.setdefault("created_at", item.get("created_at"))
            normalized_entry.setdefault("entry_id", item.get("entry_id", normalized_entry["entry_id"]))
            entries[idx] = normalized_entry
            replaced = True
            break
        if not replaced:
            entries.append(normalized_entry)

        payload["entries"] = entries
        self.write_semantic_store(payload)
        return str(normalized_entry.get("entry_id", "") or "")

    def _normalize_text(self, value: Any) -> str:
        return str(value or "").strip().lower()

    def _normalize_feature_list(self, values: Any, limit: int = 5) -> Tuple[str, ...]:
        if not isinstance(values, list):
            return ()
        normalized = [self._normalize_text(value) for value in values if self._normalize_text(value)]
        return tuple(sorted(dict.fromkeys(normalized))[:limit])

    def _detect_case_type(self, case: MemoryCase) -> str:
        tags = set(str(tag) for tag in case.get("tags", []) or [])
        if "case:run" in tags or case.get("run_summary"):
            return "run"
        if case.get("phase2_summary") and not case.get("phase1_summary"):
            return "phase2"
        if case.get("phase1_summary") and not case.get("phase2_summary"):
            return "phase1"
        if case.get("phase0_summary") and not case.get("phase1_summary") and not case.get("phase2_summary"):
            return "phase0"
        if case.get("phase2_summary"):
            return "phase2"
        if case.get("phase1_summary"):
            return "phase1"
        if case.get("phase0_summary"):
            return "phase0"
        return "generic"

    def _build_dedup_signature(self, case: MemoryCase) -> Tuple[Any, ...]:
        case_type = self._detect_case_type(case)
        disease_name = self._normalize_text(case.get("disease_name"))
        clinical_scenario = self._normalize_text(case.get("clinical_scenario"))
        fingerprint = dict(case.get("dataset_fingerprint", {}) or {})

        if case_type == "phase0":
            phase0_summary = dict(case.get("phase0_summary", {}) or {})
            return (
                case_type,
                disease_name,
                clinical_scenario,
                self._normalize_feature_list(phase0_summary.get("final_priors"), limit=5),
                self._normalize_feature_list(phase0_summary.get("target_pathways"), limit=5),
            )
        if case_type == "phase1":
            phase1_summary = dict(case.get("phase1_summary", {}) or {})
            return (
                case_type,
                disease_name,
                clinical_scenario,
                self._normalize_text(phase1_summary.get("best_model")),
                self._normalize_feature_list(phase1_summary.get("final_selected_features"), limit=5),
                int(fingerprint.get("n_samples", 0) or 0),
                int(fingerprint.get("n_features", 0) or 0),
            )
        if case_type == "phase2":
            phase2_summary = dict(case.get("phase2_summary", {}) or {})
            return (
                case_type,
                disease_name,
                clinical_scenario,
                self._normalize_feature_list(phase2_summary.get("winner_features"), limit=5),
                self._normalize_text(phase2_summary.get("metric")),
                self._normalize_text(phase2_summary.get("stop_reason")),
            )
        if case_type == "run":
            run_summary = dict(case.get("run_summary", {}) or {})
            phase2_summary = dict(case.get("phase2_summary", {}) or {})
            return (
                case_type,
                disease_name,
                clinical_scenario,
                self._normalize_feature_list(phase2_summary.get("winner_features"), limit=5),
                int(run_summary.get("final_priors_count", 0) or 0),
                int(run_summary.get("phase1_selected_feature_count", 0) or 0),
            )
        return (
            case_type,
            disease_name,
            clinical_scenario,
            self._normalize_text(case.get("case_id")),
        )

    def _select_better_case(self, left: MemoryCase, right: MemoryCase) -> MemoryCase:
        keep_strategy = str((self.memory_config.get("compaction", {}) or {}).get("keep_strategy", "highest_quality_then_latest"))
        left_quality = float(left.get("quality_score", 0.0) or 0.0)
        right_quality = float(right.get("quality_score", 0.0) or 0.0)
        if keep_strategy == "highest_quality_then_latest":
            if right_quality > left_quality:
                return right
            if left_quality > right_quality:
                return left
        left_created_at = self._normalize_text(left.get("created_at"))
        right_created_at = self._normalize_text(right.get("created_at"))
        return right if right_created_at >= left_created_at else left

    def _find_duplicate_case(self, candidate: MemoryCase, recent_cases: List[MemoryCase]) -> Optional[MemoryCase]:
        candidate_signature = self._build_dedup_signature(candidate)
        for existing in reversed(recent_cases):
            if self._build_dedup_signature(existing) != candidate_signature:
                continue
            better = self._select_better_case(existing, candidate)
            if better is existing:
                return existing
        return None

    def compact_cases(self) -> Dict[str, Any]:
        """Deduplicate the JSONL case store and keep the best case for each signature."""
        cases = self.load_cases()
        if not cases:
            return {"before_count": 0, "after_count": 0, "removed_count": 0}

        deduped: Dict[Tuple[Any, ...], MemoryCase] = {}
        for case in cases:
            signature = self._build_dedup_signature(case)
            if signature not in deduped:
                deduped[signature] = case
                continue
            deduped[signature] = self._select_better_case(deduped[signature], case)

        before_count = len(cases)
        deduped_cases = list(deduped.values())
        deduped_cases = sorted(
            deduped_cases,
            key=lambda case: self._normalize_text(case.get("created_at")),
        )

        with open(self.case_store_path, "w", encoding="utf-8") as handle:
            for payload in deduped_cases:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return {
            "before_count": before_count,
            "after_count": len(deduped_cases),
            "removed_count": max(0, before_count - len(deduped_cases)),
        }
