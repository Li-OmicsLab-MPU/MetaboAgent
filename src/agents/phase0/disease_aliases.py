"""
Shared disease alias normalization for Phase 0 runtime workflows.
"""

from __future__ import annotations

import re
from typing import Dict, List


_CANONICAL_DISEASE_ALIASES: Dict[str, List[str]] = {
    "hepatocellular carcinoma": [
        "hcc",
        "hepatocellular carcinoma",
        "hepatocellular_carcinoma",
        "hepatocellular cancer",
        "liver cancer",
        "liver_cancer",
        "liver carcinoma",
    ],
}


def _normalize_alias_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical_name, aliases in _CANONICAL_DISEASE_ALIASES.items():
    normalized_canonical = _normalize_alias_token(canonical_name)
    _ALIAS_TO_CANONICAL[normalized_canonical] = canonical_name
    for alias in aliases:
        normalized_alias = _normalize_alias_token(alias)
        if normalized_alias:
            _ALIAS_TO_CANONICAL[normalized_alias] = canonical_name


def canonicalize_phase0_disease_name(disease_name: str) -> str:
    """
    Resolve a runtime disease name to the Phase 0 canonical name when known.
    """
    original = str(disease_name or "").strip()
    if not original:
        return ""
    normalized = _normalize_alias_token(original)
    return _ALIAS_TO_CANONICAL.get(normalized, original)


def get_phase0_disease_aliases(disease_name: str) -> List[str]:
    """
    Return related aliases for a disease, prioritizing the canonical name first.
    """
    canonical_name = canonicalize_phase0_disease_name(disease_name)
    normalized_canonical = _normalize_alias_token(canonical_name)

    aliases: List[str] = []
    if canonical_name:
        aliases.append(canonical_name)

    for alias in _CANONICAL_DISEASE_ALIASES.get(canonical_name, []):
        if _normalize_alias_token(alias) != normalized_canonical:
            aliases.append(alias)

    original = str(disease_name or "").strip()
    if original:
        aliases.append(original)

    deduped: List[str] = []
    seen = set()
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if not alias_text:
            continue
        normalized_alias = _normalize_alias_token(alias_text)
        if normalized_alias in seen:
            continue
        seen.add(normalized_alias)
        deduped.append(alias_text)
    return deduped
