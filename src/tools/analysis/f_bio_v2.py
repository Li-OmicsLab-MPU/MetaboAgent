"""
Anchor-Conditional f_bio v2 scaffolding.

This module defines the core data contracts and placeholder entry points for the
next-generation biological prior scorer used by Phase 2 / Phase 3. The actual
scoring logic will be implemented incrementally in follow-up steps.
"""

from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from src.utils.config_manager import get_config
from src.utils.feature_name_standardizer import standardize_feature_name


AnchorMode = Literal["exact_anchor", "external_anchor", "disease_only"]
FeatureType = Literal["single", "sum", "taxsum", "ratio", "pathway", "unknown"]


DEFAULT_F_BIO_V2_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "mode_selection": "auto",
    "gamma_topo": 0.05,
    "delta_panel": 0.03,
    "topk_ratio": 0.3,
    "topk_min_k": 2,
    "aggregation": {
        "single": "direct",
        "sum": "topk_mean",
        "pathway": "topk_mean",
        "ratio": "minmax_blend",
    },
    "structure_bonus": {
        "single": 0.0,
        "sum": 0.02,
        "ratio": 0.04,
        "pathway": 0.05,
    },
    "calibration": {
        "enabled": True,
        "max_candidates": 200,
        "topk_values": [5, 10, 20],
    },
    "weights": {
        "exact_anchor": {
            "direct_prior": 0.10,
            "disease_pathway_align": 0.25,
            "anchor_link": 0.40,
            "coverage_gain": 0.25,
        },
        "external_anchor": {
            "direct_prior": 0.10,
            "disease_pathway_align": 0.35,
            "anchor_link": 0.20,
            "coverage_gain": 0.35,
        },
        "disease_only": {
            "direct_prior": 0.05,
            "disease_pathway_align": 0.50,
            "anchor_link": 0.0,
            "coverage_gain": 0.45,
        },
    },
}


@dataclass(frozen=True)
class FeatureDescriptor:
    """Normalized descriptor for one candidate feature."""

    feature_name: str
    feature_type: FeatureType
    member_hmdb_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BioContext:
    """Unified context object required by f_bio v2 scoring."""

    disease_name: str = ""
    anchor_mode: AnchorMode = "disease_only"
    priors_dict: Dict[str, float] = field(default_factory=dict)
    protected_anchor_ids: List[str] = field(default_factory=list)
    external_prior_seed_ids: List[str] = field(default_factory=list)
    disease_core_pathways: List[str] = field(default_factory=list)
    taxonomy_map: Dict[str, Any] = field(default_factory=dict)
    pathway_map: Dict[str, Any] = field(default_factory=dict)
    pathway_map_reverse: Dict[str, Any] = field(default_factory=dict)
    pathway_map_path: str = ""
    pathway_map_reverse_path: str = ""
    reaction_graph: Dict[str, Any] = field(default_factory=dict)
    metabolite_context_map: Dict[str, Any] = field(default_factory=dict)
    metabolite_lookup: Dict[str, str] = field(default_factory=dict)
    feature_provenance_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    taxonomy_index: Dict[str, List[str]] = field(default_factory=dict)
    pathway_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hmdb_to_pathways: Dict[str, List[str]] = field(default_factory=dict)
    feature_descriptor_cache: Dict[str, FeatureDescriptor] = field(default_factory=dict)
    support_cache: Dict[Tuple[str, str, str], Dict[str, float]] = field(default_factory=dict)
    selected_context_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pathway_similarity_cache: Dict[Tuple[str, str], float] = field(default_factory=dict)
    disease_pathway_relevance_cache: Dict[str, float] = field(default_factory=dict)
    hmdb_disease_profile_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anchor_profile_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    anchor_union_profile: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


DISEASE_PATHWAY_STOPWORDS: Set[str] = {
    "and",
    "or",
    "of",
    "the",
    "in",
    "by",
    "to",
    "with",
    "for",
    "on",
    "from",
    "pathway",
    "pathways",
    "signaling",
    "metabolism",
    "metabolic",
    "disease",
    "deficiency",
    "cancer",
}


# Reuse large JSON resources within one Python process. These payloads are
# treated as read-only by the scoring code, so returning the cached object
# avoids repeated disk I/O and JSON parsing during Phase 2 reruns.
_JSON_OPTIONAL_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}


def _resolve_project_path(path_value: str) -> Path:
    """Resolve project-relative paths against the repository root."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[3] / candidate


def _load_json_optional(path_value: str) -> Dict[str, Any]:
    """Load a JSON object from disk, returning an empty dict on failure."""
    if not path_value:
        return {}
    resolved = _resolve_project_path(path_value)
    if not resolved.exists():
        return {}
    try:
        stat = resolved.stat()
    except OSError:
        return {}

    cache_key = str(resolved)
    cached = _JSON_OPTIONAL_CACHE.get(cache_key)
    if cached is not None:
        cached_mtime_ns, cached_size, cached_payload = cached
        if cached_mtime_ns == int(stat.st_mtime_ns) and cached_size == int(stat.st_size):
            return cached_payload
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        normalized = payload if isinstance(payload, dict) else {}
        _JSON_OPTIONAL_CACHE[cache_key] = (
            int(stat.st_mtime_ns),
            int(stat.st_size),
            normalized,
        )
        return normalized
    except Exception:
        return {}


def _clamp01(value: float) -> float:
    """Clamp a numeric score into [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _normalize_text(value: str) -> str:
    """Normalize free text for exact matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _tokenize_concepts(value: str) -> Set[str]:
    """Tokenize pathway text into lightweight concept tokens."""
    tokens = {
        token
        for token in _normalize_text(value).split()
        if token and token not in DISEASE_PATHWAY_STOPWORDS and len(token) > 2
    }
    return tokens


def _extract_hmdb_ids(value: str) -> List[str]:
    """Extract HMDB identifiers from a feature or biomarker name."""
    return re.findall(r"HMDB\d+", str(value or ""), flags=re.IGNORECASE)


def _standardize_feature_name(value: str) -> str:
    """Use the shared feature-name standardizer with a safe fallback."""
    try:
        return standardize_feature_name(value)
    except Exception:
        return str(value or "").strip().upper()


def _build_feature_pool_index(feature_pool: Optional[List[str]]) -> Dict[str, Any]:
    """Build normalized lookup sets for the current feature pool."""
    normalized_names = set()
    hmdb_ids = set()
    original_features = []
    for feature in feature_pool or []:
        text = str(feature or "").strip()
        if not text:
            continue
        original_features.append(text)
        normalized_names.add(_normalize_text(text))
        hmdb_ids.update(h.upper() for h in _extract_hmdb_ids(text))
    return {
        "features": original_features,
        "normalized_names": normalized_names,
        "hmdb_ids": hmdb_ids,
    }


def _match_biomarker_to_feature_pool(biomarker: Dict[str, Any], feature_index: Dict[str, Any]) -> bool:
    """Return whether a Phase 0 biomarker has an exact counterpart in the feature pool."""
    biomarker_id = str(biomarker.get("id", "") or "").upper()
    biomarker_name = _normalize_text(str(biomarker.get("name", "") or ""))
    if biomarker_id and biomarker_id in feature_index.get("hmdb_ids", set()):
        return True
    if biomarker_name and biomarker_name in feature_index.get("normalized_names", set()):
        return True
    return False


def _extract_biomarker_prior_value(biomarker: Dict[str, Any]) -> float:
    """Extract a usable prior value from one cached biomarker record."""
    for key in ("bio_prior_norm", "confidence_score", "bio_prior_raw"):
        value = biomarker.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _load_phase0_cache_entry(disease_name: str) -> Dict[str, Any]:
    """Load the most relevant disease entry from the Phase 0 cache."""
    cache_path = get_config().get_phase0_path("cache_path") or "storage/phase0_cache.json"
    cache_data = _load_json_optional(cache_path)
    diseases = cache_data.get("diseases", {})
    if not isinstance(diseases, dict) or not diseases:
        return {}

    normalized = _normalize_text(disease_name)
    key_map = {_normalize_text(str(key)): key for key in diseases.keys()}
    matched_key = key_map.get(normalized)
    if matched_key is None:
        matches = get_close_matches(normalized, list(key_map.keys()), n=1, cutoff=0.7)
        if matches:
            matched_key = key_map[matches[0]]
    if matched_key is None:
        return {}

    matched_entry = diseases.get(matched_key, {})
    return matched_entry if isinstance(matched_entry, dict) else {}


def _runtime_summary_has_pathway_content(runtime_summary: Dict[str, Any]) -> bool:
    """Return whether a runtime summary contains usable disease-pathway content."""
    if not isinstance(runtime_summary, dict) or not runtime_summary:
        return False
    for key in ("disease_core_pathways", "top_pathways", "external_disease_pathways_broad"):
        values = runtime_summary.get(key)
        if isinstance(values, list) and values:
            return True
    return False


def _find_runtime_pathway_pack(
    disease_name: str,
    phase0_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load runtime disease pathway pack for the current disease if available."""
    bound_phase0_output = dict(phase0_output or {})
    if bound_phase0_output:
        runtime_summary = bound_phase0_output.get("runtime_pathway_summary")
        if _runtime_summary_has_pathway_content(runtime_summary):
            return runtime_summary

        explicit_full_result_path = str(bound_phase0_output.get("full_result_path", "") or "").strip()
        explicit_evidence_dir = str(bound_phase0_output.get("evidence_dir", "") or "").strip()
        bound_candidates = [explicit_full_result_path]
        if explicit_evidence_dir:
            bound_candidates.append(os.path.join(explicit_evidence_dir, "phase0_full_result.json"))

        for candidate_path in bound_candidates:
            if not candidate_path:
                continue
            payload = _load_json_optional(candidate_path)
            nested_runtime_summary = payload.get("runtime_pathway_summary") if isinstance(payload, dict) else None
            if _runtime_summary_has_pathway_content(nested_runtime_summary):
                return nested_runtime_summary

    output_latest_path = get_config().get_phase0_path("output_latest") or "storage/phase0_output_latest.json"
    latest_output = _load_json_optional(output_latest_path)
    if _normalize_text(str(latest_output.get("disease_name", "") or "")) == _normalize_text(disease_name):
        runtime_summary = latest_output.get("runtime_pathway_summary")
        # Fall back to the persisted runtime cache when Phase 0 wrote only a
        # placeholder summary into `output_latest`.
        if _runtime_summary_has_pathway_content(runtime_summary):
            return runtime_summary

    retrieval_config = get_config().get_phase0_pathway_retrieval_config()
    runtime_cache_dir = retrieval_config.get(
        "runtime_cache_dir",
        "storage/external_pathways/runtime_disease_pathways",
    )
    resolved_dir = _resolve_project_path(str(runtime_cache_dir))
    if not resolved_dir.exists():
        return {}

    disease_slug = re.sub(r"[^a-z0-9]+", "_", _normalize_text(disease_name)).strip("_")
    candidates = sorted(resolved_dir.glob(f"{disease_slug}_*.json"), reverse=True)
    for candidate in candidates:
        try:
            payload = _load_json_optional(str(candidate))
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _load_feature_provenance_index() -> Dict[str, Dict[str, Any]]:
    """Load the latest Phase 1 feature provenance artifact as a lookup index."""
    phase1_artifacts_dir = get_config().get_phase1_path("artifacts_dir") or "output/phase1/artifacts"
    provenance_path = _resolve_project_path(str(phase1_artifacts_dir)) / "feature_provenance.json"
    payload = _load_json_optional(str(provenance_path))
    records = payload.get("features", [])
    if not isinstance(records, list):
        return {}

    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        feature_name = str(record.get("feature", "") or "").strip()
        standardized_name = str(record.get("standardized_name", "") or "").strip()
        if feature_name:
            index[feature_name] = record
        if standardized_name:
            index.setdefault(standardized_name, record)
    return index


def _build_taxonomy_member_index(taxonomy_map: Dict[str, Any]) -> Dict[str, List[str]]:
    """Create class/subclass -> HMDB member index from the taxonomy map."""
    taxonomy_index: Dict[str, List[str]] = {}
    for hmdb_id, payload in (taxonomy_map or {}).items():
        if not isinstance(payload, dict):
            continue
        normalized_hmdb = str(hmdb_id or "").upper().strip()
        if not normalized_hmdb:
            continue
        for field_name in ("class", "sub_class"):
            class_name = str(payload.get(field_name, "") or "").strip()
            if not class_name:
                continue
            standardized = _standardize_feature_name(class_name)
            if standardized:
                taxonomy_index.setdefault(standardized, [])
                if normalized_hmdb not in taxonomy_index[standardized]:
                    taxonomy_index[standardized].append(normalized_hmdb)
    return taxonomy_index


def _looks_like_hmdb_to_pathways_map(payload: Dict[str, Any]) -> bool:
    """Heuristically detect HMDB -> [pathway names] map orientation."""
    if not isinstance(payload, dict) or not payload:
        return False
    sample_key = next(iter(payload.keys()))
    sample_value = payload.get(sample_key)
    return bool(
        isinstance(sample_key, str)
        and sample_key.upper().startswith("HMDB")
        and isinstance(sample_value, list)
        and (not sample_value or isinstance(sample_value[0], str))
    )


def _looks_like_pathway_to_hmdb_map(payload: Dict[str, Any]) -> bool:
    """Heuristically detect pathway -> [HMDB IDs] map orientation."""
    if not isinstance(payload, dict) or not payload:
        return False
    sample_key = next(iter(payload.keys()))
    sample_value = payload.get(sample_key)
    if not isinstance(sample_key, str) or not isinstance(sample_value, list):
        return False
    if not sample_value:
        return True
    first_item = str(sample_value[0] or "").upper().strip()
    return first_item.startswith("HMDB")


def _normalize_pathway_maps(
    pathway_map: Dict[str, Any],
    pathway_map_reverse: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Normalize pathway resources into both directions:
    - hmdb_to_pathways
    - pathway_to_hmdbs
    """
    hmdb_to_pathways: Dict[str, List[str]] = {}
    pathway_to_hmdbs: Dict[str, List[str]] = {}

    if _looks_like_hmdb_to_pathways_map(pathway_map):
        for hmdb_id, pathways in pathway_map.items():
            normalized_hmdb = str(hmdb_id or "").upper().strip()
            if not normalized_hmdb or not isinstance(pathways, list):
                continue
            hmdb_to_pathways[normalized_hmdb] = []
            for pathway_name in pathways:
                normalized_pathway = str(pathway_name or "").strip()
                if not normalized_pathway:
                    continue
                if normalized_pathway not in hmdb_to_pathways[normalized_hmdb]:
                    hmdb_to_pathways[normalized_hmdb].append(normalized_pathway)
                pathway_to_hmdbs.setdefault(normalized_pathway, [])
                if normalized_hmdb not in pathway_to_hmdbs[normalized_pathway]:
                    pathway_to_hmdbs[normalized_pathway].append(normalized_hmdb)
    elif _looks_like_pathway_to_hmdb_map(pathway_map):
        for pathway_name, members in pathway_map.items():
            normalized_pathway = str(pathway_name or "").strip()
            if not normalized_pathway or not isinstance(members, list):
                continue
            pathway_to_hmdbs.setdefault(normalized_pathway, [])
            for member in members:
                normalized_hmdb = str(member or "").upper().strip()
                if not normalized_hmdb:
                    continue
                if normalized_hmdb not in pathway_to_hmdbs[normalized_pathway]:
                    pathway_to_hmdbs[normalized_pathway].append(normalized_hmdb)
                hmdb_to_pathways.setdefault(normalized_hmdb, [])
                if normalized_pathway not in hmdb_to_pathways[normalized_hmdb]:
                    hmdb_to_pathways[normalized_hmdb].append(normalized_pathway)

    if pathway_map_reverse and _looks_like_pathway_to_hmdb_map(pathway_map_reverse):
        for pathway_name, members in pathway_map_reverse.items():
            normalized_pathway = str(pathway_name or "").strip()
            if not normalized_pathway or not isinstance(members, list):
                continue
            pathway_to_hmdbs.setdefault(normalized_pathway, [])
            for member in members:
                normalized_hmdb = str(member or "").upper().strip()
                if not normalized_hmdb:
                    continue
                if normalized_hmdb not in pathway_to_hmdbs[normalized_pathway]:
                    pathway_to_hmdbs[normalized_pathway].append(normalized_hmdb)
                hmdb_to_pathways.setdefault(normalized_hmdb, [])
                if normalized_pathway not in hmdb_to_pathways[normalized_hmdb]:
                    hmdb_to_pathways[normalized_hmdb].append(normalized_pathway)

    return hmdb_to_pathways, pathway_to_hmdbs


def _ensure_pathway_resources(
    bio_context: BioContext,
    *,
    require_hmdb_to_pathways: bool = False,
    require_pathway_index: bool = False,
) -> None:
    """Lazily load and normalize large pathway resources only when needed."""
    if require_hmdb_to_pathways and bio_context.hmdb_to_pathways:
        if not require_pathway_index or bio_context.pathway_index:
            return
    if require_pathway_index and bio_context.pathway_index:
        if not require_hmdb_to_pathways or bio_context.hmdb_to_pathways:
            return

    pathway_to_hmdbs = bio_context.pathway_map_reverse or {}
    hmdb_to_pathways = bio_context.hmdb_to_pathways or {}

    if not pathway_to_hmdbs and bio_context.pathway_map_reverse_path:
        loaded_reverse = _load_json_optional(bio_context.pathway_map_reverse_path)
        if _looks_like_pathway_to_hmdb_map(loaded_reverse):
            pathway_to_hmdbs = loaded_reverse
            bio_context.pathway_map_reverse = pathway_to_hmdbs

    if not hmdb_to_pathways and bio_context.pathway_map and _looks_like_hmdb_to_pathways_map(bio_context.pathway_map):
        hmdb_to_pathways = {
            str(hmdb_id or "").upper().strip(): [
                str(pathway_name or "").strip()
                for pathway_name in pathways
                if str(pathway_name or "").strip()
            ]
            for hmdb_id, pathways in bio_context.pathway_map.items()
            if str(hmdb_id or "").upper().strip() and isinstance(pathways, list)
        }

    if not pathway_to_hmdbs and bio_context.pathway_map_path:
        loaded_forward = _load_json_optional(bio_context.pathway_map_path)
        if _looks_like_hmdb_to_pathways_map(loaded_forward):
            bio_context.pathway_map = loaded_forward
            hmdb_to_pathways, pathway_to_hmdbs = _normalize_pathway_maps(loaded_forward, {})
        elif _looks_like_pathway_to_hmdb_map(loaded_forward):
            pathway_to_hmdbs = loaded_forward
            bio_context.pathway_map_reverse = pathway_to_hmdbs

    if not hmdb_to_pathways and pathway_to_hmdbs:
        hmdb_to_pathways = {}
        for pathway_name, members in pathway_to_hmdbs.items():
            if not isinstance(members, list):
                continue
            normalized_pathway = str(pathway_name or "").strip()
            if not normalized_pathway:
                continue
            for member in members:
                normalized_hmdb = str(member or "").upper().strip()
                if not normalized_hmdb:
                    continue
                hmdb_to_pathways.setdefault(normalized_hmdb, [])
                if normalized_pathway not in hmdb_to_pathways[normalized_hmdb]:
                    hmdb_to_pathways[normalized_hmdb].append(normalized_pathway)

    if require_hmdb_to_pathways and hmdb_to_pathways:
        bio_context.hmdb_to_pathways = hmdb_to_pathways
    if require_pathway_index and pathway_to_hmdbs:
        bio_context.pathway_map_reverse = pathway_to_hmdbs
        if not bio_context.pathway_index:
            bio_context.pathway_index = _build_pathway_member_index(pathway_to_hmdbs)

    if pathway_to_hmdbs and "pathway_to_hmdbs_size" not in bio_context.metadata:
        bio_context.metadata["pathway_to_hmdbs_size"] = len(pathway_to_hmdbs)
    if hmdb_to_pathways and "hmdb_to_pathways_size" not in bio_context.metadata:
        bio_context.metadata["hmdb_to_pathways_size"] = len(hmdb_to_pathways)
    if bio_context.pathway_index and "pathway_index_size" not in bio_context.metadata:
        bio_context.metadata["pathway_index_size"] = len(bio_context.pathway_index)


def _build_pathway_member_index(pathway_to_hmdbs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create standardized pathway-name index with HMDB members."""
    pathway_index: Dict[str, Dict[str, Any]] = {}
    for pathway_name, members in (pathway_to_hmdbs or {}).items():
        if not isinstance(pathway_name, str) or not isinstance(members, list):
            continue
        standardized = _standardize_feature_name(pathway_name)
        normalized_members = [
            str(member or "").upper().strip()
            for member in members
            if str(member or "").upper().strip()
        ]
        if not standardized:
            continue
        pathway_index[standardized] = {
            "pathway_name": pathway_name,
            "member_hmdb_ids": list(dict.fromkeys(normalized_members)),
        }
    return pathway_index


def _get_hmdb_neighbors(hmdb_id: str, reaction_graph: Dict[str, Any]) -> Set[str]:
    """Return one-hop reaction neighbors for a metabolite."""
    neighbors: Set[str] = set()
    for edge in reaction_graph.get(hmdb_id, []) or []:
        if isinstance(edge, dict):
            target = str(edge.get("target", "") or "").upper().strip()
        else:
            target = str(edge or "").upper().strip()
        if target:
            neighbors.add(target)
    return neighbors


def _get_taxonomy_signature(hmdb_id: str, taxonomy_map: Dict[str, Any]) -> Tuple[str, str]:
    """Return (class, sub_class) taxonomy signature for a metabolite."""
    payload = taxonomy_map.get(hmdb_id, {}) if isinstance(taxonomy_map, dict) else {}
    if not isinstance(payload, dict):
        return ("", "")
    return (
        str(payload.get("class", "") or "").strip(),
        str(payload.get("sub_class", "") or "").strip(),
    )


def _build_anchor_profiles(
    anchor_ids: List[str],
    bio_context: BioContext,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Precompute reusable anchor-side graph/pathway/taxonomy profiles."""
    normalized_anchor_ids = [
        str(anchor_id or "").upper().strip()
        for anchor_id in anchor_ids
        if str(anchor_id or "").upper().strip()
    ]
    if not normalized_anchor_ids:
        return {}, {
            "anchor_ids": [],
            "all_neighbors": set(),
            "all_pathways": set(),
            "all_classes": set(),
            "all_subclasses": set(),
        }

    _ensure_pathway_resources(bio_context, require_hmdb_to_pathways=True)
    hmdb_to_pathways = bio_context.hmdb_to_pathways

    anchor_profile_index: Dict[str, Dict[str, Any]] = {}
    all_neighbors: Set[str] = set()
    all_pathways: Set[str] = set()
    all_classes: Set[str] = set()
    all_subclasses: Set[str] = set()

    for anchor_id in normalized_anchor_ids:
        neighbors = _get_hmdb_neighbors(anchor_id, bio_context.reaction_graph)
        pathways = set(hmdb_to_pathways.get(anchor_id, []) or [])
        anchor_class, anchor_sub_class = _get_taxonomy_signature(anchor_id, bio_context.taxonomy_map)
        class_norm = _normalize_text(anchor_class)
        sub_class_norm = _normalize_text(anchor_sub_class)
        profile = {
            "neighbors": neighbors,
            "pathways": pathways,
            "class": anchor_class,
            "sub_class": anchor_sub_class,
            "class_norm": class_norm,
            "sub_class_norm": sub_class_norm,
        }
        anchor_profile_index[anchor_id] = profile
        all_neighbors.update(neighbors)
        all_pathways.update(pathways)
        if class_norm:
            all_classes.add(class_norm)
        if sub_class_norm:
            all_subclasses.add(sub_class_norm)

    union_profile = {
        "anchor_ids": list(dict.fromkeys(normalized_anchor_ids)),
        "all_neighbors": all_neighbors,
        "all_pathways": all_pathways,
        "all_classes": all_classes,
        "all_subclasses": all_subclasses,
    }
    return anchor_profile_index, union_profile


def _compute_pathway_similarity(left: str, right: str) -> float:
    """Compute a conservative similarity between two pathway names."""
    if not left or not right:
        return 0.0
    if _normalize_text(left) == _normalize_text(right):
        return 1.0
    left_tokens = _tokenize_concepts(left)
    right_tokens = _tokenize_concepts(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(1.0, min(len(left_tokens), len(right_tokens)))


def _compute_pathway_similarity_cached(left: str, right: str, bio_context: Optional[BioContext] = None) -> float:
    """Memoized pathway similarity lookup."""
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    cache_key = (left_norm, right_norm) if left_norm <= right_norm else (right_norm, left_norm)
    if bio_context is not None and cache_key in bio_context.pathway_similarity_cache:
        return bio_context.pathway_similarity_cache[cache_key]
    score = _compute_pathway_similarity(left, right)
    if bio_context is not None:
        bio_context.pathway_similarity_cache[cache_key] = score
    return score


def _collect_member_pathways(
    hmdb_ids: List[str],
    bio_context: BioContext,
    hmdb_to_pathways: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Collect unique pathway names for a set of HMDB members."""
    if hmdb_to_pathways is None:
        _ensure_pathway_resources(bio_context, require_hmdb_to_pathways=True)
        hmdb_to_pathways = bio_context.hmdb_to_pathways
    collected: List[str] = []
    for hmdb_id in hmdb_ids:
        for pathway_name in hmdb_to_pathways.get(hmdb_id, []) or []:
            if pathway_name not in collected:
                collected.append(pathway_name)
    return collected


def _get_pathway_disease_relevance(pathway_name: str, bio_context: BioContext) -> float:
    """Return cached relevance between one pathway and the current disease core pathways."""
    normalized_pathway = str(pathway_name or "").strip()
    if not normalized_pathway or not bio_context.disease_core_pathways:
        return 0.0

    cache_key = _normalize_text(normalized_pathway)
    if cache_key in bio_context.disease_pathway_relevance_cache:
        return float(bio_context.disease_pathway_relevance_cache[cache_key])

    best_score = 0.0
    for disease_pathway in bio_context.disease_core_pathways:
        similarity = _compute_pathway_similarity_cached(normalized_pathway, disease_pathway, bio_context)
        if similarity >= 1.0:
            best_score = 1.0
            break
        best_score = max(best_score, similarity)

    bio_context.disease_pathway_relevance_cache[cache_key] = float(best_score)
    return float(best_score)


def _get_hmdb_disease_profile(hmdb_id: str, bio_context: BioContext) -> Dict[str, Any]:
    """Compute and cache one HMDB's disease-pathway profile for reuse across scoring terms."""
    normalized_hmdb = str(hmdb_id or "").upper().strip()
    if not normalized_hmdb:
        return {
            "member_pathways": [],
            "best_relevance": 0.0,
            "disease_relevant_pathways": [],
        }

    if normalized_hmdb in bio_context.hmdb_disease_profile_cache:
        return dict(bio_context.hmdb_disease_profile_cache[normalized_hmdb])

    _ensure_pathway_resources(bio_context, require_hmdb_to_pathways=True)
    hmdb_to_pathways = bio_context.hmdb_to_pathways
    member_pathways = list(hmdb_to_pathways.get(normalized_hmdb, []) or [])
    disease_relevant_pathways: List[Tuple[str, float]] = []
    best_relevance = 0.0

    for pathway_name in member_pathways:
        relevance = _get_pathway_disease_relevance(pathway_name, bio_context)
        best_relevance = max(best_relevance, relevance)
        if relevance >= 0.2:
            disease_relevant_pathways.append((pathway_name, relevance))

    profile = {
        "member_pathways": member_pathways,
        "best_relevance": float(best_relevance),
        "disease_relevant_pathways": disease_relevant_pathways,
    }
    bio_context.hmdb_disease_profile_cache[normalized_hmdb] = dict(profile)
    return dict(profile)


def _build_selected_context(
    selected_context: Optional[Dict[str, Any]],
    bio_context: BioContext,
    hmdb_to_pathways: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Normalize selected context for coverage calculations."""
    if hmdb_to_pathways is None:
        _ensure_pathway_resources(bio_context, require_hmdb_to_pathways=True)
        hmdb_to_pathways = bio_context.hmdb_to_pathways
    cache_key = "__default__"
    if selected_context:
        selected_ids = sorted(
            [
                str(x or "").upper().strip()
                for x in selected_context.get("selected_hmdb_ids", []) or []
                if str(x or "").upper().strip()
            ]
        )
        covered = sorted(
            [
                str(x or "").strip()
                for x in selected_context.get("covered_pathways", []) or []
                if str(x or "").strip()
            ]
        )
        cache_key = f"selected:{'|'.join(selected_ids)}::covered:{'|'.join(covered)}"
    if cache_key in bio_context.selected_context_cache:
        return dict(bio_context.selected_context_cache[cache_key])
    normalized = dict(selected_context or {})
    selected_hmdb_ids = [
        str(x or "").upper().strip()
        for x in normalized.get("selected_hmdb_ids", []) or []
        if str(x or "").upper().strip()
    ]
    if not selected_hmdb_ids:
        selected_hmdb_ids = list(dict.fromkeys(bio_context.protected_anchor_ids))

    covered_pathways = [
        str(x or "").strip()
        for x in normalized.get("covered_pathways", []) or []
        if str(x or "").strip()
    ]
    if not covered_pathways:
        covered_pathways = _collect_member_pathways(selected_hmdb_ids, bio_context, hmdb_to_pathways)

    result = {
        "selected_hmdb_ids": list(dict.fromkeys(selected_hmdb_ids)),
        "covered_pathways": covered_pathways,
    }
    bio_context.selected_context_cache[cache_key] = dict(result)
    return result


def _resolve_name_to_hmdb(feature_name: str, metabolite_lookup: Dict[str, str]) -> List[str]:
    """Resolve a raw metabolite name to HMDB via exact/standardized lookup."""
    if not metabolite_lookup:
        return []
    standardized = _standardize_feature_name(feature_name)
    normalized_for_lookup = str(feature_name or "").replace("_", " ").strip()
    standardized_alt = _standardize_feature_name(normalized_for_lookup)

    candidate_keys = [
        standardized,
        standardized_alt,
        f"L_{standardized_alt}",
        f"D_{standardized_alt}",
        f"DL_{standardized_alt}",
    ]
    for candidate_key in candidate_keys:
        if candidate_key in metabolite_lookup:
            return [str(metabolite_lookup[candidate_key]).upper()]

    matches = get_close_matches(standardized_alt, list(metabolite_lookup.keys()), n=1, cutoff=0.9)
    if matches:
        return [str(metabolite_lookup[matches[0]]).upper()]
    return []


def _resolve_sum_members(feature_name: str, taxonomy_index: Dict[str, List[str]]) -> List[str]:
    """Resolve SUM_/TAXSUM_ feature members from taxonomy labels."""
    standardized = _standardize_feature_name(feature_name)
    base_name = standardized
    for prefix in ("TAXSUM_", "SUM_"):
        if standardized.startswith(prefix):
            base_name = standardized[len(prefix):]
            break

    if base_name in taxonomy_index:
        return list(taxonomy_index[base_name])

    matches = get_close_matches(base_name, list(taxonomy_index.keys()), n=1, cutoff=0.88)
    if matches:
        return list(taxonomy_index[matches[0]])
    return []


def _extract_provenance_member_hmdb_ids(
    provenance_record: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Prefer explicit member IDs exported by Phase 1 provenance/registry."""
    if not isinstance(provenance_record, dict):
        return []

    candidate_lists = [
        provenance_record.get("mapped_hmdb_ids"),
        provenance_record.get("selected_member_hmdb_ids"),
        provenance_record.get("observed_member_hmdb_ids"),
        provenance_record.get("all_member_hmdb_ids"),
    ]
    resolved: List[str] = []
    seen = set()
    for values in candidate_lists:
        if not isinstance(values, list):
            continue
        for hmdb_id in values:
            normalized = str(hmdb_id or "").upper().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                resolved.append(normalized)
    return resolved


def _resolve_pathway_members(
    feature_name: str,
    pathway_index: Dict[str, Dict[str, Any]],
    provenance_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve pathway feature metadata and members from the pathway map."""
    provenance_members = _extract_provenance_member_hmdb_ids(provenance_record)
    if provenance_members:
        mapped_pathways = provenance_record.get("mapped_pathways", []) if isinstance(provenance_record, dict) else []
        pathway_name = ""
        if isinstance(mapped_pathways, list) and mapped_pathways:
            pathway_name = str(mapped_pathways[0] or "").strip()
        return {
            "pathway_name": pathway_name,
            "member_hmdb_ids": provenance_members,
        }

    standardized = _standardize_feature_name(feature_name)
    candidate_keys: List[str] = []
    if standardized.startswith("PATHWAY_"):
        candidate_keys.append(standardized[len("PATHWAY_"):])
    candidate_keys.append(standardized)

    if provenance_record:
        for pathway_name in provenance_record.get("mapped_pathways", []) or []:
            standardized_name = _standardize_feature_name(str(pathway_name))
            if standardized_name:
                candidate_keys.append(standardized_name)

    for key in candidate_keys:
        if key in pathway_index:
            return dict(pathway_index[key])

    for key in candidate_keys:
        matches = get_close_matches(key, list(pathway_index.keys()), n=1, cutoff=0.9)
        if matches:
            return dict(pathway_index[matches[0]])
    return {"pathway_name": "", "member_hmdb_ids": []}


def get_default_f_bio_v2_config() -> Dict[str, Any]:
    """Return a copy of the default f_bio v2 configuration."""
    return {
        **DEFAULT_F_BIO_V2_CONFIG,
        "aggregation": dict(DEFAULT_F_BIO_V2_CONFIG["aggregation"]),
        "structure_bonus": dict(DEFAULT_F_BIO_V2_CONFIG["structure_bonus"]),
        "calibration": dict(DEFAULT_F_BIO_V2_CONFIG["calibration"]),
        "weights": {
            mode: dict(values)
            for mode, values in DEFAULT_F_BIO_V2_CONFIG["weights"].items()
        },
    }


def merge_f_bio_v2_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user config onto the f_bio v2 defaults."""
    merged = get_default_f_bio_v2_config()
    if not isinstance(config, dict):
        return merged

    for key, value in config.items():
        if key in {"aggregation", "structure_bonus", "calibration"} and isinstance(value, dict):
            merged[key].update(value)
        elif key == "weights" and isinstance(value, dict):
            for mode, mode_values in value.items():
                if mode in merged["weights"] and isinstance(mode_values, dict):
                    merged["weights"][mode].update(mode_values)
        else:
            merged[key] = value
    return merged


def build_empty_bio_context(
    disease_name: str = "",
    anchor_mode: AnchorMode = "disease_only",
    config: Optional[Dict[str, Any]] = None,
) -> BioContext:
    """Create an empty BioContext with normalized config."""
    return BioContext(
        disease_name=disease_name,
        anchor_mode=anchor_mode,
        config=merge_f_bio_v2_config(config),
    )


def detect_anchor_mode(
    protected_anchor_ids: Optional[List[str]] = None,
    external_prior_seed_ids: Optional[List[str]] = None,
) -> AnchorMode:
    """Infer anchor mode from exact anchors and external seeds."""
    if protected_anchor_ids:
        return "exact_anchor"
    if external_prior_seed_ids:
        return "external_anchor"
    return "disease_only"


def build_bio_context(
    disease_name: Optional[str] = None,
    feature_pool: Optional[List[str]] = None,
    priors_dict: Optional[Dict[str, float]] = None,
    taxonomy_map: Optional[Dict[str, Any]] = None,
    pathway_map: Optional[Dict[str, Any]] = None,
    reaction_graph: Optional[Dict[str, Any]] = None,
    metabolite_context_map: Optional[Dict[str, Any]] = None,
    phase0_output: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> BioContext:
    """Build a concrete BioContext from Phase 0 artifacts and the current feature pool."""
    bound_phase0_output = dict(phase0_output or {})
    disease_name = str(
        disease_name
        or bound_phase0_output.get("disease_name")
        or get_config().get_test_disease_name()
        or ""
    ).strip()
    merged_config = merge_f_bio_v2_config(config or get_config().get_phase2_f_bio_v2_config())
    phase0_paths = get_config().get_phase0_paths()

    phase0_entry = _load_phase0_cache_entry(disease_name)
    confirmed_biomarkers = bound_phase0_output.get("confirmed_biomarkers", [])
    if not isinstance(confirmed_biomarkers, list) or not confirmed_biomarkers:
        confirmed_biomarkers = phase0_entry.get("confirmed_biomarkers", [])
    confirmed_biomarkers = confirmed_biomarkers if isinstance(confirmed_biomarkers, list) else []

    runtime_pack = _find_runtime_pathway_pack(disease_name, phase0_output=bound_phase0_output)
    disease_core_pathways = runtime_pack.get("disease_core_pathways", []) or runtime_pack.get("top_pathways", [])
    disease_core_pathways = [str(x) for x in disease_core_pathways if x]

    feature_index = _build_feature_pool_index(feature_pool)
    protected_anchor_ids: List[str] = []
    external_prior_seed_ids: List[str] = []
    derived_priors = dict(priors_dict or {})
    biomarker_records: List[Dict[str, Any]] = []

    for biomarker in confirmed_biomarkers:
        if not isinstance(biomarker, dict):
            continue
        biomarker_id = str(biomarker.get("id", "") or "").upper()
        biomarker_name = str(biomarker.get("name", "") or "")
        if not biomarker_id and not biomarker_name:
            continue

        biomarker_records.append(
            {
                "id": biomarker_id,
                "name": biomarker_name,
                "matched_in_feature_pool": _match_biomarker_to_feature_pool(biomarker, feature_index),
                "prior_value": _extract_biomarker_prior_value(biomarker),
            }
        )
        if biomarker_id:
            derived_priors.setdefault(biomarker_id, _extract_biomarker_prior_value(biomarker))
            if _match_biomarker_to_feature_pool(biomarker, feature_index):
                protected_anchor_ids.append(biomarker_id)
            else:
                external_prior_seed_ids.append(biomarker_id)

    protected_anchor_ids = sorted(dict.fromkeys([x for x in protected_anchor_ids if x]))
    external_prior_seed_ids = sorted(
        dict.fromkeys([x for x in external_prior_seed_ids if x and x not in protected_anchor_ids])
    )
    anchor_mode = detect_anchor_mode(protected_anchor_ids, external_prior_seed_ids)

    if taxonomy_map is None:
        taxonomy_map = _load_json_optional(phase0_paths.get("taxonomy_map", ""))
    pathway_map_path = str(phase0_paths.get("pathway_map", "") or "")
    pathway_map_reverse_path = str(phase0_paths.get("pathway_map_reverse", "") or "")
    pathway_map_reverse: Dict[str, Any] = {}
    if pathway_map is not None:
        pathway_map = pathway_map or {}
    else:
        pathway_map = {}
    if reaction_graph is None:
        reaction_graph = _load_json_optional(phase0_paths.get("reaction_graph", ""))
    if metabolite_context_map is None:
        metabolite_context_map = _load_json_optional(phase0_paths.get("metabolite_context_map", ""))
    metabolite_lookup = _load_json_optional(phase0_paths.get("metabolite_lookup", ""))
    feature_provenance_index = _load_feature_provenance_index()
    taxonomy_index = _build_taxonomy_member_index(taxonomy_map or {})
    anchor_context = BioContext(
        disease_name=disease_name,
        anchor_mode=anchor_mode,
        taxonomy_map=taxonomy_map or {},
        pathway_map=pathway_map or {},
        pathway_map_reverse=pathway_map_reverse,
        pathway_map_path=pathway_map_path,
        pathway_map_reverse_path=pathway_map_reverse_path,
        reaction_graph=reaction_graph or {},
        config=merged_config,
        metadata={},
    )
    anchor_profile_index, anchor_union_profile = _build_anchor_profiles(
        protected_anchor_ids + external_prior_seed_ids,
        anchor_context,
    )

    return BioContext(
        disease_name=disease_name,
        anchor_mode=anchor_mode,
        priors_dict=derived_priors,
        protected_anchor_ids=protected_anchor_ids,
        external_prior_seed_ids=external_prior_seed_ids,
        disease_core_pathways=disease_core_pathways,
        taxonomy_map=taxonomy_map or {},
        pathway_map=pathway_map or {},
        pathway_map_reverse=pathway_map_reverse,
        pathway_map_path=pathway_map_path,
        pathway_map_reverse_path=pathway_map_reverse_path,
        reaction_graph=reaction_graph or {},
        metabolite_context_map=metabolite_context_map or {},
        metabolite_lookup=metabolite_lookup or {},
        feature_provenance_index=feature_provenance_index,
        taxonomy_index=taxonomy_index,
        anchor_profile_index=anchor_profile_index,
        anchor_union_profile=anchor_union_profile,
        config=merged_config,
        metadata={
            "feature_pool_size": len(feature_index.get("features", [])),
            "phase0_confirmed_biomarker_count": len(biomarker_records),
            "protected_anchor_count": len(protected_anchor_ids),
            "external_prior_seed_count": len(external_prior_seed_ids),
            "runtime_pathway_source": runtime_pack.get("cache_key") or runtime_pack.get("disease") or None,
            "phase0_cache_hit": bool(phase0_entry),
            "feature_provenance_loaded": bool(feature_provenance_index),
            "taxonomy_index_size": len(taxonomy_index),
            "pathway_index_size": 0,
            "hmdb_to_pathways_size": 0,
            "pathway_to_hmdbs_size": 0,
            "pathway_resources_lazy": True,
            "anchor_profile_count": len(anchor_profile_index),
            "anchor_union_neighbor_count": len(anchor_union_profile.get("all_neighbors", set()) or set()),
            "anchor_union_pathway_count": len(anchor_union_profile.get("all_pathways", set()) or set()),
            "biomarker_records": biomarker_records,
        },
    )


def resolve_feature_descriptor(
    feature_name: str,
    bio_context: Optional[BioContext] = None,
    taxonomy_map: Optional[Dict[str, Any]] = None,
    pathway_map: Optional[Dict[str, Any]] = None,
    metabolite_lookup: Optional[Dict[str, str]] = None,
    feature_provenance_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> FeatureDescriptor:
    """Resolve one feature into a normalized descriptor with member HMDB IDs."""
    raw_name = str(feature_name or "").strip()
    standardized_name = _standardize_feature_name(raw_name)
    if bio_context is not None:
        cached = bio_context.feature_descriptor_cache.get(raw_name) or bio_context.feature_descriptor_cache.get(standardized_name)
        if cached is not None:
            return cached

    taxonomy_map = taxonomy_map if taxonomy_map is not None else (bio_context.taxonomy_map if bio_context else {})
    pathway_map = pathway_map if pathway_map is not None else (bio_context.pathway_map if bio_context else {})
    metabolite_lookup = (
        metabolite_lookup
        if metabolite_lookup is not None
        else (bio_context.metabolite_lookup if bio_context else {})
    )
    feature_provenance_index = (
        feature_provenance_index
        if feature_provenance_index is not None
        else (bio_context.feature_provenance_index if bio_context else {})
    )

    provenance_record = (
        feature_provenance_index.get(raw_name)
        or feature_provenance_index.get(standardized_name)
        or None
    )

    taxonomy_index = (
        bio_context.taxonomy_index if bio_context and bio_context.taxonomy_index else _build_taxonomy_member_index(taxonomy_map)
    )
    if bio_context is not None:
        _ensure_pathway_resources(bio_context, require_pathway_index=True)
        pathway_index = bio_context.pathway_index
    else:
        pathway_index = _build_pathway_member_index(pathway_map)

    metadata: Dict[str, Any] = {
        "standardized_name": standardized_name,
        "resolver_source": [],
    }

    hmdb_ids = [hmdb.upper() for hmdb in _extract_hmdb_ids(raw_name)]
    if hmdb_ids:
        feature_type: FeatureType = "ratio" if standardized_name.startswith("RATIO_") else "single"
        metadata["resolver_source"].append("hmdb_pattern")
        if feature_type == "single":
            metadata["display_name"] = provenance_record.get("display_name") if provenance_record else raw_name
        descriptor = FeatureDescriptor(
            feature_name=raw_name,
            feature_type=feature_type,
            member_hmdb_ids=list(dict.fromkeys(hmdb_ids)),
            metadata=metadata,
        )
        if bio_context is not None:
            bio_context.feature_descriptor_cache[raw_name] = descriptor
            bio_context.feature_descriptor_cache[standardized_name] = descriptor
        return descriptor

    if provenance_record:
        metadata["provenance_record"] = {
            "origin_type": provenance_record.get("origin_type"),
            "origin_subtype": provenance_record.get("origin_subtype"),
            "mapped_hmdb_ids": list(provenance_record.get("mapped_hmdb_ids", []) or []),
            "mapped_pathways": list(provenance_record.get("mapped_pathways", []) or []),
        }
        mapped_hmdb_ids = _extract_provenance_member_hmdb_ids(provenance_record)
        origin_type = str(provenance_record.get("origin_type", "") or "")
        origin_subtype = str(provenance_record.get("origin_subtype", "") or "")
        if mapped_hmdb_ids and origin_type == "raw":
            metadata["resolver_source"].append("feature_provenance")
            descriptor = FeatureDescriptor(
                feature_name=raw_name,
                feature_type="single",
                member_hmdb_ids=list(dict.fromkeys(mapped_hmdb_ids)),
                metadata=metadata,
            )
            if bio_context is not None:
                bio_context.feature_descriptor_cache[raw_name] = descriptor
                bio_context.feature_descriptor_cache[standardized_name] = descriptor
            return descriptor
        if origin_type == "engineered_sum":
            members = mapped_hmdb_ids or _resolve_sum_members(raw_name, taxonomy_index)
            metadata["resolver_source"].append("feature_provenance")
            metadata["group_label"] = (
                provenance_record.get("group_label")
                if provenance_record.get("group_label")
                else (standardized_name.split("_", 1)[1] if "_" in standardized_name else standardized_name)
            )
            descriptor = FeatureDescriptor(
                feature_name=raw_name,
                feature_type="taxsum" if origin_subtype == "taxonomy_sum" else "sum",
                member_hmdb_ids=members,
                metadata=metadata,
            )
            if bio_context is not None:
                bio_context.feature_descriptor_cache[raw_name] = descriptor
                bio_context.feature_descriptor_cache[standardized_name] = descriptor
            return descriptor
        if origin_type == "engineered_pathway":
            pathway_payload = _resolve_pathway_members(raw_name, pathway_index, provenance_record)
            metadata["resolver_source"].append("feature_provenance")
            metadata["pathway_name"] = pathway_payload.get("pathway_name", "")
            descriptor = FeatureDescriptor(
                feature_name=raw_name,
                feature_type="pathway",
                member_hmdb_ids=list(pathway_payload.get("member_hmdb_ids", []) or []),
                metadata=metadata,
            )
            if bio_context is not None:
                bio_context.feature_descriptor_cache[raw_name] = descriptor
                bio_context.feature_descriptor_cache[standardized_name] = descriptor
            return descriptor

    if standardized_name.startswith("RATIO_"):
        descriptor = FeatureDescriptor(
            feature_name=raw_name,
            feature_type="ratio",
            member_hmdb_ids=[],
            metadata={**metadata, "resolver_source": metadata["resolver_source"] + ["ratio_prefix"]},
        )
        if bio_context is not None:
            bio_context.feature_descriptor_cache[raw_name] = descriptor
            bio_context.feature_descriptor_cache[standardized_name] = descriptor
        return descriptor

    if standardized_name.startswith("TAXSUM_") or standardized_name.startswith("SUM_"):
        members = _resolve_sum_members(raw_name, taxonomy_index)
        descriptor = FeatureDescriptor(
            feature_name=raw_name,
            feature_type="taxsum" if standardized_name.startswith("TAXSUM_") else "sum",
            member_hmdb_ids=members,
            metadata={**metadata, "resolver_source": metadata["resolver_source"] + ["taxonomy_index"]},
        )
        if bio_context is not None:
            bio_context.feature_descriptor_cache[raw_name] = descriptor
            bio_context.feature_descriptor_cache[standardized_name] = descriptor
        return descriptor

    if standardized_name.startswith("PATHWAY_"):
        pathway_payload = _resolve_pathway_members(raw_name, pathway_index, provenance_record)
        descriptor = FeatureDescriptor(
            feature_name=raw_name,
            feature_type="pathway",
            member_hmdb_ids=list(pathway_payload.get("member_hmdb_ids", []) or []),
            metadata={
                **metadata,
                "resolver_source": metadata["resolver_source"] + ["pathway_index"],
                "pathway_name": pathway_payload.get("pathway_name", ""),
            },
        )
        if bio_context is not None:
            bio_context.feature_descriptor_cache[raw_name] = descriptor
            bio_context.feature_descriptor_cache[standardized_name] = descriptor
        return descriptor

    resolved_hmdb = _resolve_name_to_hmdb(raw_name, metabolite_lookup or {})
    if resolved_hmdb:
        descriptor = FeatureDescriptor(
            feature_name=raw_name,
            feature_type="single",
            member_hmdb_ids=resolved_hmdb,
            metadata={**metadata, "resolver_source": metadata["resolver_source"] + ["metabolite_lookup"]},
        )
        if bio_context is not None:
            bio_context.feature_descriptor_cache[raw_name] = descriptor
            bio_context.feature_descriptor_cache[standardized_name] = descriptor
        return descriptor

    pathway_payload = _resolve_pathway_members(raw_name, pathway_index, provenance_record)
    if pathway_payload.get("member_hmdb_ids"):
        descriptor = FeatureDescriptor(
            feature_name=raw_name,
            feature_type="pathway",
            member_hmdb_ids=list(pathway_payload.get("member_hmdb_ids", []) or []),
            metadata={
                **metadata,
                "resolver_source": metadata["resolver_source"] + ["pathway_fallback"],
                "pathway_name": pathway_payload.get("pathway_name", ""),
            },
        )
        if bio_context is not None:
            bio_context.feature_descriptor_cache[raw_name] = descriptor
            bio_context.feature_descriptor_cache[standardized_name] = descriptor
        return descriptor

    descriptor = FeatureDescriptor(
        feature_name=raw_name,
        feature_type="unknown",
        member_hmdb_ids=[],
        metadata={**metadata, "resolver_source": metadata["resolver_source"] + ["unresolved"]},
    )
    if bio_context is not None:
        bio_context.feature_descriptor_cache[raw_name] = descriptor
        bio_context.feature_descriptor_cache[standardized_name] = descriptor
    return descriptor


def compute_direct_prior(hmdb_id: str, bio_context: BioContext) -> float:
    """Return direct prior support from Phase 0 priors."""
    return _clamp01(float(bio_context.priors_dict.get(str(hmdb_id or "").upper(), 0.0) or 0.0))


def compute_disease_pathway_align(hmdb_id: str, bio_context: BioContext) -> float:
    """Score alignment between a metabolite's pathways and disease core pathways."""
    hmdb_id = str(hmdb_id or "").upper().strip()
    if not hmdb_id or not bio_context.disease_core_pathways:
        return 0.0

    profile = _get_hmdb_disease_profile(hmdb_id, bio_context)
    best_score = float(profile.get("best_relevance", 0.0) or 0.0)
    if best_score <= 0.0:
        return 0.0

    # Pathway bridge rewards partial disease-mechanism overlap without requiring exact names.
    if best_score >= 0.6:
        return 0.85
    if best_score >= 0.4:
        return 0.6
    if best_score >= 0.2:
        return 0.35
    return 0.0


def compute_anchor_link(hmdb_id: str, bio_context: BioContext) -> float:
    """Score how strongly a metabolite connects to exact or external anchor seeds."""
    hmdb_id = str(hmdb_id or "").upper().strip()
    if not hmdb_id:
        return 0.0
    if bio_context.anchor_mode == "disease_only":
        return 0.0

    anchor_ids = list(dict.fromkeys(bio_context.protected_anchor_ids + bio_context.external_prior_seed_ids))
    if not anchor_ids:
        return 0.0
    if hmdb_id in anchor_ids:
        return 1.0

    _ensure_pathway_resources(bio_context, require_hmdb_to_pathways=True)
    hmdb_to_pathways = bio_context.hmdb_to_pathways
    target_pathways = set(hmdb_to_pathways.get(hmdb_id, []) or [])
    target_neighbors = _get_hmdb_neighbors(hmdb_id, bio_context.reaction_graph)
    target_class, target_sub_class = _get_taxonomy_signature(hmdb_id, bio_context.taxonomy_map)
    target_class_norm = _normalize_text(target_class)
    target_sub_class_norm = _normalize_text(target_sub_class)
    anchor_profile_index = bio_context.anchor_profile_index or {}

    best_score = 0.0
    for anchor_id in anchor_ids:
        anchor_id = str(anchor_id or "").upper().strip()
        if not anchor_id:
            continue
        anchor_profile = anchor_profile_index.get(anchor_id, {})

        score = 0.0
        if anchor_id in target_neighbors:
            score = max(score, 1.0)
        else:
            anchor_neighbors = set(anchor_profile.get("neighbors", set()) or set())
            if anchor_neighbors & target_neighbors:
                score = max(score, 0.7)

        anchor_pathways = set(anchor_profile.get("pathways", set()) or set())
        if target_pathways and anchor_pathways:
            shared = len(target_pathways & anchor_pathways)
            if shared > 0:
                score = max(score, min(0.85, 0.45 + 0.15 * shared))

        anchor_class_norm = str(anchor_profile.get("class_norm", "") or "")
        anchor_sub_class_norm = str(anchor_profile.get("sub_class_norm", "") or "")
        if target_sub_class_norm and anchor_sub_class_norm and target_sub_class_norm == anchor_sub_class_norm:
            score = max(score, 0.4)
        elif target_class_norm and anchor_class_norm and target_class_norm == anchor_class_norm:
            score = max(score, 0.25)

        best_score = max(best_score, score)

    return _clamp01(best_score)


def compute_coverage_gain(
    hmdb_id: str,
    bio_context: BioContext,
    selected_context: Optional[Dict[str, Any]] = None,
) -> float:
    """Reward metabolites that add disease-relevant pathway coverage not already covered."""
    hmdb_id = str(hmdb_id or "").upper().strip()
    if not hmdb_id:
        return 0.0

    profile = _get_hmdb_disease_profile(hmdb_id, bio_context)
    member_pathways = list(profile.get("member_pathways", []) or [])
    if not member_pathways:
        return 0.0

    _ensure_pathway_resources(bio_context, require_hmdb_to_pathways=True)
    hmdb_to_pathways = bio_context.hmdb_to_pathways
    normalized_selected = _build_selected_context(selected_context, bio_context, hmdb_to_pathways)
    covered_pathways = set(normalized_selected.get("covered_pathways", []) or [])
    disease_relevant_pathways = list(profile.get("disease_relevant_pathways", []) or [])

    if not disease_relevant_pathways:
        return 0.0

    novel_scores = [
        relevance
        for pathway_name, relevance in disease_relevant_pathways
        if pathway_name not in covered_pathways
    ]
    if not novel_scores:
        return 0.05

    novelty_ratio = len(novel_scores) / max(1.0, len(disease_relevant_pathways))
    novelty_strength = max(novel_scores)
    return _clamp01(0.5 * novelty_ratio + 0.5 * novelty_strength)


def compute_feature_support(
    hmdb_id: str,
    bio_context: BioContext,
    selected_context: Optional[Dict[str, Any]] = None,
    mode: Optional[AnchorMode] = None,
) -> Dict[str, float]:
    """Compute the four-part support score for one HMDB metabolite."""
    effective_mode = mode or bio_context.anchor_mode
    selected_context_key = "__default__"
    if selected_context:
        normalized_selected = _build_selected_context(selected_context, bio_context, bio_context.hmdb_to_pathways)
        selected_context_key = (
            f"selected:{'|'.join(sorted(normalized_selected.get('selected_hmdb_ids', [])))}"
            f"::covered:{'|'.join(sorted(normalized_selected.get('covered_pathways', [])))}"
        )
    cache_key = (str(hmdb_id or "").upper().strip(), str(effective_mode), selected_context_key)
    if cache_key in bio_context.support_cache:
        return dict(bio_context.support_cache[cache_key])
    weights = dict(
        (bio_context.config.get("weights", {}) or {}).get(effective_mode, {})
    )
    direct_prior = compute_direct_prior(hmdb_id, bio_context)
    disease_pathway_align = compute_disease_pathway_align(hmdb_id, bio_context)
    anchor_link = compute_anchor_link(hmdb_id, bio_context)
    coverage_gain = compute_coverage_gain(hmdb_id, bio_context, selected_context)

    components = {
        "direct_prior": direct_prior,
        "disease_pathway_align": disease_pathway_align,
        "anchor_link": anchor_link,
        "coverage_gain": coverage_gain,
    }
    support_score = sum(float(weights.get(key, 0.0) or 0.0) * value for key, value in components.items())
    result = {
        **components,
        "support_score": _clamp01(support_score),
    }
    bio_context.support_cache[cache_key] = dict(result)
    return result


def _topk_mean(values: List[float], topk_ratio: float, topk_min_k: int) -> float:
    """Aggregate values by top-k mean with conservative defaults."""
    if not values:
        return 0.0
    cleaned = sorted([float(v) for v in values], reverse=True)
    k = max(int(topk_min_k), int(round(len(cleaned) * float(topk_ratio))))
    k = max(1, min(len(cleaned), k))
    return sum(cleaned[:k]) / k


def aggregate_member_support(
    feature_descriptor: FeatureDescriptor,
    bio_context: BioContext,
    selected_context: Optional[Dict[str, Any]] = None,
    mode: Optional[AnchorMode] = None,
) -> Dict[str, Any]:
    """Aggregate HMDB-level support into one feature-level support score."""
    member_ids = [
        str(member or "").upper().strip()
        for member in feature_descriptor.member_hmdb_ids
        if str(member or "").upper().strip()
    ]
    if not member_ids:
        return {
            "feature_type": feature_descriptor.feature_type,
            "member_supports": {},
            "aggregate_member_support": 0.0,
            "aggregation_strategy": "empty",
        }

    member_supports = {
        hmdb_id: compute_feature_support(
            hmdb_id=hmdb_id,
            bio_context=bio_context,
            selected_context=selected_context,
            mode=mode,
        )
        for hmdb_id in member_ids
    }
    support_values = [payload["support_score"] for payload in member_supports.values()]
    aggregation_cfg = bio_context.config.get("aggregation", {}) or {}
    topk_ratio = float(bio_context.config.get("topk_ratio", 0.3) or 0.3)
    topk_min_k = int(bio_context.config.get("topk_min_k", 2) or 2)

    feature_type = feature_descriptor.feature_type
    if feature_type == "single":
        aggregate_score = float(support_values[0])
        strategy = aggregation_cfg.get("single", "direct")
    elif feature_type in {"sum", "taxsum", "pathway"}:
        aggregate_score = _topk_mean(support_values, topk_ratio=topk_ratio, topk_min_k=topk_min_k)
        strategy = aggregation_cfg.get("pathway" if feature_type == "pathway" else "sum", "topk_mean")
    elif feature_type == "ratio":
        ordered = sorted(support_values)
        if len(ordered) >= 2:
            aggregate_score = 0.7 * ordered[0] + 0.3 * ordered[-1]
        else:
            aggregate_score = float(ordered[0])
        strategy = aggregation_cfg.get("ratio", "minmax_blend")
    else:
        aggregate_score = _topk_mean(support_values, topk_ratio=topk_ratio, topk_min_k=topk_min_k)
        strategy = "fallback_topk_mean"

    return {
        "feature_type": feature_type,
        "member_supports": member_supports,
        "aggregate_member_support": _clamp01(aggregate_score),
        "aggregation_strategy": strategy,
        "member_count": len(member_ids),
    }


def compute_structure_bonus(
    feature_descriptor: FeatureDescriptor,
    bio_context: BioContext,
) -> float:
    """Return a small structural bonus by engineered-feature type."""
    structure_cfg = bio_context.config.get("structure_bonus", {}) or {}
    feature_type = feature_descriptor.feature_type
    if feature_type == "taxsum":
        feature_type = "sum"
    return _clamp01(float(structure_cfg.get(feature_type, 0.0) or 0.0))


def _extract_feature_hmdb_ids(
    feature_name: str,
    bio_context: BioContext,
    feature_descriptor: Optional[FeatureDescriptor] = None,
) -> List[str]:
    """Resolve one feature into its member HMDB IDs."""
    descriptor = feature_descriptor or resolve_feature_descriptor(feature_name, bio_context=bio_context)
    return [
        str(member or "").upper().strip()
        for member in descriptor.member_hmdb_ids
        if str(member or "").upper().strip()
    ]


def compute_panel_connectivity_gain(
    search_features: List[str],
    bio_context: BioContext,
    anchor_features: Optional[List[str]] = None,
    descriptor_map: Optional[Dict[str, FeatureDescriptor]] = None,
) -> Dict[str, Any]:
    """Estimate how much the searched features connect with themselves and anchor context."""
    descriptor_map = descriptor_map or {}
    search_hmdb_ids: Set[str] = set()
    anchor_hmdb_ids: Set[str] = set()

    for feature_name in search_features or []:
        descriptor = descriptor_map.get(feature_name)
        search_hmdb_ids.update(_extract_feature_hmdb_ids(feature_name, bio_context, descriptor))

    for feature_name in anchor_features or []:
        descriptor = descriptor_map.get(feature_name)
        anchor_hmdb_ids.update(_extract_feature_hmdb_ids(feature_name, bio_context, descriptor))

    if not search_hmdb_ids:
        return {
            "search_hmdb_ids": [],
            "anchor_hmdb_ids": sorted(anchor_hmdb_ids),
            "search_internal_edge_count": 0,
            "anchor_bridge_edge_count": 0,
            "connectivity_gain": 0.0,
        }

    search_internal_edges = 0
    seen_internal_pairs: Set[Tuple[str, str]] = set()
    for hmdb_id in sorted(search_hmdb_ids):
        neighbors = _get_hmdb_neighbors(hmdb_id, bio_context.reaction_graph)
        for neighbor in neighbors:
            if neighbor in search_hmdb_ids and neighbor != hmdb_id:
                pair = tuple(sorted((hmdb_id, neighbor)))
                if pair not in seen_internal_pairs:
                    seen_internal_pairs.add(pair)
                    search_internal_edges += 1

    anchor_bridge_edges = 0
    seen_bridge_pairs: Set[Tuple[str, str]] = set()
    for hmdb_id in sorted(search_hmdb_ids):
        neighbors = _get_hmdb_neighbors(hmdb_id, bio_context.reaction_graph)
        for neighbor in neighbors:
            if neighbor in anchor_hmdb_ids and neighbor != hmdb_id:
                pair = tuple(sorted((hmdb_id, neighbor)))
                if pair not in seen_bridge_pairs:
                    seen_bridge_pairs.add(pair)
                    anchor_bridge_edges += 1

    denominator = max(1.0, float(len(search_hmdb_ids)))
    connectivity_gain = _clamp01((search_internal_edges + anchor_bridge_edges) / denominator)
    return {
        "search_hmdb_ids": sorted(search_hmdb_ids),
        "anchor_hmdb_ids": sorted(anchor_hmdb_ids),
        "search_internal_edge_count": search_internal_edges,
        "anchor_bridge_edge_count": anchor_bridge_edges,
        "connectivity_gain": connectivity_gain,
    }


def _calculate_single_p0_v2(
    feature_name: str,
    bio_context: BioContext,
    selected_context: Optional[Dict[str, Any]] = None,
    feature_descriptor: Optional[FeatureDescriptor] = None,
    mode: Optional[AnchorMode] = None,
) -> Dict[str, Any]:
    """
    Calculate one feature's v2 biological prior contribution.

    Returns a structured payload rather than a bare float so later phases can
    inspect member-level support decomposition.
    """
    effective_mode = mode or bio_context.anchor_mode
    descriptor = feature_descriptor or resolve_feature_descriptor(feature_name, bio_context=bio_context)
    aggregation = aggregate_member_support(
        feature_descriptor=descriptor,
        bio_context=bio_context,
        selected_context=selected_context,
        mode=effective_mode,
    )
    aggregate_support = float(aggregation.get("aggregate_member_support", 0.0) or 0.0)
    structure_bonus = compute_structure_bonus(descriptor, bio_context)
    gamma_topo = float(bio_context.config.get("gamma_topo", 0.05) or 0.05)
    p0_v2 = _clamp01(aggregate_support + gamma_topo * structure_bonus)

    return {
        "feature_name": feature_name,
        "mode": effective_mode,
        "feature_type": descriptor.feature_type,
        "member_hmdb_ids": list(descriptor.member_hmdb_ids),
        "descriptor_metadata": dict(descriptor.metadata),
        "member_supports": aggregation.get("member_supports", {}),
        "aggregation_strategy": aggregation.get("aggregation_strategy"),
        "aggregate_member_support": aggregate_support,
        "structure_bonus": structure_bonus,
        "gamma_topo": gamma_topo,
        "p0_v2": p0_v2,
    }


def calculate_f_bio_v2(
    features_list: List[str],
    bio_context: Optional[BioContext] = None,
    disease_name: Optional[str] = None,
    feature_pool: Optional[List[str]] = None,
    priors_dict: Optional[Dict[str, float]] = None,
    taxonomy_map: Optional[Dict[str, Any]] = None,
    pathway_map: Optional[Dict[str, Any]] = None,
    reaction_graph: Optional[Dict[str, Any]] = None,
    metabolite_context_map: Optional[Dict[str, Any]] = None,
    selected_context: Optional[Dict[str, Any]] = None,
    mode: Optional[AnchorMode] = None,
    return_debug: bool = False,
    **_: Any,
) -> Any:
    """
    Calculate panel-level Anchor-Conditional f_bio v2.

    By default this returns a float score for compatibility with the legacy
    `calculate_f_bio()` call sites. When `return_debug=True`, a structured
    payload containing panel decomposition is returned.
    """
    try:
        if not features_list:
            empty = {
                "f_bio_v2": 0.0,
                "mode": mode or (bio_context.anchor_mode if bio_context else "disease_only"),
                "anchor_features": [],
                "search_features": [],
                "p0_items": [],
                "avg_p0_v2": 0.0,
                "delta_panel": 0.0,
                "panel_connectivity": {},
            }
            return empty if return_debug else 0.0

        if bio_context is None:
            bio_context = build_bio_context(
                disease_name=disease_name,
                feature_pool=feature_pool or features_list,
                priors_dict=priors_dict,
                taxonomy_map=taxonomy_map,
                pathway_map=pathway_map,
                reaction_graph=reaction_graph,
                metabolite_context_map=metabolite_context_map,
            )

        effective_mode = mode or bio_context.anchor_mode
        descriptor_map = {
            feature_name: resolve_feature_descriptor(feature_name, bio_context=bio_context)
            for feature_name in features_list
        }

        anchor_hmdb_ids = set(bio_context.protected_anchor_ids)
        anchor_features: List[str] = []
        search_features: List[str] = []
        for feature_name, descriptor in descriptor_map.items():
            member_ids = {
                str(member or "").upper().strip()
                for member in descriptor.member_hmdb_ids
                if str(member or "").upper().strip()
            }
            if member_ids and member_ids.issubset(anchor_hmdb_ids):
                anchor_features.append(feature_name)
            else:
                search_features.append(feature_name)

        scoring_targets = search_features or list(features_list)
        p0_items = [
            _calculate_single_p0_v2(
                feature_name=feature_name,
                bio_context=bio_context,
                selected_context=selected_context,
                feature_descriptor=descriptor_map.get(feature_name),
                mode=effective_mode,
            )
            for feature_name in scoring_targets
        ]
        avg_p0_v2 = (
            sum(float(item.get("p0_v2", 0.0) or 0.0) for item in p0_items) / len(p0_items)
            if p0_items
            else 0.0
        )

        panel_connectivity = compute_panel_connectivity_gain(
            search_features=scoring_targets,
            bio_context=bio_context,
            anchor_features=anchor_features,
            descriptor_map=descriptor_map,
        )
        delta_panel = float(bio_context.config.get("delta_panel", 0.03) or 0.03)
        f_bio_v2 = _clamp01(avg_p0_v2 + delta_panel * float(panel_connectivity.get("connectivity_gain", 0.0) or 0.0))

        debug_payload = {
            "f_bio_v2": f_bio_v2,
            "mode": effective_mode,
            "anchor_features": anchor_features,
            "search_features": scoring_targets,
            "p0_items": p0_items,
            "avg_p0_v2": avg_p0_v2,
            "delta_panel": delta_panel,
            "panel_connectivity": panel_connectivity,
        }
        return debug_payload if return_debug else float(f_bio_v2)

    except Exception:
        if return_debug:
            return {
                "f_bio_v2": 0.0,
                "mode": mode or (bio_context.anchor_mode if bio_context else "disease_only"),
                "anchor_features": [],
                "search_features": list(features_list or []),
                "p0_items": [],
                "avg_p0_v2": 0.0,
                "delta_panel": 0.0,
                "panel_connectivity": {},
            }
        return 0.0


__all__ = [
    "AnchorMode",
    "BioContext",
    "DEFAULT_F_BIO_V2_CONFIG",
    "FeatureDescriptor",
    "FeatureType",
    "build_bio_context",
    "build_empty_bio_context",
    "compute_panel_connectivity_gain",
    "calculate_f_bio_v2",
    "detect_anchor_mode",
    "get_default_f_bio_v2_config",
    "merge_f_bio_v2_config",
    "resolve_feature_descriptor",
]
