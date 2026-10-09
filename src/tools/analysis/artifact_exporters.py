"""
Artifact Exporters for Phase 1 & Phase 2

这个模块负责在 Phase 1 和 Phase 2 完成核心任务后，导出画图所需的底层数据矩阵。
Phase 3 将完全依赖这些 Artifacts，不再进行任何模型训练和预测计算。

核心设计理念：
- Phase 1/2: 计算 + 导出 Artifacts
- Phase 3: 只读取 Artifacts + 画图
- 目标: 100% 数值一致性

Author: MetaboAgent Team
Date: 2026-04-24
"""

import os
import json
import pickle
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import warnings

from src.agents.phase3.evaluator_bridge import ClinicalEvaluator
from src.utils.config_manager import get_config

# 抑制警告
warnings.filterwarnings('ignore')


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _resolve_target_column(df: pd.DataFrame, requested_target: str) -> Optional[str]:
    candidates = [requested_target, "target", "Group"]
    for candidate in _dedupe_preserve_order(candidates):
        if candidate in df.columns:
            return candidate
    return None


def _load_best_phase1_panel_dataset(
    data_path: str,
    target_column: str,
    selected_features: List[str],
) -> Tuple[str, str, pd.DataFrame]:
    """
    Resolve the best available Phase 1 final dataset using config-aware priority.

    Preference follows Phase 1 IO policy so legacy callers do not accidentally
    score stale data when the canonical final dataset already exists.
    """
    cfg = get_config()
    io_policy = cfg.get_phase1_io_policy()
    prefer_new_read = bool(io_policy.get("prefer_new_read", False))
    fallback_old_read = bool(io_policy.get("fallback_old_read", True))

    canonical_path = cfg.get_phase1_path("selected_features_csv") or data_path
    legacy_path = cfg.get_phase1_path("legacy_selected_features") or data_path

    ordered_candidates = [data_path]
    if prefer_new_read:
        ordered_candidates = [canonical_path, data_path]
        if fallback_old_read:
            ordered_candidates.append(legacy_path)
    else:
        ordered_candidates = [data_path, legacy_path]
        if fallback_old_read:
            ordered_candidates.append(canonical_path)

    candidate_paths = _dedupe_preserve_order(ordered_candidates)
    candidate_reports = []

    for candidate in candidate_paths:
        if not candidate or not os.path.exists(candidate):
            candidate_reports.append(f"{candidate or '<empty>'}: missing")
            continue

        try:
            df = pd.read_csv(candidate)
        except Exception as exc:
            candidate_reports.append(f"{candidate}: unreadable ({exc})")
            continue

        resolved_target = _resolve_target_column(df, target_column)
        feature_hits = sum(1 for feature in selected_features if feature in df.columns)

        if resolved_target is None:
            candidate_reports.append(
                f"{candidate}: target missing (requested={target_column}, columns={list(df.columns[:10])}...)"
            )
            continue

        if feature_hits == 0 and selected_features:
            candidate_reports.append(
                f"{candidate}: 0/{len(selected_features)} selected features found"
            )
            continue

        return candidate, resolved_target, df

    raise FileNotFoundError(
        "Could not resolve a valid Phase 1 panel dataset. Checked: "
        + " | ".join(candidate_reports)
    )


def _mirror_phase1_artifact(artifact_path: str, artifact_filename: str) -> None:
    """Mirror Phase 1 artifacts to both canonical and legacy directories when enabled."""
    if not artifact_path or not os.path.exists(artifact_path):
        return

    cfg = get_config()
    io_policy = cfg.get_phase1_io_policy()
    if not bool(io_policy.get("dual_write_enabled", False)):
        return

    canonical_dir = cfg.get_phase1_path("artifacts_dir") or "output/phase1/artifacts"
    legacy_dir = cfg.get_phase1_path("legacy_artifacts_dir") or "output/artifacts"
    source_abs = os.path.abspath(artifact_path)

    for artifact_dir in _dedupe_preserve_order([canonical_dir, legacy_dir]):
        target_path = os.path.join(artifact_dir, artifact_filename)
        target_abs = os.path.abspath(target_path)
        if target_abs == source_abs:
            continue
        os.makedirs(os.path.dirname(target_abs), exist_ok=True)
        shutil.copy2(source_abs, target_abs)
        print(f"✅ Dual-write mirror saved to: {target_abs}")


def _load_existing_phase2_contribution_artifact(
    output_dir: str,
    winner_features: List[str],
) -> Optional[Dict[str, Any]]:
    """Reuse an existing contribution artifact when the feature set still matches."""
    csv_path = os.path.join(output_dir, "phase2_winner_feature_contributions.csv")
    summary_path = os.path.join(output_dir, "phase2_winner_feature_contributions_summary.json")
    if not os.path.exists(csv_path):
        return None

    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return None

    existing_features = {
        str(value).strip()
        for value in frame.get("feature", [])
        if str(value).strip()
    }
    normalized_winner_features = {
        str(value).strip()
        for value in winner_features
        if str(value).strip()
    }
    if len(frame.index) != len(normalized_winner_features) or existing_features != normalized_winner_features:
        return None

    summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                summary = payload
        except Exception:
            summary = {}

    summary["contribution_csv"] = csv_path
    summary["winner_feature_count"] = int(len(frame.index))
    summary["reused_existing"] = True
    return summary


def _resolve_phase1_ag_results_path(ag_results_path: str) -> str:
    """Prefer the current canonical AutoGluon results over stale legacy mirrors."""
    cfg = get_config()
    io_policy = cfg.get_phase1_io_policy()
    prefer_new_read = bool(io_policy.get("prefer_new_read", False))
    fallback_old_read = bool(io_policy.get("fallback_old_read", True))

    canonical_path = os.path.join(
        cfg.get_phase1_path("artifacts_dir") or "output/phase1/artifacts",
        "autogluon_training_results.json",
    )
    legacy_path = cfg.get_phase1_path("autogluon_results") or ag_results_path

    if prefer_new_read:
        candidates = [canonical_path, ag_results_path]
        if fallback_old_read:
            candidates.append(legacy_path)
    else:
        candidates = [ag_results_path, legacy_path]
        if fallback_old_read:
            candidates.append(canonical_path)

    for candidate in _dedupe_preserve_order(candidates):
        if candidate and os.path.exists(candidate):
            if candidate != ag_results_path:
                print(f"[Panel Export] Using resolved AutoGluon results: {candidate}")
            return candidate

    return ag_results_path


# ============================================================================
# Phase 1 Feature Provenance Helpers
# ============================================================================

_HMDB_ID_PATTERN = re.compile(r"HMDB\d{5,9}", re.IGNORECASE)


def _safe_standardize_feature_name(feature_name: str) -> str:
    try:
        from src.utils.feature_name_standardizer import standardize_feature_name
        return standardize_feature_name(feature_name)
    except Exception:
        return str(feature_name or "").strip().upper()


def _humanize_feature_name(feature_name: str) -> str:
    text = str(feature_name or "").strip()
    if not text:
        return ""
    text = text.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if _HMDB_ID_PATTERN.fullmatch(text):
        return text.upper()
    return text[:1].upper() + text[1:].lower() if text else ""


def _load_metabolite_lookup_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return both:
    1) standardized metabolite name -> HMDB ID
    2) HMDB ID -> display name

    The lookup is best-effort and gracefully degrades to empty maps when the
    storage files are not available.
    """
    name_to_hmdb: Dict[str, str] = {}
    hmdb_to_name: Dict[str, str] = {}

    try:
        lookup_path = os.path.join("storage", "metabolite_lookup.json")
        if os.path.exists(lookup_path):
            with open(lookup_path, "r", encoding="utf-8") as f:
                lookup_map = json.load(f)
            if isinstance(lookup_map, dict):
                for metabolite_name, hmdb_id in lookup_map.items():
                    if not isinstance(metabolite_name, str) or not isinstance(hmdb_id, str):
                        continue
                    standardized_name = _safe_standardize_feature_name(metabolite_name)
                    normalized_hmdb = hmdb_id.upper().strip()
                    if standardized_name and normalized_hmdb:
                        name_to_hmdb[standardized_name] = normalized_hmdb
                        hmdb_to_name.setdefault(normalized_hmdb, metabolite_name)
    except Exception as e:
        print(f"Warning: Failed to load metabolite lookup map for provenance export: {e}")

    try:
        shorthand_path = os.path.join("storage", "shorthand_map.json")
        if os.path.exists(shorthand_path):
            with open(shorthand_path, "r", encoding="utf-8") as f:
                shorthand_map = json.load(f)
            if isinstance(shorthand_map, dict):
                for shorthand_name, payload in shorthand_map.items():
                    if not isinstance(payload, dict):
                        continue
                    hmdb_id = str(payload.get("hmdb_id", "")).upper().strip()
                    display_name = str(payload.get("name", "")).strip() or shorthand_name
                    standardized_name = _safe_standardize_feature_name(shorthand_name)
                    if standardized_name and hmdb_id:
                        name_to_hmdb[standardized_name] = hmdb_id
                        hmdb_to_name.setdefault(hmdb_id, display_name)
    except Exception as e:
        print(f"Warning: Failed to load shorthand map for provenance export: {e}")

    return name_to_hmdb, hmdb_to_name


def _load_engineered_feature_registry() -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """
    Best-effort load for the engineered feature registry generated during Phase 1.

    Returns:
        Tuple[registry_index, registry_path]
    """
    registry_index: Dict[str, Dict[str, Any]] = {}
    registry_path: Optional[str] = None

    try:
        registry_path = get_config().get_phase1_engineered_feature_registry_path()
        if not registry_path or not os.path.exists(registry_path):
            return registry_index, registry_path

        with open(registry_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not isinstance(features, list):
            return registry_index, registry_path

        for record in features:
            if not isinstance(record, dict):
                continue
            raw_feature_name = str(record.get("feature_name", "") or "").strip()
            standardized_name = _safe_standardize_feature_name(raw_feature_name)
            if raw_feature_name:
                registry_index[raw_feature_name] = record
            if standardized_name:
                registry_index[standardized_name] = record
    except Exception as e:
        print(f"Warning: Failed to load engineered feature registry for provenance export: {e}")

    return registry_index, registry_path


def _resolve_registry_record(
    feature_name: str,
    registry_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve a registry record by raw or standardized feature name."""
    if not registry_index:
        return None
    raw_feature_name = str(feature_name or "").strip()
    standardized_name = _safe_standardize_feature_name(feature_name)
    return registry_index.get(raw_feature_name) or registry_index.get(standardized_name)


def _build_hmdb_to_pathways(pathway_map: Dict[str, Any]) -> Dict[str, List[str]]:
    hmdb_to_pathways: Dict[str, List[str]] = {}
    for pathway_name, members in pathway_map.items():
        if not isinstance(pathway_name, str) or not isinstance(members, list):
            continue
        for member in members:
            member_id = str(member).upper().strip()
            if not member_id:
                continue
            hmdb_to_pathways.setdefault(member_id, [])
            if pathway_name not in hmdb_to_pathways[member_id]:
                hmdb_to_pathways[member_id].append(pathway_name)
    return hmdb_to_pathways


def _extract_hmdb_ids_from_feature_name(feature_name: str) -> List[str]:
    matches = _HMDB_ID_PATTERN.findall(str(feature_name or ""))
    normalized = []
    seen = set()
    for match in matches:
        hmdb_id = match.upper()
        if hmdb_id not in seen:
            seen.add(hmdb_id)
            normalized.append(hmdb_id)
    return normalized


def _build_phase0_support_sets(
    phase0_output: Optional[Dict[str, Any]],
    name_to_hmdb: Dict[str, str],
) -> Tuple[List[str], set[str], set[str]]:
    confirmed_biomarkers = []
    prior_name_set: set[str] = set()
    prior_hmdb_set: set[str] = set()

    if isinstance(phase0_output, dict):
        candidate_sources = [
            phase0_output.get("confirmed_biomarkers", []) or [],
            phase0_output.get("final_priors", []) or [],
            (phase0_output.get("feature_definitions", {}) or {}).get("target_metabolites", []) or [],
        ]
        for source in candidate_sources:
            if not isinstance(source, list):
                continue
            confirmed_biomarkers.extend(source)

    for biomarker in confirmed_biomarkers:
        biomarker_id = ""
        biomarker_name = ""

        if isinstance(biomarker, dict):
            biomarker_id = str(biomarker.get("id", "") or "").upper().strip()
            biomarker_name = str(biomarker.get("name", "") or "").strip()
        else:
            biomarker_name = str(biomarker).strip()

        if biomarker_id:
            prior_hmdb_set.add(biomarker_id)
            standardized_id = _safe_standardize_feature_name(biomarker_id)
            if standardized_id:
                prior_name_set.add(standardized_id)

        if biomarker_name:
            standardized_name = _safe_standardize_feature_name(biomarker_name)
            if standardized_name:
                prior_name_set.add(standardized_name)
            mapped_hmdb = name_to_hmdb.get(standardized_name)
            if mapped_hmdb:
                prior_hmdb_set.add(mapped_hmdb)

    return confirmed_biomarkers, prior_name_set, prior_hmdb_set


def _classify_feature_origin(
    feature_name: str,
    pathway_name_map: Dict[str, str],
    registry_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    standardized_name = _safe_standardize_feature_name(feature_name)

    if isinstance(registry_record, dict):
        feature_type = str(registry_record.get("feature_type", "") or "").strip()
        origin_subtype = str(registry_record.get("origin_subtype", "") or "").strip() or None
        if feature_type:
            return {
                "origin_type": feature_type,
                "origin_subtype": origin_subtype,
                "is_engineered": feature_type != "raw",
                "annotation_confidence": "high",
                "evidence_tags": ["engineered_feature_registry"],
            }

    if standardized_name.startswith("RATIO_"):
        return {
            "origin_type": "engineered_ratio",
            "origin_subtype": "reaction_ratio",
            "is_engineered": True,
            "annotation_confidence": "high",
            "evidence_tags": ["ratio_prefix"],
        }

    if standardized_name.startswith("TAXSUM_"):
        return {
            "origin_type": "engineered_sum",
            "origin_subtype": "taxonomy_sum",
            "is_engineered": True,
            "annotation_confidence": "high",
            "evidence_tags": ["taxonomy_sum_prefix"],
        }

    if standardized_name.startswith("SUM_"):
        return {
            "origin_type": "engineered_sum",
            "origin_subtype": "group_sum",
            "is_engineered": True,
            "annotation_confidence": "high",
            "evidence_tags": ["sum_prefix"],
        }

    if _HMDB_ID_PATTERN.fullmatch(standardized_name):
        return {
            "origin_type": "raw",
            "origin_subtype": "hmdb_metabolite",
            "is_engineered": False,
            "annotation_confidence": "high",
            "evidence_tags": ["hmdb_id"],
        }

    if standardized_name in pathway_name_map:
        return {
            "origin_type": "engineered_pathway",
            "origin_subtype": "pathway_score",
            "is_engineered": True,
            "annotation_confidence": "high",
            "evidence_tags": ["pathway_name_match"],
        }

    return {
        "origin_type": "raw",
        "origin_subtype": "named_metabolite",
        "is_engineered": False,
        "annotation_confidence": "medium",
        "evidence_tags": ["name_based_raw"],
    }


def _infer_feature_mappings(
    feature_name: str,
    origin_type: str,
    origin_subtype: Optional[str],
    pathway_name_map: Dict[str, str],
    hmdb_to_pathways: Dict[str, List[str]],
    name_to_hmdb: Dict[str, str],
    hmdb_to_name: Dict[str, str],
    registry_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    standardized_name = _safe_standardize_feature_name(feature_name)
    mapped_hmdb_ids: List[str] = []
    mapped_metabolites: List[str] = []
    mapped_pathways: List[str] = []

    if isinstance(registry_record, dict):
        mapped_hmdb_ids = _dedupe_preserve_order([
            str(hmdb_id).upper().strip()
            for hmdb_id in (
                registry_record.get("mapped_hmdb_ids")
                or registry_record.get("selected_member_hmdb_ids")
                or registry_record.get("observed_member_hmdb_ids")
                or []
            )
            if str(hmdb_id).strip()
        ])
        mapped_pathways = _dedupe_preserve_order([
            str(pathway_name).strip()
            for pathway_name in (registry_record.get("mapped_pathways") or [])
            if str(pathway_name).strip()
        ])

    if not mapped_hmdb_ids and origin_type == "engineered_ratio":
        mapped_hmdb_ids = _extract_hmdb_ids_from_feature_name(feature_name)
    elif not mapped_pathways and origin_type == "engineered_pathway":
        pathway_name = pathway_name_map.get(standardized_name)
        if pathway_name:
            mapped_pathways = [pathway_name]
    elif origin_type == "raw" and origin_subtype == "hmdb_metabolite":
        mapped_hmdb_ids = [standardized_name]
    elif origin_type == "raw":
        mapped_hmdb = name_to_hmdb.get(standardized_name)
        if mapped_hmdb:
            mapped_hmdb_ids = [mapped_hmdb]
        else:
            mapped_metabolites = [_humanize_feature_name(feature_name)]

    for hmdb_id in mapped_hmdb_ids:
        display_name = hmdb_to_name.get(hmdb_id)
        if display_name:
            mapped_metabolites.append(display_name)
        for pathway_name in hmdb_to_pathways.get(hmdb_id, []):
            if pathway_name not in mapped_pathways:
                mapped_pathways.append(pathway_name)

    mapped_metabolites = _dedupe_preserve_order([m for m in mapped_metabolites if m])
    mapped_pathways = _dedupe_preserve_order([p for p in mapped_pathways if p])

    return {
        "mapped_hmdb_ids": mapped_hmdb_ids,
        "mapped_metabolites": mapped_metabolites,
        "mapped_pathways": mapped_pathways,
    }


def _infer_prior_support(
    feature_name: str,
    mapped_hmdb_ids: List[str],
    mapped_metabolites: List[str],
    phase0_prior_name_set: set[str],
    phase0_prior_hmdb_set: set[str],
) -> Dict[str, Any]:
    standardized_feature_name = _safe_standardize_feature_name(feature_name)

    if standardized_feature_name in phase0_prior_name_set:
        return {
            "is_prior_supported": True,
            "prior_support_source": "phase0_confirmed_biomarkers",
        }

    for hmdb_id in mapped_hmdb_ids:
        if hmdb_id in phase0_prior_hmdb_set:
            return {
                "is_prior_supported": True,
                "prior_support_source": "phase0_confirmed_biomarkers",
            }

    for metabolite_name in mapped_metabolites:
        if _safe_standardize_feature_name(metabolite_name) in phase0_prior_name_set:
            return {
                "is_prior_supported": True,
                "prior_support_source": "derived_from_prior_metabolites",
            }

    return {
        "is_prior_supported": False,
        "prior_support_source": "none",
    }


def _build_feature_display_name(
    feature_name: str,
    origin_type: str,
    origin_subtype: Optional[str],
    mapped_metabolites: List[str],
    pathway_name_map: Dict[str, str],
    hmdb_to_name: Dict[str, str],
    registry_record: Optional[Dict[str, Any]] = None,
) -> str:
    standardized_name = _safe_standardize_feature_name(feature_name)

    if isinstance(registry_record, dict):
        if origin_type == "engineered_pathway":
            mapped_pathways = registry_record.get("mapped_pathways") or []
            if isinstance(mapped_pathways, list) and mapped_pathways:
                return str(mapped_pathways[0])
        if origin_type == "engineered_sum":
            group_label = str(registry_record.get("group_label", "") or "").strip()
            if group_label:
                return group_label

    if origin_type == "engineered_pathway" and standardized_name in pathway_name_map:
        return pathway_name_map[standardized_name]
    if origin_type == "raw" and origin_subtype == "hmdb_metabolite":
        return hmdb_to_name.get(standardized_name, standardized_name)
    if mapped_metabolites:
        return mapped_metabolites[0]
    return _humanize_feature_name(feature_name)


def _summarize_feature_provenance(feature_records: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "raw_count": 0,
        "engineered_ratio_count": 0,
        "engineered_sum_count": 0,
        "engineered_pathway_count": 0,
        "unknown_count": 0,
        "prior_supported_count": 0,
        "taxonomy_sum_count": 0,
        "group_sum_count": 0,
    }

    for record in feature_records:
        origin_type = record.get("origin_type")
        origin_subtype = record.get("origin_subtype")

        if origin_type == "raw":
            summary["raw_count"] += 1
        elif origin_type == "engineered_ratio":
            summary["engineered_ratio_count"] += 1
        elif origin_type == "engineered_sum":
            summary["engineered_sum_count"] += 1
            if origin_subtype == "taxonomy_sum":
                summary["taxonomy_sum_count"] += 1
            elif origin_subtype == "group_sum":
                summary["group_sum_count"] += 1
        elif origin_type == "engineered_pathway":
            summary["engineered_pathway_count"] += 1
        else:
            summary["unknown_count"] += 1

        if record.get("is_prior_supported"):
            summary["prior_supported_count"] += 1

    return summary


def export_feature_provenance(
    selected_features: List[str],
    data_path: str,
    target_column: str,
    phase0_output: Optional[Dict[str, Any]] = None,
    output_dir: str = "output/artifacts",
    artifact_filename: str = "feature_provenance.json",
) -> str:
    """
    Export feature provenance for the final Phase 1 panel.

    一级分类严格贴合当前 Phase 1 设计:
    - raw
    - engineered_ratio
    - engineered_sum
    - engineered_pathway
    - unknown

    其中 taxonomy-based sums 归属于 engineered_sum，并通过
    origin_subtype="taxonomy_sum" 标记细分来源。
    """
    print("\n" + "=" * 80)
    print("Phase 1: Exporting Feature Provenance Artifact")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    resolved_data_path, resolved_target_column, resolved_df = _load_best_phase1_panel_dataset(
        data_path=data_path,
        target_column=target_column,
        selected_features=selected_features,
    )
    valid_features = [feature for feature in selected_features if feature in resolved_df.columns]
    if not valid_features:
        raise ValueError("No selected features were found in the resolved dataset for provenance export")

    from src.tools.analysis.pareto_evaluator import _load_pathway_map

    pathway_map = _load_pathway_map()
    pathway_name_map = {
        _safe_standardize_feature_name(pathway_name): pathway_name
        for pathway_name in pathway_map.keys()
        if isinstance(pathway_name, str)
    }
    registry_index, registry_path = _load_engineered_feature_registry()
    hmdb_to_pathways = _build_hmdb_to_pathways(pathway_map)
    name_to_hmdb, hmdb_to_name = _load_metabolite_lookup_maps()
    confirmed_biomarkers, phase0_prior_name_set, phase0_prior_hmdb_set = _build_phase0_support_sets(
        phase0_output,
        name_to_hmdb,
    )

    feature_records: List[Dict[str, Any]] = []
    for feature_name in valid_features:
        registry_record = _resolve_registry_record(feature_name, registry_index)
        classification = _classify_feature_origin(
            feature_name,
            pathway_name_map,
            registry_record=registry_record,
        )
        mappings = _infer_feature_mappings(
            feature_name=feature_name,
            origin_type=classification["origin_type"],
            origin_subtype=classification["origin_subtype"],
            pathway_name_map=pathway_name_map,
            hmdb_to_pathways=hmdb_to_pathways,
            name_to_hmdb=name_to_hmdb,
            hmdb_to_name=hmdb_to_name,
            registry_record=registry_record,
        )
        prior_support = _infer_prior_support(
            feature_name=feature_name,
            mapped_hmdb_ids=mappings["mapped_hmdb_ids"],
            mapped_metabolites=mappings["mapped_metabolites"],
            phase0_prior_name_set=phase0_prior_name_set,
            phase0_prior_hmdb_set=phase0_prior_hmdb_set,
        )

        evidence_tags = list(classification.get("evidence_tags", []))
        if prior_support["is_prior_supported"]:
            evidence_tags.append("phase0_prior_match")
        if mappings["mapped_pathways"]:
            evidence_tags.append("pathway_mapping")

        record = {
            "feature": feature_name,
            "display_name": _build_feature_display_name(
                feature_name=feature_name,
                origin_type=classification["origin_type"],
                origin_subtype=classification["origin_subtype"],
                mapped_metabolites=mappings["mapped_metabolites"],
                pathway_name_map=pathway_name_map,
                hmdb_to_name=hmdb_to_name,
                registry_record=registry_record,
            ),
            "standardized_name": _safe_standardize_feature_name(feature_name),
            "origin_type": classification["origin_type"],
            "origin_subtype": classification["origin_subtype"],
            "is_engineered": classification["is_engineered"],
            "is_prior_supported": prior_support["is_prior_supported"],
            "prior_support_source": prior_support["prior_support_source"],
            "source_phase": "phase0+phase1" if prior_support["is_prior_supported"] else "phase1",
            "mapped_metabolites": mappings["mapped_metabolites"],
            "mapped_hmdb_ids": mappings["mapped_hmdb_ids"],
            "mapped_pathways": mappings["mapped_pathways"],
            "annotation_confidence": classification["annotation_confidence"],
            "evidence_tags": _dedupe_preserve_order(evidence_tags),
            "registry_matched": isinstance(registry_record, dict),
            "notes": None,
        }
        if isinstance(registry_record, dict):
            if "quality_filters" in registry_record:
                record["quality_filters"] = registry_record.get("quality_filters")
            if "reaction" in registry_record:
                record["reaction"] = registry_record.get("reaction")
            for extra_key in [
                "group_label",
                "taxonomy_level",
                "dominant_direction",
                "direction_consistency",
                "member_effects",
                "all_member_hmdb_ids",
                "observed_member_hmdb_ids",
                "coverage_ratio",
                "observed_count",
            ]:
                if extra_key in registry_record:
                    record[extra_key] = registry_record.get(extra_key)
        feature_records.append(record)

    artifact_path = os.path.join(output_dir, artifact_filename)
    payload = {
        "schema_version": "phase4.feature_provenance.v1",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_path": resolved_data_path,
        "target_column": resolved_target_column,
        "selected_features_count": len(valid_features),
        "selected_features": valid_features,
        "phase0_confirmed_biomarkers": confirmed_biomarkers,
        "features": feature_records,
        "summary": _summarize_feature_provenance(feature_records),
        "classification_rules_version": "phase4.provenance.rules.v2",
        "engineered_feature_registry_path": registry_path,
        "notes": [
            "Taxonomy-derived features are classified as engineered_sum with origin_subtype=taxonomy_sum.",
            "Pathway activity scores remain a separate engineered_pathway category because they are enrichment/activity scores, not direct additive sums.",
            "When the engineered feature registry is available, provenance export prefers registry-backed mapped_hmdb_ids and mapped_pathways over name-based inference.",
        ],
    }

    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    try:
        _mirror_phase1_artifact(artifact_path, artifact_filename)
    except Exception as e:
        print(f"⚠️  Warning: failed to mirror feature provenance artifact: {e}")

    print(f"\n✅ Artifact saved to: {artifact_path}")
    print(f"   Features: {len(feature_records)}")
    print(f"   Summary: {payload['summary']}")
    print("=" * 80 + "\n")
    return artifact_path


# ============================================================================
# Phase 1 Artifact Exporters
# ============================================================================

def export_phase1_stats_artifacts(
    data_path: str,
    target_column: str,
    selected_features: List[str],
    output_dir: str = 'output/artifacts'
) -> str:
    """
    导出 Phase 1 统计降维 Artifacts (PCA & PLS-DA)
    
    在 Phase 1 特征选择完成后调用，导出 PCA 和 PLS-DA 的降维坐标。
    Phase 3 的 plot_stats_scatter 将直接读取这些坐标，不再重新计算。
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名称
        selected_features: Phase 1 选出的特征列表
        output_dir: Artifact 输出目录
    
    Returns:
        str: 生成的 Artifact 文件路径
    
    Examples:
        >>> features = ['HMDB0000001', 'HMDB0000002', 'HMDB0000003']
        >>> artifact_path = export_phase1_stats_artifacts(
        ...     'data/cleaned.csv', 'Group', features
        ... )
        >>> print(f"Artifact saved to: {artifact_path}")
    """
    print("\n" + "="*80)
    print("Phase 1: Exporting Statistical Dimensionality Reduction Artifacts")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 过滤出存在的特征
    valid_features = [f for f in selected_features if f in df.columns]
    
    if not valid_features:
        print("Warning: No valid features found. Skipping artifact export.")
        return None
    
    print(f"Data loaded: {df.shape}")
    print(f"Valid features: {len(valid_features)}/{len(selected_features)}")
    
    # 提取特征和标签
    X = df[valid_features].values
    y = df[target_column].values
    
    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ========================================================================
    # 1. PCA 降维
    # ========================================================================
    print("\n[1/2] Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    pca_variance_ratio = pca.explained_variance_ratio_
    
    print(f"  PCA variance explained: PC1={pca_variance_ratio[0]:.2%}, PC2={pca_variance_ratio[1]:.2%}")
    
    # ========================================================================
    # 2. PLS-DA 降维
    # ========================================================================
    print("\n[2/2] Computing PLS-DA...")
    pls = PLSRegression(n_components=2, scale=False)  # 已经标准化过了
    pls.fit(X_scaled, y_encoded)
    plsda_coords = pls.transform(X_scaled)
    
    # 计算 PLS-DA 的方差解释率（近似）
    plsda_variance_ratio = np.var(plsda_coords, axis=0) / np.sum(np.var(plsda_coords, axis=0))
    
    print(f"  PLS-DA variance explained: LV1={plsda_variance_ratio[0]:.2%}, LV2={plsda_variance_ratio[1]:.2%}")
    
    # ========================================================================
    # 3. 保存 Artifacts
    # ========================================================================
    artifact_path = os.path.join(output_dir, 'phase1_stats_artifacts.npz')
    
    np.savez(
        artifact_path,
        pca_coords=pca_coords,
        pca_variance_ratio=pca_variance_ratio,
        plsda_coords=plsda_coords,
        plsda_variance_ratio=plsda_variance_ratio,
        labels=y,
        label_names=le.classes_,
        sample_ids=np.arange(len(y))
    )
    
    print(f"\n✅ Artifact saved to: {artifact_path}")
    print(f"   File size: {os.path.getsize(artifact_path) / 1024:.2f} KB")
    print("="*80 + "\n")
    
    return artifact_path


def export_phase1_panel_scores(
    data_path: str,
    target_column: str,
    selected_features: List[str],
    ag_results_path: str = 'data/autogluon_training_results.json',
    eval_data_path: Optional[str] = None,
    output_dir: str = 'output/artifacts',
    artifact_filename: str = 'phase1_panel_scores.json'
) -> str:
    """
    导出 Phase 1 最终特征面板的四维评分 Artifact (f_perf/f_bio/f_corr/f_cost)
 
    设计目标：
    - Phase 1/2 负责计算并落盘
    - Phase 3/测试脚本只读取 artifact 进行可视化
 
    Args:
        data_path: Phase 1 最终特征数据集路径（优先解析为当前 canonical final dataset）
        target_column: 目标列名
        selected_features: 特征列表（建议与 data_path 中列一致）
        ag_results_path: AutoGluon 训练结果路径（用于 f_perf 回溯计算）
        eval_data_path: 可选的评估数据集路径；存在时优先用于 panel scoring
        output_dir: Artifact 输出目录
        artifact_filename: 输出文件名
 
    Returns:
        str: 生成的 Artifact JSON 文件路径
    """
    print("\n" + "="*80)
    print("Phase 1: Exporting Final Panel 4D Scores Artifact")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # 🔧 修复: 自动加载 Phase 0 缓存和知识库
    # ========================================================================
    from src.tools.analysis.pareto_evaluator import (
        _load_priors_dict,
        _load_taxonomy_map,
        _load_pathway_map
    )
    
    print("\n[Loading Knowledge Base]")
    priors_dict = _load_priors_dict()
    taxonomy_map = _load_taxonomy_map()
    pathway_map = _load_pathway_map()
 
    resolved_data_path, resolved_target_column, resolved_df = _load_best_phase1_panel_dataset(
        data_path=eval_data_path or data_path,
        target_column=target_column,
        selected_features=selected_features,
    )

    if resolved_data_path != data_path:
        print(f"[Panel Export] Using resolved dataset: {resolved_data_path}")
    if resolved_target_column != target_column:
        print(
            f"[Panel Export] Target column adjusted: "
            f"{target_column} -> {resolved_target_column}"
        )

    valid_features = [f for f in selected_features if f in resolved_df.columns]
    if not valid_features:
        raise ValueError("No selected features were found in the resolved Phase 1 dataset")

    resolved_ag_results_path = _resolve_phase1_ag_results_path(ag_results_path)
    holdout_data_path = resolved_data_path if resolved_data_path != data_path else None

    evaluator = ClinicalEvaluator(
        data_path=data_path,
        holdout_data_path=holdout_data_path,
        target_column=resolved_target_column,
        priors_dict=priors_dict,      # ✅ 使用加载的先验字典
        taxonomy_map=taxonomy_map,    # ✅ 使用加载的分类映射
        pathway_map=pathway_map,      # ✅ 使用加载的通路映射
        ag_results_path=resolved_ag_results_path
    )
 
    raw_scores = evaluator.score_panel(valid_features, panel_name='Phase 1 Final Panel')
 
    artifact_path = os.path.join(output_dir, artifact_filename)
    payload = {
        'panel_name': 'Phase 1 Final Panel',
        'data_path': resolved_data_path,
        'training_data_path': data_path,
        'evaluation_data_path': resolved_data_path,
        'performance_metric': 'roc_auc',
        'performance_protocol': 'holdout_only' if holdout_data_path else 'train_5fold_cv',
        'target_column': resolved_target_column,
        'selected_features': valid_features,
        'scores': raw_scores
    }
 
    with open(artifact_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    try:
        _mirror_phase1_artifact(artifact_path, artifact_filename)
    except Exception as e:
        print(f"⚠️  Warning: failed to mirror Phase1 panel scores artifact: {e}")
 
    print(f"\n✅ Artifact saved to: {artifact_path}")
    print(f"   File size: {os.path.getsize(artifact_path) / 1024:.2f} KB")
    print("="*80 + "\n")
 
    return artifact_path


# ============================================================================
# Phase 2 Artifact Exporters
# ============================================================================

def export_phase2_winner_feature_contributions(
    winner_scores: Dict[str, Any],
    search_summary: Dict[str, Any],
    phase0_output: Optional[Dict[str, Any]] = None,
    data_path: Optional[str] = None,
    ag_results_path: Optional[str] = None,
    feature_provenance_path: Optional[str] = None,
    output_dir: str = 'output/artifacts',
) -> Optional[Dict[str, Any]]:
    """
    Export Phase 2 winner leave-one-feature-out contribution artifacts.

    This formalizes the data source required by fig3 so Phase 3 can prefer
    direct artifact reads instead of recomputing the winner drop-one analysis.
    """
    print("\n" + "=" * 80)
    print("Phase 2: Exporting Winner Feature Contribution Artifacts")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    winner_features = [str(value).strip() for value in winner_scores.get("selected_features", []) if str(value).strip()]
    if not winner_features:
        print("⚠️  Winner feature list is empty. Skipping contribution artifact export.")
        return None

    existing_summary = _load_existing_phase2_contribution_artifact(output_dir, winner_features)
    if existing_summary is not None:
        print(f"✅ Reusing existing contribution artifact: {existing_summary.get('contribution_csv')}")
        print("=" * 80 + "\n")
        return existing_summary

    # Lazy import avoids a ptot_engine <-> artifact_exporters circular import
    # because the repair module imports TOPSIS helpers from ptot_engine.
    from src.utils.phase2_artifact_repair import (
        _load_taxonomy_map,
        _read_json,
        _repair_winner_feature_contributions,
        _resolve_existing_path,
    )

    cfg = get_config()
    phase1_paths = cfg.get_phase1_paths() if hasattr(cfg, "get_phase1_paths") else {}
    canonical_artifact_dir = phase1_paths.get("artifacts_dir", output_dir)
    legacy_artifact_dir = phase1_paths.get("legacy_artifacts_dir", output_dir)

    resolved_data_path = _resolve_existing_path(
        data_path,
        winner_scores.get("evaluation_data_path"),
        winner_scores.get("training_data_path"),
        winner_scores.get("data_path"),
        phase1_paths.get("selected_features_csv"),
        phase1_paths.get("legacy_selected_features"),
    )
    resolved_ag_results_path = _resolve_existing_path(
        ag_results_path,
        winner_scores.get("ag_results_path"),
        os.path.join(output_dir, "autogluon_training_results.json"),
        os.path.join(canonical_artifact_dir, "autogluon_training_results.json"),
        phase1_paths.get("autogluon_results"),
        os.path.join("output", "phase1", "artifacts", "autogluon_training_results.json"),
        os.path.join("data", "autogluon_training_results.json"),
    )
    resolved_feature_provenance_path = _resolve_existing_path(
        feature_provenance_path,
        os.path.join(output_dir, "feature_provenance.json"),
        os.path.join(canonical_artifact_dir, "feature_provenance.json"),
        os.path.join(legacy_artifact_dir, "feature_provenance.json"),
    )
    taxonomy_map_path = _resolve_existing_path("storage/taxonomy_map.json")

    resolved_phase0_output = phase0_output if isinstance(phase0_output, dict) and phase0_output else {}
    if not resolved_phase0_output:
        for payload in (winner_scores.get("phase0_output"), search_summary.get("phase0_output")):
            if isinstance(payload, dict) and payload:
                resolved_phase0_output = payload
                break

    summary = _repair_winner_feature_contributions(
        winner_scores=winner_scores,
        search_summary=search_summary,
        phase0_output=resolved_phase0_output,
        provenance_payload=_read_json(resolved_feature_provenance_path),
        taxonomy_map=_load_taxonomy_map(taxonomy_map_path),
        data_path=resolved_data_path,
        ag_results_path=resolved_ag_results_path,
        output_dir=Path(os.path.abspath(output_dir)),
    )

    for artifact_filename in (
        "phase2_winner_feature_contributions.csv",
        "phase2_winner_feature_contributions_summary.json",
    ):
        artifact_path = os.path.join(output_dir, artifact_filename)
        try:
            _mirror_phase1_artifact(artifact_path, artifact_filename)
        except Exception as e:
            print(f"⚠️  Warning: failed to mirror {artifact_filename}: {e}")

    print(f"✅ Contribution artifact saved: {summary.get('contribution_csv')}")
    print("=" * 80 + "\n")
    return summary


def export_winner_cv_predictions(
    data_path: str,
    target_column: str,
    winner_features: List[str],
    champion_model_family: str,
    k_folds: int = 5,
    output_dir: str = 'output/artifacts',
    use_phase1_config: bool = True
) -> str:
    """
    导出 Winner Panel 的交叉验证预测概率 (OOF Predictions)
    
    在 Phase 2 选出 Winner Panel 后调用，使用 cross_val_predict 计算全量样本的
    袋外预测概率。Phase 3 的 plot_final_roc 和 plot_dca 将直接读取这些概率。
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名称
        winner_features: Winner Panel 特征列表
        champion_model_family: 冠军模型家族
        k_folds: 交叉验证折数
        output_dir: Artifact 输出目录
        use_phase1_config: 是否使用 Phase 1 配置（重采样+标准化）
    
    Returns:
        str: 生成的 Artifact 文件路径
    
    Examples:
        >>> features = ['HMDB0000001', 'HMDB0000002']
        >>> artifact_path = export_winner_cv_predictions(
        ...     'data/cleaned.csv', 'Group', features, 'LightGBM'
        ... )
        >>> print(f"Artifact saved to: {artifact_path}")
    """
    print("\n" + "="*80)
    print("Phase 2: Exporting Winner Panel CV Predictions (OOF)")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 过滤出存在的特征
    valid_features = [f for f in winner_features if f in df.columns]
    
    if not valid_features:
        print("Warning: No valid features found. Skipping artifact export.")
        return None
    
    print(f"Data loaded: {df.shape}")
    print(f"Winner Panel features: {len(valid_features)}")
    print(f"Champion model: {champion_model_family}")
    print(f"K-folds: {k_folds}")
    print(f"Use Phase 1 config: {use_phase1_config}")
    
    # 提取特征和标签
    X = df[valid_features].values
    y = df[target_column].values
    
    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # ========================================================================
    # 获取模型（使用 Phase 1 配置）
    # ========================================================================
    if use_phase1_config:
        # 导入 Phase 1 配置
        from src.tools.analysis.pareto_evaluator import _get_dynamic_classifier, _get_resampling_strategy
        
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            HAS_IMBLEARN = True
        except ImportError:
            HAS_IMBLEARN = False
        
        clf = _get_dynamic_classifier(champion_model_family)
        resampler = _get_resampling_strategy('data/autogluon_training_results.json')
        
        # 构建 Pipeline
        if resampler is not None and HAS_IMBLEARN:
            if isinstance(resampler, tuple) and resampler[0] == 'smote_nearmiss':
                _, smote, nearmiss = resampler
                from imblearn.pipeline import Pipeline as ImbPipeline
                model = ImbPipeline([
                    ('smote', smote),
                    ('nearmiss', nearmiss),
                    ('classifier', clf)
                ])
            else:
                from imblearn.pipeline import Pipeline as ImbPipeline
                model = ImbPipeline([
                    ('resampler', resampler),
                    ('classifier', clf)
                ])
        else:
            model = clf
    else:
        # 使用简单的 LightGBM
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=42, verbose=-1)
    
    print(f"\nModel: {model.__class__.__name__}")
    
    # ========================================================================
    # 使用 cross_val_predict 获取 OOF 预测概率
    # ========================================================================
    print(f"\nComputing OOF predictions with {k_folds}-fold CV...")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 存储每个样本的预测概率和所属 fold
    all_predictions = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"  Fold {fold_idx + 1}/{k_folds}...", end=' ')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测概率
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # 存储结果
        for i, test_sample_idx in enumerate(test_idx):
            all_predictions.append({
                'sample_id': test_sample_idx,
                'true_label': y[test_sample_idx],
                'pred_proba_class0': y_pred_proba[i, 0],
                'pred_proba_class1': y_pred_proba[i, 1],
                'fold': fold_idx + 1
            })
        
        print(f"Done ({len(test_idx)} samples)")
    
    # ========================================================================
    # 保存为 CSV
    # ========================================================================
    artifact_path = os.path.join(output_dir, 'winner_cv_predictions.csv')
    
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.sort_values('sample_id').reset_index(drop=True)
    predictions_df.to_csv(artifact_path, index=False)
    
    print(f"\n✅ Artifact saved to: {artifact_path}")
    print(f"   Total samples: {len(predictions_df)}")
    print(f"   File size: {os.path.getsize(artifact_path) / 1024:.2f} KB")
    print("="*80 + "\n")
    
    return artifact_path


def export_winner_shap_artifacts(
    data_path: str,
    target_column: str,
    winner_features: List[str],
    champion_model_family: str,
    output_dir: str = 'output/artifacts',
    use_phase1_config: bool = True
) -> str:
    """
    导出 Winner Panel 的 SHAP values
    
    在 Phase 2 选出 Winner Panel 后调用，在全量数据上训练模型并计算 SHAP values。
    Phase 3 的 plot_shap 将直接读取这些 SHAP values。
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名称
        winner_features: Winner Panel 特征列表
        champion_model_family: 冠军模型家族
        output_dir: Artifact 输出目录
        use_phase1_config: 是否使用 Phase 1 配置
    
    Returns:
        str: 生成的 Artifact 文件路径
    
    Examples:
        >>> features = ['HMDB0000001', 'HMDB0000002']
        >>> artifact_path = export_winner_shap_artifacts(
        ...     'data/cleaned.csv', 'Group', features, 'LightGBM'
        ... )
        >>> print(f"Artifact saved to: {artifact_path}")
    """
    print("\n" + "="*80)
    print("Phase 2: Exporting Winner Panel SHAP Artifacts")
    print("="*80)
    
    try:
        import shap
    except ImportError:
        print("Warning: SHAP not installed. Skipping SHAP artifact export.")
        print("Install with: pip install shap")
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 过滤出存在的特征
    valid_features = [f for f in winner_features if f in df.columns]
    
    if not valid_features:
        print("Warning: No valid features found. Skipping artifact export.")
        return None
    
    print(f"Data loaded: {df.shape}")
    print(f"Winner Panel features: {len(valid_features)}")
    print(f"Champion model: {champion_model_family}")
    
    # 提取特征和标签
    X = df[valid_features].values
    y = df[target_column].values
    
    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ========================================================================
    # 训练模型
    # ========================================================================
    if use_phase1_config:
        from src.tools.analysis.pareto_evaluator import _get_dynamic_classifier
        model = _get_dynamic_classifier(champion_model_family)
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=42, verbose=-1)
    
    print(f"\nTraining model on full dataset...")
    model.fit(X_scaled, y_encoded)
    print("  Model trained successfully")
    
    # ========================================================================
    # 计算 SHAP values
    # ========================================================================
    print(f"\nComputing SHAP values...")
    
    # 使用 TreeExplainer（适用于树模型）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # 如果是二分类，shap_values 是一个列表，取第二个类别的 SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print(f"  SHAP values shape: {shap_values.shape}")
    
    # ========================================================================
    # 保存为 pickle
    # ========================================================================
    artifact_path = os.path.join(output_dir, 'winner_shap_artifacts.pkl')
    
    shap_artifact = {
        'shap_values': shap_values,
        'feature_names': valid_features,
        'base_value': explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
        'data': X_scaled,
        'model_type': champion_model_family
    }
    
    with open(artifact_path, 'wb') as f:
        pickle.dump(shap_artifact, f)
    
    print(f"\n✅ Artifact saved to: {artifact_path}")
    print(f"   File size: {os.path.getsize(artifact_path) / 1024:.2f} KB")
    print("="*80 + "\n")
    
    return artifact_path


# ============================================================================
# Artifact Reading Functions (Artifact-Based Architecture)
# ============================================================================

def read_phase1_panel_scores(
    artifact_dir: str = 'output/artifacts',
    artifact_filename: str = 'phase1_panel_scores.json'
) -> Optional[Dict[str, Any]]:
    """
    从 artifact 读取 Phase1 panel scores
    
    Args:
        artifact_dir: Artifact 目录
        artifact_filename: Artifact 文件名
    
    Returns:
        Dict: Phase1 panel scores artifact，如果不存在则返回 None
    
    Example:
        >>> scores = read_phase1_panel_scores()
        >>> if scores:
        ...     print(f"Phase1 scores: {scores['scores']}")
        ... else:
        ...     print("Artifact not found, will recompute")
    """
    artifact_path = os.path.join(artifact_dir, artifact_filename)
    
    if not os.path.exists(artifact_path):
        print(f"⏭️  Phase1 panel scores artifact not found: {artifact_path}")
        return None
    
    try:
        with open(artifact_path, 'r', encoding='utf-8') as f:
            artifact = json.load(f)
        
        print(f"✅ Phase1 panel scores artifact loaded: {artifact_path}")
        print(f"   Panel: {artifact.get('panel_name')}")
        print(f"   Features: {len(artifact.get('selected_features', []))} features")
        print(f"   Scores: {artifact.get('scores', {})}")
        
        return artifact
    except Exception as e:
        print(f"⚠️  Failed to read Phase1 panel scores artifact: {e}")
        return None


def read_phase2_winner_scores(
    artifact_dir: str = 'output/artifacts',
    artifact_filename: str = 'phase2_winner_scores.json'
) -> Optional[Dict[str, Any]]:
    """
    从 artifact 读取 Phase2 winner scores
    
    Args:
        artifact_dir: Artifact 目录
        artifact_filename: Artifact 文件名
    
    Returns:
        Dict: Phase2 winner scores artifact，如果不存在则返回 None
    """
    artifact_path = os.path.join(artifact_dir, artifact_filename)
    
    if not os.path.exists(artifact_path):
        print(f"⏭️  Phase2 winner scores artifact not found: {artifact_path}")
        return None
    
    try:
        with open(artifact_path, 'r', encoding='utf-8') as f:
            artifact = json.load(f)
        
        print(f"✅ Phase2 winner scores artifact loaded: {artifact_path}")
        print(f"   Panel: {artifact.get('panel_name')}")
        print(f"   Features: {len(artifact.get('selected_features', []))} features")
        print(f"   Scores: {artifact.get('scores', {})}")
        
        return artifact
    except Exception as e:
        print(f"⚠️  Failed to read Phase2 winner scores artifact: {e}")
        return None


def read_phase2_bio_debug(
    artifact_dir: str = 'output/artifacts',
    artifact_filename: str = 'phase2_bio_debug.json'
) -> Optional[Dict[str, Any]]:
    """
    从 artifact 读取 Phase2 biological debug artifact

    Args:
        artifact_dir: Artifact 目录
        artifact_filename: Artifact 文件名

    Returns:
        Dict: Phase2 biological debug artifact，如果不存在则返回 None
    """
    artifact_path = os.path.join(artifact_dir, artifact_filename)

    if not os.path.exists(artifact_path):
        print(f"⏭️  Phase2 biological debug artifact not found: {artifact_path}")
        return None

    try:
        with open(artifact_path, 'r', encoding='utf-8') as f:
            artifact = json.load(f)

        print(f"✅ Phase2 biological debug artifact loaded: {artifact_path}")
        print(f"   Panel: {artifact.get('panel_name')}")
        print(f"   Bio score version: {artifact.get('bio_score_version')}")
        print(f"   Protected anchors: {artifact.get('n_protected_anchor_features', 0)}")

        return artifact
    except Exception as e:
        print(f"⚠️  Failed to read Phase2 biological debug artifact: {e}")
        return None


def read_phase2_bio_calibration(
    artifact_dir: str = 'output/artifacts',
    artifact_filename: str = 'phase2_bio_calibration.json'
) -> Optional[Dict[str, Any]]:
    """
    从 artifact 读取 Phase2 biological calibration artifact

    Args:
        artifact_dir: Artifact 目录
        artifact_filename: Artifact 文件名

    Returns:
        Dict: Phase2 biological calibration artifact，如果不存在则返回 None
    """
    artifact_path = os.path.join(artifact_dir, artifact_filename)

    if not os.path.exists(artifact_path):
        print(f"⏭️  Phase2 biological calibration artifact not found: {artifact_path}")
        return None

    try:
        with open(artifact_path, 'r', encoding='utf-8') as f:
            artifact = json.load(f)
        sample_summary = artifact.get('sample_summary') if isinstance(artifact.get('sample_summary'), dict) else {}
        winner_comparison = artifact.get('winner_comparison') if isinstance(artifact.get('winner_comparison'), dict) else {}

        print(f"✅ Phase2 biological calibration artifact loaded: {artifact_path}")
        print(f"   Disease: {artifact.get('disease_name')}")
        print(f"   Sampled candidates: {sample_summary.get('sampled_candidate_count', 0)}")
        print(f"   Winner changed: {winner_comparison.get('winner_changed', False)}")

        return artifact
    except Exception as e:
        print(f"⚠️  Failed to read Phase2 biological calibration artifact: {e}")
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def list_artifacts(artifact_dir: str = 'output/artifacts') -> Dict[str, Any]:
    """
    列出所有可用的 Artifacts
    
    Args:
        artifact_dir: Artifact 目录
    
    Returns:
        Dict: Artifact 信息字典
    """
    artifacts = {}
    
    if not os.path.exists(artifact_dir):
        return artifacts
    
    # Phase 1 artifacts
    phase1_path = os.path.join(artifact_dir, 'phase1_stats_artifacts.npz')
    if os.path.exists(phase1_path):
        artifacts['phase1_stats'] = {
            'path': phase1_path,
            'size_kb': os.path.getsize(phase1_path) / 1024,
            'mtime': os.path.getmtime(phase1_path)
        }
    
    # Phase 2 artifacts
    winner_cv_path = os.path.join(artifact_dir, 'winner_cv_predictions.csv')
    if os.path.exists(winner_cv_path):
        artifacts['winner_cv_predictions'] = {
            'path': winner_cv_path,
            'size_kb': os.path.getsize(winner_cv_path) / 1024,
            'mtime': os.path.getmtime(winner_cv_path)
        }
    
    winner_shap_path = os.path.join(artifact_dir, 'winner_shap_artifacts.pkl')
    if os.path.exists(winner_shap_path):
        artifacts['winner_shap'] = {
            'path': winner_shap_path,
            'size_kb': os.path.getsize(winner_shap_path) / 1024,
            'mtime': os.path.getmtime(winner_shap_path)
        }

    winner_bio_debug_path = os.path.join(artifact_dir, 'phase2_bio_debug.json')
    if os.path.exists(winner_bio_debug_path):
        artifacts['phase2_bio_debug'] = {
            'path': winner_bio_debug_path,
            'size_kb': os.path.getsize(winner_bio_debug_path) / 1024,
            'mtime': os.path.getmtime(winner_bio_debug_path)
        }

    winner_bio_calibration_path = os.path.join(artifact_dir, 'phase2_bio_calibration.json')
    if os.path.exists(winner_bio_calibration_path):
        artifacts['phase2_bio_calibration'] = {
            'path': winner_bio_calibration_path,
            'size_kb': os.path.getsize(winner_bio_calibration_path) / 1024,
            'mtime': os.path.getmtime(winner_bio_calibration_path)
        }

    winner_contrib_path = os.path.join(artifact_dir, 'phase2_winner_feature_contributions.csv')
    if os.path.exists(winner_contrib_path):
        artifacts['phase2_winner_feature_contributions'] = {
            'path': winner_contrib_path,
            'size_kb': os.path.getsize(winner_contrib_path) / 1024,
            'mtime': os.path.getmtime(winner_contrib_path)
        }

    winner_contrib_summary_path = os.path.join(artifact_dir, 'phase2_winner_feature_contributions_summary.json')
    if os.path.exists(winner_contrib_summary_path):
        artifacts['phase2_winner_feature_contributions_summary'] = {
            'path': winner_contrib_summary_path,
            'size_kb': os.path.getsize(winner_contrib_summary_path) / 1024,
            'mtime': os.path.getmtime(winner_contrib_summary_path)
        }
    
    return artifacts


if __name__ == '__main__':
    print("="*80)
    print("Artifact Exporters Module")
    print("="*80)
    print()
    print("This module provides functions to export artifacts from Phase 1 and Phase 2.")
    print("Phase 3 will read these artifacts instead of recomputing predictions.")
    print()
    print("Available functions:")
    print("  1. export_phase1_stats_artifacts() - Export PCA/PLS-DA coordinates")
    print("  2. export_winner_cv_predictions() - Export OOF predictions")
    print("  3. export_winner_shap_artifacts() - Export SHAP values")
    print("  4. list_artifacts() - List all available artifacts")
    print()
    print("="*80)
