"""Shared helpers for resolving and rebuilding Phase 2 report artifacts."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.phase2.ptot_engine import calculate_topsis_scores
from src.tools.analysis.f_bio_v2 import build_bio_context, calculate_f_bio_v2
from src.tools.analysis.pareto_evaluator import (
    calculate_f_corr,
    calculate_f_cost,
    calculate_f_perf,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _normalize_features(features: Sequence[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for feature in features:
        value = str(feature or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _feature_key(features: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(_normalize_features(features)))


def _feature_signature(features: Iterable[str]) -> str:
    return " | ".join(_normalize_features(features))


def _resolve_search_details(search_summary: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(search_summary.get("search_details"), dict):
        return search_summary["search_details"]
    final_result = search_summary.get("final_result", {})
    if isinstance(final_result, dict) and isinstance(final_result.get("search_details"), dict):
        return final_result["search_details"]
    prior_anchor_union = search_summary.get("prior_anchor_union_evaluation", {})
    phase1_eval = prior_anchor_union.get("phase1_model_evaluation", {})
    if isinstance(phase1_eval, dict) and isinstance(phase1_eval.get("search_details"), dict):
        return phase1_eval["search_details"]
    return {}


def _project_aliases(path_str: str) -> List[Path]:
    value = str(path_str or "").strip()
    if not value:
        return []

    candidates = [Path(value)]
    if not os.path.isabs(value):
        candidates.append(PROJECT_ROOT / value)

    aliases: List[Path] = []
    for candidate in candidates:
        aliases.append(candidate)
        as_posix = str(candidate)

    seen: set[str] = set()
    unique_aliases: List[Path] = []
    for alias in aliases:
        key = str(alias)
        if key in seen:
            continue
        seen.add(key)
        unique_aliases.append(alias)
    return unique_aliases


def _resolve_existing_path(*candidates: Optional[str]) -> Path:
    for candidate in candidates:
        for alias in _project_aliases(candidate or ""):
            if alias.exists():
                return alias
    raise FileNotFoundError(f"Unable to resolve an existing path from candidates: {candidates}")


def _load_feature_provenance_index(provenance_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in provenance_payload.get("features", []):
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature", "") or "").strip()
        standardized = str(row.get("standardized_name", "") or "").strip()
        if feature:
            index[feature] = row
        if standardized:
            index.setdefault(standardized, row)
    return index


def _choose_primary(values: Sequence[str]) -> str:
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _load_taxonomy_map(taxonomy_path: Path) -> Dict[str, Dict[str, Any]]:
    payload = _read_json(taxonomy_path)
    if not payload:
        return {}
    first_value = next(iter(payload.values()))
    if isinstance(first_value, dict):
        return {str(k).upper(): v for k, v in payload.items()}
    return {}


def _extract_search_index(search_summary: Dict[str, Any]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    index: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    search_details = _resolve_search_details(search_summary)
    layers = search_details.get("layers", []) or search_summary.get("layers", [])
    for layer in layers:
        depth = int(layer.get("depth", 0) or 0)
        for node in layer.get("candidate_trace", []) or []:
            features = _normalize_features(node.get("features", []) or [])
            if not features:
                continue
            key = _feature_key(features)
            candidate = {
                "depth": depth,
                "status": str(node.get("status", "") or ""),
                "perf": float(node.get("perf", 0.0) or 0.0),
                "bio": float(node.get("bio", 0.0) or 0.0),
                "corr": float(node.get("corr", 0.0) or 0.0),
                "cost": float(node.get("cost", 0.0) or 0.0),
                "feature_signature": _feature_signature(features),
            }
            existing = index.get(key)
            if existing is None or candidate["perf"] > existing["perf"]:
                index[key] = candidate
    return index


def _normalize_biomarker_key(value: str) -> str:
    return str(value or "").strip().upper()


def _build_confirmed_biomarker_index(phase0_output: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for item in phase0_output.get("confirmed_biomarkers", []) or []:
        if not isinstance(item, dict):
            continue
        biomarker_id = _normalize_biomarker_key(item.get("id", ""))
        biomarker_name = _normalize_biomarker_key(item.get("name", ""))
        if biomarker_id:
            index[biomarker_id] = item
        if biomarker_name:
            index[biomarker_name] = item
    return index


def _extract_prior_value(payload: Dict[str, Any]) -> float:
    for key in ("bio_prior_norm", "confidence_score", "bio_prior_raw"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if key == "bio_prior_norm":
            return max(0.0, min(1.0, numeric))
        if numeric <= 1.0:
            return max(0.0, min(1.0, numeric))
        return max(0.0, min(1.0, numeric / 10.0))
    return 0.0


def _enrich_support_payload(
    *,
    feature: str,
    support_payload: Dict[str, Any],
    provenance_row: Dict[str, Any],
    protected_anchor_features: set[str],
    confirmed_biomarker_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    enriched = {
        "direct_prior": float(support_payload.get("direct_prior", 0.0) or 0.0),
        "disease_pathway_align": float(support_payload.get("disease_pathway_align", 0.0) or 0.0),
        "anchor_link": float(support_payload.get("anchor_link", 0.0) or 0.0),
        "coverage_gain": float(support_payload.get("coverage_gain", 0.0) or 0.0),
        "support_score": float(support_payload.get("support_score", 0.0) or 0.0),
    }
    candidate_keys = {
        _normalize_biomarker_key(feature),
        _normalize_biomarker_key(provenance_row.get("display_name", "")),
        _normalize_biomarker_key(provenance_row.get("standardized_name", "")),
    }
    for hmdb_id in provenance_row.get("mapped_hmdb_ids", []) or []:
        candidate_keys.add(_normalize_biomarker_key(hmdb_id))
    matched_biomarker = next(
        (confirmed_biomarker_index[key] for key in candidate_keys if key and key in confirmed_biomarker_index),
        {},
    )
    if enriched["direct_prior"] <= 0.0 and matched_biomarker:
        enriched["direct_prior"] = _extract_prior_value(matched_biomarker)
    if feature in protected_anchor_features and enriched["anchor_link"] <= 0.0:
        enriched["anchor_link"] = 1.0
    if enriched["support_score"] <= 0.0:
        enriched["support_score"] = max(
            enriched["direct_prior"] * 0.5,
            enriched["anchor_link"] * 0.4,
            enriched["disease_pathway_align"] * 0.35,
            enriched["coverage_gain"] * 0.3,
        )
    return enriched


def _repair_topsis(
    search_summary: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    search_details = _resolve_search_details(search_summary)
    layers = search_details.get("layers", []) or search_summary.get("layers", [])
    topsis_weights = search_details.get("topsis_weights", {}) or {}

    layer_rows: List[Dict[str, Any]] = []
    global_rank_inputs: List[Dict[str, Any]] = []

    for layer in layers:
        depth = int(layer.get("depth", 0) or 0)
        trace = layer.get("candidate_trace", []) or []
        epsilon_candidates = [
            {
                "features": _normalize_features(node.get("features", []) or []),
                "perf": float(node.get("perf", 0.0) or 0.0),
                "bio": float(node.get("bio", 0.0) or 0.0),
                "corr": float(node.get("corr", 0.0) or 0.0),
                "cost": float(node.get("cost", 0.0) or 0.0),
                "status": str(node.get("status", "") or ""),
            }
            for node in trace
            if _boolish(node.get("in_epsilon_feasible_front", False))
        ]

        topsis_source = "epsilon_feasible_front"
        if not epsilon_candidates:
            epsilon_candidates = [
                {
                    "features": _normalize_features(node.get("features", []) or []),
                    "perf": float(node.get("perf", 0.0) or 0.0),
                    "bio": float(node.get("bio", 0.0) or 0.0),
                    "corr": float(node.get("corr", 0.0) or 0.0),
                    "cost": float(node.get("cost", 0.0) or 0.0),
                    "status": str(node.get("status", "") or ""),
                }
                for node in trace
                if _boolish(node.get("in_pareto_front", False))
            ]
            topsis_source = "pareto_fallback"

        ranked = calculate_topsis_scores(epsilon_candidates, weights=topsis_weights) if epsilon_candidates else []
        ranked_index = {_feature_key(item.get("features", [])): item for item in ranked}

        for node in trace:
            features = _normalize_features(node.get("features", []) or [])
            key = _feature_key(features)
            repaired = ranked_index.get(key, {})
            row = {
                "depth": depth,
                "feature_signature": _feature_signature(features),
                "status": str(node.get("status", "") or ""),
                "perf": float(node.get("perf", 0.0) or 0.0),
                "bio": float(node.get("bio", 0.0) or 0.0),
                "corr": float(node.get("corr", 0.0) or 0.0),
                "cost": float(node.get("cost", 0.0) or 0.0),
                "in_pareto_front": _boolish(node.get("in_pareto_front", False)),
                "in_epsilon_feasible_front": _boolish(node.get("in_epsilon_feasible_front", False)),
                "selected_beam": _boolish(node.get("selected_beam", False)),
                "topsis_score_repaired": repaired.get("topsis_score"),
                "topsis_rank_repaired": repaired.get("topsis_rank"),
                "distance_to_ideal_best_repaired": repaired.get("distance_to_ideal_best"),
                "distance_to_ideal_worst_repaired": repaired.get("distance_to_ideal_worst"),
                "topsis_source": topsis_source if repaired else "not_ranked",
            }
            layer_rows.append(row)

        global_rank_inputs.extend(
            [
                {
                    "depth": depth,
                    "features": candidate.get("features", []),
                    "perf": float(candidate.get("perf", 0.0) or 0.0),
                    "bio": float(candidate.get("bio", 0.0) or 0.0),
                    "corr": float(candidate.get("corr", 0.0) or 0.0),
                    "cost": float(candidate.get("cost", 0.0) or 0.0),
                }
                for candidate in epsilon_candidates
            ]
        )

    global_ranked = calculate_topsis_scores(global_rank_inputs, weights=topsis_weights) if global_rank_inputs else []
    global_rows = [
        {
            "depth": int(row.get("depth", 0) or 0),
            "feature_signature": _feature_signature(row.get("features", [])),
            "perf": float(row.get("perf", 0.0) or 0.0),
            "bio": float(row.get("bio", 0.0) or 0.0),
            "corr": float(row.get("corr", 0.0) or 0.0),
            "cost": float(row.get("cost", 0.0) or 0.0),
            "topsis_score_repaired": float(row.get("topsis_score", 0.0) or 0.0),
            "topsis_rank_repaired": int(row.get("topsis_rank", 0) or 0),
        }
        for row in global_ranked
    ]

    layer_csv = output_dir / "phase2_topsis_repaired_candidates.csv"
    global_csv = output_dir / "phase2_topsis_global.csv"
    pd.DataFrame(layer_rows).to_csv(layer_csv, index=False)
    pd.DataFrame(global_rows).to_csv(global_csv, index=False)

    summary = {
        "topsis_weights": topsis_weights,
        "layer_candidate_count": len(layer_rows),
        "global_ranked_candidate_count": len(global_rows),
        "layer_csv": str(layer_csv),
        "global_csv": str(global_csv),
    }
    _write_json(output_dir / "phase2_topsis_repaired.json", summary)
    return summary


def _build_feature_taxonomy(
    provenance_payload: Dict[str, Any],
    taxonomy_map: Dict[str, Dict[str, Any]],
    winner_features: Sequence[str],
    output_dir: Path,
) -> Dict[str, Any]:
    winner_set = set(_normalize_features(winner_features))
    rows: List[Dict[str, Any]] = []
    for row in provenance_payload.get("features", []) or []:
        if not isinstance(row, dict):
            continue
        feature = str(row.get("feature", "") or "").strip()
        hmdb_ids = [str(x).upper().strip() for x in (row.get("mapped_hmdb_ids") or []) if str(x).strip()]
        mapped_classes = [
            str((taxonomy_map.get(hmdb_id) or {}).get("class", "") or "").strip()
            for hmdb_id in hmdb_ids
        ]
        mapped_subclasses = [
            str((taxonomy_map.get(hmdb_id) or {}).get("sub_class", "") or "").strip()
            for hmdb_id in hmdb_ids
        ]
        taxonomy_class_set = sorted({value for value in mapped_classes if value})
        taxonomy_subclass_set = sorted({value for value in mapped_subclasses if value})

        rows.append(
            {
                "feature": feature,
                "display_name": str(row.get("display_name", "") or feature),
                "origin_type": str(row.get("origin_type", "") or ""),
                "origin_subtype": str(row.get("origin_subtype", "") or ""),
                "is_engineered": bool(row.get("is_engineered", False)),
                "is_prior_supported": bool(row.get("is_prior_supported", False)),
                "mapped_hmdb_ids": " | ".join(hmdb_ids),
                "mapped_metabolites": " | ".join(str(x) for x in (row.get("mapped_metabolites") or []) if str(x).strip()),
                "mapped_pathways": " | ".join(str(x) for x in (row.get("mapped_pathways") or []) if str(x).strip()),
                "taxonomy_member_count": len(hmdb_ids),
                "taxonomy_class_set": " | ".join(taxonomy_class_set),
                "taxonomy_subclass_set": " | ".join(taxonomy_subclass_set),
                "primary_class": _choose_primary(taxonomy_class_set) or str(row.get("origin_type", "") or ""),
                "primary_sub_class": _choose_primary(taxonomy_subclass_set),
                "feature_category": _choose_primary(taxonomy_class_set) or str(row.get("origin_type", "") or "unknown"),
                "is_phase2_winner": feature in winner_set,
            }
        )

    csv_path = output_dir / "phase2_feature_taxonomy.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "feature_count": len(rows),
        "winner_feature_count": sum(1 for row in rows if row["is_phase2_winner"]),
        "taxonomy_csv": str(csv_path),
    }
    _write_json(output_dir / "phase2_feature_taxonomy_summary.json", summary)
    return summary


def _extract_full_support_map(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    support_map: Dict[str, Dict[str, Any]] = {}
    bio_debug = payload.get("bio_debug", payload) if isinstance(payload, dict) else {}
    for item in bio_debug.get("p0_items", []) or []:
        if not isinstance(item, dict):
            continue
        feature_name = str(item.get("feature_name", "") or "").strip()
        if not feature_name:
            continue
        member_supports = item.get("member_supports", {}) or {}
        flattened: Dict[str, Any] = {}
        if isinstance(member_supports, dict):
            candidates = [value for value in member_supports.values() if isinstance(value, dict)]
            if candidates:
                flattened = max(
                    candidates,
                    key=lambda value: float(value.get("support_score", 0.0) or 0.0)
                    + float(value.get("direct_prior", 0.0) or 0.0)
                    + float(value.get("disease_pathway_align", 0.0) or 0.0)
                    + float(value.get("anchor_link", 0.0) or 0.0)
                    + float(value.get("coverage_gain", 0.0) or 0.0),
                )
        support_map[feature_name] = {
            "direct_prior": float(flattened.get("direct_prior", item.get("direct_prior", 0.0)) or 0.0),
            "disease_pathway_align": float(
                flattened.get("disease_pathway_align", item.get("disease_pathway_align", 0.0)) or 0.0
            ),
            "anchor_link": float(flattened.get("anchor_link", item.get("anchor_link", 0.0)) or 0.0),
            "coverage_gain": float(flattened.get("coverage_gain", item.get("coverage_gain", 0.0)) or 0.0),
            "support_score": float(
                flattened.get(
                    "support_score",
                    item.get("aggregate_member_support", item.get("p0_v2", 0.0)),
                )
                or 0.0
            ),
        }
    return support_map


def _merge_support_maps(
    preferred_map: Dict[str, Dict[str, Any]],
    fallback_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for feature in sorted(set(preferred_map) | set(fallback_map)):
        preferred = preferred_map.get(feature, {}) or {}
        fallback = fallback_map.get(feature, {}) or {}
        merged[feature] = {
            # Preserve legitimate recomputed zeros instead of falling back to stale
            # non-zero values from historical artifacts.
            key: float(preferred[key] if key in preferred else fallback.get(key, 0.0) or 0.0)
            for key in ("direct_prior", "disease_pathway_align", "anchor_link", "coverage_gain", "support_score")
        }
    return merged


def _evaluate_panel(
    *,
    features: Sequence[str],
    data_path: Path,
    target_column: str,
    champion_model_family: str,
    ag_results_path: Path,
    metric: str,
    taxonomy_map: Dict[str, Dict[str, Any]],
    pathway_map: Optional[Dict[str, Any]],
    bio_context: Any,
) -> Dict[str, Any]:
    normalized_features = _normalize_features(features)
    perf_score, predictions = calculate_f_perf(
        data_path=str(data_path),
        target_column=target_column,
        features_list=normalized_features,
        champion_model_family=champion_model_family,
        k_folds=5,
        ag_results_path=str(ag_results_path),
        metric=metric,
        use_phase1_config=True,
        return_predictions=True,
    )
    bio_debug = calculate_f_bio_v2(
        normalized_features,
        bio_context=bio_context,
        return_debug=True,
    )
    corr_score = calculate_f_corr(str(data_path), normalized_features)
    cost_score = calculate_f_cost(
        features_list=normalized_features,
        taxonomy_map=taxonomy_map,
        pathway_map=pathway_map,
        n_max=20,
        k=2.0,
        data_path=str(data_path),
        aggregation_discount=True,
    )
    return {
        "features": normalized_features,
        "perf": float(perf_score),
        "bio": float(bio_debug.get("f_bio_v2", 0.0) or 0.0),
        "corr": float(corr_score),
        "cost": float(cost_score),
        "independence": 1.0 - float(corr_score),
        "bio_debug": bio_debug,
        "cv_predictions": predictions,
    }


def _repair_winner_feature_contributions(
    *,
    winner_scores: Dict[str, Any],
    search_summary: Dict[str, Any],
    phase0_output: Dict[str, Any],
    provenance_payload: Dict[str, Any],
    taxonomy_map: Dict[str, Dict[str, Any]],
    data_path: Path,
    ag_results_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    target_column = str(winner_scores.get("target_column", "group") or "group")
    winner_features = _normalize_features(winner_scores.get("selected_features", []) or [])
    metric = str(winner_scores.get("metric", "roc_auc") or "roc_auc")
    champion_model_family = str(winner_scores.get("selected_model", "") or "").strip()
    if not champion_model_family:
        champion_model_family = str(_read_json(ag_results_path).get("best_model", "RandomForest") or "RandomForest")
    champion_model_family = champion_model_family.split(" (", 1)[0].strip()

    feature_pool = provenance_payload.get("selected_features", []) or winner_features
    provenance_index = _load_feature_provenance_index(provenance_payload)
    bio_context = build_bio_context(
        disease_name=str(phase0_output.get("disease_name", "") or ""),
        feature_pool=_normalize_features(feature_pool),
        taxonomy_map=taxonomy_map,
        phase0_output=phase0_output,
    )
    if provenance_index:
        bio_context.feature_provenance_index = dict(provenance_index)

    full_eval = _evaluate_panel(
        features=winner_features,
        data_path=data_path,
        target_column=target_column,
        champion_model_family=champion_model_family,
        ag_results_path=ag_results_path,
        metric=metric,
        taxonomy_map=taxonomy_map,
        pathway_map=bio_context.pathway_map,
        bio_context=bio_context,
    )
    stored_scores = winner_scores.get("scores", {}) or {}
    stored_support_map = _extract_full_support_map(winner_scores)
    recomputed_support_map = _extract_full_support_map(full_eval.get("bio_debug", {}) or {})
    support_map = _merge_support_maps(recomputed_support_map, stored_support_map)
    search_index = _extract_search_index(search_summary)
    confirmed_biomarker_index = _build_confirmed_biomarker_index(phase0_output)
    protected_anchor_features = set(_normalize_features(winner_scores.get("protected_anchor_features", []) or []))

    rows: List[Dict[str, Any]] = []
    for feature in winner_features:
        subset = [value for value in winner_features if value != feature]
        drop_eval = _evaluate_panel(
            features=subset,
            data_path=data_path,
            target_column=target_column,
            champion_model_family=champion_model_family,
            ag_results_path=ag_results_path,
            metric=metric,
            taxonomy_map=taxonomy_map,
            pathway_map=bio_context.pathway_map,
            bio_context=bio_context,
        )
        support_payload = support_map.get(feature, {})
        provenance_row = provenance_index.get(feature) or {}
        support_payload = _enrich_support_payload(
            feature=feature,
            support_payload=support_payload,
            provenance_row=provenance_row,
            protected_anchor_features=protected_anchor_features,
            confirmed_biomarker_index=confirmed_biomarker_index,
        )
        matched_search = search_index.get(_feature_key(subset), {})

        rows.append(
            {
                "feature": feature,
                "display_name": str(provenance_row.get("display_name", "") or feature),
                "drop1_subset_signature": _feature_signature(subset),
                "drop1_subset_size": len(subset),
                "winner_perf": full_eval["perf"],
                "winner_bio": full_eval["bio"],
                "winner_corr": full_eval["corr"],
                "winner_cost": full_eval["cost"],
                "drop1_perf": drop_eval["perf"],
                "drop1_bio": drop_eval["bio"],
                "drop1_corr": drop_eval["corr"],
                "drop1_cost": drop_eval["cost"],
                "delta_keep_perf": full_eval["perf"] - drop_eval["perf"],
                "delta_keep_bio": full_eval["bio"] - drop_eval["bio"],
                "delta_keep_corr": full_eval["corr"] - drop_eval["corr"],
                "delta_keep_independence": drop_eval["corr"] - full_eval["corr"],
                "extra_cost_of_keeping": full_eval["cost"] - drop_eval["cost"],
                "direct_prior": support_payload.get("direct_prior", 0.0),
                "disease_pathway_align": support_payload.get("disease_pathway_align", 0.0),
                "anchor_link": support_payload.get("anchor_link", 0.0),
                "coverage_gain": support_payload.get("coverage_gain", 0.0),
                "winner_support_score": support_payload.get("support_score", 0.0),
                "matched_search_depth": matched_search.get("depth"),
                "matched_search_status": matched_search.get("status"),
                "matched_search_perf": matched_search.get("perf"),
                "matched_search_bio": matched_search.get("bio"),
                "matched_search_corr": matched_search.get("corr"),
                "matched_search_cost": matched_search.get("cost"),
            }
        )

    csv_path = output_dir / "phase2_winner_feature_contributions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "winner_feature_count": len(rows),
        "contribution_csv": str(csv_path),
        "stored_winner_scores": {
            "f_perf": float(stored_scores.get("f_perf", 0.0) or 0.0),
            "f_bio": float(stored_scores.get("f_bio", 0.0) or 0.0),
            "f_corr": float(stored_scores.get("f_corr", 0.0) or 0.0),
            "f_cost": float(stored_scores.get("f_cost", 0.0) or 0.0),
        },
        "recomputed_winner_scores": {
            "f_perf": full_eval["perf"],
            "f_bio": full_eval["bio"],
            "f_corr": full_eval["corr"],
            "f_cost": full_eval["cost"],
        },
        "winner_score_drift": {
            "f_perf": full_eval["perf"] - float(stored_scores.get("f_perf", 0.0) or 0.0),
            "f_bio": full_eval["bio"] - float(stored_scores.get("f_bio", 0.0) or 0.0),
            "f_corr": full_eval["corr"] - float(stored_scores.get("f_corr", 0.0) or 0.0),
            "f_cost": full_eval["cost"] - float(stored_scores.get("f_cost", 0.0) or 0.0),
        },
    }
    _write_json(output_dir / "phase2_winner_feature_contributions_summary.json", summary)
    return summary


