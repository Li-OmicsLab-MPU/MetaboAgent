"""
Feature Generator Tools for Phase 1

This module provides high-level tools for generating knowledge-driven features.
These tools encapsulate complex logic to prevent LLM from generating incorrect code.
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

# Add src to path for standalone execution
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.tools.domain.reaction_checker import ReactionCheckerTool
from src.tools.domain.taxonomy_group_tool import TaxonomyGroupTool
from src.tools.domain.pathway_scorer import calculate_pathway_scores, PathwayScorer
from src.utils.config_manager import get_config


def _resolve_id_column(df: pd.DataFrame) -> str:
    """Return a stable join key. Create one from row index when no ID column exists."""
    for col in ("id", "ID", "Sample_ID", "sample_id", "SampleID", "sampleid", "ROW_ID", "row_id"):
        if col in df.columns:
            return col
    row_id_col = "__row_id__"
    if row_id_col not in df.columns:
        df[row_id_col] = range(len(df))
    return row_id_col


def _build_output_df(
    base_df: pd.DataFrame,
    feature_payload: Dict[str, pd.Series],
    include_label: bool,
) -> pd.DataFrame:
    """
    Build engineered feature output without leaking synthetic row IDs as features.

    The internal `__row_id__` key is only for merge alignment. When there is no
    real sample-id column, downstream merge logic can fall back to row index, so
    the synthetic key must not be exported into the feature space.
    """
    id_col = _resolve_id_column(base_df)
    label_col = _resolve_label_column(base_df)

    output_df = pd.DataFrame()
    if id_col != "__row_id__":
        output_df[id_col] = base_df[id_col]
    if include_label and label_col:
        output_df[label_col] = base_df[label_col]
    for feature_name, feature_values in feature_payload.items():
        output_df[feature_name] = feature_values
    if output_df.shape[1] == 0:
        # Keep the file readable even when zero engineered features survive. This
        # internal key is excluded downstream from feature space and only serves
        # as a merge/index placeholder.
        output_df[id_col] = base_df[id_col]
    return output_df


def _resolve_label_column(df: pd.DataFrame) -> str:
    """Return the best available label column name, or empty string if absent."""
    for col in ("Group", "group", "target", "Target", "Class", "class", "Label", "label", "Factors", "factors", "FACTOR"):
        if col in df.columns:
            return col
    return ""


def _load_json_file(file_path: str, default: Any) -> Any:
    """Load a JSON file and return a fallback on failure."""
    if not file_path or not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _resolve_ratio_rules(rules_path: Optional[str] = None) -> Dict[str, Any]:
    """Resolve engineered ratio rules from config-driven JSON."""
    if not rules_path:
        try:
            rules_path = get_config().get_phase1_engineered_feature_rules_path()
        except Exception:
            rules_path = ""

    rules_payload = _load_json_file(rules_path, {})
    ratio_rules = rules_payload.get("ratio", {}) if isinstance(rules_payload, dict) else {}
    return {
        "use_log_ratio": bool(ratio_rules.get("use_log_ratio", True)),
        "max_missing_rate": float(ratio_rules.get("max_missing_rate", 0.20)),
        "min_variance": float(ratio_rules.get("min_variance", 1e-8)),
    }


def _resolve_sum_rules(rules_path: Optional[str] = None) -> Dict[str, Any]:
    """Resolve engineered sum rules from config-driven JSON."""
    if not rules_path:
        try:
            rules_path = get_config().get_phase1_engineered_feature_rules_path()
        except Exception:
            rules_path = ""

    rules_payload = _load_json_file(rules_path, {})
    sum_rules = rules_payload.get("sum", {}) if isinstance(rules_payload, dict) else {}
    return {
        "taxonomy_level": str(sum_rules.get("taxonomy_level", "sub_class") or "sub_class"),
        "exclude_labels": [
            str(label or "").strip()
            for label in (sum_rules.get("exclude_labels", []) or [])
            if str(label or "").strip()
        ],
        "min_raw_members": int(sum_rules.get("min_raw_members", 3)),
        "min_informative_members": int(sum_rules.get("min_informative_members", 3)),
        "min_abs_effect": float(sum_rules.get("min_abs_effect", 0.10)),
        "min_direction_consistency": float(sum_rules.get("min_direction_consistency", 0.70)),
        "max_members_per_feature": int(sum_rules.get("max_members_per_feature", 8)),
    }


def _resolve_pathway_rules(rules_path: Optional[str] = None) -> Dict[str, Any]:
    """Resolve engineered pathway rules from config-driven JSON."""
    if not rules_path:
        try:
            rules_path = get_config().get_phase1_engineered_feature_rules_path()
        except Exception:
            rules_path = ""

    rules_payload = _load_json_file(rules_path, {})
    pathway_rules = rules_payload.get("pathway", {}) if isinstance(rules_payload, dict) else {}
    return {
        "source": str(pathway_rules.get("source", "phase0_target_pathways_only") or "phase0_target_pathways_only"),
        "min_observed_members": int(pathway_rules.get("min_observed_members", 4)),
        "min_coverage_ratio": float(pathway_rules.get("min_coverage_ratio", 0.30)),
        "postcheck_auc_gain": float(pathway_rules.get("postcheck_auc_gain", 0.01)),
    }


def _resolve_registry_path(registry_path: Optional[str] = None) -> str:
    """Resolve engineered feature registry output path."""
    if registry_path:
        return registry_path
    try:
        return get_config().get_phase1_engineered_feature_registry_path()
    except Exception:
        return ""


def _load_registry_payload(registry_path: str) -> Dict[str, Any]:
    """Load the engineered feature registry payload or create a fresh shell."""
    default_payload = {
        "schema_version": "phase1.engineered_feature_registry.v1",
        "generated_at": "",
        "features": [],
    }
    payload = _load_json_file(registry_path, default_payload)
    if not isinstance(payload, dict):
        return dict(default_payload)
    features = payload.get("features", [])
    if not isinstance(features, list):
        payload["features"] = []
    return payload


def _merge_registry_entries(registry_path: str, entries: List[Dict[str, Any]]) -> None:
    """Upsert engineered feature records into the shared registry."""
    if not registry_path:
        return

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    payload = _load_registry_payload(registry_path)

    merged: Dict[str, Dict[str, Any]] = {}
    for record in payload.get("features", []):
        feature_name = str(record.get("feature_name", "") or "").strip()
        if feature_name:
            merged[feature_name] = record

    for record in entries:
        feature_name = str(record.get("feature_name", "") or "").strip()
        if feature_name:
            merged[feature_name] = record

    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["features"] = list(merged.values())

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _ensure_registry_file(registry_path: str) -> None:
    """Materialize an empty registry file when no entries are produced."""
    if not registry_path:
        return
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    payload = _load_registry_payload(registry_path)
    if not payload.get("generated_at"):
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _normalize_taxonomy_label(label: str) -> str:
    """Create a stable feature suffix from a taxonomy label."""
    normalized = str(label or "").strip().replace(" ", "_").replace(",", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _compute_directional_effect(
    feature_series: pd.Series,
    label_series: pd.Series,
) -> Optional[float]:
    """
    Compute a robust binary effect size using median difference scaled by global IQR.
    """
    numeric_feature = pd.to_numeric(feature_series, errors="coerce")
    numeric_label = pd.to_numeric(label_series, errors="coerce")
    valid_mask = numeric_feature.notna() & numeric_label.notna()
    if valid_mask.sum() == 0:
        return None

    feature_values = numeric_feature[valid_mask]
    label_values = numeric_label[valid_mask]
    unique_labels = sorted(set(label_values.astype(int).tolist()))
    if unique_labels != [0, 1]:
        return None

    pos_values = feature_values[label_values == 1]
    neg_values = feature_values[label_values == 0]
    if len(pos_values) == 0 or len(neg_values) == 0:
        return None

    iqr = float(feature_values.quantile(0.75) - feature_values.quantile(0.25))
    effect = (float(pos_values.median()) - float(neg_values.median())) / (iqr + 1e-8)
    return effect


def _extract_phase0_target_pathways(phase0_output: Optional[Dict[str, Any]]) -> List[str]:
    """Extract pathway candidates from Phase 0 output payload."""
    if not isinstance(phase0_output, dict):
        return []

    candidate_lists: List[List[Any]] = []

    feature_definitions = phase0_output.get("feature_definitions", {})
    if isinstance(feature_definitions, dict):
        nested_target_pathways = feature_definitions.get("target_pathways", [])
        if isinstance(nested_target_pathways, list):
            candidate_lists.append(nested_target_pathways)

    # Compatibility fallback:
    # some upstream payloads or LLM-reconstructed summaries may flatten the
    # Phase 0 pathway list to a top-level field.
    for fallback_key in ("target_pathways", "enriched_pathways"):
        fallback_values = phase0_output.get(fallback_key, [])
        if isinstance(fallback_values, list):
            candidate_lists.append(fallback_values)

    resolved = []
    for pathway_list in candidate_lists:
        for pathway_name in pathway_list:
            if isinstance(pathway_name, dict):
                pathway_name = pathway_name.get("name") or pathway_name.get("pathway_name")
            clean_name = str(pathway_name or "").strip()
            if clean_name and clean_name not in resolved:
                resolved.append(clean_name)
    return resolved


def generate_reaction_ratios(
    data_path: str,
    output_path: str = 'data/ratio_features_data.csv',
    registry_path: Optional[str] = None,
    rules_path: Optional[str] = None,
    include_label: bool = False,
) -> Dict:
    """
    Generate ratio features based on biochemical reactions.
    
    Only creates ratios for metabolite pairs that are connected by biochemical reactions.
    When the input matrix is already log-transformed, the ratio is represented as
    a stable difference (log-ratio): A - B.
    
    Args:
        data_path: Path to input data CSV file
        output_path: Path to save ratio features (default: data/ratio_features_data.csv)
        registry_path: Optional engineered feature registry output path
        rules_path: Optional engineered feature rules JSON path
        include_label: Whether to keep the target/group column in the output
        
    Returns:
        dict: {
            'status': 'success' or 'error',
            'output_path': path to output file,
            'num_ratios': number of ratio features generated,
            'ratio_names': list of ratio feature names,
            'registry_path': registry output path,
            'message': status message
        }
    """
    try:
        ratio_rules = _resolve_ratio_rules(rules_path=rules_path)
        use_log_ratio = bool(ratio_rules["use_log_ratio"])
        max_missing_rate = float(ratio_rules["max_missing_rate"])
        min_variance = float(ratio_rules["min_variance"])

        # Load data
        base_df = pd.read_csv(data_path)
        print(f"✓ Loaded data: {len(base_df)} rows, {len(base_df.columns)} columns")
        
        id_col = _resolve_id_column(base_df)
        label_col = _resolve_label_column(base_df)

        # Initialize reaction checker
        checker = ReactionCheckerTool()
        
        # Extract HMDB ID columns only.
        hmdb_cols = [col for col in base_df.columns if col.startswith("HMDB")]
        print(f"✓ Found {len(hmdb_cols)} HMDB metabolite columns")

        # Find connected pairs and create ratios
        valid_pairs = checker.find_connected_pairs(hmdb_cols)
        ratio_features = {}
        registry_entries: List[Dict[str, Any]] = []
        skipped_pairs: List[Dict[str, Any]] = []

        for pair in valid_pairs:
            id_a = pair["id_a"]
            id_b = pair["id_b"]

            series_a = pd.to_numeric(base_df[id_a], errors="coerce")
            series_b = pd.to_numeric(base_df[id_b], errors="coerce")

            missing_rate_a = float(series_a.isna().mean())
            missing_rate_b = float(series_b.isna().mean())
            variance_a = float(series_a.var(ddof=0)) if len(series_a) > 1 else 0.0
            variance_b = float(series_b.var(ddof=0)) if len(series_b) > 1 else 0.0

            if (
                missing_rate_a > max_missing_rate
                or missing_rate_b > max_missing_rate
                or pd.isna(variance_a)
                or pd.isna(variance_b)
                or variance_a <= min_variance
                or variance_b <= min_variance
            ):
                skipped_pairs.append(
                    {
                        "id_a": id_a,
                        "id_b": id_b,
                        "reason": "quality_filter",
                        "missing_rate_a": round(missing_rate_a, 4),
                        "missing_rate_b": round(missing_rate_b, 4),
                        "variance_a": variance_a,
                        "variance_b": variance_b,
                    }
                )
                continue

            if use_log_ratio:
                ratio_values = series_a - series_b
                transform_mode = "difference"
            else:
                safe_denominator = series_b.replace(0, 1e-10)
                ratio_values = (series_a / safe_denominator).replace([pd.NA, float("inf"), float("-inf")], 0.0)
                transform_mode = "division"

            if ratio_values.isna().any():
                skipped_pairs.append(
                    {
                        "id_a": id_a,
                        "id_b": id_b,
                        "reason": "nan_after_transform",
                    }
                )
                continue

            ratio_name = f"Ratio_{id_a}_{id_b}"
            ratio_features[ratio_name] = ratio_values
            registry_entries.append(
                {
                    "feature_name": ratio_name,
                    "feature_type": "engineered_ratio",
                    "origin_subtype": "reaction_ratio",
                    "mapped_hmdb_ids": [id_a, id_b],
                    "reaction": {
                        "rc_id": pair.get("rc_id", ""),
                        "reaction_type": pair.get("reaction_type", ""),
                    },
                    "quality_filters": {
                        "use_log_ratio": use_log_ratio,
                        "transform_mode": transform_mode,
                        "max_missing_rate": max_missing_rate,
                        "min_variance": min_variance,
                        "missing_rate_a": round(missing_rate_a, 4),
                        "missing_rate_b": round(missing_rate_b, 4),
                        "variance_a": variance_a,
                        "variance_b": variance_b,
                    },
                }
            )
        
        print(f"✓ Found {len(valid_pairs)} connected HMDB pairs")
        print(f"✓ Generated {len(ratio_features)} ratio features after quality filters")
        if skipped_pairs:
            print(f"✓ Skipped {len(skipped_pairs)} pairs due to quality filters")
        
        # Create output DataFrame with real id/label columns plus engineered ratios.
        output_df = _build_output_df(
            base_df=base_df,
            feature_payload=ratio_features,
            include_label=include_label,
        )

        # Save to file
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(output_df.columns)} columns to {output_path}")

        resolved_registry_path = _resolve_registry_path(registry_path)
        if registry_entries:
            _merge_registry_entries(resolved_registry_path, registry_entries)
            print(f"✓ Updated engineered feature registry: {resolved_registry_path}")
        
        return {
            'status': 'success',
            'output_path': output_path,
            'num_ratios': len(ratio_features),
            'ratio_names': list(ratio_features.keys()),
            'registry_path': resolved_registry_path,
            'n_connected_pairs': len(valid_pairs),
            'n_skipped_pairs': len(skipped_pairs),
            'message': (
                f"Generated {len(ratio_features)} ratio features from "
                f"{len(valid_pairs)} connected pairs"
            ),
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'output_path': None,
            'num_ratios': 0,
            'ratio_names': [],
            'registry_path': _resolve_registry_path(registry_path),
            'message': f"Error: {str(e)}"
        }


def generate_taxonomy_sums(
    data_path: str,
    output_path: str = 'data/taxonomy_sum_features_data.csv',
    registry_path: Optional[str] = None,
    rules_path: Optional[str] = None,
    include_label: bool = False,
) -> Dict:
    """
    Generate sum features based on HMDB taxonomy classification.
    
    Groups metabolites by taxonomy classification and retains only directionally
    consistent burden-style groups.
    
    Args:
        data_path: Path to input data CSV file
        output_path: Path to save sum features (default: data/taxonomy_sum_features_data.csv)
        registry_path: Optional engineered feature registry output path
        rules_path: Optional engineered feature rules JSON path
        include_label: Whether to keep the target/group column in the output
        
    Returns:
        dict: {
            'status': 'success' or 'error',
            'output_path': path to output file,
            'num_sums': number of sum features generated,
            'sum_names': list of sum feature names,
            'registry_path': registry output path,
            'message': status message
        }
    """
    try:
        sum_rules = _resolve_sum_rules(rules_path=rules_path)
        taxonomy_level = sum_rules["taxonomy_level"]
        exclude_labels = {
            str(label).strip().lower() for label in sum_rules["exclude_labels"]
        }
        min_raw_members = int(sum_rules["min_raw_members"])
        min_informative_members = int(sum_rules["min_informative_members"])
        min_abs_effect = float(sum_rules["min_abs_effect"])
        min_direction_consistency = float(sum_rules["min_direction_consistency"])
        max_members_per_feature = int(sum_rules["max_members_per_feature"])

        # Load data
        base_df = pd.read_csv(data_path)
        print(f"✓ Loaded data: {len(base_df)} rows, {len(base_df.columns)} columns")
        
        id_col = _resolve_id_column(base_df)
        label_col = _resolve_label_column(base_df)
        label_series = pd.to_numeric(base_df[label_col], errors="coerce") if label_col else None

        # Initialize taxonomy tool
        tool = TaxonomyGroupTool()
        
        # Extract HMDB ID columns
        hmdb_ids = [col for col in base_df.columns if col.startswith('HMDB')]
        print(f"✓ Found {len(hmdb_ids)} HMDB ID columns")
        
        # Group metabolites by taxonomy
        groups = tool.group_metabolites(hmdb_ids, level=taxonomy_level)
        print(f"✓ Found {len(groups)} taxonomy groups")
        
        # Create controlled burden sum features
        sum_features = {}
        registry_entries: List[Dict[str, Any]] = []
        skipped_groups: List[Dict[str, Any]] = []

        for group_name, metabolite_list in groups.items():
            normalized_group = str(group_name or "").strip()
            if not normalized_group:
                skipped_groups.append({"group_name": group_name, "reason": "empty_label"})
                continue

            if normalized_group.lower() in exclude_labels:
                skipped_groups.append({"group_name": normalized_group, "reason": "excluded_label"})
                continue

            if len(metabolite_list) < min_raw_members:
                skipped_groups.append(
                    {
                        "group_name": normalized_group,
                        "reason": "insufficient_raw_members",
                        "raw_members": len(metabolite_list),
                    }
                )
                continue

            informative_members: List[Dict[str, Any]] = []
            if label_col and label_series is not None:
                for hmdb_id in metabolite_list:
                    effect = _compute_directional_effect(base_df[hmdb_id], label_series)
                    if effect is None or abs(effect) < min_abs_effect:
                        continue
                    informative_members.append(
                        {
                            "hmdb_id": hmdb_id,
                            "effect": float(effect),
                            "direction": "positive" if effect > 0 else "negative",
                        }
                    )
            else:
                informative_members = [
                    {"hmdb_id": hmdb_id, "effect": 0.0, "direction": "positive"}
                    for hmdb_id in metabolite_list
                ]

            if len(informative_members) < min_informative_members:
                skipped_groups.append(
                    {
                        "group_name": normalized_group,
                        "reason": "insufficient_informative_members",
                        "informative_members": len(informative_members),
                    }
                )
                continue

            n_positive = sum(1 for member in informative_members if member["direction"] == "positive")
            n_negative = sum(1 for member in informative_members if member["direction"] == "negative")
            dominant_direction = "positive" if n_positive >= n_negative else "negative"
            direction_consistency = max(n_positive, n_negative) / max(len(informative_members), 1)

            if direction_consistency < min_direction_consistency:
                skipped_groups.append(
                    {
                        "group_name": normalized_group,
                        "reason": "low_direction_consistency",
                        "direction_consistency": round(direction_consistency, 4),
                    }
                )
                continue

            retained_members = [
                member for member in informative_members
                if member["direction"] == dominant_direction
            ]
            retained_members = sorted(
                retained_members,
                key=lambda item: abs(float(item["effect"])),
                reverse=True,
            )[:max_members_per_feature]

            if len(retained_members) < min_informative_members:
                skipped_groups.append(
                    {
                        "group_name": normalized_group,
                        "reason": "insufficient_retained_members",
                        "retained_members": len(retained_members),
                    }
                )
                continue

            selected_hmdb_ids = [member["hmdb_id"] for member in retained_members]
            sum_name = f"Sum_{_normalize_taxonomy_label(normalized_group)}"
            sum_features[sum_name] = base_df[selected_hmdb_ids].sum(axis=1)
            registry_entries.append(
                {
                    "feature_name": sum_name,
                    "feature_type": "engineered_sum",
                    "origin_subtype": "taxonomy_sum",
                    "mapped_hmdb_ids": selected_hmdb_ids,
                    "group_label": normalized_group,
                    "taxonomy_level": taxonomy_level,
                    "candidate_member_hmdb_ids": list(metabolite_list),
                    "selected_member_hmdb_ids": selected_hmdb_ids,
                    "dominant_direction": dominant_direction,
                    "direction_consistency": round(direction_consistency, 4),
                    "member_effects": {
                        member["hmdb_id"]: round(float(member["effect"]), 6)
                        for member in retained_members
                    },
                    "quality_filters": {
                        "min_raw_members": min_raw_members,
                        "min_informative_members": min_informative_members,
                        "min_abs_effect": min_abs_effect,
                        "min_direction_consistency": min_direction_consistency,
                        "max_members_per_feature": max_members_per_feature,
                    },
                }
            )

        print(f"✓ Created {len(sum_features)} controlled sum features")
        if skipped_groups:
            print(f"✓ Skipped {len(skipped_groups)} taxonomy groups after filtering")
        
        # Create output DataFrame with real id/label columns plus controlled sums.
        output_df = _build_output_df(
            base_df=base_df,
            feature_payload=sum_features,
            include_label=include_label,
        )
        
        # Save to file
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(output_df.columns)} columns to {output_path}")

        resolved_registry_path = _resolve_registry_path(registry_path)
        if registry_entries:
            _merge_registry_entries(resolved_registry_path, registry_entries)
            print(f"✓ Updated engineered feature registry: {resolved_registry_path}")
        
        return {
            'status': 'success',
            'output_path': output_path,
            'num_sums': len(sum_features),
            'sum_names': list(sum_features.keys()),
            'registry_path': resolved_registry_path,
            'n_group_candidates': len(groups),
            'n_skipped_groups': len(skipped_groups),
            'message': f"Generated {len(sum_features)} sum features from {len(groups)} groups"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'output_path': None,
            'num_sums': 0,
            'sum_names': [],
            'registry_path': _resolve_registry_path(registry_path),
            'message': f"Error: {str(e)}"
        }


def generate_pathway_scores(
    data_path: str,
    output_path: str = 'data/pathway_scores.csv',
    pathway_names: Optional[List[str]] = None,
    phase0_output: Optional[Dict[str, Any]] = None,
    registry_path: Optional[str] = None,
    rules_path: Optional[str] = None,
    include_label: bool = False,
) -> Dict:
    """
    Generate pathway activity scores with a strict observed-members coverage gate.

    Pathway candidates default to Phase 0 target pathways and are retained only
    if they have enough observed HMDB members in the current dataset.
    """
    try:
        pathway_rules = _resolve_pathway_rules(rules_path=rules_path)
        min_observed_members = int(pathway_rules["min_observed_members"])
        min_coverage_ratio = float(pathway_rules["min_coverage_ratio"])

        base_df = pd.read_csv(data_path)
        print(f"✓ Loaded data: {len(base_df)} rows, {len(base_df.columns)} columns")

        id_col = _resolve_id_column(base_df)
        label_col = _resolve_label_column(base_df)
        dataset_hmdb_ids = {col for col in base_df.columns if col.startswith("HMDB")}
        print(f"✓ Found {len(dataset_hmdb_ids)} HMDB metabolite columns")

        resolved_pathway_names: List[str] = []
        if isinstance(pathway_names, list) and pathway_names:
            for pathway_name in pathway_names:
                clean_name = str(pathway_name or "").strip()
                if clean_name and clean_name not in resolved_pathway_names:
                    resolved_pathway_names.append(clean_name)
        else:
            resolved_pathway_names = _extract_phase0_target_pathways(phase0_output)

        if not resolved_pathway_names:
            print("✓ No pathway candidates available; exporting empty pathway feature file")
            output_df = _build_output_df(
                base_df=base_df,
                feature_payload={},
                include_label=include_label,
            )
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            output_df.to_csv(output_path, index=False)
            resolved_registry_path = _resolve_registry_path(registry_path)
            _ensure_registry_file(resolved_registry_path)
            return {
                "status": "success",
                "output_path": output_path,
                "num_pathways": 0,
                "pathway_names": [],
                "registry_path": resolved_registry_path,
                "n_candidates": 0,
                "n_skipped_pathways": 0,
                "message": "No pathway candidates available from pathway_names or phase0_output",
            }

        scorer = PathwayScorer()
        retained_pathways: List[str] = []
        registry_entries: List[Dict[str, Any]] = []
        skipped_pathways: List[Dict[str, Any]] = []

        for pathway_name in resolved_pathway_names:
            all_members = scorer.get_pathway_members(pathway_name)
            if not all_members:
                skipped_pathways.append(
                    {"pathway_name": pathway_name, "reason": "pathway_not_found"}
                )
                continue

            observed_members = sorted(set(all_members).intersection(dataset_hmdb_ids))
            coverage_ratio = len(observed_members) / max(len(all_members), 1)
            if (
                len(observed_members) < min_observed_members
                or coverage_ratio < min_coverage_ratio
            ):
                skipped_pathways.append(
                    {
                        "pathway_name": pathway_name,
                        "reason": "coverage_filter",
                        "observed_members": len(observed_members),
                        "total_members": len(all_members),
                        "coverage_ratio": round(coverage_ratio, 4),
                    }
                )
                continue

            retained_pathways.append(pathway_name)
            registry_entries.append(
                {
                    "feature_name": pathway_name,
                    "feature_type": "engineered_pathway",
                    "origin_subtype": "pathway_score",
                    "mapped_hmdb_ids": observed_members,
                    "mapped_pathways": [pathway_name],
                    "all_member_hmdb_ids": list(all_members),
                    "observed_member_hmdb_ids": observed_members,
                    "coverage_ratio": round(coverage_ratio, 4),
                    "observed_count": len(observed_members),
                    "quality_filters": {
                        "min_observed_members": min_observed_members,
                        "min_coverage_ratio": min_coverage_ratio,
                    },
                }
            )

        if not retained_pathways:
            print("✓ No pathways passed the coverage gate; exporting empty pathway feature file")
            output_df = _build_output_df(
                base_df=base_df,
                feature_payload={},
                include_label=include_label,
            )
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            output_df.to_csv(output_path, index=False)
            resolved_registry_path = _resolve_registry_path(registry_path)
            _ensure_registry_file(resolved_registry_path)
            return {
                "status": "success",
                "output_path": output_path,
                "num_pathways": 0,
                "pathway_names": [],
                "registry_path": resolved_registry_path,
                "n_candidates": len(resolved_pathway_names),
                "n_skipped_pathways": len(skipped_pathways),
                "message": "No pathways passed the coverage gate",
            }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result_json = calculate_pathway_scores(
            data_matrix_path=data_path,
            pathway_names=retained_pathways,
            output_path=output_path,
            include_group=include_label,
        )
        result = json.loads(result_json)
        if not result.get("success"):
            raise ValueError(result.get("error", "Pathway score calculation failed"))

        resolved_registry_path = _resolve_registry_path(registry_path)
        _merge_registry_entries(resolved_registry_path, registry_entries)
        print(f"✓ Updated engineered feature registry: {resolved_registry_path}")

        return {
            "status": "success",
            "output_path": output_path,
            "num_pathways": int(result.get("n_pathways", len(retained_pathways)) or 0),
            "pathway_names": retained_pathways,
            "registry_path": resolved_registry_path,
            "n_candidates": len(resolved_pathway_names),
            "n_skipped_pathways": len(skipped_pathways),
            "message": f"Generated pathway scores for {len(retained_pathways)} pathways",
        }

    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "num_pathways": 0,
            "pathway_names": [],
            "registry_path": _resolve_registry_path(registry_path),
            "message": f"Error: {str(e)}",
        }


def merge_feature_files(
    base_data_path: str,
    ratio_path: str = 'data/ratio_features_data.csv',
    taxonomy_path: str = 'data/taxonomy_sum_features_data.csv',
    pathway_path: str = 'data/pathway_scores.csv',
    output_path: str = 'data/enriched_data.csv',
    metadata_path: str = 'data/feature_metadata.json'
) -> Dict:
    """
    Merge base data with all generated feature files.
    
    Args:
        base_data_path: Path to base data CSV file
        ratio_path: Path to ratio features CSV (optional)
        taxonomy_path: Path to taxonomy sum features CSV (optional)
        pathway_path: Path to pathway scores CSV (optional)
        output_path: Path to save merged data (default: data/enriched_data.csv)
        metadata_path: Path to save feature metadata JSON (default: data/feature_metadata.json)
        
    Returns:
        dict: {
            'status': 'success' or 'error',
            'output_path': path to merged data file,
            'metadata_path': path to metadata file,
            'num_features': total number of features,
            'feature_breakdown': dict with counts by type,
            'message': status message
        }
    """
    try:
        # Load base data
        base_df = pd.read_csv(base_data_path)
        print(f"✓ Loaded base data: {len(base_df)} rows, {len(base_df.columns)} columns")

        id_col = _resolve_id_column(base_df)
        label_col = _resolve_label_column(base_df)
        protected_cols = {id_col}
        # CSV round-trips can change identifier dtypes (e.g. int64 vs object).
        # Normalize join keys before every merge while preserving the labels.
        base_df[id_col] = base_df[id_col].astype(str)
        if label_col:
            protected_cols.add(label_col)
        
        feature_counts = {
            'original': len([c for c in base_df.columns if c not in protected_cols]),
            'ratio': 0,
            'taxonomy_sum': 0,
            'pathway_score': 0
        }
        
        def _read_optional_feature_table(path: str, feature_type: str) -> Optional[pd.DataFrame]:
            """Read an engineered-feature table when it contains real columns.

            Feature generators are allowed to emit an empty placeholder when a
            dataset has no valid candidates for one engineered-feature family.
            ``pd.read_csv`` raises ``EmptyDataError`` for a zero-byte
            placeholder, which previously aborted the entire Phase 1 handoff.
            Treat that case as a valid zero-feature result and let the other
            feature families continue to merge.
            """
            if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
                print(f"ℹ No {feature_type} feature table available; continuing with zero features")
                return None
            try:
                table = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(f"ℹ Empty {feature_type} feature table; continuing with zero features")
                return None
            if table.shape[1] == 0:
                print(f"ℹ {feature_type.capitalize()} feature table has no columns; continuing with zero features")
                return None
            return table

        # Merge ratio features (if a non-empty table exists)
        ratio_df = _read_optional_feature_table(ratio_path, 'ratio')
        if ratio_df is not None:
            ratio_id_col = _resolve_id_column(ratio_df)
            ratio_df[ratio_id_col] = ratio_df[ratio_id_col].astype(str)
            if ratio_id_col not in ratio_df.columns:
                ratio_df[ratio_id_col] = range(len(ratio_df))
            ratio_label_col = _resolve_label_column(ratio_df)
            ratio_excluded = {ratio_id_col}
            if ratio_label_col:
                ratio_excluded.add(ratio_label_col)
            ratio_cols = [c for c in ratio_df.columns if c not in ratio_excluded]
            if ratio_cols:
                if ratio_id_col != id_col and len(ratio_df) == len(base_df):
                    ratio_df[id_col] = base_df[id_col].values
                    ratio_id_col = id_col
                merge_cols = [ratio_id_col] + ratio_cols
                if ratio_id_col != id_col:
                    print(f"⚠ Ratio file ID column '{ratio_id_col}' does not match base '{id_col}', skipping")
                else:
                    base_df = base_df.merge(ratio_df[merge_cols], on=id_col, how='left', suffixes=('', '_ratio'))
                feature_counts['ratio'] = len(ratio_cols)
                print(f"✓ Merged {len(ratio_cols)} ratio features")
        
        # Merge taxonomy sums (if a non-empty table exists)
        taxonomy_df = _read_optional_feature_table(taxonomy_path, 'taxonomy-sum')
        if taxonomy_df is not None:
            taxonomy_id_col = _resolve_id_column(taxonomy_df)
            taxonomy_df[taxonomy_id_col] = taxonomy_df[taxonomy_id_col].astype(str)
            if taxonomy_id_col not in taxonomy_df.columns:
                taxonomy_df[taxonomy_id_col] = range(len(taxonomy_df))
            taxonomy_label_col = _resolve_label_column(taxonomy_df)
            taxonomy_excluded = {taxonomy_id_col}
            if taxonomy_label_col:
                taxonomy_excluded.add(taxonomy_label_col)
            taxonomy_cols = [c for c in taxonomy_df.columns if c not in taxonomy_excluded]
            if taxonomy_cols:
                if taxonomy_id_col != id_col and len(taxonomy_df) == len(base_df):
                    taxonomy_df[id_col] = base_df[id_col].values
                    taxonomy_id_col = id_col
                merge_cols = [taxonomy_id_col] + taxonomy_cols
                if taxonomy_id_col != id_col:
                    print(f"⚠ Taxonomy file ID column '{taxonomy_id_col}' does not match base '{id_col}', skipping")
                else:
                    base_df = base_df.merge(taxonomy_df[merge_cols], on=id_col, how='left', suffixes=('', '_tax'))
                feature_counts['taxonomy_sum'] = len(taxonomy_cols)
                print(f"✓ Merged {len(taxonomy_cols)} taxonomy sum features")
        
        # Merge pathway scores (if a non-empty table exists)
        pathway_df = _read_optional_feature_table(pathway_path, 'pathway-score')
        if pathway_df is not None:
            pathway_id_col = _resolve_id_column(pathway_df)
            pathway_df[pathway_id_col] = pathway_df[pathway_id_col].astype(str)
            if pathway_id_col not in pathway_df.columns:
                pathway_df[pathway_id_col] = range(len(pathway_df))
            pathway_label_col = _resolve_label_column(pathway_df)
            pathway_excluded = {pathway_id_col}
            if pathway_label_col:
                pathway_excluded.add(pathway_label_col)
            pathway_cols = [c for c in pathway_df.columns if c not in pathway_excluded]
            if pathway_cols:
                if pathway_id_col != id_col and len(pathway_df) == len(base_df):
                    pathway_df[id_col] = base_df[id_col].values
                    pathway_id_col = id_col
                merge_cols = [pathway_id_col] + pathway_cols
                if pathway_id_col != id_col:
                    print(f"⚠ Pathway file ID column '{pathway_id_col}' does not match base '{id_col}', skipping")
                else:
                    base_df = base_df.merge(pathway_df[merge_cols], on=id_col, how='left', suffixes=('', '_path'))
                feature_counts['pathway_score'] = len(pathway_cols)
                print(f"✓ Merged {len(pathway_cols)} pathway score features")
        
        # Do not persist the synthetic row-index join key into downstream matrices.
        # It is only needed transiently to align temp engineered files when the
        # source data has no real sample identifier column.
        save_df = base_df.copy()
        if id_col == "__row_id__" and "__row_id__" in save_df.columns:
            save_df = save_df.drop(columns=["__row_id__"])

        # Save merged data
        save_df.to_csv(output_path, index=False)
        print(f"✓ Saved merged data: {len(save_df)} rows, {len(save_df.columns)} columns")
        
        # Create feature metadata
        metadata = {}
        for col in save_df.columns:
            if col in protected_cols:
                continue
            elif col.startswith('Ratio_'):
                metadata[col] = {'type': 'ratio', 'source': 'reaction_based'}
            elif col.startswith('Sum_'):
                metadata[col] = {'type': 'taxonomy_sum', 'source': 'taxonomy_based'}
            elif 'pathway' in col.lower() or 'score' in col.lower():
                metadata[col] = {'type': 'pathway_score', 'source': 'pathway_based'}
            else:
                metadata[col] = {'type': 'original', 'source': 'base_data'}
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved feature metadata: {len(metadata)} features")
        
        return {
            'status': 'success',
            'output_path': output_path,
            'metadata_path': metadata_path,
            'num_features': len(metadata),
            'feature_breakdown': feature_counts,
            'message': f"Merged {sum(feature_counts.values())} features successfully"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'output_path': None,
            'metadata_path': None,
            'num_features': 0,
            'feature_breakdown': {},
            'message': f"Error: {str(e)}"
        }


def merge_engineered_features(
    base_data_path: str,
    ratio_path: str = 'data/ratio_features_data.csv',
    taxonomy_path: str = 'data/taxonomy_sum_features_data.csv',
    pathway_path: str = 'data/pathway_scores.csv',
    output_path: str = 'data/enriched_data.csv',
    metadata_path: str = 'data/feature_metadata.json'
) -> Dict:
    """
    Preferred alias for Phase 1 engineered-feature merge.

    Keeps backward compatibility with the older `merge_feature_files(...)`
    implementation while exposing a clearer, centralized entrypoint name for
    SOP-generated code.
    """
    return merge_feature_files(
        base_data_path=base_data_path,
        ratio_path=ratio_path,
        taxonomy_path=taxonomy_path,
        pathway_path=pathway_path,
        output_path=output_path,
        metadata_path=metadata_path,
    )


def _load_registry_entries(registry_path: str) -> List[Dict[str, Any]]:
    payload = _load_registry_payload(registry_path)
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []
    return [record for record in features if isinstance(record, dict)]


def _project_ratio_features_from_registry(
    base_df: pd.DataFrame,
    registry_entries: List[Dict[str, Any]],
) -> Dict[str, pd.Series]:
    ratio_features: Dict[str, pd.Series] = {}
    for record in registry_entries:
        if str(record.get("feature_type") or "") != "engineered_ratio":
            continue
        mapped_ids = list(record.get("mapped_hmdb_ids", []) or [])
        if len(mapped_ids) < 2:
            continue
        id_a, id_b = str(mapped_ids[0]), str(mapped_ids[1])
        if id_a not in base_df.columns or id_b not in base_df.columns:
            continue
        series_a = pd.to_numeric(base_df[id_a], errors="coerce")
        series_b = pd.to_numeric(base_df[id_b], errors="coerce")
        transform_mode = str(
            ((record.get("quality_filters") or {}).get("transform_mode"))
            or "difference"
        ).strip().lower()
        if transform_mode == "division":
            safe_denominator = series_b.replace(0, 1e-10)
            feature_values = (series_a / safe_denominator).replace(
                [pd.NA, float("inf"), float("-inf")],
                0.0,
            )
        else:
            feature_values = series_a - series_b
        feature_name = str(record.get("feature_name") or "").strip()
        if feature_name:
            ratio_features[feature_name] = feature_values
    return ratio_features


def _project_taxonomy_features_from_registry(
    base_df: pd.DataFrame,
    registry_entries: List[Dict[str, Any]],
) -> Dict[str, pd.Series]:
    taxonomy_features: Dict[str, pd.Series] = {}
    for record in registry_entries:
        if str(record.get("origin_subtype") or "") != "taxonomy_sum":
            continue
        feature_name = str(record.get("feature_name") or "").strip()
        member_ids = [
            str(member_id or "").strip()
            for member_id in (record.get("selected_member_hmdb_ids", []) or [])
            if str(member_id or "").strip()
        ]
        available_members = [member_id for member_id in member_ids if member_id in base_df.columns]
        if not feature_name or not available_members:
            continue
        taxonomy_features[feature_name] = base_df[available_members].sum(axis=1)
    return taxonomy_features


def _project_pathway_features_from_registry(
    base_df: pd.DataFrame,
    registry_entries: List[Dict[str, Any]],
) -> Dict[str, pd.Series]:
    pathway_records = [
        record
        for record in registry_entries
        if str(record.get("feature_type") or "") == "engineered_pathway"
    ]
    if not pathway_records:
        return {}

    id_col = _detect_id_column(base_df)
    label_col = _detect_label_column(base_df)
    metadata_cols = {"Batch", "batch", "Class", "class", "Label", "label", "Target", "target"}
    if id_col:
        metadata_cols.add(id_col)
    if label_col:
        metadata_cols.add(label_col)

    feature_cols: List[str] = []
    for col in base_df.columns:
        if col in metadata_cols:
            continue
        if col.startswith("HMDB") or pd.api.types.is_numeric_dtype(base_df[col]):
            feature_cols.append(col)
    if not feature_cols:
        return {}

    feature_data = base_df[feature_cols].copy()
    if id_col:
        feature_data.index = base_df[id_col]

    pathway_names = [
        str(record.get("feature_name") or "").strip()
        for record in pathway_records
        if str(record.get("feature_name") or "").strip()
    ]
    if not pathway_names:
        return {}

    scorer = PathwayScorer()
    scores = scorer.calculate_scores(feature_data, pathway_names)
    return {
        str(pathway_name): scores[str(pathway_name)].reset_index(drop=True)
        for pathway_name in pathway_names
        if str(pathway_name) in scores.columns
    }


def project_engineered_features_from_registry(
    data_path: str,
    output_path: str,
    registry_path: str,
    feature_types: Optional[List[str]] = None,
    include_label: bool = False,
) -> Dict[str, Any]:
    """
    Materialize engineered features on a new dataset using train-derived registry entries.

    This is used to project train-selected engineered feature definitions onto the
    holdout pool without re-running any label- or full-data-dependent selection.
    """
    try:
        base_df = pd.read_csv(data_path)
        registry_entries = _load_registry_entries(registry_path)
        feature_type_filter = {str(item or "").strip() for item in (feature_types or []) if str(item or "").strip()}

        projected_payload: Dict[str, pd.Series] = {}
        if not feature_type_filter or "engineered_ratio" in feature_type_filter:
            projected_payload.update(_project_ratio_features_from_registry(base_df, registry_entries))
        if not feature_type_filter or "engineered_sum" in feature_type_filter:
            projected_payload.update(_project_taxonomy_features_from_registry(base_df, registry_entries))
        if not feature_type_filter or "engineered_pathway" in feature_type_filter:
            projected_payload.update(_project_pathway_features_from_registry(base_df, registry_entries))

        output_df = _build_output_df(
            base_df=base_df,
            feature_payload=projected_payload,
            include_label=include_label,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output_df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "output_path": output_path,
            "num_features": len(projected_payload),
            "feature_names": list(projected_payload.keys()),
            "message": f"Projected {len(projected_payload)} engineered features from registry",
        }
    except Exception as e:
        return {
            "status": "error",
            "output_path": None,
            "num_features": 0,
            "feature_names": [],
            "message": f"Error: {str(e)}",
        }


def generate_train_only_engineered_feature_bundle(
    train_data_path: str,
    holdout_data_path: str,
    output_dir: str,
    phase0_output: Optional[Dict[str, Any]] = None,
    registry_path: Optional[str] = None,
    rules_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate Phase 1 engineered features using only the train pool, then project
    the learned engineered definitions onto the holdout pool.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        resolved_registry_path = registry_path or os.path.join(output_dir, "engineered_feature_registry.json")
        if os.path.exists(resolved_registry_path):
            os.remove(resolved_registry_path)

        train_ratio_path = os.path.join(output_dir, "train_temp_ratios.csv")
        train_taxonomy_path = os.path.join(output_dir, "train_temp_taxonomy_sums.csv")
        train_pathway_path = os.path.join(output_dir, "train_temp_pathway_scores.csv")
        train_merged_path = os.path.join(output_dir, "train_data_with_engineered_features_unscaled.csv")
        train_metadata_path = os.path.join(output_dir, "train_feature_metadata_unscaled.json")

        holdout_ratio_path = os.path.join(output_dir, "holdout_temp_ratios.csv")
        holdout_taxonomy_path = os.path.join(output_dir, "holdout_temp_taxonomy_sums.csv")
        holdout_pathway_path = os.path.join(output_dir, "holdout_temp_pathway_scores.csv")
        holdout_merged_path = os.path.join(output_dir, "holdout_data_with_engineered_features_unscaled.csv")
        holdout_metadata_path = os.path.join(output_dir, "holdout_feature_metadata_unscaled.json")

        train_ratio_result = generate_reaction_ratios(
            data_path=train_data_path,
            output_path=train_ratio_path,
            registry_path=resolved_registry_path,
            rules_path=rules_path,
            include_label=False,
        )
        train_taxonomy_result = generate_taxonomy_sums(
            data_path=train_data_path,
            output_path=train_taxonomy_path,
            registry_path=resolved_registry_path,
            rules_path=rules_path,
            include_label=False,
        )
        train_pathway_result = generate_pathway_scores(
            data_path=train_data_path,
            output_path=train_pathway_path,
            phase0_output=phase0_output,
            registry_path=resolved_registry_path,
            rules_path=rules_path,
            include_label=False,
        )

        for result in (train_ratio_result, train_taxonomy_result, train_pathway_result):
            if not isinstance(result, dict) or result.get("status") != "success":
                raise ValueError(f"Train-only engineered feature generation failed: {result}")

        train_merge_result = merge_feature_files(
            base_data_path=train_data_path,
            ratio_path=train_ratio_path,
            taxonomy_path=train_taxonomy_path,
            pathway_path=train_pathway_path,
            output_path=train_merged_path,
            metadata_path=train_metadata_path,
        )
        if not isinstance(train_merge_result, dict) or train_merge_result.get("status") != "success":
            raise ValueError(f"Train engineered feature merge failed: {train_merge_result}")

        holdout_ratio_result = project_engineered_features_from_registry(
            data_path=holdout_data_path,
            output_path=holdout_ratio_path,
            registry_path=resolved_registry_path,
            feature_types=["engineered_ratio"],
            include_label=False,
        )
        holdout_taxonomy_result = project_engineered_features_from_registry(
            data_path=holdout_data_path,
            output_path=holdout_taxonomy_path,
            registry_path=resolved_registry_path,
            feature_types=["engineered_sum"],
            include_label=False,
        )
        holdout_pathway_result = project_engineered_features_from_registry(
            data_path=holdout_data_path,
            output_path=holdout_pathway_path,
            registry_path=resolved_registry_path,
            feature_types=["engineered_pathway"],
            include_label=False,
        )
        for result in (holdout_ratio_result, holdout_taxonomy_result, holdout_pathway_result):
            if not isinstance(result, dict) or result.get("status") != "success":
                raise ValueError(f"Holdout engineered feature projection failed: {result}")

        holdout_merge_result = merge_feature_files(
            base_data_path=holdout_data_path,
            ratio_path=holdout_ratio_path,
            taxonomy_path=holdout_taxonomy_path,
            pathway_path=holdout_pathway_path,
            output_path=holdout_merged_path,
            metadata_path=holdout_metadata_path,
        )
        if not isinstance(holdout_merge_result, dict) or holdout_merge_result.get("status") != "success":
            raise ValueError(f"Holdout engineered feature merge failed: {holdout_merge_result}")

        summary = {
            "schema_version": "phase1.train_only_engineering_bundle.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "train_data_path": train_data_path,
            "holdout_data_path": holdout_data_path,
            "registry_path": resolved_registry_path,
            "train_outputs": {
                "ratio_path": train_ratio_path,
                "taxonomy_path": train_taxonomy_path,
                "pathway_path": train_pathway_path,
                "merged_path": train_merged_path,
                "metadata_path": train_metadata_path,
            },
            "holdout_outputs": {
                "ratio_path": holdout_ratio_path,
                "taxonomy_path": holdout_taxonomy_path,
                "pathway_path": holdout_pathway_path,
                "merged_path": holdout_merged_path,
                "metadata_path": holdout_metadata_path,
            },
            "counts": {
                "train_ratio_features": int(train_ratio_result.get("num_ratios", 0) or 0),
                "train_taxonomy_features": int(train_taxonomy_result.get("num_sums", 0) or 0),
                "train_pathway_features": int(train_pathway_result.get("num_pathways", 0) or 0),
                "holdout_ratio_features": int(holdout_ratio_result.get("num_features", 0) or 0),
                "holdout_taxonomy_features": int(holdout_taxonomy_result.get("num_features", 0) or 0),
                "holdout_pathway_features": int(holdout_pathway_result.get("num_features", 0) or 0),
            },
        }
        summary_path = os.path.join(output_dir, "train_only_engineering_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return {
            "status": "success",
            "registry_path": resolved_registry_path,
            "summary_path": summary_path,
            "train_merged_path": train_merged_path,
            "holdout_merged_path": holdout_merged_path,
            "train_metadata_path": train_metadata_path,
            "holdout_metadata_path": holdout_metadata_path,
            "counts": summary["counts"],
            "message": "Train-only engineered feature bundle generated successfully",
        }
    except Exception as e:
        return {
            "status": "error",
            "registry_path": registry_path,
            "summary_path": "",
            "train_merged_path": "",
            "holdout_merged_path": "",
            "train_metadata_path": "",
            "holdout_metadata_path": "",
            "counts": {},
            "message": f"Error: {str(e)}",
        }


# Convenience function for testing
if __name__ == '__main__':
    print("Feature Generator Tools")
    print("=" * 50)
    
    # Test generate_reaction_ratios
    print("\n1. Testing generate_reaction_ratios...")
    result = generate_reaction_ratios('data/mapped_data.csv')
    print(f"Result: {result}")
    
    # Test generate_taxonomy_sums
    print("\n2. Testing generate_taxonomy_sums...")
    result = generate_taxonomy_sums('data/mapped_data.csv')
    print(f"Result: {result}")
    
    # Test merge_feature_files
    print("\n3. Testing merge_feature_files...")
    result = merge_feature_files('data/mapped_data.csv')
    print(f"Result: {result}")
