"""
Phase 4 normalization.

This module converts heterogeneous Phase 0-3 artifacts into a stable,
report-oriented `report_context` structure.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config_manager import ConfigManager


_HMDB_PATTERN = re.compile(r'^HMDB\d{5,9}(?:_\d+)?$', re.IGNORECASE)

_FEATURE_NAME_OVERRIDES = {
    '4_HYDROXYPROLINE': '4-Hydroxyproline',
    'INOSITOL_MYO': 'myo-Inositol',
    'TRYPTOPHAN': 'Tryptophan',
    'HMDB0000684': 'Kynurenine',
    'HMDB0000696': 'Methionine',
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DISEASE_PATHWAY_MAP_PATH = _PROJECT_ROOT / 'storage' / 'disease_pathway_map.json'
_DISEASE_PATHWAY_MAP_CACHE: Optional[Dict[str, Any]] = None


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_str(value: Any) -> str:
    if value is None:
        return ''
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coalesce_non_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == '':
            continue
        return value
    return None


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = _safe_str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _normalize_key(value: str) -> str:
    return re.sub(r'[^A-Z0-9]+', '', _safe_str(value).upper())


def _humanize_feature_token(token: str) -> str:
    normalized = _safe_str(token).strip()
    if not normalized:
        return ''
    if _HMDB_PATTERN.fullmatch(normalized):
        return normalized.upper()
    normalized = normalized.replace('__', '_').strip('_')
    words = [part for part in normalized.replace('_', ' ').split() if part]
    if not words:
        return normalized
    return ' '.join(word.capitalize() if not any(ch.isdigit() for ch in word) else word for word in words)


def _standardize_metabolite_name(name: str, *, feature_token: str = '') -> str:
    candidate = _safe_str(name).strip()
    if not candidate:
        return ''
    token_key = _safe_str(feature_token).strip().upper()
    if token_key in _FEATURE_NAME_OVERRIDES:
        return _FEATURE_NAME_OVERRIDES[token_key]

    candidate = re.sub(r'^\(\s*±\s*\)-?', '', candidate).strip()
    candidate = re.sub(r'^\+\s*/\s*-\s*', '', candidate).strip()
    if candidate.startswith(('L-', 'D-')):
        candidate = candidate[2:].strip()

    if candidate.lower() == 'inositol myo':
        return 'myo-Inositol'
    if re.match(r'^\d+\s+[A-Za-z]', candidate):
        prefix, remainder = candidate.split(' ', 1)
        return f'{prefix}-{remainder[:1].upper()}{remainder[1:]}'
    if _HMDB_PATTERN.fullmatch(candidate):
        return candidate.upper()
    return candidate


def _build_name_index(values: List[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for value in values:
        normalized = _normalize_key(value)
        if normalized and normalized not in index:
            index[normalized] = value
    return index


def _load_disease_pathway_map() -> Dict[str, Any]:
    global _DISEASE_PATHWAY_MAP_CACHE
    if _DISEASE_PATHWAY_MAP_CACHE is not None:
        return _DISEASE_PATHWAY_MAP_CACHE
    try:
        with open(_DISEASE_PATHWAY_MAP_PATH, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
            _DISEASE_PATHWAY_MAP_CACHE = payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        _DISEASE_PATHWAY_MAP_CACHE = {}
    return _DISEASE_PATHWAY_MAP_CACHE


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    candidate = _safe_str(path).strip()
    if not candidate:
        return {}
    try:
        with open(candidate, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _figure_label_from_id(figure_id: str) -> str:
    normalized = _safe_str(figure_id).strip()
    if normalized.lower().startswith('fig'):
        return f'Fig. {normalized[3:]}'
    return normalized or 'Fig.'


def _figure_reference_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    primary = _safe_dict(record.get('primary_output'))
    path = _safe_str(primary.get('path')).strip()
    return {
        'figure_id': _safe_str(record.get('figure_id')).strip(),
        'figure_label': _figure_label_from_id(record.get('figure_id')),
        'title': _safe_str(record.get('title')).strip(),
        'asset_name': Path(path).name if path else '',
        'path': path,
        'section': _safe_str(record.get('section')).strip(),
    }


def _build_figure_evidence_summary(
    *,
    figure_records: List[Dict[str, Any]],
    task_records: List[Dict[str, Any]],
    feature_selection_summary: Dict[str, Any],
    adaptive_selection_report: Dict[str, Any],
    phase2_search_summary: Dict[str, Any],
    phase2_clinical_utility: Dict[str, Any],
    winner_feature_dictionary: List[Dict[str, Any]],
    winner_perf: Any,
) -> Dict[str, Any]:
    figure_index = {
        _safe_str(record.get('figure_id')).strip(): record
        for record in figure_records
        if isinstance(record, dict) and _safe_str(record.get('figure_id')).strip()
    }
    task_index = {
        _safe_str(record.get('task')).strip(): record
        for record in task_records
        if isinstance(record, dict) and _safe_str(record.get('task')).strip()
    }

    # Phase4Pipeline changes cwd to the explicit run root.  Runtime artifacts
    # must therefore resolve from cwd, never from the source-code checkout.
    radar_scores = _read_json_if_exists(str(Path.cwd() / 'output' / 'figures' / 'radar_scores.json'))
    rcs_figure = _safe_dict(figure_index.get('fig4d'))
    rcs_metadata = _safe_dict(rcs_figure.get('metadata_output'))
    rcs_renderer_metadata = _safe_dict(rcs_figure.get('renderer_metadata'))
    rcs_summary_path = _safe_str(
        rcs_metadata.get('path')
        or rcs_renderer_metadata.get('rcs_summary_path')
    ).strip()
    rcs_summary = _read_json_if_exists(rcs_summary_path)

    holdout_dca_summary = _safe_dict(_safe_dict(task_index.get('plot_final_holdout_dca')).get('renderer_metadata')).get('dca_summary')
    cv_dca_summary = _safe_dict(_safe_dict(task_index.get('plot_dca')).get('renderer_metadata')).get('dca_summary')
    cv_calibration_summary = _safe_dict(_safe_dict(task_index.get('plot_calibration')).get('renderer_metadata')).get('calibration_summary')
    # The report-facing probability evaluation is the fixed holdout result.
    # Keep CV/OOF evidence separately available for transparent comparison.
    dca_summary = holdout_dca_summary if isinstance(holdout_dca_summary, dict) else cv_dca_summary
    calibration_summary = cv_calibration_summary
    threshold_summary = _safe_dict(_safe_dict(task_index.get('plot_threshold_performance')).get('renderer_metadata')).get('threshold_performance_summary')
    shap_figure = _safe_dict(figure_index.get('fig4c'))
    shap_metadata = _safe_dict(shap_figure.get('metadata_output'))
    shap_renderer_metadata = _safe_dict(shap_figure.get('renderer_metadata'))
    shap_summary_path = _safe_str(
        shap_metadata.get('path')
        or shap_renderer_metadata.get('shap_summary_path')
    ).strip()
    shap_summary = _read_json_if_exists(shap_summary_path)

    if not isinstance(dca_summary, dict):
        dca_summary = _safe_dict(
            phase2_clinical_utility.get('decision_curve_relative')
            or phase2_clinical_utility.get('decision_curve_summary')
        )
    if not isinstance(calibration_summary, dict):
        calibration_summary = _safe_dict(
            phase2_clinical_utility.get('probability_calibration_adjusted')
            or phase2_clinical_utility.get('probability_calibration')
        )
    if not isinstance(threshold_summary, dict):
        threshold_summary = {}

    methods_used = _safe_list(adaptive_selection_report.get('methods_used'))
    shap_reference_names = [
        _safe_str(item.get('preferred_report_name')).strip()
        for item in winner_feature_dictionary
        if isinstance(item, dict) and not bool(item.get('is_engineered')) and _safe_str(item.get('preferred_report_name')).strip()
    ]
    shap_name_index: Dict[str, str] = {}
    for item in winner_feature_dictionary:
        if not isinstance(item, dict):
            continue
        preferred_name = _safe_str(item.get('preferred_report_name') or item.get('report_label') or item.get('feature_name')).strip()
        if not preferred_name:
            continue
        for candidate in (
            item.get('preferred_report_name'),
            item.get('report_label'),
            item.get('feature_token'),
            item.get('feature_name'),
        ):
            normalized_candidate = _normalize_key(candidate)
            if normalized_candidate and normalized_candidate not in shap_name_index:
                shap_name_index[normalized_candidate] = preferred_name

    def _normalize_shap_feature_name(value: Any) -> str:
        raw_name = _safe_str(value).strip()
        if not raw_name:
            return ''
        mapped_name = shap_name_index.get(_normalize_key(raw_name))
        if mapped_name:
            return mapped_name
        return _standardize_metabolite_name(raw_name, feature_token=raw_name)
    nonlinear_features = [
        _safe_str(item.get('display_feature')).strip()
        for item in _safe_list(rcs_summary.get('features'))
        if isinstance(item, dict) and _safe_float(item.get('p_nonlinearity')) is not None and _safe_float(item.get('p_nonlinearity')) < 0.05
    ]
    overall_associated_features = [
        _safe_str(item.get('display_feature')).strip()
        for item in _safe_list(rcs_summary.get('features'))
        if isinstance(item, dict) and _safe_float(item.get('p_overall')) is not None and _safe_float(item.get('p_overall')) < 0.05
    ]

    return {
        'methodology_figure_ids': ['fig1a', 'fig1b', 'fig1c', 'fig2b'],
        'results_figure_ids': ['fig1c', 'fig2a', 'fig2b', 'fig3', 'fig4a_holdout', 'fig4b_holdout', 'fig4c', 'fig4d', 'fig4e', 'fig4f'],
        'discussion_figure_ids': ['fig4a_holdout', 'fig4b_holdout', 'fig4c', 'fig4d', 'fig4e', 'fig4f'],
        'figures': {
            figure_id: _figure_reference_payload(record)
            for figure_id, record in figure_index.items()
        },
        'prior_evidence_atlas': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig1c'))),
        },
        'upset': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig1c'))),
            'methods_count': len(methods_used) if methods_used else 3,
            'methods_used': methods_used,
            'union_features_count': feature_selection_summary.get('union_features_count'),
            'at_least_two_methods_count': feature_selection_summary.get('intersection_2_of_3_count'),
            'all_methods_count': feature_selection_summary.get('intersection_all_3_count'),
            'consensus_features_count': adaptive_selection_report.get('consensus_features_count'),
        },
        'autogluon_roc': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig2a'))),
            'available': 'fig2a' in figure_index,
        },
        'pareto': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig2b'))),
            'beam_width': phase2_search_summary.get('beam_width'),
            'actual_depth': phase2_search_summary.get('actual_depth'),
            'max_depth': phase2_search_summary.get('max_depth'),
            'stop_reason': phase2_search_summary.get('stop_reason'),
            'early_stopping_type': phase2_search_summary.get('early_stopping_type'),
            'layer_count': len(_safe_list(phase2_search_summary.get('layer_summary'))),
        },
        'radar': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig3'))),
            'raw_scores': _safe_dict(radar_scores.get('raw_scores')),
            'normalized_scores': _safe_dict(radar_scores.get('normalized_scores')),
        },
        'final_roc': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig4a'))),
            'auc': winner_perf,
            'feature_count': len(winner_feature_dictionary),
        },
        'dca': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig4b_holdout') or figure_index.get('fig4b'))),
            **_safe_dict(dca_summary),
            'winner_better_than_treat_none_ranges': _safe_list(_safe_dict(dca_summary).get('winner_better_than_treat_none_ranges')),
            'winner_better_than_treat_all_ranges': _safe_list(_safe_dict(dca_summary).get('winner_better_than_treat_all_ranges')),
            'winner_better_than_baseline_ranges': _safe_list(_safe_dict(dca_summary).get('winner_better_than_baseline_ranges')),
            'summary_type': _safe_dict(dca_summary).get('summary_type'),
        },
        'dca_internal_cv': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig4b'))),
            **_safe_dict(cv_dca_summary),
        },
        'shap': {
            **_figure_reference_payload(shap_figure),
            'waterfall_asset_name': Path(_safe_str(_safe_list(shap_figure.get('auxiliary_outputs'))[0].get('path'))).name
            if _safe_list(shap_figure.get('auxiliary_outputs')) else '',
            'summary_path': shap_summary_path,
            'feature_names': _dedupe_preserve_order([
                _normalize_shap_feature_name(item.get('feature'))
                for item in _safe_list(shap_summary.get('ranked_features'))
                if isinstance(item, dict) and _normalize_shap_feature_name(item.get('feature'))
            ]) or shap_reference_names,
            'ranked_features': [
                {
                    'feature': _normalize_shap_feature_name(item.get('feature')),
                    'rank': item.get('rank'),
                    'mean_abs_shap': _safe_float(item.get('mean_abs_shap')),
                    'mean_shap': _safe_float(item.get('mean_shap')),
                    'direction': _safe_str(item.get('direction')).strip(),
                    'direction_correlation': _safe_float(item.get('direction_correlation')),
                }
                for item in _safe_list(shap_summary.get('ranked_features'))
                if isinstance(item, dict) and _normalize_shap_feature_name(item.get('feature'))
            ],
            'top_positive_risk_features': [
                _normalize_shap_feature_name(item.get('feature'))
                for item in _safe_list(shap_summary.get('top_positive_risk_features'))
                if isinstance(item, dict) and _normalize_shap_feature_name(item.get('feature'))
            ],
            'top_negative_risk_features': [
                _normalize_shap_feature_name(item.get('feature'))
                for item in _safe_list(shap_summary.get('top_negative_risk_features'))
                if isinstance(item, dict) and _normalize_shap_feature_name(item.get('feature'))
            ],
            'patient_index_used': _coalesce_non_null(
                shap_summary.get('patient_index_used'),
                shap_renderer_metadata.get('patient_index_used'),
            ),
            'local_example': {
                'patient_index': _coalesce_non_null(
                    _safe_dict(shap_summary.get('local_example')).get('patient_index'),
                    shap_summary.get('patient_index_used'),
                    shap_renderer_metadata.get('patient_index_used'),
                ),
                'top_contributors': [
                    {
                        'feature': _normalize_shap_feature_name(item.get('feature')),
                        'feature_value': _safe_float(item.get('feature_value')),
                        'shap_value': _safe_float(item.get('shap_value')),
                        'direction': _safe_str(item.get('direction')).strip(),
                        'abs_shap': _safe_float(item.get('abs_shap')),
                    }
                    for item in _safe_list(_safe_dict(shap_summary.get('local_example')).get('top_contributors'))
                    if isinstance(item, dict) and _normalize_shap_feature_name(item.get('feature'))
                ],
            },
            'feature_rank_summary_available': bool(_safe_list(shap_summary.get('ranked_features'))),
        },
        'rcs': {
            **_figure_reference_payload(rcs_figure),
            'summary_path': rcs_summary_path,
            'nonlinear_features': _dedupe_preserve_order([item for item in nonlinear_features if item]),
            'overall_associated_features': _dedupe_preserve_order([item for item in overall_associated_features if item]),
            'feature_summaries': _safe_list(rcs_summary.get('features')),
        },
        'calibration': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig4e'))),
            **_safe_dict(calibration_summary),
            'brier_score': _safe_dict(calibration_summary).get('brier_score'),
            'expected_calibration_error': _safe_dict(calibration_summary).get('expected_calibration_error'),
            'integrated_calibration_index': _safe_dict(calibration_summary).get('integrated_calibration_index'),
            'calibration_slope': _safe_dict(calibration_summary).get('calibration_slope'),
            'calibration_intercept': _safe_dict(calibration_summary).get('calibration_intercept'),
        },
        'calibration_internal_cv': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig4e'))),
            **_safe_dict(cv_calibration_summary),
        },
        'threshold_performance': {
            **_figure_reference_payload(_safe_dict(figure_index.get('fig4f'))),
            'default_threshold': _coalesce_non_null(
                _safe_dict(threshold_summary).get('default_threshold'),
                _safe_dict(phase2_clinical_utility.get('recommended_threshold_summary')).get('selected_threshold'),
                _safe_dict(phase2_clinical_utility.get('scenario')).get('default_action_threshold'),
            ),
            'companion_threshold': _coalesce_non_null(
                _safe_dict(threshold_summary).get('companion_threshold'),
                _safe_dict(phase2_clinical_utility.get('data_driven_companion_threshold_summary')).get('selected_threshold'),
            ),
            'sample_threshold_metrics': _safe_list(_safe_dict(threshold_summary).get('sample_threshold_metrics')),
        },
    }


def _extract_pathway_names(phase0: Dict[str, Any]) -> List[str]:
    runtime_summary = _safe_dict(phase0.get('runtime_pathway_summary'))
    for field_name in ('disease_core_pathways', 'top_pathways'):
        candidate_values = _dedupe_preserve_order([
            _safe_str(item).strip()
            for item in _safe_list(runtime_summary.get(field_name))
            if _safe_str(item).strip()
        ])
        if candidate_values:
            return candidate_values

    for field_name in ('disease_core_pathways', 'top_pathways'):
        candidate_values = _dedupe_preserve_order([
            _safe_str(item).strip()
            for item in _safe_list(phase0.get(field_name))
            if _safe_str(item).strip()
        ])
        if candidate_values:
            return candidate_values

    disease_name = _safe_str(phase0.get('normalized_disease_name') or phase0.get('disease_name')).strip().lower()
    disease_pathway_map = _load_disease_pathway_map()
    disease_entry = _safe_dict(disease_pathway_map.get(disease_name))
    for field_name in ('disease_core_pathways', 'top_pathways', 'core_pathways', 'pathways'):
        candidate_values = _dedupe_preserve_order([
            _safe_str(item).strip()
            for item in _safe_list(disease_entry.get(field_name))
            if _safe_str(item).strip()
        ])
        if candidate_values:
            return candidate_values

    feature_definitions = _safe_dict(phase0.get('feature_definitions'))
    return _dedupe_preserve_order([
        _safe_str(item).strip()
        for item in _safe_list(feature_definitions.get('target_pathways'))
        if _safe_str(item).strip()
    ])


def _normalize_confirmed_biomarkers(phase0: Dict[str, Any]) -> Dict[str, Any]:
    confirmed_raw = _safe_list(phase0.get('confirmed_biomarkers'))
    normalized_records: List[Dict[str, Any]] = []
    biomarker_names: List[str] = []
    biomarker_ids: List[str] = []

    for item in confirmed_raw:
        if isinstance(item, dict):
            biomarker_id = _safe_str(item.get('id') or item.get('hmdb_id')).strip()
            biomarker_name = _safe_str(item.get('name') or item.get('metabolite') or biomarker_id).strip()
            if biomarker_name:
                biomarker_names.append(biomarker_name)
            if biomarker_id:
                biomarker_ids.append(biomarker_id)
            normalized_records.append(
                {
                    **item,
                    'id': biomarker_id,
                    'name': biomarker_name,
                }
            )
            continue

        biomarker_name = _safe_str(item).strip()
        if not biomarker_name:
            continue
        biomarker_names.append(biomarker_name)
        normalized_records.append({'id': '', 'name': biomarker_name})

    final_priors = _dedupe_preserve_order([
        _safe_str(item).strip()
        for item in _safe_list(phase0.get('final_priors'))
        if _safe_str(item).strip()
    ])

    return {
        'records': normalized_records,
        'names': _dedupe_preserve_order(biomarker_names),
        'ids': _dedupe_preserve_order(biomarker_ids),
        'final_priors': final_priors,
    }


def _canonicalize_feature_record(record: Dict[str, Any]) -> Dict[str, Any]:
    feature_name = _safe_str(
        record.get('feature_name')
        or record.get('feature')
        or record.get('standardized_name')
        or record.get('display_name')
    ).strip()
    mapped_pathways = [
        _safe_str(item).strip()
        for item in _safe_list(record.get('mapped_pathways'))
        if _safe_str(item).strip()
    ]
    mapped_hmdb_ids = [
        _safe_str(item).strip()
        for item in _safe_list(record.get('mapped_hmdb_ids'))
        if _safe_str(item).strip()
    ]
    mapped_metabolites = [
        _safe_str(item).strip()
        for item in _safe_list(record.get('mapped_metabolites'))
        if _safe_str(item).strip()
    ]
    prior_supported = bool(record.get('prior_supported', record.get('is_prior_supported', False)))

    matched_phase0_biomarker = _safe_str(record.get('matched_phase0_biomarker')).strip()
    if not matched_phase0_biomarker and prior_supported:
        matched_phase0_biomarker = (mapped_hmdb_ids or mapped_metabolites or [''])[0]

    matched_pathway = _safe_str(record.get('matched_pathway')).strip()
    if not matched_pathway and mapped_pathways:
        matched_pathway = mapped_pathways[0]

    return {
        **record,
        'feature_name': feature_name,
        'feature_token': _safe_str(record.get('feature') or feature_name).strip(),
        'display_name': _safe_str(record.get('display_name')).strip(),
        'standardized_name': _safe_str(record.get('standardized_name') or feature_name).strip(),
        'origin_type': _safe_str(record.get('origin_type')).strip(),
        'origin_subtype': _safe_str(record.get('origin_subtype')).strip(),
        'is_engineered': bool(record.get('is_engineered', False)),
        'prior_supported': prior_supported,
        'mapped_metabolites': mapped_metabolites,
        'mapped_hmdb_ids': mapped_hmdb_ids,
        'mapped_pathways': mapped_pathways,
        'matched_phase0_biomarker': matched_phase0_biomarker or None,
        'matched_pathway': matched_pathway or None,
    }


def _classify_feature_record(
    feature_name: str,
    pathway_index: Dict[str, str],
    biomarker_index: Dict[str, str],
) -> Dict[str, Any]:
    normalized_feature = _normalize_key(feature_name)
    matched_pathway = pathway_index.get(normalized_feature)
    matched_biomarker = biomarker_index.get(normalized_feature)
    standardized_name = _safe_str(feature_name).strip().upper()

    if standardized_name.startswith('RATIO_'):
        origin_type = 'engineered_ratio'
        origin_subtype = 'reaction_ratio'
        is_engineered = True
    elif standardized_name.startswith('TAXSUM_'):
        origin_type = 'engineered_sum'
        origin_subtype = 'taxonomy_sum'
        is_engineered = True
    elif standardized_name.startswith('SUM_'):
        origin_type = 'engineered_sum'
        origin_subtype = 'group_sum'
        is_engineered = True
    elif matched_pathway:
        origin_type = 'engineered_pathway'
        origin_subtype = 'pathway_score'
        is_engineered = True
    elif _HMDB_PATTERN.fullmatch(standardized_name):
        origin_type = 'raw'
        origin_subtype = 'hmdb_metabolite'
        is_engineered = False
    else:
        origin_type = 'raw'
        origin_subtype = 'named_metabolite'
        is_engineered = False

    return {
        'feature_name': feature_name,
        'origin_type': origin_type,
        'origin_subtype': origin_subtype,
        'is_engineered': is_engineered,
        'prior_supported': matched_biomarker is not None,
        'matched_phase0_biomarker': matched_biomarker,
        'matched_pathway': matched_pathway,
        'annotation_source': 'phase4.normalizer.inference',
    }


def _normalize_topsis_weights(weights: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    key_map = {
        'perf': 'f_perf',
        'f_perf': 'f_perf',
        'bio': 'f_bio',
        'f_bio': 'f_bio',
        'cost': 'f_cost',
        'f_cost': 'f_cost',
        'corr': 'f_corr',
        'f_corr': 'f_corr',
    }
    for raw_key, value in _safe_dict(weights).items():
        mapped_key = key_map.get(_safe_str(raw_key).strip(), _safe_str(raw_key).strip())
        normalized[mapped_key] = value
    return normalized


def _summarize_feature_records(feature_records: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        'raw_count': 0,
        'engineered_ratio_count': 0,
        'engineered_sum_count': 0,
        'engineered_pathway_count': 0,
        'unknown_count': 0,
        'prior_supported_count': 0,
        'taxonomy_sum_count': 0,
        'group_sum_count': 0,
    }

    for record in feature_records:
        origin_type = _safe_str(record.get('origin_type'))
        origin_subtype = _safe_str(record.get('origin_subtype'))
        if origin_type == 'raw':
            summary['raw_count'] += 1
        elif origin_type == 'engineered_ratio':
            summary['engineered_ratio_count'] += 1
        elif origin_type == 'engineered_sum':
            summary['engineered_sum_count'] += 1
        elif origin_type == 'engineered_pathway':
            summary['engineered_pathway_count'] += 1
        else:
            summary['unknown_count'] += 1

        if origin_subtype == 'taxonomy_sum':
            summary['taxonomy_sum_count'] += 1
        if origin_subtype == 'group_sum':
            summary['group_sum_count'] += 1
        if bool(record.get('prior_supported')):
            summary['prior_supported_count'] += 1
    return summary


def _materialize_feature_records(
    selected_features: List[str],
    feature_provenance: Dict[str, Any],
    pathway_index: Dict[str, str],
    biomarker_index: Dict[str, str],
) -> List[Dict[str, Any]]:
    provenance_records = _safe_list(feature_provenance.get('features'))
    if provenance_records:
        return [
            _canonicalize_feature_record(record)
            for record in provenance_records
            if isinstance(record, dict)
        ]
    return [
        _classify_feature_record(feature_name, pathway_index, biomarker_index)
        for feature_name in selected_features
    ]


def _select_feature_records(feature_names: List[str], available_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _select_feature_records_with_fallback(
        feature_names=feature_names,
        available_records=available_records,
        pathway_index={},
        biomarker_index={},
    )


def _select_feature_records_with_fallback(
    feature_names: List[str],
    available_records: List[Dict[str, Any]],
    pathway_index: Dict[str, str],
    biomarker_index: Dict[str, str],
) -> List[Dict[str, Any]]:
    record_index = {
        _normalize_key(_safe_str(record.get('feature_name'))): record
        for record in available_records
        if _normalize_key(_safe_str(record.get('feature_name')))
    }
    selected_records: List[Dict[str, Any]] = []
    for feature_name in feature_names:
        normalized = _normalize_key(feature_name)
        record = record_index.get(normalized)
        if record is None:
            record = _classify_feature_record(
                feature_name=feature_name,
                pathway_index=pathway_index,
                biomarker_index=biomarker_index,
            )
        selected_records.append(record)
    return selected_records


def _infer_phase1_selection_policy(feature_selection_strategy: str) -> Dict[str, Any]:
    strategy = _safe_str(feature_selection_strategy).lower()
    if 'union' in strategy:
        return {
            'selection_policy': 'union',
            'selection_policy_reason': 'union-style aggregation was used to preserve a sufficiently large Phase 2 candidate pool.',
        }
    if 'intersection' in strategy:
        return {
            'selection_policy': 'intersection',
            'selection_policy_reason': 'intersection-style aggregation was used to retain only features consistently selected across methods.',
        }
    if 'consensus' in strategy or 'voting' in strategy:
        return {
            'selection_policy': 'consensus_voting',
            'selection_policy_reason': 'consensus-style aggregation was used to prioritize repeated multi-method support.',
        }
    return {
        'selection_policy': 'unspecified',
        'selection_policy_reason': 'selection policy was not explicitly derivable from the available Phase 1 artifacts.',
    }


def _infer_phase2_early_stopping_type(
    stop_reason: str,
    max_depth: Optional[int],
    physical_feature_limit: Optional[int],
) -> str:
    reason = _safe_str(stop_reason).strip().lower()
    if 'delong' in reason:
        return 'delong_patience'
    if 'pool_exhausted' in reason:
        return 'pool_exhausted'
    if 'hard_prune' in reason:
        return 'hard_prune_exhausted'
    if 'pareto_empty' in reason:
        return 'pareto_empty'
    if 'beam_empty' in reason:
        return 'beam_empty'
    if 'fallback' in reason:
        return 'fallback_to_global_best'
    if reason == 'max_depth_reached':
        if max_depth is not None and physical_feature_limit is not None and int(max_depth) == int(physical_feature_limit):
            return 'epv_limit_reached'
        return 'max_depth_reached'
    return reason or 'unknown'


def _build_phase2_search_summary_fallback(
    phase2_result: Dict[str, Any],
    winner: Dict[str, Any],
) -> Dict[str, Any]:
    search_details = _safe_dict(winner.get('search_details')) or _safe_dict(phase2_result.get('search_details'))
    layers = [layer for layer in _safe_list(search_details.get('layers')) if isinstance(layer, dict)]
    max_depth = search_details.get('max_depth', phase2_result.get('max_depth'))
    physical_feature_limit = search_details.get('physical_feature_limit')
    stop_reason = _safe_str(winner.get('stop_reason') or phase2_result.get('stop_reason'))
    early_stopping_type = _infer_phase2_early_stopping_type(
        stop_reason=stop_reason,
        max_depth=max_depth if isinstance(max_depth, int) else None,
        physical_feature_limit=physical_feature_limit if isinstance(physical_feature_limit, int) else None,
    )

    return {
        'schema_version': 'phase4.phase2_search_summary.fallback.v1',
        'generation_mode': 'fallback_from_phase2_result',
        'engine': search_details.get('engine', 'ParetoSFS_Engine'),
        'disease_name': phase2_result.get('disease_name'),
        'bio_score_version': winner.get('bio_score_version') or phase2_result.get('bio_score_version'),
        'f_bio_v2_enabled': bool(
            search_details.get('f_bio_v2_enabled', (winner.get('bio_score_version') or phase2_result.get('bio_score_version')) == 'f_bio_v2')
        ),
        'metric': search_details.get('metric', phase2_result.get('metric')),
        'beam_width': search_details.get('beam_width', phase2_result.get('beam_width')),
        'max_depth': max_depth,
        'max_depth_additional_features': search_details.get('max_depth_additional_features', phase2_result.get('max_depth_additional_features')),
        'actual_depth': len(layers),
        'last_completed_depth': layers[-1].get('depth') if layers else None,
        'minority_class_count': search_details.get('minority_class_count'),
        'physical_feature_limit': physical_feature_limit,
        'expansion_feature_limit': search_details.get('expansion_feature_limit'),
        'stop_reason': stop_reason,
        'early_stopping_type': early_stopping_type,
        'rollback_triggered': bool(winner.get('rollback_triggered', phase2_result.get('rollback_triggered', False))),
        'min_feature_guard_applied': bool(
            winner.get(
                'min_feature_guard_applied',
                phase2_result.get('min_feature_guard_applied', search_details.get('min_feature_guard_applied', False))
            )
        ),
        'global_best_depth': winner.get('global_best_depth', phase2_result.get('global_best_depth')),
        'min_final_feature_count': search_details.get('min_final_feature_count'),
        'search_start_mode': search_details.get('search_start_mode'),
        'initial_panel_features': _safe_list(search_details.get('initial_panel_features')),
        'bio_context_summary': _safe_dict(search_details.get('bio_context_summary')),
        'protected_anchor_features': _safe_list(
            winner.get('protected_anchor_features', phase2_result.get('protected_anchor_features', search_details.get('protected_features')))
        ),
        'n_protected_anchor_features': winner.get(
            'n_protected_anchor_features',
            phase2_result.get('n_protected_anchor_features', search_details.get('n_protected_features')),
        ),
        'missing_protected_anchor_features': _safe_list(
            winner.get('missing_protected_anchor_features', phase2_result.get('missing_protected_anchor_features', search_details.get('missing_protected_features')))
        ),
        'winner_feature_count': len(_safe_list(winner.get('features'))),
        'winner_features': _safe_list(winner.get('features')),
        'n_added_features_beyond_protected': winner.get(
            'n_added_features_beyond_protected',
            phase2_result.get('n_added_features_beyond_protected'),
        ),
        'winner_scores': {
            'f_perf': winner.get('perf'),
            'f_bio': winner.get('bio'),
            'f_corr': winner.get('corr'),
            'f_cost': winner.get('cost'),
        },
        'dual_criteria_summary': {
            'epv_guard_enabled': physical_feature_limit is not None,
            'delong_patience_enabled': phase2_result.get('patience') is not None,
            'delong_patience': phase2_result.get('patience'),
            'delong_improvement_delta': phase2_result.get('epsilon'),
            'epv_guard_triggered': early_stopping_type == 'epv_limit_reached',
            'delong_guard_triggered': early_stopping_type == 'delong_patience',
        },
        'layer_summary': [
            {
                'depth': layer.get('depth'),
                'expanded_count': layer.get('expanded_count'),
                'survivor_count': layer.get('survivor_count'),
                'pareto_count': layer.get('pareto_count'),
                'beam_count': layer.get('beam_count'),
                'top1_features': _safe_list(layer.get('top1_features')),
                'top1_perf': layer.get('top1_perf'),
                'delong_p_value': layer.get('delong_p_value'),
                'improvement_status': layer.get('improvement_status'),
            }
            for layer in layers
        ],
    }


def _build_figure_section_summary(figure_records: List[Dict[str, Any]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in figure_records:
        section = _safe_str(record.get('section')) or 'Unassigned'
        summary[section] = summary.get(section, 0) + 1
    return summary


def _top_pathway_candidates(feature_records: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    seen = set()
    output: List[str] = []
    for record in feature_records:
        pathway = _safe_str(record.get('matched_pathway'))
        if not pathway or pathway in seen:
            continue
        seen.add(pathway)
        output.append(pathway)
        if len(output) >= limit:
            break
    return output


def _select_preferred_feature_name(record: Dict[str, Any]) -> str:
    display_name = _safe_str(record.get('display_name')).strip()
    mapped_metabolites = [
        _safe_str(item).strip()
        for item in _safe_list(record.get('mapped_metabolites'))
        if _safe_str(item).strip()
    ]
    matched_biomarker = _safe_str(record.get('matched_phase0_biomarker')).strip()
    feature_name = _safe_str(record.get('feature_name') or record.get('feature_token')).strip()

    for candidate in [display_name, *mapped_metabolites, matched_biomarker]:
        if candidate and not _HMDB_PATTERN.fullmatch(candidate):
            return candidate
    if feature_name:
        return _humanize_feature_token(feature_name)
    return 'Unnamed feature'


def _build_feature_report_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    feature_token = _safe_str(record.get('feature_token') or record.get('feature_name')).strip()
    preferred_report_name = _standardize_metabolite_name(
        _select_preferred_feature_name(record),
        feature_token=feature_token,
    )
    hmdb_ids = [
        _safe_str(item).strip()
        for item in _safe_list(record.get('mapped_hmdb_ids'))
        if _safe_str(item).strip()
    ]
    if not hmdb_ids and _HMDB_PATTERN.fullmatch(feature_token):
        hmdb_ids = [feature_token.upper()]

    report_label = preferred_report_name
    composition_label = report_label
    if hmdb_ids and preferred_report_name.upper() not in {item.upper() for item in hmdb_ids}:
        composition_label = f'{preferred_report_name} ({hmdb_ids[0]})'

    evidence_tags: List[str] = []
    if bool(record.get('prior_supported')):
        evidence_tags.append('prior_supported')
    if _safe_str(record.get('matched_pathway')).strip():
        evidence_tags.append('pathway_linked')
    if bool(record.get('is_engineered')):
        evidence_tags.append('engineered')
    if not evidence_tags:
        evidence_tags.append('panel_selected')

    return {
        'feature_token': feature_token,
        'preferred_report_name': preferred_report_name,
        'report_label': report_label,
        'composition_label': composition_label,
        'display_name': _standardize_metabolite_name(_safe_str(record.get('display_name')).strip() or preferred_report_name, feature_token=feature_token),
        'origin_type': _safe_str(record.get('origin_type')).strip(),
        'origin_subtype': _safe_str(record.get('origin_subtype')).strip(),
        'is_engineered': bool(record.get('is_engineered')),
        'hmdb_ids': hmdb_ids,
        'mapped_metabolites': [
            _standardize_metabolite_name(_safe_str(item).strip(), feature_token=feature_token)
            for item in _safe_list(record.get('mapped_metabolites'))
            if _safe_str(item).strip()
        ],
        'mapped_pathways': [
            _safe_str(item).strip()
            for item in _safe_list(record.get('mapped_pathways'))
            if _safe_str(item).strip()
        ],
        'prior_supported': bool(record.get('prior_supported')),
        'matched_phase0_biomarker': _safe_str(record.get('matched_phase0_biomarker')).strip() or None,
        'matched_pathway': _safe_str(record.get('matched_pathway')).strip() or None,
        'evidence_tags': evidence_tags,
    }


def _build_feature_dictionary(feature_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_build_feature_report_entry(_safe_dict(record)) for record in feature_records if isinstance(record, dict)]


def _build_panel_name_summary(feature_dictionary: List[Dict[str, Any]]) -> Dict[str, Any]:
    readable_names = [
        _safe_str(item.get('preferred_report_name')).strip()
        for item in feature_dictionary
        if _safe_str(item.get('preferred_report_name')).strip()
    ]
    raw_names = [
        _safe_str(item.get('preferred_report_name')).strip()
        for item in feature_dictionary
        if not bool(item.get('is_engineered')) and _safe_str(item.get('preferred_report_name')).strip()
    ]
    engineered_names = [
        _safe_str(item.get('preferred_report_name')).strip()
        for item in feature_dictionary
        if bool(item.get('is_engineered')) and _safe_str(item.get('preferred_report_name')).strip()
    ]
    return {
        'readable_feature_names': readable_names,
        'raw_feature_names': raw_names,
        'engineered_feature_names': engineered_names,
        'raw_count': len(raw_names),
        'engineered_count': len(engineered_names),
    }


def _build_preprocessing_summary(
    *,
    missing_value_metadata: Dict[str, Any],
    modeling_readiness_report: Dict[str, Any],
    data_analysis_for_modeling: Dict[str, Any],
    preprocessing_report: Dict[str, Any],
    preprocessing_decision_pack: Dict[str, Any],
) -> Dict[str, Any]:
    missing_meta = _safe_dict(missing_value_metadata)
    readiness = _safe_dict(modeling_readiness_report)
    analysis = _safe_dict(data_analysis_for_modeling)
    data_quality = _safe_dict(analysis.get('data_quality'))
    report = _safe_dict(preprocessing_report)
    decision_pack = _safe_dict(preprocessing_decision_pack)
    reportable_facts = _safe_dict(
        report.get('phase4_reportable_facts') or decision_pack.get('phase4_reportable_facts')
    )
    decision_context = _safe_dict(decision_pack.get('context'))
    decisions = _safe_dict(decision_pack.get('decisions'))
    thresholds = _safe_dict(report.get('thresholds'))

    zero_handling_strategy = _safe_str(_coalesce_non_null(
        reportable_facts.get('zero_handling_strategy'),
        decisions.get('zero_handling_strategy'),
    )).strip()
    imputation_method = _safe_str(_coalesce_non_null(
        reportable_facts.get('imputation_method'),
        decisions.get('imputation_method'),
        missing_meta.get('imputation_method'),
    )).strip()
    columns_imputed = missing_meta.get('columns_imputed')
    if columns_imputed is None:
        columns_imputed = missing_meta.get('cols_imputed', 0)
    rows_removed = missing_meta.get('rows_removed')
    columns_removed = missing_meta.get('columns_removed', missing_meta.get('cols_removed'))
    columns_with_missing = _safe_dict(next(
        (step for step in _safe_list(missing_meta.get('steps')) if _safe_str(_safe_dict(step).get('action')) == 'mnar_assessment'),
        {}
    )).get('columns_with_missing', data_quality.get('missing_values'))
    if not imputation_method:
        if columns_with_missing in (0, '0', None) and columns_imputed in (0, '0', None):
            imputation_method = 'none_needed'
        elif readiness.get('missing_values') in (0, '0') or data_quality.get('missing_values') in (0, '0'):
            imputation_method = 'none_needed'
        else:
            imputation_method = 'unknown'

    normalization_method = _safe_str(_coalesce_non_null(
        reportable_facts.get('normalization_method'),
        decisions.get('normalization_method'),
        'not_recorded',
    )).strip() or 'not_recorded'
    transformation_method = _safe_str(_coalesce_non_null(
        reportable_facts.get('transformation_method'),
        decisions.get('transformation_method'),
        'not_recorded',
    )).strip() or 'not_recorded'
    outlier_method = _safe_str(_coalesce_non_null(
        reportable_facts.get('outlier_method'),
        decisions.get('outlier_method'),
        'not_recorded',
    )).strip() or 'not_recorded'
    sample_missingness_threshold = _coalesce_non_null(
        reportable_facts.get('sample_missing_rate_threshold'),
        thresholds.get('sample_missing_rate_threshold'),
        0.35,
    )
    feature_missingness_threshold = _coalesce_non_null(
        reportable_facts.get('feature_missing_rate_threshold'),
        thresholds.get('feature_missing_rate_threshold'),
        0.3,
    )
    batch_correction_applied = bool(_coalesce_non_null(
        reportable_facts.get('batch_correction_applied'),
        report.get('batch_correction_applied'),
        False,
    ))
    qc_rsd_filter_applied = bool(_coalesce_non_null(
        reportable_facts.get('qc_rsd_filter_applied'),
        report.get('qc_rsd_filter_applied'),
        False,
    ))
    qc_rsd_threshold = _coalesce_non_null(
        reportable_facts.get('qc_rsd_threshold'),
        report.get('qc_rsd_threshold'),
    )

    suspected_mechanism = _safe_str(decisions.get('suspected_missingness_mechanism')).strip()
    normalized_outlier_label = 'IQR' if outlier_method.lower() == 'iqr' else outlier_method

    configured_workflow: List[str] = []
    if normalization_method not in {'', 'not_recorded', 'unknown'}:
        configured_workflow.append(f'{normalization_method} normalization')
    if transformation_method not in {'', 'not_recorded', 'unknown'}:
        configured_workflow.append(f'{transformation_method} transformation')
    if outlier_method not in {'', 'not_recorded', 'unknown'}:
        configured_workflow.append(f'{normalized_outlier_label}-based outlier capping')
    configured_workflow.append('standard auto-scaling (z-score)')

    return {
        'context': {
            'branch_used': report.get('branch_used', decision_pack.get('branch_used')),
            'platform_source': _coalesce_non_null(
                reportable_facts.get('platform_source'),
                decision_context.get('platform_source'),
                'unknown',
            ),
            'data_level': _coalesce_non_null(
                reportable_facts.get('data_level'),
                decision_context.get('data_level'),
                'unknown',
            ),
            'has_pooled_qc': decision_context.get('has_pooled_qc'),
            'has_blank_samples': decision_context.get('has_blank_samples'),
            'has_run_order': decision_context.get('has_run_order'),
            'has_batch_metadata': decision_context.get('has_batch_metadata'),
        },
        'zero_handling': {
            'strategy': zero_handling_strategy or 'unknown',
            'performed': zero_handling_strategy not in {'', 'keep_zero', 'not_recorded', 'unknown'},
            'description': (
                'Zero values were reclassified as missing before formal missingness assessment.'
                if zero_handling_strategy == 'convert_zero_to_nan'
                else 'Zero values were retained as observed values during missingness assessment.'
                if zero_handling_strategy == 'keep_zero'
                else 'Zero-handling strategy was not explicitly recorded.'
            ),
        },
        'missingness_detection': {
            'sample_missingness_threshold': sample_missingness_threshold,
            'feature_missingness_threshold': feature_missingness_threshold,
            'sample_filter_step_recorded': True,
            'rows_removed': rows_removed,
            'columns_removed': columns_removed,
            'mnar_assessed': bool(missing_meta.get('mnar_detected', False)) or any(
                _safe_str(_safe_dict(step).get('action')) == 'mnar_assessment'
                for step in _safe_list(missing_meta.get('steps'))
            ) or bool(suspected_mechanism),
            'mnar_detected': bool(missing_meta.get('mnar_detected', False)) or suspected_mechanism.upper() == 'MNAR',
            'columns_with_missing': columns_with_missing,
            'suspected_mechanism': suspected_mechanism,
        },
        'imputation': {
            'method': imputation_method,
            'performed': imputation_method not in {'', 'none_needed', 'not_needed'},
            'recommended_family': decisions.get('recommended_imputation_family'),
            'selection_basis': {
                'mnar': 'QRILC / half-minimum fallback for below-detection-limit style MNAR patterns',
                'multivariate': 'KNN for correlated multivariate missingness when data size permits',
                'normal_like': 'mean imputation for approximately symmetric distributions',
                'skewed': 'median imputation for skewed distributions or as robust fallback',
            },
            'columns_imputed': columns_imputed,
            'remaining_missing_values': missing_meta.get('remaining_missing_values', data_quality.get('missing_values')),
        },
        'normalization_scaling': {
            'configured_workflow': configured_workflow,
            'normalization_method': normalization_method,
            'transformation_method': transformation_method,
            'outlier_method': outlier_method,
            'pqn_rationale': 'reduce sample-wise dilution/systematic effects before downstream modeling',
            'log2_rationale': 'stabilize variance and reduce right skew',
            'outlier_handling': 'IQR rule with winsorization-style capping',
            'scaling_method': 'standard',
            'scaling_description': 'z-score standardization applied to numeric metabolite features while preserving protected columns',
        },
        'qc_batch': {
            'batch_correction_applied': batch_correction_applied,
            'qc_rsd_filter_applied': qc_rsd_filter_applied,
            'qc_rsd_threshold': qc_rsd_threshold,
        },
        'report_artifacts': {
            'stage_status': _safe_dict(report.get('stage_status')),
            'warnings': _safe_list(report.get('warnings')),
        },
        'data_quality_snapshot': {
            'missing_values': data_quality.get('missing_values', readiness.get('missing_values')),
            'missing_percentage': data_quality.get('missing_percentage'),
            'constant_features': data_quality.get('constant_features', readiness.get('constant_features')),
            'data_ready': readiness.get('data_ready'),
        },
    }


def _build_stability_search_context(
    *,
    phase0_phase1_test_result: Dict[str, Any],
    phase2_result: Dict[str, Any],
    adaptive_selection_report: Dict[str, Any],
    selection_policy: Dict[str, Any],
) -> Dict[str, Any]:
    test_config = _safe_dict(phase0_phase1_test_result.get('test_config'))
    phase1_result = _safe_dict(phase0_phase1_test_result.get('phase1_result'))
    sop_config = _safe_dict(phase1_result.get('sop_config'))
    execution_notes = _safe_dict(
        phase1_result.get('execution_notes')
        or sop_config.get('execution_notes')
    )
    feature_selection_config = _safe_dict(
        phase0_phase1_test_result.get('feature_selection_config')
        or execution_notes.get('feature_selection_config')
    )
    stability_iterations = _safe_dict(feature_selection_config.get('stability_iterations'))
    n_selection_methods = _safe_dict(feature_selection_config.get('n_selection_methods'))
    dataset_fingerprint = _safe_dict(phase2_result.get('dataset_fingerprint'))
    class_counts = _safe_dict(dataset_fingerprint.get('class_counts'))
    imbalance_ratio = _safe_float(dataset_fingerprint.get('imbalance_ratio'))
    archived_methods = _safe_list(adaptive_selection_report.get('methods_used'))

    return {
        'source_available': bool(phase0_phase1_test_result),
        'test_config': {
            'disease_name': test_config.get('disease_name'),
            'data_path': test_config.get('data_path'),
            'target_column': test_config.get('target_column'),
        },
        'dataset_fingerprint': {
            'class_counts': class_counts,
            'imbalance_ratio': imbalance_ratio,
            'n_sum_features': dataset_fingerprint.get('n_sum_features'),
        },
        'imbalance_detected': imbalance_ratio is not None and imbalance_ratio > 2.0,
        'stability_iterations': stability_iterations.get('default'),
        'stability_iterations_description': stability_iterations.get('description'),
        'selector_family_count': n_selection_methods.get('default'),
        'selector_family_count_description': n_selection_methods.get('description'),
        'selector_family_options': _safe_dict(n_selection_methods.get('options')),
        'archived_methods_used': archived_methods,
        'archived_method_count': len(archived_methods),
        'selection_policy': selection_policy.get('selection_policy'),
        'selection_policy_reason': selection_policy.get('selection_policy_reason'),
    }


def build_report_context(collected_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize collected Phase 0-3 artifacts into report_context v1."""

    phase0 = _safe_dict(collected_artifacts.get('phase0'))
    phase1 = _safe_dict(collected_artifacts.get('phase1'))
    phase2 = _safe_dict(collected_artifacts.get('phase2'))
    phase3 = _safe_dict(collected_artifacts.get('phase3'))
    paths = _safe_dict(collected_artifacts.get('paths'))
    source_registry = _safe_dict(collected_artifacts.get('source_registry'))
    collection_summary = _safe_dict(collected_artifacts.get('collection_summary'))
    config_path = _safe_str(collected_artifacts.get('config_path') or paths.get('config_path') or 'config.yaml').strip() or 'config.yaml'
    try:
        config_manager = ConfigManager(config_path=config_path)
    except Exception:
        config_manager = None

    phase1_panel_scores = _safe_dict(phase1.get('panel_scores'))
    feature_provenance = _safe_dict(phase1.get('feature_provenance'))
    phase2_result = _safe_dict(phase2.get('result'))
    phase2_search_summary = _safe_dict(phase2.get('search_summary'))
    phase2_bio_debug_artifact = _safe_dict(phase2.get('bio_debug'))
    phase2_clinical_utility = _safe_dict(phase2.get('clinical_utility'))
    phase2_probability_recalibration = _safe_dict(phase2.get('probability_recalibration'))
    phase2_bio_calibration = _safe_dict(phase2.get('bio_calibration'))
    figure_manifest = _safe_dict(phase3.get('figure_manifest'))
    phase0_defaults_config = config_manager.get_phase0_defaults() if config_manager else {}
    phase0_stage1_config = config_manager.get_phase0_stage1_config() if config_manager else {}
    phase0_literature_config = config_manager.get_phase0_literature_config() if config_manager else {}
    phase1_model_report = _read_json_if_exists(
        str(Path.cwd() / 'output' / 'phase1' / 'intermediate' / 'latest' / 'model_artifacts' / 'model_report.json')
    )
    phase0_phase1_test_result = _safe_dict(phase1.get('phase0_phase1_test_result'))

    phase0_biomarker_summary = _normalize_confirmed_biomarkers(phase0)
    confirmed_biomarkers = _safe_list(phase0_biomarker_summary.get('records'))
    confirmed_biomarker_names = _safe_list(phase0_biomarker_summary.get('names'))
    confirmed_biomarker_ids = _safe_list(phase0_biomarker_summary.get('ids'))
    final_priors = _safe_list(phase0_biomarker_summary.get('final_priors'))
    phase0_pathways = _extract_pathway_names(phase0)
    pathway_index = _build_name_index(phase0_pathways)
    biomarker_index = _build_name_index(confirmed_biomarker_names + confirmed_biomarker_ids + final_priors)

    winner = _safe_dict(phase2_result.get('final_result'))
    phase1_selected_features = _safe_list(
        phase1_panel_scores.get('selected_features') or phase2_result.get('phase1_selected_features')
    )
    winner_features = _safe_list(winner.get('features') or phase2_result.get('selected_features'))

    feature_records = _materialize_feature_records(
        selected_features=phase1_selected_features,
        feature_provenance=feature_provenance,
        pathway_index=pathway_index,
        biomarker_index=biomarker_index,
    )
    feature_provenance_summary = _safe_dict(feature_provenance.get('summary')) or _summarize_feature_records(feature_records)
    winner_feature_records = _select_feature_records_with_fallback(
        feature_names=winner_features,
        available_records=feature_records,
        pathway_index=pathway_index,
        biomarker_index=biomarker_index,
    )
    winner_feature_dictionary = _build_feature_dictionary(winner_feature_records)
    winner_panel_name_summary = _build_panel_name_summary(winner_feature_dictionary)
    figure_records = _safe_list(figure_manifest.get('figures'))
    phase2_search_summary = phase2_search_summary or _build_phase2_search_summary_fallback(phase2_result, winner)

    feature_selection_summary = _safe_dict(phase1.get('feature_selection_summary'))
    final_selection_summary = _safe_dict(phase1.get('final_selection_summary'))
    adaptive_selection_report = _safe_dict(phase1.get('adaptive_selection_report'))
    missing_value_metadata = _safe_dict(phase1.get('missing_value_metadata'))
    preprocessing_report = _safe_dict(phase1.get('preprocessing_report'))
    preprocessing_decision_pack = _safe_dict(phase1.get('preprocessing_decision_pack'))
    modeling_readiness_report = _safe_dict(phase1.get('modeling_readiness_report'))
    data_analysis_for_modeling = _safe_dict(phase1.get('data_analysis_for_modeling'))
    selection_policy = _infer_phase1_selection_policy(feature_selection_summary.get('feature_selection_strategy'))
    phase1_stability_search_context = _build_stability_search_context(
        phase0_phase1_test_result=phase0_phase1_test_result,
        phase2_result=phase2_result,
        adaptive_selection_report=adaptive_selection_report,
        selection_policy=selection_policy,
    )
    figure_section_summary = _build_figure_section_summary(figure_records)
    top_winner_pathways = _top_pathway_candidates(winner_feature_records)
    phase2_bio_score_version = _safe_str(winner.get('bio_score_version') or phase2_result.get('bio_score_version'))
    phase2_bio_debug = _safe_dict(
        phase2_bio_debug_artifact.get('winner_bio_debug')
        or winner.get('bio_debug')
        or phase2_result.get('bio_debug')
    )
    incremental_value = _safe_dict(
        winner.get('incremental_value')
        or phase2_result.get('incremental_value')
    )
    protected_anchor_features = _safe_list(
        winner.get('protected_anchor_features') or phase2_result.get('protected_anchor_features')
    )
    missing_protected_anchor_features = _safe_list(
        winner.get('missing_protected_anchor_features') or phase2_result.get('missing_protected_anchor_features')
    )
    n_protected_anchor_features = winner.get(
        'n_protected_anchor_features',
        phase2_result.get('n_protected_anchor_features', len(protected_anchor_features)),
    )
    memory_retrieval_summary = _safe_dict(_safe_dict(phase2_result.get('memory_debug')).get('phase2'))
    memory_prior_summary = {
        'memory_prior_enabled': bool(phase2_search_summary.get('memory_prior_enabled', False)),
        'memory_prior_confidence': phase2_search_summary.get('memory_prior_confidence'),
        'memory_matched_case_ids': _safe_list(phase2_search_summary.get('memory_matched_cases')),
        'memory_reranked_candidate_count': phase2_search_summary.get('memory_reranked_candidate_count'),
        'memory_anchor_hints': _safe_list(phase2_search_summary.get('memory_anchor_hints')),
        'memory_applied_actions': _safe_list(phase2_search_summary.get('memory_applied_actions')),
        'retrieval_summary': memory_retrieval_summary,
        'memory_case_id': _safe_str(phase2_result.get('memory_case_id')).strip(),
    }
    phase0_selection_threshold = _safe_dict(phase0.get('selection_threshold'))
    phase0_runtime_pathway_summary = _safe_dict(phase0.get('runtime_pathway_summary'))
    phase0_screening_summary = _safe_dict(phase0.get('screening_summary'))
    if phase0_pathways:
        if not _safe_list(phase0_runtime_pathway_summary.get('disease_core_pathways')):
            phase0_runtime_pathway_summary['disease_core_pathways'] = phase0_pathways
        if not _safe_list(phase0_runtime_pathway_summary.get('top_pathways')):
            phase0_runtime_pathway_summary['top_pathways'] = phase0_pathways
    preprocessing_summary = _build_preprocessing_summary(
        missing_value_metadata=missing_value_metadata,
        modeling_readiness_report=modeling_readiness_report,
        data_analysis_for_modeling=data_analysis_for_modeling,
        preprocessing_report=preprocessing_report,
        preprocessing_decision_pack=preprocessing_decision_pack,
    )
    figure_evidence_summary = _build_figure_evidence_summary(
        figure_records=figure_records,
        task_records=_safe_list(figure_manifest.get('task_records')),
        feature_selection_summary=feature_selection_summary,
        adaptive_selection_report=adaptive_selection_report,
        phase2_search_summary=phase2_search_summary,
        phase2_clinical_utility=phase2_clinical_utility,
        winner_feature_dictionary=winner_feature_dictionary,
        winner_perf=winner.get('perf', _safe_dict(phase2_result.get('scores')).get('f_perf')),
    )
    phase2_strategy_summary = {
        'clinical_scenario': phase2_search_summary.get('clinical_scenario') or phase2_result.get('clinical_scenario'),
        'clinical_scenario_definition': _safe_dict(
            phase2_search_summary.get('clinical_scenario_definition')
            or phase2_result.get('clinical_scenario_definition')
        ),
        'topsis_weights': _normalize_topsis_weights(_safe_dict(phase2_search_summary.get('topsis_weights'))),
        'topsis_weight_source': phase2_search_summary.get('topsis_weight_source'),
        'soft_epsilon_feasible_enabled': bool(phase2_search_summary.get('soft_epsilon_feasible_enabled', False)),
        'soft_epsilon_perf_margin': phase2_search_summary.get('soft_epsilon_perf_margin'),
        'soft_epsilon_perf_margin_source': phase2_search_summary.get('soft_epsilon_perf_margin_source'),
    }

    missing_required_inputs = [
        name for name, value in {
            'phase0_output': bool(phase0),
            'phase1_panel_scores': bool(phase1_panel_scores),
            'feature_provenance': bool(feature_records),
            'phase2_result': bool(phase2_result),
            'phase2_search_summary': bool(phase2_search_summary),
            'figure_manifest': bool(figure_manifest),
        }.items() if not value
    ]

    return {
        'schema_version': 'phase4.report_context.v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'run': {
            'run_id': collected_artifacts.get('run_id', ''),
            'report_dir': collected_artifacts.get('report_dir', ''),
            'config_path': collected_artifacts.get('config_path', 'config.yaml'),
        },
        'sources': paths,
        'source_registry': source_registry,
        'collection_summary': collection_summary,
        'phase0': {
            'disease_name': phase0.get('disease_name'),
            'normalized_disease_name': phase0.get('normalized_disease_name'),
            'confirmed_biomarkers': confirmed_biomarkers,
            'confirmed_biomarker_names': confirmed_biomarker_names,
            'confirmed_biomarker_ids': confirmed_biomarker_ids,
            'confirmed_biomarker_count': len(confirmed_biomarkers),
            'final_priors': final_priors,
            'final_priors_count': len(final_priors),
            'selection_threshold': phase0_selection_threshold,
            'runtime_pathway_summary': phase0_runtime_pathway_summary,
            'disease_core_pathways': phase0_pathways,
            'top_pathways': phase0_pathways,
        },
        'phase1': {
            'selected_feature_count': phase1_panel_scores.get('panel_size', len(phase1_selected_features)),
            'selected_features': phase1_selected_features,
            'panel_scores': _safe_dict(phase1_panel_scores.get('scores')),
            'feature_provenance_summary': feature_provenance_summary,
            'feature_provenance_notes': _safe_list(feature_provenance.get('notes')),
            'engineered_feature_registry_path': feature_provenance.get('engineered_feature_registry_path'),
            'feature_records': feature_records,
            'missing_value_metadata': missing_value_metadata,
            'preprocessing_report': preprocessing_report,
            'preprocessing_decision_pack': preprocessing_decision_pack,
            'modeling_readiness_report': modeling_readiness_report,
            'data_analysis_for_modeling': data_analysis_for_modeling,
            'preprocessing_summary': preprocessing_summary,
            'engineering_profile': {
                'raw_count': feature_provenance_summary.get('raw_count', 0),
                'engineered_ratio_count': feature_provenance_summary.get('engineered_ratio_count', 0),
                'engineered_sum_count': feature_provenance_summary.get('engineered_sum_count', 0),
                'engineered_pathway_count': feature_provenance_summary.get('engineered_pathway_count', 0),
                'taxonomy_sum_count': feature_provenance_summary.get('taxonomy_sum_count', 0),
                'group_sum_count': feature_provenance_summary.get('group_sum_count', 0),
                'prior_supported_count': feature_provenance_summary.get('prior_supported_count', 0),
            },
            'selection_strategy': {
                'feature_selection_summary': feature_selection_summary,
                'final_selection_summary': final_selection_summary,
                'adaptive_selection_report': adaptive_selection_report,
                **selection_policy,
                'methods_used': _safe_list(adaptive_selection_report.get('methods_used')),
                'fallback_strategy_used': bool(final_selection_summary.get('fallback_strategy_used', False)),
                'fallback_selection_method': final_selection_summary.get('selection_method'),
                'union_features_count': feature_selection_summary.get('union_features_count'),
                'intersection_all_3_count': feature_selection_summary.get('intersection_all_3_count'),
                'intersection_2_of_3_count': feature_selection_summary.get('intersection_2_of_3_count'),
                'consensus_features_count': adaptive_selection_report.get('consensus_features_count'),
            },
            'stability_search_context': phase1_stability_search_context,
        },
        'phase2': {
            'winner_features': winner_features,
            'winner_feature_count': len(winner_features),
            'winner_feature_records': winner_feature_records,
            'winner_feature_dictionary': winner_feature_dictionary,
            'winner_panel_name_summary': winner_panel_name_summary,
            'phase1_candidate_pool_count': phase2_result.get('phase1_output_size', len(_safe_list(phase2_result.get('phase1_selected_features')))),
            'dataset_fingerprint': _safe_dict(phase2_result.get('dataset_fingerprint')),
            'bio_score_version': phase2_bio_score_version,
            'bio_debug': phase2_bio_debug,
            'bio_debug_artifact': phase2_bio_debug_artifact,
            'bio_calibration': phase2_bio_calibration,
            'incremental_value': incremental_value,
            'bio_context': _safe_dict(phase2_bio_debug_artifact.get('bio_context')),
            'search_model_bio_debug': _safe_dict(phase2_bio_debug_artifact.get('search_model_bio_debug')),
            'phase1_model_bio_debug': _safe_dict(phase2_bio_debug_artifact.get('phase1_model_bio_debug')),
            'protected_anchor_features': protected_anchor_features,
            'n_protected_anchor_features': n_protected_anchor_features,
            'missing_protected_anchor_features': missing_protected_anchor_features,
            'memory_prior_summary': memory_prior_summary,
            'strategy_summary': phase2_strategy_summary,
            'n_added_features_beyond_protected': winner.get(
                'n_added_features_beyond_protected',
                phase2_result.get('n_added_features_beyond_protected'),
            ),
            'winner_scores': {
                'f_perf': winner.get('perf', _safe_dict(phase2_result.get('scores')).get('f_perf')),
                'f_bio': winner.get('bio', _safe_dict(phase2_result.get('scores')).get('f_bio')),
                'f_cost': winner.get('cost', _safe_dict(phase2_result.get('scores')).get('f_cost')),
                'f_corr': winner.get('corr', _safe_dict(phase2_result.get('scores')).get('f_corr')),
            },
            'selected_model': winner.get('selected_model') or phase2_result.get('selected_model') or phase2_result.get('champion_model_family'),
            'search_summary': phase2_search_summary,
            'clinical_utility': phase2_clinical_utility,
            'probability_recalibration': phase2_probability_recalibration,
            'runtime_summary': {
                'search_time_seconds': phase2_result.get('search_time_seconds'),
                'total_time_seconds': phase2_result.get('total_time_seconds'),
                'k_folds': phase2_result.get('k_folds'),
                'clinical_scenario': phase2_result.get('clinical_scenario'),
            },
        },
        'phase3': {
            'figure_count': len(figure_records),
            'figures': figure_records,
            'task_records': _safe_list(figure_manifest.get('task_records')),
            'figures_by_section': figure_section_summary,
            'figure_evidence_summary': figure_evidence_summary,
            'summary_markdown': phase3.get('pipeline_summary_markdown', ''),
        },
        'report_inputs': {
            'executive_summary': {
                'queue_size': len(final_priors or confirmed_biomarker_names),
                'confirmed_biomarker_count': len(confirmed_biomarkers),
                'final_priors_count': len(final_priors or confirmed_biomarker_names),
                'final_feature_count': len(winner_features),
                'best_auc': winner.get('perf', _safe_dict(phase2_result.get('scores')).get('f_perf')),
                'core_pathways': top_winner_pathways or phase0_pathways[:5],
                'selected_model': winner.get('selected_model') or phase2_result.get('selected_model') or phase2_result.get('champion_model_family'),
                'bio_score_version': phase2_bio_score_version,
                'winner_feature_dictionary': winner_feature_dictionary,
                'winner_panel_name_summary': winner_panel_name_summary,
                'figure_evidence_summary': figure_evidence_summary,
                'protected_anchor_count': n_protected_anchor_features,
                'anchor_mode': _safe_dict(phase2_search_summary.get('bio_context_summary')).get('anchor_mode'),
                'memory_prior_enabled': memory_prior_summary.get('memory_prior_enabled'),
                'nri': _safe_dict(incremental_value.get('nri')),
                'idi': _safe_dict(incremental_value.get('idi')),
            },
            'methodology': {
                'phase0': {
                    'disease_name': phase0.get('disease_name'),
                    'phase0_prior_count': len(final_priors or confirmed_biomarker_names),
                    'confirmed_biomarker_count': len(confirmed_biomarkers),
                    'confirmed_biomarkers': confirmed_biomarkers,
                    'final_priors': final_priors or confirmed_biomarker_names,
                    'target_pathway_count': len(phase0_pathways),
                    'target_pathways': phase0_pathways,
                    'selection_threshold': phase0_selection_threshold,
                    'screening_summary': phase0_screening_summary,
                    'runtime_pathway_summary': phase0_runtime_pathway_summary,
                    'literature_protocol': {
                        'query_field': phase0_literature_config.get('query_field', 'tiab'),
                        'top_k_abstracts': phase0_literature_config.get('top_k_abstracts'),
                        'max_abstracts_per_candidate': phase0_defaults_config.get('max_abstracts_per_candidate'),
                        'relevance_filter_enabled': phase0_literature_config.get('relevance_filter_enabled'),
                        'min_relevant_abstracts': phase0_literature_config.get('min_relevant_abstracts'),
                    },
                    'stage1_screening_policy': {
                        'hit_count_sources': _safe_list(phase0_stage1_config.get('hit_count_sources')),
                        'hard_drop_require_pathway_if_zero_hit': phase0_stage1_config.get('hard_drop_require_pathway_if_zero_hit'),
                        'pathway_relevance_min_score': phase0_stage1_config.get('pathway_relevance_min_score'),
                        'retain_low_hit_with_strong_pathway_only': phase0_stage1_config.get('retain_low_hit_with_strong_pathway_only'),
                        'low_hit_max_hits': phase0_stage1_config.get('low_hit_max_hits'),
                        'low_hit_pathway_relevance_min_score': phase0_stage1_config.get('low_hit_pathway_relevance_min_score'),
                    },
                    'rubric_framework': {
                        'dimensions': [
                            {'name': 'Clinical_Evidence', 'range': '0-3'},
                            {'name': 'Disease_Specificity', 'range': '0-3'},
                            {'name': 'Mechanistic_Plausibility', 'range': '0-3'},
                            {'name': 'Consistency', 'range': '-1 to 1'},
                        ],
                        'selection_threshold_rule': phase0_selection_threshold.get('rule'),
                        'selection_threshold_value': phase0_selection_threshold.get('threshold'),
                        'dynamic_weighting_enabled': True,
                        'dynamic_weighting_basis': 'literature hit-count zone',
                    },
                },
                'phase1': {
                    'phase1_selected_feature_count': len(phase1_selected_features),
                    'feature_engineering_summary': {
                        'raw_count': feature_provenance_summary.get('raw_count', 0),
                        'engineered_ratio_count': feature_provenance_summary.get('engineered_ratio_count', 0),
                        'engineered_sum_count': feature_provenance_summary.get('engineered_sum_count', 0),
                        'engineered_pathway_count': feature_provenance_summary.get('engineered_pathway_count', 0),
                        'taxonomy_sum_count': feature_provenance_summary.get('taxonomy_sum_count', 0),
                        'group_sum_count': feature_provenance_summary.get('group_sum_count', 0),
                    },
                    'feature_provenance_notes': _safe_list(feature_provenance.get('notes')),
                    'engineered_feature_registry_path': feature_provenance.get('engineered_feature_registry_path'),
                    'preprocessing_summary': preprocessing_summary,
                    'figure_evidence_summary': figure_evidence_summary,
                    'selection_policy': selection_policy.get('selection_policy'),
                    'selection_policy_reason': selection_policy.get('selection_policy_reason'),
                    'methods_used': _safe_list(adaptive_selection_report.get('methods_used')),
                    'stability_search_context': phase1_stability_search_context,
                    'feature_selection_summary': feature_selection_summary,
                    'adaptive_selection_report': adaptive_selection_report,
                    'fallback_strategy_used': bool(final_selection_summary.get('fallback_strategy_used', False)),
                    'fallback_selection_method': final_selection_summary.get('selection_method'),
                    'baseline_model_screening': {
                        'best_model': _safe_dict(_safe_dict(phase2_result.get('phase1_context')).get('training_results_json')).get('best_model')
                        or phase1_model_report.get('best_model'),
                        'num_models_trained': phase1_model_report.get('num_models_trained'),
                        'eval_metric': phase1_model_report.get('eval_metric')
                        or _safe_dict(_safe_dict(phase2_result.get('phase1_context')).get('training_results_json')).get('primary_metric'),
                        'best_model_is_weighted_ensemble': 'weightedensemble' in str(
                            _safe_dict(_safe_dict(phase2_result.get('phase1_context')).get('training_results_json')).get('best_model')
                            or phase1_model_report.get('best_model')
                            or ''
                        ).lower(),
                        'best_model_explanation': (
                            'The top-performing classifier was an AutoGluon WeightedEnsemble, '
                            'i.e., an automatically learned weighted combination of multiple '
                            'high-ranking base learners selected according to validation performance.'
                            if 'weightedensemble' in str(
                                _safe_dict(_safe_dict(phase2_result.get('phase1_context')).get('training_results_json')).get('best_model')
                                or phase1_model_report.get('best_model')
                                or ''
                            ).lower()
                            else ''
                        ),
                    },
                },
                'phase2': {
                    'disease_name': phase2_search_summary.get('disease_name') or phase0.get('disease_name'),
                    'phase2_input_feature_count': phase2_result.get('phase1_output_size', len(_safe_list(phase2_result.get('phase1_selected_features')))),
                    'dataset_fingerprint': _safe_dict(phase2_result.get('dataset_fingerprint')),
                    'winner_feature_count': len(winner_features),
                    'bio_score_version': phase2_bio_score_version,
                    'f_bio_v2_enabled': phase2_search_summary.get('f_bio_v2_enabled', phase2_bio_score_version == 'f_bio_v2'),
                    'search_start_mode': phase2_search_summary.get('search_start_mode'),
                    'protected_anchor_count': n_protected_anchor_features,
                    'protected_anchor_features': protected_anchor_features,
                    'missing_protected_anchor_features': missing_protected_anchor_features,
                    'anchor_mode': _safe_dict(phase2_search_summary.get('bio_context_summary')).get('anchor_mode'),
                    'bio_context': _safe_dict(phase2_bio_debug_artifact.get('bio_context')),
                    'bio_calibration': phase2_bio_calibration,
                    'probability_recalibration': phase2_probability_recalibration,
                    'scenario_strategy': phase2_strategy_summary,
                    'memory_prior_summary': memory_prior_summary,
                    'incremental_value': incremental_value,
                    'winner_feature_dictionary': winner_feature_dictionary,
                    'search_engine': phase2_search_summary.get('engine'),
                    'beam_width': phase2_search_summary.get('beam_width'),
                    'max_depth': phase2_search_summary.get('max_depth'),
                    'max_depth_additional_features': phase2_search_summary.get('max_depth_additional_features'),
                    'actual_depth': phase2_search_summary.get('actual_depth'),
                    'early_stopping_type': phase2_search_summary.get('early_stopping_type'),
                    'stop_reason': phase2_search_summary.get('stop_reason'),
                    'runtime_summary': {
                        'k_folds': phase2_result.get('k_folds'),
                        'patience': phase2_result.get('patience'),
                        'epsilon': phase2_result.get('epsilon'),
                    },
                },
                'phase2_search_summary': phase2_search_summary,
                'statistical_evaluation_framework': {
                    'k_folds': phase2_result.get('k_folds'),
                    'discrimination_metrics': ['ROC AUC'],
                    'incremental_metrics': ['continuous NRI', 'IDI'],
                    'clinical_utility_metrics': ['decision-curve analysis', 'threshold-dependent sensitivity/specificity/PPV/NPV'],
                    'calibration_metrics': ['calibration curve', 'Brier score', 'expected calibration error', 'calibration slope', 'calibration intercept'],
                    'interpretability_tools': ['SHAP global attribution', 'SHAP indexed waterfall example'],
                    'nonlinearity_tools': ['restricted cubic spline'],
                },
                'figure_evidence_summary': figure_evidence_summary,
            },
            'results': {
                'winner_panel': {
                    'selected_model': winner.get('selected_model') or phase2_result.get('selected_model') or phase2_result.get('champion_model_family'),
                    'winner_feature_count': len(winner_features),
                    'winner_features': winner_features,
                    'winner_feature_records': winner_feature_records,
                    'winner_feature_dictionary': winner_feature_dictionary,
                    'winner_panel_name_summary': winner_panel_name_summary,
                    'bio_score_version': phase2_bio_score_version,
                    'bio_debug': phase2_bio_debug,
                    'bio_debug_artifact': phase2_bio_debug_artifact,
                    'bio_calibration': phase2_bio_calibration,
                    'incremental_value': incremental_value,
                    'memory_prior_summary': memory_prior_summary,
                    'scenario_strategy': phase2_strategy_summary,
                    'protected_anchor_features': protected_anchor_features,
                    'n_protected_anchor_features': n_protected_anchor_features,
                    'missing_protected_anchor_features': missing_protected_anchor_features,
                    'winner_scores': {
                        'f_perf': winner.get('perf', _safe_dict(phase2_result.get('scores')).get('f_perf')),
                        'f_bio': winner.get('bio', _safe_dict(phase2_result.get('scores')).get('f_bio')),
                        'f_cost': winner.get('cost', _safe_dict(phase2_result.get('scores')).get('f_cost')),
                        'f_corr': winner.get('corr', _safe_dict(phase2_result.get('scores')).get('f_corr')),
                    },
                },
                'winner_features': winner_features,
                'feature_records': winner_feature_records or feature_records,
                'feature_dictionary': winner_feature_dictionary,
                'winner_panel_name_summary': winner_panel_name_summary,
                'phase1_candidate_pool_count': phase2_result.get('phase1_output_size', len(_safe_list(phase2_result.get('phase1_selected_features')))),
                'incremental_value': incremental_value,
                'memory_prior_summary': memory_prior_summary,
                'scenario_strategy': phase2_strategy_summary,
                'probability_recalibration': phase2_probability_recalibration,
                'figures': figure_records,
                'figure_count': len(figure_records),
                'figure_evidence_summary': figure_evidence_summary,
            },
            'clinical_utility': phase2_clinical_utility,
            'probability_recalibration': phase2_probability_recalibration,
            'discussion': {
                'disease_name': phase0.get('disease_name'),
                'winner_feature_dictionary': winner_feature_dictionary,
                'incremental_value': incremental_value,
                'clinical_utility': phase2_clinical_utility,
                'probability_recalibration': phase2_probability_recalibration,
                'figure_evidence_summary': figure_evidence_summary,
                'disease_core_pathways': phase0_pathways,
            },
        },
        'quality_checks': {
            'missing_required_inputs': missing_required_inputs,
            'missing_source_files': _safe_list(collection_summary.get('missing_sources')),
            'ready_for_llm': len(missing_required_inputs) == 0,
            'ready_for_rendering': bool(figure_records or winner_features),
        },
    }
