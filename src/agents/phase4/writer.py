"""
Phase 4 writers.

This module persists structured Phase 4 JSON artifacts. At this stage it emits
LLM-ready section contracts rather than final prose, so later LLM integration
can focus on writing while staying grounded in report_context facts.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .prompts import PHASE4_REPORT_SYSTEM_PROMPT, build_section_generation_prompt
from src.utils.config_manager import ConfigManager


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEEDED_LLM_POLICY = 'seeded_llm'
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')

logger = logging.getLogger(__name__)

DEFAULT_SECTION_KEYS = [
    'executive_summary',
    'methodology_workflow',
    'results',
    'clinical_utility_decision_support',
    'discussion_mechanistic_insights',
]

SECTION_TITLES = {
    'executive_summary': 'Executive Summary',
    'methodology_workflow': 'Methodology & Workflow',
    'results': 'Results',
    'clinical_utility_decision_support': 'Clinical Utility & Decision Support',
    'discussion_mechanistic_insights': 'Discussion & Evidence-scoped Biological Interpretation',
}


def _write_json(payload: Dict[str, Any], file_path: Path) -> str:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return str(file_path)


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_str(value: Any) -> str:
    if value is None:
        return ''
    return str(value)


_CANONICAL_DISEASE_NAMES = {
    'cfs': 'Myalgic encephalomyelitis/chronic fatigue syndrome (ME/CFS)',
    'me/cfs': 'Myalgic encephalomyelitis/chronic fatigue syndrome (ME/CFS)',
    'chronic fatigue syndrome': 'Myalgic encephalomyelitis/chronic fatigue syndrome (ME/CFS)',
    'hcc': 'Hepatocellular carcinoma',
    't2dm': 'Type 2 diabetes mellitus',
    'crc': 'Colorectal cancer',
    'luad': 'Lung adenocarcinoma',
    'pdac': 'Pancreatic ductal adenocarcinoma',
}


def _display_disease_name(value: Any) -> str:
    """Use a full disease name in reader-facing prose, never a run shorthand."""

    raw = _safe_str(value).strip()
    return _CANONICAL_DISEASE_NAMES.get(raw.lower(), raw or 'the target disease')


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any, digits: int = 3) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return 'N/A'
    return f'{numeric:.{digits}f}'


def _format_metric_or_unknown(value: Any, digits: int = 3, unknown: str = 'not archived') -> str:
    """Render missing evidence explicitly without programmer placeholders."""
    numeric = _safe_float(value)
    return unknown if numeric is None else f'{numeric:.{digits}f}'


def _select_calibration_payload(clinical_utility: Dict[str, Any]) -> Dict[str, Any]:
    """Return the first calibration payload with an actual metric.

    Historical runs often emit ``probability_calibration_adjusted`` as a
    non-empty placeholder with ``available=false`` and all metrics null.  A
    truthiness-based ``adjusted or primary`` lookup therefore hid the valid
    raw/primary OOF calibration and produced ``Brier not archived`` in the
    report.  Selection is now based on metric availability, not dictionary
    presence.
    """
    candidates = (
        _safe_dict(clinical_utility.get('probability_calibration_adjusted')),
        _safe_dict(clinical_utility.get('probability_calibration')),
        _safe_dict(clinical_utility.get('probability_calibration_raw')),
    )
    for payload in candidates:
        if _safe_float(payload.get('brier_score')) is not None:
            return payload
    for payload in candidates:
        if payload.get('available') is True:
            return payload
    return {}


def _repair_narrative_facts(report_context: Dict[str, Any], text: str) -> str:
    """Apply deterministic fact repairs to both seed and LLM prose.

    Phase 4 may be run with or without an LLM, but the same report can expose
    the generated section through HTML, Markdown and PDF.  This final repair
    layer prevents legacy/generated prose from reintroducing known conflicts
    after the structured facts have already been resolved.
    """
    value = _safe_str(text)
    if not value:
        return value
    report_inputs = _safe_dict(report_context.get('report_inputs'))
    phase0 = _safe_dict(report_context.get('phase0'))
    results = _safe_dict(report_inputs.get('results'))
    winner_panel = _safe_dict(results.get('winner_panel'))
    feature_dictionary = _safe_list(results.get('feature_dictionary'))
    final_priors = _safe_list(phase0.get('final_priors'))
    prior_supported_count = sum(
        1 for item in feature_dictionary
        if isinstance(item, dict) and bool(item.get('prior_supported'))
    )
    selected_model = _safe_str(
        _safe_dict(report_context.get('phase2')).get('selected_model')
        or winner_panel.get('selected_model')
    ).strip()

    # Repair the specific malformed Phase 0 sentence emitted by older writer
    # versions when stage1_candidate_count was absent.
    if re.search(r'not archived advanced after title/abstract-level evidence triage', value, flags=re.IGNORECASE):
        methodology_phase0 = _safe_dict(_safe_dict(report_inputs.get('methodology')).get('phase0'))
        phase0_count = _coalesce_non_null(
            _safe_dict(methodology_phase0.get('screening_summary')).get('candidate_count'),
            phase0.get('confirmed_biomarker_count'),
            len(_safe_list(phase0.get('confirmed_biomarkers'))) or None,
        )
        retained_count = _coalesce_non_null(
            phase0.get('phase0_prior_count'),
            phase0.get('final_priors_count'),
            len(final_priors) or None,
        )
        replacement = (
            f'{phase0_count} candidates were evaluated in the archived Phase 0 screening workflow, and '
            f'{retained_count} high-confidence priors were ultimately retained for downstream modeling.'
            if phase0_count is not None and retained_count is not None
            else 'The archived Phase 0 screening workflow retained the confirmed prior set for downstream modeling.'
        )
        value = re.sub(
            r'[^.]*not archived advanced after title/abstract-level evidence triage[^.]*\.',
            replacement,
            value,
            flags=re.IGNORECASE,
        )

    if 'selected not archived as the reference engine' in value.lower():
        if selected_model:
            value = re.sub(
                r'In parallel, an AutoGluon-based baseline screening stage compared learner families using not archived, and selected not archived as the reference engine for downstream panel optimization\.',
                f'The archived baseline screening stage carried forward {selected_model} as the reference engine for downstream panel optimization; learner-level comparison metrics were not archived.',
                value,
                flags=re.IGNORECASE,
            )
        else:
            value = re.sub(
                r'In parallel, an AutoGluon-based baseline screening stage compared learner families using not archived, and selected not archived as the reference engine for downstream panel optimization\.',
                'The archived baseline screening stage was referenced, but its selected learner and learner-level comparison metrics were not archived.',
                value,
                flags=re.IGNORECASE,
            )

    if final_priors and re.search(r'none of the \d+ archived Phase 0 priors were retained', value, flags=re.IGNORECASE):
        value = re.sub(
            r'meaning that none of the \d+ archived Phase 0 priors were retained in the final panel',
            f'this corresponds to {prior_supported_count} of the {len(final_priors)} archived Phase 0 priors represented in the final panel',
            value,
            flags=re.IGNORECASE,
        )

    clinical_utility = _safe_dict(report_inputs.get('clinical_utility'))
    calibration = _select_calibration_payload(clinical_utility)
    brier = _safe_float(calibration.get('brier_score'))
    slope = _safe_float(calibration.get('calibration_slope'))
    intercept = _safe_float(calibration.get('calibration_intercept'))
    if brier is not None:
        value = re.sub(r'\bBrier(?: score)? not archived\b', f'Brier score {brier:.3f}', value, flags=re.IGNORECASE)
        value = re.sub(
            r'Internal calibration metrics were Brier not archived, slope not archived, and intercept not archived\.',
            f'Internal raw OOF calibration metrics were Brier {brier:.3f}, slope {slope:.3f} and intercept {intercept:.3f}.' if slope is not None and intercept is not None else f'Internal calibration Brier score was {brier:.3f}; slope and intercept were not archived.',
            value,
            flags=re.IGNORECASE,
        )
    return _sanitize_narrative_text(value)


def _infer_cv_folds(report_context: Dict[str, Any]) -> Optional[int]:
    """Infer a declared fold count from archived protocol strings/fields."""
    # Only inspect fields whose *keys* declare a fold count.  The previous
    # recursive implementation appended every string in the context, so a
    # PubMed PMID such as ``36184594`` was interpreted as the number of CV
    # folds and rendered as ``36184594-fold``.
    candidates: List[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).lower()
                if key_text in {'k_folds', 'cv_folds', 'n_folds', 'folds', 'n_splits', 'cv_splits'}:
                    candidates.append(item)
                elif any(token in key_text for token in ('fold_count', 'nfold', 'n_fold')):
                    candidates.append(item)
                # A protocol string may be stored under a descriptive key,
                # but arbitrary values (PMIDs, paths, feature names) must not
                # be treated as fold counts.
                if key_text in {'protocol', 'cv_protocol', 'validation_protocol', 'performance_protocol', 'description'}:
                    if isinstance(item, str):
                        candidates.append(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(report_context)
    for value in candidates:
        try:
            number = int(value)
            if number > 1:
                return number
        except (TypeError, ValueError):
            pass
        match = re.search(r'(?<!\d)(\d+)\s*[-_ ]?fold(?:s)?', str(value), flags=re.IGNORECASE)
        if match:
            number = int(match.group(1))
            if number > 1:
                return number
    return None


def _infer_holdout_auc(report_context: Dict[str, Any]) -> Optional[float]:
    """Read the archived internal-holdout AUC without treating it as external."""
    phase3 = _safe_dict(report_context.get('phase3'))
    for item in _safe_list(phase3.get('task_records')):
        record = _safe_dict(item)
        if record.get('task') == 'plot_final_holdout_roc':
            metadata = _safe_dict(record.get('renderer_metadata'))
            value = _safe_float(metadata.get('pooled_auc') or metadata.get('auc'))
            if value is not None:
                return value
    for key in ('internal_holdout_auc', 'holdout_auc'):
        value = _safe_float(phase3.get(key))
        if value is not None:
            return value
    return None


def _phase1_handoff_facts(phase1: Dict[str, Any], phase2: Dict[str, Any]) -> Dict[str, Any]:
    """Separate measured metabolites, engineered candidates and final handoff."""
    engineering = _safe_dict(phase1.get('feature_engineering_summary'))
    raw = _safe_float(engineering.get('raw_count'))
    engineered_parts = [
        _safe_float(engineering.get('engineered_sum_count')) or 0,
        _safe_float(engineering.get('engineered_ratio_count')) or 0,
        _safe_float(engineering.get('engineered_pathway_count')) or 0,
    ]
    engineered = int(sum(engineered_parts)) if any(item > 0 for item in engineered_parts) else 0
    selected = _safe_float(
        phase2.get('phase2_input_feature_count')
        if phase2.get('phase2_input_feature_count') not in (None, '')
        else phase1.get('phase1_selected_feature_count')
    )
    if selected is None:
        selected = _safe_float(phase1.get('selected_feature_count'))
    if (selected is None or selected == 0) and raw is not None:
        selected = raw + engineered
    if raw is None and selected is not None and engineered == 0:
        raw = selected
    return {
        'raw_count': int(raw) if raw is not None else None,
        'engineered_count': engineered,
        'handoff_count': int(selected) if selected is not None else None,
    }


def _coalesce_non_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == '':
            continue
        return value
    return None


def _format_ci_range(value: Any, digits: int = 3) -> str:
    points = _safe_list(value)
    if len(points) != 2:
        return 'N/A'
    left = _safe_float(points[0])
    right = _safe_float(points[1])
    if left is None or right is None:
        return 'N/A'
    return f'{left:.{digits}f} to {right:.{digits}f}'


def _format_threshold_range(ranges: Any) -> str:
    normalized = _safe_list(ranges)
    if not normalized:
        return 'none identified'
    rendered: List[str] = []
    for item in normalized:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            left = _format_metric(item[0], 2)
            right = _format_metric(item[1], 2)
            rendered.append(f'{left}-{right}')
    return ', '.join(rendered) if rendered else 'none identified'


def _build_threshold_metrics_summary(threshold_metrics: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    lines: List[str] = []
    for row in threshold_metrics[:limit]:
        if not isinstance(row, dict):
            continue
        threshold = _format_metric(row.get('threshold'), 2)
        sensitivity = _format_metric(row.get('sensitivity'))
        specificity = _format_metric(row.get('specificity'))
        ppv = _format_metric(row.get('ppv'))
        npv = _format_metric(row.get('npv'))
        flagged = _format_metric(row.get('flagged_rate'))
        lines.append(
            f'Threshold {threshold}: sensitivity {sensitivity}, specificity {specificity}, '
            f'PPV {ppv}, NPV {npv}, flagged rate {flagged}.'
        )
    return lines


def _format_percentage(value: Any, digits: int = 1) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return 'N/A'
    return f'{numeric * 100:.{digits}f}%'


def _join_readable_list(items: List[str], max_items: Optional[int] = None) -> str:
    normalized = [str(item).strip() for item in items if str(item).strip()]
    if max_items is not None and len(normalized) > max_items:
        normalized = normalized[:max_items]
    if not normalized:
        return 'none'
    if len(normalized) == 1:
        return normalized[0]
    if len(normalized) == 2:
        return f'{normalized[0]} and {normalized[1]}'
    return ', '.join(normalized[:-1]) + f', and {normalized[-1]}'


def _is_opaque_feature_label(value: Any) -> bool:
    label = _safe_str(value).strip()
    if not label:
        return True
    if re.fullmatch(r'HMDB\d+(?:\.\d+)?', label, flags=re.IGNORECASE):
        return True
    return False


def _readable_method_name(value: Any) -> str:
    raw = _safe_str(value).strip()
    if not raw:
        return ''
    alias_map = {
        'elasticnet': 'Elastic Net Logistic',
        'run_elasticnet_selector': 'Elastic Net Logistic',
        'elastic_net_logistic': 'Elastic Net Logistic',
        'random_forest': 'Random Forest',
        'run_random_forest_selector': 'Random Forest',
        'lasso': 'L1 Logistic/Lasso',
        'run_lasso_selector': 'L1 Logistic/Lasso',
        'l1_logistic_lasso': 'L1 Logistic/Lasso',
        'mrmr': 'mRMR',
        'run_mrmr_selector': 'mRMR',
        'lightgbm': 'LightGBM',
        'run_lightgbm_selector': 'LightGBM',
        't_test': 'Welch t-test',
        'run_t_test_selector': 'Welch t-test',
        'welch_t_test': 'Welch t-test',
        'fdr_effect_size': 'FDR/effect-size',
        'run_fdr_effect_size_selector': 'FDR/effect-size',
    }
    normalized = raw.lower()
    if normalized in alias_map:
        return alias_map[normalized]
    return raw.replace('_', ' ')


def _format_class_counts(class_counts: Dict[str, Any]) -> str:
    parts: List[str] = []
    for label, count in _safe_dict(class_counts).items():
        if count is None or count == '':
            continue
        parts.append(f'{label}={count}')
    return ', '.join(parts)


def _is_weighted_ensemble_model(model_name: Any) -> bool:
    return 'weightedensemble' in _safe_str(model_name).strip().lower()


def _baseline_model_screening_sentence(screening: Dict[str, Any], fallback_model: Any = None) -> str:
    """Describe baseline screening without turning missing fields into prose.

    Some runs archive the selected model in the Phase 2 winner artifact while
    the optional Phase 1 baseline summary has ``best_model=null``.  In that
    case we report the run-aligned selected model and explicitly qualify the
    missing learner-level comparison metrics.
    """
    eval_metric = screening.get('eval_metric') or 'the prespecified discrimination metric'
    best_model = screening.get('best_model') or fallback_model
    if not best_model:
        return (
            'The archived baseline screening stage was referenced for model selection, '
            'but its selected learner name and learner-level comparison metrics were not archived.'
        )
    if not screening.get('eval_metric'):
        return (
            f'The archived baseline screening stage carried forward {best_model} as the reference engine for downstream panel optimization; '
            'the learner-level comparison metric was not archived.'
        )
    if _is_weighted_ensemble_model(best_model):
        return (
            f'In parallel, an AutoGluon-based baseline screening stage compared learner families using {eval_metric}, '
            f'and selected {best_model} as the reference engine for downstream panel optimization. '
            'Here, the AutoGluon WeightedEnsemble denotes an automatically learned weighted combination of multiple '
            'high-ranking base learners selected according to validation performance, rather than a single standalone learner.'
        )
    return (
        f'In parallel, an AutoGluon-based baseline screening stage compared learner families using {eval_metric}, '
        f'and selected {best_model} as the reference engine for downstream panel optimization.'
    )


def _top_ranked_shap_features(shap_summary: Dict[str, Any], limit: int = 3) -> List[str]:
    ranked = []
    for item in _safe_list(shap_summary.get('ranked_features')):
        if not isinstance(item, dict):
            continue
        feature = _safe_str(item.get('feature')).strip()
        if feature:
            ranked.append(feature)
    return ranked[:limit]


def _shap_direction_sentence_fragment(shap_summary: Dict[str, Any]) -> str:
    positive = [
        _safe_str(item).strip()
        for item in _safe_list(shap_summary.get('top_positive_risk_features'))
        if _safe_str(item).strip()
    ][:3]
    negative = [
        _safe_str(item).strip()
        for item in _safe_list(shap_summary.get('top_negative_risk_features'))
        if _safe_str(item).strip()
    ][:3]

    if positive and negative:
        return (
            f'higher values of {_join_readable_list(positive)} tended to push the model toward the case class, '
            f'whereas higher values of {_join_readable_list(negative)} tended to shift predictions in the opposite direction'
        )
    if positive:
        return f'higher values of {_join_readable_list(positive)} tended to push the model toward the case class'
    if negative:
        return f'higher values of {_join_readable_list(negative)} tended to shift predictions away from the case class'
    return ''


def _shap_model_direction_sentence(shap_summary: Dict[str, Any]) -> str:
    positive = [
        _safe_str(item).strip()
        for item in _safe_list(shap_summary.get('top_positive_risk_features'))
        if _safe_str(item).strip()
    ][:3]
    negative = [
        _safe_str(item).strip()
        for item in _safe_list(shap_summary.get('top_negative_risk_features'))
        if _safe_str(item).strip()
    ][:3]

    if positive and negative:
        return (
            f'Within the archived model-based direction summary, higher values of {_join_readable_list(positive)} were associated with higher predicted case probability, '
            f'whereas higher values of {_join_readable_list(negative)} were associated with lower predicted case probability'
        )
    if positive:
        return (
            f'Within the archived model-based direction summary, higher values of {_join_readable_list(positive)} were associated with higher predicted case probability'
        )
    if negative:
        return (
            f'Within the archived model-based direction summary, no retained metabolite showed a stable higher-value and higher-risk pattern, '
            f'whereas higher values of {_join_readable_list(negative)} were associated with lower predicted case probability'
        )
    return ''


def _shap_local_example_sentence(shap_summary: Dict[str, Any]) -> str:
    local_example = _safe_dict(shap_summary.get('local_example'))
    patient_index = local_example.get('patient_index') or shap_summary.get('patient_index_used')
    top_contributors = [
        item for item in _safe_list(local_example.get('top_contributors'))
        if isinstance(item, dict) and _safe_str(item.get('feature')).strip()
    ]
    if not patient_index:
        patient_index = 1
    if not top_contributors:
        return f'The companion waterfall plot for Patient {patient_index} provides a concrete patient-level decomposition of the final prediction'

    increase_features = [
        _safe_str(item.get('feature')).strip()
        for item in top_contributors
        if _safe_str(item.get('direction')).strip() == 'increase_prediction' and _safe_str(item.get('feature')).strip()
    ][:3]
    decrease_features = [
        _safe_str(item.get('feature')).strip()
        for item in top_contributors
        if _safe_str(item.get('direction')).strip() == 'decrease_prediction' and _safe_str(item.get('feature')).strip()
    ][:3]

    if increase_features and decrease_features:
        return (
            f'The companion waterfall plot for Patient {patient_index} shows how {_join_readable_list(increase_features)} increased the predicted case probability, '
            f'whereas {_join_readable_list(decrease_features)} pulled the prediction downward in that individual example'
        )
    if increase_features:
        return f'The companion waterfall plot for Patient {patient_index} shows how {_join_readable_list(increase_features)} increased the predicted case probability in that individual example'
    if decrease_features:
        return f'The companion waterfall plot for Patient {patient_index} shows how {_join_readable_list(decrease_features)} pulled the predicted case probability downward in that individual example'
    return f'The companion waterfall plot for Patient {patient_index} provides a concrete patient-level decomposition of the final prediction'


def _english_scenario_label(scenario: Dict[str, Any], fallback: str = '') -> str:
    scenario_dict = _safe_dict(scenario)
    name = _safe_str(scenario_dict.get('name')).strip()
    key = _safe_str(scenario_dict.get('key') or fallback).strip()
    intended_use = _safe_str(scenario_dict.get('intended_use')).strip().lower()

    parenthetical = re.search(r'\(([^()]*)\)', name)
    if parenthetical:
        candidate = parenthetical.group(1).strip()
        if re.search(r'[A-Za-z]', candidate):
            return candidate

    if name and re.search(r'^[\x00-\x7F]+$', name):
        return name

    scenario_key_map = {
        'primary_care': 'Primary Care Screening',
    }
    if key:
        mapped = scenario_key_map.get(key)
        if mapped:
            return mapped
        if re.search(r'[A-Za-z0-9]', key):
            humanized = key.replace('_', ' ').replace('-', ' ').strip().title()
            if 'screen' in intended_use and 'screen' not in humanized.lower():
                return f'{humanized} Screening'
            return humanized

    if 'screen' in intended_use:
        return 'Primary Care Screening'
    return 'the prespecified clinical scenario'


def _feature_report_names(feature_dictionary: List[Dict[str, Any]], *, include_engineered: Optional[bool] = None) -> List[str]:
    names: List[str] = []
    for item in feature_dictionary:
        if not isinstance(item, dict):
            continue
        if include_engineered is not None and bool(item.get('is_engineered')) != include_engineered:
            continue
        label = str(item.get('preferred_report_name') or item.get('report_label') or '').strip()
        if label:
            names.append(label)
    return names


def _result_figure_titles(figures: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    titles: List[str] = []
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        title = str(figure.get('title') or '').strip()
        if title:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _figure_evidence_summary(report_context: Dict[str, Any]) -> Dict[str, Any]:
    report_inputs = _safe_dict(report_context.get('report_inputs'))
    results = _safe_dict(report_inputs.get('results'))
    methodology = _safe_dict(report_inputs.get('methodology'))
    discussion = _safe_dict(report_inputs.get('discussion'))
    return (
        _safe_dict(results.get('figure_evidence_summary'))
        or _safe_dict(methodology.get('figure_evidence_summary'))
        or _safe_dict(discussion.get('figure_evidence_summary'))
        or _safe_dict(_safe_dict(report_context.get('phase3')).get('figure_evidence_summary'))
    )


def _figure_evidence(report_context: Dict[str, Any], key: str) -> Dict[str, Any]:
    return _safe_dict(_figure_evidence_summary(report_context).get(key))


def _figure_citation_from_summary(summary: Dict[str, Any]) -> str:
    return _safe_str(summary.get('figure_label')).strip() or 'the corresponding figure'


def _select_figures_by_ids(report_context: Dict[str, Any], figure_ids: List[str]) -> List[Dict[str, Any]]:
    figures = _safe_list(_safe_dict(report_context.get('phase3')).get('figures'))
    figure_index = {
        _safe_str(figure.get('figure_id')).strip(): figure
        for figure in figures
        if isinstance(figure, dict) and _safe_str(figure.get('figure_id')).strip()
    }
    selected: List[Dict[str, Any]] = []
    for figure_id in figure_ids:
        figure = figure_index.get(_safe_str(figure_id).strip())
        if figure:
            selected.append(figure)
    return selected


def _sanitize_narrative_text(text: str) -> str:
    normalized = str(text or '').replace('\r\n', '\n').replace('\r', '\n')
    normalized = re.sub(r'\s*\(HMDB\d{5,9}(?:_\d+)?\)', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bHMDB\d{5,9}(?:_\d+)?\b', '', normalized, flags=re.IGNORECASE)
    cleaned_lines: List[str] = []
    for line in normalized.split('\n'):
        compact = re.sub(r'[^\S\n]{2,}', ' ', line)
        compact = re.sub(r' \,', ',', compact)
        cleaned_lines.append(compact.strip())
    normalized = '\n'.join(cleaned_lines)
    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    # Never expose unresolved programmer placeholders in reader-facing prose.
    normalized = re.sub(r'\b\d{4,}[- ]fold\b', 'out-of-fold', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bNone[- ]fold\b', 'cross-validation with fold count not archived', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bN/A\b', 'not archived', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bNone\b', 'not archived', normalized)
    normalized = re.sub(r'\bFig\.\s*(?=[,.;:)])', 'the corresponding figure', normalized)
    normalized = re.sub(r'\bstrong internal discrimination\b', 'modest development discrimination', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bclear incremental value\b', 'a directional incremental signal', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bclinically actionable\b', 'exploratory for workflow planning', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\bimproved discrimination\b', 'showed a higher point estimate for discrimination', normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _mechanistic_theme_blueprints(feature_dictionary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = {str(item.get('preferred_report_name') or '').strip().lower(): item for item in feature_dictionary if isinstance(item, dict)}
    themes: List[Dict[str, Any]] = []

    def present(*candidates: str) -> List[str]:
        found: List[str] = []
        for candidate in candidates:
            item = names.get(candidate.lower())
            if item:
                found.append(str(item.get('preferred_report_name')).strip())
        return found

    theme_specs = [
        {
            'title': 'Collagen Turnover And Extracellular Matrix Remodeling',
            'features': present('4-Hydroxyproline'),
            'direct': 'Hydroxyproline is a collagen-derived metabolite and therefore provides direct panel-level support for extracellular matrix remodeling.',
            'contextual': 'This theme is biologically consistent with tissue remodeling and stromal activity, but the current run does not include an explicit pathway-score feature for collagen turnover.',
        },
        {
            'title': 'Tryptophan-Kynurenine Immune Axis',
            'features': present('Tryptophan', 'Kynurenine'),
            'direct': 'The co-occurrence of tryptophan and kynurenine supports a panel-level signal related to tryptophan catabolism and immune-metabolic regulation.',
            'contextual': 'This theme is mechanistically plausible for cancer-associated immune adaptation, but still remains interpretive without direct flux or enzyme measurements.',
        },
        {
            'title': 'One-Carbon, Methyl-Donor, And Osmolyte Biology',
            'features': present('Methionine', 'N-Methylalanine', 'myo-Inositol'),
            'direct': 'Methionine, N-methylalanine, and myo-inositol together point toward methyl-donor handling, amino-acid turnover, and osmolyte-linked metabolic stress responses.',
            'contextual': 'The Phase 0 disease context emphasized betaine-related pathways, so this theme should be framed as contextual convergence rather than direct pathway-matched evidence in the selected panel.',
        },
        {
            'title': 'Lipid And Small-Molecule Stress Signals',
            'features': present('Lauric acid', 'Hydroxylamine'),
            'direct': 'Lauric acid and hydroxylamine are retained as isolated panel components, suggesting complementary lipid-related and small-molecule stress information.',
            'contextual': 'These features are currently best treated as hypothesis-generating because direct pathway linkage is limited in the available artifact set.',
        },
    ]

    for spec in theme_specs:
        if spec['features']:
            themes.append(spec)
    return themes


def _feature_evidence_tiers(feature_dictionary: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    available = {name for name in _feature_report_names(feature_dictionary)}
    tier_1 = [
        name for name in ['4-Hydroxyproline', 'Tryptophan', 'Kynurenine', 'Methionine']
        if name in available
    ]
    tier_2 = [
        name for name in ['N-Methylalanine', 'myo-Inositol', 'Lauric acid']
        if name in available
    ]
    tier_3 = [
        name for name in ['Hydroxylamine']
        if name in available
    ]
    assigned = set(tier_1 + tier_2 + tier_3)
    residual = [name for name in _feature_report_names(feature_dictionary) if name not in assigned]
    tier_3.extend(residual)
    return {
        'tier_1': tier_1,
        'tier_2': tier_2,
        'tier_3': tier_3,
    }


def _build_prior_selection_interpretation(phase0: Dict[str, Any]) -> List[str]:
    final_priors = _safe_list(phase0.get('final_priors') or phase0.get('confirmed_biomarker_names'))
    selection_threshold = _safe_dict(phase0.get('selection_threshold'))
    runtime_pathway_summary = _safe_dict(phase0.get('runtime_pathway_summary'))

    lines: List[str] = []
    if final_priors:
        lines.append(
            f'Phase 0 retained {len(final_priors)} prior biomarkers after score thresholding, '
            f'forming the disease-aligned biomarker seed set carried into later phases.'
        )

    threshold_value = _safe_float(selection_threshold.get('threshold'))
    threshold_rule = str(selection_threshold.get('rule') or '').strip()
    if threshold_value is not None or threshold_rule:
        rule_text = threshold_rule or 'unspecified rule'
        lines.append(
            f'Prior selection used `{rule_text}` with cutoff {_format_metric(threshold_value)} to translate literature/mechanism evidence into the final prior queue.'
        )

    if runtime_pathway_summary:
        lines.append(
            f'Runtime pathway retrieval considered {runtime_pathway_summary.get("pathway_count", "N/A")} pathways; '
            f'LLM rerank used={runtime_pathway_summary.get("llm_rerank_used", False)}.'
        )
    return lines


def _build_incremental_value_interpretation(incremental_value: Dict[str, Any]) -> List[str]:
    if not incremental_value or not incremental_value.get('enabled'):
        return []

    nri = _safe_dict(incremental_value.get('nri'))
    idi = _safe_dict(incremental_value.get('idi'))
    baseline_name = str(incremental_value.get('baseline_name') or 'baseline panel').strip()
    scope = str(incremental_value.get('comparison_scope') or 'unspecified scope').strip()

    lines = [
        f'Reclassification analysis versus `{baseline_name}` ({scope}) yielded continuous NRI {_format_metric(nri.get("value"))} and IDI {_format_metric(idi.get("value"))}.'
    ]

    nri_ci = nri.get('ci_95')
    idi_ci = idi.get('ci_95')
    if nri_ci or idi_ci:
        lines.append(
            f'Bootstrap uncertainty was tracked for incremental value: NRI 95% CI {nri_ci}, IDI 95% CI {idi_ci}.'
        )
    return lines


def _build_memory_interpretation(memory_prior_summary: Dict[str, Any]) -> List[str]:
    if not memory_prior_summary:
        return []

    enabled = bool(memory_prior_summary.get('memory_prior_enabled', False))
    retrieval = _safe_dict(memory_prior_summary.get('retrieval_summary'))
    if not enabled and not retrieval:
        return []

    lines = [
        f'Memory-aware search prior enabled={enabled}, reranking {memory_prior_summary.get("memory_reranked_candidate_count", 0)} candidates before deterministic Pareto-SFS expansion.'
    ]
    matched_cases = _safe_list(memory_prior_summary.get('memory_matched_case_ids'))
    if matched_cases:
        lines.append(
            f'Memory retrieval matched {len(matched_cases)} historical Phase 2 case(s) with advisory anchor hints {memory_prior_summary.get("memory_anchor_hints", [])}.'
        )
    if retrieval:
        lines.append(
            f'Retrieval trace: considered {retrieval.get("considered_case_count", "N/A")}, eligible {retrieval.get("eligible_case_count", "N/A")}, selected {retrieval.get("selected_case_count", "N/A")} cases.'
        )
    return lines


def _build_scenario_strategy_interpretation(strategy: Dict[str, Any]) -> List[str]:
    if not strategy:
        return []

    scenario = _safe_dict(strategy.get('clinical_scenario_definition'))
    weights = _safe_dict(strategy.get('topsis_weights'))
    lines: List[str] = []
    if scenario:
        lines.append(
            f'Phase 2 used the `{scenario.get("key") or strategy.get("clinical_scenario") or "unspecified"}` scenario, with an exploratory optimization threshold of {_format_metric(scenario.get("default_action_threshold"), 2)} and risk thresholds {scenario.get("risk_thresholds")}. This value is an optimization input, not a validated clinical threshold.'
        )
        scenario_weights = _safe_dict(scenario.get('weights'))
        if scenario_weights:
            lines.append(
                f'Scenario-defined optimization weights were f_perf={_format_metric(scenario_weights.get("f_perf"), 2)}, f_bio={_format_metric(scenario_weights.get("f_bio"), 2)}, f_cost={_format_metric(scenario_weights.get("f_cost"), 2)}, and f_corr={_format_metric(scenario_weights.get("f_corr"), 2)}.'
            )
    elif weights:
        lines.append(f'Phase 2 TOPSIS weights were {weights}.')

    if strategy.get('soft_epsilon_feasible_enabled'):
        lines.append(
            f'Soft-epsilon feasible filtering was enabled with performance margin {_format_metric(strategy.get("soft_epsilon_perf_margin"), 3)} ({strategy.get("soft_epsilon_perf_margin_source") or "source unspecified"}).'
        )
    return lines


def _build_executive_summary_content(report_context: Dict[str, Any]) -> str:
    executive = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('executive_summary'))
    clinical_utility = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('clinical_utility'))
    best_auc = _format_metric(executive.get('best_auc'))
    nri = _safe_dict(executive.get('nri'))
    idi = _safe_dict(executive.get('idi'))
    feature_dictionary = _safe_list(executive.get('winner_feature_dictionary'))
    feature_names = _feature_report_names(feature_dictionary)
    readable_feature_names = [name for name in feature_names if not _is_opaque_feature_label(name)]
    scenario = _safe_dict(clinical_utility.get('scenario'))
    scenario_name = _english_scenario_label(scenario)
    default_threshold = _format_metric(scenario.get('default_action_threshold'), 2)
    companion = _safe_dict(clinical_utility.get('data_driven_companion_threshold_summary'))
    companion_threshold = _format_metric(companion.get('selected_threshold'), 2)

    panel_count = len(feature_dictionary) or executive.get("final_feature_count", "N/A")
    sentence_1 = (
        f'In this internal biomarker-discovery study of {_display_disease_name(report_context.get("phase0", {}).get("disease_name"))}, '
        f'the archived Phase 2 workflow selected a {panel_count}-feature panel through scenario-aware multi-objective feature selection, '
        f'with an archived cross-validated AUC of {best_auc}.'
    )
    auc_ci = _safe_list(executive.get('auc_ci_95') or executive.get('cv_auc_ci_95'))
    holdout_auc = _safe_float(executive.get('holdout_auc') or executive.get('internal_holdout_auc'))
    nri_ci = _safe_list(nri.get('ci_95'))
    idi_ci = _safe_list(idi.get('ci_95'))
    nri_crosses_zero = len(nri_ci) == 2 and _safe_float(nri_ci[0]) is not None and _safe_float(nri_ci[1]) is not None and _safe_float(nri_ci[0]) <= 0 <= _safe_float(nri_ci[1])
    idi_crosses_zero = len(idi_ci) == 2 and _safe_float(idi_ci[0]) is not None and _safe_float(idi_ci[1]) is not None and _safe_float(idi_ci[0]) <= 0 <= _safe_float(idi_ci[1])
    if holdout_auc is not None:
        sentence_1 = sentence_1.rstrip('.') + f', while the internal holdout AUC was {_format_metric(holdout_auc)}.'
    if nri or idi:
        if nri_crosses_zero and idi_crosses_zero:
            sentence_2 = (
                f'Against the Phase 1 baseline, continuous NRI of {_format_metric(nri.get("value"))} and IDI of {_format_metric(idi.get("value"))}; both uncertainty intervals included zero, so neither estimate established a non-zero incremental effect.'
            )
        elif nri_crosses_zero != idi_crosses_zero:
            crossing_name = 'NRI' if nri_crosses_zero else 'IDI'
            excluding_name = 'IDI' if nri_crosses_zero else 'NRI'
            sentence_2 = (
                f'Against the Phase 1 baseline, continuous NRI of {_format_metric(nri.get("value"))} and IDI of {_format_metric(idi.get("value"))}. '
                f'The {crossing_name} interval included zero, whereas the {excluding_name} interval excluded zero; these secondary internal reclassification measures must therefore be interpreted separately.'
            )
        else:
            sentence_2 = (
                f'Against the Phase 1 baseline, the point estimates were continuous NRI of {_format_metric(nri.get("value"))} and IDI of {_format_metric(idi.get("value"))}; these remain secondary internal reclassification measures.'
            )
    else:
        sentence_2 = 'The principal evidence is the internal discrimination estimate; incremental reclassification was not treated as a primary endpoint.'
    if readable_feature_names:
        panel_sentence_prefix = f'The report-facing panel includes {_join_readable_list(readable_feature_names, max_items=panel_count if isinstance(panel_count, int) else 20)}'
        unresolved_count = max(0, len(feature_names) - len(readable_feature_names))
        if unresolved_count:
            panel_sentence_prefix += f', together with {unresolved_count} additional features currently recorded as assay identifiers'
    else:
        panel_sentence_prefix = f'The report-facing panel contains {panel_count} features'
    sentence_3 = (
        f'{panel_sentence_prefix}. An illustrative high-sensitivity operating point of {default_threshold} is shown for the archived {scenario_name} scenario; '
        'it is not validated for clinical use, and its operating characteristics require target-population external validation and recalibration.'
    )
    return _sanitize_narrative_text(' '.join([sentence_1, sentence_2, sentence_3]))


def _build_methodology_content(report_context: Dict[str, Any]) -> str:
    report_inputs = _safe_dict(report_context.get('report_inputs'))
    methodology = _safe_dict(report_inputs.get('methodology'))
    clinical_utility = _safe_dict(report_inputs.get('clinical_utility'))
    phase0 = _safe_dict(methodology.get('phase0'))
    phase1 = _safe_dict(methodology.get('phase1'))
    phase2 = _safe_dict(methodology.get('phase2'))
    statistical_framework = _safe_dict(methodology.get('statistical_evaluation_framework'))

    figure_evidence = _safe_dict(methodology.get('figure_evidence_summary'))
    fig1c = _safe_dict(figure_evidence.get('prior_evidence_atlas') or figure_evidence.get('upset'))
    fig2b = _safe_dict(figure_evidence.get('pareto'))
    lines: List[str] = []
    engineering = _safe_dict(phase1.get('feature_engineering_summary'))
    preprocessing = _safe_dict(phase1.get('preprocessing_summary'))
    preprocessing_context = _safe_dict(preprocessing.get('context'))
    zero_handling = _safe_dict(preprocessing.get('zero_handling'))
    missingness_detection = _safe_dict(preprocessing.get('missingness_detection'))
    imputation = _safe_dict(preprocessing.get('imputation'))
    normalization_scaling = _safe_dict(preprocessing.get('normalization_scaling'))
    qc_batch = _safe_dict(preprocessing.get('qc_batch'))
    scenario_strategy = _safe_dict(phase2.get('scenario_strategy'))
    scenario = _safe_dict(scenario_strategy.get('clinical_scenario_definition'))
    scenario_name = _english_scenario_label(scenario, _safe_str(scenario_strategy.get('clinical_scenario')))
    if scenario_name == 'the prespecified clinical scenario':
        scenario_name = _english_scenario_label(_safe_dict(clinical_utility.get('scenario')))
    search_summary = _safe_dict(methodology.get('phase2_search_summary'))
    dual_criteria = _safe_dict(search_summary.get('dual_criteria_summary'))
    phase0_pathways = _safe_list(phase0.get('target_pathways') or phase0.get('disease_core_pathways') or phase0.get('top_pathways'))
    phase0_screening = _safe_dict(phase0.get('screening_summary'))
    phase0_literature = _safe_dict(phase0.get('literature_protocol'))
    phase0_stage1_policy = _safe_dict(phase0.get('stage1_screening_policy'))
    phase0_rubric = _safe_dict(phase0.get('rubric_framework'))
    baseline_model_screening = _safe_dict(phase1.get('baseline_model_screening'))
    # The optional Phase 1 baseline summary may omit ``best_model`` even when
    # the authoritative Phase 2 winner artifact records the model carried
    # forward into optimization.
    results_input = _safe_dict(report_inputs.get('results'))
    winner_panel_input = _safe_dict(results_input.get('winner_panel'))
    selected_model_fallback = _safe_str(
        phase2.get('selected_model')
        or winner_panel_input.get('selected_model')
    ).strip()
    runtime_summary = _safe_dict(phase2.get('runtime_summary'))
    methods_used = [_readable_method_name(item) for item in _safe_list(phase1.get('methods_used')) if _safe_str(item).strip()]
    stability_context = _safe_dict(phase1.get('stability_search_context'))
    stability_dataset = _safe_dict(stability_context.get('dataset_fingerprint'))
    class_counts = _safe_dict(stability_dataset.get('class_counts'))
    imbalance_ratio = _safe_float(stability_dataset.get('imbalance_ratio'))
    handoff = _phase1_handoff_facts(phase1, phase2)
    cv_folds = _infer_cv_folds(report_context)
    phase0_confirmed = _safe_list(phase0.get('confirmed_biomarkers'))
    phase0_final_priors = _safe_list(phase0.get('final_priors'))
    phase0_candidate_count = _coalesce_non_null(
        phase0_screening.get('candidate_count'),
        phase0.get('confirmed_biomarker_count'),
        len(phase0_confirmed) or None,
    )
    phase0_advanced_count = _coalesce_non_null(
        phase0_screening.get('stage1_candidate_count'),
        phase0_screening.get('advanced_candidate_count'),
    )
    phase0_prior_count = _coalesce_non_null(
        phase0.get('phase0_prior_count'),
        phase0.get('final_priors_count'),
        len(phase0_final_priors) or None,
    )
    if phase0_advanced_count is not None:
        phase0_screening_sentence = (
            f'{phase0_candidate_count if phase0_candidate_count is not None else "the archived candidate set"} candidates entered coarse screening, '
            f'{phase0_advanced_count} advanced after title/abstract-level evidence triage, and '
            f'{phase0_prior_count if phase0_prior_count is not None else "the archived"} high-confidence priors were ultimately retained for downstream modeling.'
        )
    else:
        # Do not invent an intermediate Stage-1 count when the source record
        # only archives the confirmed/final-prior list.
        phase0_screening_sentence = (
            f'{phase0_candidate_count if phase0_candidate_count is not None else "The archived candidate set"} candidates were evaluated in the archived screening workflow, and '
            f'{phase0_prior_count if phase0_prior_count is not None else "the archived"} high-confidence priors were ultimately retained for downstream modeling.'
        )

    lines.extend([
        '### Prior Biomarker Discovery and Evidence Scoring',
        (
            f'Phase 0 began by assembling a disease-linked candidate metabolite set for {_display_disease_name(phase0.get("disease_name"))}; '
            f'{phase0_screening_sentence}'
        ),
    ])
    if phase0_literature:
        lines.append(
            f'For each retained candidate, the literature module queried {phase0_literature.get("query_field", "tiab")} fields in PubMed, retrieved up to '
            f'{phase0_literature.get("max_abstracts_per_candidate", "N/A")} abstracts per metabolite-disease pair, and carried forward up to '
            f'{phase0_literature.get("top_k_abstracts", "N/A")} relevance-filtered abstracts into a structured evidence pack.'
        )
    if phase0_stage1_policy:
        lines.append(
            f'The Stage 1 coarse screen enforced an explicit hit-count gate: zero-hit candidates required disease-core pathway support '
            f'(minimum pathway relevance score {_format_metric(phase0_stage1_policy.get("pathway_relevance_min_score"), 1)}), and low-hit candidates '
            f'with <= {phase0_stage1_policy.get("low_hit_max_hits", "N/A")} literature hits were only retained when stronger pathway support was present.'
        )
    if phase0_rubric:
        lines.append(
            'Each evidence pack was then adjudicated by an LLM acting as a virtual reviewer under a fixed 3+1 rubric comprising '
            'Clinical_Evidence (0-3), Disease_Specificity (0-3), Mechanistic_Plausibility (0-3), and a Consistency modifier (-1 to 1), '
            'after which dynamic evidence-zone weighting and a mu+0.5sigma threshold were used to define the final prior queue.'
        )
    if phase0_pathways:
        lines.append(
            f'The prior layer was anchored to curated disease-core programs, including {_join_readable_list([str(item) for item in phase0_pathways[:4]])}, '
            'so biological context entered the pipeline before panel search rather than being added post hoc.'
        )

    lines.extend([
        '',
        '### Data Preprocessing and Stable Candidate Generation',
        (
            f'Phase 1 did not restrict candidate generation to directly measured metabolites alone; starting from the observed metabolite matrix, the workflow also generated combination-style candidates such as sums, ratios, and pathway-level summaries so that grouped biochemical signals could compete against single-analyte features during screening.'
        ),
        (
            f'After preprocessing and feature screening, the run-aligned data flow contained {handoff.get("handoff_count") if handoff.get("handoff_count") is not None else "not archived"} candidate features entering Phase 2: '
            f'{handoff.get("raw_count") if handoff.get("raw_count") is not None else "not archived"} measured metabolite features and {handoff.get("engineered_count", 0)} engineered features. '
            'The final Phase 2 panel count is reported separately and must not be conflated with the Phase 1 handoff count.'
        ),
        (
            f'In this run, the run-aligned Phase 1 handoff into Phase 2 comprised {handoff.get("handoff_count") if handoff.get("handoff_count") is not None else "not archived"} retained features.'
        ),
    ])
    if not bool(imputation.get('performed')):
        lines.append(
            f'Data quality control first applied a sample-level missingness threshold of {_format_percentage(missingness_detection.get("sample_missingness_threshold"), 0)} '
            f'and a feature-level threshold of {_format_percentage(missingness_detection.get("feature_missingness_threshold"), 0)}; '
            'all samples passed this audit without requiring downstream imputation in the present run.'
        )
    else:
        lines.append(
            f'Missingness was first screened with a sample-level exclusion threshold of {_format_percentage(missingness_detection.get("sample_missingness_threshold"), 0)} '
            f'and a feature-level threshold of {_format_percentage(missingness_detection.get("feature_missingness_threshold"), 0)}; '
            f'the resulting pattern was treated as {missingness_detection.get("suspected_mechanism", "the prespecified")} missingness, after which '
            f'{imputation.get("method", "the prespecified")} imputation strategy was applied according to the observed missing-data pattern.'
        )
    if zero_handling.get('strategy') == 'convert_zero_to_nan':
        lines.append(
            'Before missingness assessment, zero intensities were treated as non-detect-like measurements and converted to missing values rather than being carried forward as true observed abundances.'
        )
    lines.append(
        f'The preprocessing workflow then proceeded through {_join_readable_list(_safe_list(normalization_scaling.get("configured_workflow")))}, '
        'a sequence selected to control sample-dilution effects, stabilize skewed metabolite distributions, cap extreme observations, and place analytes on a common standardized scale before feature ranking.'
    )
    if (
        not bool(qc_batch.get('batch_correction_applied'))
        and not bool(qc_batch.get('qc_rsd_filter_applied'))
        and not bool(preprocessing_context.get('has_pooled_qc'))
    ):
        lines.append(
            'Because the current public matrix did not expose pooled-QC support or sufficient drift-tracking metadata, no QC-RSD filtering or batch-drift correction was applied in this run.'
        )
    if bool(stability_context.get('imbalance_detected')) and (stability_context.get('stability_iterations') or stability_context.get('selector_family_count')):
        balance_text = _format_class_counts(class_counts)
        lines.append(
            f'Because the current test dataset was imbalanced ({balance_text}; imbalance ratio {_format_metric(imbalance_ratio, 2)} in the run-aligned Phase 2 fingerprint), candidate generation was framed as a stability-oriented search rather than a single-pass selector vote. '
            f'The matched Phase 0/1 test configuration specified repeated stability iterations ({stability_context.get("stability_iterations", "N/A")}) together with a multi-selector search pool ({stability_context.get("selector_family_count", "N/A")} configured methods), after which cross-method consensus and a {phase1.get("selection_policy", "prespecified")} handoff policy were used to retain a sufficiently robust candidate pool.'
        )
    if methods_used:
        lines.append(
            f'The archived Phase 1 selection report recorded {_join_readable_list(methods_used)} as the selector families contributing explicit evidence to the final archived handoff, so candidate retention reflected cross-method support rather than reliance on a single selector family.'
        )
    if baseline_model_screening or selected_model_fallback:
        lines.append(_baseline_model_screening_sentence(baseline_model_screening, selected_model_fallback))
    if fig1c:
        lines.append(
            f'The corresponding Prior Evidence Atlas was archived to visualize how adaptive Phase 0 prior scores decomposed across mechanistic plausibility, disease specificity, clinical evidence, and consistency before the retained prior queue was handed forward ({_figure_citation_from_summary(_safe_dict(fig1c))}).'
        )

    lines.extend([
        '',
        '### Scenario-Aware Multi-objective Panel Optimization',
        (
            f'Phase 2 was designed for {scenario_name}, a scenario that prioritizes clinically actionable sensitivity, parsimonious assay burden, and interpretable panel composition rather than discrimination alone.'
        ),
        (
            'The final panel was selected by full Pareto-guided sequential forward search with TOPSIS-based scalarization over four prespecified dimensions: discrimination, biological coherence, analytical burden, and redundancy control.'
        ),
    ])
    if scenario:
        lines.append(
            f'The optimization target was anchored to prespecified exploratory risk thresholds {_safe_list(scenario.get("risk_thresholds"))} and an optimization threshold of {_format_metric(scenario.get("default_action_threshold"), 2)}, '
            'so scenario constraints were encoded directly into the search objective rather than applied only after model fitting.'
        )
    if dual_criteria:
        lines.append(
            f'Dual-criteria early stopping combined a dynamic Events-Per-Variable guard with a DeLong-based patience rule '
            f'({dual_criteria.get("delong_patience", runtime_summary.get("patience", "N/A"))} consecutive layers; delta {_format_metric(dual_criteria.get("delong_improvement_delta", runtime_summary.get("epsilon")), 3)}), '
            f'with the search capped at depth {phase2.get("actual_depth", phase2.get("max_depth", "N/A"))} in the present run.'
        )
    if fig2b:
        lines.append(
            f'The corresponding Pareto trajectory provides a visual audit trail of how the search traversed the feasible solution space before converging on the final panel ({_figure_citation_from_summary(fig2b)}).'
        )
    lines.extend([
        '',
        '### Statistical Evaluation and Validation Framework',
        (
            f'Internal validation used a {cv_folds}-fold cross-validation framework with out-of-fold evaluation plus internal holdout evaluation.' if cv_folds is not None else
            'Internal validation used out-of-fold cross-validation plus internal holdout evaluation; the archived fold count was not recorded.'
        ),
        (
            'Probability calibration and clinical utility were evaluated through calibration plots, Brier score and expected calibration error summaries, decision-curve analysis, threshold-dependent operating characteristics, and continuous NRI and IDI, '
            'whereas model interpretability and exposure-response structure were examined using SHAP-based global and indexed local attribution together with restricted cubic spline modeling of continuous analyte-risk relationships.'
        ),
    ])
    return _sanitize_narrative_text('\n'.join(lines))


def _build_results_content(report_context: Dict[str, Any]) -> str:
    report_inputs = _safe_dict(report_context.get('report_inputs'))
    results = _safe_dict(report_inputs.get('results'))
    methodology_phase2 = _safe_dict(_safe_dict(report_inputs.get('methodology')).get('phase2'))
    phase0 = _safe_dict(report_context.get('phase0'))
    winner_panel = _safe_dict(results.get('winner_panel'))
    feature_dictionary = _safe_list(results.get('feature_dictionary'))
    feature_names = _feature_report_names(feature_dictionary)
    incremental_value = _safe_dict(results.get('incremental_value'))
    figure_evidence = _safe_dict(results.get('figure_evidence_summary'))
    upset = _safe_dict(figure_evidence.get('prior_evidence_atlas') or figure_evidence.get('upset'))
    autogluon_roc = _safe_dict(figure_evidence.get('autogluon_roc'))
    pareto = _safe_dict(figure_evidence.get('pareto'))
    radar = _safe_dict(figure_evidence.get('radar'))
    final_roc = _safe_dict(figure_evidence.get('final_roc'))
    dca = _safe_dict(figure_evidence.get('dca'))
    shap = _safe_dict(figure_evidence.get('shap'))
    rcs = _safe_dict(figure_evidence.get('rcs'))
    calibration = _safe_dict(figure_evidence.get('calibration'))
    threshold = _safe_dict(figure_evidence.get('threshold_performance'))
    recommended = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('clinical_utility')).get('recommended_threshold_summary') or {}
    companion = _safe_dict(_safe_dict(_safe_dict(report_context.get('report_inputs')).get('clinical_utility')).get('data_driven_companion_threshold_summary'))
    winner_panel_name_summary = _safe_dict(results.get('winner_panel_name_summary'))
    raw_count = winner_panel_name_summary.get('raw_count')
    engineered_count = winner_panel_name_summary.get('engineered_count')
    engineered_names = _safe_list(winner_panel_name_summary.get('engineered_feature_names'))
    prior_supported_count = sum(
        1 for item in feature_dictionary
        if isinstance(item, dict) and bool(item.get('prior_supported'))
    )
    nri = _safe_dict(incremental_value.get('nri'))
    idi = _safe_dict(incremental_value.get('idi'))
    default_threshold = _coalesce_non_null(
        threshold.get('default_threshold'),
        _safe_dict(recommended).get('selected_threshold'),
    )
    companion_threshold = _coalesce_non_null(
        threshold.get('companion_threshold'),
        companion.get('selected_threshold'),
    )
    phase1_candidate_pool_count = _coalesce_non_null(
        results.get('phase1_candidate_pool_count'),
        _safe_dict(report_context.get('phase2')).get('phase1_candidate_pool_count'),
        methodology_phase2.get('phase2_input_feature_count'),
    )
    handoff_facts = _phase1_handoff_facts(_safe_dict(report_context.get('phase1')), _safe_dict(report_context.get('phase2')))
    if phase1_candidate_pool_count in (None, 0, '0') and handoff_facts.get('handoff_count') is not None:
        phase1_candidate_pool_count = handoff_facts.get('handoff_count')
    final_count = len(feature_dictionary) or len(_safe_list(winner_panel.get('winner_features'))) or _safe_float(results.get('final_feature_count'))
    holdout_auc = _infer_holdout_auc(report_context)
    final_priors = _safe_list(phase0.get('final_priors'))
    selected_model = _safe_str(
        winner_panel.get('selected_model')
        or _safe_dict(report_context.get('phase2')).get('selected_model')
    ).strip()

    baseline_radar = _safe_dict(_safe_dict(radar.get('normalized_scores')).get('Phase 1 Baseline'))
    winner_radar = _safe_dict(_safe_dict(radar.get('normalized_scores')).get('PToT Winner'))
    overall_improved = bool(
        winner_radar and baseline_radar and
        _safe_float(winner_radar.get('AUC')) is not None and _safe_float(baseline_radar.get('AUC')) is not None and
        _safe_float(winner_radar.get('AUC')) > _safe_float(baseline_radar.get('AUC')) and
        _safe_float(winner_radar.get('f_bio')) is not None and _safe_float(baseline_radar.get('f_bio')) is not None and
        _safe_float(winner_radar.get('f_bio')) > _safe_float(baseline_radar.get('f_bio')) and
        _safe_float(winner_radar.get('1-f_cost')) is not None and _safe_float(baseline_radar.get('1-f_cost')) is not None and
        _safe_float(winner_radar.get('1-f_cost')) > _safe_float(baseline_radar.get('1-f_cost')) and
        _safe_float(winner_radar.get('1-f_corr')) is not None and _safe_float(baseline_radar.get('1-f_corr')) is not None and
        _safe_float(winner_radar.get('1-f_corr')) > _safe_float(baseline_radar.get('1-f_corr'))
    )

    content_lines = [
        '### Candidate-to-Panel Selection Pathway',
    ]
    if phase1_candidate_pool_count is not None:
        content_lines.append(
            f'The run-aligned Phase 1 handoff supplied {phase1_candidate_pool_count} candidate features to Phase 2, comprising {handoff_facts.get("raw_count") if handoff_facts.get("raw_count") is not None else "not archived"} measured metabolites and {handoff_facts.get("engineered_count", 0)} engineered features. This current-run count is kept separate from overlap counts across the archived Phase 1 screening routes.'
        )
    if upset:
        content_lines.append(
            f'Upstream prior discovery retained {len(final_priors) if final_priors else "N/A"} disease-aligned candidate biomarker(s), and the accompanying Prior Evidence Atlas provides a ranked audit of the adaptive Phase 0 evidence structure rather than a second count of the downstream modeling matrix ({_figure_citation_from_summary(upset)}).'
        )
    if final_priors:
        content_lines.append(
            f'Upstream prior discovery had already retained {len(final_priors)} disease-aligned prior biomarker(s), namely {_join_readable_list([_safe_str(item) for item in final_priors], max_items=10)}, to seed later phases of the workflow.'
        )
    if autogluon_roc:
        content_lines.append(
            f'At the baseline model-screening stage, multi-model ROC comparison selected {selected_model or "the best-performing reference learner"} as the reference learner that was subsequently carried forward into constrained panel optimization ({_figure_citation_from_summary(autogluon_roc)}).'
        )
    if pareto:
        content_lines.append(
            f'The Pareto trajectory recorded a beam-width-{pareto.get("beam_width", "N/A")} search that traversed {pareto.get("layer_count", pareto.get("actual_depth", "N/A"))} completed layers before converging at depth {pareto.get("actual_depth", "N/A")}, supporting the view that the final panel emerged from orderly exploration of the feasible solution space rather than from a single-step heuristic choice ({_figure_citation_from_summary(pareto)}).'
        )
    if radar:
        if overall_improved:
            content_lines.append(
                f'The integrated Phase 2 objective-shift summary further showed that the selected panel improved all four prespecified optimization dimensions relative to the Phase 1 baseline, combining higher discrimination and biological coherence with lower analytical burden and slightly reduced redundancy, while also exposing the winner-feature contribution structure that supported the final choice ({_figure_citation_from_summary(radar)}).'
            )
        else:
            content_lines.append(
                f'The integrated Phase 2 objective-shift summary showed that the selected panel achieved the most favorable overall balance between discrimination, biological coherence, analytical burden, and redundancy control among the shortlisted candidates, together with a feature-level contribution profile for the retained winner features ({_figure_citation_from_summary(radar)}).'
            )

    content_lines.extend([
        '',
        '### Final Panel Performance and Incremental Value',
        (
            f'The final {int(final_count) if final_count is not None else "not archived"}-feature panel consisted of {_join_readable_list(feature_names, max_items=12)}, and achieved a cross-validated AUC of {_format_metric(final_roc.get("auc", winner_panel.get("winner_scores", {}).get("f_perf")))} in internal evaluation ({_figure_citation_from_summary(final_roc)}).'
        ),
    ])
    if holdout_auc is not None:
        cv_auc = _safe_float(final_roc.get('auc', winner_panel.get('winner_scores', {}).get('f_perf')))
        if cv_auc is not None and holdout_auc < cv_auc:
            content_lines.append(
                f'Performance was weaker in the internal holdout (AUC {_format_metric(holdout_auc)}) than in development cross-validation (AUC {_format_metric(cv_auc)}), so the principal result is best described as modest development discrimination with substantial within-study attenuation rather than strong generalization.'
            )
        else:
            content_lines.append(
                f'The internal holdout AUC was {_format_metric(holdout_auc)}; this remains development evidence and not external validation.'
            )
    if raw_count is not None or engineered_count is not None:
        if final_priors:
            prior_retention_clause = (
                f'under the current normalized Phase 0 mapping, {prior_supported_count} of the retained analytes were directly labeled as prior-supported; '
                f'this corresponds to {prior_supported_count} of the {len(final_priors)} archived Phase 0 priors represented in the final panel'
            )
        else:
            prior_retention_clause = (
                f'under the current normalized Phase 0 mapping, {prior_supported_count} of the retained analytes were directly labeled as prior-supported'
            )
        if _safe_float(engineered_count) and int(engineered_count) > 0:
            engineered_text = _join_readable_list([_safe_str(item) for item in engineered_names if _safe_str(item).strip()])
            content_lines.append(
                f'Panel provenance analysis showed that {raw_count if raw_count is not None else "N/A"} retained members were raw metabolites and {engineered_count if engineered_count is not None else "N/A"} was an engineered composite feature, specifically {engineered_text or "an engineered summary feature"}; {prior_retention_clause} ({_figure_citation_from_summary(pareto)} and {_figure_citation_from_summary(final_roc)}).'
            )
        else:
            content_lines.append(
                f'Panel provenance analysis showed that all {raw_count if raw_count is not None else "N/A"} retained members were raw metabolites and that no engineered sums, ratios, or pathway scores survived the final Pareto frontier; {prior_retention_clause} ({_figure_citation_from_summary(pareto)} and {_figure_citation_from_summary(final_roc)}).'
            )
    if incremental_value.get('enabled'):
        content_lines.append(
            f'Relative to the Phase 1 baseline panel, the point estimates were continuous NRI of {_format_metric(nri.get("value"))} (95% CI {_format_ci_range(nri.get("ci_95"))}) and IDI of {_format_metric(idi.get("value"))} (95% CI {_format_ci_range(idi.get("ci_95"))}); these are secondary internal reclassification measures and are not treated as definitive improvement when their intervals include zero ({_figure_citation_from_summary(radar)} and {_figure_citation_from_summary(final_roc)}).'
        )

    content_lines.extend([
        '',
        '### Clinical Operating Characteristics and Utility',
    ])
    if dca:
        content_lines.append(
            f'Exploratory decision-curve analysis under the observed case-control sampling prevalence showed relative net benefit over treat-none across {_format_threshold_range(dca.get("winner_better_than_treat_none_ranges"))}, and over treat-all across {_format_threshold_range(dca.get("winner_better_than_treat_all_ranges"))}; these ranges are descriptive and do not establish transportable clinical utility ({_figure_citation_from_summary(dca)}).'
        )
    if calibration:
        content_lines.append(
            f'The calibration plot provides an internal description of agreement between predicted and observed risk; local departures from the diagonal and the absence of independent external calibration support external recalibration before deployment ({_figure_citation_from_summary(calibration)}).'
        )
    if threshold:
        content_lines.append(
            f'Threshold-performance analysis showed the expected inverse sensitivity-specificity pattern. The default threshold of {_format_metric(default_threshold, 2)} is used as an illustrative operating point that favors case detection, while the companion threshold of {_format_metric(companion_threshold, 2)} is only a relatively more selective comparison point; neither is a validated screening rule ({_figure_citation_from_summary(threshold)}).'
        )

    content_lines.extend([
        '',
        '### Model Interpretability and Exposure-Response Structure',
    ])
    if shap:
        ranked_shap_features = _top_ranked_shap_features(shap, limit=3)
        shap_direction_fragment = _shap_model_direction_sentence(shap)
        local_example_fragment = _shap_local_example_sentence(shap)
        if ranked_shap_features and shap_direction_fragment:
            content_lines.append(
                f'The SHAP ranking placed {_join_readable_list(ranked_shap_features)} among the dominant contributors to model output; {shap_direction_fragment}, and this inverse model-based direction should be interpreted as a multivariable attribution pattern rather than as direct evidence of biological protection, with mechanistic implications considered further in the Discussion ({_figure_citation_from_summary(shap)}).'
            )
            content_lines.append(
                f'{local_example_fragment} ({_figure_citation_from_summary(shap)}).'
            )
        elif ranked_shap_features:
            content_lines.append(
                f'The SHAP summary plot ranked {_join_readable_list(ranked_shap_features)} among the dominant contributors to model output, and {local_example_fragment.lower()} ({_figure_citation_from_summary(shap)}).'
            )
        else:
            content_lines.append(
                f'The SHAP summary plot showed that the retained analytes collectively drove model output through bidirectional local contributions across the cohort, and {local_example_fragment.lower()} ({_figure_citation_from_summary(shap)}).'
            )
    if rcs:
        nonlinear_features = [
            str(item).strip()
            for item in _safe_list(rcs.get('nonlinear_features'))
            if str(item).strip()
        ]
        if nonlinear_features:
            content_lines.append(
                f'Restricted cubic spline analysis identified {_join_readable_list(nonlinear_features)} as the clearest feature(s) with statistically supported non-linear association with the outcome, indicating that at least part of the panel behaves in a threshold-dependent rather than purely linear manner across the observed concentration range ({_figure_citation_from_summary(rcs)}).'
            )
        else:
            content_lines.append(
                f'Restricted cubic spline analysis suggested that the retained metabolites were dominated by approximately monotonic or weakly curved exposure-response patterns within the observed range, with non-linearity remaining limited in the archived feature-level summaries ({_figure_citation_from_summary(rcs)}).'
            )
    return _sanitize_narrative_text('\n'.join(content_lines))


def _build_discussion_content(report_context: Dict[str, Any]) -> str:
    discussion = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('discussion'))
    clinical_utility = _safe_dict(discussion.get('clinical_utility'))
    feature_dictionary = _safe_list(discussion.get('winner_feature_dictionary'))
    figure_evidence = _safe_dict(discussion.get('figure_evidence_summary'))
    disease_name = _display_disease_name(discussion.get('disease_name') or _safe_dict(report_context.get('phase0')).get('disease_name'))
    feature_names = _feature_report_names(feature_dictionary)
    pathways = _safe_list(discussion.get('disease_core_pathways'))
    fig4a = _safe_dict(figure_evidence.get('final_roc'))
    fig4b = _safe_dict(figure_evidence.get('dca'))
    fig4c = _safe_dict(figure_evidence.get('shap'))
    fig4d = _safe_dict(figure_evidence.get('rcs'))
    fig4e = _safe_dict(figure_evidence.get('calibration'))
    fig4f = _safe_dict(figure_evidence.get('threshold_performance'))
    scenario_name = _english_scenario_label(_safe_dict(clinical_utility.get('scenario')))
    nonlinear_features = [
        str(item).strip()
        for item in _safe_list(fig4d.get('nonlinear_features'))
        if str(item).strip()
    ]
    ranked_shap_features = _top_ranked_shap_features(fig4c, limit=4)
    shap_direction_fragment = _shap_direction_sentence_fragment(fig4c)

    lines = [
        '### Summary of Principal Findings',
        (
            f'In the {scenario_name.lower()} setting for {disease_name}, this study identified a {len(feature_dictionary) if feature_dictionary else "not archived"}-feature panel that showed modest development discrimination and a directional incremental signal over the Phase 1 baseline, while threshold behavior remained exploratory across downstream utility analyses ({_figure_citation_from_summary(fig4a)}, {_figure_citation_from_summary(fig4b)}, and {_figure_citation_from_summary(fig4f)}).'
        ),
        '',
        '### Evidence-scoped Biological Interpretation',
    ]
    mapped_pathways = [
        _safe_str(pathway).strip()
        for item in feature_dictionary
        for pathway in _safe_list(_safe_dict(item).get('mapped_pathways'))
        if _safe_str(pathway).strip()
    ]
    prior_supported = [
        _safe_str(_safe_dict(item).get('preferred_report_name') or _safe_dict(item).get('display_name')).strip()
        for item in feature_dictionary
        if _safe_dict(item).get('prior_supported')
    ]
    if pathways or mapped_pathways or prior_supported:
        evidence_labels = _join_readable_list(sorted(set(pathways + mapped_pathways + prior_supported)), max_items=8)
        lines.append(
            f'The retained panel includes {_join_readable_list(feature_names, max_items=12)}. Its biological interpretation is constrained by run-local pathway or prior-evidence annotations, including {evidence_labels}; these annotations provide context for the observed model associations but do not establish pathway activation or disease causality ({_figure_citation_from_summary(fig4c)} and {_figure_citation_from_summary(fig4d)}).'
        )
    else:
        lines.append(
            f'The retained panel includes {_join_readable_list(feature_names, max_items=12)}. This run does not archive Phase 0 prior support, HMDB-linked pathway mapping, or disease-core pathway evidence for the selected features. Accordingly, the report limits this section to model attribution and within-cohort association rather than issuing a disease-mechanistic explanation ({_figure_citation_from_summary(fig4c)} and {_figure_citation_from_summary(fig4d)}).'
        )
    if ranked_shap_features or shap_direction_fragment:
        ranked_phrase = _join_readable_list(ranked_shap_features) if ranked_shap_features else 'the dominant SHAP-ranked metabolites'
        direction_phrase = _shap_model_direction_sentence(fig4c) or 'the archived directionality summary still supports heterogeneous contribution patterns across the retained analytes'
        lines.append(
            f'The SHAP profile further suggests that {ranked_phrase} sit close to the interpretive center of the model, while {direction_phrase}; importantly, this multivariable attribution direction should not be conflated with a simple causal or protective biological effect ({_figure_citation_from_summary(fig4c)}).'
        )
    if nonlinear_features:
        lines.append(
            f'The restricted cubic spline analysis particularly highlighted {_join_readable_list(nonlinear_features)} as feature(s) with non-linear association, indicating that at least part of the panel may operate through threshold-dependent rather than purely linear risk behavior across the observed concentration range ({_figure_citation_from_summary(fig4d)}).'
        )
    if pathways:
        lines.append(
            f'These observations can be interpreted against the archived disease-core programs {_join_readable_list([str(item) for item in pathways[:4]])}, but remain pathway-informed hypotheses rather than direct proof of pathway activation without orthogonal validation ({_figure_citation_from_summary(fig4c)} and {_figure_citation_from_summary(fig4d)}).'
        )

    lines.extend([
        '',
        '### Clinical Translation and Utility',
        (
            f'From a translational perspective, the decision-curve profile indicates that the panel outperforms both no-testing and test-all strategies across broad threshold intervals that are relevant to triage-style {scenario_name.lower()} deployment ({_figure_citation_from_summary(fig4b)}).'
        ),
        (
            f'The threshold-performance plot shows that lowering the decision threshold prioritizes sensitivity, whereas the companion threshold offers a more selective operating rule with improved specificity and positive predictive value, providing a pragmatic way to tune confirmatory workload in real-world primary-care screening ({_figure_citation_from_summary(fig4f)}).'
        ),
        (
            f'The calibration plot supports internal consistency between predicted and observed risk while also underscoring the need for external recalibration before the panel is treated as a transportable probability model or used in formal cost-effectiveness arguments ({_figure_citation_from_summary(fig4e)}).'
        ),
    ])
    return _sanitize_narrative_text('\n'.join(lines))


def _build_limitations_content(report_context: Dict[str, Any]) -> str:
    quality_checks = _safe_dict(report_context.get('quality_checks'))
    clinical_utility = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('clinical_utility'))
    calibration_status = _safe_dict(clinical_utility.get('calibration_status'))
    external_validation = _safe_dict(_safe_dict(report_context.get('phase2')).get('external_validation'))
    missing_inputs = _safe_list(quality_checks.get('missing_required_inputs'))

    lines = []
    if external_validation:
        external_discrimination = _safe_dict(external_validation.get('discrimination_summary'))
        lines.append(
            'An external holdout validation artifact is available for the final panel, '
            f'with ROC-AUC {_format_metric(external_discrimination.get("roc_auc"))} '
            f'and AUPRC {_format_metric(external_discrimination.get("auprc"))}.'
        )
        external_calibration_status = _safe_dict(external_validation.get('calibration_status'))
        if external_calibration_status:
            lines.append(
                'External validation is now available, but site-level transportability and recalibration review '
                f'still remain relevant: recalibration_recommended={external_calibration_status.get("probability_recalibration_recommended", False)}.'
            )
    elif calibration_status:
        lines.append(
            f'Probability calibration remains bounded by internal validation only: external_validation_available={calibration_status.get("external_validation_available", False)}, recalibration_recommended={calibration_status.get("probability_recalibration_recommended", False)}.'
        )
    if external_validation:
        lines.append(
            'Clinical utility and threshold summaries now include an external-holdout assessment, '
            'but broader multi-site transportability should still be verified before deployment claims.'
        )
    else:
        lines.append('Clinical utility, NRI/IDI, and threshold summaries currently reflect internal out-of-fold evaluation and still require external validation before deployment claims.')
    if missing_inputs:
        lines.append(f'Missing upstream artifacts still affecting report completeness: {missing_inputs}.')
    return '\n'.join(lines)


def _build_calibration_interpretation(probability_calibration: Dict[str, Any], calibration_status: Dict[str, Any]) -> str:
    if not probability_calibration:
        warning = str(calibration_status.get('warning', '')).strip()
        if warning:
            return f'Probability calibration assessment is not available; current caveat: {warning}'
        return 'Probability calibration assessment is not available.'

    ece = _safe_float(probability_calibration.get('expected_calibration_error'))
    ici = _safe_float(probability_calibration.get('integrated_calibration_index'))
    slope = _safe_float(probability_calibration.get('calibration_slope'))
    intercept = _safe_float(probability_calibration.get('calibration_intercept'))
    brier = _safe_float(probability_calibration.get('brier_score'))
    transport_applied = bool(probability_calibration.get('transport_applied', False))

    agreement = 'moderate'
    if ece is not None:
        if ece < 0.05:
            agreement = 'good'
        elif ece >= 0.10:
            agreement = 'limited'

    slope_note = 'slope unavailable'
    if slope is not None:
        if 0.9 <= slope <= 1.1:
            slope_note = 'slope is close to ideal'
        elif slope > 1.1:
            slope_note = 'slope is above 1, suggesting imperfect probability scaling'
        else:
            slope_note = 'slope is below 1, suggesting probability over-dispersion/overfitting'

    intercept_note = 'intercept unavailable'
    if intercept is not None:
        if abs(intercept) <= 0.05:
            intercept_note = 'intercept is close to zero'
        elif intercept > 0:
            intercept_note = 'intercept is positive, indicating upward baseline probability shift'
        else:
            intercept_note = 'intercept is negative, indicating downward baseline probability shift'

    warning = str(calibration_status.get('warning', '')).strip()
    if transport_applied:
        sentence = (
            f'Prevalence-adjusted calibration appears {agreement} '
            f'(Brier {_format_metric(brier)}, ICI {_format_metric(ici)}, '
            f'slope {_format_metric(slope)}, intercept {_format_metric(intercept)}); '
            f'{slope_note}; {intercept_note}.'
        )
    else:
        sentence = (
            f'Internal study-prevalence calibration is reported for diagnostic transparency only '
            f'(Brier {_format_metric(brier)}, slope {_format_metric(slope)}, intercept {_format_metric(intercept)}); '
            f'{slope_note}; {intercept_note}. ECE is suppressed from the main report because it is unstable under case-control prevalence distortion.'
        )
    if warning:
        sentence += f' {warning}'
    return sentence


def _build_recalibration_transparency_lines(probability_recalibration: Dict[str, Any]) -> List[str]:
    if not probability_recalibration:
        return []

    raw_calibration = _safe_dict(probability_recalibration.get('raw_probability_calibration'))
    calibrated_calibration = _safe_dict(probability_recalibration.get('calibrated_probability_calibration'))
    improvement = _safe_dict(probability_recalibration.get('calibration_improvement_summary'))
    migration = _safe_dict(probability_recalibration.get('threshold_migration_summary'))
    probability_source = _safe_dict(probability_recalibration.get('recalibration_status'))

    if not raw_calibration and not calibrated_calibration:
        return []

    lines: List[str] = []
    if calibrated_calibration:
        lines.append(
            'Default report interpretations are anchored to recalibrated out-of-fold probabilities rather than the raw winner scores.'
        )

    if raw_calibration or calibrated_calibration:
        lines.append(
            'For transparency, the raw versus calibrated calibration profiles were: '
            f'Brier {_format_metric(raw_calibration.get("brier_score"))} vs {_format_metric(calibrated_calibration.get("brier_score"))}, '
            f'ECE {_format_metric(raw_calibration.get("expected_calibration_error"))} vs {_format_metric(calibrated_calibration.get("expected_calibration_error"))}, '
            f'slope {_format_metric(raw_calibration.get("calibration_slope"))} vs {_format_metric(calibrated_calibration.get("calibration_slope"))}, '
            f'intercept {_format_metric(raw_calibration.get("calibration_intercept"))} vs {_format_metric(calibrated_calibration.get("calibration_intercept"))}.'
        )

    if improvement:
        lines.append(
            f'Recalibration summary: delta Brier {_format_metric(improvement.get("delta_brier_score"))}, '
            f'delta ECE {_format_metric(improvement.get("delta_expected_calibration_error"))}, '
            f'delta slope-distance-to-1 {_format_metric(improvement.get("delta_abs_slope_distance_to_1"))}, '
            f'delta |intercept| {_format_metric(improvement.get("delta_abs_intercept"))}.'
        )

    mappings = _safe_list(migration.get('mappings'))
    if mappings:
        mapping_text = '; '.join(
            f'{_format_metric(item.get("raw_threshold"), 2)} -> {_format_metric(item.get("calibrated_equivalent_threshold"), 2)}'
            for item in mappings[:4]
            if isinstance(item, dict)
        )
        if mapping_text:
            lines.append(
                'Threshold migration guidance for users carrying forward historical cut points is: '
                f'{mapping_text}.'
            )

    warning = _safe_str(probability_source.get('warning')).strip()
    if warning:
        lines.append(warning)
    return lines


def _translate_calibration_warning_to_english(warning: str) -> str:
    normalized = _safe_str(warning).strip()
    if not normalized:
        return ''
    lowered = normalized.lower()
    if 'risk thresholds are derived from internal out-of-fold probabilities without formal probability calibration' in lowered:
        return 'The current risk thresholds are derived from internal out-of-fold probabilities without formal probability calibration; they are therefore suitable for internal decision support, but should not yet be interpreted as stable individual-level absolute risk estimates.'
    if 'external validation' in lowered and 'required' in lowered:
        return 'External validation is still required, and deployment should be preceded by recalibration and threshold confirmation in an independent cohort.'
    return normalized


def _build_dual_threshold_interpretation(
    scenario: Dict[str, Any],
    recommended: Dict[str, Any],
    companion: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    scenario_name = str(scenario.get('name') or scenario.get('key') or 'current scenario')
    intended_use = str(scenario.get('intended_use') or '').strip()
    default_threshold = _safe_float(recommended.get('selected_threshold', scenario.get('default_action_threshold')))
    default_metrics = _safe_dict(recommended.get('operating_characteristics'))
    companion_threshold = _safe_float(companion.get('selected_threshold'))
    companion_metrics = _safe_dict(companion.get('operating_characteristics'))

    if default_threshold is not None:
        prefix = f'For {scenario_name}'
        if intended_use:
            prefix += f' ({intended_use})'
        lines.append(
            f'{prefix}, an illustrative scenario operating point is {_format_metric(default_threshold, 2)}, '
            f'with sensitivity {_format_metric(default_metrics.get("sensitivity"))}, '
            f'specificity {_format_metric(default_metrics.get("specificity"))}, '
            f'PPV {_format_metric(default_metrics.get("ppv"))}, and '
            f'NPV {_format_metric(default_metrics.get("npv"))}.'
        )

    if companion_threshold is not None:
        lines.append(
            f'A data-driven companion threshold of {_format_metric(companion_threshold, 2)} '
            f'is relatively more selective but remains exploratory, with sensitivity '
            f'{_format_metric(companion_metrics.get("sensitivity"))}, specificity '
            f'{_format_metric(companion_metrics.get("specificity"))}, PPV '
            f'{_format_metric(companion_metrics.get("ppv"))}, and NPV '
            f'{_format_metric(companion_metrics.get("npv"))}.'
        )
        if companion.get('narrative'):
            lines.append(str(companion.get('narrative')))
    return lines


def _build_clinical_utility_interpretation(clinical_utility: Dict[str, Any]) -> Dict[str, Any]:
    scenario = _safe_dict(clinical_utility.get('scenario'))
    recommended = _safe_dict(clinical_utility.get('recommended_threshold_summary'))
    companion = _safe_dict(clinical_utility.get('data_driven_companion_threshold_summary'))
    dca = _safe_dict(clinical_utility.get('decision_curve_relative') or clinical_utility.get('decision_curve_summary'))
    resource = _safe_dict(clinical_utility.get('resource_impact_per_1000'))
    probability_calibration = _select_calibration_payload(clinical_utility)
    calibration_status = _safe_dict(clinical_utility.get('calibration_status'))
    threshold_metrics = _safe_list(clinical_utility.get('threshold_metrics_table'))
    probability_recalibration = _safe_dict(clinical_utility.get('probability_recalibration'))
    if probability_calibration.get('evaluation_scope') == 'internal_holdout':
        # OOF recalibration diagnostics are a separate development analysis;
        # do not mix them into the fixed-holdout calibration narrative.
        probability_recalibration = {}
    prevalence_context = _safe_dict(clinical_utility.get('prevalence_context'))

    default_threshold = _safe_float(
        scenario.get('default_action_threshold', recommended.get('selected_threshold'))
    )
    threshold_summary_lines = _build_threshold_metrics_summary(threshold_metrics)
    dca_summary = (
        f'Exploratory decision-curve analysis under the observed case-control sampling prevalence shows relative net benefit versus treat-none across '
        f'{_format_threshold_range(dca.get("winner_better_than_treat_none_ranges"))} '
        f'and versus treat-all across {_format_threshold_range(dca.get("winner_better_than_treat_all_ranges"))}; these ranges are descriptive and not transportable clinical utility evidence.'
    )
    baseline_ranges = _format_threshold_range(dca.get('winner_better_than_baseline_ranges'))
    if baseline_ranges != 'none identified':
        dca_summary += f' Versus baseline, additional favorable relative-utility ranges are {baseline_ranges}.'
    if prevalence_context.get('prevalence_transport_applied'):
        dca_summary += (
            f' Absolute clinical net-benefit interpretation is anchored to a target prevalence of '
            f'{_format_metric(prevalence_context.get("target_prevalence"))}.'
        )

    delta_cost = resource.get("cost_per_additional_high_risk_identified")
    resource_summary = (
        f'At the illustrative operating point {_format_metric(default_threshold, 2)}, '
        f'per-{int(resource.get("population_size", 1000) or 1000)} screening impact includes '
        f'{resource.get("high_risk_identified", "not archived")} flagged individuals requiring review, '
        f'{resource.get("confirmatory_tests_triggered", "not archived")} confirmatory tests triggered, '
        f'and cost per additional high-risk identified of '
        f'{_format_metric(delta_cost, 2) if delta_cost is not None else "N/A"}.'
    )

    overview = (
        f'Intended use: {scenario.get("name") or scenario.get("key") or "unspecified scenario"}; '
        f'illustrative optimization threshold {_format_metric(default_threshold, 2)}; this is not a validated clinical threshold.'
    )
    dual_threshold_lines = _build_dual_threshold_interpretation(scenario, recommended, companion)
    calibration_summary = _build_calibration_interpretation(probability_calibration, calibration_status)
    recalibration_lines = _build_recalibration_transparency_lines(probability_recalibration)

    bullets = [overview]
    bullets.extend(dual_threshold_lines)
    bullets.extend(threshold_summary_lines)
    bullets.append(dca_summary)
    bullets.append(resource_summary)
    bullets.append(calibration_summary)
    bullets.extend(recalibration_lines)
    return {
        'overview': overview,
        'dual_threshold_lines': dual_threshold_lines,
        'threshold_summary_lines': threshold_summary_lines,
        'dca_summary_text': dca_summary,
        'resource_summary_text': resource_summary,
        'calibration_summary_text': calibration_summary,
        'recalibration_transparency_lines': recalibration_lines,
        'auto_summary_markdown': '\n'.join(f'- {line}' for line in bullets if line),
    }


def _merge_fixed_holdout_probability_evidence(
    report_context: Dict[str, Any],
    clinical_utility: Dict[str, Any],
) -> Dict[str, Any]:
    """Make the fixed holdout renderer metadata authoritative for report prose."""
    figure_evidence = _safe_dict(
        _safe_dict(report_context.get('phase3')).get('figure_evidence_summary')
    )
    holdout_calibration = _safe_dict(figure_evidence.get('calibration'))
    holdout_dca = _safe_dict(figure_evidence.get('dca'))
    merged = dict(clinical_utility)
    if (
        holdout_calibration.get('evaluation_scope') == 'internal_holdout'
        and _safe_float(holdout_calibration.get('brier_score')) is not None
    ):
        merged['probability_calibration_adjusted'] = {}
        merged['probability_calibration'] = holdout_calibration
    if holdout_dca.get('evaluation_scope') == 'internal_holdout':
        merged['decision_curve_relative'] = holdout_dca
    return merged


def _build_clinical_utility_chinese_content(report_context: Dict[str, Any]) -> str:
    clinical_utility = _merge_fixed_holdout_probability_evidence(
        report_context,
        _safe_dict(_safe_dict(report_context.get('report_inputs')).get('clinical_utility')),
    )
    # Keep the reader-facing clinical-utility narrative on the fixed holdout
    # audit source used by the report-facing calibration figure.
    run_root = _safe_str(_safe_dict(report_context.get('run')).get('run_root')).strip()
    uncertainty_path = Path(run_root) / 'output' / 'audit' / 'performance_uncertainty.json' if run_root else None
    if uncertainty_path and uncertainty_path.exists():
        try:
            uncertainty = _safe_dict(json.loads(uncertainty_path.read_text(encoding='utf-8')))
            brier = _safe_dict(uncertainty.get('internal_holdout')).get('brier_score')
            if brier is not None:
                primary_calibration = dict(_safe_dict(clinical_utility.get('probability_calibration')))
                primary_calibration['brier_score'] = brier
                clinical_utility = {**clinical_utility, 'probability_calibration': primary_calibration}
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    probability_recalibration_from_report = _safe_dict(
        _safe_dict(report_context.get('report_inputs')).get('probability_recalibration')
    )
    holdout_calibration_is_primary = (
        _safe_dict(clinical_utility.get('probability_calibration')).get('evaluation_scope')
        == 'internal_holdout'
    )
    if probability_recalibration_from_report and not holdout_calibration_is_primary:
        clinical_utility = {
            **clinical_utility,
            'probability_recalibration': probability_recalibration_from_report,
        }
    if not clinical_utility:
        return 'Complete clinical-utility artifacts were not available for this run, so only a placeholder summary can be provided.'

    scenario = _safe_dict(clinical_utility.get('scenario'))
    scenario_name = _english_scenario_label(scenario)
    recommended = _safe_dict(clinical_utility.get('recommended_threshold_summary'))
    companion = _safe_dict(clinical_utility.get('data_driven_companion_threshold_summary'))
    dca = _safe_dict(clinical_utility.get('decision_curve_relative') or clinical_utility.get('decision_curve_summary'))
    resource = _safe_dict(clinical_utility.get('resource_impact_per_1000'))
    calibration = _select_calibration_payload(clinical_utility)
    calibration_status = _safe_dict(clinical_utility.get('calibration_status'))
    probability_recalibration = _safe_dict(clinical_utility.get('probability_recalibration'))
    if calibration.get('evaluation_scope') == 'internal_holdout':
        # OOF recalibration diagnostics are a separate development analysis;
        # do not mix them into the fixed-holdout calibration narrative.
        probability_recalibration = {}
    prevalence_context = _safe_dict(clinical_utility.get('prevalence_context'))

    default_metrics = _safe_dict(recommended.get('operating_characteristics'))
    companion_metrics = _safe_dict(companion.get('operating_characteristics'))
    lines = [
        '### Illustrative High-sensitivity Operating Point',
        (
            f'The default action threshold is {_format_metric(recommended.get("selected_threshold", scenario.get("default_action_threshold")), 2)} in the archived scenario; it is displayed as an illustrative high-sensitivity operating point, '
            f'with sensitivity {_format_metric(default_metrics.get("sensitivity"))}, specificity {_format_metric(default_metrics.get("specificity"))}, '
            f'PPV {_format_metric(default_metrics.get("ppv"))}, and NPV {_format_metric(default_metrics.get("npv"))}. '
            'This operating point favors sensitivity, but it is not a recommended or validated clinical threshold.'
        ),
    ]
    if companion:
        lines.extend([
            '',
            '### Illustrative Higher-specificity Comparison Point',
            (
                f'A data-driven companion threshold of {_format_metric(companion.get("selected_threshold"), 2)} yields sensitivity {_format_metric(companion_metrics.get("sensitivity"))}, '
                f'specificity {_format_metric(companion_metrics.get("specificity"))}, PPV {_format_metric(companion_metrics.get("ppv"))}, and NPV {_format_metric(companion_metrics.get("npv"))}. '
                'This more selective operating point may be preferable when confirmatory resources are limited and false-positive downstream work-up needs to be contained.'
            ),
        ])

    lines.extend([
        '',
        '### Decision-Curve and Workflow Impact',
        (
            f'Decision-curve analysis shows net benefit relative to treat-none across {_format_threshold_range(dca.get("winner_better_than_treat_none_ranges"))} '
            f'and relative to treat-all across {_format_threshold_range(dca.get("winner_better_than_treat_all_ranges"))} under the observed case-control sampling scheme.'
        ),
        (
            f'On a per-{int(resource.get("population_size", 1000) or 1000)} screened population basis within the archived sampling assumptions, the illustrative operating point identifies '
            f'{resource.get("high_risk_identified", "N/A")} individuals as high risk and triggers {resource.get("confirmatory_tests_triggered", "N/A")} confirmatory evaluations, '
            'so threshold selection should be understood as a practical balance between case finding and downstream workflow burden.'
        ),
        '',
        '### Calibration and Deployment Caveat',
        (
            (
                f'Prevalence-adjusted calibration metrics were Brier {_format_metric(calibration.get("brier_score"))}, ICI {_format_metric(calibration.get("integrated_calibration_index"))}, '
                f'slope {_format_metric(calibration.get("calibration_slope"))}, and intercept {_format_metric(calibration.get("calibration_intercept"))}. '
                'These estimates are more suitable for deployment-facing interpretation because prevalence transport was applied.'
            )
            if prevalence_context.get('prevalence_transport_applied')
            else
            (
                f'Internal calibration metrics were Brier {_format_metric(calibration.get("brier_score"))}, '
                f'slope {_format_metric(calibration.get("calibration_slope"))}, and intercept {_format_metric(calibration.get("calibration_intercept"))}. '
                'These values describe the observed study prevalence; ECE is intentionally omitted from the main report, and the estimates should not be interpreted as target-population absolute-risk calibration.'
            )
        ),
    ])
    lines.extend([''] + _build_recalibration_transparency_lines(probability_recalibration) if probability_recalibration else [])
    warning = _safe_str(calibration_status.get('warning')).strip()
    if warning:
        lines.append(f'Deployment note: {_translate_calibration_warning_to_english(warning)}')
    lines.append('Clinical recommendation: at the current stage, the panel is best used for adjunctive risk stratification and prioritization of confirmatory follow-up, not as a stand-alone diagnostic test.')
    return _sanitize_narrative_text('\n'.join(lines))


def _select_figures_by_section(report_context: Dict[str, Any], section_name: str) -> List[Dict[str, Any]]:
    figures = _safe_list(_safe_dict(report_context.get('phase3')).get('figures'))
    return [
        figure for figure in figures
        if isinstance(figure, dict) and figure.get('section') == section_name
    ]


def _figure_narrative_contract(report_context: Dict[str, Any], figure_ids: List[str]) -> Dict[str, Any]:
    figure_summary = _figure_evidence_summary(report_context)
    figures = _safe_dict(figure_summary.get('figures'))
    contract: Dict[str, Any] = {}
    role_map = {
        'fig1a': 'sample structure after preprocessing',
        'fig1b': 'supervised sample separation after preprocessing',
        'fig1c': 'adaptive prior evidence atlas for retained versus excluded metabolites',
        'fig2a': 'baseline multi-model ROC comparison',
        'fig2b': 'Pareto-guided search trajectory',
        'fig3': 'Phase 2 objective-shift summary and winner feature contribution',
        'fig4a': 'final panel discrimination',
        'fig4b': 'decision-curve utility',
        'fig4c': 'global feature attribution and local explanation example',
        'fig4d': 'continuous exposure-response structure',
        'fig4e': 'internal calibration',
        'fig4f': 'threshold-dependent operating characteristics',
    }
    for figure_id in figure_ids:
        payload = _safe_dict(figures.get(figure_id))
        if not payload:
            continue
        contract[figure_id] = {
            'figure_label': payload.get('figure_label'),
            'asset_name': payload.get('asset_name'),
            'title': payload.get('title'),
            'narrative_role': role_map.get(figure_id, 'supporting visual evidence'),
        }
    return contract


def _figure_refs(figures: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    for figure in figures:
        primary = _safe_dict(figure.get('primary_output'))
        refs.append(
            {
                'figure_id': str(figure.get('figure_id', '')),
                'title': str(figure.get('title', '')),
                'path': str(primary.get('path', '')),
            }
        )
    return refs


def _global_constraints(report_context: Dict[str, Any]) -> List[str]:
    quality_checks = _safe_dict(report_context.get('quality_checks'))
    missing_inputs = _safe_list(quality_checks.get('missing_required_inputs'))
    constraints = [
        'Only use facts present in report_context.json and referenced upstream artifacts.',
        'Do not invent biological mechanisms, p-values, confidence intervals, or sample counts that are not explicitly available.',
        'When an upstream artifact is missing, acknowledge the gap instead of fabricating a claim.',
        'Keep methodology sections objective and reproducible; reserve interpretation for the discussion section.',
        'Use preferred human-readable feature names from the feature dictionary instead of raw HMDB or token identifiers whenever those names are available.',
        'In Results and Discussion, empirical statements should end with supporting figure citations in parentheses whenever figure-backed evidence is available.',
    ]
    if missing_inputs:
        constraints.append(
            'Current missing upstream artifacts: ' + ', '.join(str(item) for item in missing_inputs) + '.'
        )
    return constraints


def _methodology_section_blueprint() -> List[Dict[str, str]]:
    return [
        {
            'subsection': 'Prior Biomarker Discovery and Evidence Scoring',
            'purpose': 'Explain candidate prior discovery, PubMed-based coarse screening, structured evidence-pack construction, and the 3+1 rubric used to retain high-confidence priors.',
        },
        {
            'subsection': 'Data Preprocessing and Stable Candidate Generation',
            'purpose': 'Describe preprocessing, missingness handling, normalization, scaling, and multi-method candidate reduction before panel optimization.',
        },
        {
            'subsection': 'Scenario-Aware Multi-objective Panel Optimization',
            'purpose': 'Describe scenario encoding, Pareto-guided search, TOPSIS scalarization, and dual-criteria early stopping without presenting downstream performance outcomes as results.',
        },
        {
            'subsection': 'Statistical Evaluation and Validation Framework',
            'purpose': 'State how ROC, incremental value, calibration, decision-curve, threshold, SHAP, and RCS analyses were defined before the Results narrative.',
        },
    ]


def _results_section_blueprint() -> List[Dict[str, str]]:
    return [
        {
            'subsection': 'Candidate-to-Panel Selection Pathway',
            'purpose': 'Explain how the workflow moved from Phase 1 candidate features to the final Pareto-selected panel using Fig. 1c, Fig. 2a, Fig. 2b, and Fig. 3.',
        },
        {
            'subsection': 'Final Panel Performance and Incremental Value',
            'purpose': 'Report the final panel composition, discrimination, incremental value, and panel provenance using explicit numbers rather than vague summary language.',
        },
        {
            'subsection': 'Clinical Operating Characteristics and Utility',
            'purpose': 'Explain DCA, calibration, threshold behavior, and clinical operating points in one coherent results module.',
        },
        {
            'subsection': 'Model Interpretability and Exposure-Response Structure',
            'purpose': 'Interpret SHAP and RCS outputs while clearly distinguishing model-based attribution from biological causality.',
        },
    ]


def _discussion_section_blueprint() -> List[Dict[str, str]]:
    return [
        {
            'subsection': 'Summary of Principal Findings',
            'purpose': 'Summarize the core empirical findings of the final selected panel and its incremental value.',
        },
        {
            'subsection': 'Evidence-scoped Biological Interpretation',
            'purpose': 'Interpret retained metabolites only through registered pathway, prior-evidence and model-attribution sources; otherwise expose the lack of mechanistic evidence.',
        },
        {
            'subsection': 'Clinical Translation and Utility',
            'purpose': 'Discuss threshold selection, DCA, calibration, and implementation implications for real-world screening workflows.',
        },
    ]


def _build_executive_summary_section(report_context: Dict[str, Any]) -> Dict[str, Any]:
    executive = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('executive_summary'))
    winner_panel = _safe_dict(_safe_dict(_safe_dict(report_context.get('report_inputs')).get('results')).get('winner_panel'))
    return {
        'title': SECTION_TITLES['executive_summary'],
        'status': 'pending_llm',
        'generation_policy': SEEDED_LLM_POLICY,
        'content': _build_executive_summary_content(report_context),
        'objective': 'Summarize the winner panel, discrimination gain, readable panel composition, and near-clinical interpretation in one concise paragraph.',
        'style_constraints': [
            'Write exactly four sentences in polished abstract style.',
            'Lead with study context and final panel performance rather than implementation detail.',
            'Use report-facing feature names instead of raw feature tokens.',
            'Keep mechanistic interpretation concise and integrated into the final sentence.',
        ],
        'required_quantitative_mentions': [
            'Cross-validated AUC of the final panel.',
            'Continuous NRI and IDI versus the Phase 1 baseline panel.',
            'Primary screening threshold and more selective companion threshold when available.',
        ],
        'forbidden_phrasings': [
            'Do not describe the panel only as "better" without quantitative anchors.',
            'Do not expose raw optimization/debug variable names.',
        ],
        'required_facts': {
            'queue_size': executive.get('queue_size'),
            'confirmed_biomarker_count': executive.get('confirmed_biomarker_count'),
            'final_priors_count': executive.get('final_priors_count'),
            'final_feature_count': executive.get('final_feature_count'),
            'best_auc': executive.get('best_auc'),
            'selected_model': executive.get('selected_model'),
            'bio_score_version': executive.get('bio_score_version'),
            'protected_anchor_count': executive.get('protected_anchor_count'),
            'anchor_mode': executive.get('anchor_mode'),
            'memory_prior_enabled': executive.get('memory_prior_enabled'),
            'nri': _safe_dict(executive.get('nri')),
            'idi': _safe_dict(executive.get('idi')),
            'core_pathways': _safe_list(executive.get('core_pathways')),
            'winner_features': _safe_list(winner_panel.get('winner_features')),
            'winner_feature_dictionary': _safe_list(executive.get('winner_feature_dictionary')),
            'winner_panel_name_summary': _safe_dict(executive.get('winner_panel_name_summary')),
        },
        'recommended_figures': _figure_refs(_select_figures_by_section(report_context, 'Results')),
        'missing_evidence': [],
        'prompt_scaffold': (
            'Use the seed as a compact abstract outline, but rewrite it into stronger abstract-style prose while preserving all named quantitative anchors and readable metabolite names.'
        ),
    }


def _build_methodology_section(report_context: Dict[str, Any]) -> Dict[str, Any]:
    methodology = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('methodology'))
    phase0 = _safe_dict(methodology.get('phase0'))
    phase1 = _safe_dict(methodology.get('phase1'))
    phase2 = _safe_dict(methodology.get('phase2'))
    phase2_search_summary = _safe_dict(methodology.get('phase2_search_summary'))
    return {
        'title': SECTION_TITLES['methodology_workflow'],
        'status': 'pending_llm',
        'generation_policy': SEEDED_LLM_POLICY,
        'content': _build_methodology_content(report_context),
        'objective': 'Describe the end-to-end manuscript methods as four scientific stages: prior biomarker evidence scoring, preprocessing and stable candidate generation, scenario-aware multi-objective optimization, and statistical evaluation.',
        'style_constraints': [
            'Use exactly four subsections in this order: Prior Biomarker Discovery and Evidence Scoring; Data Preprocessing and Stable Candidate Generation; Scenario-Aware Multi-objective Panel Optimization; Statistical Evaluation and Validation Framework.',
            'Use objective scientific reporting tone in smooth academic prose.',
            'Translate structured system parameters into readable methodological description rather than key-value narration.',
            'Do not expose internal optimization variable names unless they are indispensable for reproducibility.',
            'Do not discuss downstream biological interpretation in this section.',
            'When figures are mentioned in Methodology, use them only to document workflow or search behavior rather than endpoint performance claims.',
            'Make the methods read like a manuscript workflow rather than a Phase 0/1/2/3 engineering log.',
        ],
        'section_blueprint': _methodology_section_blueprint(),
        'section_boundaries': [
            'The first subsection should cover candidate prior discovery, literature triage, and rubric-based evidence scoring rather than downstream panel outcomes.',
            'The second subsection should cover missingness handling, normalization, scaling, selector voting, and baseline learner screening rather than final ROC or utility claims.',
            'The third subsection should cover scenario encoding, TOPSIS/Pareto search, and early stopping rules rather than downstream model interpretation.',
            'The fourth subsection may define ROC, NRI/IDI, calibration, DCA, threshold metrics, SHAP, and RCS as prespecified analyses, but should not report their results.',
        ],
        'required_qualitative_mentions': [
            'Translate missingness/imputation settings into fluent academic prose rather than key-value narration.',
            'Explain the 3+1 prior-evidence rubric in prose when the rubric framework is available.',
            'Clarify that statistical validation and interpretability analyses were prespecified before the Results section reports them.',
            'Explain that Phase 1 candidate generation deliberately allowed engineered candidates such as sums, ratios, and pathway-level features before reporting how many survived screening.',
            'When imbalance-aware stability-search facts are available, describe the rationale for repeated stability iterations and cross-method consensus without overstating unsupported execution details.',
            'If the selected AutoGluon baseline learner is a WeightedEnsemble, explain once that it is an automatically learned weighted combination of multiple high-ranking base learners selected according to validation performance rather than a single standalone learner.',
        ],
        'forbidden_phrasings': [
            'Avoid rows_removed=0, columns_removed=0, imputation_method=none_needed style wording.',
            'Avoid programmer/debug phrasing such as f_perf or f_cost unless strictly required for reproducibility.',
            'Avoid presenting the methods as a raw Phase 0/1/2/3 execution trace.',
            'Avoid jumping directly to engineered-feature counts without first explaining why combination features were generated.',
        ],
        'figure_writing_instructions': [
            'Do not discuss PCA or PLS-DA in Methodology unless the contract explicitly requires them.',
            'Fig. 1c should be described as feature-overlap context, not as a final performance figure or as a second count of the final Phase 1 handoff set.',
            'Fig. 2b should be described as the search trajectory and audit trail of Pareto-guided panel search.',
        ],
        'required_facts': {
            'phase0': phase0,
            'phase1': phase1,
            'phase2': phase2,
            'phase2_search_summary': phase2_search_summary,
            'figure_evidence_summary': _safe_dict(methodology.get('figure_evidence_summary')),
            'figure_narrative_contract': _figure_narrative_contract(report_context, ['fig1c', 'fig2b']),
        },
        'recommended_figures': _figure_refs(_select_figures_by_ids(report_context, ['fig1c', 'fig2b'])),
        'missing_evidence': [],
        'prompt_scaffold': (
            'Treat the seed as a methodological outline, not as final wording. Keep the four fixed subsection titles unchanged, write the section as a manuscript methods workflow, and ensure that prior evidence scoring, preprocessing, optimization, and statistical evaluation are each described explicitly when their facts are available. If the selected baseline learner is an AutoGluon WeightedEnsemble, preserve the explanation that it represents an automatically learned weighted combination of high-ranking base learners selected according to validation performance.'
        ),
    }


def _build_results_section(report_context: Dict[str, Any]) -> Dict[str, Any]:
    results = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('results'))
    winner_panel = _safe_dict(results.get('winner_panel'))
    figures = _safe_list(results.get('figures'))
    result_figure_ids = _safe_list(_figure_evidence_summary(report_context).get('results_figure_ids')) or ['fig1c', 'fig2a', 'fig2b', 'fig3', 'fig4a', 'fig4b', 'fig4c', 'fig4d', 'fig4e', 'fig4f']
    return {
        'title': SECTION_TITLES['results'],
        'status': 'pending_llm',
        'generation_policy': SEEDED_LLM_POLICY,
        'content': _build_results_content(report_context),
        'objective': 'Report the candidate-to-panel pathway, final panel performance, clinical operating characteristics, and interpretability outputs in a fixed figure-linked evidence chain.',
        'style_constraints': [
            'Use exactly four subsections in this order: Candidate-to-Panel Selection Pathway; Final Panel Performance and Incremental Value; Clinical Operating Characteristics and Utility; Model Interpretability and Exposure-Response Structure.',
            'Use readable metabolite names, with HMDB identifiers only in dedicated composition lists.',
            'Separate clinically interpretable validation results from internal optimization logic.',
            'Do not expose internal objective variable names such as f_perf or f_cost in narrative prose.',
            'Mention every available Results figure at least once when supported by archived facts.',
            'All result statements should end with supporting figure citations in parentheses.',
            'Retain key quantitative details from the seed summary, including NRI/IDI with confidence intervals, threshold values, panel provenance counts, and the indexed SHAP waterfall example.',
        ],
        'section_blueprint': _results_section_blueprint(),
        'section_boundaries': [
            'Candidate-to-Panel Selection Pathway should cover candidate overlap, baseline learner screening, Pareto search path, and the Phase 2 objective-shift summary, not downstream clinical utility.',
            'Final Panel Performance and Incremental Value should report the final panel composition, AUC, NRI/IDI, and provenance counts.',
            'Clinical Operating Characteristics and Utility should integrate DCA, calibration, and threshold operating points as one coherent result module.',
            'Model Interpretability and Exposure-Response Structure should cover SHAP and RCS while clearly separating model attribution from biological causality.',
        ],
        'required_quantitative_mentions': [
            'AUC of the final panel.',
            'Continuous NRI and IDI with 95% confidence intervals when available.',
            'Default threshold and companion threshold values.',
            'Panel provenance counts: raw_count, engineered_count, and prior_supported_count.',
            'Indexed SHAP waterfall example when a patient index is available.',
        ],
        'required_qualitative_mentions': [
            'State explicitly that no engineered features survived if engineered_count is zero.',
            'If engineered_count is greater than zero, say so directly and name the surviving engineered feature when available.',
            'If prior_supported_count is zero, say so directly rather than implying prior confirmation.',
            'When Phase 0 final priors are available, report how many were retained upstream and name them before explaining whether any survived into the final panel.',
            'If SHAP direction points away from case probability, state that this is a model-based attribution pattern and leave room for mechanistic discussion later.',
            'Clarify that the candidate-pool size must come from the current Phase 1 handoff facts, not from older runs or from the UpSet overlap counts.',
        ],
        'forbidden_phrasings': [
            'Do not write vague lines such as "better overall reclassification" without the NRI/IDI numbers.',
            'Do not write generic phrases such as "established the reference learner family" without naming the selected learner when available.',
            'Do not describe SHAP direction as direct biological protection or causality.',
            'Do not omit the patient index when the SHAP waterfall example is indexed.',
            'Do not claim that all retained features are raw metabolites when engineered_count is non-zero.',
        ],
        'figure_writing_instructions': [
            'Fig. 1c should be used to describe archived overlap across Phase 1 screening routes, but do not mix that visualization with the dynamic Phase 1 handoff count supplied in the required facts.',
            'Fig. 2a must identify the baseline learner family when the selected model name is available.',
            'Fig. 2b must explain the Pareto-guided search trajectory and orderly solution-space exploration.',
            'Fig. 3 must explain the multi-objective improvement relative to the Phase 1 baseline and, when relevant, note the winner-feature contribution structure shown in the same figure.',
            'Fig. 4a must report final discrimination numerically.',
            'Fig. 4b must describe the threshold ranges with net benefit over treat-none and treat-all.',
            'Fig. 4c must cover global SHAP ranking plus the indexed waterfall example.',
            'Fig. 4d must state which retained analytes show non-linear association when supported.',
            'Fig. 4e must comment on agreement between predicted and observed risk.',
            'Fig. 4f must report threshold-dependent operating characteristics with explicit operating points.',
        ],
        'required_facts': {
            'winner_panel': winner_panel,
            'phase0_final_priors': _safe_list(_safe_dict(report_context.get('phase0')).get('final_priors')),
            'phase0_final_priors_count': _safe_dict(report_context.get('phase0')).get('final_priors_count'),
            'feature_records': _safe_list(results.get('feature_records')),
            'feature_dictionary': _safe_list(results.get('feature_dictionary')),
            'winner_panel_name_summary': _safe_dict(results.get('winner_panel_name_summary')),
            'phase1_candidate_pool_count': results.get('phase1_candidate_pool_count'),
            'incremental_value': _safe_dict(results.get('incremental_value')),
            'memory_prior_summary': _safe_dict(results.get('memory_prior_summary')),
            'scenario_strategy': _safe_dict(results.get('scenario_strategy')),
            'figure_count': len(figures),
            'figures': figures,
            'figure_evidence_summary': _safe_dict(results.get('figure_evidence_summary')),
            'figure_narrative_contract': _figure_narrative_contract(report_context, result_figure_ids),
        },
        'recommended_figures': _figure_refs(_select_figures_by_ids(report_context, result_figure_ids)),
        'missing_evidence': [],
        'prompt_scaffold': (
            'Use the seed as a structured evidence outline rather than a final draft. Preserve the fixed four-subsection structure, keep every empirical statement figure-linked, carry forward all mandated numbers, and favor manuscript-style prose over rigid sentence-by-sentence paraphrase.'
        ),
    }


def _build_clinical_utility_section(report_context: Dict[str, Any]) -> Dict[str, Any]:
    clinical_utility = _merge_fixed_holdout_probability_evidence(
        report_context,
        _safe_dict(_safe_dict(report_context.get('report_inputs')).get('clinical_utility')),
    )
    probability_recalibration = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('probability_recalibration'))
    # Prefer the run-scoped fixed-holdout audit for the primary Brier score.
    run_root = _safe_str(_safe_dict(report_context.get('run')).get('run_root')).strip()
    uncertainty_path = Path(run_root) / 'output' / 'audit' / 'performance_uncertainty.json' if run_root else None
    if uncertainty_path and uncertainty_path.exists():
        try:
            uncertainty = _safe_dict(json.loads(uncertainty_path.read_text(encoding='utf-8')))
            holdout = _safe_dict(uncertainty.get('internal_holdout'))
            brier = holdout.get('brier_score')
            if brier is not None:
                primary_calibration = dict(_safe_dict(clinical_utility.get('probability_calibration')))
                primary_calibration['brier_score'] = brier
                clinical_utility = {**clinical_utility, 'probability_calibration': primary_calibration}
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    holdout_calibration_is_primary = (
        _safe_dict(clinical_utility.get('probability_calibration')).get('evaluation_scope')
        == 'internal_holdout'
    )
    if probability_recalibration and not holdout_calibration_is_primary:
        clinical_utility = {
            **clinical_utility,
            'probability_recalibration': probability_recalibration,
        }
    missing_evidence: List[str] = []
    if not clinical_utility:
        missing_evidence.append('phase2_clinical_utility')
    interpretation = _build_clinical_utility_interpretation(clinical_utility) if clinical_utility else {}

    return {
        'title': SECTION_TITLES['clinical_utility_decision_support'],
        'status': 'pending_llm',
        'generation_policy': SEEDED_LLM_POLICY,
        'content': _build_clinical_utility_chinese_content(report_context),
        'objective': (
            'Describe intended use, prespecified risk thresholds, operating characteristics, '
            'decision-curve utility, resource impact, and deployment caveats for the recommended panel.'
        ),
        'style_constraints': [
            'Use concise clinical decision-support English rather than generic manuscript prose.',
            'State clearly that threshold-based utility is derived from internal model evaluation when calibration or external validation is unavailable.',
            'Prefer scenario-specific threshold interpretation over generic statements about risk.',
        ],
        'section_boundaries': [
            'Focus on intended use, thresholds, workflow burden, decision curves, and calibration caveats.',
            'Do not repeat the full biological discussion here.',
        ],
        'required_quantitative_mentions': [
            'Default threshold and companion threshold with operating characteristics when available.',
            'Decision-curve threshold ranges.',
            'Resource impact per 1000 screened individuals when available.',
        ],
        'forbidden_phrasings': [
            'Avoid generic clinical claims such as "clinically useful" without threshold or operating-characteristic detail.',
        ],
        'required_facts': {
            'clinical_utility': clinical_utility,
            'scenario': _safe_dict(clinical_utility.get('scenario')),
            'recommended_threshold_summary': _safe_dict(clinical_utility.get('recommended_threshold_summary')),
            'data_driven_companion_threshold_summary': _safe_dict(
                clinical_utility.get('data_driven_companion_threshold_summary')
            ),
            'decision_curve_summary': _safe_dict(clinical_utility.get('decision_curve_summary')),
            'resource_impact_per_1000': _safe_dict(clinical_utility.get('resource_impact_per_1000')),
            'threshold_metrics_table': _safe_list(clinical_utility.get('threshold_metrics_table')),
            'probability_calibration': _safe_dict(clinical_utility.get('probability_calibration')),
            'calibration_status': _safe_dict(clinical_utility.get('calibration_status')),
            'probability_recalibration': probability_recalibration,
            'auto_interpretation': interpretation,
        },
        'recommended_figures': _figure_refs(_select_figures_by_section(report_context, 'Results')),
        'missing_evidence': missing_evidence,
        'prompt_scaffold': (
            'Use the seed as a decision-support outline, but rewrite it into fluent clinical prose that preserves thresholds, operating characteristics, and deployment caveats.'
        ),
    }


def _build_discussion_section(report_context: Dict[str, Any]) -> Dict[str, Any]:
    discussion = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('discussion'))
    missing_inputs = _safe_list(_safe_dict(report_context.get('quality_checks')).get('missing_required_inputs'))
    discussion_figure_ids = _safe_list(_figure_evidence_summary(report_context).get('discussion_figure_ids')) or ['fig4a', 'fig4b', 'fig4c', 'fig4d', 'fig4e', 'fig4f']
    return {
        'title': SECTION_TITLES['discussion_mechanistic_insights'],
        'status': 'pending_llm',
        'generation_policy': SEEDED_LLM_POLICY,
        'content': _build_discussion_content(report_context),
        'objective': 'Provide mechanistic interpretation of the winner panel while staying grounded in the identified pathways, feature types, and known Phase 0 disease context.',
        'style_constraints': [
            'Write as cohesive academic prose in a Nature Medicine-style discussion voice.',
            'Use exactly three subsections in this order: Summary of Principal Findings; Evidence-scoped Biological Interpretation; Clinical Translation and Utility.',
            'Do not use tier labels, bullet points, or feature-by-feature enumeration.',
            'Build the biological interpretation from the retained panel and disease-context facts of the current run rather than from disease-specific stock narratives.',
            'Use preferred human-readable feature names from the feature dictionary and avoid raw pathway-debug language.',
            'Do not overstate causality; maintain mechanistic restraint where direct validation is absent.',
            'All empirical or translational statements should end with supporting figure citations in parentheses.',
        ],
        'section_blueprint': _discussion_section_blueprint(),
        'section_boundaries': [
            'Summary of Principal Findings should restate the headline empirical findings rather than re-describing the full workflow.',
            'Evidence-scoped Biological Interpretation should describe only registered pathway or prior-evidence links. When those sources are absent, it must explicitly restrict interpretation to model association rather than inventing a mechanism.',
            'Clinical Translation and Utility should synthesize DCA, threshold behavior, and calibration into deployment-oriented reasoning.',
        ],
        'required_qualitative_mentions': [
            'Base biological interpretation on the actual retained panel and registered evidence of the current run instead of older disease templates.',
            'If SHAP directionality appears counterintuitive for a known marker, explicitly frame it as multivariable attribution rather than simple biological protection.',
        ],
        'forbidden_phrasings': [
            'Do not use Tier 1/Tier 2/Tier 3 style labels.',
            'Do not switch into bullet-style feature enumeration.',
            'Do not convert model-attribution direction into direct biological causality.',
        ],
        'figure_writing_instructions': [
            'Use Fig. 4a, Fig. 4b, and Fig. 4f to anchor the principal empirical findings.',
            'Use Fig. 4c and Fig. 4d to support mechanistic interpretation while preserving causal restraint.',
            'Use Fig. 4b, Fig. 4e, and Fig. 4f to discuss translational utility and deployment caveats.',
        ],
        'required_facts': {
            'discussion': discussion,
            'figure_narrative_contract': _figure_narrative_contract(report_context, discussion_figure_ids),
        },
        'recommended_figures': _figure_refs(_select_figures_by_ids(report_context, discussion_figure_ids)),
        'missing_evidence': [str(item) for item in missing_inputs if item in {'feature_provenance', 'figure_manifest'}],
        'prompt_scaffold': (
            'Use the seed as a mechanistic outline rather than as final wording. Write elegant academic prose under the fixed three-subsection Discussion structure, keep mechanism and translation continuous rather than list-like, and preserve figure-anchored empirical grounding.'
        ),
    }


def _build_limitations_section(report_context: Dict[str, Any]) -> Dict[str, Any]:
    quality_checks = _safe_dict(report_context.get('quality_checks'))
    phase2_runtime = _safe_dict(_safe_dict(report_context.get('phase2')).get('runtime_summary'))
    clinical_utility = _safe_dict(_safe_dict(report_context.get('report_inputs')).get('clinical_utility'))
    return {
        'title': SECTION_TITLES['limitations'],
        'status': 'pending_llm',
        'generation_policy': SEEDED_LLM_POLICY,
        'content': _build_limitations_content(report_context),
        'objective': 'Summarize current analytical limitations, missing artifacts, and next validation steps without weakening factual results.',
        'style_constraints': [
            'Focus on sample size, artifact completeness, and validation needs.',
            'Keep the tone balanced and practical.',
            'Do not introduce limitations that contradict available facts.',
        ],
        'required_facts': {
            'missing_required_inputs': _safe_list(quality_checks.get('missing_required_inputs')),
            'missing_source_files': _safe_list(quality_checks.get('missing_source_files')),
            'runtime_summary': phase2_runtime,
            'calibration_status': _safe_dict(clinical_utility.get('calibration_status')),
        },
        'recommended_figures': [],
        'missing_evidence': _safe_list(quality_checks.get('missing_required_inputs')),
        'prompt_scaffold': (
            'Write a limitations paragraph that notes any missing upstream artifacts, the need for external validation, '
            'the internal-only nature of current calibration / NRI / IDI / threshold utility estimates, and the caution required when interpreting a computationally selected biomarker panel.'
        ),
    }


def build_placeholder_llm_sections(report_context: Dict[str, Any]) -> Dict[str, Any]:
    """Create an LLM-ready `llm_sections.json` contract for later integration."""

    quality_checks = _safe_dict(report_context.get('quality_checks'))
    sections = {
        'executive_summary': _build_executive_summary_section(report_context),
        'methodology_workflow': _build_methodology_section(report_context),
        'results': _build_results_section(report_context),
        'clinical_utility_decision_support': _build_clinical_utility_section(report_context),
        'discussion_mechanistic_insights': _build_discussion_section(report_context),
    }

    return {
        'schema_version': 'phase4.llm_sections.v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'run_id': report_context.get('run', {}).get('run_id', ''),
        'generation_mode': 'structured_placeholder',
        'section_order': DEFAULT_SECTION_KEYS,
        'global_guidance': {
            'audience': 'biomedical research readers and downstream report-writing LLMs',
            'language': 'english',
            'constraints': _global_constraints(report_context),
            'ready_for_llm': bool(quality_checks.get('ready_for_llm', False)),
            'missing_required_inputs': _safe_list(quality_checks.get('missing_required_inputs')),
        },
        'sections': sections,
        'summary': {
            'section_count': len(sections),
            'pending_section_count': sum(
                1 for section in sections.values()
                if isinstance(section, dict) and section.get('status') == 'pending_llm'
            ),
        },
    }


def _llm_disabled() -> bool:
    return os.getenv('METABOAGENT_PHASE4_DISABLE_LLM', '').strip().lower() in {'1', 'true', 'yes'}


def _initialize_phase4_llm_client(config_path: str) -> Optional[Any]:
    if _llm_disabled():
        logger.info("Phase 4 LLM generation disabled by METABOAGENT_PHASE4_DISABLE_LLM")
        return None

    config = ConfigManager(config_path)
    llm_config = (
        config.get_phase4_llm_config() if hasattr(config, 'get_phase4_llm_config') else {}
    ) or (
        config.get_phase0_llm_config() if hasattr(config, 'get_phase0_llm_config') else {}
    )
    provider = _safe_str(llm_config.get('provider')).strip().lower() or 'openai_compatible'
    api_key = _safe_str(llm_config.get('api_key') or os.getenv('OPENAI_API_KEY')).strip()
    api_base = _safe_str(llm_config.get('api_base') or os.getenv('OPENAI_API_BASE')).strip()
    if not api_key and provider == 'openai_compatible':
        logger.info("OPENAI_API_KEY not found; Phase 4 will keep structured placeholder sections")
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("langchain_openai not installed; Phase 4 will keep structured placeholder sections")
        return None

    model = llm_config.get('model', 'gpt-4o')
    temperature = float(llm_config.get('temperature', 0.0))
    max_retries = int(llm_config.get('max_retries', 2))
    try:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base if api_base else None,
            max_retries=max_retries,
        )
    except Exception as exc:
        logger.warning("Failed to initialize Phase 4 LLM client: %s", exc)
        return None


def _generate_section_content_with_llm(
    llm_client: Any,
    section_key: str,
    section_payload: Dict[str, Any],
    global_guidance: Dict[str, Any],
) -> str:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        raise RuntimeError("langchain_core is required for Phase 4 LLM generation") from exc

    prompt = build_section_generation_prompt(
        section_key=section_key,
        section_payload=section_payload,
        global_guidance=global_guidance,
    )
    response = llm_client.invoke(
        [
            SystemMessage(content=PHASE4_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    content = getattr(response, 'content', response)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    return str(content).strip()


def _materialize_seed_fallback_sections(report_context: Dict[str, Any], sections: Dict[str, Any]) -> int:
    materialized_count = 0
    for section_key, payload in list(sections.items()):
        section_payload = _safe_dict(payload)
        if section_payload.get('generation_policy') in {'locked_seed', SEEDED_LLM_POLICY}:
            section_payload['content'] = _repair_narrative_facts(
                report_context,
                _safe_str(section_payload.get('content')),
            )
            section_payload['status'] = 'deterministic'
            sections[section_key] = section_payload
            materialized_count += 1
    return materialized_count


def generate_llm_sections(
    report_context: Dict[str, Any],
    config_path: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Generate section prose with LLM when available, otherwise keep placeholders."""

    llm_sections = build_placeholder_llm_sections(report_context)
    sections = _safe_dict(llm_sections.get('sections'))
    if not use_llm:
        deterministic_count = _materialize_seed_fallback_sections(report_context, sections)
        llm_sections['sections'] = sections
        llm_sections['generation_mode'] = 'structured_placeholder'
        llm_sections['summary']['generated_section_count'] = deterministic_count
        return llm_sections

    resolved_config_path = config_path or report_context.get('run', {}).get('config_path', 'config.yaml')
    llm_client = _initialize_phase4_llm_client(str(resolved_config_path))
    if llm_client is None:
        deterministic_count = _materialize_seed_fallback_sections(report_context, sections)
        llm_sections['sections'] = sections
        llm_sections['generation_mode'] = 'structured_placeholder'
        llm_sections['summary']['generated_section_count'] = deterministic_count
        return llm_sections

    llm_sections['sections'] = sections
    global_guidance = _safe_dict(llm_sections.get('global_guidance'))
    generated_section_count = 0
    failed_sections: List[str] = []

    for section_key in DEFAULT_SECTION_KEYS:
        section_payload = _safe_dict(sections.get(section_key))
        if not section_payload:
            continue
        try:
            content = _generate_section_content_with_llm(
                llm_client=llm_client,
                section_key=section_key,
                section_payload=section_payload,
                global_guidance=global_guidance,
            )
            if content:
                section_payload['content'] = _repair_narrative_facts(report_context, content)
                section_payload['status'] = 'generated'
                generated_section_count += 1
            else:
                section_payload['status'] = 'llm_empty'
                failed_sections.append(section_key)
        except Exception as exc:
            seed_content = _repair_narrative_facts(report_context, _safe_str(section_payload.get('content')))
            if seed_content:
                section_payload['content'] = seed_content
                section_payload['status'] = 'llm_failed_seed_retained'
            else:
                section_payload['status'] = 'llm_failed'
            section_payload['generation_error'] = str(exc)
            failed_sections.append(section_key)
            logger.warning("Phase 4 LLM generation failed for section %s: %s", section_key, exc)
        sections[section_key] = section_payload

    llm_sections['sections'] = sections
    llm_sections['generation_mode'] = 'llm_generated' if generated_section_count == len(DEFAULT_SECTION_KEYS) else 'llm_generated_partial'
    llm_sections['summary']['generated_section_count'] = generated_section_count
    llm_sections['summary']['failed_sections'] = failed_sections
    llm_sections['global_guidance']['llm_attempted'] = True
    llm_sections['global_guidance']['llm_generation_succeeded'] = generated_section_count > 0
    return llm_sections


def write_report_context(report_context: Dict[str, Any], report_dir: str) -> str:
    return _write_json(report_context, Path(report_dir) / 'report_context.json')


def write_llm_sections(
    report_context: Dict[str, Any],
    report_dir: str,
    llm_sections: Optional[Dict[str, Any]] = None,
) -> str:
    payload = llm_sections or build_placeholder_llm_sections(report_context)
    return _write_json(payload, Path(report_dir) / 'llm_sections.json')
