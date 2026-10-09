"""
Phase 3: 综合临床评估报告生成器 (Final Report Generator)

该模块是 Phase 3 的正式主入口，负责：
1. 加载 Phase 0/1/2 的关键状态与配置
2. 评估基线面板与 Winner 面板得分
3. 生成报告所需图表与评分文件
4. 输出 Phase 3 结果摘要

根目录的 `generate_final_report.py` 仅保留为兼容 wrapper。
"""

import json
import os
import inspect
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ''):
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.agents.phase3.adaptive.render_config_builder import build_render_config
from src.agents.phase3.evaluator_bridge import ClinicalEvaluator, normalize_scores_for_radar
from src.agents.phase3.renderers.phase1_renderers import (
    load_phase1_stability_plot_data,
)
from src.agents.phase3.review.vlm_reviewer import VLMReviewer
from src.agents.phase3.router import FigureRouter
from src.agents.phase3.styles.journal_registry import get_journal_style
from src.tools.visualization import PLOT_FUNCTIONS
from src.tools.visualization.journal_theme import style_to_theme, write_theme_json
from src.utils.config_manager import ConfigManager
from src.utils.run_registry import archive_artifacts, record_stage_run
from src.utils.evaluation_audit import (
    build_artifact_manifest,
    build_audit_summary,
    describe_path,
    environment_snapshot,
    package_final_delivery,
    write_audit_json,
    write_execution_trace,
)

TASK_TO_PLOT_KEY = {
    'plot_stats_scatter': 'plot_stats_scatter',
    'plot_phase0_prior_evidence_atlas': 'plot_phase0_prior_evidence_atlas',
    'plot_stability_landscape': 'plot_stability_landscape',
    'plot_method_feature_heatmap': 'plot_method_feature_heatmap',
    'plot_phase1_final_panel_correlation_heatmap': 'plot_phase1_final_panel_correlation_heatmap',
    'plot_autogluon_roc': 'plot_autogluon_roc',
    'plot_phase1_selection_baseline_composite': 'plot_phase1_selection_baseline_composite',
    'plot_phase2_clinical_validation_composite': 'plot_phase2_clinical_validation_composite',
    'plot_phase2_radar_validation_composite': 'plot_phase2_radar_validation_composite',
    'plot_phase3_shap_interpretation_composite': 'plot_phase3_shap_interpretation_composite',
    'plot_radar_4d': 'plot_phase2_objective_shift_summary',
    'plot_final_roc': 'plot_final_roc',
    'plot_final_holdout_roc': 'plot_final_holdout_roc',
    'plot_final_holdout_dca': 'plot_final_holdout_dca',
    'plot_dca': 'plot_dca',
    'plot_calibration': 'plot_calibration',
    'plot_threshold_performance': 'plot_threshold_performance',
    'plot_incremental_value_summary': 'plot_phase2_incremental_value_summary',
    'plot_shap': 'plot_shap',
    'plot_rcs': 'plot_rcs_curves',
}

SINGLE_FIGURE_DIR = 'output/figures/single_panels'
COMPOSITE_FIGURE_DIR = 'output/figures/composite_panels'

TASK_OUTPUT_HINTS = {
    'plot_stats_scatter': [
        f'{SINGLE_FIGURE_DIR}/phase1_pca_scatter.pdf',
        f'{SINGLE_FIGURE_DIR}/phase1_plsda_scatter.pdf',
    ],
    'plot_phase0_prior_evidence_atlas': [f'{SINGLE_FIGURE_DIR}/phase0_prior_evidence_atlas.pdf'],
    'plot_stability_landscape': [f'{SINGLE_FIGURE_DIR}/phase1_stability_landscape.pdf'],
    'plot_method_feature_heatmap': [f'{SINGLE_FIGURE_DIR}/phase1_method_support_matrix.pdf'],
    'plot_phase1_final_panel_correlation_heatmap': [f'{SINGLE_FIGURE_DIR}/phase1_final_panel_correlation.pdf'],
    'plot_autogluon_roc': [f'{SINGLE_FIGURE_DIR}/phase1_autogluon_baseline_roc.pdf'],
    'plot_phase1_selection_baseline_composite': [f'{COMPOSITE_FIGURE_DIR}/phase1_selection_and_baseline_composite.pdf'],
    'plot_radar_4d': [f'{SINGLE_FIGURE_DIR}/phase2_objective_shift_summary.pdf'],
    'plot_phase2_clinical_validation_composite': [f'{COMPOSITE_FIGURE_DIR}/phase2_objective_shift_clinical_validation_composite.pdf'],
    'plot_phase2_radar_validation_composite': [f'{COMPOSITE_FIGURE_DIR}/phase2_radar_clinical_validation_composite.pdf'],
    'plot_phase3_shap_interpretation_composite': [f'{COMPOSITE_FIGURE_DIR}/phase2_shap_interpretation_composite.pdf'],
    'plot_final_roc': [f'{SINGLE_FIGURE_DIR}/phase2_winner_roc.pdf'],
    'plot_final_holdout_roc': [f'{SINGLE_FIGURE_DIR}/phase2_winner_holdout_roc.pdf'],
    'plot_final_holdout_dca': [f'{SINGLE_FIGURE_DIR}/phase2_winner_holdout_decision_curve.pdf'],
    'plot_dca': [f'{SINGLE_FIGURE_DIR}/phase2_winner_decision_curve.pdf'],
    'plot_calibration': [f'{SINGLE_FIGURE_DIR}/phase2_winner_calibration.pdf'],
    'plot_threshold_performance': [f'{SINGLE_FIGURE_DIR}/phase2_winner_threshold_performance.pdf'],
    'plot_incremental_value_summary': [f'{SINGLE_FIGURE_DIR}/phase2_incremental_value_summary.pdf'],
    'plot_shap': [
        f'{SINGLE_FIGURE_DIR}/phase2_winner_shap_summary.pdf',
        f'{SINGLE_FIGURE_DIR}/phase2_winner_shap_summary.json',
        f'{SINGLE_FIGURE_DIR}/phase2_winner_shap_summary_dependence_panels.pdf',
    ],
    'plot_rcs': [
        f'{SINGLE_FIGURE_DIR}/phase2_winner_rcs_curves.pdf',
        f'{SINGLE_FIGURE_DIR}/phase2_winner_rcs_curves.json',
    ],
}

TASK_MANIFEST_SPECS = {
    'plot_stats_scatter': [
        {
            'figure_id': 'fig1a',
            'title': 'PCA Scatter Plot',
            'section': 'Methodology & Workflow',
            'output_index': 0,
            'caption_seed': 'PCA-based visualization of sample separation after Phase 1 feature engineering.',
        },
        {
            'figure_id': 'fig1b',
            'title': 'PLS-DA Scatter Plot',
            'section': 'Methodology & Workflow',
            'output_index': 1,
            'caption_seed': 'PLS-DA-based visualization of sample separation after Phase 1 feature engineering.',
        },
    ],
    'plot_phase0_prior_evidence_atlas': [
        {
            'figure_id': 'fig1c',
            'title': 'Prior Evidence Atlas',
            'section': 'Methodology & Workflow',
            'output_index': 0,
            'caption_seed': 'Ranked atlas of Phase 0 adaptive prior scores, with retained versus excluded metabolites and aligned evidence tracks for mechanistic plausibility, disease specificity, clinical evidence, and consistency bonus.',
        },
    ],
    'plot_stability_landscape': [
        {
            'figure_id': 'fig1d',
            'title': 'Phase 1 Stability Landscape',
            'section': 'Methodology & Workflow',
            'output_index': 0,
            'caption_seed': 'Selection-frequency versus method-consensus landscape of Phase 1 candidate features, highlighting the stable core and final panel.',
        },
    ],
    'plot_method_feature_heatmap': [
        {
            'figure_id': 'fig1e',
            'title': 'Method Support Dot Matrix',
            'section': 'Methodology & Workflow',
            'output_index': 0,
            'caption_seed': 'Dot-matrix summary of per-method selection frequencies for final-panel features retained by the Phase 1 stability-selection workflow.',
        },
    ],
    'plot_phase1_final_panel_correlation_heatmap': [
        {
            'figure_id': 'fig1g',
            'title': 'Final Panel Correlation Heatmap',
            'section': 'Methodology & Workflow',
            'output_index': 0,
            'caption_seed': 'Clustered Pearson-correlation heatmap of the final Phase 1 feature panel, highlighting internal redundancy structure and correlation blocks among retained metabolites.',
        },
    ],
    'plot_autogluon_roc': [
        {
            'figure_id': 'fig2a',
            'title': 'AutoGluon Multi-Model ROC Comparison',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'ROC comparison of baseline AutoGluon candidate models.',
        },
    ],
    'plot_phase1_selection_baseline_composite': [
        {
            'figure_id': 'fig2b',
            'title': 'Stable Panel Selection and Baseline Model Performance',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Composite overview linking Phase 1 feature stability, cross-method support, panel correlation structure, and baseline AutoGluon ROC performance.',
        },
    ],
    'plot_radar_4d': [
        {
            'figure_id': 'fig3',
            'title': 'Phase 2 Objective Shift and Winner Feature Contribution',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Integrated Phase 2 summary comparing the Phase 1 baseline against the selected winner across predictive performance, biological relevance, cost efficiency, and redundancy control, together with leave-one-feature-out importance and mechanistic evidence tracks for retained winner features.',
        },
    ],
    'plot_phase2_clinical_validation_composite': [
        {
            'figure_id': 'fig4h',
            'title': 'Phase 2 Decision and Clinical Validation Composite',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Composite overview linking Phase 2 objective-shift evidence with final ROC, decision-curve, and calibration performance of the winner panel.',
        },
    ],
    'plot_phase2_radar_validation_composite': [
        {
            'figure_id': 'fig4h_alt',
            'title': 'Phase 2 Radar and Clinical Validation Composite',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Composite overview linking the whole-panel 4D radar comparison with final ROC, decision-curve, and calibration performance of the winner panel.',
        },
    ],
    'plot_phase3_shap_interpretation_composite': [
        {
            'figure_id': 'fig4i',
            'title': 'SHAP Summary and Dependence Composite',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Composite overview linking global SHAP feature importance with per-feature dependence patterns of the final winner panel.',
        },
    ],
    'plot_final_roc': [
        {
            'figure_id': 'fig4a',
            'title': 'Final ROC Curve',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'ROC curve of the final Phase 2 winner panel.',
        },
    ],
    'plot_final_holdout_roc': [
        {
            'figure_id': 'fig4a_holdout',
            'title': 'Final Holdout ROC Curve',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Internal holdout ROC curve of the final Phase 2 winner panel. This held-out subset is development evidence, not independent external validation.',
        },
    ],
    'plot_final_holdout_dca': [
        {
            'figure_id': 'fig4b_holdout',
            'title': 'Internal Holdout Decision Curve Analysis',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Decision curve analysis computed from the fixed internal holdout probabilities of the final Phase 2 winner panel. This held-out subset is development evidence, not independent external validation.',
        },
    ],
    'plot_dca': [
        {
            'figure_id': 'fig4b',
            'title': 'Decision Curve Analysis',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Clinical utility of the final model evaluated by decision curve analysis.',
        },
    ],
    'plot_calibration': [
        {
            'figure_id': 'fig4e',
            'title': 'Calibration Plot',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Internal probability calibration assessment of the final winner panel.',
        },
    ],
    'plot_threshold_performance': [
        {
            'figure_id': 'fig4f',
            'title': 'Threshold Performance Plot',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Sensitivity, specificity, PPV, and NPV across prespecified decision thresholds.',
        },
    ],
    'plot_incremental_value_summary': [
        {
            'figure_id': 'fig4j',
            'title': 'Incremental Value of the Phase 2 Panel',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Incremental reclassification and discrimination gains of the Phase 2 panel relative to the Phase 1 panel, shown with empirical bootstrap confidence intervals and p values for continuous NRI and IDI.',
        },
    ],
    'plot_shap': [
        {
            'figure_id': 'fig4c',
            'title': 'SHAP Summary Plot',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Global SHAP summary plot for the final winner panel.',
            'auxiliary_output_indices': [2],
            'metadata_output_index': 1,
        },
    ],
    'plot_rcs': [
        {
            'figure_id': 'fig4d',
            'title': 'Restricted Cubic Spline Curves',
            'section': 'Results',
            'output_index': 0,
            'caption_seed': 'Restricted cubic spline analysis of continuous exposure-response patterns for winner features.',
            'metadata_output_index': 1,
        },
    ],
}

SMART_RENDERERS = {}


def _call_renderer_compat(renderer, **kwargs):
    """Call renderers while tolerating legacy/minimal renderer signatures."""
    try:
        signature = inspect.signature(renderer)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    except (TypeError, ValueError):
        pass
    return renderer(**kwargs)


def _find_latest_phase2_result(
    results_dir: str = 'output/runs',
    pattern_prefix: str = 'phase2_result_',
    pattern_suffix: str = '.json',
) -> str:
    candidates = []
    if not os.path.isdir(results_dir):
        return ''

    for name in os.listdir(results_dir):
        if name.startswith(pattern_prefix) and name.endswith(pattern_suffix):
            full_path = os.path.join(results_dir, name)
            try:
                candidates.append((os.path.getmtime(full_path), full_path))
            except OSError:
                continue

    if not candidates:
        return ''

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _get_canonical_phase0_output_path() -> str:
    """Resolve the current run's Phase 0 output without falling back to shared latest state."""
    runtime_root = str(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or '').strip()
    candidates = []
    if runtime_root:
        candidates.extend([
            os.path.join(runtime_root, 'phase0', 'phase0_output_latest.json'),
            os.path.join(runtime_root, 'output', 'phase0', 'phase0_output_latest.json'),
        ])
    candidates.extend([
        'output/phase0/phase0_output_latest.json',
        'output/phase0/outputs/phase0_output_latest.json',
    ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ''


def _get_canonical_phase2_winner_path() -> str:
    runtime_root = str(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or '').strip()
    candidates = []
    if runtime_root:
        candidates.extend([
            os.path.join(runtime_root, 'phase1', 'artifacts', 'phase2_winner_scores.json'),
            os.path.join(runtime_root, 'phase1', 'legacy', 'artifacts', 'phase2_winner_scores.json'),
            os.path.join(runtime_root, 'output', 'phase1', 'artifacts', 'phase2_winner_scores.json'),
            os.path.join(runtime_root, 'output', 'artifacts', 'phase2_winner_scores.json'),
        ])
    candidates.extend([
        'output/phase1/artifacts/phase2_winner_scores.json',
        'output/artifacts/phase2_winner_scores.json',
    ])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ''


def _extract_run_id_from_phase2_path(phase2_path: str) -> str:
    if not phase2_path:
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.basename(phase2_path)
    stem = os.path.splitext(base_name)[0]
    for marker in ('phase2_correct_architecture_test_', 'phase2_', 'test_'):
        if marker in stem:
            tail = stem.split(marker, 1)[-1]
            if tail:
                return tail
    return stem or datetime.now().strftime('%Y%m%d_%H%M%S')


def _normalize_manifest_path(path: str) -> str:
    if not path:
        return ''
    normalized = os.path.normpath(str(path))
    if os.path.isabs(normalized):
        try:
            return os.path.relpath(normalized, os.getcwd())
        except ValueError:
            return normalized
    return normalized


def _materialize_run_figure(source_path: str, run_figure_dir: Path) -> str:
    """Copy a generated figure/metadata asset into the current run directory."""
    source = Path(source_path)
    if not source.exists() or not source.is_file():
        return _normalize_manifest_path(source_path)
    run_figure_dir.mkdir(parents=True, exist_ok=True)
    destination = run_figure_dir / source.name
    try:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    except OSError:
        return _normalize_manifest_path(source_path)
    return _normalize_manifest_path(str(destination))


def _infer_output_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == '.pdf':
        return 'pdf'
    if suffix in {'.png', '.jpg', '.jpeg', '.svg', '.webp'}:
        return 'image'
    if suffix == '.json':
        return 'json'
    if suffix in {'.md', '.markdown'}:
        return 'markdown'
    if suffix in {'.html', '.htm'}:
        return 'html'
    return suffix.lstrip('.') or 'unknown'


def _build_output_descriptor(path: str, role: str) -> Dict[str, Any]:
    normalized_path = _normalize_manifest_path(path)
    return {
        'role': role,
        'path': normalized_path,
        'exists': bool(normalized_path) and os.path.exists(normalized_path),
        'kind': _infer_output_kind(normalized_path) if normalized_path else 'unknown',
    }


def _get_canonical_phase1_engineered_data_path() -> str:
    candidates = [
        'output/phase1/intermediate/latest/engineered/data_with_engineered_features.csv',
        'data/data_with_engineered_features.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def _load_latest_phase1_method_feature_sets() -> Dict[str, List[str]]:
    """Load latest real Phase 1 method-level feature sets for UpSet plotting."""
    display_name_map = {
        'run_elasticnet_selector': 'Elastic Net Logistic',
        'elastic net logistic': 'Elastic Net Logistic',
        'elasticnet': 'Elastic Net Logistic',
        'elastic_net_logistic': 'Elastic Net Logistic',
        'run_lasso_selector': 'L1 Logistic/Lasso',
        'lasso': 'L1 Logistic/Lasso',
        'l1_logistic_lasso': 'L1 Logistic/Lasso',
        'run_random_forest_selector': 'Random Forest',
        'random_forest': 'Random Forest',
        'rf': 'Random Forest',
        'run_lightgbm_selector': 'LightGBM',
        'lightgbm': 'LightGBM',
        'run_mrmr_selector': 'mRMR',
        'mrmr': 'mRMR',
        'run_t_test_selector': 'Welch t-test',
        't_test': 'Welch t-test',
        'welch_t_test': 'Welch t-test',
        'welch t-test': 'Welch t-test',
        'run_fdr_effect_size_selector': 'FDR/effect-size',
        'fdr_effect_size': 'FDR/effect-size',
    }
    canonical_order = ['Elastic Net Logistic', 'L1 Logistic/Lasso', 'Random Forest', 'LightGBM', 'mRMR', 'Welch t-test', 'FDR/effect-size']
    candidates = [
        'output/phase1/intermediate/latest/feature_selection/method_results.json',
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        normalized: Dict[str, List[str]] = {}
        for method_name, features in data.items():
            if isinstance(features, list) and all(isinstance(item, str) for item in features):
                display_name = display_name_map.get(str(method_name).strip().lower())
                if display_name:
                    normalized[display_name] = features
        normalized = {
            label: normalized[label]
            for label in canonical_order
            if label in normalized
        }
        if len(normalized) > 1:
            return normalized
    return {}


def _load_json_if_exists(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_phase0_output_from_any_source(source_path: str) -> Dict[str, Any]:
    payload = _load_json_if_exists(source_path)
    if payload.get('confirmed_biomarkers') or payload.get('final_priors'):
        return payload

    explicit_phase0_path = str(payload.get('phase0_output_path', '') or '').strip()
    if explicit_phase0_path and os.path.exists(explicit_phase0_path):
        nested = _load_json_if_exists(explicit_phase0_path)
        if nested:
            return nested

    phase0_output = payload.get('phase0_output', {})
    return phase0_output if isinstance(phase0_output, dict) else {}


def load_phase0_features(phase0_or_phase2_json_path: str) -> List[str]:
    """从 Phase 0 / Phase 2 输出中提取可展示的生物标志物名称列表。"""
    phase0_output = _load_phase0_output_from_any_source(phase0_or_phase2_json_path)

    confirmed_biomarkers = phase0_output.get('confirmed_biomarkers', []) or []
    normalized_biomarkers: List[str] = []
    for biomarker in confirmed_biomarkers:
        if isinstance(biomarker, dict):
            biomarker_name = str(biomarker.get('name', '') or biomarker.get('id', '') or '').strip()
            if biomarker_name:
                normalized_biomarkers.append(biomarker_name)
            continue
        biomarker_text = str(biomarker or '').strip()
        if biomarker_text:
            normalized_biomarkers.append(biomarker_text)
    if normalized_biomarkers:
        return normalized_biomarkers

    for prior_name in phase0_output.get('final_priors', []) or []:
        prior_text = str(prior_name or '').strip()
        if prior_text:
            normalized_biomarkers.append(prior_text)
    return normalized_biomarkers


def load_phase1_features(csv_path: str) -> List[str]:
    """从 Phase 1 输出 CSV 中提取特征列表。"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    protected_columns = ['Sample_ID', 'Group', 'Batch']
    return [col for col in df.columns if col not in protected_columns]


def load_phase2_winner(json_path: str) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
    """从 Phase 2 输出 JSON 中提取 Winner 特征和四维得分。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_result = data.get('final_result', {})
    phase0_output = _load_phase0_output_from_any_source(json_path)
    if final_result:
        features = final_result.get('features', [])
        comprehensive_metrics = (
            final_result.get('comprehensive_metrics')
            if isinstance(final_result.get('comprehensive_metrics'), dict)
            else {}
        )
        display_roc_auc = comprehensive_metrics.get('roc_auc')
        if display_roc_auc is None:
            display_roc_auc = final_result.get('roc_auc')
        if display_roc_auc is None:
            display_roc_auc = final_result.get('perf', 0.0)
        scores = {
            'f_perf': display_roc_auc,
            'f_bio': final_result.get('bio', 0.0),
            'f_cost': final_result.get('cost', 0.0),
            'f_corr': final_result.get('corr', 0.0),
            'roc_auc': display_roc_auc,
            'search_mean_cv_score': final_result.get('perf', 0.0),
            'brier_score': comprehensive_metrics.get('brier_score', 0.0),
            'auprc': comprehensive_metrics.get('auprc', 0.0),
            'f1_score': comprehensive_metrics.get('f1_score', 0.0),
            'sensitivity': comprehensive_metrics.get('sensitivity', 0.0),
            'specificity': comprehensive_metrics.get('specificity', 0.0),
        }
    else:
        features = data.get('selected_features', [])
        scores = dict(data.get('scores', {}) or {})
        comprehensive_metrics = (
            data.get('comprehensive_metrics')
            if isinstance(data.get('comprehensive_metrics'), dict)
            else {}
        )
        display_roc_auc = comprehensive_metrics.get('roc_auc', scores.get('roc_auc', scores.get('f_perf', 0.0)))
        scores['f_perf'] = display_roc_auc
        scores.setdefault('roc_auc', display_roc_auc)
        scores.setdefault('f_bio', data.get('bio', 0.0))
        scores.setdefault('f_cost', data.get('cost', 0.0))
        scores.setdefault('f_corr', data.get('corr', 0.0))
        scores.setdefault('search_mean_cv_score', data.get('perf', scores.get('search_mean_cv_score', display_roc_auc)))
        scores.setdefault('brier_score', comprehensive_metrics.get('brier_score', 0.0))
        scores.setdefault('auprc', comprehensive_metrics.get('auprc', 0.0))
        scores.setdefault('f1_score', comprehensive_metrics.get('f1_score', 0.0))
        scores.setdefault('sensitivity', comprehensive_metrics.get('sensitivity', 0.0))
        scores.setdefault('specificity', comprehensive_metrics.get('specificity', 0.0))
    return features, scores, phase0_output


def build_priors_dict(phase0_biomarkers: List[Any]) -> Dict[str, float]:
    """构建 Phase 0 先验字典，兼容字符串和 confirmed_biomarker 记录。"""
    priors: Dict[str, float] = {}
    for biomarker in phase0_biomarkers or []:
        if isinstance(biomarker, dict):
            biomarker_id = str(biomarker.get('id', '') or '').strip().upper()
            biomarker_name = str(biomarker.get('name', '') or '').strip()
            prior_value = biomarker.get('bio_prior_norm')
            if prior_value is None:
                prior_value = biomarker.get('confidence_score')
            if prior_value is None:
                prior_value = biomarker.get('bio_prior_raw')
            try:
                prior_float = float(prior_value) if prior_value is not None else 1.0
            except (TypeError, ValueError):
                prior_float = 1.0

            if biomarker_id:
                priors[biomarker_id] = prior_float
            elif biomarker_name:
                priors[biomarker_name] = prior_float
            continue

        biomarker_text = str(biomarker or '').strip()
        if biomarker_text:
            priors[biomarker_text] = 1.0
    return priors


class Phase3Pipeline:
    """Phase 3 正式主控流水线。"""

    def __init__(
        self,
        config_path: str = 'config.yaml',
        use_mock_vlm: bool = True,
        max_acval_retries: int = 3,
    ):
        self.config_path = config_path
        self.use_mock_vlm = use_mock_vlm
        self.max_acval_retries = max_acval_retries

        self.config: ConfigManager | None = None
        self.router = None
        self.auditor = None

        self.phase0_state: Dict[str, Any] = {}
        self.phase1_state: Dict[str, Any] = {}
        self.phase2_state: Dict[str, Any] = {}
        self.required_figures: List[str] = []
        self.loaded_state_paths: Dict[str, str] = {}
        self.generated_figures: Dict[str, Dict[str, Any]] = {}

    def run(self) -> str:
        print("\n" + "=" * 80)
        print("PHASE 3: COMPREHENSIVE CLINICAL EVALUATION REPORT GENERATION")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        self._load_config_and_states()
        required_figures = self._determine_required_figures()
        self._initialize_acval_engine()
        self._generate_all_figures(required_figures)
        report_context = self._generate_report_artifacts()
        report_path = self._generate_final_report(report_context)

        print("\n" + "=" * 80)
        print("PHASE 3 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Report Path: {report_path}")
        print("=" * 80 + "\n")
        return report_path

    def _load_config_and_states(self):
        print("\n" + "=" * 80)
        print("[Step 1] Loading Configuration and States")
        print("=" * 80)

        print(f"\n[1.1] Loading config from: {self.config_path}")
        self.config = ConfigManager(self.config_path)
        print("✅ Config loaded successfully")

        phase0_state_path = (
            os.environ.get('METABOAGENT_PHASE0_RESULT_PATH', '').strip()
            or os.environ.get('METABOAGENT_PHASE0_STATE_PATH', '').strip()
            or 'output/phase0/phase0_state.json'
        )
        phase1_state_path = (
            os.environ.get('METABOAGENT_PHASE1_RESULT_PATH', '').strip()
            or os.environ.get('METABOAGENT_PHASE1_STATE_PATH', '').strip()
            or 'output/phase1/phase1_state.json'
        )
        phase2_state_path = (
            os.environ.get('METABOAGENT_PHASE2_RESULT_PATH', '').strip()
            or _get_canonical_phase2_winner_path()
            or _find_latest_phase2_result()
        )
        if not phase2_state_path:
            phase2_state_path = 'output/phase1/artifacts/phase2_winner_scores.json'

        self.loaded_state_paths = {
            'phase0': phase0_state_path,
            'phase1': phase1_state_path,
            'phase2': phase2_state_path,
        }

        self.phase0_state = self._load_state_file(phase0_state_path)
        self.phase1_state = self._load_state_file(phase1_state_path)
        self.phase2_state = self._load_state_file(phase2_state_path)

    def _load_state_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            print(f"  ⏭️  State file not found: {file_path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        print(f"  ✅ State loaded: {file_path}")
        return state

    def _determine_required_figures(self) -> List[str]:
        print("\n" + "=" * 80)
        print("[Step 2] Determining Required Figures")
        print("=" * 80)

        self.router = FigureRouter(
            config_manager=self.config,
            phase0_state=self.phase0_state,
            phase1_state=self.phase1_state,
            phase2_state=self.phase2_state,
        )
        self.required_figures = self.router.determine_required_figures()
        # The router historically looked only at ``phase0_state``.  In
        # run-scoped executions that state file can be absent even though the
        # canonical Phase 0 output is available under the runtime root (or is
        # discoverable through the configured Phase 0 path).  Make the atlas
        # requirement data-driven so a valid Phase 0 result cannot silently
        # lose its figure.
        phase0_candidates: List[Path] = []
        runtime_root = str(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or '').strip()
        if runtime_root:
            phase0_candidates.extend([
                Path(runtime_root) / 'phase0' / 'phase0_output_latest.json',
                Path(runtime_root) / 'phase0' / 'outputs' / 'phase0_output_latest.json',
                Path(runtime_root) / 'output' / 'phase0' / 'phase0_output_latest.json',
            ])
        loaded_phase0 = str(self.loaded_state_paths.get('phase0') or '').strip()
        if loaded_phase0:
            phase0_candidates.append(Path(loaded_phase0))
        if self.config is not None and hasattr(self.config, 'get_phase0_path'):
            try:
                configured_phase0 = str(self.config.get_phase0_path('phase0_output') or '').strip()
            except TypeError:
                configured_phase0 = str(self.config.get_phase0_path() or '').strip()
            if configured_phase0:
                phase0_candidates.append(Path(configured_phase0))
        phase0_available = False
        for candidate in phase0_candidates:
            try:
                if not candidate.exists():
                    continue
                payload = _load_json_if_exists(str(candidate))
                if payload and any(
                    key in payload
                    for key in (
                        'candidate_scores_summary',
                        'confirmed_biomarkers',
                        'final_priors',
                        'phase0_prior_count',
                    )
                ):
                    phase0_available = True
                    break
            except OSError:
                continue
        if not phase0_available and self.phase0_state:
            phase0_available = any(
                key in self.phase0_state
                for key in (
                    'candidate_scores_summary',
                    'confirmed_biomarkers',
                    'final_priors',
                    'phase0_prior_count',
                )
            )
        if phase0_available and 'plot_phase0_prior_evidence_atlas' not in self.required_figures:
            self.required_figures.insert(0, 'plot_phase0_prior_evidence_atlas')
        return self.required_figures

    def _initialize_acval_engine(self):
        print("\n" + "=" * 80)
        print("[Step 3] Initializing ACVAL Engine")
        print("=" * 80)
        self.auditor = VLMReviewer(use_mock=self.use_mock_vlm)
        print(f"✅ ACVAL Engine initialized ({'Mock' if self.use_mock_vlm else 'Real'} VLM)")

    def _generate_all_figures(self, required_figures: List[str]):
        print("\n" + "=" * 80)
        print("[Step 4] Executing Figure Tasks")
        print("=" * 80)

        runtime_paths = self._resolve_runtime_paths()
        report_inputs = self._load_report_inputs(runtime_paths)
        self.generated_figures = {}
        # Composite panels may internally reuse legacy single-panel assets.
        # Render the canonical P0 panels last so no composite can overwrite
        # their run-scoped ROC/calibration/DCA/threshold evidence.
        p0_tasks = {
            'plot_final_roc',
            'plot_final_holdout_roc',
            'plot_final_holdout_dca',
            'plot_calibration',
            'plot_dca',
            'plot_threshold_performance',
        }
        composite_tasks = {
            'plot_phase2_clinical_validation_composite',
            'plot_phase2_radar_validation_composite',
        }
        ordered_tasks = [
            task for task in required_figures
            if task not in p0_tasks and task not in composite_tasks
        ]
        ordered_tasks.extend(task for task in required_figures if task in p0_tasks)
        # Phase 2 composites consume the canonical P0 single-panel outputs
        # above; render them only after those source panels exist.
        ordered_tasks.extend(task for task in required_figures if task in composite_tasks)
        for task in ordered_tasks:
            result = self._execute_figure_task(task, runtime_paths, report_inputs)
            self.generated_figures[task] = result
            print(f"  • {task} -> {result['renderer_name'] or 'N/A'} [{result['status']}]")

    def _resolve_visualization_data_path(self, runtime_paths: Dict[str, str], task: str) -> str:
        phase2_data_path = self.phase2_state.get('data_path')
        # Winner-centric plots must use the exact Phase 2 winner dataset.
        if task in {'plot_final_roc', 'plot_final_holdout_roc', 'plot_final_holdout_dca', 'plot_shap', 'plot_rcs'}:
            if phase2_data_path and os.path.exists(phase2_data_path):
                # Phase 2 may retain the original upload path even when its
                # winner features are the engineered columns materialized by
                # Phase 1.  Do not pass a matrix that cannot contain the
                # selected feature IDs to SHAP/RCS renderers.
                try:
                    import pandas as pd
                    phase2_features = (
                        (self.phase2_state.get('final_result') or {}).get('features')
                        or self.phase2_state.get('selected_features')
                        or []
                    )
                    source_columns = set(pd.read_csv(phase2_data_path, nrows=0).columns)
                    if all(feature in source_columns for feature in phase2_features):
                        return phase2_data_path
                except Exception:
                    pass
            return runtime_paths['phase1_csv']

        # Phase 1/full-dataset plots should use latest engineered feature matrix.
        if task in {'plot_stats_scatter', 'plot_autogluon_roc', 'plot_dca', 'plot_threshold_performance'}:
            return runtime_paths['engineered_data_path']

        return runtime_paths['engineered_data_path']

    def _extract_phase2_metadata(self, runtime_paths: Dict[str, str]) -> Dict[str, Any]:
        final_result = self.phase2_state.get('final_result', {})
        selected_model = (
            final_result.get('selected_model')
            or self.phase2_state.get('selected_model')
            or self.phase2_state.get('champion_model_family')
            or 'RandomForest'
        )
        return {
            'features': final_result.get('features', []) or self.phase2_state.get('selected_features', []),
            'selected_model': selected_model,
            'phase2_result_path': self.loaded_state_paths.get('phase2') or runtime_paths['phase2_json'],
        }

    def _build_upset_feature_sets(self) -> Optional[Dict[str, List[str]]]:
        candidates = [
            self.phase1_state.get('method_feature_sets'),
            self.phase1_state.get('feature_sets'),
            self.phase1_state.get('selection_results'),
        ]

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            normalized: Dict[str, List[str]] = {}
            for method_name, features in candidate.items():
                if isinstance(features, list) and all(isinstance(item, str) for item in features):
                    normalized[str(method_name)] = features
            if len(normalized) > 1:
                return normalized
        latest_feature_sets = _load_latest_phase1_method_feature_sets()
        return latest_feature_sets or None

    def _build_radar_profiles(
        self,
        runtime_paths: Dict[str, str],
        report_inputs: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        evaluator = self._build_evaluator(runtime_paths, report_inputs)
        raw_scores = self._collect_raw_scores(
            runtime_paths,
            evaluator,
            report_inputs,
            include_phase0_baseline=False,
        )
        normalized_scores, _ = self._normalize_scores(raw_scores)
        return normalized_scores

    @staticmethod
    def _unwrap_renderer_output(result: Any) -> str:
        if isinstance(result, tuple):
            return str(result[0])
        return str(result)

    @staticmethod
    def _extract_renderer_metadata(result: Any) -> Dict[str, Any]:
        if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict):
            return dict(result[1])
        return {}

    def _get_phase3_visual_optimization_config(self) -> Dict[str, Any]:
        if self.config is None or not hasattr(self.config, 'config'):
            return {}
        phase3_cfg = self.config.config.get('phase3', {}) or {}
        return dict(phase3_cfg.get('visualization', {}) or {})

    def _get_journal_style_key(self) -> str:
        visual_cfg = self._get_phase3_visual_optimization_config()
        return str(
            visual_cfg.get('journal_style')
            or visual_cfg.get('style_preset')
            or 'nature'
        )

    def _get_journal_style(self):
        return get_journal_style(self._get_journal_style_key())

    def _get_journal_theme(self, task: str | None = None) -> Dict[str, Any]:
        visual_cfg = self._get_phase3_visual_optimization_config()
        theme = style_to_theme(self._get_journal_style(), task_name=task)
        font_family = str(visual_cfg.get('font_family') or '').strip()
        if font_family:
            theme['font_family'] = font_family
            theme = style_to_theme(theme, task_name=task)
        return theme

    def _write_task_theme_json(self, task: str, save_path: str) -> str:
        theme_path = Path(save_path).with_suffix('').with_name(f"{Path(save_path).stem}_theme.json")
        return write_theme_json(self._get_journal_theme(task), theme_path, task_name=task)

    def _build_smart_plot_data(self, task: str, runtime_paths: Dict[str, str]) -> Dict[str, Any]:
        if task in {'plot_stability_landscape', 'plot_method_feature_heatmap'}:
            return load_phase1_stability_plot_data(
                stability_scores_path=runtime_paths['phase1_stability_scores_path'],
                stability_summary_path=runtime_paths['phase1_stability_summary_path'],
                feature_provenance_path=runtime_paths.get('phase1_feature_provenance_path', ''),
            )
        raise ValueError(f'No smart plot data builder implemented for {task}')

    def _build_render_config_for_task(self, task: str, plot_data: Dict[str, Any]):
        visual_cfg = self._get_phase3_visual_optimization_config()
        style_preset = self._get_journal_style_key()
        render_policy = dict(visual_cfg.get('render_policy', {}) or {})
        heatmap_policy = dict(visual_cfg.get('heatmap_policy', {}) or {})
        landscape_policy = dict(visual_cfg.get('landscape_policy', {}) or {})

        policy = dict(render_policy)
        if task == 'plot_method_feature_heatmap':
            policy.update(heatmap_policy)
            figure_type = 'method_feature_heatmap'
        elif task == 'plot_stability_landscape':
            policy.update(landscape_policy)
            figure_type = 'stability_landscape'
        else:
            figure_type = task

        return build_render_config(
            journal=style_preset,
            figure_type=figure_type,
            data_summary=plot_data.get('data_summary', {}),
            policy=policy,
        )

    def _should_trigger_vlm_review(self, task: str, render_result) -> bool:
        visual_cfg = self._get_phase3_visual_optimization_config()
        review_policy = dict(visual_cfg.get('review_policy', {}) or {})
        if not bool(review_policy.get('enable_vlm_review', True)):
            return False
        if render_result.self_check_passed:
            return False
        high_risk_figures = set(review_policy.get('high_risk_figures', []) or [])
        return task in high_risk_figures

    def _execute_figure_task(
        self,
        task: str,
        runtime_paths: Dict[str, str],
        report_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        plot_key = TASK_TO_PLOT_KEY.get(task, task)
        renderer = PLOT_FUNCTIONS.get(plot_key)
        output_hints = list(TASK_OUTPUT_HINTS.get(task, []))
        runtime_root = Path(
            str(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or Path.cwd())
        ).expanduser().resolve()
        expected_outputs = [
            str(path if path.is_absolute() else runtime_root / path)
            for path in (Path(item) for item in output_hints)
        ]
        single_figure_dir = runtime_root / SINGLE_FIGURE_DIR
        record: Dict[str, Any] = {
            'task': task,
            'plot_key': plot_key,
            'renderer_name': getattr(renderer, '__name__', ''),
            'renderer_available': renderer is not None,
            'audit_enabled': self.auditor is not None,
            'expected_outputs': output_hints,
            'resolved_outputs': expected_outputs,
            'actual_outputs': [],
            'renderer_metadata': {},
            'status': 'planned' if renderer is not None else 'missing_renderer',
        }

        smart_renderer = SMART_RENDERERS.get(task)

        if renderer is None and smart_renderer is None:
            return record

        viz_data_path = self._resolve_visualization_data_path(runtime_paths, task)
        phase2_meta = self._extract_phase2_metadata(runtime_paths)
        journal_theme = self._get_journal_theme(task)
        theme_json_path = self._write_task_theme_json(task, expected_outputs[0]) if expected_outputs else ''
        record['journal_style'] = journal_theme.get('journal_style')
        record['theme_json_path'] = theme_json_path

        try:
            if smart_renderer is not None:
                plot_data = self._build_smart_plot_data(task, runtime_paths)
                render_config = self._build_render_config_for_task(task, plot_data)
                render_result = smart_renderer.render(plot_data, render_config, expected_outputs[0])
                record['actual_outputs'] = [render_result.save_path]
                record['renderer_name'] = smart_renderer.__class__.__name__
                record['renderer_metadata'] = {
                    'self_check_passed': render_result.self_check_passed,
                    'self_check_issues': render_result.self_check_issues,
                    'retry_count': render_result.retry_count,
                    'render_config': {
                        'figure_width': render_result.render_config.figure_width,
                        'figure_height': render_result.render_config.figure_height,
                        'label_strategy': render_result.render_config.label_strategy,
                        'annotate_top_n': render_result.render_config.annotate_top_n,
                        'focused_mode': render_result.render_config.focused_mode,
                        'focused_rejected_n': render_result.render_config.focused_rejected_n,
                        'style_preset': render_result.render_config.style.key,
                    },
                }

                if self.auditor is not None and self._should_trigger_vlm_review(task, render_result):
                    review_result = self.auditor.audit(
                        image_path=render_result.save_path,
                        scope='single',
                        figure_type=task,
                        metadata=render_result.metadata,
                    )
                    record['renderer_metadata']['vlm_review'] = review_result
                    if review_result.get('status') == 'FAIL' and review_result.get('suggested_actions'):
                        patched_config = render_result.render_config.patch(**dict(review_result.get('suggested_actions') or {}))
                        rerender_result = smart_renderer.render(plot_data, patched_config, expected_outputs[0])
                        record['actual_outputs'] = [rerender_result.save_path]
                        record['renderer_metadata']['self_check_passed'] = rerender_result.self_check_passed
                        record['renderer_metadata']['self_check_issues'] = rerender_result.self_check_issues
                        record['renderer_metadata']['retry_count'] = rerender_result.retry_count
                        record['renderer_metadata']['render_config'].update({
                            'figure_width': rerender_result.render_config.figure_width,
                            'figure_height': rerender_result.render_config.figure_height,
                            'label_strategy': rerender_result.render_config.label_strategy,
                            'annotate_top_n': rerender_result.render_config.annotate_top_n,
                            'focused_mode': rerender_result.render_config.focused_mode,
                            'focused_rejected_n': rerender_result.render_config.focused_rejected_n,
                            'style_preset': rerender_result.render_config.style.key,
                        })

                record['status'] = 'completed'
                return record

            if task == 'plot_stats_scatter':
                actual_outputs = []
                for method, save_path in (
                    ('pca', expected_outputs[0]),
                    ('plsda', expected_outputs[1]),
                ):
                    result = renderer(
                        data_path=viz_data_path,
                        target_column=runtime_paths['target_column'],
                        method=method,
                        save_path=save_path,
                    )
                    actual_outputs.append(self._unwrap_renderer_output(result))
                record['actual_outputs'] = actual_outputs
                record['status'] = 'completed'
                return record

            if task == 'plot_phase0_prior_evidence_atlas':
                phase0_output = report_inputs.get('phase0_output', {})
                if not phase0_output:
                    record['status'] = 'skipped'
                    record['skip_reason'] = 'No Phase 0 output available for Prior Evidence Atlas generation'
                    return record
                result = renderer(
                    phase0_output=phase0_output,
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_stability_landscape':
                result = renderer(
                    stability_scores_path=runtime_paths['phase1_stability_scores_path'],
                    stability_summary_path=runtime_paths['phase1_stability_summary_path'],
                    feature_provenance_path=runtime_paths.get('phase1_feature_provenance_path', ''),
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_method_feature_heatmap':
                result = renderer(
                    stability_scores_path=runtime_paths['phase1_stability_scores_path'],
                    stability_summary_path=runtime_paths['phase1_stability_summary_path'],
                    feature_provenance_path=runtime_paths.get('phase1_feature_provenance_path', ''),
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_phase1_final_panel_correlation_heatmap':
                result = renderer(
                    phase1_selected_features_path=runtime_paths['phase1_csv'],
                    feature_provenance_path=runtime_paths.get('phase1_feature_provenance_path', ''),
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_autogluon_roc':
                result = renderer(
                    ag_results_path=runtime_paths['ag_results_path'],
                    data_path=viz_data_path,
                    target_column=runtime_paths['target_column'],
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_phase1_selection_baseline_composite':
                result = renderer(
                    stability_landscape_path=str(single_figure_dir / 'phase1_stability_landscape.png'),
                    method_support_path=str(single_figure_dir / 'phase1_method_support_matrix.png'),
                    panel_correlation_path=str(single_figure_dir / 'phase1_final_panel_correlation.png'),
                    autogluon_roc_path=str(single_figure_dir / 'phase1_autogluon_baseline_roc.png'),
                    save_path=expected_outputs[0],
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                actual_outputs = renderer_metadata.get('generated_paths') or [self._unwrap_renderer_output(result)]
                record['actual_outputs'] = [str(path) for path in actual_outputs]
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            if task == 'plot_radar_4d':
                result = renderer(
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    artifact_dir=runtime_paths['artifact_dir'],
                    ag_results_path=runtime_paths['ag_results_path'],
                    feature_provenance_path=runtime_paths.get('phase1_feature_provenance_path', ''),
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                actual_outputs = [self._unwrap_renderer_output(result)]
                actual_outputs.extend(str(path) for path in renderer_metadata.get('legacy_radar_4d_paths', []) if path)
                record['actual_outputs'] = actual_outputs
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            if task == 'plot_phase2_clinical_validation_composite':
                result = renderer(
                    objective_shift_path=str(single_figure_dir / 'phase2_objective_shift_summary.png'),
                    final_roc_path=str(single_figure_dir / 'phase2_winner_roc.png'),
                    dca_path=str(single_figure_dir / 'phase2_winner_holdout_decision_curve.png'),
                    calibration_path=str(single_figure_dir / 'phase2_winner_calibration.png'),
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                actual_outputs = renderer_metadata.get('generated_paths') or [self._unwrap_renderer_output(result)]
                record['actual_outputs'] = [str(path) for path in actual_outputs]
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            if task == 'plot_phase2_radar_validation_composite':
                result = renderer(
                    radar_4d_path=str(single_figure_dir / 'phase2_4d_radar_profile.pdf'),
                    final_roc_path=str(single_figure_dir / 'phase2_winner_roc.png'),
                    dca_path=str(single_figure_dir / 'phase2_winner_holdout_decision_curve.png'),
                    calibration_path=str(single_figure_dir / 'phase2_winner_calibration.png'),
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                actual_outputs = renderer_metadata.get('generated_paths') or [self._unwrap_renderer_output(result)]
                record['actual_outputs'] = [str(path) for path in actual_outputs]
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            if task == 'plot_phase3_shap_interpretation_composite':
                result = renderer(
                    shap_summary_path=str(single_figure_dir / 'phase2_winner_shap_summary.pdf'),
                    shap_dependence_path=str(single_figure_dir / 'phase2_winner_shap_summary_dependence_panels.pdf'),
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                actual_outputs = renderer_metadata.get('generated_paths') or [self._unwrap_renderer_output(result)]
                record['actual_outputs'] = [str(path) for path in actual_outputs]
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            if task == 'plot_final_roc':
                result = renderer(
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_final_holdout_roc':
                result = renderer(
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_final_holdout_dca':
                result = _call_renderer_compat(renderer,
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_dca':
                result = _call_renderer_compat(renderer,
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_calibration':
                result = _call_renderer_compat(renderer,
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_threshold_performance':
                result = _call_renderer_compat(renderer,
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    save_path=expected_outputs[0],
                    journal_theme=journal_theme,
                )
                record['actual_outputs'] = [self._unwrap_renderer_output(result)]
                record['renderer_metadata'] = self._extract_renderer_metadata(result)
                record['status'] = 'completed'
                return record

            if task == 'plot_incremental_value_summary':
                result = renderer(
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    artifact_dir=runtime_paths['artifact_dir'],
                    save_path=expected_outputs[0],
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                actual_outputs = renderer_metadata.get('generated_paths') or [self._unwrap_renderer_output(result)]
                actual_outputs = [str(path) for path in actual_outputs if path and Path(path).exists()]
                if not actual_outputs:
                    actual_outputs = [self._unwrap_renderer_output(result)]
                record['actual_outputs'] = actual_outputs
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            if task == 'plot_shap':
                if not phase2_meta['features']:
                    record['status'] = 'skipped'
                    record['skip_reason'] = 'No winner features found in Phase 2 result'
                    return record

                shap_config = self.config.get_visualization_config('shap_plot') if self.config else {}
                config_patient_index = shap_config.get('patient_index', 1)
                env_patient_index = os.environ.get('METABOAGENT_SHAP_PATIENT_INDEX')
                raw_patient_index = env_patient_index if env_patient_index is not None else config_patient_index
                try:
                    shap_patient_index = int(raw_patient_index)
                except (TypeError, ValueError):
                    shap_patient_index = 1

                result = _call_renderer_compat(renderer,
                    data_path=viz_data_path,
                    target_column=runtime_paths['target_column'],
                    features=phase2_meta['features'],
                    champion_model_family=phase2_meta['selected_model'],
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    save_path=expected_outputs[0],
                    patient_index=shap_patient_index,
                    journal_theme=journal_theme,
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                primary_output = self._unwrap_renderer_output(result)
                # Preserve the historical report-facing alias while retaining
                # the canonical single-panel artifact on disk.
                legacy_primary = 'output/figures/fig4c_shap.pdf'
                if primary_output and str(primary_output) != legacy_primary:
                    try:
                        if Path(str(primary_output)).exists():
                            Path(legacy_primary).parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(primary_output), legacy_primary)
                    except Exception:
                        pass
                actual_outputs = [legacy_primary if primary_output else '']
                waterfall_path = renderer_metadata.get('waterfall_path')
                if waterfall_path:
                    actual_outputs.append(str(waterfall_path))
                shap_summary_path = renderer_metadata.get('shap_summary_path')
                if shap_summary_path:
                    actual_outputs.append(str(shap_summary_path))
                dependence_panels_pdf_path = renderer_metadata.get('dependence_panels_pdf_path')
                if dependence_panels_pdf_path:
                    actual_outputs.append(str(dependence_panels_pdf_path))
                record['actual_outputs'] = actual_outputs
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            if task == 'plot_rcs':
                if not phase2_meta['features']:
                    record['status'] = 'skipped'
                    record['skip_reason'] = 'No winner features found in Phase 2 result'
                    return record

                rcs_config = self.config.get_visualization_config('rcs_plot') if self.config else {}
                rcs_backend = str(rcs_config.get('backend', 'r')).strip().lower()
                rcs_knots = 3
                try:
                    rcs_n_grid = int(rcs_config.get('n_grid', 200))
                except (TypeError, ValueError):
                    rcs_n_grid = 200
                output_filename = rcs_config.get('output_filename', 'phase2_winner_rcs_curves.pdf')
                configured_output_path = Path(str(output_filename))
                if configured_output_path.parent == Path('.'):
                    rcs_save_path = str(single_figure_dir / output_filename)
                elif configured_output_path.is_absolute():
                    rcs_save_path = str(configured_output_path)
                else:
                    rcs_save_path = str(runtime_root / configured_output_path)

                result = renderer(
                    data_path=viz_data_path,
                    target_column=runtime_paths['target_column'],
                    features=phase2_meta['features'],
                    save_path=rcs_save_path,
                    n_knots=rcs_knots,
                    n_grid=rcs_n_grid,
                    backend=rcs_backend,
                    phase2_result_path=phase2_meta['phase2_result_path'],
                    journal_theme=journal_theme,
                )
                renderer_metadata = self._extract_renderer_metadata(result)
                actual_outputs = [self._unwrap_renderer_output(result)]
                summary_path = renderer_metadata.get('rcs_summary_path')
                if summary_path:
                    actual_outputs.append(str(summary_path))
                png_path = renderer_metadata.get('rcs_png_path')
                if png_path:
                    actual_outputs.append(str(png_path))
                svg_path = renderer_metadata.get('rcs_svg_path')
                if svg_path:
                    actual_outputs.append(str(svg_path))
                record['actual_outputs'] = actual_outputs
                record['renderer_metadata'] = renderer_metadata
                record['status'] = 'completed'
                return record

            record['status'] = 'skipped'
            record['skip_reason'] = f'No dispatcher implemented for task {task}'
            return record
        except Exception as exc:
            message = str(exc)
            if task == 'plot_phase0_prior_evidence_atlas' and 'No non-zero prior evidence rows remain after filtering.' in message:
                record['status'] = 'skipped'
                record['skip_reason'] = 'No non-zero prior evidence rows were available for the Phase 0 Prior Evidence Atlas'
                return record
            record['status'] = 'failed'
            record['error'] = message
            return record

    def _generate_report_artifacts(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("[Step 5] Generating Report Artifacts")
        print("=" * 80)

        runtime_paths = self._resolve_runtime_paths()
        report_inputs = self._load_report_inputs(runtime_paths)
        evaluator = self._build_evaluator(runtime_paths, report_inputs)
        raw_scores = self._collect_raw_scores(runtime_paths, evaluator, report_inputs)
        normalized_scores, max_cost = self._normalize_scores(raw_scores)
        generated_files = self._collect_task_generated_files()
        scores_json_path = self._write_scores_json(
            runtime_paths=runtime_paths,
            report_inputs=report_inputs,
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
            max_cost=max_cost,
        )
        generated_files['radar_scores_json'] = scores_json_path
        figure_manifest = self._build_figure_manifest(runtime_paths)
        manifest_paths = self._write_figure_manifest(figure_manifest)
        generated_files['figure_manifest'] = manifest_paths['latest_manifest_path']
        generated_files['figure_manifest_run'] = manifest_paths['run_manifest_path']

        return {
            'runtime_paths': runtime_paths,
            'report_inputs': report_inputs,
            'raw_scores': raw_scores,
            'normalized_scores': normalized_scores,
            'max_cost': max_cost,
            'generated_files': generated_files,
            'figure_manifest': figure_manifest,
            'figure_manifest_paths': manifest_paths,
        }

    def _collect_task_generated_files(self) -> Dict[str, str]:
        generated: Dict[str, str] = {}
        for task, record in self.generated_figures.items():
            if record.get('status') != 'completed':
                continue
            outputs = record.get('actual_outputs') or []
            if not outputs:
                continue
            generated[task] = outputs[0]
        return generated

    def _build_figure_manifest(self, runtime_paths: Dict[str, str]) -> Dict[str, Any]:
        phase2_run_source = (
            runtime_paths.get('phase2_history_json')
            or self.loaded_state_paths.get('phase2')
            or runtime_paths.get('phase2_json')
            or ''
        )
        run_id = (
            str(os.environ.get('METABOAGENT_RUN_TAG', '') or '').strip()
            or _extract_run_id_from_phase2_path(phase2_run_source)
        )
        report_root = Path('output/reports')
        run_report_dir = report_root / run_id
        run_figure_dir = Path('output/figures') / run_id

        figures: List[Dict[str, Any]] = []
        task_records: List[Dict[str, Any]] = []

        task_order = self.required_figures or list(self.generated_figures.keys())
        for task in task_order:
            record = self.generated_figures.get(task, {})
            actual_outputs = [
                _materialize_run_figure(path, run_figure_dir)
                for path in (record.get('actual_outputs') or [])
                if path
            ]
            task_records.append(
                {
                    'task': task,
                    'plot_key': record.get('plot_key', ''),
                    'renderer_name': record.get('renderer_name', ''),
                    'status': record.get('status', 'unknown'),
                    'outputs': actual_outputs,
                    'renderer_metadata': dict(record.get('renderer_metadata') or {}),
                    'skip_reason': record.get('skip_reason', ''),
                    'error': record.get('error', ''),
                }
            )

            if record.get('status') != 'completed':
                continue

            task_specs = TASK_MANIFEST_SPECS.get(task, [])
            if not task_specs and actual_outputs:
                task_specs = [
                    {
                        'figure_id': task,
                        'title': task,
                        'section': 'Results',
                        'output_index': 0,
                        'caption_seed': '',
                    }
                ]

            for spec in task_specs:
                output_index = int(spec.get('output_index', 0))
                primary_index = output_index if 0 <= output_index < len(actual_outputs) else None
                primary_path = actual_outputs[primary_index] if primary_index is not None else ''

                # Renderer output layouts are allowed to evolve (for example,
                # SHAP may return [primary, waterfall, summary-json] in one
                # renderer and [primary, summary-json, dependence-pdf] in
                # another).  Resolve metadata by its semantic path hint first
                # instead of relying only on a positional index.
                renderer_metadata = dict(record.get('renderer_metadata') or {})
                metadata_index = spec.get('metadata_output_index')
                metadata_hint = renderer_metadata.get('shap_summary_path')
                if metadata_hint:
                    hint_norm = os.path.normpath(str(metadata_hint))
                    hint_base = os.path.basename(hint_norm)
                    for index, candidate in enumerate(actual_outputs):
                        candidate_norm = os.path.normpath(str(candidate))
                        if (
                            candidate_norm == hint_norm
                            or os.path.basename(candidate_norm) == hint_base
                            or candidate_norm.endswith(hint_norm)
                            or hint_norm.endswith(candidate_norm)
                        ):
                            metadata_index = index
                            break
                if not isinstance(metadata_index, int) or not (0 <= metadata_index < len(actual_outputs)):
                    metadata_index = None

                configured_auxiliary_indices = spec.get('auxiliary_output_indices') or []
                if metadata_index is not None:
                    # Once metadata is identified semantically, every other
                    # generated output is an auxiliary artifact.  This keeps
                    # both legacy and current SHAP renderer layouts valid.
                    auxiliary_indices = [
                        index for index in range(len(actual_outputs))
                        if index != primary_index and index != metadata_index
                    ]
                else:
                    auxiliary_indices = [
                        index for index in configured_auxiliary_indices
                        if index != primary_index
                    ]
                auxiliary_outputs = [
                    _build_output_descriptor(actual_outputs[index], 'auxiliary')
                    for index in auxiliary_indices
                    if 0 <= index < len(actual_outputs) and actual_outputs[index]
                ]

                metadata_output = None
                if metadata_index is not None:
                    metadata_output = _build_output_descriptor(actual_outputs[metadata_index], 'metadata')

                figures.append(
                    {
                        'figure_id': spec.get('figure_id', task),
                        'title': spec.get('title', task),
                        'section': spec.get('section', 'Results'),
                        'source_task': task,
                        'plot_key': record.get('plot_key', ''),
                        'renderer_name': record.get('renderer_name', ''),
                        'status': 'completed' if primary_path else 'missing_primary_output',
                        'caption_seed': spec.get('caption_seed', ''),
                        'primary_output': _build_output_descriptor(primary_path, 'primary') if primary_path else None,
                        'auxiliary_outputs': auxiliary_outputs,
                        'metadata_output': metadata_output,
                        'renderer_metadata': dict(record.get('renderer_metadata') or {}),
                        'all_outputs': [
                            _build_output_descriptor(path, 'generated')
                            for path in actual_outputs
                        ],
                    }
                )

        return {
            'schema_version': 'phase4.figure_manifest.v1',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'run_id': run_id,
            'phase2_run_source': _normalize_manifest_path(phase2_run_source),
            'phase2_result_path': _normalize_manifest_path(runtime_paths.get('phase2_json', '')),
            'phase2_history_path': _normalize_manifest_path(runtime_paths.get('phase2_history_json', '')),
            'figures_dir': _normalize_manifest_path(str(run_figure_dir)),
            'source_figures_dir': _normalize_manifest_path(runtime_paths.get('figures_dir', '')),
            'report_root': _normalize_manifest_path(str(report_root)),
            'run_report_dir': _normalize_manifest_path(str(run_report_dir)),
            'required_tasks': task_order,
            'task_records': task_records,
            'figures': figures,
            'summary': {
                'required_task_count': len(task_order),
                'completed_task_count': sum(1 for item in task_records if item.get('status') == 'completed'),
                'figure_count': len(figures),
            },
            'notes': [
                'Each figure entry maps a Phase 3 plotting task to report-ready figure metadata.',
                'primary_output is the main asset referenced by Phase 4, while auxiliary_outputs and metadata_output preserve secondary deliverables.',
                'All manifest outputs are materialized under the current run-specific figures_dir; source_figures_dir is retained for provenance only.',
            ],
        }

    def _write_figure_manifest(self, manifest: Dict[str, Any]) -> Dict[str, str]:
        report_root = Path('output/reports')
        run_report_dir = report_root / manifest['run_id']
        report_root.mkdir(parents=True, exist_ok=True)
        run_report_dir.mkdir(parents=True, exist_ok=True)

        run_manifest_path = run_report_dir / 'figure_manifest.json'
        latest_manifest_path = report_root / 'figure_manifest.json'

        payload = json.dumps(manifest, indent=2, ensure_ascii=False)
        run_manifest_path.write_text(payload + "\n", encoding='utf-8')
        latest_manifest_path.write_text(payload + "\n", encoding='utf-8')

        return {
            'run_manifest_path': str(run_manifest_path),
            'latest_manifest_path': str(latest_manifest_path),
        }

    def _resolve_runtime_paths(self) -> Dict[str, str]:
        assert self.config is not None
        runtime_root = Path(str(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or '').strip()) if os.environ.get('METABOAGENT_RUNTIME_ROOT') else None
        phase1_cfg_paths = self.config.get_phase1_paths() if hasattr(self.config, "get_phase1_paths") else {}
        phase1_io_policy = self.config.get_phase1_io_policy() if hasattr(self.config, "get_phase1_io_policy") else {}

        prefer_new_read = bool(phase1_io_policy.get("prefer_new_read", False))
        fallback_old_read = bool(phase1_io_policy.get("fallback_old_read", True))

        runtime_phase1_csv = runtime_root / 'phase1' / 'final' / 'selected_features_final.csv' if runtime_root else None
        runtime_holdout_csv = runtime_root / 'phase1' / 'final' / 'selected_features_holdout.csv' if runtime_root else None
        legacy_phase1_csv = str(runtime_phase1_csv) if runtime_phase1_csv and runtime_phase1_csv.exists() else (phase1_cfg_paths.get('legacy_selected_features') or self.config.get_path('phase1_selected_features'))
        canonical_phase1_csv = str(runtime_phase1_csv) if runtime_phase1_csv and runtime_phase1_csv.exists() else (phase1_cfg_paths.get('selected_features_csv') or legacy_phase1_csv)

        runtime_legacy_artifacts = runtime_root / 'phase1' / 'legacy' / 'artifacts' if runtime_root else None
        runtime_canonical_artifacts = runtime_root / 'phase1' / 'artifacts' if runtime_root else None
        legacy_artifact_dir = str(runtime_legacy_artifacts) if runtime_legacy_artifacts and runtime_legacy_artifacts.exists() else phase1_cfg_paths.get('legacy_artifacts_dir', 'output/artifacts')
        canonical_artifact_dir = str(runtime_canonical_artifacts) if runtime_canonical_artifacts and runtime_canonical_artifacts.exists() else phase1_cfg_paths.get('artifacts_dir', legacy_artifact_dir)
        artifact_filename = 'phase1_panel_scores.json'
        runtime_intermediate = runtime_root / 'phase1' / 'intermediate' / 'latest' if runtime_root else None
        intermediate_latest_dir = str(runtime_intermediate) if runtime_intermediate and runtime_intermediate.exists() else phase1_cfg_paths.get('intermediate_latest_dir', 'output/phase1/intermediate/latest')

        phase1_csv = legacy_phase1_csv
        if prefer_new_read and os.path.exists(canonical_phase1_csv):
            phase1_csv = canonical_phase1_csv
        elif fallback_old_read and os.path.exists(legacy_phase1_csv):
            phase1_csv = legacy_phase1_csv
        phase1_holdout_csv = os.path.join(os.path.dirname(phase1_csv), 'selected_features_holdout.csv')
        if not os.path.exists(phase1_holdout_csv):
            phase1_holdout_csv = ''

        phase1_artifact_dir = legacy_artifact_dir
        canonical_artifact_path = os.path.join(canonical_artifact_dir, artifact_filename)
        legacy_artifact_path = os.path.join(legacy_artifact_dir, artifact_filename)
        if prefer_new_read and os.path.exists(canonical_artifact_path):
            phase1_artifact_dir = canonical_artifact_dir
        elif fallback_old_read and os.path.exists(legacy_artifact_path):
            phase1_artifact_dir = legacy_artifact_dir

        runtime_ag_results = runtime_root / 'phase1' / 'artifacts' / 'autogluon_training_results.json' if runtime_root else None
        ag_results_path = str(runtime_ag_results) if runtime_ag_results and runtime_ag_results.exists() else phase1_cfg_paths.get('autogluon_results', 'output/phase1/artifacts/autogluon_training_results.json')
        if not os.path.exists(ag_results_path):
            ag_results_path = 'data/autogluon_training_results.json'

        phase1_feature_provenance_path = os.path.join(canonical_artifact_dir, 'feature_provenance.json')
        if not os.path.exists(phase1_feature_provenance_path):
            phase1_feature_provenance_path = os.path.join(legacy_artifact_dir, 'feature_provenance.json')

        phase1_stability_summary_path = os.path.join(
            intermediate_latest_dir,
            'feature_selection',
            'stability_selection_summary.json',
        )
        phase1_stability_scores_path = os.path.join(
            intermediate_latest_dir,
            'feature_selection',
            'stability_scores.json',
        )

        # Reuse the resolved Phase 2 state path so report inputs and plots stay
        # aligned with the latest run context instead of drifting to config.yaml.
        phase0_output_json = (
            self.loaded_state_paths.get('phase0')
            or self.config.get_phase0_path('phase0_output')
            or _get_canonical_phase0_output_path()
        )
        phase0_output_json = phase0_output_json if os.path.exists(phase0_output_json) else _get_canonical_phase0_output_path()
        phase0_payload = _load_json_if_exists(phase0_output_json)

        phase2_json = (
            _get_canonical_phase2_winner_path()
            or self.loaded_state_paths.get('phase2')
            or self.config.get_path('phase2_winner_panel')
        )
        phase2_history_json = phase2_json or _find_latest_phase2_result()
        phase2_payload = _load_json_if_exists(phase2_json)
        phase2_history_payload = (
            _load_json_if_exists(phase2_history_json)
            if phase2_history_json and phase2_history_json != phase2_json
            else {}
        )
        resolved_test_data_path = (
            str(phase0_payload.get('data_path', '') or '').strip()
            or str(phase2_payload.get('data_path', '') or '').strip()
            or str(phase2_history_payload.get('data_path', '') or '').strip()
            or self.config.get_test_data_path()
        )
        resolved_target_column = (
            str(phase0_payload.get('target_column', '') or '').strip()
            or str(phase2_payload.get('target_column', '') or '').strip()
            or str((phase2_payload.get('dataset_fingerprint', {}) or {}).get('target_column', '') or '').strip()
            or str(phase2_history_payload.get('target_column', '') or '').strip()
            or str((phase2_history_payload.get('dataset_fingerprint', {}) or {}).get('target_column', '') or '').strip()
            or self.config.get_target_column()
        )
        return {
            'phase0_output_json': phase0_output_json,
            'phase1_csv': phase1_csv,
            'phase1_holdout_csv': phase1_holdout_csv,
            'phase2_json': phase2_json,
            'phase2_history_json': phase2_history_json,
            'test_data_path': resolved_test_data_path,
            'engineered_data_path': _get_canonical_phase1_engineered_data_path(),
            'target_column': resolved_target_column,
            'figures_dir': str(runtime_root / 'output' / 'figures') if runtime_root else self.config.get_figures_dir(),
            'artifact_dir': phase1_artifact_dir,
            'ag_results_path': ag_results_path,
            'phase1_intermediate_latest_dir': intermediate_latest_dir,
            'phase1_feature_provenance_path': phase1_feature_provenance_path,
            'phase1_stability_summary_path': phase1_stability_summary_path,
            'phase1_stability_scores_path': phase1_stability_scores_path,
        }

    def _load_report_inputs(self, runtime_paths: Dict[str, str]) -> Dict[str, Any]:
        phase0_source_path = runtime_paths.get('phase0_output_json') or runtime_paths['phase2_json']
        phase0_biomarkers = load_phase0_features(phase0_source_path)
        phase1_features = load_phase1_features(runtime_paths['phase1_csv'])
        phase2_features, phase2_scores, embedded_phase0_output = load_phase2_winner(runtime_paths['phase2_json'])
        # The current run's phase0 file is authoritative. Phase2 JSONs can
        # contain a stale embedded payload from the shared legacy storage path.
        phase0_output = _load_phase0_output_from_any_source(phase0_source_path)
        if not phase0_output:
            phase0_output = embedded_phase0_output
        return {
            'phase0_biomarkers': phase0_biomarkers,
            'phase1_features': phase1_features,
            'phase2_features': phase2_features,
            'phase2_scores': phase2_scores,
            'phase0_output': phase0_output,
        }

    def _build_evaluator(self, runtime_paths: Dict[str, str], report_inputs: Dict[str, Any]) -> ClinicalEvaluator:
        priors_source = report_inputs.get('phase0_output', {}).get('confirmed_biomarkers') or report_inputs['phase0_biomarkers']
        priors_dict = build_priors_dict(priors_source)
        return ClinicalEvaluator(
            data_path=runtime_paths['test_data_path'],
            holdout_data_path=runtime_paths.get('phase1_holdout_csv') or None,
            target_column=runtime_paths['target_column'],
            priors_dict=priors_dict,
            taxonomy_map={},
            pathway_map={},
            ag_results_path=runtime_paths['ag_results_path'],
            disease_name=report_inputs.get('phase0_output', {}).get('disease_name'),
            phase0_output=report_inputs.get('phase0_output', {}),
        )

    def _collect_raw_scores(
        self,
        runtime_paths: Dict[str, str],
        evaluator: ClinicalEvaluator,
        report_inputs: Dict[str, Any],
        include_phase0_baseline: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        print("\n[5.1] Collecting raw scores")
        raw_scores: Dict[str, Dict[str, float]] = {}
        use_artifacts = os.environ.get('USE_ARTIFACTS', 'true').lower() == 'true'

        if include_phase0_baseline:
            phase0_scores = evaluator.score_panel(
                features=report_inputs['phase0_biomarkers'],
                panel_name='Phase 0 Baseline',
            )
            raw_scores['Phase 0 Baseline'] = phase0_scores
        else:
            print('  ⏭️  Skipping Phase 0 Baseline rescoring for radar generation')

        phase1_scores = None
        if use_artifacts:
            from src.tools.analysis.artifact_exporters import read_phase1_panel_scores

            phase1_artifact = read_phase1_panel_scores(runtime_paths['artifact_dir'])
            if phase1_artifact:
                phase1_scores = phase1_artifact.get('scores')
                print(f"  ✓ Loaded Phase 1 scores from artifact: {runtime_paths['artifact_dir']}")

        if phase1_scores is None:
            print('  ⏭️  Artifact not found or disabled, recomputing Phase 1 scores...')
            phase1_scores = evaluator.score_panel(
                features=report_inputs['phase1_features'],
                panel_name='Phase 1 Baseline',
            )
        raw_scores['Phase 1 Baseline'] = phase1_scores

        raw_scores['PToT Winner'] = report_inputs['phase2_scores']
        print('  ✓ Using Phase 2 winner scores from result JSON')
        return raw_scores

    def _normalize_scores(self, raw_scores: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Dict[str, float]], float]:
        max_cost = max(score_bundle['f_cost'] for score_bundle in raw_scores.values())
        normalized_scores = {
            model_name: normalize_scores_for_radar(score_bundle, max_cost)
            for model_name, score_bundle in raw_scores.items()
        }
        return normalized_scores, max_cost

    def _write_scores_json(
        self,
        runtime_paths: Dict[str, str],
        report_inputs: Dict[str, Any],
        raw_scores: Dict[str, Dict[str, float]],
        normalized_scores: Dict[str, Dict[str, float]],
        max_cost: float,
    ) -> str:
        output_json = Path(runtime_paths['figures_dir']) / 'radar_scores.json'
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'raw_scores': raw_scores,
                    'normalized_scores': normalized_scores,
                    'max_cost': max_cost,
                    'config': {
                        'data_path': runtime_paths['test_data_path'],
                        'target_column': runtime_paths['target_column'],
                        'phase0_biomarkers_count': len(report_inputs['phase0_biomarkers']),
                        'phase1_features_count': len(report_inputs['phase1_features']),
                        'phase2_features_count': len(report_inputs['phase2_features']),
                    },
                },
                f,
                indent=2,
            )
        return str(output_json)

    def _generate_final_report(self, context: Dict[str, Any]) -> str:
        print("\n" + "=" * 80)
        print("[Step 6] Writing Final Phase 3 Report")
        print("=" * 80)

        report_dir = Path('output/reports')
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / 'phase3_pipeline_summary.md'

        report_inputs = context['report_inputs']
        generated_files = context['generated_files']
        raw_scores = context['raw_scores']
        normalized_scores = context['normalized_scores']
        figure_manifest = context.get('figure_manifest', {})
        figure_manifest_paths = context.get('figure_manifest_paths', {})
        # The figure manifest may use a stable artifact label (for example
        # ``winner_scores``), which is not safe as a delivery-package ID.
        # Prefer the orchestration run tag when supplied.
        run_id = (
            str(os.environ.get('METABOAGENT_RUN_TAG', '') or '').strip()
            or figure_manifest.get('run_id')
            or datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        run_report_dir = report_dir / run_id
        run_report_dir.mkdir(parents=True, exist_ok=True)
        run_report_path = run_report_dir / 'phase3_pipeline_summary.md'

        code_paths = [
            'src/agents/phase1/state.py',
            'src/agents/phase2/ptot_engine.py',
            'src/agents/phase3/generate_final_report.py',
            'src/utils/evaluation_audit.py',
        ]
        write_audit_json('run_manifest.json', {
            'assessment_domain': 'run_identity',
            'run_id': run_id,
            'config': describe_path(self.config_path),
            'phase_state_paths': self.loaded_state_paths,
            'run_tag': str(os.environ.get('METABOAGENT_RUN_TAG', '') or ''),
            'validation_cohort_type': str(os.environ.get('METABOAGENT_VALIDATION_COHORT_TYPE', 'unspecified')),
        })
        write_audit_json('environment_manifest.json', {
            'assessment_domain': 'reproducibility',
            'status': 'PASS',
            **environment_snapshot(code_paths),
        })
        write_audit_json('figure_table_manifest.json', {
            'assessment_domain': 'figures_and_tables',
            'status': 'PASS' if figure_manifest.get('summary', {}).get('figure_count', 0) else 'PARTIAL',
            'figure_manifest': describe_path(figure_manifest_paths.get('latest_manifest_path', '')),
            'run_figure_manifest': describe_path(figure_manifest_paths.get('run_manifest_path', '')),
            'generated_files': {label: describe_path(path) for label, path in generated_files.items()},
            'evaluation_scope': 'see source artifacts; no visual claims are inferred by audit layer',
        })
        phase1_context = self.phase1_state.get('context_variables', {}) if isinstance(self.phase1_state, dict) else {}
        selection_log_candidates = [
            phase1_context.get('feature_selection_log_path', ''),
            'output/phase1/intermediate/latest/stability_selection/stability_selection_log.json',
            'data/feature_selection_decision.json',
        ]
        selected_evidence = next((path for path in selection_log_candidates if path and Path(path).exists()), '')
        write_audit_json('feature_selection_audit.json', {
            'assessment_domain': 'feature_selection',
            'status': 'PASS' if selected_evidence else 'NOT_ASSESSED',
            'policy': 'train_only_stability_selection_required_by_phase1_executor',
            'train_pool': describe_path(phase1_context.get('train_pool_data_path', '')),
            'holdout_pool': describe_path(phase1_context.get('holdout_data_path', '')),
            'selection_evidence': describe_path(selected_evidence),
            'selected_feature_count': len(report_inputs.get('phase1_features', [])),
        })
        audit_summary = build_audit_summary()

        lines = [
            '# Phase 3 Pipeline Summary',
            '',
            f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- Config path: `{self.config_path}`",
            f"- Use mock VLM: `{self.use_mock_vlm}`",
            f"- Max ACVAL retries: `{self.max_acval_retries}`",
            '',
            '## Loaded States',
            '',
        ]

        for phase_name in ['phase0', 'phase1', 'phase2']:
            state_path = self.loaded_state_paths.get(phase_name, '')
            state = getattr(self, f'{phase_name}_state', {})
            status = 'loaded' if state else 'missing'
            lines.append(f'- `{phase_name}`: `{status}` from `{state_path}`')

        lines.extend([
            '',
            '## Planned Figure Tasks',
            '',
        ])
        if not self.required_figures:
            lines.append('- No figure tasks were selected.')
        else:
            for task in self.required_figures:
                plan = self.generated_figures.get(task, {})
                outputs = plan.get('actual_outputs') or plan.get('expected_outputs', [])
                output_text = ', '.join(f'`{path}`' for path in outputs) if outputs else '`N/A`'
                note = plan.get('skip_reason') or plan.get('error')
                note_text = f" | note: `{note}`" if note else ''
                lines.append(
                    f"- `{task}` -> `{plan.get('renderer_name', 'N/A')}` | "
                    f"status: `{plan.get('status', 'unknown')}` | outputs: {output_text}{note_text}"
                )

        lines.extend([
            '',
            '## Report Inputs',
            '',
            f"- Phase 0 biomarkers: `{len(report_inputs['phase0_biomarkers'])}`",
            f"- Phase 1 features: `{len(report_inputs['phase1_features'])}`",
            f"- Phase 2 winner features: `{len(report_inputs['phase2_features'])}`",
            f"- Phase 2 bio score version: `{self.phase2_state.get('bio_score_version', 'unknown')}`",
            f"- Phase 2 protected anchors: `{self.phase2_state.get('n_protected_anchor_features', 0)}`",
            f"- Phase 2 winner ROC-AUC: `{(report_inputs.get('phase2_scores', {}) or {}).get('roc_auc', (report_inputs.get('phase2_scores', {}) or {}).get('f_perf', 0.0))}`",
            f"- Phase 2 search mean CV score: `{(report_inputs.get('phase2_scores', {}) or {}).get('search_mean_cv_score', 0.0)}`",
            f"- Phase 2 winner Brier score: `{(report_inputs.get('phase2_scores', {}) or {}).get('brier_score', 0.0)}`",
            '',
            '## Raw Scores',
            '',
        ])
        for model_name, score_bundle in raw_scores.items():
            lines.append(f"- `{model_name}`: `{score_bundle}`")

        lines.extend([
            '',
            '## Normalized Scores',
            '',
        ])
        for model_name, score_bundle in normalized_scores.items():
            lines.append(f"- `{model_name}`: `{score_bundle}`")

        lines.extend([
            '',
            '## Figure Manifest',
            '',
            f"- Run ID: `{figure_manifest.get('run_id', 'N/A')}`",
            f"- Figure count: `{figure_manifest.get('summary', {}).get('figure_count', 0)}`",
            f"- Completed tasks: `{figure_manifest.get('summary', {}).get('completed_task_count', 0)}` / `{figure_manifest.get('summary', {}).get('required_task_count', 0)}`",
            '',
            '## Generated Files',
            '',
        ])
        for label, path in generated_files.items():
            lines.append(f'- `{label}`: `{path}`')

        # Deterministic, non-LLM audit summary.  Raw audit evidence remains in
        # output/audit; the report contains only a compact, fixed-rule view.
        lines.extend(['', '## Analysis Traceability & Quality Control', ''])
        lines.append('| Domain | Status | Evidence |')
        lines.append('|---|---|---|')
        for item in audit_summary.get('domains', []):
            lines.append(
                f"| {item['domain']} | {item['status']} | `{item['evidence_file']}` |"
            )
        lines.extend([
            '',
            '- Audit statuses are deterministic artifact-completeness checks; they do not replace methodological review.',
            '- Validation cohort labels must be interpreted from the corresponding audit evidence.',
        ])

        payload = "\n".join(lines) + "\n"
        report_path.write_text(payload, encoding='utf-8')
        run_report_path.write_text(payload, encoding='utf-8')
        print(f"✅ Final report written: {report_path}")

        # Record phase-level execution provenance before final packaging.  The
        # trace is filesystem-backed and intentionally separate from the LLM
        # report, so it can be checked without trusting narrative text.
        runtime_root = Path(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or Path.cwd())
        phase2_path = (
            figure_manifest.get('phase2_result_path', '')
            or str(runtime_root / 'output' / 'artifacts' / 'phase2_winner_scores.json')
        )
        phase0_trace_path = self.loaded_state_paths.get('phase0', '')
        phase1_trace_path = self.loaded_state_paths.get('phase1', '')
        if not phase0_trace_path or not Path(phase0_trace_path).exists():
            phase0_trace_path = str(runtime_root / 'phase0')
        if not phase1_trace_path or not Path(phase1_trace_path).exists():
            phase1_trace_path = str(runtime_root / 'phase1')
        write_execution_trace(
            run_id=run_id,
            stage_paths={
                'phase0': phase0_trace_path,
                'phase1': phase1_trace_path,
                'phase2': str(phase2_path),
                'phase3_report': str(run_report_path),
            },
            status='PASS',
            extra={
                'phase_state_paths': self.loaded_state_paths,
                'generated_file_count': len(generated_files),
                'figure_task_count': len(self.required_figures),
            },
        )

        write_audit_json('artifact_manifest.json', build_artifact_manifest())
        audit_summary = build_audit_summary()
        audit_summary_path = write_audit_json('audit_summary.json', audit_summary)
        generated_paths = [str(path) for path in generated_files.values() if path]
        figure_paths = [path for path in generated_paths if Path(path).suffix.lower() in {'.png', '.jpg', '.jpeg', '.pdf', '.svg'}]
        table_paths = [path for path in generated_paths if Path(path).suffix.lower() in {'.csv', '.tsv', '.xlsx'}]
        delivery_package = package_final_delivery(
            run_id=run_id,
            report_paths=[str(report_path), str(run_report_path)],
            figure_paths=figure_paths,
            table_paths=table_paths,
        )

        disease_name = (
            context.get('report_inputs', {}).get('phase0_output', {}).get('disease_name')
            or self.phase2_state.get('phase0_output', {}).get('disease_name')
            or 'unknown'
        )
        stage_outputs = {
            'phase3_pipeline_summary_md': str(report_path),
            'phase3_pipeline_summary_run_md': str(run_report_path),
            'figure_manifest_json': figure_manifest_paths.get('latest_manifest_path', ''),
            'figure_manifest_run_json': figure_manifest_paths.get('run_manifest_path', ''),
            'radar_scores_json': generated_files.get('radar_scores_json', ''),
            'audit_summary_json': audit_summary_path,
            'final_delivery_root': delivery_package.get('delivery_root', ''),
            'final_delivery_audit_readme': delivery_package.get('readme', ''),
            'final_delivery_manifest': delivery_package.get('delivery_manifest', ''),
        }
        for label, path in generated_files.items():
            if label == 'radar_scores_json':
                continue
            stage_outputs[f'generated_{label}'] = path
        archived_outputs = archive_artifacts(
            run_id=run_id,
            disease_name=disease_name,
            stage='phase3',
            artifact_paths=stage_outputs,
        )
        run_tag = str(os.environ.get("METABOAGENT_RUN_TAG", "") or "").strip()
        record_stage_run(
            run_id=run_id,
            disease_name=disease_name,
            stage='phase3',
            parameters={
                'config_path': self.config_path,
                'use_mock_vlm': self.use_mock_vlm,
                'max_acval_retries': self.max_acval_retries,
                'run_tag': run_tag,
            },
            inputs={
                'phase0_state_path': self.loaded_state_paths.get('phase0', ''),
                'phase1_state_path': self.loaded_state_paths.get('phase1', ''),
                'phase2_state_path': self.loaded_state_paths.get('phase2', ''),
                'test_data_path': context.get('runtime_paths', {}).get('test_data_path', ''),
                'target_column': context.get('runtime_paths', {}).get('target_column', ''),
            },
            outputs=stage_outputs,
            archived_outputs=archived_outputs,
            metrics={
                'phase0_biomarkers_count': len(report_inputs['phase0_biomarkers']),
                'phase1_features_count': len(report_inputs['phase1_features']),
                'phase2_features_count': len(report_inputs['phase2_features']),
                'figure_count': figure_manifest.get('summary', {}).get('figure_count', 0),
                'completed_task_count': figure_manifest.get('summary', {}).get('completed_task_count', 0),
                'required_task_count': figure_manifest.get('summary', {}).get('required_task_count', 0),
                'ptot_winner_f_perf': raw_scores.get('PToT Winner', {}).get('f_perf'),
                'ptot_winner_roc_auc': raw_scores.get('PToT Winner', {}).get('roc_auc'),
                'ptot_winner_search_mean_cv_score': raw_scores.get('PToT Winner', {}).get('search_mean_cv_score'),
                'ptot_winner_f_bio': raw_scores.get('PToT Winner', {}).get('f_bio'),
                'ptot_winner_f_corr': raw_scores.get('PToT Winner', {}).get('f_corr'),
                'ptot_winner_f_cost': raw_scores.get('PToT Winner', {}).get('f_cost'),
                'ptot_winner_brier_score': raw_scores.get('PToT Winner', {}).get('brier_score'),
            },
            lineage={
                'phase2_run_source': figure_manifest.get('phase2_run_source', ''),
                'phase2_result_path': figure_manifest.get('phase2_result_path', ''),
            },
            notes={
                'run_tag': run_tag,
                'normalized_scores': normalized_scores,
            },
        )
        print("✅ Run registry updated: storage/run_registry")
        return str(report_path)


def generate_final_report(
    config_path: str = 'config.yaml',
    use_mock_vlm: bool = True,
    max_acval_retries: int = 3,
) -> str:
    """运行 Phase 3 最终报告生成流程。"""
    # A run may be launched from the project root while its Phase 0–2 and
    # audit artifacts live under a run-specific directory.  All figure hints
    # are relative paths (``output/figures/...``), so leaving the process in
    # the project root allows a stale global figure to be materialized into a
    # new report.  When the runtime root is declared, make it the temporary
    # working directory for the whole Phase 3 render and restore the caller's
    # directory afterwards.  This does not alter the analytical workflow; it
    # only makes figure and report assets run-scoped and provenance-safe.
    runtime_root = str(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or '').strip()
    original_cwd = os.getcwd()
    resolved_config = config_path
    if runtime_root and Path(runtime_root).is_dir():
        config_candidate = Path(config_path)
        if not config_candidate.is_absolute():
            project_root = str(os.environ.get('METABOAGENT_PROJECT_ROOT', '') or '').strip()
            if project_root:
                config_candidate = Path(project_root) / config_candidate
            else:
                config_candidate = Path(original_cwd) / config_candidate
        resolved_config = str(config_candidate.resolve())
        # evaluator_bridge obtains the process-wide ConfigManager through
        # get_config() with its default relative path.  Prime that singleton
        # with the resolved project config before changing cwd to the run
        # directory, otherwise it would look for run_root/config.yaml.
        try:
            from src.utils.config_manager import get_config
            get_config(resolved_config)
        except Exception:
            # Phase 3 will surface the original configuration error if the
            # project config cannot be initialized.
            pass
        os.chdir(runtime_root)
    pipeline = Phase3Pipeline(
        config_path=resolved_config,
        use_mock_vlm=use_mock_vlm,
        max_acval_retries=max_acval_retries,
    )
    try:
        return pipeline.run()
    finally:
        if os.getcwd() != original_cwd:
            os.chdir(original_cwd)


def main() -> str:
    """CLI 入口。"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3 final report generation')
    parser.add_argument('--config', default='config.yaml', help='Path to the Phase 3 config file')
    parser.add_argument('--use-mock-vlm', action='store_true', default=True, help='Use mock VLM reviewer')
    parser.add_argument('--max-acval-retries', type=int, default=3, help='Max ACVAL retries')
    args = parser.parse_args()

    return generate_final_report(
        config_path=args.config,
        use_mock_vlm=args.use_mock_vlm,
        max_acval_retries=args.max_acval_retries,
    )


if __name__ == '__main__':
    main()
