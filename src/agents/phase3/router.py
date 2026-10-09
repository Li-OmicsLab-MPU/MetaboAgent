"""
Phase 3: Figure Router (图表路由中心)

核心功能:
根据 Phase 0/1/2 的历史状态，动态决定需要生成哪些图表。

设计理念:
- 高内聚: 所有图表任务分发逻辑集中在一个类中
- 可扩展: 新增图表类型只需添加新的判断分支
- 配置驱动: 支持从 config.yaml 读取可视化配置

Author: MetaboAgent Team
Date: 2026-04-24
"""

import os
import json
from typing import List, Dict, Any, Optional


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


def _load_json_if_exists(file_path: str) -> Dict[str, Any]:
    if not file_path or not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class FigureRouter:
    """
    图表路由中心 (Figure Router)

    根据上游 Phase 的执行状态，动态决定需要生成哪些图表。

    核心图表矩阵 (Figure Matrix):
    - Fig 1 (Phase 1 产出):
      * 1A: PCA/PLS-DA 散点图 (统计学显著性)
      * 1B: UpSet Plot (多方法投票特征交集 + Prior Biomarker 高亮)

    - Fig 2 (Phase 1 产出):
      * AutoGluon 多模型基础 ROC 曲线对比图

    - Fig 3 (Phase 2 产出):
      * 3A: 帕累托进化轨迹散点图 (Pareto Trajectory)
      * 3B: 多模型 4D 雷达对比图 (4D Radar)

    - Fig 4 (Winner 产出):
      * 4A: 最终 ROC 曲线
      * 4B: DCA 决策曲线 (Decision Curve Analysis)
      * 4C: SHAP 可解释性分析摘要图
      * 4D: Winner 特征 RCS 非线性风险曲线

    Attributes:
        config: ConfigManager 实例
        phase0_state: Phase 0 状态字典
        phase1_state: Phase 1 状态字典
        phase2_state: Phase 2 状态字典

    Examples:
        >>> from src.utils.config_manager import ConfigManager
        >>> config = ConfigManager('config.yaml')
        >>> router = FigureRouter(config, phase0_state, phase1_state, phase2_state)
        >>> tasks = router.determine_required_figures()
        >>> print(tasks)
        ['plot_autogluon_roc', 'plot_radar_4d', 'plot_final_roc', 'plot_dca', 'plot_shap']
    """

    def __init__(
        self,
        config_manager: Any,
        phase0_state: Optional[Dict] = None,
        phase1_state: Optional[Dict] = None,
        phase2_state: Optional[Dict] = None
    ):
        """
        初始化图表路由器

        Args:
            config_manager: ConfigManager 实例
            phase0_state: Phase 0 状态字典（可选）
            phase1_state: Phase 1 状态字典（可选）
            phase2_state: Phase 2 状态字典（可选）
        """
        self.config = config_manager
        self.phase0_state = phase0_state or {}
        self.phase1_state = phase1_state or {}
        self.phase2_state = phase2_state or {}

        # 从配置文件读取可视化开关（如果存在）
        self.viz_config = self._load_viz_config()

    def _phase1_feature_selection_path(self, filename: str) -> str:
        try:
            if hasattr(self.config, 'get_phase1_paths'):
                phase1_paths = self.config.get_phase1_paths()
                latest_dir = str(phase1_paths.get('intermediate_latest_dir', '') or '').strip()
                if latest_dir:
                    return os.path.join(latest_dir, 'feature_selection', filename)
        except Exception:
            pass
        return os.path.join('output', 'phase1', 'intermediate', 'latest', 'feature_selection', filename)

    def _load_viz_config(self) -> Dict:
        """
        从配置文件加载可视化配置

        Returns:
            Dict: 可视化配置字典
        """
        # 尝试从 config_manager 读取 phase3 配置
        try:
            if hasattr(self.config, 'config') and 'phase3' in self.config.config:
                return self.config.config['phase3'].get('visualization', {})
        except Exception as e:
            print(f"Warning: Failed to load visualization config: {e}")

        # 默认配置：所有图表都启用
        return {
            'enable_stats_scatter': True,
            'enable_phase0_prior_evidence_atlas': True,
            'enable_stability_landscape': True,
            'enable_method_feature_heatmap': True,
            'enable_phase1_final_panel_correlation_heatmap': True,
            'enable_phase2_final_panel_correlation_heatmap': True,
            'enable_autogluon_roc': True,
            'enable_phase2_clinical_validation_composite': False,
            'enable_phase2_radar_validation_composite': False,
            'enable_radar_4d': True,
            'enable_final_roc': True,
            'enable_final_holdout_roc': True,
            'enable_dca': True,
            'enable_shap': True,
            'enable_rcs': True,
            'enable_calibration': True,
            'enable_threshold_performance': True,
            'enable_incremental_value_summary': True,
        }

    def determine_required_figures(self) -> List[str]:
        """
        根据上游状态，动态决定需要生成哪些图表

        决策逻辑:
        1. 检查 Phase 1 状态:
           - 如果使用了统计学方法 (PCA/PLS-DA) -> 'plot_stats_scatter'
           - 如果记录了 AutoGluon baseline 结果 -> 'plot_autogluon_roc'

        2. 检查 Phase 2 状态:
           - 如果包含多目标搜索日志 -> 'plot_radar_4d'
           - 如果成功锁定了最终的 winner panel -> 'plot_final_roc', 'plot_dca', 'plot_shap', 'plot_rcs', 'plot_calibration', 'plot_threshold_performance'

        Returns:
            List[str]: 需要绘制的图表任务标识符列表

        Examples:
            >>> router = FigureRouter(config, {}, phase1_state, phase2_state)
            >>> tasks = router.determine_required_figures()
            >>> 'plot_radar_4d' in tasks
            True
        """
        tasks = []

        print("\n" + "="*80)
        print("Phase 3: Figure Router - Determining Required Figures")
        print("="*80)

        # ====================================================================
        # 1. 检查 Phase 1 状态
        # ====================================================================
        print("\n[Step 1] Checking Phase 1 State...")

        # 1.1 PCA / PLS-DA scatter 已从正式 Phase 3 输出中移除
        if self._check_stats_methods_used():
            print("  ⏭️  plot_stats_scatter: Retired from Phase 3 output")
        else:
            print("  ⏭️  plot_stats_scatter: No statistical methods used")

        has_phase0_prior_outputs = self._check_phase0_prior_outputs()
        has_stability_selection_outputs = self._check_stability_selection_outputs()
        has_phase1_final_panel_outputs = self._check_phase1_final_panel_outputs()
        has_autogluon_baseline = self._check_autogluon_baseline()

        # 1.2 Phase 0 prior evidence atlas
        if has_phase0_prior_outputs:
            if self.viz_config.get('enable_phase0_prior_evidence_atlas', True):
                tasks.append('plot_phase0_prior_evidence_atlas')
                print("  ✅ plot_phase0_prior_evidence_atlas: Phase 0 prior evidence detected")
            else:
                print("  ⏭️  plot_phase0_prior_evidence_atlas: Disabled in config")
        else:
            print("  ⏭️  plot_phase0_prior_evidence_atlas: No Phase 0 prior evidence output")

        # 1.2b Stability-based Phase 1 diagnostics
        if has_stability_selection_outputs:
            if self.viz_config.get('enable_stability_landscape', True):
                tasks.append('plot_stability_landscape')
                print("  ✅ plot_stability_landscape: Stability-selection artifacts detected")
            else:
                print("  ⏭️  plot_stability_landscape: Disabled in config")

            if self.viz_config.get('enable_method_feature_heatmap', True):
                tasks.append('plot_method_feature_heatmap')
                print("  ✅ plot_method_feature_heatmap: Stability-selection artifacts detected")
            else:
                print("  ⏭️  plot_method_feature_heatmap: Disabled in config")
        else:
            print("  ⏭️  plot_stability_landscape: No stability-selection artifacts")
            print("  ⏭️  plot_method_feature_heatmap: No stability-selection artifacts")

        if has_phase1_final_panel_outputs:
            if self.viz_config.get('enable_phase1_final_panel_correlation_heatmap', True):
                tasks.append('plot_phase1_final_panel_correlation_heatmap')
                print("  ✅ plot_phase1_final_panel_correlation_heatmap: Phase 1 final panel detected")
            else:
                print("  ⏭️  plot_phase1_final_panel_correlation_heatmap: Disabled in config")
        else:
            print("  ⏭️  plot_phase1_final_panel_correlation_heatmap: No Phase 1 final panel found")

        # 1.3 AutoGluon baseline 检查
        if has_autogluon_baseline:
            if self.viz_config.get('enable_autogluon_roc', True):
                tasks.append('plot_autogluon_roc')
                print("  ✅ plot_autogluon_roc: AutoGluon baseline results found")
            else:
                print("  ⏭️  plot_autogluon_roc: Disabled in config")
        else:
            print("  ⏭️  plot_autogluon_roc: No AutoGluon baseline results")

        if has_stability_selection_outputs and has_phase1_final_panel_outputs and has_autogluon_baseline:
            if self.viz_config.get('enable_phase1_selection_baseline_composite', True):
                tasks.append('plot_phase1_selection_baseline_composite')
                print("  ✅ plot_phase1_selection_baseline_composite: fig1d/fig1e/fig1g/fig2a composite enabled")
            else:
                print("  ⏭️  plot_phase1_selection_baseline_composite: Disabled in config")
        else:
            print("  ⏭️  plot_phase1_selection_baseline_composite: Missing fig1d/fig1e/fig1g/fig2a prerequisites")

        # ====================================================================
        # 2. 检查 Phase 2 状态
        # ====================================================================
        print("\n[Step 2] Checking Phase 2 State...")

        if self._check_phase2_objective_shift_inputs():
            if self.viz_config.get('enable_radar_4d', True):
                tasks.append('plot_radar_4d')
                print("  ✅ plot_radar_4d: Phase 2 objective-shift inputs detected")
            else:
                print("  ⏭️  plot_radar_4d: Disabled in config")
        else:
            print("  ⏭️  plot_radar_4d: Missing Phase 2 objective-shift inputs")

        has_phase2_objective_shift_inputs = self._check_phase2_objective_shift_inputs()
        has_legacy_radar_4d_output = has_phase2_objective_shift_inputs or any(
            os.path.exists(path)
            for path in (
                os.path.join('output', 'figures', 'single_panels', 'phase2_4d_radar_profile.pdf'),
                os.path.join('output', 'figures', 'single_panels', 'phase2_4d_radar_profile.png'),
            )
        )
        has_winner_panel = self._check_winner_panel()

        # 2.2 Winner panel 检查
        if has_winner_panel:
            if self.viz_config.get('enable_final_roc', True):
                tasks.append('plot_final_roc')
                print("  ✅ plot_final_roc: Winner panel locked")
            else:
                print("  ⏭️  plot_final_roc: Disabled in config")

            if self.viz_config.get('enable_final_holdout_roc', True):
                tasks.append('plot_final_holdout_roc')
                print("  ✅ plot_final_holdout_roc: Winner panel locked")
            else:
                print("  ⏭️  plot_final_holdout_roc: Disabled in config")

            if self.viz_config.get('enable_final_holdout_dca', True):
                tasks.append('plot_final_holdout_dca')
                print("  ✅ plot_final_holdout_dca: Winner panel locked")
            else:
                print("  ⏭️  plot_final_holdout_dca: Disabled in config")

            if self.viz_config.get('enable_dca', True):
                tasks.append('plot_dca')
                print("  ✅ plot_dca: Winner panel locked")
            else:
                print("  ⏭️  plot_dca: Disabled in config")

            if self.viz_config.get('enable_shap', True):
                tasks.append('plot_shap')
                print("  ✅ plot_shap: Winner panel locked")
            else:
                print("  ⏭️  plot_shap: Disabled in config")

            if self.viz_config.get('enable_phase3_shap_interpretation_composite', True):
                tasks.append('plot_phase3_shap_interpretation_composite')
                print("  ✅ plot_phase3_shap_interpretation_composite: fig4c summary/dependence composite enabled")
            else:
                print("  ⏭️  plot_phase3_shap_interpretation_composite: Disabled in config")

            if self.viz_config.get('enable_rcs', True):
                tasks.append('plot_rcs')
                print("  ✅ plot_rcs: Winner panel locked")
            else:
                print("  ⏭️  plot_rcs: Disabled in config")

            if self.viz_config.get('enable_calibration', True):
                tasks.append('plot_calibration')
                print("  ✅ plot_calibration: Winner panel locked")
            else:
                print("  ⏭️  plot_calibration: Disabled in config")

            if self.viz_config.get('enable_threshold_performance', True):
                tasks.append('plot_threshold_performance')
                print("  ✅ plot_threshold_performance: Winner panel locked")
            else:
                print("  ⏭️  plot_threshold_performance: Disabled in config")

            if self.viz_config.get('enable_incremental_value_summary', True):
                tasks.append('plot_incremental_value_summary')
                print("  ✅ plot_incremental_value_summary: Winner panel locked")
            else:
                print("  ⏭️  plot_incremental_value_summary: Disabled in config")
        else:
            print("  ⏭️  plot_final_roc: No winner panel")
            print("  ⏭️  plot_final_holdout_roc: No winner panel")
            print("  ⏭️  plot_final_holdout_dca: No winner panel")
            print("  ⏭️  plot_dca: No winner panel")
            print("  ⏭️  plot_shap: No winner panel")
            print("  ⏭️  plot_phase3_shap_interpretation_composite: No winner panel")
            print("  ⏭️  plot_rcs: No winner panel")
            print("  ⏭️  plot_calibration: No winner panel")
            print("  ⏭️  plot_threshold_performance: No winner panel")
            print("  ⏭️  plot_incremental_value_summary: No winner panel")

        if has_phase2_objective_shift_inputs and has_winner_panel:
            if self.viz_config.get('enable_phase2_clinical_validation_composite', False):
                tasks.append('plot_phase2_clinical_validation_composite')
                print("  ✅ plot_phase2_clinical_validation_composite: fig3/fig4a/fig4b/fig4e composite enabled")
            else:
                print("  ⏭️  plot_phase2_clinical_validation_composite: Disabled in config")
        else:
            print("  ⏭️  plot_phase2_clinical_validation_composite: Missing fig3/fig4a/fig4b/fig4e prerequisites")

        if has_legacy_radar_4d_output and has_winner_panel:
            if self.viz_config.get('enable_phase2_radar_validation_composite', False):
                tasks.append('plot_phase2_radar_validation_composite')
                print("  ✅ plot_phase2_radar_validation_composite: phase2_4d_radar_profile/ROC/DCA/calibration composite enabled")
            else:
                print("  ⏭️  plot_phase2_radar_validation_composite: Disabled in config")
        else:
            print("  ⏭️  plot_phase2_radar_validation_composite: Missing phase2_4d_radar_profile or winner-panel prerequisites")

        # ====================================================================
        # 3. 总结
        # ====================================================================
        print("\n" + "="*80)
        print(f"Figure Router Summary: {len(tasks)} figures to generate")
        print("="*80)
        for idx, task in enumerate(tasks, 1):
            print(f"  {idx}. {task}")
        print("="*80 + "\n")

        return tasks

    # ========================================================================
    # 辅助方法: Phase 1 状态检查
    # ========================================================================

    def _check_stats_methods_used(self) -> bool:
        """
        检查 Phase 1 是否使用了统计学方法 (PCA/PLS-DA)

        Returns:
            bool: 如果使用了统计学方法，返回 True
        """
        # 检查 phase1_state 中是否有 stats_methods 或 statistical_analysis 字段
        if 'stats_methods' in self.phase1_state:
            return len(self.phase1_state['stats_methods']) > 0

        if 'statistical_analysis' in self.phase1_state:
            return self.phase1_state['statistical_analysis'].get('enabled', False)

        # 检查是否有 PCA/PLS-DA 相关的结果文件
        if 'output_files' in self.phase1_state:
            for file_path in self.phase1_state['output_files']:
                if 'pca' in file_path.lower() or 'plsda' in file_path.lower():
                    return True

        # Fallback to the latest engineered matrix when stale phase1_state is missing.
        if os.path.exists('output/phase1/intermediate/latest/engineered/data_with_engineered_features.csv'):
            return True

        return False

    def _check_multi_method_voting(self) -> bool:
        """
        检查 Phase 1 是否使用了多模型投票

        Returns:
            bool: 如果使用了多模型投票，返回 True
        """
        # 检查 phase1_state 中是否有 method_used 字段
        if 'method_used' in self.phase1_state:
            methods = self.phase1_state['method_used']
            if isinstance(methods, list):
                return len(methods) > 1
            elif isinstance(methods, int):
                return methods > 1

        # 检查是否有 feature_selection_methods 字段
        if 'feature_selection_methods' in self.phase1_state:
            methods = self.phase1_state['feature_selection_methods']
            if isinstance(methods, list):
                return len(methods) > 1

        method_results = _load_json_if_exists(self._phase1_feature_selection_path('method_results.json'))
        valid_methods = [
            method_name for method_name, features in method_results.items()
            if isinstance(features, list) and len(features) > 0
        ]
        if len(valid_methods) > 1:
            return True

        return False

    def _check_phase0_prior_outputs(self) -> bool:
        def has_phase0_payload(payload: Dict[str, Any]) -> bool:
            if not isinstance(payload, dict):
                return False
            return any(
                key in payload
                for key in (
                    'candidate_scores_summary',
                    'confirmed_biomarkers',
                    'final_priors',
                    'phase0_prior_count',
                )
            )

        phase0_output = self.phase2_state.get('phase0_output', {})
        if has_phase0_payload(phase0_output):
            return True

        explicit_phase0_path = str(self.phase2_state.get('phase0_output_path', '') or '').strip()
        if explicit_phase0_path and os.path.exists(explicit_phase0_path):
            payload = _load_json_if_exists(explicit_phase0_path)
            if has_phase0_payload(payload):
                return True

        if has_phase0_payload(self.phase0_state):
            return True

        return False

    def _check_stability_selection_outputs(self) -> bool:
        candidate_paths = [
            self._phase1_feature_selection_path('stability_selection_summary.json'),
            self._phase1_feature_selection_path('stability_scores.json'),
        ]
        return all(os.path.exists(path) for path in candidate_paths)

    def _check_phase1_qc_outputs(self) -> bool:
        try:
            if hasattr(self.config, 'get_phase1_paths'):
                phase1_paths = self.config.get_phase1_paths()
                latest_dir = str(phase1_paths.get('intermediate_latest_dir', '') or '').strip()
            else:
                latest_dir = ''
        except Exception:
            latest_dir = ''

        if not latest_dir:
            latest_dir = os.path.join('output', 'phase1', 'intermediate', 'latest')

        candidate_paths = [
            os.path.join(latest_dir, 'preprocessing_context.json'),
            os.path.join(latest_dir, 'normalization_decision_report.json'),
            os.path.join(latest_dir, 'transformation_report.json'),
            os.path.join(latest_dir, 'outlier_audit_report.json'),
            os.path.join(latest_dir, 'missingness_report.json'),
            os.path.join(latest_dir, 'stage1_1_imputed.csv'),
        ]
        return all(os.path.exists(path) for path in candidate_paths)

    def _check_phase1_final_panel_outputs(self) -> bool:
        try:
            if hasattr(self.config, 'get_phase1_paths'):
                phase1_paths = self.config.get_phase1_paths()
                selected_features_path = str(phase1_paths.get('selected_features_csv', '') or '').strip()
            else:
                selected_features_path = ''
        except Exception:
            selected_features_path = ''

        if not selected_features_path:
            selected_features_path = os.path.join('output', 'phase1', 'final', 'selected_features_final.csv')
        return os.path.exists(selected_features_path)

    def _check_autogluon_baseline(self) -> bool:
        """
        检查是否记录了 AutoGluon baseline 结果

        Returns:
            bool: 如果有 AutoGluon baseline 结果，返回 True
        """
        # 检查 phase1_state 中是否有 autogluon_results 字段
        if 'autogluon_results' in self.phase1_state:
            return True

        # 检查是否有 AutoGluon 结果文件（优先 canonical 路径）
        candidate_paths = [
            'output/phase1/artifacts/autogluon_training_results.json',
            'data/autogluon_training_results.json',
        ]
        for ag_results_path in candidate_paths:
            if os.path.exists(ag_results_path):
                return True

        return False

    # ========================================================================
    # 辅助方法: Phase 2 状态检查
    # ========================================================================

    def _check_ptot_search_log(self) -> bool:
        """
        检查 Phase 2 是否包含多目标搜索日志

        Returns:
            bool: 如果包含 PToT 搜索日志，返回 True
        """
        # 检查 phase2_state 中是否有 search_details 字段
        if 'search_details' in self.phase2_state:
            search_details = self.phase2_state['search_details']
            if 'layers' in search_details and len(search_details['layers']) > 0:
                return True

        # 检查 phase2_state 中是否有 final_result 字段
        if 'final_result' in self.phase2_state:
            final_result = self.phase2_state['final_result']
            if 'search_details' in final_result:
                search_details = final_result['search_details']
                if 'layers' in search_details and len(search_details['layers']) > 0:
                    return True

        latest_history = _load_json_if_exists(_find_latest_phase2_result())
        if 'search_details' in latest_history and latest_history['search_details'].get('layers'):
            return True
        if 'final_result' in latest_history:
            search_details = latest_history['final_result'].get('search_details', {})
            if search_details.get('layers'):
                return True

        return False

    def _check_winner_panel(self) -> bool:
        """
        检查 Phase 2 是否成功锁定了最终的 winner panel

        Returns:
            bool: 如果成功锁定了 winner panel，返回 True
        """
        # 检查 phase2_state 中是否有 final_result 字段
        if 'final_result' in self.phase2_state:
            final_result = self.phase2_state['final_result']
            # 检查是否有 features 和 perf 字段
            if 'features' in final_result and 'perf' in final_result:
                # 检查 features 是否非空
                if len(final_result['features']) > 0:
                    return True

        # 检查 phase2_state 中是否直接有 features 字段
        if 'features' in self.phase2_state and 'perf' in self.phase2_state:
            if len(self.phase2_state['features']) > 0:
                return True

        selected_features = self.phase2_state.get('selected_features', [])
        canonical_perf = self.phase2_state.get('scores', {}).get('f_perf')
        if selected_features and canonical_perf is not None:
            return True

        return False

    def _check_phase2_objective_shift_inputs(self) -> bool:
        """
        检查 Figure F / fig3 的正式输入是否齐备。

        该图在正式 Phase 3 中不应强依赖 PToT history log；只要 baseline
        score、winner score 与 Phase 2 search summary 存在，就可以生成。
        """
        if self._check_ptot_search_log():
            return True

        candidate_search_summary_paths = [
            str(self.phase2_state.get('phase2_search_summary_path', '') or '').strip(),
            str(self.phase2_state.get('search_summary_path', '') or '').strip(),
            os.path.join('output', 'artifacts', 'phase2_search_summary.json'),
            os.path.join('output', 'phase1', 'artifacts', 'phase2_search_summary.json'),
        ]
        has_search_summary = any(path and os.path.exists(path) for path in candidate_search_summary_paths)
        if not has_search_summary:
            return False

        candidate_phase1_score_paths = [
            os.path.join('output', 'artifacts', 'phase1_panel_scores.json'),
            os.path.join('output', 'phase1', 'artifacts', 'phase1_panel_scores.json'),
        ]
        has_phase1_scores = any(os.path.exists(path) for path in candidate_phase1_score_paths)
        if not has_phase1_scores:
            return False

        return self._check_winner_panel()

    # ========================================================================
    # 辅助方法: 从文件加载状态
    # ========================================================================

    @staticmethod
    def load_state_from_file(file_path: str) -> Dict:
        """
        从 JSON 文件加载状态

        Args:
            file_path: JSON 文件路径

        Returns:
            Dict: 状态字典

        Examples:
            >>> state = FigureRouter.load_state_from_file('output/phase1/artifacts/phase2_winner_scores.json')
            >>> 'final_result' in state
            True
        """
        if not os.path.exists(file_path):
            print(f"Warning: State file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error: Failed to load state from {file_path}: {e}")
            return {}


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("Testing Phase 3 Figure Router")
    print("="*80)

    # 测试 1: 从文件加载 Phase 2 状态
    print("\n[Test 1] Loading Phase 2 state from file...")
    phase2_state = FigureRouter.load_state_from_file(
        'output/phase1/artifacts/phase2_winner_scores.json'
    )

    if phase2_state:
        print(f"✅ Phase 2 state loaded successfully")
        print(f"   Keys: {list(phase2_state.keys())}")
    else:
        print("❌ Failed to load Phase 2 state")

    # 测试 2: 创建 FigureRouter 并决定需要生成的图表
    print("\n[Test 2] Creating FigureRouter and determining required figures...")

    # 创建一个简单的 config mock
    class MockConfig:
        def __init__(self):
            self.config = {}

    config = MockConfig()

    # 创建 router
    router = FigureRouter(
        config_manager=config,
        phase0_state={},
        phase1_state={},
        phase2_state=phase2_state
    )

    # 决定需要生成的图表
    tasks = router.determine_required_figures()

    print(f"\n✅ Test completed!")
    print(f"   Total tasks: {len(tasks)}")
    print(f"   Tasks: {tasks}")

    print("\n" + "="*80)
    print("All Tests Completed!")
    print("="*80)
