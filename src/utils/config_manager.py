"""
配置管理器 (Configuration Manager)

功能：
1. 加载和解析 config.yaml
2. 提供全局配置访问接口
3. 支持配置验证和默认值

Author: MetaboAgent Team
Date: 2026-04-23
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigManager:
    """全局配置管理器"""

    DEFAULT_SCENARIO_RISK_THRESHOLDS = [0.10, 0.15, 0.20]
    DEFAULT_SCENARIO_RESOURCE_ASSUMPTIONS = {
        'screen_population_size': 1000,
        'panel_test_cost_per_person': 0.0,
        'confirmatory_test_cost_per_person': 0.0,
        'baseline_strategy': 'phase1_panel',
    }

    @staticmethod
    def _get_llm_model_override() -> str:
        """
        Resolve an environment-driven LLM model override for runtime workflows.
        """
        return str(
            os.getenv("METABOAGENT_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or ""
        ).strip()

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径（默认为项目根目录的 config.yaml）
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件

        Returns:
            dict: 配置字典
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_path}\n"
                f"请确保在项目根目录下创建 config.yaml 文件"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证必需的配置项
        self._validate_config(config)

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置文件的完整性

        Args:
            config: 配置字典
        """
        required_keys = ['paths', 'phase0', 'phase1', 'evaluation', 'visualization', 'models']

        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必需的键: {key}")

        # 验证路径配置
        if 'phase1_selected_features' not in config['paths']:
            raise ValueError("配置文件缺少 paths.phase1_selected_features")

        if 'phase2_winner_panel' not in config['paths']:
            raise ValueError("配置文件缺少 paths.phase2_winner_panel")

        # 验证评估配置
        if 'active_scenario' not in config['evaluation']:
            raise ValueError("配置文件缺少 evaluation.active_scenario")
        if 'scenarios' not in config['evaluation'] or not isinstance(config['evaluation']['scenarios'], dict) or not config['evaluation']['scenarios']:
            raise ValueError("配置文件缺少 evaluation.scenarios 或其内容为空")
        if config['evaluation']['active_scenario'] not in config['evaluation']['scenarios']:
            raise ValueError("evaluation.active_scenario 必须出现在 evaluation.scenarios 中")

        for scenario_name, scenario_config in config['evaluation']['scenarios'].items():
            if not isinstance(scenario_config, dict):
                raise ValueError(f"evaluation.scenarios.{scenario_name} 必须为字典")
            if 'weights' not in scenario_config or not isinstance(scenario_config['weights'], dict):
                raise ValueError(f"配置文件缺少 evaluation.scenarios.{scenario_name}.weights")

        # 验证 Phase 0 配置
        phase0_required_sections = ['defaults', 'paths', 'pubmed', 'llm']
        for section in phase0_required_sections:
            if section not in config['phase0']:
                raise ValueError(f"配置文件缺少 phase0.{section}")

        phase0_required_paths = ['disease_map', 'pathway_map', 'cache_path', 'output_dir', 'output_latest']
        for key in phase0_required_paths:
            if key not in config['phase0']['paths']:
                raise ValueError(f"配置文件缺少 phase0.paths.{key}")

        # 验证 Phase 1 配置
        phase1_required_sections = ['io_policy', 'paths']
        for section in phase1_required_sections:
            if section not in config['phase1']:
                raise ValueError(f"配置文件缺少 phase1.{section}")

        phase1_required_paths = [
            'legacy_selected_features',
            'legacy_artifacts_dir',
            'autogluon_results',
            'output_root',
            'intermediate_root',
            'intermediate_latest_dir',
            'stability_temp_dir',
            'artifacts_dir',
            'autogluon_models_dir',
            'final_dir',
            'selected_features_csv',
            'panel_scores_json',
            'phase2_winner_scores_json',
            'engineered_feature_rules',
            'engineered_feature_registry',
            'intermediate_legacy_dirs',
        ]
        for key in phase1_required_paths:
            if key not in config['phase1']['paths']:
                raise ValueError(f"配置文件缺少 phase1.paths.{key}")

        # 验证 Phase 2 配置（可选，但如果存在则校验关键字段）
        if 'phase2' in config and 'f_bio_v2' in config['phase2']:
            fbio_v2_config = config['phase2']['f_bio_v2']
            required_fbio_v2_keys = ['enabled', 'mode_selection', 'gamma_topo', 'delta_panel', 'weights']
            for key in required_fbio_v2_keys:
                if key not in fbio_v2_config:
                    raise ValueError(f"配置文件缺少 phase2.f_bio_v2.{key}")

        # 验证 Memory 配置（可选，但如果存在则校验关键字段）
        if 'memory' in config:
            memory_required_keys = [
                'enabled',
                'read_enabled',
                'write_enabled',
                'storage_dir',
                'case_store_path',
                'semantic_store_path',
            ]
            for key in memory_required_keys:
                if key not in config['memory']:
                    raise ValueError(f"配置文件缺少 memory.{key}")

    def get_path(self, key: str) -> str:
        """
        获取路径配置

        Args:
            key: 路径键名（如 'phase1_selected_features'）

        Returns:
            str: 路径字符串
        """
        return self.config['paths'].get(key, '')

    def get_active_scenario(self) -> str:
        """
        获取当前激活的临床场景

        Returns:
            str: 场景名称（如 'primary_care'）
        """
        return self.config['evaluation']['active_scenario']

    def _resolve_scenario_name(self, scenario: Optional[str] = None) -> str:
        """解析场景名称，如未指定则回退到当前激活场景。"""
        resolved = str(scenario or self.get_active_scenario()).strip()
        scenarios = self.config['evaluation']['scenarios']
        if resolved not in scenarios:
            raise ValueError(
                f"未知的场景: {resolved}\n"
                f"可用场景: {list(scenarios.keys())}"
            )
        return resolved

    def _normalize_thresholds(self, thresholds: Any) -> List[float]:
        """标准化阈值列表，过滤非法值并保持升序去重。"""
        normalized: List[float] = []
        for value in thresholds or []:
            try:
                threshold = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 < threshold < 1.0:
                normalized.append(threshold)
        unique_sorted = sorted(set(normalized))
        return unique_sorted or list(self.DEFAULT_SCENARIO_RISK_THRESHOLDS)

    def _normalize_resource_assumptions(self, payload: Any) -> Dict[str, Any]:
        """为 resource assumptions 填充默认值并做基本类型归一化。"""
        normalized = dict(self.DEFAULT_SCENARIO_RESOURCE_ASSUMPTIONS)
        if isinstance(payload, dict):
            normalized.update(payload)

        try:
            normalized['screen_population_size'] = max(1, int(normalized.get('screen_population_size', 1000)))
        except (TypeError, ValueError):
            normalized['screen_population_size'] = self.DEFAULT_SCENARIO_RESOURCE_ASSUMPTIONS['screen_population_size']

        for key in ('panel_test_cost_per_person', 'confirmatory_test_cost_per_person'):
            try:
                normalized[key] = float(normalized.get(key, 0.0))
            except (TypeError, ValueError):
                normalized[key] = float(self.DEFAULT_SCENARIO_RESOURCE_ASSUMPTIONS[key])

        normalized['baseline_strategy'] = str(
            normalized.get('baseline_strategy', self.DEFAULT_SCENARIO_RESOURCE_ASSUMPTIONS['baseline_strategy'])
        )
        return normalized

    def get_scenario_definition(self, scenario: Optional[str] = None) -> Dict[str, Any]:
        """
        获取完整的临床场景定义，并补齐 clinical utility 默认字段。

        Args:
            scenario: 场景名称（如果为 None，则使用当前激活的场景）

        Returns:
            dict: 归一化后的场景定义
        """
        scenario_name = self._resolve_scenario_name(scenario)
        raw_definition = dict(self.config['evaluation']['scenarios'][scenario_name])
        normalized = dict(raw_definition)
        normalized['key'] = scenario_name
        normalized['name'] = str(normalized.get('name', scenario_name))
        normalized['description'] = str(normalized.get('description', ''))
        normalized['intended_use'] = str(normalized.get('intended_use', 'risk stratification'))
        normalized['clinical_action'] = str(
            normalized.get('clinical_action', 'flag high-risk individuals for confirmatory assessment')
        )
        normalized['weights'] = dict(normalized.get('weights', {}))
        normalized['risk_thresholds'] = self._normalize_thresholds(normalized.get('risk_thresholds'))

        default_threshold = normalized.get('default_action_threshold')
        try:
            default_threshold_float = float(default_threshold)
        except (TypeError, ValueError):
            default_threshold_float = float(normalized['risk_thresholds'][0])
        if not (0.0 < default_threshold_float < 1.0):
            default_threshold_float = float(normalized['risk_thresholds'][0])
        normalized['default_action_threshold'] = default_threshold_float
        normalized['study_design'] = str(normalized.get('study_design', 'case_control') or 'case_control').strip().lower()
        normalized['use_observed_prevalence'] = bool(normalized.get('use_observed_prevalence', True))
        target_prevalence = normalized.get('target_prevalence')
        try:
            target_prevalence_float = float(target_prevalence)
        except (TypeError, ValueError):
            target_prevalence_float = None
        if target_prevalence_float is not None and not (0.0 < target_prevalence_float < 1.0):
            target_prevalence_float = None
        normalized['target_prevalence'] = target_prevalence_float
        normalized['prevalence_source'] = str(normalized.get('prevalence_source', ''))
        normalized['prevalence_reference_population'] = str(
            normalized.get('prevalence_reference_population', '')
        )
        normalized['report_ece'] = bool(normalized.get('report_ece', False))
        normalized['resource_assumptions'] = self._normalize_resource_assumptions(
            normalized.get('resource_assumptions')
        )
        return normalized

    def get_scenario_weights(self, scenario: Optional[str] = None) -> Dict[str, float]:
        """
        获取指定场景的权重配置

        Args:
            scenario: 场景名称（如果为 None，则使用当前激活的场景）

        Returns:
            dict: 权重字典 {'f_perf': 0.35, 'f_bio': 0.40, ...}
        """
        return dict(self.get_scenario_definition(scenario).get('weights', {}))

    def get_scenario_thresholds(self, scenario: Optional[str] = None) -> List[float]:
        """
        获取指定场景的预设风险阈值列表。

        Args:
            scenario: 场景名称（如果为 None，则使用当前激活的场景）

        Returns:
            List[float]: 升序去重后的风险阈值列表
        """
        return list(self.get_scenario_definition(scenario).get('risk_thresholds', []))

    def get_scenario_default_action_threshold(self, scenario: Optional[str] = None) -> float:
        """
        获取指定场景的默认行动阈值。

        Args:
            scenario: 场景名称（如果为 None，则使用当前激活的场景）

        Returns:
            float: 默认行动阈值
        """
        return float(self.get_scenario_definition(scenario).get('default_action_threshold', 0.0))

    def get_scenario_resource_assumptions(self, scenario: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定场景的资源影响模拟假设。

        Args:
            scenario: 场景名称（如果为 None，则使用当前激活的场景）

        Returns:
            dict: 归一化后的资源假设
        """
        return dict(self.get_scenario_definition(scenario).get('resource_assumptions', {}))

    def get_visualization_config(self, plot_type: str) -> Dict[str, Any]:
        """
        获取可视化配置

        Args:
            plot_type: 图表类型（'radar_plot' 或 'pareto_plot'）

        Returns:
            dict: 可视化配置
        """
        return self.config['visualization'].get(plot_type, {})

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        获取模型配置

        Args:
            model_type: 模型类型（'baselines' 或 'winner'）

        Returns:
            dict: 模型配置
        """
        return self.config['models'].get(model_type, {})

    def get_phase0_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 总配置

        Returns:
            dict: Phase 0 配置字典
        """
        return self.config.get('phase0', {})

    def get_phase0_defaults(self) -> Dict[str, Any]:
        """
        获取 Phase 0 默认运行参数

        Returns:
            dict: 默认参数字典
        """
        return self.get_phase0_config().get('defaults', {})

    def get_phase0_paths(self) -> Dict[str, str]:
        """
        获取 Phase 0 路径配置

        Returns:
            dict: 路径配置字典
        """
        paths = dict(self.get_phase0_config().get('paths', {}))

        runtime_root = str(os.environ.get("METABOAGENT_RUNTIME_ROOT", "") or "").strip()
        if runtime_root:
            runtime_root_path = Path(runtime_root)
            phase0_root = runtime_root_path / "phase0"
            paths["output_dir"] = str(phase0_root / "outputs")
            paths["output_latest"] = str(phase0_root / "phase0_output_latest.json")
            paths["evidence_dir"] = str(phase0_root / "evidence")
            paths["full_result_dir"] = str(phase0_root / "evidence")

        return paths

    def get_phase0_path(self, key: str) -> str:
        """
        获取 Phase 0 路径配置项

        Args:
            key: 路径键名

        Returns:
            str: 路径字符串
        """
        return self.get_phase0_paths().get(key, '')

    def get_phase0_pubmed_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 的 PubMed 配置

        Returns:
            dict: PubMed 配置
        """
        return self.get_phase0_config().get('pubmed', {})

    def get_phase0_llm_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 的 LLM 配置

        Returns:
            dict: LLM 配置
        """
        llm_config = dict(self.get_phase0_config().get('llm', {}))
        model_override = self._get_llm_model_override()
        if model_override:
            llm_config['model'] = model_override
            llm_config['extraction_model'] = model_override
            llm_config['scoring_model'] = model_override
        return llm_config

    def get_phase0_stage1_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 Stage 1 粗筛配置

        Returns:
            dict: Stage 1 配置
        """
        return self.get_phase0_config().get('stage1', {})

    def get_phase0_literature_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 文献检索配置

        Returns:
            dict: Literature 配置
        """
        return self.get_phase0_config().get('literature', {})

    def get_phase0_scoring_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 评分配置

        Returns:
            dict: Scoring 配置
        """
        phase0_scoring = self.get_phase0_config().get('scoring', {})
        if isinstance(phase0_scoring, dict) and phase0_scoring:
            return dict(phase0_scoring)
        # Backward compatibility for legacy configs that incorrectly stored
        # Phase 0 scoring knobs under `phase4.scoring`.
        legacy_scoring = self.get_phase4_config().get('scoring', {})
        return dict(legacy_scoring) if isinstance(legacy_scoring, dict) else {}

    def get_phase0_weights_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 动态权重配置

        Returns:
            dict: Weights 配置
        """
        phase0_weights = self.get_phase0_config().get('weights', {})
        if isinstance(phase0_weights, dict) and phase0_weights:
            return dict(phase0_weights)
        # Backward compatibility for legacy configs that incorrectly stored
        # Phase 0 dynamic evidence weights under `phase4.weights`.
        legacy_weights = self.get_phase4_config().get('weights', {})
        return dict(legacy_weights) if isinstance(legacy_weights, dict) else {}

    def get_phase0_pathway_retrieval_config(self) -> Dict[str, Any]:
        """
        获取 Phase 0 disease pathway retrieval 配置

        Returns:
            dict: pathway retrieval 配置
        """
        pathway_config = dict(self.get_phase0_config().get('pathway_retrieval', {}))
        model_override = self._get_llm_model_override()
        if model_override:
            pathway_config['llm_rerank_model'] = model_override
        return pathway_config

    def get_phase1_config(self) -> Dict[str, Any]:
        """
        获取 Phase 1 总配置

        Returns:
            dict: Phase 1 配置字典
        """
        return self.config.get('phase1', {})

    def get_phase1_paths(self) -> Dict[str, Any]:
        """
        获取 Phase 1 路径配置

        Returns:
            dict: 路径配置字典
        """
        paths = dict(self.get_phase1_config().get('paths', {}))

        runtime_root = str(os.environ.get("METABOAGENT_RUNTIME_ROOT", "") or "").strip()
        if runtime_root:
            runtime_root_path = Path(runtime_root)
            phase1_root = runtime_root_path / "phase1"
            intermediate_root = phase1_root / "intermediate"
            intermediate_latest_dir = intermediate_root / "latest"
            artifacts_dir = phase1_root / "artifacts"
            final_dir = phase1_root / "final"
            legacy_root = phase1_root / "legacy"

            paths["legacy_selected_features"] = str(legacy_root / "selected_features_final.csv")
            paths["legacy_artifacts_dir"] = str(legacy_root / "artifacts")
            paths["autogluon_results"] = str(legacy_root / "autogluon_training_results.json")
            paths["output_root"] = str(phase1_root)
            paths["intermediate_root"] = str(intermediate_root)
            paths["intermediate_latest_dir"] = str(intermediate_latest_dir)
            paths["stability_temp_dir"] = str(intermediate_latest_dir / "stability_logs")
            paths["artifacts_dir"] = str(artifacts_dir)
            paths["autogluon_models_dir"] = str(artifacts_dir / "models" / "autogluon")
            paths["final_dir"] = str(final_dir)
            paths["selected_features_csv"] = str(final_dir / "selected_features_final.csv")
            paths["panel_scores_json"] = str(artifacts_dir / "phase1_panel_scores.json")
            paths["phase2_winner_scores_json"] = str(artifacts_dir / "phase2_winner_scores.json")
            paths["engineered_feature_registry"] = str(
                intermediate_latest_dir / "engineered" / "engineered_feature_registry.json"
            )

        return paths

    def get_phase1_path(self, key: str) -> str:
        """
        获取 Phase 1 路径配置项

        Args:
            key: 路径键名

        Returns:
            str: 路径字符串
        """
        return self.get_phase1_paths().get(key, '')

    def get_phase1_io_policy(self) -> Dict[str, Any]:
        """
        获取 Phase 1 的 IO 策略配置

        Returns:
            dict: IO 策略配置
        """
        io_policy = dict(self.get_phase1_config().get('io_policy', {}))
        runtime_root = str(os.environ.get("METABOAGENT_RUNTIME_ROOT", "") or "").strip()
        if runtime_root:
            # Runtime-scoped evaluation runs should not mirror back into shared
            # legacy paths, otherwise concurrent repeats can overwrite each other.
            io_policy["dual_write_enabled"] = False
        return io_policy

    def get_phase1_engineered_features_config(self) -> Dict[str, Any]:
        """
        获取 Phase 1 工程特征配置

        Returns:
            dict: 工程特征配置
        """
        return self.get_phase1_config().get('engineered_features', {})

    def get_phase1_engineered_feature_rules_path(self) -> str:
        """
        获取工程特征规则文件路径

        Returns:
            str: 规则文件路径
        """
        return self.get_phase1_path('engineered_feature_rules')

    def get_phase1_engineered_feature_registry_path(self) -> str:
        """
        获取工程特征注册表输出路径

        Returns:
            str: 注册表输出路径
        """
        return self.get_phase1_path('engineered_feature_registry')

    def get_phase2_config(self) -> Dict[str, Any]:
        """
        获取 Phase 2 总配置

        Returns:
            dict: Phase 2 配置字典
        """
        return self.config.get('phase2', {})

    def get_phase4_config(self) -> Dict[str, Any]:
        """
        获取 Phase 4 配置

        Returns:
            dict: Phase 4 配置
        """
        return self.config.get('phase4', {})

    def get_phase4_llm_config(self) -> Dict[str, Any]:
        """
        获取 Phase 4 的 LLM 配置

        Returns:
            dict: LLM 配置
        """
        return self.get_phase4_config().get('llm', {})

    def get_phase2_f_bio_v2_config(self) -> Dict[str, Any]:
        """
        获取 Phase 2 f_bio v2 配置

        Returns:
            dict: f_bio v2 配置字典
        """
        return self.get_phase2_config().get('f_bio_v2', {})

    def get_memory_config(self) -> Dict[str, Any]:
        """
        获取长期记忆总配置

        Returns:
            dict: Memory 配置字典
        """
        memory_cfg = dict(self.config.get('memory', {}) or {})

        def _parse_env_bool(name: str) -> Optional[bool]:
            raw_value = os.getenv(name)
            if raw_value is None:
                return None
            normalized = str(raw_value).strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return None

        env_overrides = {
            "enabled": _parse_env_bool("METABOAGENT_MEMORY_ENABLED"),
            "read_enabled": _parse_env_bool("METABOAGENT_MEMORY_READ_ENABLED"),
            "write_enabled": _parse_env_bool("METABOAGENT_MEMORY_WRITE_ENABLED"),
        }
        for key, value in env_overrides.items():
            if value is not None:
                memory_cfg[key] = value
        return memory_cfg

    def get_memory_path(self, key: str) -> str:
        """
        获取长期记忆路径配置项

        Args:
            key: 路径键名

        Returns:
            str: 路径字符串
        """
        return str(self.get_memory_config().get(key, ''))

    def get_phase1_memory_config(self) -> Dict[str, Any]:
        """
        获取 Phase 1 记忆配置

        Returns:
            dict: Phase 1 记忆配置
        """
        return self.get_memory_config().get('phase1', {})

    def get_phase2_memory_config(self) -> Dict[str, Any]:
        """
        获取 Phase 2 记忆配置

        Returns:
            dict: Phase 2 记忆配置
        """
        return self.get_memory_config().get('phase2', {})

    def get_figures_dir(self) -> str:
        """
        获取图表输出目录

        Returns:
            str: 输出目录路径
        """
        return self.config['paths']['figures_dir']

    def get_test_data_path(self) -> str:
        """
        获取测试数据路径

        Returns:
            str: 测试数据路径
        """
        return self.config['paths'].get('test_data_path', '')

    def get_target_column(self) -> str:
        """
        获取目标列名

        Returns:
            str: 目标列名
        """
        return self.config['paths'].get('target_column', 'Group')

    def get_test_disease_name(self) -> str:
        """Return the optional configured disease name without inventing one."""
        paths = self.config.get('paths', {})
        return str(paths.get('test_disease_name') or paths.get('default_disease_name') or '').strip()

    def get_default_disease_name(self) -> str:
        """Backward-compatible alias for the optional disease name."""
        return self.get_test_disease_name()

    def reload(self) -> None:
        """重新加载配置文件"""
        self.config = self._load_config()

    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConfigManager(config_path='{self.config_path}')"


# ============================================================================
# 全局配置实例
# ============================================================================
_global_config: Optional[ConfigManager] = None


def get_config(config_path: str = "config.yaml") -> ConfigManager:
    """
    获取全局配置实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigManager: 配置管理器实例
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigManager(config_path)

    return _global_config
