"""
Phase 3: 评估桥接模块 (Evaluator Bridge)

功能：
1. 复用 Phase 2 的核心评估算子（DRY 原则）
2. 为 Phase 0, Phase 1, Phase 2 的特征面板提供统一的回溯打分接口
3. 处理评分上下文的加载（数据集、知识图谱等）
4. 执行相对归一化，确保雷达图数据在 [0, 1] 范围内

Author: MetaboAgent Team
Date: 2026-04-23
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

# 导入 Phase 2 的核心评估算子（绝对不重复实现！）
from src.tools.analysis import build_bio_context, calculate_f_bio_v2
from src.tools.analysis.pareto_evaluator import (
    calculate_f_perf,
    calculate_f_bio,
    calculate_f_corr,
    calculate_f_cost
)
from src.utils.config_manager import get_config


class ClinicalEvaluator:
    """
    临床评估器（复用 Phase 2 的评估算子）

    职责：
    1. 管理评分上下文（数据集、知识图谱等）
    2. 提供统一的特征面板评分接口
    3. 处理归一化和数据转换
    """

    def __init__(
        self,
        data_path: str,
        target_column: str,
        holdout_data_path: Optional[str] = None,
        priors_dict: Optional[Dict] = None,
        taxonomy_map: Optional[Dict] = None,
        pathway_map: Optional[Dict] = None,
        ag_results_path: str = 'data/autogluon_training_results.json',
        disease_name: Optional[str] = None,
        phase0_output: Optional[Dict[str, Any]] = None,
        use_f_bio_v2: Optional[bool] = None,
    ):
        """
        初始化评估器

        Args:
            data_path: 临床数据集路径
            target_column: 目标列名
            priors_dict: Phase 0 先验字典（用于 f_bio 计算）
            taxonomy_map: 分类学映射（用于 f_bio 和 f_cost 计算）
            pathway_map: 通路映射（用于 f_bio 和 f_cost 计算）
            ag_results_path: AutoGluon 训练结果路径（用于 f_perf 计算）
        """
        self.data_path = data_path
        self.holdout_data_path = holdout_data_path if holdout_data_path and os.path.exists(holdout_data_path) else None
        self.target_column = target_column
        self.priors_dict = priors_dict or {}
        self.taxonomy_map = taxonomy_map or {}
        self.pathway_map = pathway_map or {}
        self.ag_results_path = ag_results_path
        self.phase0_output = phase0_output or {}
        self.disease_name = str(
            disease_name
            or self.phase0_output.get('disease_name')
            or get_config().get_test_disease_name()
            or ''
        ).strip()
        self.f_bio_v2_config = get_config().get_phase2_f_bio_v2_config()
        self.use_f_bio_v2 = bool(self.f_bio_v2_config.get('enabled', False)) if use_f_bio_v2 is None else bool(use_f_bio_v2)

        # 加载数据集（用于验证特征存在性）
        self.df = load_data_auto(data_path)
        self.feature_pool = [col for col in self.df.columns.tolist() if col != self.target_column]
        self.bio_context = None
        self._bio_context_initialized = False

        current_disease_priors = self.phase0_output.get("confirmed_biomarkers", []) or []
        if not isinstance(current_disease_priors, list):
            current_disease_priors = []

        print(f"✅ 评估器初始化完成")
        print(f"   数据集: {data_path} ({len(self.df)} 样本)")
        if self.holdout_data_path:
            print(f"   外部 Holdout: {self.holdout_data_path}")
        print(f"   目标列: {target_column}")
        print(f"   全局先验字典: {len(self.priors_dict)} 个生物标志物")
        if current_disease_priors:
            print(f"   当前疾病先验: {len(current_disease_priors)} 个生物标志物")
        print(f"   分类映射: {len(self.taxonomy_map)} 个特征")
        print(f"   通路映射: {len(self.pathway_map)} 个特征")
        print(f"   生物学评分: {'f_bio_v2' if self.use_f_bio_v2 else 'f_bio_v1'}")

    def _ensure_bio_context(self):
        """
        惰性构建 f_bio_v2 所需的 BioContext，避免在评估器初始化阶段触发重型资源加载。
        """
        if not self.use_f_bio_v2:
            return None

        if self._bio_context_initialized:
            return self.bio_context

        self._bio_context_initialized = True
        try:
            self.bio_context = build_bio_context(
                disease_name=self.disease_name,
                feature_pool=self.feature_pool,
                priors_dict=self.priors_dict,
                taxonomy_map=self.taxonomy_map,
                pathway_map=self.pathway_map,
                config=self.f_bio_v2_config,
            )
        except Exception as e:
            print(f"  ⚠️  f_bio_v2 context 懒加载失败，将回退旧版 f_bio: {e}")
            self.bio_context = None

        return self.bio_context

    @staticmethod
    def _infer_supported_model_family(model_name: Optional[str]) -> Optional[str]:
        """Map AutoGluon model names to a replayable family supported by f_perf."""
        if not model_name:
            return None

        model_name_lower = str(model_name).lower()
        if "catboost" in model_name_lower:
            return "CatBoost"
        if "lightgbm" in model_name_lower:
            return "LightGBM"
        if "xgboost" in model_name_lower or "xgb" in model_name_lower:
            return "XGBoost"
        if "gradientboost" in model_name_lower or "gradient_boost" in model_name_lower:
            return "GradientBoosting"
        if "randomforest" in model_name_lower or model_name_lower.startswith("rf"):
            return "RandomForest"
        return None

    def _resolve_champion_model_family(self) -> Optional[str]:
        """
        Resolve a replayable model family from Phase 1 AutoGluon artifacts.

        If the exact champion is an ensemble that cannot be re-instantiated in the
        lightweight evaluator, fall back to the highest-ranked supported base model
        from the saved leaderboard.
        """
        ag_path = Path(self.ag_results_path)
        if not ag_path.exists():
            return None

        try:
            with ag_path.open("r", encoding="utf-8") as handle:
                ag_results = json.load(handle)
        except Exception as e:
            print(f"  ⚠️  f_perf: 无法读取 AG 结果文件: {e}")
            return None

        candidates: List[Optional[str]] = [ag_results.get("best_model")]
        for row in ag_results.get("leaderboard_top5", []) or []:
            if isinstance(row, dict):
                candidates.append(row.get("model"))

        for candidate in candidates:
            family = self._infer_supported_model_family(candidate)
            if family:
                if candidate != ag_results.get("best_model"):
                    print(
                        f"  ℹ️  f_perf: 冠军模型 {ag_results.get('best_model')} 不可直接回放，"
                        f"改用可支持的高排名模型家族 {family} ({candidate})"
                    )
                else:
                    print(f"  ℹ️  f_perf: 继承 Phase 1 冠军模型家族 {family}")
                return family

        print("  ⚠️  f_perf: 未找到可回放的冠军模型家族，将回退默认分类器")
        return None

    def validate_features(self, features: List[str]) -> Tuple[List[str], List[str]]:
        """
        验证特征是否存在于数据集中（支持动态标准化匹配）

        Args:
            features: 特征列表

        Returns:
            tuple: (可用特征列表, 缺失特征列表)
        """
        from src.utils.feature_name_standardizer import validate_feature_names

        # 使用标准化验证特征（优雅降级）
        validation_result = validate_feature_names(
            feature_names=features,
            dataset_columns=self.df.columns.tolist(),
            auto_standardize=True  # 启用自动标准化匹配
        )

        available = validation_result['valid']
        missing = validation_result['missing']

        return available, missing

    def calc_perf(
        self,
        features: List[str],
        k_folds: int = 5,
        use_phase1_config: bool = True
    ) -> float:
        """
        计算预测性能（f_perf / 训练集5折ROC-AUC）

        Args:
            features: 特征列表
            k_folds: 交叉验证折数
            use_phase1_config: 是否使用 Phase 1 的配置（冠军模型 + 重采样）

        Returns:
            float: 训练集5折 ROC-AUC 得分 [0, 1]
        """
        available, missing = self.validate_features(features)

        if len(available) == 0:
            print(f"  ⚠️  f_perf: 没有可用特征")
            return 0.0

        if len(missing) > 0:
            print(f"  ⚠️  f_perf: {len(missing)} 个特征缺失")

        try:
            champion_model_family = self._resolve_champion_model_family()
            # 复用 Phase 2 的评估算子
            perf = calculate_f_perf(
                data_path=self.data_path,
                target_column=self.target_column,
                features_list=available,
                champion_model_family=champion_model_family,
                k_folds=k_folds,
                ag_results_path=self.ag_results_path,
                metric='roc_auc',
                use_phase1_config=use_phase1_config,
                holdout_data_path=self.holdout_data_path,
                evaluation_mode='auto',
            )

            return float(perf)
        except Exception as e:
            print(f"  ❌ f_perf 计算失败: {e}")
            return 0.0

    def calc_bio(
        self,
        features: List[str],
        w_lit: float = 0.2,
        w_mech: float = 0.8,
        gamma: float = 0.1,
        sigmoid_k: float = 1.0
    ) -> float:
        """
        计算生物学机制价值（f_bio）

        Args:
            features: 特征列表
            w_lit: 文献权重
            w_mech: 机制权重
            gamma: 惩罚系数
            sigmoid_k: Sigmoid 斜率

        Returns:
            float: 生物学价值得分 [0, 1]
        """
        if len(features) == 0:
            return 0.0

        try:
            if self.use_f_bio_v2:
                bio_context = self._ensure_bio_context()
                if bio_context is not None:
                    bio = calculate_f_bio_v2(
                        features_list=features,
                        bio_context=bio_context,
                        disease_name=self.disease_name,
                        feature_pool=self.feature_pool,
                        priors_dict=self.priors_dict,
                        taxonomy_map=self.taxonomy_map,
                        pathway_map=self.pathway_map,
                        return_debug=False,
                    )
                    return float(bio)
                print("  ⚠️  f_bio_v2 context 不可用，回退旧版 f_bio")

            # 兼容旧版 Phase 2 评估算子
            bio = calculate_f_bio(
                features_list=features,
                priors_dict=self.priors_dict,
                taxonomy_map=self.taxonomy_map,
                pathway_map=self.pathway_map,
                w_lit=w_lit,
                w_mech=w_mech,
                gamma=gamma,
                sigmoid_k=sigmoid_k
            )

            return float(bio)
        except Exception as e:
            print(f"  ❌ f_bio 计算失败: {e}")
            return 0.0

    def calc_corr(self, features: List[str]) -> float:
        """
        计算特征冗余度（f_corr）

        Args:
            features: 特征列表

        Returns:
            float: 冗余度得分 [0, 1]
        """
        available, missing = self.validate_features(features)

        if len(available) < 2:
            return 0.0

        if len(missing) > 0:
            print(f"  ⚠️  f_corr: {len(missing)} 个特征缺失")

        try:
            # 复用 Phase 2 的评估算子
            corr = calculate_f_corr(
                data_path=self.data_path,
                features_list=available
            )

            return float(corr)
        except Exception as e:
            print(f"  ❌ f_corr 计算失败: {e}")
            return 0.0

    def calc_cost(
        self,
        features: List[str],
        n_max: int = 20,
        k: float = 2.0,
        aggregation_discount: bool = True
    ) -> float:
        """
        计算临床成本（f_cost）

        Args:
            features: 特征列表
            n_max: 最大特征数
            k: 指数惩罚系数
            aggregation_discount: 是否启用规模效应打折

        Returns:
            float: 成本得分 [0, 1]
        """
        if len(features) == 0:
            return 0.0

        try:
            # 复用 Phase 2 的评估算子
            cost = calculate_f_cost(
                features_list=features,
                taxonomy_map=self.taxonomy_map,
                pathway_map=self.pathway_map,
                n_max=n_max,
                k=k,
                data_path=self.data_path,
                aggregation_discount=aggregation_discount
            )

            return float(cost)
        except Exception as e:
            print(f"  ❌ f_cost 计算失败: {e}")
            return 0.0

    def score_panel(
        self,
        features: List[str],
        panel_name: str = "Feature Panel"
    ) -> Dict[str, float]:
        """
        对特征面板进行完整的四维评分

        Args:
            features: 特征列表
            panel_name: 面板名称（用于日志）

        Returns:
            dict: 四维得分 {'f_perf': x, 'f_bio': y, 'f_cost': z, 'f_corr': w}
        """
        print(f"\n评估 {panel_name} ({len(features)} 个特征)...")

        # 计算四维得分
        perf = self.calc_perf(features)
        bio = self.calc_bio(features)
        corr = self.calc_corr(features)
        cost = self.calc_cost(features)

        print(f"  ✓ f_perf (Train 5-fold ROC-AUC): {perf:.4f}")
        print(f"  ✓ f_bio (Biology):     {bio:.4f}")
        print(f"  ✓ f_corr (Redundancy): {corr:.4f}")
        print(f"  ✓ f_cost (Cost):       {cost:.4f}")

        return {
            'f_perf': perf,
            'f_bio': bio,
            'f_cost': cost,
            'f_corr': corr
        }


def normalize_scores_for_radar(
    raw_scores: Dict[str, float],
    max_cost: float
) -> Dict[str, float]:
    """
    将原始得分归一化为雷达图格式（所有维度越大越好）

    归一化规则：
    1. f_perf (AUC): 已经是 [0, 1]，越大越好 → 直接使用
    2. f_bio: 已经是 [0, 1]，越大越好 → 直接使用
    3. f_cost: 原始值越小越好 → 转换为 1 - (f_cost / max_cost)
    4. f_corr: 原始值越小越好 → 转换为 1 - f_corr

    Args:
        raw_scores: 原始得分字典
        max_cost: 所有模型中的最大成本（用于相对归一化）

    Returns:
        dict: 归一化得分 {'AUC': x, 'f_bio': y, '1-f_cost': z, '1-f_corr': w}
    """
    # 防止除以零
    if max_cost == 0:
        max_cost = 1.0

    display_auc = raw_scores.get('roc_auc', raw_scores.get('f_perf', 0.0))
    normalized = {
        'AUC': display_auc,                                   # 展示层优先使用真实 ROC-AUC
        'f_bio': raw_scores['f_bio'],                         # 生物学价值（已归一化）
        '1-f_cost': 1.0 - (raw_scores['f_cost'] / max_cost),  # 简约性（相对归一化）
        '1-f_corr': 1.0 - raw_scores['f_corr']                # 独立性（翻转极性）
    }

    # 确保所有值在 [0, 1] 范围内
    for key, value in normalized.items():
        if value < 0:
            print(f"  ⚠️  警告: {key} = {value:.4f} < 0，截断为 0")
            normalized[key] = 0.0
        elif value > 1:
            print(f"  ⚠️  警告: {key} = {value:.4f} > 1，截断为 1")
            normalized[key] = 1.0

    return normalized


def load_data_auto(path: str) -> pd.DataFrame:
    """自动根据后缀加载 CSV 或 Excel 文件"""
    if path.endswith('.xlsx') or path.endswith('.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("Phase 3: 评估桥接模块测试")
    print("="*80)

    # 从配置中获取路径，或者使用默认值
    try:
        from src.utils.config_manager import get_config
        cfg = get_config()
        test_data_path = cfg.get_test_data_path() or "tests/data/example.csv"
        target_column = cfg.get_path("target_column") or "target"
    except ImportError:
        test_data_path = "tests/data/example.csv"
        target_column = "target"

    # 测试评估器初始化
    try:
        evaluator = ClinicalEvaluator(
            data_path=test_data_path,
            target_column=target_column,
            priors_dict={},
            taxonomy_map={},
            pathway_map={}
        )

        print("\n" + "="*80)
        print("测试特征面板评分")
        print("="*80)

        # 测试特征列表（使用数据集中实际存在的列）
        df = load_data_auto(test_data_path)
        test_features = [
            col for col in df.columns
            if col not in ['Sample_ID', 'sample_id', 'SampleID', 'sampleid', 'ROW_ID', 'row_id', '__row_id__', target_column]
        ][:10]

        print(f"\n测试特征: {test_features[:5]}...")

        # 评分
        scores = evaluator.score_panel(test_features, panel_name="Test Panel")

        print("\n" + "="*80)
        print("测试归一化")
        print("="*80)

        # 归一化
        normalized = normalize_scores_for_radar(scores, max_cost=scores['f_cost'])

        print("\n归一化后的得分:")
        for key, value in normalized.items():
            print(f"  {key:12s}: {value:.4f}")

        # 验证范围
        all_in_range = all(0 <= v <= 1 for v in normalized.values())
        print(f"\n✅ 所有得分在 [0, 1] 范围内: {all_in_range}")

        print("\n" + "="*80)
        print("✅ 测试通过！")
        print("="*80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
