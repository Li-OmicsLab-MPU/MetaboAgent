"""
Pareto Multi-Objective Feature Search Evaluator

Phase 2 核心评估函数库，用于基于 Pareto 优化的多目标特征搜索。
此模块提供四个静态评估函数：
- f_perf: 模型预测性能评估
- f_size: 特征集大小评估
- f_interp: 特征可解释性评估
- f_stab: 特征稳定性评估

设计理念：
1. 动态继承 Phase 1 的最优算法基因
2. 毫秒级极速评估，支持 ToT 搜索
3. 避免 LLM 在沙盒中手写评估代码导致的 Token 爆炸

Author: MetaboAgent Team
Date: 2026-04-16
"""

import pandas as pd
import numpy as np
import json
import os
import re
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import List, Dict, Any, Optional, Union, Tuple, Sequence
from src.utils.config_manager import get_config

# 导入原生分类器
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# CRITICAL: 必须使用 imblearn.pipeline.Pipeline，不能使用 sklearn.pipeline.Pipeline
# 因为 sklearn 的 Pipeline 不支持在训练时对样本和标签同时进行重采样改变形状
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
    from imblearn.combine import SMOTEENN, SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("Warning: imbalanced-learn not installed. Resampling strategies will be disabled.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    TabularPredictor = None
    HAS_AUTOGLUON = False


# Keep RF single-threaded during repeated Phase 2 evaluations to avoid
# oversubscription across CV folds and search iterations.
PHASE2_RF_N_JOBS = 1


# ============================================================================
# 辅助函数：生物学先验评分（双轨制）
# ============================================================================


def _load_feature_provenance_index() -> Dict[str, Dict[str, Any]]:
    """Best-effort load for Phase 1 feature provenance."""
    provenance_index: Dict[str, Dict[str, Any]] = {}
    try:
        phase1_artifacts_dir = get_config().get_phase1_path("artifacts_dir") or "output/phase1/artifacts"
        provenance_path = os.path.join(str(phase1_artifacts_dir), "feature_provenance.json")
        if not os.path.exists(provenance_path):
            return provenance_index

        with open(provenance_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not isinstance(features, list):
            return provenance_index

        for record in features:
            if not isinstance(record, dict):
                continue
            feature_name = str(record.get("feature", "") or "").strip()
            standardized_name = str(record.get("standardized_name", "") or "").strip()
            if feature_name:
                provenance_index[feature_name] = record
            if standardized_name:
                provenance_index[standardized_name] = record
    except Exception as e:
        print(f"[Cost] Warning: failed to load feature provenance index: {e}")
    return provenance_index


def _load_engineered_feature_registry_index() -> Dict[str, Dict[str, Any]]:
    """Best-effort load for Phase 1 engineered feature registry."""
    registry_index: Dict[str, Dict[str, Any]] = {}
    try:
        cfg = get_config()
        candidate_paths: List[str] = []

        configured_path = (
            cfg.get_phase1_engineered_feature_registry_path()
            or cfg.get_phase1_path("engineered_feature_registry")
        )
        if configured_path:
            candidate_paths.append(str(configured_path))

        intermediate_latest_dir = cfg.get_phase1_path("intermediate_latest_dir")
        if intermediate_latest_dir:
            candidate_paths.append(
                os.path.join(str(intermediate_latest_dir), "engineered", "engineered_feature_registry.json")
            )

        registry_path = next((path for path in candidate_paths if path and os.path.exists(path)), None)
        if not registry_path:
            return registry_index

        with open(registry_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not isinstance(features, list):
            return registry_index

        for record in features:
            if not isinstance(record, dict):
                continue
            feature_name = str(record.get("feature_name", "") or "").strip()
            standardized_name = feature_name.upper() if feature_name else ""
            if feature_name:
                registry_index[feature_name] = record
            if standardized_name:
                registry_index[standardized_name] = record
    except Exception as e:
        print(f"[Cost] Warning: failed to load engineered feature registry index: {e}")
    return registry_index


def _get_provenance_record(
    feature_name: str,
    feature_provenance_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a feature provenance record by raw or uppercase standardized name."""
    if feature_provenance_index is None:
        feature_provenance_index = _load_feature_provenance_index()
    raw_name = str(feature_name or "").strip()
    standardized_name = raw_name.upper()
    return (
        feature_provenance_index.get(raw_name)
        or feature_provenance_index.get(standardized_name)
        or None
    )


def _extract_provenance_hmdb_ids(provenance_record: Optional[Dict[str, Any]]) -> List[str]:
    """Extract explicit HMDB members from provenance when available."""
    if not isinstance(provenance_record, dict):
        return []
    candidate_lists = [
        provenance_record.get("mapped_hmdb_ids"),
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


def _extract_registry_hmdb_ids(registry_record: Optional[Dict[str, Any]]) -> List[str]:
    """Extract explicit HMDB members from engineered feature registry when available."""
    if not isinstance(registry_record, dict):
        return []
    candidate_lists = [
        registry_record.get("selected_member_hmdb_ids"),
        registry_record.get("observed_member_hmdb_ids"),
        registry_record.get("mapped_hmdb_ids"),
        registry_record.get("all_member_hmdb_ids"),
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


def _resolve_cost_observed_hmdb_set(
    data_path: Optional[str],
    taxonomy_map: Dict[str, dict],
    feature_provenance_index: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_registry_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[set, str]:
    """
    Resolve a richer observed-HMDB universe for f_cost.

    Priority:
    1. Phase 1 intermediate full matrix (`data_with_engineered_features.csv`)
    2. Phase 1 pre-engineering matrix
    3. Caller-provided data_path
    4. Union of provenance / registry HMDB members
    5. taxonomy_map keys as coarse fallback
    """
    cfg = get_config()
    candidate_paths: List[str] = []
    intermediate_latest_dir = cfg.get_phase1_path("intermediate_latest_dir")
    if intermediate_latest_dir:
        candidate_paths.extend([
            os.path.join(str(intermediate_latest_dir), "engineered", "data_with_engineered_features.csv"),
            os.path.join(str(intermediate_latest_dir), "pre_engineering_matrix.csv"),
            os.path.join(str(intermediate_latest_dir), "cleaned_pre_engineering.csv"),
        ])
    if data_path:
        candidate_paths.append(str(data_path))

    seen_paths = set()
    dataset_hmdb_set: set = set()
    source_tags: List[str] = []

    for path in candidate_paths:
        normalized_path = os.path.abspath(path)
        if normalized_path in seen_paths or not os.path.exists(normalized_path):
            continue
        seen_paths.add(normalized_path)
        try:
            df_columns = pd.read_csv(normalized_path, nrows=0).columns
            extracted = set()
            for col in df_columns:
                extracted.update(re.findall(r'HMDB\d+', str(col)))
            if extracted:
                dataset_hmdb_set.update(extracted)
                source_tags.append(os.path.basename(normalized_path))
        except Exception as e:
            print(f"[Cost] Warning: failed to inspect observed HMDB set from {normalized_path}: {e}")

    if dataset_hmdb_set:
        return dataset_hmdb_set, "+".join(source_tags)

    fallback_ids: set = set()
    if feature_provenance_index:
        for record in feature_provenance_index.values():
            fallback_ids.update(_extract_provenance_hmdb_ids(record))
    if feature_registry_index:
        for record in feature_registry_index.values():
            fallback_ids.update(_extract_registry_hmdb_ids(record))
    if fallback_ids:
        return fallback_ids, "feature_provenance_or_registry"

    return set(taxonomy_map.keys()), "taxonomy_map_fallback"


def _get_registry_record(
    feature_name: str,
    feature_registry_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve an engineered feature registry record by raw or uppercase standardized name."""
    if feature_registry_index is None:
        feature_registry_index = _load_engineered_feature_registry_index()
    raw_name = str(feature_name or "").strip()
    standardized_name = raw_name.upper()
    return (
        feature_registry_index.get(raw_name)
        or feature_registry_index.get(standardized_name)
        or None
    )


def _resolve_feature_entity_ids(
    feature_name: str,
    dataset_hmdb_set: set,
    taxonomy_map: Dict[str, dict],
    pathway_map: Dict[str, list],
    feature_provenance_index: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_registry_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, List[str]]:
    """
    Resolve the underlying measurable analytes for a feature.

    Returns:
        (resolution_source, hmdb_ids)
    """
    provenance_record = _get_provenance_record(feature_name, feature_provenance_index)
    provenance_hmdb_ids = _extract_provenance_hmdb_ids(provenance_record)
    if provenance_hmdb_ids:
        return "feature_provenance", sorted(set(provenance_hmdb_ids))

    registry_record = _get_registry_record(feature_name, feature_registry_index)
    registry_hmdb_ids = _extract_registry_hmdb_ids(registry_record)
    if registry_hmdb_ids:
        return "engineered_feature_registry", sorted(set(registry_hmdb_ids))

    feature_upper = str(feature_name or "").strip().upper()

    if feature_upper.startswith('SUM_') or feature_upper.startswith('TAXSUM_'):
        group_name = feature_name.split('_', 1)[1].replace('_', ' ')
        raw_ids = [
            h_id for h_id, info in taxonomy_map.items()
            if str(info.get('class', '')) == group_name or
               str(info.get('sub_class', '')) == group_name
        ]
        actual_ids = sorted(set(raw_ids).intersection(dataset_hmdb_set)) if dataset_hmdb_set else sorted(set(raw_ids))
        return "taxonomy_map", actual_ids

    if feature_upper.startswith('PATHWAY_'):
        pathway_name = feature_upper.replace('PATHWAY_', '').replace('_', ' ')
        matched_pw_key = next(
            (key for key in pathway_map.keys() if pathway_name.lower() in key.lower()),
            None
        )
        raw_ids = pathway_map.get(matched_pw_key, []) if matched_pw_key else []
        actual_ids = sorted(set(raw_ids).intersection(dataset_hmdb_set)) if dataset_hmdb_set else sorted(set(raw_ids))
        return "pathway_map", actual_ids

    if feature_upper.startswith('RATIO_'):
        ratio_ids = re.findall(r'HMDB\d+', feature_name)
        actual_ids = sorted(set(ratio_ids).intersection(dataset_hmdb_set)) if dataset_hmdb_set else sorted(set(ratio_ids))
        return "ratio_name", actual_ids

    raw_ids = re.findall(r'HMDB\d+', feature_name)
    actual_ids = sorted(set(raw_ids).intersection(dataset_hmdb_set)) if dataset_hmdb_set else sorted(set(raw_ids))
    return "feature_name", actual_ids

def _sigmoid_mapping(raw_score: float, k: float = 1.0) -> float:
    """
    Sigmoid 映射，将无界 LogProb 压入 (0,1) 区间。
    
    Args:
        raw_score: 原始 LogProb 分数
        k: 温度系数（控制曲线陡峭程度）
    
    Returns:
        float: 映射后的概率值 [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-k * raw_score))


def _coerce_prior_probability(score: Optional[float], sigmoid_k: float = 1.0) -> float:
    """
    将 Phase 0 先验值统一映射为 [0,1] 概率。

    兼容两类输入：
    1. 新结构 `bio_prior_norm`，本身已在 [0,1]
    2. 旧结构 `confidence_score/raw logprob`，需要 sigmoid 压缩
    """
    if score is None:
        return 0.0

    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return 0.0

    if 0.0 <= numeric <= 1.0:
        return numeric
    return _sigmoid_mapping(numeric, sigmoid_k)


def _get_top_k_average(scores: list, k_ratio: float = 0.3, min_k: int = 2) -> float:
    """
    计算 Top-K 核心驱动力的均值，防止大量低分背景代谢物稀释信号。
    
    Args:
        scores: 分数列表
        k_ratio: Top-K 比例（默认 30%）
        min_k: 最小 K 值（至少取 2 个）
    
    Returns:
        float: Top-K 均值
    """
    if not scores:
        return 0.0
    
    scores = sorted(scores, reverse=True)
    k = max(min_k, int(len(scores) * k_ratio))
    k = min(k, len(scores))
    
    return float(np.mean(scores[:k]))


def _calculate_single_p0(
    feature_name: str,
    priors_dict: dict,
    taxonomy_map: dict,
    pathway_map: dict,
    w_lit: float = 0.7,
    w_mech: float = 0.3,
    sigmoid_k: float = 1.0
) -> float:
    """
    基于方案 B 的双轨先验评分 P0(i)。
    
    核心设计：
    1. Prior Track (先验轨): 基于 Phase 0 的 `bio_prior_norm`
    2. Topology Track (拓扑轨): 基于特征的工程拓扑维度阶梯

    迁移说明：
    - 为保持向后兼容，函数参数名仍保留 `w_lit` / `w_mech`
    - 但语义已迁移为 `w_prior` / `w_topo`
    - 若输入仍是旧版 raw confidence/logprob，会自动使用 sigmoid 压缩
    
    拓扑维度定义（硬编码）：
    - 零维（单体特征 Single HMDB）: I_topo = 0.25 (1/4)
    - 一维无序（分类学加和 Sum/TaxSum）: I_topo = 0.50 (2/4)
    - 一维有向（反应比值 Ratio）: I_topo = 0.75 (3/4)
    - 高维子图（通路富集 Pathway）: I_topo = 1.00 (4/4)
    
    Args:
        feature_name: 特征名称
        priors_dict: Phase 0 先验字典 {HMDB_ID: bio_prior_norm 或 legacy confidence}
        taxonomy_map: 分类学映射 {HMDB_ID: {class, sub_class, ...}}
        pathway_map: 通路映射 {pathway_name: [HMDB_IDs]}
        w_lit: 先验轨权重（兼容旧命名，默认 0.7）
        w_mech: 拓扑轨权重（兼容旧命名，默认 0.3）
        sigmoid_k: 旧版 raw prior 的 Sigmoid 温度系数（默认 1.0）
    
    Returns:
        float: 先验评分 P0(i) ∈ [0, 1]
    
    Examples:
        >>> priors = {"HMDB0000001": 2.5, "HMDB0000002": -1.5}
        >>> p0 = _calculate_single_p0("Ratio_HMDB0000001_HMDB0000002", priors, {}, {})
        >>> print(f"P0 = {p0:.4f}")
    """
    # 强制权重归一化（防呆设计）
    w_prior = w_lit
    w_topo = w_mech
    total_w = w_prior + w_topo
    if total_w > 0:
        w_prior, w_topo = w_prior / total_w, w_topo / total_w
    else:
        w_prior, w_topo = 0.7, 0.3  # 极端的除零保护
    
    p_prior = 0.0
    i_topo = 0.0
    
    # 辅助闭包：安全获取单体代谢物的 P_prior
    def _get_single_p_prior(h_id: str) -> float:
        if h_id in priors_dict:
            return _coerce_prior_probability(priors_dict[h_id], sigmoid_k)
        return 0.0  # 纯数据驱动，先验分为 0
    
    # ==========================================================================
    # 1. 反应比值 (Directed Edge / Ratio) - 维度 3/4
    # ==========================================================================
    feature_upper = str(feature_name or "").strip().upper()

    if feature_upper.startswith('RATIO_'):
        i_topo = 0.75
        hmdb_ids = re.findall(r'HMDB\d+', feature_name)
        
        if hmdb_ids:
            # 概率并集 Noisy-OR: 1 - product(1 - p_i)
            inverse_prob = 1.0
            for h_id in hmdb_ids:
                p_i = _get_single_p_prior(h_id)
                inverse_prob *= (1.0 - p_i)
            p_prior = float(1.0 - inverse_prob)
    
    # ==========================================================================
    # 2. 分类加和 (Undirected Set / Sum) - 维度 2/4
    # ==========================================================================
    elif feature_upper.startswith('SUM_') or feature_upper.startswith('TAXSUM_'):
        i_topo = 0.50
        group_name = feature_name.split('_', 1)[1].replace('_', ' ')
        
        # 从 taxonomy_map 中查找属于该分类的所有 HMDB ID
        involved_hmdbs = [
            h_id for h_id, info in taxonomy_map.items()
            if str(info.get('class', '')) == group_name or 
               str(info.get('sub_class', '')) == group_name
        ]
        
        if involved_hmdbs:
            sub_scores = [_get_single_p_prior(h_id) for h_id in involved_hmdbs]
            # 取 Top-30% 均值
            p_prior = _get_top_k_average(sub_scores, k_ratio=0.3, min_k=2)
    
    # ==========================================================================
    # 3. 通路富集 (Functional Subgraph / Pathway) - 维度 4/4
    # ==========================================================================
    elif feature_upper.startswith('PATHWAY_'):
        i_topo = 1.00
        pathway_name = feature_upper.replace('PATHWAY_', '').replace('_', ' ')
        
        # 模糊匹配通路名称
        matched_pw_key = next(
            (k for k in pathway_map.keys() if pathway_name.lower() in k.lower()),
            None
        )
        
        if matched_pw_key:
            involved_hmdbs = pathway_map[matched_pw_key]
            sub_scores = [_get_single_p_prior(h_id) for h_id in involved_hmdbs]
            # 取 Top-30% 均值
            p_prior = _get_top_k_average(sub_scores, k_ratio=0.3, min_k=2)
    
    # ==========================================================================
    # 4. 单体代谢物 (Isolated Node / Single) - 维度 1/4
    # ==========================================================================
    else:
        i_topo = 0.25
        clean_id_match = re.search(r'HMDB\d+', feature_name)
        clean_id = clean_id_match.group(0) if clean_id_match else feature_name
        p_prior = _get_single_p_prior(clean_id)
    
    # ==========================================================================
    # 双轨融合计算
    # ==========================================================================
    final_p0 = (w_prior * p_prior) + (w_topo * i_topo)
    
    return min(1.0, float(final_p0))


# ============================================================================
# 辅助函数：动态分类器选择
# ============================================================================

def _get_dynamic_classifier(model_family: str):
    """
    根据指定的模型家族返回对应的轻量级分类器。
    
    PERFORMANCE OPTIMIZED: 
    - 移除了内部的文件读取逻辑，避免重复 I/O
    - 直接接收模型家族字符串参数
    - 适用于高频调用场景（如 PToT 搜索的数百次迭代）
    
    策略：
    1. 根据 model_family 字符串匹配对应的分类器
    2. 支持的模型家族: "LightGBM", "XGBoost", "CatBoost", "RandomForest", "GradientBoosting"
    3. 如果无法匹配，返回默认的 RandomForest
    
    Args:
        model_family: 模型家族名称 (如 "LightGBM", "XGBoost", "CatBoost", "RandomForest", "GradientBoosting")
    
    Returns:
        sklearn-compatible classifier instance
    
    Examples:
        >>> clf = _get_dynamic_classifier('LightGBM')
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """
    # 默认保底模型：RandomForest
    default_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=PHASE2_RF_N_JOBS,
        random_state=42
    )
    
    # 如果没有提供模型家族，使用默认模型
    if not model_family:
        print("Info: No model family specified, using default RandomForest")
        return default_clf
    
    # 转换为小写以便匹配
    model_family_lower = model_family.lower()
    
    # 动态匹配策略
    if 'lightgbm' in model_family_lower and HAS_LGB:
        print(f"Info: Using LightGBM classifier (champion model from Phase 1)")
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=10,
            num_leaves=31,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
    
    elif 'xgb' in model_family_lower and HAS_XGB:
        print(f"Info: Using XGBoost classifier (champion model from Phase 1)")
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )

    elif 'catboost' in model_family_lower and HAS_CATBOOST:
        print(f"Info: Using CatBoost classifier (champion model from Phase 1)")
        return CatBoostClassifier(
            iterations=100,
            depth=10,
            learning_rate=0.1,
            loss_function='Logloss',
            verbose=False,
            random_seed=42
        )
    
    elif 'randomforest' in model_family_lower or 'rf' in model_family_lower:
        print(f"Info: Using RandomForest classifier (champion model from Phase 1)")
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=PHASE2_RF_N_JOBS,
            random_state=42
        )

    elif 'gradientboost' in model_family_lower or 'gradient_boost' in model_family_lower:
        print(f"Info: Using GradientBoosting classifier (champion model from Phase 1)")
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    
    else:
        # 如果无法匹配，使用默认模型
        print(f"Warning: Unknown model family '{model_family}', using default RandomForest")
        return default_clf


def _get_resampling_strategy(ag_results_path: str = 'data/autogluon_training_results.json'):
    """
    解析 Phase 1 的重采样策略，动态返回对应的 imblearn 重采样器。
    
    CRITICAL: 必须使用 imblearn 的重采样器，确保与 Phase 1 的评估一致性。
    
    Args:
        ag_results_path: AutoGluon 训练结果 JSON 文件路径
    
    Returns:
        imblearn resampler instance or None (if no resampling)
    
    Examples:
        >>> resampler = _get_resampling_strategy()
        >>> if resampler:
        ...     X_res, y_res = resampler.fit_resample(X, y)
    """
    if not HAS_IMBLEARN:
        print("Warning: imbalanced-learn not installed, resampling disabled")
        return None
    
    # 检查文件是否存在
    if not os.path.exists(ag_results_path):
        print(f"Info: AG results not found, no resampling strategy applied")
        return None
    
    try:
        # 读取 Phase 1 结果
        with open(ag_results_path, 'r') as f:
            ag_results = json.load(f)
        
        # Phase 1 使用 'resampling_applied' 字段，兼容旧版本的 'balancing_method'
        balancing_method = ag_results.get('resampling_applied', 
                                         ag_results.get('balancing_method', 'none'))
        
        # 处理字符串格式（如 "Skipped (Categorical Data)" 或 "none"）
        balancing_method_lower = str(balancing_method).lower()
        
        print(f"Info: Phase 1 balancing method = {balancing_method}")
        
        # 动态匹配重采样策略
        if ('none' in balancing_method_lower or 
            'skip' in balancing_method_lower or 
            balancing_method is None):
            return None
        
        elif 'smote' in balancing_method_lower and 'nearmiss' in balancing_method_lower:
            print("Info: Using SMOTE+NearMiss resampling (inherited from Phase 1)")
            # 组合策略：先 SMOTE 过采样，再 NearMiss 欠采样
            # 注意：这里返回一个元组，后续需要特殊处理
            return ('smote_nearmiss', SMOTE(random_state=42), NearMiss(version=2))
        
        elif 'smote' in balancing_method_lower:
            print("Info: Using SMOTE resampling (inherited from Phase 1)")
            return SMOTE(random_state=42, k_neighbors=5)
        
        elif 'smoteenn' in balancing_method_lower:
            print("Info: Using SMOTEENN resampling (inherited from Phase 1)")
            return SMOTEENN(random_state=42)
        
        elif 'smotetomek' in balancing_method_lower:
            print("Info: Using SMOTETomek resampling (inherited from Phase 1)")
            return SMOTETomek(random_state=42)
        
        else:
            print(f"Warning: Unknown balancing method '{balancing_method}', no resampling applied")
            return None
    
    except Exception as e:
        print(f"Warning: Failed to parse resampling strategy: {e}")
        return None


def _load_phase1_autogluon_metadata(ag_results_path: str) -> Dict[str, Any]:
    if not ag_results_path or not os.path.exists(ag_results_path):
        return {}
    try:
        with open(ag_results_path, "r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception as exc:
        print(f"Warning: Failed to load Phase 1 AutoGluon metadata: {exc}")
        return {}


def _predict_phase1_autogluon_holdout(
    *,
    ag_results_path: str,
    champion_model_family: str,
    X_holdout: pd.DataFrame,
) -> Optional[np.ndarray]:
    if not HAS_AUTOGLUON or TabularPredictor is None:
        return None
    metadata = _load_phase1_autogluon_metadata(ag_results_path)
    model_path = metadata.get("model_path")
    if not model_path or not os.path.exists(model_path):
        return None
    model_name = champion_model_family or metadata.get("best_model")
    try:
        predictor = TabularPredictor.load(model_path)
        proba = predictor.predict_proba(X_holdout, model=model_name)
        if hasattr(proba, "columns"):
            if 1 in proba.columns:
                return proba[1].to_numpy(dtype=float)
            if "1" in proba.columns:
                return proba["1"].to_numpy(dtype=float)
            return proba.iloc[:, -1].to_numpy(dtype=float)
        proba_array = np.asarray(proba)
        return proba_array[:, -1].astype(float) if proba_array.ndim == 2 else proba_array.astype(float)
    except Exception as exc:
        print(f"Warning: Failed to predict holdout with Phase 1 AutoGluon winner '{model_name}': {exc}")
        return None


def _build_f_perf_model(
    *,
    use_phase1_config: bool,
    champion_model_family: str,
    ag_results_path: str,
    n_samples: int,
):
    dynamic_max_depth = max(3, min(10, n_samples // 10))

    if use_phase1_config:
        clf = _get_dynamic_classifier(champion_model_family)
        resampler = _get_resampling_strategy(ag_results_path)
    else:
        print("Info: Using RandomForest classifier (fast search mode)")
        print(
            "Info: Dynamic max_depth = "
            f"{dynamic_max_depth} (n_samples={n_samples}, formula: max(3, min(10, n_samples//10)))"
        )
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=dynamic_max_depth,
            random_state=42,
            n_jobs=PHASE2_RF_N_JOBS,
        )
        resampler = None

    if resampler is not None and HAS_IMBLEARN:
        if isinstance(resampler, tuple) and resampler[0] == 'smote_nearmiss':
            _, smote, nearmiss = resampler
            return ImbPipeline([
                ('smote', smote),
                ('nearmiss', nearmiss),
                ('classifier', clf)
            ])
        return ImbPipeline([
            ('resampler', resampler),
            ('classifier', clf)
        ])

    return clf


# ============================================================================
# 目标函数 1: f_perf - 模型预测性能评估
# ============================================================================

def _resolve_prediction_sample_ids(frame: pd.DataFrame) -> Tuple[List[str], str]:
    """Resolve a stable per-row identifier for prediction artifacts.

    IDs are metadata, not model features. Prefer an explicit identifier
    column and only fall back to the row index when the input truly has no
    usable unique identifier. Returning the source column lets downstream
    audit files expose whether pairing is safe.
    """
    candidate_columns = (
        "Sample_ID",
        "sample_id",
        "SampleID",
        "sampleid",
        "ID",
        "id",
        "subject_id",
        "subjectid",
        "patient_id",
        "patientid",
        "participant_id",
        "participantid",
        "record_id",
        "recordid",
        "ROW_ID",
        "row_id",
        "__row_id__",
    )
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        if values.isna().any():
            continue
        normalized = values.astype(str).str.strip()
        if normalized.empty or (normalized == "").any() or not normalized.is_unique:
            continue
        return normalized.tolist(), column
    return [str(index) for index in range(len(frame))], "row_index"

def _label_key(value: Any) -> str:
    """Normalize UI-provided labels for comparison with pandas values."""
    text = str(value).strip()
    if text.endswith(".0"):
        integer_part = text[:-2]
        if integer_part.lstrip("-").isdigit():
            text = integer_part
    return text.casefold()


def _fit_target_label_state(y: Sequence[Any], positive_class: Optional[str] = None) -> Dict[str, Any]:
    """Fit the binary target mapping, honoring the selected positive class.

    The existing default LabelEncoder behavior is preserved when no positive
    class is supplied.  When supplied, the complementary class is explicitly
    placed at 0 and the selected class at 1 so ROC/AUPRC semantics are stable.
    """
    encoder = LabelEncoder()
    encoder.fit(y)
    original_classes = encoder.classes_.tolist()
    if len(original_classes) != 2 or not str(positive_class or "").strip():
        return {
            "classes": original_classes,
            "mapping": {_label_key(label): int(index) for index, label in enumerate(original_classes)},
            "positive_class": None,
        }

    requested_key = _label_key(positive_class)
    positive_raw = next((label for label in original_classes if _label_key(label) == requested_key), None)
    if positive_raw is None:
        raise ValueError(
            f"正类标签 '{positive_class}' 不在目标列实际类别中: "
            f"{[str(label) for label in original_classes]}"
        )
    negative_raw = next(label for label in original_classes if _label_key(label) != requested_key)
    classes = [negative_raw, positive_raw]
    return {
        "classes": classes,
        "mapping": {_label_key(negative_raw): 0, _label_key(positive_raw): 1},
        "positive_class": str(positive_raw),
    }


def _transform_target_labels(values: Sequence[Any], label_state: Dict[str, Any]) -> np.ndarray:
    mapping = dict(label_state.get("mapping", {}) or {})
    encoded: List[int] = []
    for value in values:
        key = _label_key(value)
        if key not in mapping:
            raise ValueError(f"标签 '{value}' 不在训练集类别中")
        encoded.append(int(mapping[key]))
    return np.asarray(encoded, dtype=int)


def calculate_f_perf(
    data_path: str,
    target_column: str,
    features_list: List[str],
    champion_model_family: str,
    k_folds: int = 5,
    ag_results_path: str = 'data/autogluon_training_results.json',
    metric: str = 'auprc',
    use_phase1_config: bool = False,
    return_predictions: bool = False,
    holdout_data_path: Optional[str] = None,
    evaluation_mode: str = 'auto',
    positive_class: Optional[str] = None,
    use_phase1_autogluon_holdout: bool = True,
) -> Union[float, Tuple[float, Dict]]:
    """
    计算特征子集 S 的预测性能 f_perf(S)。
    
    核心设计（两阶段策略）：
    
    【搜索阶段】(use_phase1_config=False, 默认):
    1. 使用 RandomForest（快速、稳定）
    2. 不使用重采样（避免额外开销）
    3. 目的：快速评估候选特征组合的相对优劣
    
    【最终评估阶段】(use_phase1_config=True):
    1. 使用 Phase 1 的冠军模型（LightGBM/XGBoost/RandomForest）
    2. 使用 Phase 1 的重采样策略（SMOTE/NearMiss/SMOTEENN等）
    3. 目的：对最终选出的特征集合进行准确的性能评估
    
    PERFORMANCE OPTIMIZED:
    - champion_model_family 参数直接传入，避免重复文件读取
    - 搜索阶段使用轻量级配置，适用于高频调用场景（如 PToT 搜索的数百次迭代）
    
    CRITICAL: 
    - 搜索阶段：RandomForest + 无重采样（快速）
    - 最终评估：Phase 1 配置（准确）
    - k_folds=5 保持与 Phase 1 一致
    
    NEW (2026-04-24):
    - 添加 return_predictions 参数，用于保存交叉验证预测概率
    - 当 return_predictions=True 时，返回 (score, predictions_dict)
    - predictions_dict 包含: y_true, y_pred_proba, sample_ids, fold_ids
    
    Args:
        data_path: 数据文件路径（CSV格式）
        target_column: 目标列名称
        features_list: 特征子集列表
        champion_model_family: Phase 1 选出的冠军模型家族 (如 "LightGBM", "XGBoost")
        k_folds: 交叉验证折数（默认5，与 Phase 1 一致）
        ag_results_path: AutoGluon 训练结果 JSON 文件路径
        metric: 评估指标 ('auprc' 或 'roc_auc')
        use_phase1_config: 是否使用 Phase 1 的完整配置（冠军模型+重采样）
                          False=搜索阶段（RandomForest+无重采样，默认）
                          True=最终评估阶段（Phase 1 配置）
        return_predictions: 是否返回交叉验证预测概率（默认 False）
                           True=返回 (score, predictions_dict)
                           False=只返回 score
        use_phase1_autogluon_holdout: holdout_only 模式下是否直接调用
                           Phase 1 AutoGluon predictor。Phase 2 的最终 panel
                           默认关闭该路径，以避免模型 schema/版本变化影响锁定留出集。
    
    Returns:
        float: 预测性能得分，范围 [0.0, 1.0]
        或
        Tuple[float, Dict]: (score, predictions_dict) 当 return_predictions=True
            predictions_dict 包含:
                - y_true: 真实标签数组
                - y_pred_proba: 预测概率数组
                - sample_ids: 样本索引数组
                - fold_ids: 所属 fold 数组
                - label_names: 类别名称列表
    
    Examples:
        >>> # 搜索阶段：快速评估
        >>> features = ['HMDB0000001', 'HMDB0000002', 'HMDB0000003']
        >>> score = calculate_f_perf('data/cleaned.csv', 'Group', features, 'LightGBM')
        >>> print(f"Search Performance: {score:.4f}")
        Search Performance: 0.8523
        
        >>> # 最终评估：使用 Phase 1 配置 + 保存预测
        >>> score, predictions = calculate_f_perf(
        ...     'data/cleaned.csv', 'Group', features, 'LightGBM',
        ...     use_phase1_config=True, return_predictions=True
        ... )
        >>> print(f"Final Performance: {score:.4f}")
        >>> print(f"Predictions shape: {predictions['y_pred_proba'].shape}")
        Final Performance: 0.8645
        Predictions shape: (200,)
    """
    def _empty_predictions_payload(error: str) -> Dict[str, Any]:
        return {
            "predictions": [],
            "mean_score": 0.0,
            "label_names": [],
            "n_samples": 0,
            "k_folds": k_folds,
            "metric": metric,
            "evaluation_mode": evaluation_mode,
            "error": error,
        }

    def _return_empty(error: str) -> Union[float, Tuple[float, Dict[str, Any]]]:
        if return_predictions:
            return 0.0, _empty_predictions_payload(error)
        return 0.0

    try:
        # 抑制 LGBMClassifier 的 feature names warning
        # 原因：imblearn 的重采样器会将 DataFrame 转换为 numpy array
        import warnings
        warnings.filterwarnings('ignore', message='X does not have valid feature names')
        
        # 1. 加载数据
        df = pd.read_csv(data_path)
        
        # ====================================================================
        # 🔧 修复 1: 特征去重（保持顺序）
        # ====================================================================
        # 使用 dict.fromkeys() 去重，保持原始顺序
        features_list = list(dict.fromkeys(features_list))
        
        if len(features_list) == 0:
            print(f"Warning: 特征列表为空，返回 0.0")
            return _return_empty("empty_feature_list")
        
        # 验证目标列存在
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
        
        # ====================================================================
        # 🔧 修复 2: 验证特征列存在（优雅降级：支持标准化匹配）
        # ====================================================================
        # Phase 1 已在源头标准化特征名，但为了向后兼容和鲁棒性，
        # 仍然保留动态标准化匹配能力
        from src.utils.feature_name_standardizer import validate_feature_names
        
        # 使用标准化验证特征（优雅降级）
        validation_result = validate_feature_names(
            feature_names=features_list,
            dataset_columns=df.columns.tolist(),
            auto_standardize=True  # 启用自动标准化匹配（向后兼容）
        )
        
        valid_features = validation_result['valid']
        missing_features = validation_result['missing']
        feature_mapping = validation_result['mapping']
        
        if not valid_features:
            print(f"Warning: 没有有效的特征列，返回 0.0")
            print(f"  请求的特征: {features_list[:5]}{'...' if len(features_list) > 5 else ''}")
            print(f"  数据列: {df.columns.tolist()[:10]}{'...' if len(df.columns) > 10 else ''}")
            return _return_empty("no_valid_features")
        
        if missing_features:
            print(f"Warning: {len(missing_features)} 个特征在数据中不存在:")
            for feat in missing_features[:3]:
                print(f"  - {feat}")
            if len(missing_features) > 3:
                print(f"  ... 还有 {len(missing_features) - 3} 个")
        
        # 使用映射后的列名提取数据
        mapped_features = [feature_mapping[f] for f in valid_features]
        
        # ====================================================================
        # 🔧 修复：清理特征名称（LightGBM 兼容性 - 二次保险）
        # ====================================================================
        # Phase 1 已标准化，但为了确保 LightGBM 兼容性，再次清理
        from src.utils.feature_name_standardizer import standardize_feature_name
        
        feature_name_mapping = {}
        clean_feature_names = []
        
        for original_feature, mapped_feature in zip(valid_features, mapped_features):
            # 使用标准化函数清理名称（应该已经是干净的）
            clean_name = standardize_feature_name(mapped_feature)
            feature_name_mapping[mapped_feature] = clean_name
            clean_feature_names.append(clean_name)
        
        # 提取特征和目标（使用映射后的名称提取数据）
        X = df[mapped_features].copy()
        
        # 重命名列为清理后的名称（应该已经是干净的，这是二次保险）
        X.columns = clean_feature_names
        
        y = df[target_column].values
        
        # 2. 预处理
        # 编码目标变量；如果用户指定正类，强制其编码为 1。
        label_state = _fit_target_label_state(y, positive_class=positive_class)
        y_encoded = _transform_target_labels(y, label_state)
        label_names = list(label_state.get("classes", []) or [])
        
        # 检查类别数量
        n_classes = len(np.unique(y_encoded))
        if n_classes < 2:
            print(f"Warning: 目标列只有 {n_classes} 个类别，无法进行分类")
            return _return_empty(f"insufficient_classes: {n_classes}")
        
        resolved_mode = str(evaluation_mode or 'auto').strip().lower()
        if resolved_mode == 'auto':
            resolved_mode = 'holdout_only' if holdout_data_path and os.path.exists(holdout_data_path) else 'cv'
        if resolved_mode not in {'cv', 'holdout_only'}:
            raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")

        n_samples = len(X)

        if resolved_mode == 'holdout_only':
            if not holdout_data_path or not os.path.exists(holdout_data_path):
                raise FileNotFoundError(f"holdout_data_path not found for holdout_only mode: {holdout_data_path}")

            holdout_df = pd.read_csv(holdout_data_path)
            if target_column not in holdout_df.columns:
                raise ValueError(f"目标列 '{target_column}' 不存在于 holdout 数据中")
            holdout_sample_ids, holdout_sample_id_source = _resolve_prediction_sample_ids(holdout_df)

            holdout_validation = validate_feature_names(
                feature_names=features_list,
                dataset_columns=holdout_df.columns.tolist(),
                auto_standardize=True,
            )
            holdout_valid_features = holdout_validation['valid']
            holdout_feature_mapping = holdout_validation['mapping']
            if not holdout_valid_features:
                return _return_empty("no_valid_features_in_holdout")

            train_feature_to_clean_name = {
                requested_feature: clean_name
                for requested_feature, clean_name in zip(valid_features, clean_feature_names)
            }
            X_holdout = pd.DataFrame(0.0, index=holdout_df.index, columns=clean_feature_names)
            for feature_name in holdout_valid_features:
                clean_name = train_feature_to_clean_name.get(feature_name)
                holdout_column = holdout_feature_mapping.get(feature_name)
                if not clean_name or not holdout_column or holdout_column not in holdout_df.columns:
                    continue
                X_holdout[clean_name] = pd.to_numeric(holdout_df[holdout_column], errors="coerce").fillna(0.0)
            y_holdout_raw = holdout_df[target_column].values

            try:
                y_holdout = _transform_target_labels(y_holdout_raw, label_state)
            except ValueError as exc:
                raise ValueError(f"holdout 标签包含训练集中未见类别: {exc}") from exc

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=clean_feature_names,
                index=X.index,
            )
            X_holdout_scaled = pd.DataFrame(
                scaler.transform(X_holdout),
                columns=clean_feature_names,
                index=X_holdout.index,
            )

            if n_classes == 2:
                y_probs_full = None
                y_probs = None
                if use_phase1_config and use_phase1_autogluon_holdout:
                    ag_probs = _predict_phase1_autogluon_holdout(
                        ag_results_path=ag_results_path,
                        champion_model_family=champion_model_family,
                        X_holdout=holdout_df[mapped_features].copy(),
                    )
                    if ag_probs is not None and len(ag_probs) == len(y_holdout):
                        y_probs = np.asarray(ag_probs, dtype=float)
                        y_probs_full = np.column_stack([1.0 - y_probs, y_probs])
                if y_probs_full is None:
                    model = _build_f_perf_model(
                        use_phase1_config=use_phase1_config,
                        champion_model_family=champion_model_family,
                        ag_results_path=ag_results_path,
                        n_samples=n_samples,
                    )
                    model.fit(X_train_scaled, y_encoded)
                    y_probs_full = model.predict_proba(X_holdout_scaled)
                    y_probs = y_probs_full[:, 1]
                score = roc_auc_score(y_holdout, y_probs) if metric == 'roc_auc' else average_precision_score(y_holdout, y_probs)
            else:
                model = _build_f_perf_model(
                    use_phase1_config=use_phase1_config,
                    champion_model_family=champion_model_family,
                    ag_results_path=ag_results_path,
                    n_samples=n_samples,
                )
                model.fit(X_train_scaled, y_encoded)
                y_probs_full = None
                y_probs = None
                score = model.score(X_holdout_scaled, y_holdout)

            if return_predictions:
                predictions = []
                if n_classes == 2 and y_probs_full is not None:
                    for row_idx in range(len(X_holdout_scaled)):
                        predictions.append({
                            'sample_id': holdout_sample_ids[row_idx],
                            'true_label': y_holdout_raw[row_idx],
                            'true_label_encoded': int(y_holdout[row_idx]),
                            'pred_proba_class0': float(y_probs_full[row_idx, 0]),
                            'pred_proba_class1': float(y_probs_full[row_idx, 1]),
                            'fold': 0,
                        })
                predictions_dict = {
                    'predictions': predictions,
                    'mean_score': float(score),
                    'label_names': label_names,
                    'n_samples': len(X_holdout_scaled),
                    'k_folds': 1,
                    'metric': metric,
                    'evaluation_mode': resolved_mode,
                    'holdout_data_path': holdout_data_path,
                    'sample_id_source': holdout_sample_id_source,
                }
                return round(float(score), 4), predictions_dict
            return round(float(score), 4)

        # 3. 无外部 holdout 时，继续使用训练集内部 CV。
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        scores = []
        all_predictions = [] if return_predictions else None
        cv_sample_ids, cv_sample_id_source = _resolve_prediction_sample_ids(df)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=clean_feature_names,
                index=X_train.index,
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=clean_feature_names,
                index=X_val.index,
            )

            model = _build_f_perf_model(
                use_phase1_config=use_phase1_config,
                champion_model_family=champion_model_family,
                ag_results_path=ag_results_path,
                n_samples=len(X_train),
            )
            model.fit(X_train_scaled, y_train)

            if n_classes == 2:
                y_probs_full = model.predict_proba(X_val_scaled)
                y_probs = y_probs_full[:, 1]
                score = roc_auc_score(y_val, y_probs) if metric == 'roc_auc' else average_precision_score(y_val, y_probs)
                scores.append(float(score))

                if return_predictions:
                    for i, sample_idx in enumerate(val_idx):
                        all_predictions.append({
                            'sample_id': cv_sample_ids[sample_idx],
                            'true_label': y[sample_idx],
                            'true_label_encoded': int(y_val[i]),
                            'pred_proba_class0': float(y_probs_full[i, 0]),
                            'pred_proba_class1': float(y_probs_full[i, 1]),
                            'fold': fold_idx + 1,
                        })
            else:
                score = model.score(X_val_scaled, y_val)
                scores.append(float(score))
                if return_predictions:
                    print("Warning: return_predictions not supported for multi-class classification")

        mean_score = float(np.mean(scores))

        if return_predictions:
            predictions_dict = {
                'predictions': all_predictions or [],
                'mean_score': mean_score,
                'label_names': label_names,
                'n_samples': len(all_predictions or []),
                'k_folds': k_folds,
                'metric': metric,
                'evaluation_mode': resolved_mode,
                'sample_id_source': cv_sample_id_source,
            }
            return round(mean_score, 4), predictions_dict
        return round(mean_score, 4)
    
    except Exception as e:
        print(f"Error in calculate_f_perf: {str(e)}")
        import traceback
        traceback.print_exc()
        return _return_empty(f"exception: {str(e)}")


# ============================================================================
# 目标函数 2: f_size - 特征集大小评估（待实现）
# ============================================================================

def calculate_f_size(features_list: List[str]) -> float:
    """
    计算特征子集的大小惩罚 f_size(S)。
    
    设计：特征数量越少越好，归一化到 [0, 1]。
    
    Args:
        features_list: 特征子集列表
    
    Returns:
        float: 大小惩罚得分，范围 [0.0, 1.0]
    
    TODO: Phase 2 后续实现
    """
    # 占位实现
    return 1.0 - min(len(features_list) / 100.0, 1.0)


# ============================================================================
# 目标函数 3: f_bio - 生物学可解释性评估
# ============================================================================

def calculate_f_bio(
    features_list: List[str],
    priors_dict: Optional[Dict[str, float]] = None,
    taxonomy_map: Optional[Dict] = None,
    pathway_map: Optional[Dict] = None,
    w_lit: float = 0.7,
    w_mech: float = 0.3,
    gamma: float = 0.1,
    sigmoid_k: float = 1.0
) -> float:
    """
    计算特征子集的生物学可解释性 f_bio(S)。
    
    核心公式:
        f_bio(S) = (1/|S|) × Σ P0(i) + γ × E(S)/|S|
    
    其中:
        - P0(i): 单体先验评分（方案 B：P_prior + topology bonus）
        - E(S): 网络连通度（反应边数量）
        - γ: 网络连通度权重系数（默认 0.1）
    
    设计理念:
        1. 先验得分项: 评估特征的综合生物学先验与工程拓扑奖励
        2. 网络连通度项: 评估特征间的生化反应关系
        3. 归一化: 除以 |S| 确保不同大小的特征子集可比
    
    Args:
        features_list: 特征子集列表
        priors_dict: Phase 0 先验字典 {HMDB_ID: bio_prior_norm 或 legacy confidence}
        taxonomy_map: 分类学映射 {HMDB_ID: {class, sub_class, ...}}
        pathway_map: 通路映射 {pathway_name: [HMDB_IDs]}
        w_lit: 先验轨权重（兼容旧命名，默认 0.7）
        w_mech: 拓扑轨权重（兼容旧命名，默认 0.3）
        gamma: 网络连通度权重系数（默认 0.1）
        sigmoid_k: 旧版 raw prior 的 Sigmoid 温度系数（默认 1.0）
    
    Returns:
        float: 生物学可解释性得分，通常在 [0, 2] 区间
    
    Examples:
        >>> features = ['HMDB0000001', 'HMDB0000002', 'Ratio_HMDB0000001_HMDB0000002']
        >>> priors = {'HMDB0000001': 2.5, 'HMDB0000002': -1.5}
        >>> score = calculate_f_bio(features, priors_dict=priors)
        >>> print(f"f_bio = {score:.4f}")
        f_bio = 0.8234
    
    Notes:
        - 如果知识库未提供，会尝试从 storage/ 目录自动加载
        - 网络连通度计算使用 ReactionCheckerTool
        - 除零保护: |S| = 0 时返回 0.0
    """
    try:
        # 边界情况: 空特征集
        if not features_list or len(features_list) == 0:
            return 0.0
        
        n_features = len(features_list)
        
        # ======================================================================
        # 1. 加载知识库（如果未提供）
        # ======================================================================
        if priors_dict is None:
            priors_dict = _load_priors_dict()
        
        if taxonomy_map is None:
            taxonomy_map = _load_taxonomy_map()
        
        if pathway_map is None:
            pathway_map = _load_pathway_map()
        
        # ======================================================================
        # 2. 计算先验得分项: (1/|S|) × Σ P0(i)
        # ======================================================================
        p0_scores = []
        for feature in features_list:
            p0 = _calculate_single_p0(
                feature_name=feature,
                priors_dict=priors_dict,
                taxonomy_map=taxonomy_map,
                pathway_map=pathway_map,
                w_lit=w_lit,
                w_mech=w_mech,
                sigmoid_k=sigmoid_k
            )
            p0_scores.append(p0)
        
        # 计算平均先验得分
        avg_p0 = float(np.mean(p0_scores))
        
        # ======================================================================
        # 3. 计算网络连通度项: γ × E(S)/|S|
        # ======================================================================
        # 3.1 提取所有唯一的 HMDB ID
        hmdb_ids = set()
        for feature in features_list:
            # 使用正则表达式提取 HMDB ID
            matches = re.findall(r'HMDB\d+', feature)
            hmdb_ids.update(matches)
        
        # 3.2 计算反应边数量
        if len(hmdb_ids) >= 2:
            # 只有至少 2 个代谢物才可能有连接
            from src.tools.domain.reaction_checker import ReactionCheckerTool
            
            try:
                checker = ReactionCheckerTool()
                connected_pairs = checker.find_connected_pairs(list(hmdb_ids))
                edges_count = len(connected_pairs)
            except Exception as e:
                print(f"Warning: Failed to calculate network connectivity: {e}")
                edges_count = 0
        else:
            edges_count = 0
        
        # 3.3 计算网络连通度（除零保护）
        if n_features > 0:
            network_connectivity = edges_count / n_features
        else:
            network_connectivity = 0.0
        
        # ======================================================================
        # 4. 组装最终得分
        # ======================================================================
        f_bio = avg_p0 + gamma * network_connectivity
        
        return float(f_bio)
    
    except Exception as e:
        print(f"Error in calculate_f_bio: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0


def _load_priors_dict() -> Dict[str, float]:
    """
    从 storage/phase0_cache.json 加载先验字典。
    
    支持两种数据结构:
    1. 旧结构: cache_data['priors']
    2. 新结构: cache_data['diseases'][disease_key]['confirmed_biomarkers']
    
    Returns:
        dict: {HMDB_ID: bio_prior_norm 或 legacy confidence_score}
    """
    try:
        cache_path = os.path.join('storage', 'phase0_cache.json')
        
        if not os.path.exists(cache_path):
            print(f"Warning: Phase 0 cache not found at {cache_path}, using empty priors")
            return {}
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 尝试旧结构（向后兼容）
        if 'priors' in cache_data:
            priors_dict = cache_data['priors']
            print(f"Info: Loaded {len(priors_dict)} priors from Phase 0 cache (legacy structure)")
            return priors_dict
        
        # 使用新结构（正确的数据契约）
        if 'diseases' in cache_data:
            priors_dict = {}
            
            # 安全遍历所有疾病
            diseases = cache_data.get('diseases', {})
            
            if not isinstance(diseases, dict):
                print(f"Warning: 'diseases' is not a dict, using empty priors")
                return {}
            
            for disease_key, disease_data in diseases.items():
                # 确保 disease_data 是字典
                if not isinstance(disease_data, dict):
                    print(f"Warning: Disease data for '{disease_key}' is not a dict, skipping")
                    continue
                
                # 提取 confirmed_biomarkers 数组
                biomarkers = disease_data.get('confirmed_biomarkers', [])
                
                # 确保 biomarkers 是列表
                if not isinstance(biomarkers, list):
                    print(f"Warning: confirmed_biomarkers for '{disease_key}' is not a list, skipping")
                    continue
                
                # 遍历每个生物标志物
                for biomarker in biomarkers:
                    # 确保 biomarker 是字典
                    if not isinstance(biomarker, dict):
                        continue
                    
                    hmdb_id = biomarker.get('id', '')
                    prior_value = biomarker.get('bio_prior_norm', None)
                    if prior_value is None:
                        prior_value = biomarker.get('confidence_score', biomarker.get('bio_prior_raw', 0.0))
                    
                    # 只添加有效的 HMDB ID
                    if hmdb_id and isinstance(hmdb_id, str) and hmdb_id.strip():
                        priors_dict[hmdb_id] = float(prior_value)
            
            print(f"Info: Loaded {len(priors_dict)} priors from Phase 0 cache (diseases structure)")
            return priors_dict
        
        # 数据结构不匹配
        print(f"Warning: Unknown data structure in {cache_path}, using empty priors")
        return {}
    
    except Exception as e:
        print(f"Warning: Failed to load priors dict: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _load_taxonomy_map() -> Dict:
    """
    从 storage/taxonomy_map.json 加载分类学映射。
    
    支持两种数据结构:
    1. 直接映射: {HMDB_ID: {class, sub_class, ...}}
    2. 嵌套结构: 从 diseases 中提取
    
    Returns:
        dict: {HMDB_ID: {class, sub_class, ...}}
    """
    try:
        taxonomy_path = os.path.join('storage', 'taxonomy_map.json')
        
        if not os.path.exists(taxonomy_path):
            print(f"Warning: Taxonomy map not found at {taxonomy_path}, using empty map")
            return {}
        
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy_data = json.load(f)
        
        # 如果是直接映射结构，直接返回
        if taxonomy_data and not isinstance(list(taxonomy_data.values())[0], dict):
            print(f"Info: Loaded {len(taxonomy_data)} entries from taxonomy map")
            return taxonomy_data
        
        # 如果是嵌套结构，提取映射
        taxonomy_map = {}
        for key, value in taxonomy_data.items():
            if isinstance(value, dict):
                taxonomy_map[key] = value
        
        print(f"Info: Loaded {len(taxonomy_map)} entries from taxonomy map")
        return taxonomy_map
    
    except Exception as e:
        print(f"Warning: Failed to load taxonomy map: {e}")
        return {}


def _load_pathway_map() -> Dict:
    """
    从 storage/pathbank_pathway_map.json 加载通路映射。
    
    支持两种数据结构:
    1. 直接映射: {pathway_name: [HMDB_IDs]}
    2. 嵌套结构: 从 diseases 中提取
    
    Returns:
        dict: {pathway_name: [HMDB_IDs]}
    """
    try:
        pathway_path = os.path.join('storage', 'pathbank_pathway_map.json')
        
        if not os.path.exists(pathway_path):
            print(f"Warning: Pathway map not found at {pathway_path}, using empty map")
            return {}
        
        with open(pathway_path, 'r', encoding='utf-8') as f:
            pathway_data = json.load(f)
        
        # 如果是直接映射结构，直接返回
        if pathway_data and isinstance(list(pathway_data.values())[0], list):
            print(f"Info: Loaded {len(pathway_data)} pathways from pathway map")
            return pathway_data
        
        # 如果是嵌套结构，提取映射
        pathway_map = {}
        for key, value in pathway_data.items():
            if isinstance(value, list):
                pathway_map[key] = value
        
        print(f"Info: Loaded {len(pathway_map)} pathways from pathway map")
        return pathway_map
    
    except Exception as e:
        print(f"Warning: Failed to load pathway map: {e}")
        return {}


# ============================================================================
# 目标函数 4: f_stab - 特征稳定性评估（待实现）
# ============================================================================

def calculate_f_stab(
    data_path: str,
    target_column: str,
    features_list: List[str],
    n_bootstrap: int = 10
) -> float:
    """
    计算特征子集的稳定性 f_stab(S)。
    
    设计：通过 Bootstrap 采样评估特征重要性的稳定性。
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名称
        features_list: 特征子集列表
        n_bootstrap: Bootstrap 采样次数
    
    Returns:
        float: 稳定性得分，范围 [0.0, 1.0]
    
    TODO: Phase 2 后续实现
    """
    # 占位实现
    return 0.8


# ============================================================================
# 综合评估函数
# ============================================================================

def evaluate_feature_subset(
    data_path: str,
    target_column: str,
    features_list: List[str],
    ag_results_path: str = 'data/autogluon_training_results.json',
    weights: Optional[Dict[str, float]] = None,
    priors_dict: Optional[Dict[str, float]] = None,
    taxonomy_map: Optional[Dict] = None,
    pathway_map: Optional[Dict] = None
) -> Dict[str, float]:
    """
    综合评估特征子集的多个目标函数。
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名称
        features_list: 特征子集列表
        ag_results_path: AutoGluon 训练结果路径
        weights: 各目标函数的权重字典
        priors_dict: Phase 0 先验字典（可选）
        taxonomy_map: 分类学映射（可选）
        pathway_map: 通路映射（可选）
    
    Returns:
        dict: 包含各目标函数得分的字典
    
    Examples:
        >>> features = ['HMDB0000001', 'HMDB0000002']
        >>> results = evaluate_feature_subset('data/cleaned.csv', 'Group', features)
        >>> print(results)
        {'f_perf': 0.8523, 'f_size': 0.98, 'f_bio': 0.75, 'f_stab': 0.8}
    """
    if weights is None:
        weights = {'f_perf': 1.0, 'f_size': 0.3, 'f_bio': 0.5, 'f_stab': 0.2}
    
    results = {
        'f_perf': calculate_f_perf(data_path, target_column, features_list, ag_results_path=ag_results_path),
        'f_size': calculate_f_size(features_list),
        'f_bio': calculate_f_bio(features_list, priors_dict, taxonomy_map, pathway_map),
        'f_stab': calculate_f_stab(data_path, target_column, features_list)
    }
    
    # 计算加权总分
    weighted_score = sum(results[k] * weights.get(k, 0) for k in results)
    results['weighted_total'] = round(weighted_score, 4)
    
    return results


# ============================================================================
# 工具函数：返回 JSON 格式结果
# ============================================================================

def calculate_f_perf_json(
    data_path: str,
    target_column: str,
    features_list: List[str],
    k_folds: int = 5,
    ag_results_path: str = 'data/autogluon_training_results.json'
) -> str:
    """
    JSON 格式的 f_perf 计算函数，用于与其他工具保持一致的接口。
    
    Returns:
        str: JSON 字符串，包含 success, f_perf, n_features 等信息
    """
    try:
        score = calculate_f_perf(
            data_path=data_path,
            target_column=target_column,
            features_list=features_list,
            champion_model_family='RandomForest',
            k_folds=k_folds,
            ag_results_path=ag_results_path,
        )
        
        result = {
            'success': True,
            'f_perf': score,
            'n_features': len(features_list),
            'metric': 'auprc',
            'k_folds': k_folds
        }
        
        return json.dumps(result)
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'f_perf': 0.0
        }
        
        return json.dumps(result)


# ============================================================================
# 目标函数 5: f_corr - 特征冗余度评估
# ============================================================================

# 全局数据缓存，避免高频 I/O 导致性能崩塌
_DATA_CACHE = {}


def _get_cached_data(data_path: str) -> pd.DataFrame:
    """
    获取缓存的 DataFrame，若无则加载并缓存。
    
    CRITICAL: 该函数是性能优化的核心，避免每次调用都从磁盘读取 CSV。
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        pd.DataFrame: 缓存的数据框
    
    Examples:
        >>> df1 = _get_cached_data('data/cleaned.csv')  # 第一次从磁盘加载
        >>> df2 = _get_cached_data('data/cleaned.csv')  # 第二次从缓存获取（极速）
        >>> assert df1 is df2  # 同一个对象
    """
    if data_path not in _DATA_CACHE:
        print(f"Info: Loading data from {data_path} into cache...")
        _DATA_CACHE[data_path] = pd.read_csv(data_path)
        print(f"Info: Data cached successfully ({len(_DATA_CACHE[data_path])} rows)")
    
    return _DATA_CACHE[data_path]


def calculate_f_corr(data_path: str, features_list: List[str]) -> float:
    """
    计算特征子集 S 的冗余度 f_corr(S)。
    
    核心公式:
        f_corr(S) = (1/C(|S|,2)) × Σ |ρ(i,j)|
    
    其中:
        - ρ(i,j): 特征 i 和 j 的 Spearman 相关系数
        - C(|S|,2): 组合数，即 |S|×(|S|-1)/2
        - 取绝对值后求平均，范围 [0, 1]
    
    设计理念:
        1. 内存缓存机制: 使用 _DATA_CACHE 避免高频 I/O
        2. 矩阵化运算: 利用 df.corr() 和 np.triu_indices_from 避免循环
        3. 边界处理: 特征数 < 2 时返回 0.0
        4. 异常保护: 使用 np.nanmean 防止 NaN 毒化，崩溃时返回 1.0（最大惩罚）
    
    性能优化:
        - 第一次调用: ~100ms（读取 CSV）
        - 后续调用: ~5ms（从缓存获取）
        - 相关系数计算: O(n²) 但底层为 C 语言优化
    
    Args:
        data_path: 清洗后的完整数据集路径
        features_list: 当前评估的特征子集
    
    Returns:
        float: 冗余度得分，范围 [0.0, 1.0]。越低越好。
    
    Examples:
        >>> features = ['HMDB0000001', 'HMDB0000002', 'HMDB0000003']
        >>> redundancy = calculate_f_corr('data/cleaned.csv', features)
        >>> print(f"Redundancy: {redundancy:.4f}")
        Redundancy: 0.3245
        
        >>> # 单个特征，无冗余
        >>> single_feature = ['HMDB0000001']
        >>> redundancy = calculate_f_corr('data/cleaned.csv', single_feature)
        >>> print(f"Redundancy: {redundancy:.4f}")
        Redundancy: 0.0000
    
    Notes:
        - 该函数会被外层 Agent 高频调用数百次，缓存机制至关重要
        - 使用 Spearman 相关系数（秩相关），对非线性关系更鲁棒
        - 只提取上三角矩阵（k=1），避免重复计算和自相关
        - 异常时返回 1.0（最大惩罚），迫使优化器放弃该危险组合
    """
    # ======================================================================
    # 1. 边界条件：如果特征不足 2 个，不存在内部冗余
    # ======================================================================
    if not features_list or len(features_list) < 2:
        return 0.0
    
    try:
        # ==================================================================
        # 2. 极速获取数据（从缓存）
        # ==================================================================
        df = _get_cached_data(data_path)
        
        # ==================================================================
        # 3. 过滤有效特征（优雅降级：支持标准化匹配）
        # ==================================================================
        # Phase 1 已在源头标准化特征名，但为了向后兼容和鲁棒性，
        # 仍然保留动态标准化匹配能力
        from src.utils.feature_name_standardizer import validate_feature_names
        
        # 使用标准化验证特征（优雅降级）
        validation_result = validate_feature_names(
            feature_names=features_list,
            dataset_columns=df.columns.tolist(),
            auto_standardize=True  # 启用自动标准化匹配（向后兼容）
        )
        
        valid_features = validation_result['valid']
        feature_mapping = validation_result['mapping']
        
        # 使用映射后的列名
        mapped_features = [feature_mapping[f] for f in valid_features]
        
        # 再次检查过滤后的数量
        if len(mapped_features) < 2:
            print(f"Warning: Only {len(mapped_features)} valid features found, returning 0.0")
            return 0.0
        
        # ==================================================================
        # 4. 提取特征矩阵（使用映射后的列名）
        # ==================================================================
        X = df[mapped_features]
        
        # ==================================================================
        # 5. 矩阵化计算 Spearman 相关系数，并取绝对值
        # 这一步极其高效，底层为 C 语言优化
        # ==================================================================
        corr_matrix = X.corr(method='spearman').abs()
        
        # ==================================================================
        # 6. 提取右上三角元素的索引（k=1 表示不包含对角线的自相关 1.0）
        # ==================================================================
        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        
        # ==================================================================
        # 7. 获取上三角的值
        # ==================================================================
        pairwise_correlations = corr_matrix.values[upper_tri_indices]
        
        # ==================================================================
        # 8. 计算平均值（免疫 NaN）
        # ==================================================================
        mean_redundancy = float(np.nanmean(pairwise_correlations))
        
        # 如果全部都是 NaN（极端罕见），返回 0.0
        if np.isnan(mean_redundancy):
            print(f"Warning: All correlations are NaN, returning 0.0")
            return 0.0
        
        return round(mean_redundancy, 4)
    
    except Exception as e:
        print(f"Warning: calculate_f_corr failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # 若发生异常计算，返回 1.0（最大冗余惩罚），迫使优化器放弃该危险组合
        return 1.0


def calculate_f_corr_json(data_path: str, features_list: List[str]) -> str:
    """
    JSON 格式的 f_corr 计算函数，用于与其他工具保持一致的接口。
    
    Args:
        data_path: 数据文件路径
        features_list: 特征子集列表
    
    Returns:
        str: JSON 字符串，包含 success, f_corr, n_features 等信息
    
    Examples:
        >>> result_json = calculate_f_corr_json('data/cleaned.csv', ['HMDB0000001', 'HMDB0000002'])
        >>> import json
        >>> result = json.loads(result_json)
        >>> print(f"Redundancy: {result['f_corr']:.4f}")
        Redundancy: 0.3245
    """
    try:
        score = calculate_f_corr(data_path, features_list)
        
        result = {
            'success': True,
            'f_corr': score,
            'n_features': len(features_list),
            'metric': 'spearman_correlation'
        }
        
        return json.dumps(result)
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'f_corr': 1.0  # 异常时返回最大惩罚
        }
        
        return json.dumps(result)


# ============================================================================
# 目标函数 6: f_cost - 临床成本评估
# ============================================================================

def calculate_f_cost(
    features_list: List[str],
    taxonomy_map: Optional[Dict[str, dict]] = None,
    pathway_map: Optional[Dict[str, list]] = None,
    n_max: int = 20,
    k: float = 2.0,
    data_path: Optional[str] = None,
    aggregation_discount: bool = True
) -> float:
    """
    计算特征子集 S 的临床成本 f_cost(S)。
    
    核心设计理念（改进版）：
    在临床代谢组学中，成本取决于试剂盒需要检测的**"实体分子数量"**。
    
    改进版引入两个关键修正：
    1. **观测集交集 (Observed Intersection)**:
       - Sum/Pathway 特征不应按理论字典中的全部分子计算
       - 必须与当前数据集中实际存在的 HMDB ID 取交集
       - 避免将"理论分子总数"等同于"实际检测成本"
    
    2. **规模效应打折 (Economy of Scale Discount)**:
       - 同类/同通路分子在临床质谱中可"打包检测"
       - 检测 16 个同类脂质的成本 ≠ 检测 1 个脂质的 16 倍
       - 使用平方根折算: sqrt(16) = 4 个等效实体
    
    核心公式:
        Cost = (N_effective / N_max)^k
    
    其中:
        N_effective = sum of effective entities for all features
        
        For each feature:
        - 单体特征: 1 个实体
        - Ratio 特征: 2 个实体（必须测两个分子）
        - Sum 特征: ceil(sqrt(len(actual_ids_in_data))) 个等效实体
        - Pathway 特征: ceil(sqrt(len(actual_ids_in_data))) 个等效实体
    
    Args:
        features_list: 特征子集列表
        taxonomy_map: 分类学映射 {HMDB_ID: {class, sub_class, ...}}
        pathway_map: 通路映射 {pathway_name: [HMDB_IDs]}
        n_max: 最大特征预算（默认 20）
        k: 惩罚指数（默认 2.0）
        data_path: 数据文件路径（用于提取实际存在的 HMDB ID）
        aggregation_discount: 是否启用规模效应打折（默认 True）
    
    Returns:
        float: 临床成本得分，范围 [0.0, +∞)。越低越好。
    
    Examples:
        >>> # 示例 1: 3 个单体特征
        >>> features = ['HMDB0000001', 'HMDB0000002', 'HMDB0000003']
        >>> cost = calculate_f_cost(features, n_max=20, k=2.0)
        >>> print(f"Cost: {cost:.4f}")
        Cost: 0.0225  # (3/20)^2 = 0.0225
        
        >>> # 示例 2: Sum 特征（假设数据集中有 16 个该类分子）
        >>> features = ['Sum_Amino_acids_peptides_and_analogues']
        >>> cost = calculate_f_cost(features, data_path='data.csv')
        >>> # N_effective = ceil(sqrt(16)) = 4
        >>> # Cost = (4/20)^2 = 0.04
        
        >>> # 示例 3: 混合特征
        >>> features = ['HMDB0000001', 'HMDB0000002', 'Sum_Lipids', 'Pathway_X']
        >>> # N_effective = 1 + 1 + ceil(sqrt(25)) + ceil(sqrt(10)) = 2 + 5 + 4 = 11
        >>> # Cost = (11/20)^2 = 0.3025
    
    Notes:
        - 如果 data_path 未提供，Sum/Pathway 特征使用保守估计（固定值）
        - 异常时返回 999.0（极端惩罚），迫使优化器放弃该组合
        - 空特征列表返回 0.0
    """
    # ======================================================================
    # 1. 边界条件：空特征列表
    # ======================================================================
    if not features_list:
        return 0.0
    
    try:
        # ==================================================================
        # 2. 加载知识库（如果未提供）
        # ==================================================================
        if taxonomy_map is None:
            taxonomy_map = _load_taxonomy_map()
        
        if pathway_map is None:
            pathway_map = _load_pathway_map()
        
        # ==================================================================
        # 3. 提取数据集中实际存在的 HMDB ID（观测集）
        # ==================================================================
        feature_provenance_index = _load_feature_provenance_index()
        feature_registry_index = _load_engineered_feature_registry_index()
        dataset_hmdb_set, observed_source = _resolve_cost_observed_hmdb_set(
            data_path=data_path,
            taxonomy_map=taxonomy_map,
            feature_provenance_index=feature_provenance_index,
            feature_registry_index=feature_registry_index,
        )
        print(
            f"[Cost] Observed HMDB universe: {len(dataset_hmdb_set)} "
            f"(source={observed_source})"
        )
        
        # ==================================================================
        # 4. 计算有效实体数量（引入观测集交集和规模效应打折）
        # ==================================================================
        total_effective_n = 0
        for feature in features_list:
            feature_upper = str(feature or "").strip().upper()
            # ==============================================================
            # 4.1 组合特征：加和类 (Sum/TaxSum)
            # ==============================================================
            if feature_upper.startswith('SUM_') or feature_upper.startswith('TAXSUM_'):
                resolution_source, actual_ids_in_data = _resolve_feature_entity_ids(
                    feature_name=feature,
                    dataset_hmdb_set=dataset_hmdb_set,
                    taxonomy_map=taxonomy_map,
                    pathway_map=pathway_map,
                    feature_provenance_index=feature_provenance_index,
                    feature_registry_index=feature_registry_index,
                )
                if actual_ids_in_data:
                    if aggregation_discount:
                        # 关键修正2: 规模效应打折（平方根）
                        discounted_n = math.ceil(math.sqrt(len(actual_ids_in_data)))
                    else:
                        # 不打折，直接使用实际数量
                        discounted_n = len(actual_ids_in_data)
                    
                    total_effective_n += discounted_n
                    print(
                        f"[Cost] {feature}: {len(actual_ids_in_data)} actual "
                        f"({resolution_source}) → {discounted_n} effective"
                    )
                else:
                    # 实体降级保护：如果找不到对应的 HMDB ID，记为 1 个实体
                    total_effective_n += 1
                    print(f"[Cost] {feature}: No matching IDs, using 1 entity")
            
            # ==============================================================
            # 4.2 组合特征：通路类 (Pathway)
            # ==============================================================
            elif feature_upper.startswith('PATHWAY_'):
                resolution_source, actual_ids_in_data = _resolve_feature_entity_ids(
                    feature_name=feature,
                    dataset_hmdb_set=dataset_hmdb_set,
                    taxonomy_map=taxonomy_map,
                    pathway_map=pathway_map,
                    feature_provenance_index=feature_provenance_index,
                    feature_registry_index=feature_registry_index,
                )
                if actual_ids_in_data:
                    if aggregation_discount:
                        discounted_n = math.ceil(math.sqrt(len(actual_ids_in_data)))
                    else:
                        discounted_n = len(actual_ids_in_data)

                    total_effective_n += discounted_n
                    print(
                        f"[Cost] {feature}: {len(actual_ids_in_data)} actual "
                        f"({resolution_source}) → {discounted_n} effective"
                    )
                else:
                    total_effective_n += 1
                    print(f"[Cost] {feature}: No matching IDs, using 1 entity")
            
            # ==============================================================
            # 4.3 组合特征：比值类 (Ratio)
            # ==============================================================
            elif feature_upper.startswith('RATIO_'):
                _, ratio_ids = _resolve_feature_entity_ids(
                    feature_name=feature,
                    dataset_hmdb_set=dataset_hmdb_set,
                    taxonomy_map=taxonomy_map,
                    pathway_map=pathway_map,
                    feature_provenance_index=feature_provenance_index,
                    feature_registry_index=feature_registry_index,
                )
                total_effective_n += max(2, len(ratio_ids)) if ratio_ids else 2
            
            # ==============================================================
            # 4.4 单体特征
            # ==============================================================
            else:
                # 单体特征成本为 1
                total_effective_n += 1
        
        # ==================================================================
        # 5. 计算指数惩罚成本
        # ==================================================================
        if total_effective_n == 0:
            return 1.0
        
        cost_score = (total_effective_n / n_max) ** k
        
        print(f"[Cost] Total effective N: {total_effective_n}, Cost: {cost_score:.4f}")
        
        return round(float(cost_score), 4)
    
    except Exception as e:
        print(f"Warning: calculate_f_cost failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # 异常重罚
        return 999.0


def calculate_f_cost_json(
    features_list: List[str],
    taxonomy_map: Optional[Dict[str, dict]] = None,
    pathway_map: Optional[Dict[str, list]] = None,
    n_max: int = 20,
    k: float = 2.0,
    data_path: Optional[str] = None,
    aggregation_discount: bool = True
) -> str:
    """
    JSON 格式的 f_cost 计算函数，用于与其他工具保持一致的接口。
    
    Args:
        features_list: 特征子集列表
        taxonomy_map: 分类学映射（可选）
        pathway_map: 通路映射（可选）
        n_max: 最大特征预算（默认 20）
        k: 惩罚指数（默认 2.0）
        data_path: 数据文件路径（可选）
        aggregation_discount: 是否启用规模效应打折
    
    Returns:
        str: JSON 字符串，包含 success, f_cost, n_base, n_features 等信息
    
    Examples:
        >>> result_json = calculate_f_cost_json(['HMDB0000001', 'HMDB0000002'])
        >>> import json
        >>> result = json.loads(result_json)
        >>> print(f"Cost: {result['f_cost']:.4f}, Base molecules: {result['n_base']}")
        Cost: 0.0100, Base molecules: 2
    """
    try:
        # 加载知识库
        if taxonomy_map is None:
            taxonomy_map = _load_taxonomy_map()
        
        if pathway_map is None:
            pathway_map = _load_pathway_map()
        
        # 计算成本
        cost = calculate_f_cost(
            features_list=features_list,
            taxonomy_map=taxonomy_map,
            pathway_map=pathway_map,
            n_max=n_max,
            k=k,
            data_path=data_path,
            aggregation_discount=aggregation_discount,
        )
        
        # 计算底层分子数量（用于调试）
        unique_base_molecules = set()
        feature_provenance_index = _load_feature_provenance_index()
        feature_registry_index = _load_engineered_feature_registry_index()
        for feature in features_list:
            feature_upper = str(feature or "").strip().upper()
            resolution_source, involved = _resolve_feature_entity_ids(
                feature_name=feature,
                dataset_hmdb_set=set(),
                taxonomy_map=taxonomy_map,
                pathway_map=pathway_map,
                feature_provenance_index=feature_provenance_index,
                feature_registry_index=feature_registry_index,
            )
            if feature_upper.startswith('SUM_') or feature_upper.startswith('TAXSUM_') or feature_upper.startswith('PATHWAY_'):
                unique_base_molecules.update(involved if involved else [feature])
            elif feature_upper.startswith('RATIO_'):
                found_ids = re.findall(r'HMDB\d+', feature)
                unique_base_molecules.update(found_ids if found_ids else feature.replace('Ratio_', '').split('_'))
            else:
                found_ids = re.findall(r'HMDB\d+', feature)
                unique_base_molecules.update(found_ids if found_ids else [feature])
        
        n_base = len(unique_base_molecules)
        
        result = {
            'success': True,
            'f_cost': cost,
            'n_base': n_base,
            'n_features': len(features_list),
            'n_max': n_max,
            'k': k,
            'aggregation_discount': aggregation_discount,
            'cost_formula': f'({n_base}/{n_max})^{k}'
        }
        
        return json.dumps(result)
    
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'f_cost': 999.0  # 异常时返回极端惩罚
        }
        
        return json.dumps(result)
