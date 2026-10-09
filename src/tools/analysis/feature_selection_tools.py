# src/tools/analysis/feature_selection_tools.py
"""
Feature Selection Tools for MetaboAgent

This module provides advanced feature selection methods for metabolomics data analysis.
All tools follow a privacy-preserving design pattern: they accept file paths instead of 
DataFrames, process data locally, and return only metadata (no raw data).

Methods include:
- Core methods: Random Forest, Lasso, Mutual Information, Welch t-test, ANOVA
- Composite statistical methods: Statistical (F-test + MI)
- Advanced methods: CFS, FCBF, ReliefF, Linear SVM, SBS, mRMR, Genetic Algorithm
- Stability selection: Iterative resampling with consensus feature identification
- Dataset filtering: Create filtered datasets with selected features

Additional dependencies required:
pip install mlxtend skrebate skfeature-chappers sklearn-genetic-opt mrmr-selection

All tools are decorated with @tool for seamless LangChain/LangGraph integration.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
import io
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from scipy import stats
import json
import os
import glob
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


PHASE1_DEFAULT_SELECTOR_FAMILY = [
    "run_elasticnet_selector",
    "run_lasso_selector",
    "run_random_forest_selector",
    "run_lightgbm_selector",
    "run_mrmr_selector",
    "run_t_test_selector",
    "run_fdr_effect_size_selector",
]

PHASE1_SELECTOR_DISPLAY_NAMES = {
    "run_elasticnet_selector": "Elastic Net Logistic",
    "run_lasso_selector": "L1 Logistic/Lasso",
    "run_random_forest_selector": "Random Forest",
    "run_lightgbm_selector": "LightGBM",
    "run_mrmr_selector": "mRMR",
    "run_t_test_selector": "Welch t-test",
    "run_fdr_effect_size_selector": "FDR/effect-size",
}

PHASE1_SELECTOR_ALIASES = {
    "elastic net logistic": "run_elasticnet_selector",
    "elastic_net_logistic": "run_elasticnet_selector",
    "elastic net": "run_elasticnet_selector",
    "elasticnet": "run_elasticnet_selector",
    "run_elasticnet_selector": "run_elasticnet_selector",
    "l1 logistic/lasso": "run_lasso_selector",
    "l1 logistic": "run_lasso_selector",
    "l1_lasso": "run_lasso_selector",
    "random_forest": "run_random_forest_selector",
    "rf": "run_random_forest_selector",
    "run_random_forest_selector": "run_random_forest_selector",
    "lightgbm": "run_lightgbm_selector",
    "run_lightgbm_selector": "run_lightgbm_selector",
    "lasso": "run_lasso_selector",
    "run_lasso_selector": "run_lasso_selector",
    "mrmr": "run_mrmr_selector",
    "run_mrmr_selector": "run_mrmr_selector",
    "welch t-test": "run_t_test_selector",
    "welch_t_test": "run_t_test_selector",
    "t_test": "run_t_test_selector",
    "run_t_test_selector": "run_t_test_selector",
    "fdr/effect-size": "run_fdr_effect_size_selector",
    "fdr_effect_size": "run_fdr_effect_size_selector",
    "effect size": "run_fdr_effect_size_selector",
    "run_fdr_effect_size_selector": "run_fdr_effect_size_selector",
}


def _run_mrmr_classif_silently(X, y, K: int, n_jobs: int = 1):
    """
    Run mrmr-selection while suppressing any residual tqdm/progress output.

    `show_progress=False` is not sufficient in some environments because the
    underlying library may still emit progress bars to stdout/stderr.
    """
    import mrmr

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return mrmr.mrmr_classif(
            X=X,
            y=y,
            K=K,
            n_jobs=n_jobs,
            show_progress=False,
        )

# --- [优化] 提取公共的数据加载和预处理逻辑 ---
def _selector_success(result: Dict[str, Any]) -> str:
    """Normalize selector success payloads for SOP-generated code."""
    payload = {"success": True, **result}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _selector_error(message: str) -> str:
    """Normalize selector error payloads for SOP-generated code."""
    return json.dumps({"success": False, "error": message}, ensure_ascii=False, indent=2)


def _load_and_prepare_data(data_path: str, group_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    内部帮助函数，用于加载、清洗和准备数据。
    这个函数假设缺失值已经在之前的步骤中处理过了。
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("不支持的文件格式，请使用 .csv 或 .xlsx")

    return _prepare_selection_inputs_from_df(df, group_col)


def _prepare_selection_inputs_from_df(
    df: pd.DataFrame,
    group_col: str,
    apply_standard_scaling: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare selector inputs from an in-memory dataframe to avoid repeated CSV I/O."""
    working_df = df.copy()

    if group_col not in working_df.columns:
        if "target" in working_df.columns:
            group_col = "target"
        else:
            raise ValueError(f"分组列 '{group_col}' 不存在")

    if group_col != "target" and "target" in working_df.columns:
        group_non_na = int(working_df[group_col].notna().sum())
        target_non_na = int(working_df["target"].notna().sum())
        if target_non_na > group_non_na:
            group_col = "target"

    feature_names = _resolve_feature_columns(working_df, group_col)
    if not feature_names:
        raise ValueError("没有可用于特征选择的数值特征")

    X = working_df.loc[:, feature_names].copy()
    y = working_df[group_col]

    valid_label_mask = y.notna()
    X = X.loc[valid_label_mask].copy()
    y = y.loc[valid_label_mask].copy()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        median_values = X.median(numeric_only=True)
        X = X.fillna(median_values)
        X = X.fillna(0.0)

    if apply_standard_scaling:
        scaler = StandardScaler()
        X_prepared = scaler.fit_transform(X)
        X_prepared = pd.DataFrame(X_prepared, columns=feature_names, index=X.index)
    else:
        X_prepared = X.copy()
    return X_prepared, y, feature_names


def _resolve_requested_feature_count(
    n_features: Optional[int],
    feature_names: List[str],
) -> int:
    if n_features is None:
        n_features = max(10, int(len(feature_names) * 0.30))
    return max(1, min(int(n_features), len(feature_names)))


def _canonicalize_phase1_selector_name(method_name: str) -> str:
    normalized = str(method_name or "").strip()
    return PHASE1_SELECTOR_ALIASES.get(normalized.lower(), normalized)


def _run_phase1_selector_on_prepared_data(
    method_name: str,
    X_scaled: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_features: Optional[int] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute core Phase 1 selectors on already prepared inputs."""
    canonical_method = _canonicalize_phase1_selector_name(method_name)
    resolved_n_features = _resolve_requested_feature_count(n_features, feature_names)
    cache = cache if cache is not None else {}

    if canonical_method == "run_random_forest_selector":
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        sorted_features = sorted(
            zip(feature_names, rf.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
        selected_features = sorted_features[:resolved_n_features]
        return {
            "success": True,
            "method": "random_forest",
            "n_features_requested": resolved_n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features],
        }

    if canonical_method == "run_lasso_selector":
        sorted_features, metadata = _rank_sparse_logistic_features(
            X_scaled,
            y,
            feature_names,
            penalty="l1",
            random_state=42,
        )
        selected_features = sorted_features[:resolved_n_features]
        return {
            "success": True,
            "method": "l1_logistic_lasso",
            "n_features_requested": resolved_n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_coefficients": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features],
            **metadata,
        }

    if canonical_method == "run_elasticnet_selector":
        sorted_features, metadata = _rank_sparse_logistic_features(
            X_scaled,
            y,
            feature_names,
            penalty="elasticnet",
            random_state=42,
        )
        selected_features = sorted_features[:resolved_n_features]
        return {
            "success": True,
            "method": "elastic_net_logistic",
            "n_features_requested": resolved_n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_coefficients": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features],
            **metadata,
        }

    if canonical_method == "run_lightgbm_selector":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            return {"success": False, "error": f"请安装 'lightgbm' 库: {exc}"}

        clean_feature_names: List[str] = []
        feature_name_map: Dict[str, str] = {}
        used_clean_names: set[str] = set()
        for original_name in feature_names:
            clean_name = re.sub(r"[^\w\-]", "_", str(original_name))
            if clean_name in used_clean_names:
                clean_name = f"{clean_name}_{len(clean_feature_names)}"
            clean_feature_names.append(clean_name)
            feature_name_map[clean_name] = original_name
            used_clean_names.add(clean_name)

        X_lgb = X_scaled.copy()
        X_lgb.columns = clean_feature_names
        y_encoded = LabelEncoder().fit_transform(y)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_lgb, y_encoded)

        sorted_features = sorted(
            (
                (feature_name_map[clean_name], float(importance))
                for clean_name, importance in zip(clean_feature_names, lgb_model.feature_importances_)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        selected_features = sorted_features[:resolved_n_features]
        return {
            "success": True,
            "method": "lightgbm",
            "n_features_requested": resolved_n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features],
        }

    if canonical_method == "run_mrmr_selector":
        selected_features = _run_mrmr_classif_silently(
            X=X_scaled,
            y=y,
            K=resolved_n_features,
            n_jobs=1,
        )
        return {
            "success": True,
            "method": "mrmr",
            "n_features_requested": resolved_n_features,
            "n_features_selected": len(selected_features),
            "selected_features": selected_features,
            "note": f"Selected the top {len(selected_features)} features via mRMR.",
        }

    if canonical_method in {"run_t_test_selector", "run_fdr_effect_size_selector"}:
        if "welch_statistics" not in cache:
            cache["welch_statistics"] = _rank_welch_statistics(X_scaled, y, feature_names)
        t_stats, p_values, effect_sizes, classes = cache["welch_statistics"]

        if canonical_method == "run_t_test_selector":
            abs_t_stats = np.abs(t_stats)
            sorted_features = sorted(
                zip(feature_names, abs_t_stats),
                key=lambda item: item[1],
                reverse=True,
            )
            selected_features = sorted_features[:resolved_n_features]
            feature_pvalues = dict(zip(feature_names, p_values))
            return {
                "success": True,
                "method": "welch_t_test",
                "group_labels": [str(classes[0]), str(classes[1])],
                "n_features_requested": resolved_n_features,
                "n_features_selected": len(selected_features),
                "selected_features": [item[0] for item in selected_features],
                "t_scores": {item[0]: float(item[1]) for item in selected_features},
                "p_values": {item[0]: float(feature_pvalues[item[0]]) for item in selected_features},
                "all_features_ranked": [item[0] for item in sorted_features],
            }

        q_values = _benjamini_hochberg(p_values)
        sort_order = np.lexsort((-np.abs(effect_sizes), q_values))
        sorted_features = [(feature_names[idx], float(np.abs(effect_sizes[idx]))) for idx in sort_order]
        selected_features = sorted_features[:resolved_n_features]
        return {
            "success": True,
            "method": "fdr_effect_size",
            "group_labels": [str(classes[0]), str(classes[1])],
            "n_features_requested": resolved_n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "effect_sizes": {feature_names[idx]: float(effect_sizes[idx]) for idx in sort_order[:resolved_n_features]},
            "abs_effect_sizes": {item[0]: float(item[1]) for item in selected_features},
            "q_values": {feature_names[idx]: float(q_values[idx]) for idx in sort_order[:resolved_n_features]},
            "p_values": {feature_names[idx]: float(p_values[idx]) for idx in sort_order[:resolved_n_features]},
            "all_features_ranked": [item[0] for item in sorted_features],
        }

    return {"success": False, "error": f"不支持的预处理 selector: {canonical_method}"}


def _rank_sparse_logistic_features(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    *,
    penalty: str,
    random_state: int = 42,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """Rank features with sparse logistic models and aggregate absolute coefficients."""
    from sklearn.linear_model import LogisticRegressionCV

    y_encoded = LabelEncoder().fit_transform(y)
    unique_classes = np.unique(y_encoded)
    scoring = "roc_auc" if len(unique_classes) == 2 else "accuracy"
    fit_kwargs: Dict[str, Any] = {
        "cv": 5,
        "penalty": penalty,
        "solver": "saga",
        "random_state": random_state,
        "max_iter": 5000,
        "scoring": scoring,
        "n_jobs": -1,
    }
    if penalty == "elasticnet":
        fit_kwargs["l1_ratios"] = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]

    model = LogisticRegressionCV(**fit_kwargs)
    model.fit(X_scaled, y_encoded)

    coef_matrix = np.asarray(model.coef_, dtype=float)
    if coef_matrix.ndim == 1:
        coef_matrix = coef_matrix.reshape(1, -1)
    abs_coefficients = np.max(np.abs(coef_matrix), axis=0)
    sorted_features = sorted(
        zip(feature_names, abs_coefficients),
        key=lambda item: item[1],
        reverse=True,
    )

    metadata: Dict[str, Any] = {
        "scoring": scoring,
        "best_c": float(np.ravel(model.C_)[0]),
    }
    if penalty == "elasticnet" and getattr(model, "l1_ratio_", None) is not None:
        metadata["best_l1_ratio"] = float(np.ravel(model.l1_ratio_)[0])
    return sorted_features, metadata


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    if n == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Compute per-feature Cohen's d for two groups."""
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]
    if n_a < 2 or n_b < 2:
        return np.zeros(group_a.shape[1], dtype=float)

    mean_diff = np.nanmean(group_a, axis=0) - np.nanmean(group_b, axis=0)
    var_a = np.nanvar(group_a, axis=0, ddof=1)
    var_b = np.nanvar(group_b, axis=0, ddof=1)
    pooled_var = (((n_a - 1) * var_a) + ((n_b - 1) * var_b)) / max(1, n_a + n_b - 2)
    pooled_std = np.sqrt(np.clip(pooled_var, a_min=0.0, a_max=None))
    effect_size = np.divide(
        mean_diff,
        pooled_std,
        out=np.zeros_like(mean_diff, dtype=float),
        where=pooled_std > 0,
    )
    return np.nan_to_num(effect_size, nan=0.0, posinf=0.0, neginf=0.0)


def _rank_welch_statistics(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Compute Welch t statistics, raw p-values, effect sizes and class labels."""
    classes = list(pd.Series(y).dropna().unique())
    if len(classes) != 2:
        raise ValueError(f"Welch t-test requires exactly 2 groups, but got {len(classes)}")

    y_array = pd.Series(y).to_numpy()
    mask_a = y_array == classes[0]
    mask_b = y_array == classes[1]
    group_a = X_scaled.iloc[mask_a]
    group_b = X_scaled.iloc[mask_b]

    t_stats, p_values = stats.ttest_ind(
        group_a.values,
        group_b.values,
        axis=0,
        equal_var=False,
        nan_policy='omit'
    )
    t_stats = np.nan_to_num(t_stats, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    effect_sizes = _cohens_d(group_a.values, group_b.values)
    return t_stats, p_values, effect_sizes, classes


def check_balance(data_path: str, group_col: str) -> str:
    """
    检查数据是否平衡... (函数逻辑基本正确，保持不变)
    """
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"error": "不支持的文件格式"})
        
        if group_col not in df.columns:
            return json.dumps({"error": f"分组列 '{group_col}' 不存在"})
        
        class_counts = df[group_col].value_counts()
        
        if len(class_counts) < 2:
            return json.dumps({"error": "分组列中只有一个类别，无法进行比较"})

        majority_class_size = class_counts.max()
        minority_class_size = class_counts.min()
        balance_ratio = majority_class_size / minority_class_size
        
        is_balanced = balance_ratio <= 2.0
        
        recommendation = "数据平衡，可以直接进行特征选择" if is_balanced else f"数据不平衡（比例 {balance_ratio:.2f}），建议进行稳定性选择（重采样）"
        
        result = {
            "is_balanced": is_balanced,
            "majority_class_size": int(majority_class_size),
            "minority_class_size": int(minority_class_size),
            "balance_ratio": float(balance_ratio),
            "class_distribution": class_counts.to_dict(),
            "recommendation": recommendation
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"检查数据平衡性时出错: {str(e)}"})


def run_random_forest_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Random Forest for feature selection based on feature importance scores.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and scores, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and importance scores
    """
    try:
        # [优化] 调用公共函数
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        feature_importance = dict(zip(feature_names, rf.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "random_forest",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"随机森林特征选择时出错: {str(e)}")


def run_lasso_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use L1-regularized logistic regression for feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and coefficients, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and coefficient values
    """
    try:
        # [优化] 调用公共函数
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)

        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))

        sorted_features, metadata = _rank_sparse_logistic_features(
            X_scaled,
            y,
            feature_names,
            penalty="l1",
            random_state=42,
        )
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "l1_logistic_lasso",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_coefficients": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        result.update(metadata)
        
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"L1 logistic/Lasso特征选择时出错: {str(e)}")


def run_lightgbm_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use LightGBM for feature selection based on feature importance scores.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and scores, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and importance scores
    """
    try:
        import lightgbm as lgb
        import re
        
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        # Clean feature names to avoid LightGBM special character issues
        # LightGBM doesn't support special JSON characters in feature names
        clean_feature_names = []
        feature_name_map = {}  # Map clean names back to original names
        
        for original_name in feature_names:
            # Replace special characters with underscore
            clean_name = re.sub(r'[^\w\-]', '_', str(original_name))
            # Ensure unique names
            if clean_name in feature_name_map.values():
                clean_name = f"{clean_name}_{len(clean_feature_names)}"
            clean_feature_names.append(clean_name)
            feature_name_map[clean_name] = original_name
        
        # Update DataFrame with clean column names
        X_scaled.columns = clean_feature_names
        
        # Encode labels for LightGBM
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Train LightGBM model
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_scaled, y_encoded)
        
        # Map clean names back to original names
        feature_importance = {}
        for clean_name, importance in zip(clean_feature_names, lgb_model.feature_importances_):
            original_name = feature_name_map[clean_name]
            feature_importance[original_name] = importance
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "lightgbm",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'lightgbm' 库: pip install lightgbm")
    except Exception as e:
        return _selector_error(f"LightGBM特征选择时出错: {str(e)}")


def run_xgboost_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use XGBoost for feature selection based on feature importance scores.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and scores, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and importance scores
    """
    try:
        import xgboost as xgb
        
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        # Encode labels for XGBoost
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Train XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_scaled, y_encoded)
        
        feature_importance = dict(zip(feature_names, xgb_model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "xgboost",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'xgboost' 库: pip install xgboost")
    except Exception as e:
        return _selector_error(f"XGBoost特征选择时出错: {str(e)}")


def run_elasticnet_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Elastic Net logistic regression for feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and coefficients, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and coefficient values
    """
    try:
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        sorted_features, metadata = _rank_sparse_logistic_features(
            X_scaled,
            y,
            feature_names,
            penalty="elasticnet",
            random_state=42,
        )
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "elastic_net_logistic",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "feature_coefficients": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features],
        }
        result.update(metadata)
        
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"Elastic Net logistic特征选择时出错: {str(e)}")


def run_mutual_info_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Mutual Information with SelectKBest for feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and MI scores, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and MI scores
    """
    try:
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        
        feature_scores = dict(zip(feature_names, mi_scores))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "mutual_information",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "mi_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"互信息特征选择时出错: {str(e)}")


def run_t_test_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Welch's t-test for binary-group feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and t-statistics/p-values, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select.
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and t-test statistics
    """
    try:
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)

        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))

        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))

        t_stats, p_values, _, classes = _rank_welch_statistics(X_scaled, y, feature_names)
        abs_t_stats = np.abs(t_stats)

        feature_scores = dict(zip(feature_names, abs_t_stats))
        feature_pvalues = dict(zip(feature_names, p_values))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "welch_t_test",
            "group_labels": [str(classes[0]), str(classes[1])],
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "t_scores": {item[0]: float(item[1]) for item in selected_features},
            "p_values": {item[0]: float(feature_pvalues[item[0]]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }

        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"Welch t-test特征选择时出错: {str(e)}")


def run_fdr_effect_size_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Welch's t-test with BH-FDR correction and effect-size ranking.

    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and summary statistics, no raw data)
    """
    try:
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)

        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        n_features = min(n_features, len(feature_names))

        _, p_values, effect_sizes, classes = _rank_welch_statistics(X_scaled, y, feature_names)
        q_values = _benjamini_hochberg(p_values)
        sort_order = np.lexsort((-np.abs(effect_sizes), q_values))
        sorted_features = [(feature_names[idx], float(np.abs(effect_sizes[idx]))) for idx in sort_order]

        selected_features = sorted_features[:n_features]
        result = {
            "method": "fdr_effect_size",
            "group_labels": [str(classes[0]), str(classes[1])],
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "effect_sizes": {feature_names[idx]: float(effect_sizes[idx]) for idx in sort_order[:n_features]},
            "abs_effect_sizes": {item[0]: float(item[1]) for item in selected_features},
            "q_values": {feature_names[idx]: float(q_values[idx]) for idx in sort_order[:n_features]},
            "p_values": {feature_names[idx]: float(p_values[idx]) for idx in sort_order[:n_features]},
            "all_features_ranked": [item[0] for item in sorted_features],
        }
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"FDR/effect-size特征选择时出错: {str(e)}")


def run_f_statistic_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use F-statistic (ANOVA F-value) with SelectKBest for feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and F-scores, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and F-scores
    """
    try:
        from sklearn.feature_selection import SelectKBest, f_classif
        
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        # Calculate F-statistic scores
        f_scores, p_values = f_classif(X_scaled, y)
        
        feature_scores = dict(zip(feature_names, f_scores))
        feature_pvalues = dict(zip(feature_names, p_values))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "f_statistic",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "f_scores": {item[0]: float(item[1]) for item in selected_features},
            "p_values": {item[0]: float(feature_pvalues[item[0]]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"F统计量特征选择时出错: {str(e)}")


def run_anova_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use ANOVA F-test for feature selection.
    
    This is an explicit alias around the Phase 1 ANOVA implementation so the
    method can be referenced directly as `anova` in voting pools and prompts.
    """
    try:
        result = json.loads(run_f_statistic_selector(data_path, group_col, n_features))
        if not result.get("success"):
            return json.dumps(result, ensure_ascii=False, indent=2)

        result["method"] = "anova"
        result["alias_of"] = "f_statistic"
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return _selector_error(f"ANOVA特征选择时出错: {str(e)}")


def run_fcbf_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Fast Correlation-based Filter (FCBF) for feature selection.
    
    Note: Due to compatibility issues with scikit-feature library, this implementation
    uses a simplified correlation-based approach as a fallback.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (selected feature names, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and count
    """
    try:
        from scipy.stats import spearmanr
        
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        # Calculate correlation with target for each feature
        correlations = {}
        for col in X_scaled.columns:
            corr, _ = spearmanr(X_scaled[col], y_encoded)
            correlations[col] = abs(corr)  # Use absolute correlation
        
        # Sort by correlation strength
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        
        result = {
            "method": "fcbf",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "correlation_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features],
            "note": "Using correlation-based approach (simplified FCBF)"
        }
        
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"FCBF特征选择时出错: {str(e)}")


def run_cfs_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Correlation-based Feature Selection (CFS) for feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (selected feature names, no raw data)
    
    Note: CFS selects features highly correlated with target but minimally correlated with each other.
    Requires: skfeature-chappers library
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and count
    """
    try:
        from skfeature.function.statistical_based import CFS
        
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        # CFS returns selected feature indices
        selected_indices = CFS.cfs(X_scaled.values, y_encoded)
        
        # Get all selected features
        all_selected_features = [feature_names[idx] for idx in selected_indices]
        
        # Limit to top N features
        selected_features = all_selected_features[:n_features]
        
        result = {
            "method": "cfs",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": selected_features,
            "all_features_ranked": all_selected_features,
            "note": "CFS selects features with high correlation to target and low inter-correlation"
        }
        
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'skfeature-chappers' 库: pip install skfeature-chappers")
    except Exception as e:
        return _selector_error(f"CFS特征选择时出错: {str(e)}")


def run_relief_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use ReliefF algorithm for feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and importance scores, no raw data)
    
    Requires: skrebate library
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, ranked features, and importance scores
    """
    try:
        from skrebate import ReliefF
        
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        y_encoded = LabelEncoder().fit_transform(y)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        relieff = ReliefF(n_features_to_select=n_features, n_neighbors=100)
        relieff.fit(X_scaled.values, y_encoded)
        
        # Get feature importance scores
        scores = dict(zip(feature_names, relieff.feature_importances_))
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        result = {
            "method": "relief",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "importance_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'skrebate' 库: pip install skrebate")
    except Exception as e:
        return _selector_error(f"ReliefF特征选择时出错: {str(e)}")


def run_cfs_selector_old(data_path: str, group_col: str) -> str:
    """
    Use Correlation-based Feature Selection (CFS) to select features highly correlated 
    with the target but minimally correlated with each other.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (selected feature names, no raw data)
    
    Note: CFS returns a feature subset, not a complete ranking.
    Requires: skfeature-chappers library
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        
    Returns:
        JSON string with method name, selected features, and count
    """
    try:
        from skfeature.function.statistical_based.CFS import cfs

        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # CFS需要数值型的y
        y_encoded = LabelEncoder().fit_transform(y)
        
        # cfs返回选中特征的索引
        selected_indices = cfs(X_scaled.values, y_encoded)
        selected_features = [feature_names[i] for i in selected_indices]
        
        result = {
            "method": "cfs",
            "selected_features": selected_features,
            "count": len(selected_features),
            "note": "CFS selects a subset of features, no ranking is provided."
        }
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'skfeature-chappers' 库: pip install skfeature-chappers")
    except Exception as e:
        return _selector_error(f"CFS特征选择时出错: {str(e)}")


def run_linear_svm_selector(data_path: str, group_col: str) -> str:
    """
    Use Linear SVM with L1 penalty (Lasso) for feature selection. L1 regularization 
    drives unimportant feature coefficients to zero.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and coefficients, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        
    Returns:
        JSON string with method name, ranked features with non-zero coefficients
    """
    try:
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # 使用L1惩罚。dual=False对于n_samples > n_features的情况更优。
        lsvc = LinearSVC(penalty='l1', dual=False, random_state=42, max_iter=2000)
        lsvc.fit(X_scaled, y)
        
        # lsvc.coef_ 是一个二维数组，对于二分类问题，取第一行即可
        coefficients = lsvc.coef_[0]
        scores = dict(zip(feature_names, np.abs(coefficients)))
        
        # 筛选出系数不为零的特征并排序
        non_zero_features = {k: v for k, v in scores.items() if v > 1e-5}
        sorted_features = sorted(non_zero_features.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            "method": "linear_svc_l1",
            "features": [item[0] for item in sorted_features],
            "coefficients": [item[1] for item in sorted_features]
        }
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"LinearSVC特征选择时出错: {str(e)}")


def run_sbs_selector(data_path: str, group_col: str, n_features_to_select: int = 20) -> str:
    """
    Use Sequential Backward Selection (SBS) for wrapper-based feature selection. 
    Iteratively removes the least important features starting from all features.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (selected feature names, no raw data)
    
    Requires: mlxtend library
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features_to_select: Number of features to select (default: 20)
        
    Returns:
        JSON string with method name, selected features, count, and final score
    """
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS

        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # SBS需要一个评估器
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        sbs = SFS(estimator,
                  k_features=n_features_to_select,
                  forward=False,  # False表示后向选择
                  floating=False,
                  verbose=0,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)
        
        sbs.fit(X_scaled, y)
        
        # 获取选中的特征名
        selected_feature_indices = sbs.k_feature_idx_
        selected_features = [feature_names[i] for i in selected_feature_indices]
        
        result = {
            "method": "sbs",
            "selected_features": selected_features,
            "count": len(selected_features),
            "final_score": sbs.k_score_,
            "note": f"Selected the best {len(selected_features)} features via SBS."
        }
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'mlxtend' 库: pip install mlxtend")
    except Exception as e:
        return _selector_error(f"SBS特征选择时出错: {str(e)}")



def run_mutual_information_selector(data_path: str, group_col: str) -> str:
    """
    Use Mutual Information for feature selection to capture non-linear dependencies.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and MI scores, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        
    Returns:
        JSON string with method name, ranked features, and mutual information scores
    """
    try:
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # 计算互信息分数
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        
        scores = dict(zip(feature_names, mi_scores))
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            "method": "mutual_information",
            "features": [item[0] for item in sorted_features],
            "mi_scores": [float(item[1]) for item in sorted_features]
        }
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"互信息特征选择时出错: {str(e)}")


def run_mrmr_selector(data_path: str, group_col: str, n_features: int = None) -> str:
    """
    Use Minimum Redundancy Maximum Relevance (mRMR) algorithm for feature selection.
    Selects features with maximum relevance to target and minimum redundancy among themselves.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (selected feature names, no raw data)
    
    Requires: mrmr-selection library
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of features to select. 
                   If None, selects 30% of total features with minimum of 10 (default: None)
        
    Returns:
        JSON string with method name, selected features, and count
    """
    try:
        import mrmr

        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        # Dynamic default: 30% of total features, minimum 10
        if n_features is None:
            n_features = max(10, int(len(feature_names) * 0.30))
        
        # Boundary check: don't exceed total features
        n_features = min(n_features, len(feature_names))
        
        # mRMR库要求X是DataFrame, y是Series，我们的辅助函数正好满足
        # 它直接返回选定特征的名称列表
        selected_features = _run_mrmr_classif_silently(
            X=X_scaled,
            y=y,
            K=n_features,
            n_jobs=1,
        )
        
        result = {
            "method": "mrmr",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": selected_features,
            "note": f"Selected the top {len(selected_features)} features via mRMR."
        }
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'mrmr-selection' 库: pip install mrmr-selection")
    except Exception as e:
        return _selector_error(f"mRMR特征选择时出错: {str(e)}")

def run_ga_selector(data_path: str, group_col: str) -> str:
    """
    Use Genetic Algorithm (GA) for feature selection. Simulates natural selection 
    to evolve optimal feature subsets over multiple generations.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (selected feature names, no raw data)
    
    Requires: sklearn-genetic-opt library
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        
    Returns:
        JSON string with method name, selected features, and count
    """
    try:
        from sklearn_genetic import GeneticSelectorCV

        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        y_encoded = LabelEncoder().fit_transform(y)
        
        # GA需要一个评估器
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # 参数可以根据需要调整，这里使用了一些合理的默认值
        selector = GeneticSelectorCV(
            estimator,
            cv=3,
            scoring="accuracy",
            population_size=50,
            generations=20,
            n_jobs=-1,
            verbose=False
        )
        
        selector.fit(X_scaled.values, y_encoded)
        
        # 获取选中的特征
        selected_features = list(X_scaled.columns[selector.support_])
        
        result = {
            "method": "genetic_algorithm",
            "selected_features": selected_features,
            "count": len(selected_features),
            "note": "Selected features are the best found by the genetic algorithm."
        }
        return _selector_success(result)
    except ImportError:
        return _selector_error("请安装 'sklearn-genetic-opt' 库: pip install sklearn-genetic-opt")
    except Exception as e:
        return _selector_error(f"GA特征选择时出错: {str(e)}")

def run_statistical_selector(data_path: str, group_col: str, n_features: int = 50) -> str:
    """
    Use statistical methods (F-test + Mutual Information) for feature selection.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally
    - Returns only metadata (feature names and combined scores, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        n_features: Number of top features to select (default: 50)
        
    Returns:
        JSON string with method name, ranked features, and combined statistical scores
    """
    try:
        # [优化] 调用公共函数
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        f_scores, _ = f_classif(X_scaled, y)
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        combined_scores = (f_scores + mi_scores) / 2
        
        feature_scores = dict(zip(feature_names, combined_scores))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = sorted_features[:n_features]
        
        result = {
            "method": "statistical",
            "n_features_requested": n_features,
            "n_features_selected": len(selected_features),
            "selected_features": [item[0] for item in selected_features],
            "combined_scores": {item[0]: float(item[1]) for item in selected_features},
            "all_features_ranked": [item[0] for item in sorted_features]
        }
        return _selector_success(result)
    except Exception as e:
        return _selector_error(f"统计特征选择时出错: {str(e)}")


def perform_stability_selection(data_path: str, group_col: str, iterations: int, methods: List[str], top_percentage: float, temp_dir: str, mandatory_features: List[str] = None) -> str:
    """
    Perform stability selection through iterative resampling and feature selection.
    
    Creates balanced subsets via down-sampling, runs feature selection methods across 
    multiple iterations, and logs the top-N% features from each iteration to separate 
    log files for each method.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally through resampling
    - Saves results to local log files
    - Returns only metadata (success status, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file
        group_col: Name of the target/group column
        iterations: Number of resampling iterations to perform
        methods: List of feature selection method names to use
        top_percentage: Percentage of top features to select each iteration (e.g., 0.3 for top 30%)
        temp_dir: Directory path to store selection log files
        mandatory_features: List of feature names that must be kept (protected from filtering)
        
    Returns:
        JSON string with success status, iterations completed, methods used, and temp directory
    """
    try:
        # ═══════════════════════════════════════════════════════════════════════
        # 🧹 [IO Defense] 破窗重置：强制清理旧缓存，杜绝幽灵数据污染
        # ═══════════════════════════════════════════════════════════════════════
        # 
        # 问题背景：
        # - 2026-04-02 的旧日志文件包含原始格式特征名（CAR 14:0）
        # - 2026-04-30 的新测试使用了HMDB IDs（HMDB0005066）
        # - 但因为旧日志文件存在，导致特征名称"时空错乱"
        # 
        # 解决方案：
        # - 每次执行前，无条件删除整个 temp_dir 目录
        # - 确保工作空间绝对是"无菌"的
        # - 即使中途报错退出，下次启动也会重新清理
        # 
        # 设计哲学：
        # - "破窗理论"：一个破窗不修，整栋楼都会烂掉
        # - 稳定性选择的日志文件必须与当前数据文件完全同步
        # - 宁可重新计算，也不能使用过期数据
        # ═══════════════════════════════════════════════════════════════════════
        
        import shutil
        
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 [IO Defense] Purged stale cache directory: {temp_dir}")
                print(f"   Reason: Prevent ghost data contamination from previous runs")
            except Exception as cleanup_error:
                print(f"⚠️  [IO Defense] Failed to clean {temp_dir}: {cleanup_error}")
                print(f"   Attempting to continue with existing directory...")
        
        os.makedirs(temp_dir, exist_ok=True)
        print(f"✅ [IO Defense] Created fresh cache directory: {temp_dir}")
        
        # Initialize mandatory features protection
        if mandatory_features is None:
            mandatory_features = []
        
        # Convert to set for faster lookup
        mandatory_features_set = set(mandatory_features)
        
        if mandatory_features:
            print(f"[Mandatory Features] Protecting {len(mandatory_features)} prior biomarkers")
            print(f"  Examples: {', '.join(list(mandatory_features)[:3])}...")
        
        df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)

        # [优化] 为每个方法打开一个文件句柄，使用追加模式
        log_files = {}
        for method in methods:
            log_files[method] = open(os.path.join(temp_dir, f"{method}_selections.log"), 'w')

        for i in range(iterations):
            # --- [核心逻辑修正] ---
            # 1. 创建平衡的重采样数据集
            case_df = df[df[group_col] == df[group_col].unique()[0]]
            control_df = df[df[group_col] == df[group_col].unique()[1]]
            
            if len(case_df) > len(control_df):
                majority_df, minority_df = case_df, control_df
            else:
                majority_df, minority_df = control_df, case_df

            majority_downsampled = resample(majority_df, replace=True, n_samples=len(minority_df), random_state=i)
            balanced_df = pd.concat([minority_df, majority_downsampled])
            balanced_df = balanced_df.reset_index(drop=True)
            # --- 修正结束 ---

            # 2. 在平衡数据集上进行预处理
            # 排除非特征列（Sample_ID, Group等）
            non_feature_cols = [group_col]
            
            # 排除所有样本ID/技术列的变体
            if 'Sample_ID' in balanced_df.columns:
                non_feature_cols.append('Sample_ID')
            if 'sample_id' in balanced_df.columns:
                non_feature_cols.append('sample_id')
            if 'SampleID' in balanced_df.columns:
                non_feature_cols.append('SampleID')
            if 'sampleid' in balanced_df.columns:
                non_feature_cols.append('sampleid')
            for technical_col in ('__row_id__', 'ROW_ID', 'row_id'):
                if technical_col in balanced_df.columns and technical_col not in non_feature_cols:
                    non_feature_cols.append(technical_col)
            
            # 排除所有Group列的变体（包括merge产生的Group_x, Group_y等）
            for col in balanced_df.columns:
                if col.startswith('Group') or col.startswith('group'):
                    if col not in non_feature_cols:
                        non_feature_cols.append(col)
            
            X_balanced = balanced_df.drop(columns=non_feature_cols)
            y_balanced = balanced_df[group_col]
            feature_names = X_balanced.columns.tolist()
            
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_balanced), columns=feature_names)
            
            num_to_select = max(1, int(len(feature_names) * top_percentage))

            # 3. 运行选择器并记录结果
            for method in methods:
                top_features = []
                try:
                    # 处理工具名称映射
                    method_name = method
                    if method == "run_random_forest_selector":
                        method_name = "random_forest"
                    elif method == "run_lasso_selector":
                        method_name = "lasso"
                    elif method == "run_statistical_selector":
                        method_name = "statistical"
                    elif method == "run_mutual_information_selector":
                        method_name = "mutual_information"
                    elif method == "run_linear_svm_selector":
                        method_name = "linear_svc"
                    elif method == "run_relief_selector":
                        method_name = "relief_f"
                    elif method == "run_cfs_selector":
                        method_name = "cfs"
                    elif method == "run_fcbf_selector":
                        method_name = "fcbf"
                    elif method == "run_mrmr_selector":
                        method_name = "mrmr"
                    # === NEW MAPPINGS: Add missing method mappings ===
                    elif method == "run_lightgbm_selector":
                        method_name = "lightgbm"
                    elif method == "run_xgboost_selector":
                        method_name = "xgboost"
                    elif method == "run_elasticnet_selector":
                        method_name = "elasticnet"
                    elif method == "run_fdr_effect_size_selector":
                        method_name = "fdr_effect_size"
                    elif method == "run_mutual_info_selector":
                        method_name = "mutual_info"
                    elif method == "run_f_statistic_selector":
                        method_name = "f_statistic"
                    elif method == "run_t_test_selector":
                        method_name = "t_test"
                    elif method == "run_anova_selector":
                        method_name = "anova"
                    # === END NEW MAPPINGS ===
                    
                    if method_name == "random_forest":
                        rf = RandomForestClassifier(n_estimators=100, random_state=i).fit(X_scaled, y_balanced)
                        scores = dict(zip(feature_names, rf.feature_importances_))
                        top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                    
                    elif method_name == "lasso":
                        sorted_features, _ = _rank_sparse_logistic_features(
                            X_scaled,
                            y_balanced,
                            feature_names,
                            penalty="l1",
                            random_state=i,
                        )
                        top_features = [f for f, _ in sorted_features[:num_to_select]]

                    elif method_name == "statistical":
                        f_scores, _ = f_classif(X_scaled, y_balanced)
                        mi_scores = mutual_info_classif(X_scaled, y_balanced, random_state=i)
                        scores = dict(zip(feature_names, (f_scores + mi_scores) / 2))
                        top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                    
                    elif method_name == "mutual_information":
                        mi_scores = mutual_info_classif(X_scaled, y_balanced, random_state=i)
                        scores = dict(zip(feature_names, mi_scores))
                        top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                    
                    elif method_name == "linear_svc":
                        lsvc = LinearSVC(penalty='l1', dual=False, random_state=i, max_iter=2000)
                        lsvc.fit(X_scaled, y_balanced)
                        coefficients = lsvc.coef_[0]
                        scores = dict(zip(feature_names, np.abs(coefficients)))
                        # 筛选出系数不为零的特征并排序
                        non_zero_features = {k: v for k, v in scores.items() if v > 1e-5}
                        top_features = [f for f, s in sorted(non_zero_features.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                    
                    elif method_name == "relief_f":
                        try:
                            from skrebate import ReliefF
                            y_enc = LabelEncoder().fit_transform(y_balanced)
                            relieff = ReliefF()
                            relieff.fit(X_scaled.values, y_enc)
                            scores = dict(zip(feature_names, relieff.feature_importances_))
                            top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                        except ImportError:
                            print(f"警告: ReliefF需要安装skrebate库，跳过方法 {method}")
                            continue
                    
                    elif method_name == "cfs":
                        try:
                            from skfeature.function.statistical_based.CFS import cfs
                            y_enc = LabelEncoder().fit_transform(y_balanced)
                            selected_indices = cfs(X_scaled.values, y_enc)
                            selected_features = [feature_names[idx] for idx in selected_indices]
                            top_features = selected_features[:num_to_select]
                        except ImportError:
                            print(f"警告: CFS需要安装skfeature-chappers库，跳过方法 {method}")
                            continue
                    
                    elif method_name == "fcbf":
                        try:
                            from skfeature.function.information_theoretical_based.FCBF import fcbf
                            y_enc = LabelEncoder().fit_transform(y_balanced)
                            selected_indices, _ = fcbf(X_scaled.values, y_enc, n_selected_features=None)
                            selected_features = [feature_names[idx] for idx in selected_indices]
                            top_features = selected_features[:num_to_select]
                        except ImportError:
                            print(f"警告: FCBF需要安装skfeature-chappers库，跳过方法 {method}")
                            continue
                    
                    elif method_name == "mrmr":
                        try:
                            import mrmr
                            selected_features = _run_mrmr_classif_silently(
                                X=X_scaled,
                                y=y_balanced,
                                K=num_to_select,
                                n_jobs=1,
                            )
                            top_features = selected_features
                        except ImportError:
                            print(f"警告: mRMR需要安装mrmr-selection库，跳过方法 {method}")
                            continue
                    
                    # === NEW METHOD IMPLEMENTATIONS ===
                    elif method_name == "lightgbm":
                        try:
                            import lightgbm as lgb
                            import re
                            
                            # Clean feature names to avoid LightGBM special character issues
                            clean_feature_names = []
                            feature_name_map = {}
                            
                            for original_name in feature_names:
                                clean_name = re.sub(r'[^\w\-]', '_', str(original_name))
                                # Ensure unique names
                                if clean_name in feature_name_map.values():
                                    clean_name = f"{clean_name}_{len(clean_feature_names)}"
                                clean_feature_names.append(clean_name)
                                feature_name_map[clean_name] = original_name
                            
                            # Create DataFrame with clean names
                            X_clean = X_scaled.copy()
                            X_clean.columns = clean_feature_names
                            
                            y_enc = LabelEncoder().fit_transform(y_balanced)
                            lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=i, verbose=-1)
                            lgb_model.fit(X_clean, y_enc)
                            
                            # Map back to original names
                            scores = {}
                            for clean_name, importance in zip(clean_feature_names, lgb_model.feature_importances_):
                                original_name = feature_name_map[clean_name]
                                scores[original_name] = importance
                            
                            top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                        except ImportError:
                            print(f"警告: LightGBM需要安装lightgbm库，跳过方法 {method}")
                            continue
                        except Exception as lgb_error:
                            import traceback
                            error_msg = f"[Iteration {i+1}] LightGBM failed: {str(lgb_error)}\n"
                            error_msg += f"  Data shape: {X_scaled.shape}\n"
                            error_msg += f"  Feature names sample: {feature_names[:5]}\n"
                            error_msg += f"  Has NaN: {X_scaled.isnull().any().any()}\n"
                            error_msg += f"  Has Inf: {np.isinf(X_scaled.values).any()}\n"
                            error_msg += f"  Traceback:\n{traceback.format_exc()}\n"
                            print(error_msg)
                            # Write to error log
                            error_log_path = os.path.join(temp_dir, "lightgbm_errors.log")
                            with open(error_log_path, 'a') as error_log:
                                error_log.write(error_msg)
                            continue
                    
                    elif method_name == "xgboost":
                        try:
                            import xgboost as xgb
                            y_enc = LabelEncoder().fit_transform(y_balanced)
                            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=i, verbosity=0, use_label_encoder=False, eval_metric='logloss')
                            xgb_model.fit(X_scaled, y_enc)
                            scores = dict(zip(feature_names, xgb_model.feature_importances_))
                            top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                        except ImportError:
                            print(f"警告: XGBoost需要安装xgboost库，跳过方法 {method}")
                            continue
                    
                    elif method_name == "elasticnet":
                        try:
                            sorted_features, _ = _rank_sparse_logistic_features(
                                X_scaled,
                                y_balanced,
                                feature_names,
                                penalty="elasticnet",
                                random_state=i,
                            )
                            top_features = [f for f, _ in sorted_features[:num_to_select]]
                        except Exception as e:
                            print(f"警告: ElasticNet在迭代 {i} 中失败: {e}")
                            continue
                    
                    elif method_name == "mutual_info":
                        try:
                            from sklearn.feature_selection import mutual_info_classif
                            mi_scores = mutual_info_classif(X_scaled, y_balanced, random_state=i)
                            scores = dict(zip(feature_names, mi_scores))
                            top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                        except Exception as e:
                            print(f"警告: Mutual Info在迭代 {i} 中失败: {e}")
                            continue
                    
                    elif method_name == "f_statistic":
                        try:
                            from sklearn.feature_selection import f_classif
                            f_scores, _ = f_classif(X_scaled, y_balanced)
                            scores = dict(zip(feature_names, f_scores))
                            top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                        except Exception as e:
                            print(f"警告: F-statistic在迭代 {i} 中失败: {e}")
                            continue

                    elif method_name == "t_test":
                        try:
                            t_stats, _, _, _ = _rank_welch_statistics(X_scaled, y_balanced, feature_names)
                            scores = dict(zip(feature_names, np.abs(t_stats)))
                            top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                        except Exception as e:
                            print(f"警告: t-test在迭代 {i} 中失败: {e}")
                            continue

                    elif method_name == "fdr_effect_size":
                        try:
                            _, p_values, effect_sizes, _ = _rank_welch_statistics(X_scaled, y_balanced, feature_names)
                            q_values = _benjamini_hochberg(p_values)
                            sort_order = np.lexsort((-np.abs(effect_sizes), q_values))
                            top_features = [feature_names[idx] for idx in sort_order[:num_to_select]]
                        except Exception as e:
                            print(f"警告: FDR/effect-size在迭代 {i} 中失败: {e}")
                            continue

                    elif method_name == "anova":
                        try:
                            from sklearn.feature_selection import f_classif
                            f_scores, _ = f_classif(X_scaled, y_balanced)
                            scores = dict(zip(feature_names, f_scores))
                            top_features = [f for f, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_to_select]]
                        except Exception as e:
                            print(f"警告: ANOVA在迭代 {i} 中失败: {e}")
                            continue
                    # === END NEW METHOD IMPLEMENTATIONS ===
                    
                    else:
                        print(f"警告: 未知的特征选择方法 {method}，跳过")
                        continue

                    # [优化] 追加到日志文件 - 修复字符串字面量问题
                    if top_features:
                        # 确保特征名称不包含特殊字符，避免字符串字面量问题
                        safe_features = [str(f).replace('\n', ' ').replace('\r', ' ') for f in top_features]
                        log_files[method].write('\n'.join(safe_features) + '\n')

                except Exception as e:
                    print(f"警告: 方法 {method} 在迭代 {i} 中失败: {e}")
                    continue
            
            if (i + 1) % 100 == 0:
                print(f"稳定性选择: 已完成 {i + 1}/{iterations} 次迭代")
        
        # [优化] 关闭所有文件
        for f in log_files.values():
            f.close()

        return json.dumps({
            "success": True, 
            "iterations_completed": iterations, 
            "methods_used": methods, 
            "temp_dir": temp_dir,
            "mandatory_features_protected": len(mandatory_features)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"稳定性选择时出错: {str(e)}"})


def calculate_frequencies_from_logs(temp_dir: str) -> str:
    """
    Calculate feature selection frequencies from stability selection log files.
    
    Privacy-Preserving Design:
    - Reads log files from local directory
    - Processes data locally
    - Returns only metadata (feature frequencies, no raw data)
    
    Args:
        temp_dir: Directory path containing selection log files from stability selection
        
    Returns:
        JSON string with feature frequencies for each method
    """
    try:
        log_files = glob.glob(os.path.join(temp_dir, "*.log"))
        if not log_files:
            return json.dumps({"error": "临时目录中没有找到 .log 文件"})
        
        method_frequencies = {}
        for file_path in log_files:
            method_name = os.path.basename(file_path).replace('_selections.log', '')
            with open(file_path, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            method_frequencies[method_name] = dict(Counter(features))
            
        return json.dumps(method_frequencies, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"从日志计算频率时出错: {str(e)}"})


def _load_raw_dataframe(data_path: str) -> pd.DataFrame:
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    if data_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(data_path)
    raise ValueError("不支持的文件格式，请使用 .csv 或 .xlsx")


def _resolve_feature_columns(df: pd.DataFrame, group_col: str) -> List[str]:
    non_feature_cols = [group_col]

    for sample_col in ("Sample_ID", "sample_id", "SampleID", "sampleid", "__row_id__", "ROW_ID", "row_id", "ID", "id"):
        if sample_col in df.columns and sample_col not in non_feature_cols:
            non_feature_cols.append(sample_col)

    for col in df.columns:
        if (col.startswith("Group") or col.startswith("group")) and col not in non_feature_cols:
            non_feature_cols.append(col)

    return [
        col for col in df.columns
        if col not in non_feature_cols and pd.api.types.is_numeric_dtype(df[col])
    ]


def _get_selector_callable(method_name: str):
    selector = globals().get(method_name)
    if callable(selector):
        return selector
    return None


def _stratified_resample_training_pool(
    df: pd.DataFrame,
    group_col: str,
    sample_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    sampled_parts: List[pd.DataFrame] = []
    for _, class_df in df.groupby(group_col, dropna=False):
        class_df = class_df.copy()
        if class_df.empty:
            continue
        if len(class_df) <= 2:
            sampled_parts.append(class_df)
            continue

        n_samples = int(round(len(class_df) * float(sample_fraction)))
        n_samples = max(2, n_samples)
        n_samples = min(len(class_df), n_samples)
        sampled_parts.append(
            class_df.sample(n=n_samples, replace=False, random_state=random_state)
        )

    if not sampled_parts:
        raise ValueError("分层重采样失败：没有可用的类别样本")

    return (
        pd.concat(sampled_parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def run_train_only_stability_selection(
    data_path: str,
    group_col: str,
    iterations: int,
    methods: List[str],
    n_features_per_method: int,
    temp_dir: str,
    sample_fraction: float = 0.8,
    mandatory_features: Optional[List[str]] = None,
    random_state: int = 42,
    data_already_scaled: bool = False,
) -> str:
    """
    Unified Phase 1 train-only stability selection.

    This routine operates strictly on the training pool passed from Step 5.0.
    Each iteration performs a stratified subsample of the training pool and runs
    a fixed selector family with a fixed per-method width. No class balancing,
    no adaptive threshold schedule, and no latest-artifact fallback are used.
    """
    try:
        import shutil

        if iterations <= 0:
            raise ValueError("iterations 必须大于 0")
        if not methods:
            raise ValueError("methods 不能为空")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        mandatory_features = [
            str(feature).strip()
            for feature in (mandatory_features or [])
            if str(feature).strip()
        ]

        df = _load_raw_dataframe(data_path)
        if group_col not in df.columns:
            raise ValueError(f"分组列 '{group_col}' 不存在于训练集")
        if int(df[group_col].nunique(dropna=True)) < 2:
            raise ValueError("训练集标签类别数不足，无法执行稳定性选择")

        feature_columns = _resolve_feature_columns(df, group_col)
        if not feature_columns:
            raise ValueError("训练集中没有可用于筛选的数值特征")

        n_features_per_method = max(1, min(int(n_features_per_method), len(feature_columns)))
        sample_fraction = float(sample_fraction)
        if sample_fraction <= 0 or sample_fraction > 1:
            raise ValueError("sample_fraction 必须位于 (0, 1] 区间")
        data_already_scaled = bool(data_already_scaled)

        iteration_trace = []
        resolved_methods = [_canonicalize_phase1_selector_name(method) for method in methods]
        method_log_paths = {
            method: os.path.join(temp_dir, f"{method}_selections.log")
            for method in resolved_methods
        }

        for iteration_idx in range(iterations):
            iteration_seed = int(random_state + iteration_idx)
            sampled_df = _stratified_resample_training_pool(
                df=df,
                group_col=group_col,
                sample_fraction=sample_fraction,
                random_state=iteration_seed,
            )
            sampled_path = os.path.join(temp_dir, f"iteration_{iteration_idx + 1:03d}.csv")
            sampled_df.to_csv(sampled_path, index=False)
            X_scaled, y_sampled, feature_names = _prepare_selection_inputs_from_df(
                sampled_df,
                group_col,
                apply_standard_scaling=not data_already_scaled,
            )
            selector_cache: Dict[str, Any] = {}

            iteration_record = {
                "iteration": int(iteration_idx + 1),
                "seed": iteration_seed,
                "n_rows": int(len(sampled_df)),
                "selectors": {},
            }

            for method in resolved_methods:
                selector = _get_selector_callable(method)
                if selector is None:
                    raise ValueError(f"未知的特征选择方法: {method}")

                if method in PHASE1_DEFAULT_SELECTOR_FAMILY:
                    result = _run_phase1_selector_on_prepared_data(
                        method_name=method,
                        X_scaled=X_scaled,
                        y=y_sampled,
                        feature_names=feature_names,
                        n_features=n_features_per_method,
                        cache=selector_cache,
                    )
                else:
                    result = json.loads(
                        selector(
                            data_path=sampled_path,
                            group_col=group_col,
                            n_features=n_features_per_method,
                        )
                    )
                if not result.get("success"):
                    raise ValueError(f"{method} 执行失败: {result.get('error')}")

                selected_features = [
                    str(feature).strip()
                    for feature in (result.get("selected_features") or [])
                    if str(feature).strip()
                ]
                selected_features = selected_features[:n_features_per_method]

                with open(method_log_paths[method], "a", encoding="utf-8") as handle:
                    if selected_features:
                        handle.write("\n".join(selected_features) + "\n")

                iteration_record["selectors"][method] = {
                    "n_selected": int(len(selected_features)),
                    "selected_features": selected_features,
                }

            iteration_trace.append(iteration_record)

        summary_path = os.path.join(temp_dir, "stability_run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "success": True,
                    "iterations_completed": int(iterations),
                    "methods_used": list(resolved_methods),
                    "n_features_per_method": int(n_features_per_method),
                    "sample_fraction": float(sample_fraction),
                    "data_already_scaled": data_already_scaled,
                    "mandatory_features": list(mandatory_features),
                    "iteration_trace": iteration_trace,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            {
                "success": True,
                "iterations_completed": int(iterations),
                "methods_used": list(resolved_methods),
                "n_features_per_method": int(n_features_per_method),
                "sample_fraction": float(sample_fraction),
                "data_already_scaled": data_already_scaled,
                "temp_dir": temp_dir,
                "summary_path": summary_path,
                "mandatory_features_protected": int(len(mandatory_features)),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return _selector_error(f"Train-only 稳定性选择失败: {str(e)}")


def _build_abs_correlation_matrix(
    data_path: str,
    group_col: str,
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    if not data_path or not os.path.exists(data_path):
        return {}

    try:
        df = _load_raw_dataframe(data_path)
        feature_cols = [feature for feature in feature_names if feature in df.columns]
        feature_cols = [
            feature for feature in feature_cols
            if feature != group_col and pd.api.types.is_numeric_dtype(df[feature])
        ]
        if len(feature_cols) < 2:
            return {}
        corr = (
            df[feature_cols]
            .corr(method="pearson")
            .abs()
            .fillna(0.0)
        )
        return {
            row_name: {col_name: float(corr.loc[row_name, col_name]) for col_name in corr.columns}
            for row_name in corr.index
        }
    except Exception:
        return {}


def _greedy_complete_panel(
    ranked_rows: List[Dict[str, Any]],
    mandatory_features: List[str],
    stable_core_features: List[str],
    final_panel_size: int,
    correlation_matrix: Dict[str, Dict[str, float]],
    max_pairwise_correlation: float,
) -> Tuple[List[str], Dict[str, Any]]:
    ordered_mandatory = list(dict.fromkeys([f for f in mandatory_features if f]))
    ordered_core = list(dict.fromkeys([f for f in stable_core_features if f]))
    selected = list(dict.fromkeys(ordered_mandatory + ordered_core))

    effective_target_size = max(int(final_panel_size), len(selected))
    decision_trace = {
        "requested_final_panel_size": int(final_panel_size),
        "effective_final_panel_size": int(effective_target_size),
        "expanded_for_protected_or_core": bool(effective_target_size > int(final_panel_size)),
        "redundancy_threshold": float(max_pairwise_correlation),
        "redundancy_skipped": [],
    }

    def _is_redundant(candidate: str) -> bool:
        if candidate in ordered_mandatory:
            return False
        for existing in selected:
            if existing in ordered_mandatory:
                continue
            corr_value = (
                correlation_matrix.get(candidate, {}).get(existing)
                or correlation_matrix.get(existing, {}).get(candidate)
                or 0.0
            )
            if corr_value >= max_pairwise_correlation:
                decision_trace["redundancy_skipped"].append(
                    {
                        "candidate": candidate,
                        "paired_with": existing,
                        "abs_correlation": float(corr_value),
                    }
                )
                return True
        return False

    for row in ranked_rows:
        feature = row["feature"]
        if feature in selected:
            continue
        if len(selected) >= effective_target_size:
            break
        if _is_redundant(feature):
            continue
        selected.append(feature)

    if len(selected) < effective_target_size:
        for row in ranked_rows:
            feature = row["feature"]
            if feature in selected:
                continue
            selected.append(feature)
            if len(selected) >= effective_target_size:
                break

    return selected, decision_trace


def build_stable_panel_from_logs(
    temp_dir: str,
    iterations: int,
    methods: List[str],
    final_panel_size: int,
    mandatory_features: Optional[List[str]] = None,
    selection_frequency_threshold: float = 0.60,
    method_consensus_threshold: int = 3,
    data_path: str = "",
    group_col: str = "",
    max_pairwise_correlation: float = 0.90,
    n_features_per_method: Optional[int] = None,
) -> str:
    """
    Convert selector logs into stable-core and final-panel outputs.
    """
    try:
        if iterations <= 0:
            raise ValueError("iterations 必须大于 0")
        methods = list(methods or [])
        if not methods:
            raise ValueError("methods 不能为空")

        mandatory_features = list(
            dict.fromkeys(
                [
                    str(feature).strip()
                    for feature in (mandatory_features or [])
                    if str(feature).strip()
                ]
            )
        )

        frequency_payload = json.loads(calculate_frequencies_from_logs(temp_dir))
        if isinstance(frequency_payload, dict) and frequency_payload.get("error"):
            raise ValueError(str(frequency_payload.get("error")))

        all_features = set(mandatory_features)
        for method in methods:
            method_counts = frequency_payload.get(method, {}) if isinstance(frequency_payload, dict) else {}
            if isinstance(method_counts, dict):
                all_features.update(str(feature).strip() for feature in method_counts.keys() if str(feature).strip())

        if data_path and os.path.exists(data_path):
            df = _load_raw_dataframe(data_path)
            if group_col and group_col in df.columns:
                all_features.update(_resolve_feature_columns(df, group_col))

        if not all_features:
            raise ValueError("未能从稳定性日志中解析任何候选特征")

        n_methods = max(1, len(methods))
        stability_rows: List[Dict[str, Any]] = []
        for feature in sorted(all_features):
            per_method_counts = {
                method: int((frequency_payload.get(method, {}) or {}).get(feature, 0))
                for method in methods
            }
            per_method_freq = {
                method: float(count) / float(iterations)
                for method, count in per_method_counts.items()
            }
            method_consensus = int(sum(1 for value in per_method_counts.values() if value > 0))
            selection_frequency = float(sum(per_method_counts.values())) / float(iterations * n_methods)
            is_prior = feature in mandatory_features
            stability_score = (
                0.75 * selection_frequency
                + 0.15 * (float(method_consensus) / float(n_methods))
                + 0.10 * (1.0 if is_prior else 0.0)
            )
            stability_rows.append(
                {
                    "feature": feature,
                    "selection_frequency": round(selection_frequency, 6),
                    "method_consensus": int(method_consensus),
                    "is_prior_protected": bool(is_prior),
                    "panel_score": round(float(stability_score), 6),
                    "per_method_frequency": {
                        method: round(float(value), 6)
                        for method, value in per_method_freq.items()
                    },
                    "per_method_count": per_method_counts,
                }
            )

        stability_rows.sort(
            key=lambda row: (
                -float(row["panel_score"]),
                -float(row["selection_frequency"]),
                -int(row["method_consensus"]),
                row["feature"],
            )
        )

        stable_core_features = list(
            dict.fromkeys(
                mandatory_features
                + [
                    row["feature"]
                    for row in stability_rows
                    if float(row["selection_frequency"]) >= float(selection_frequency_threshold)
                    and int(row["method_consensus"]) >= int(method_consensus_threshold)
                ]
            )
        )

        correlation_matrix = _build_abs_correlation_matrix(
            data_path=data_path,
            group_col=group_col,
            feature_names=[row["feature"] for row in stability_rows],
        )
        final_panel_features, completion_trace = _greedy_complete_panel(
            ranked_rows=stability_rows,
            mandatory_features=mandatory_features,
            stable_core_features=stable_core_features,
            final_panel_size=int(final_panel_size),
            correlation_matrix=correlation_matrix,
            max_pairwise_correlation=float(max_pairwise_correlation),
        )

        return json.dumps(
            {
                "success": True,
                "selector_family": list(methods),
                "iterations": int(iterations),
                "n_methods": int(n_methods),
                "n_features_per_method": None if n_features_per_method is None else int(n_features_per_method),
                "thresholds": {
                    "selection_frequency": float(selection_frequency_threshold),
                    "method_consensus": int(method_consensus_threshold),
                    "max_pairwise_correlation": float(max_pairwise_correlation),
                },
                "mandatory_features": mandatory_features,
                "stable_core_features": stable_core_features,
                "stable_core_count": int(len(stable_core_features)),
                "final_panel_features": final_panel_features,
                "final_panel_count": int(len(final_panel_features)),
                "panel_completion": completion_trace,
                "stability_scores": stability_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return _selector_error(f"稳定性面板构建失败: {str(e)}")


def get_adaptive_consensus_features(frequencies_json: str, min_features: int = 10) -> str:
    """
    Adaptive consensus feature selection with unified threshold scheduling.
    
    This function is the single source of truth for Phase 1 consensus selection.
    It supports both:
    1. Legacy single-run input: {method: {feature: count}}
    2. Multi-threshold scheduled input:
       {
         "threshold_runs": [
           {"selection_threshold": 0.30, "run_mode": "balanced", "frequencies": {...}},
           {"selection_threshold": 0.35, "run_mode": "balanced", "frequencies": {...}},
           {"selection_threshold": 0.40, "run_mode": "balanced", "frequencies": {...}}
         ]
       }

    For each threshold run, it performs:
    - Balanced mode: selection threshold schedule + vote threshold descent
    - Imbalanced mode: selection threshold schedule + frequency threshold descent + vote threshold descent
    
    Scaling Strategy:
    1. For each selection threshold run (e.g. 30%, 35%, 40%)
    2. Balanced mode: Votes: N -> ... -> 3
    3. Imbalanced mode: Try Freq >= 70% of iterations, then 60%; each with Votes: N -> ... -> 3
    4. Fallback: Union of all methods from the widest threshold run
    
    Note: Frequency thresholds are now relative to the number of iterations.
    For example, with 50 iterations:
    - 70% threshold = 35 times
    - 60% threshold = 30 times
    
    Privacy-Preserving Design:
    - Accepts JSON string of frequencies (not raw data)
    - Processes data locally
    - Returns only metadata (consensus feature names, no raw data)
    
    Args:
        frequencies_json: JSON string containing either:
                         - single-run frequencies: {"method1": {"feature1": count1, ...}, ...}
                         - scheduled threshold runs: {"threshold_runs": [{"selection_threshold": 0.30, "run_mode": "balanced|imbalanced", "frequencies": {...}}, ...]}
        min_features: Minimum number of features required in final consensus set.
                     Enforced lower bound is >10 (i.e., at least 11).
        
    Returns:
        JSON string with consensus features, strategy used, and metadata
        
    Example:
        freq_json = calculate_frequencies_from_logs('temp_stability')
        result_json = get_adaptive_consensus_features(freq_json, min_features=10)
        result = json.loads(result_json)
        if result['success']:
            features = result['consensus_features']
            strategy = result['strategy_used']
    """
    try:
        payload = json.loads(frequencies_json)
        
        # Handle error in input
        if isinstance(payload, dict) and 'error' in payload:
            return json.dumps({
                "success": False,
                "error": f"输入数据包含错误: {payload['error']}"
            })

        # Support unified multi-threshold payload while preserving backward compatibility.
        if isinstance(payload, dict) and "threshold_runs" in payload:
            threshold_runs = payload.get("threshold_runs", [])
        else:
            threshold_runs = [{"selection_threshold": None, "frequencies": payload}]

        valid_runs = []
        for run in threshold_runs:
            if not isinstance(run, dict):
                continue
            freq_data = run.get("frequencies", {})
            if isinstance(freq_data, dict) and freq_data:
                valid_runs.append({
                    "selection_threshold": run.get("selection_threshold", run.get("threshold", run.get("top_percentage"))),
                    "run_mode": str(run.get("run_mode", run.get("mode", ""))).strip().lower() or None,
                    "frequencies": freq_data
                })

        if not valid_runs:
            return json.dumps({
                "success": False,
                "error": "没有找到任何有效的频率数据或 threshold_runs 为空"
            })

        required_feature_count = max(int(min_features), 11)
        min_vote_floor = 3
        freq_percentages = [0.70, 0.60]
        threshold_attempts = []

        def _infer_iterations(freq_data: dict) -> int:
            max_count = 0
            for method_freqs in freq_data.values():
                if method_freqs:
                    method_max = max(method_freqs.values())
                    max_count = max(max_count, method_max)
            return max_count if max_count > 0 else 100

        for run in valid_runs:
            freq_data = run["frequencies"]
            total_methods = len(freq_data)
            if total_methods == 0:
                continue

            n_iterations = _infer_iterations(freq_data)
            vote_thresholds = [v for v in range(total_methods, min_vote_floor - 1, -1)]
            run_mode = run.get("run_mode")

            # Robust mode inference:
            # - explicit run_mode wins
            # - otherwise, max_count<=1 is treated as balanced one-shot selection
            inferred_balanced = (run_mode == "balanced") or (run_mode is None and n_iterations <= 1)
            if inferred_balanced:
                freq_schedule = [(None, None)]
            else:
                freq_thresholds = [max(1, int(n_iterations * pct)) for pct in freq_percentages]
                freq_schedule = list(zip(freq_thresholds, [70, 60]))

            for freq_t, freq_pct in freq_schedule:
                feature_votes = Counter()
                for method_freqs in freq_data.values():
                    if freq_t is None:
                        stable_feats = list(method_freqs.keys())
                    else:
                        stable_feats = [f for f, count in method_freqs.items() if count >= freq_t]
                    feature_votes.update(stable_feats)

                for vote_t in vote_thresholds:
                    consensus = [f for f, votes in feature_votes.items() if votes >= vote_t]
                    attempt_record = {
                        "selection_threshold": run["selection_threshold"],
                        "run_mode": "balanced" if inferred_balanced else "imbalanced",
                        "freq_threshold": freq_t,
                        "freq_percentage": freq_pct,
                        "vote_threshold": vote_t,
                        "count": len(consensus)
                    }
                    threshold_attempts.append(attempt_record)

                    if len(consensus) >= required_feature_count:
                        selection_threshold = run["selection_threshold"]
                        threshold_desc = (
                            f"Top {int(selection_threshold * 100)}% -> "
                            if isinstance(selection_threshold, (int, float))
                            else ""
                        )
                        strategy_desc = (
                            f"{threshold_desc}Votes>={vote_t}"
                            if inferred_balanced
                            else f"{threshold_desc}Freq>={freq_t} ({freq_pct}% of {n_iterations} iterations), Votes>={vote_t}"
                        )
                        return json.dumps({
                            "success": True,
                            "consensus_features": consensus,
                            "count": len(consensus),
                            "strategy_used": strategy_desc,
                            "selection_threshold": selection_threshold,
                            "run_mode": "balanced" if inferred_balanced else "imbalanced",
                            "freq_threshold": freq_t,
                            "freq_percentage": freq_pct,
                            "n_iterations": n_iterations,
                            "vote_threshold": vote_t,
                            "min_vote_floor": min_vote_floor,
                            "required_feature_count": required_feature_count,
                            "total_methods": total_methods,
                            "threshold_attempts": threshold_attempts,
                            "is_fallback": False
                        }, ensure_ascii=False, indent=2)

        # Ultimate fallback: union of the widest threshold run
        fallback_run = valid_runs[-1]
        fallback_freq_data = fallback_run["frequencies"]
        all_features = list(set([
            f for method_freqs in fallback_freq_data.values()
            for f in method_freqs.keys()
        ]))
        total_methods = len(fallback_freq_data)
        n_iterations = _infer_iterations(fallback_freq_data)

        return json.dumps({
            "success": True,
            "consensus_features": all_features,
            "count": len(all_features),
            "strategy_used": "Ultimate Fallback (Union of all methods from widest threshold run)",
            "selection_threshold": fallback_run["selection_threshold"],
            "run_mode": fallback_run.get("run_mode"),
            "freq_threshold": 0,
            "freq_percentage": 0,
            "n_iterations": n_iterations,
            "vote_threshold": 1,
            "min_vote_floor": min_vote_floor,
            "required_feature_count": required_feature_count,
            "total_methods": total_methods,
            "threshold_attempts": threshold_attempts,
            "is_fallback": True,
            "warning": (
                "在 30%/35%/40% 选择阈值调度下，"
                "并结合对应模式的投票/频率阈值下降尝试后，仍无法得到“>=3方法且特征数>10”的共识集合，"
                "已回退为最宽阈值下所有方法并集。"
            )
        }, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"JSON解析错误: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"自适应共识计算出错: {str(e)}"
        })


def filter_stable_features(frequencies: str, threshold: int) -> str:
    """
    Filter features based on selection frequency threshold from stability selection.
    
    Privacy-Preserving Design:
    - Accepts JSON string of frequencies (not raw data)
    - Processes data locally
    - Returns only metadata (stable feature names, no raw data)
    
    Args:
        frequencies: JSON string containing feature frequencies from calculate_frequencies_from_logs
        threshold: Minimum frequency threshold for a feature to be considered stable
        
    Returns:
        JSON string with stable features for each method, threshold used, and summary counts
    """
    try:
        freq_data = json.loads(frequencies)
        stable_features = {}
        for method, feature_freqs in freq_data.items():
            stable_features[method] = [f for f, freq in feature_freqs.items() if freq >= threshold]
        
        result = {
            "stable_features": stable_features,
            "threshold": threshold,
            "summary": {m: len(f) for m, f in stable_features.items()}
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"筛选稳定特征时出错: {str(e)}"})


def find_consensus_features(stable_features_map: str, min_required_features: int) -> str:
    """
    Find consensus features across multiple selection methods with graceful degradation.
    
    Starts with the strictest consensus (all methods agree) and progressively relaxes 
    the voting requirement until the minimum feature count is met.
    
    Privacy-Preserving Design:
    - Accepts JSON string of stable features (not raw data)
    - Processes data locally
    - Returns only metadata (consensus feature names, no raw data)
    
    Args:
        stable_features_map: JSON string containing stable features from filter_stable_features
        min_required_features: Minimum number of features required in final consensus set
        
    Returns:
        JSON string with consensus features, count, minimum votes used, total methods, and strategy
    """
    try:
        stable_data = json.loads(stable_features_map)
        stable_features = stable_data.get("stable_features", {})
        if not stable_features:
            return json.dumps({"error": "没有找到稳定特征数据"})
        
        total_methods = len(stable_features)
        feature_votes = Counter()
        for features in stable_features.values():
            feature_votes.update(features)

        required_feature_count = max(int(min_required_features), 11)
        min_vote_floor = 3

        # 从最严格投票数开始，逐步降低到 3 票，直到满足“特征数>10”
        consensus_features = []
        votes_needed = min_vote_floor
        for votes_needed in range(total_methods, min_vote_floor - 1, -1):
            consensus_features = [f for f, v in feature_votes.items() if v >= votes_needed]
            if len(consensus_features) >= required_feature_count:
                break
        else:
            # fallback to union of all methods
            consensus_features = list(feature_votes.keys())
            votes_needed = 1

        result = {
            "consensus_features": consensus_features,
            "count": len(consensus_features),
            "min_votes_used": votes_needed,
            "min_vote_floor": min_vote_floor,
            "required_feature_count": required_feature_count,
            "total_methods": total_methods,
            "strategy": (
                f"Found with >= {votes_needed} votes"
                if votes_needed >= min_vote_floor
                else "Fallback to union after vote-floor descent"
            )
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"查找共识特征时出错: {str(e)}"})


def run_rank_aggregation(data_path: str, group_col: str, n_features: int) -> str:
    """
    保底工具：秩和排序...
    """
    try:
        # [优化] 调用公共函数
        X_scaled, y, feature_names = _load_and_prepare_data(data_path, group_col)
        
        method_ranks = {}
        
        # 随机森林排名
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            rf_importance = dict(zip(feature_names, rf.feature_importances_))
            rf_ranked = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
            method_ranks["random_forest"] = {feature: rank+1 for rank, (feature, _) in enumerate(rf_ranked)}
        except Exception as e:
            print(f"随机森林排名失败: {str(e)}")
        
        # Lasso排名
        try:
            y_encoded, _ = pd.factorize(y)
            lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
            lasso.fit(X_scaled, y_encoded)
            lasso_coef = dict(zip(feature_names, np.abs(lasso.coef_)))
            lasso_ranked = sorted(lasso_coef.items(), key=lambda x: x[1], reverse=True)
            method_ranks["lasso"] = {feature: rank+1 for rank, (feature, _) in enumerate(lasso_ranked)}
        except Exception as e:
            print(f"Lasso排名失败: {str(e)}")
        
        # 统计方法排名
        try:
            f_scores, _ = f_classif(X_scaled, y)
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            combined_scores = (f_scores + mi_scores) / 2
            stat_scores = dict(zip(feature_names, combined_scores))
            stat_ranked = sorted(stat_scores.items(), key=lambda x: x[1], reverse=True)
            method_ranks["statistical"] = {feature: rank+1 for rank, (feature, _) in enumerate(stat_ranked)}
        except Exception as e:
            print(f"统计方法排名失败: {str(e)}")
        
        # 互信息排名
        try:
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            mi_scores_dict = dict(zip(feature_names, mi_scores))
            mi_ranked = sorted(mi_scores_dict.items(), key=lambda x: x[1], reverse=True)
            method_ranks["mutual_information"] = {feature: rank+1 for rank, (feature, _) in enumerate(mi_ranked)}
        except Exception as e:
            print(f"互信息排名失败: {str(e)}")
        
        # LinearSVC排名
        try:
            lsvc = LinearSVC(penalty='l1', dual=False, random_state=42, max_iter=2000)
            lsvc.fit(X_scaled, y)
            coefficients = lsvc.coef_[0]
            svc_scores = dict(zip(feature_names, np.abs(coefficients)))
            # 筛选出系数不为零的特征并排序
            non_zero_features = {k: v for k, v in svc_scores.items() if v > 1e-5}
            svc_ranked = sorted(non_zero_features.items(), key=lambda x: x[1], reverse=True)
            method_ranks["linear_svc"] = {feature: rank+1 for rank, (feature, _) in enumerate(svc_ranked)}
        except Exception as e:
            print(f"LinearSVC排名失败: {str(e)}")
        
        # ReliefF排名
        try:
            from skrebate import ReliefF
            y_encoded = LabelEncoder().fit_transform(y)
            relieff = ReliefF()
            relieff.fit(X_scaled.values, y_encoded)
            relief_scores = dict(zip(feature_names, relieff.feature_importances_))
            relief_ranked = sorted(relief_scores.items(), key=lambda x: x[1], reverse=True)
            method_ranks["relief_f"] = {feature: rank+1 for rank, (feature, _) in enumerate(relief_ranked)}
        except ImportError:
            print("ReliefF需要安装skrebate库，跳过")
        except Exception as e:
            print(f"ReliefF排名失败: {str(e)}")
        
        # CFS排名 (CFS返回的是特征子集，不是排名，所以我们需要特殊处理)
        try:
            from skfeature.function.statistical_based.CFS import cfs
            y_encoded = LabelEncoder().fit_transform(y)
            selected_indices = cfs(X_scaled.values, y_encoded)
            selected_features = [feature_names[idx] for idx in selected_indices]
            # 为选中的特征分配排名，未选中的特征排名靠后
            cfs_ranks = {}
            for rank, feature in enumerate(selected_features, 1):
                cfs_ranks[feature] = rank
            for feature in feature_names:
                if feature not in cfs_ranks:
                    cfs_ranks[feature] = len(feature_names) + 1
            method_ranks["cfs"] = cfs_ranks
        except ImportError:
            print("CFS需要安装skfeature-chappers库，跳过")
        except Exception as e:
            print(f"CFS排名失败: {str(e)}")
        
        # FCBF排名 (类似CFS的处理)
        try:
            from skfeature.function.information_theoretical_based.FCBF import fcbf
            y_encoded = LabelEncoder().fit_transform(y)
            selected_indices, _ = fcbf(X_scaled.values, y_encoded, n_selected_features=None)
            selected_features = [feature_names[idx] for idx in selected_indices]
            fcbf_ranks = {}
            for rank, feature in enumerate(selected_features, 1):
                fcbf_ranks[feature] = rank
            for feature in feature_names:
                if feature not in fcbf_ranks:
                    fcbf_ranks[feature] = len(feature_names) + 1
            method_ranks["fcbf"] = fcbf_ranks
        except ImportError:
            print("FCBF需要安装skfeature-chappers库，跳过")
        except Exception as e:
            print(f"FCBF排名失败: {str(e)}")
        
        # mRMR排名
        try:
            import mrmr
            selected_features = _run_mrmr_classif_silently(
                X=X_scaled,
                y=y,
                K=min(n_features, len(feature_names)),
                n_jobs=1,
            )
            mrmr_ranks = {}
            for rank, feature in enumerate(selected_features, 1):
                mrmr_ranks[feature] = rank
            for feature in feature_names:
                if feature not in mrmr_ranks:
                    mrmr_ranks[feature] = len(feature_names) + 1
            method_ranks["mrmr"] = mrmr_ranks
        except ImportError:
            print("mRMR需要安装mrmr-selection库，跳过")
        except Exception as e:
            print(f"mRMR排名失败: {str(e)}")

        if not method_ranks:
            return json.dumps({"error": "所有排名方法都失败了"})
            
        rank_scores = {}
        for feature in feature_names:
            ranks = [method_ranks[method].get(feature, len(feature_names)) for method in method_ranks.keys()]
            rank_scores[feature] = sum(ranks) / len(ranks)

        top_features = [f for f, s in sorted(rank_scores.items(), key=lambda x: x[1])[:n_features]]
        
        return json.dumps({
            "method": "rank_aggregation",
            "selected_features": top_features,
            "count": len(top_features)
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"秩和排序时出错: {str(e)}"})


def create_filtered_dataset(
    data_path: str,
    target_column: str,
    selected_features: List[str],
    output_path: str = None
) -> str:
    """
    Create a filtered dataset containing only selected features and the target column.
    
    This tool is called after feature selection to extract selected features from the 
    complete dataset, creating a new dataset ready for model building.
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame
    - Reads data locally from CSV/Excel file
    - Processes data locally (filters columns)
    - Saves filtered dataset to local file
    - Returns only metadata (output path, shape, column names, no raw data)
    
    Args:
        data_path: Path to input CSV or Excel file (typically cleaned data)
        target_column: Name of the target/label column
        selected_features: List of feature names to keep
        output_path: Optional output file path (default: input_path_selected.csv)
        
    Returns:
        JSON string with success status, output path, feature count, shape, and column names
        
    Example:
        result = create_filtered_dataset(
            data_path="data_cleaned.csv",
            target_column="label",
            selected_features=["feature1", "feature2", "feature3"]
        )
        # Returns: {"success": true, "output_path": "data_cleaned_selected.csv", ...}
    """
    try:
        # 加载数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"error": "不支持的文件格式，请使用 .csv 或 .xlsx"})
        
        # 检查目标列
        if target_column not in df.columns:
            return json.dumps({"error": f"目标列 '{target_column}' 不存在于数据集中"})
        
        # 检查选定的特征是否存在
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            return json.dumps({
                "error": f"以下特征不存在于数据集中: {missing_features}"
            })
        
        # 只保留选定的特征和目标列
        columns_to_keep = selected_features + [target_column]
        df_filtered = df[columns_to_keep]
        
        # 确定输出路径
        if output_path is None:
            if '_cleaned.csv' in data_path:
                output_path = data_path.replace('_cleaned.csv', '_cleaned_selected.csv')
            elif '_selected.csv' in data_path:
                # 如果已经是筛选后的文件，替换
                output_path = data_path
            else:
                output_path = data_path.replace('.csv', '_selected.csv')
        
        # 保存筛选后的数据
        df_filtered.to_csv(output_path, index=False)
        
        return json.dumps({
            "success": True,
            "output_path": output_path,
            "n_features": len(selected_features),
            "shape": list(df_filtered.shape),
            "columns": df_filtered.columns.tolist(),
            "message": f"成功创建筛选后的数据集，包含 {len(selected_features)} 个特征"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"创建筛选数据集时发生错误: {str(e)}"})


# ============================================================================
# LangChain Tool Wrappers
# ============================================================================
from langchain_core.tools import StructuredTool

# Create tool wrappers for LangChain/LangGraph
check_balance_tool = StructuredTool.from_function(func=check_balance, name="check_balance", description=check_balance.__doc__)
run_random_forest_selector_tool = StructuredTool.from_function(func=run_random_forest_selector, name="run_random_forest_selector", description=run_random_forest_selector.__doc__)
run_lasso_selector_tool = StructuredTool.from_function(func=run_lasso_selector, name="run_lasso_selector", description=run_lasso_selector.__doc__)
run_lightgbm_selector_tool = StructuredTool.from_function(func=run_lightgbm_selector, name="run_lightgbm_selector", description=run_lightgbm_selector.__doc__)
run_xgboost_selector_tool = StructuredTool.from_function(func=run_xgboost_selector, name="run_xgboost_selector", description=run_xgboost_selector.__doc__)
run_elasticnet_selector_tool = StructuredTool.from_function(func=run_elasticnet_selector, name="run_elasticnet_selector", description=run_elasticnet_selector.__doc__)
run_mutual_info_selector_tool = StructuredTool.from_function(func=run_mutual_info_selector, name="run_mutual_info_selector", description=run_mutual_info_selector.__doc__)
run_t_test_selector_tool = StructuredTool.from_function(func=run_t_test_selector, name="run_t_test_selector", description=run_t_test_selector.__doc__)
run_fdr_effect_size_selector_tool = StructuredTool.from_function(func=run_fdr_effect_size_selector, name="run_fdr_effect_size_selector", description=run_fdr_effect_size_selector.__doc__)
run_f_statistic_selector_tool = StructuredTool.from_function(func=run_f_statistic_selector, name="run_f_statistic_selector", description=run_f_statistic_selector.__doc__)
run_anova_selector_tool = StructuredTool.from_function(func=run_anova_selector, name="run_anova_selector", description=run_anova_selector.__doc__)
run_fcbf_selector_tool = StructuredTool.from_function(func=run_fcbf_selector, name="run_fcbf_selector", description=run_fcbf_selector.__doc__)
run_cfs_selector_tool = StructuredTool.from_function(func=run_cfs_selector, name="run_cfs_selector", description=run_cfs_selector.__doc__)
run_relief_selector_tool = StructuredTool.from_function(func=run_relief_selector, name="run_relief_selector", description=run_relief_selector.__doc__)
run_statistical_selector_tool = StructuredTool.from_function(func=run_statistical_selector, name="run_statistical_selector", description=run_statistical_selector.__doc__)
run_mutual_information_selector_tool = StructuredTool.from_function(func=run_mutual_information_selector, name="run_mutual_information_selector", description=run_mutual_information_selector.__doc__)
run_linear_svm_selector_tool = StructuredTool.from_function(func=run_linear_svm_selector, name="run_linear_svm_selector", description=run_linear_svm_selector.__doc__)
run_sbs_selector_tool = StructuredTool.from_function(func=run_sbs_selector, name="run_sbs_selector", description=run_sbs_selector.__doc__)
run_mrmr_selector_tool = StructuredTool.from_function(func=run_mrmr_selector, name="run_mrmr_selector", description=run_mrmr_selector.__doc__)
run_ga_selector_tool = StructuredTool.from_function(func=run_ga_selector, name="run_ga_selector", description=run_ga_selector.__doc__)
perform_stability_selection_tool = StructuredTool.from_function(func=perform_stability_selection, name="perform_stability_selection", description=perform_stability_selection.__doc__)
calculate_frequencies_from_logs_tool = StructuredTool.from_function(func=calculate_frequencies_from_logs, name="calculate_frequencies_from_logs", description=calculate_frequencies_from_logs.__doc__)
get_adaptive_consensus_features_tool = StructuredTool.from_function(func=get_adaptive_consensus_features, name="get_adaptive_consensus_features", description=get_adaptive_consensus_features.__doc__)
filter_stable_features_tool = StructuredTool.from_function(func=filter_stable_features, name="filter_stable_features", description=filter_stable_features.__doc__)
find_consensus_features_tool = StructuredTool.from_function(func=find_consensus_features, name="find_consensus_features", description=find_consensus_features.__doc__)
run_rank_aggregation_tool = StructuredTool.from_function(func=run_rank_aggregation, name="run_rank_aggregation", description=run_rank_aggregation.__doc__)
create_filtered_dataset_tool = StructuredTool.from_function(func=create_filtered_dataset, name="create_filtered_dataset", description=create_filtered_dataset.__doc__)

# 工具列表，用于LangGraph注册（使用包装器）
FEATURE_SELECTION_TOOLS = [
    # 数据诊断工具
    check_balance_tool,
    
    # 核心特征选择工具（树模型）
    run_random_forest_selector_tool,
    run_lightgbm_selector_tool,
    run_xgboost_selector_tool,
    
    # 核心特征选择工具（线性模型）
    run_lasso_selector_tool,
    run_elasticnet_selector_tool,
    
    # 核心特征选择工具（统计方法）
    run_mutual_info_selector_tool,
    run_t_test_selector_tool,
    run_fdr_effect_size_selector_tool,
    run_f_statistic_selector_tool,
    run_anova_selector_tool,
    run_statistical_selector_tool,
    run_mutual_information_selector_tool,
    
    # 高级特征选择工具
    run_fcbf_selector_tool,
    run_cfs_selector_tool,
    run_relief_selector_tool,
    run_linear_svm_selector_tool,
    run_sbs_selector_tool,
    run_mrmr_selector_tool,
    run_ga_selector_tool,
    
    # 稳定性选择工具
    perform_stability_selection_tool,
    calculate_frequencies_from_logs_tool,
    get_adaptive_consensus_features_tool,  # 🔑 2D梯度阈值缩放共识特征选择
    filter_stable_features_tool,
    find_consensus_features_tool,
    
    # 保底工具
    run_rank_aggregation_tool,
    
    # 数据集筛选工具（特征选择完成后使用）
    create_filtered_dataset_tool  # 🔑 在特征选择完成后创建筛选数据集
]
