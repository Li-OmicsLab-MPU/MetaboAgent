"""
Tool Contracts Registry

This module defines the API contracts for all tools used in Phase 1.
Contracts are organized in three levels of detail:
1. Minimal: Function signature only
2. Standard: Signature + parameter descriptions + return format
3. Full: Complete with examples and edge cases

This enables progressive disclosure based on retry level:
- First attempt: Minimal (save tokens)
- First retry: Standard (add clarity)
- Second retry: Full (maximum guidance)
"""

from typing import Dict, Any, List

# ============================================================================
# Tool Contracts - Organized by Module
# ============================================================================

TOOL_CONTRACTS = {
    # ========== Data Analysis Tools ==========
    "analyze_dataset_quality": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "analyze_dataset_quality(file_path: str) -> str",
        "standard": {
            "signature": "analyze_dataset_quality(file_path: str) -> str",
            "params": {
                "file_path": "数据文件路径（CSV或Excel）"
            },
            "returns": "JSON字符串: {missing_values: dict, numeric_columns_stats: dict, potential_outlier_columns: list, ...}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import analyze_dataset_quality
import json

# 分析数据质量
result_json = analyze_dataset_quality(file_path='data/input.csv')
result = json.loads(result_json)

# 访问结果
missing_info = result['missing_values']  # {'col_name': {'missing_count': int, 'missing_percentage': float}}
stats = result['numeric_columns_stats']  # {'col_name': {'mean': float, 'std': float, ...}}
"""
    },

    "data_context_resolver_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "data_context_resolver_tool(dataset_path: str, study_metadata_path: str = '', column_roles_path: str = '', source_hint: str = '', user_declared_data_level: str = '') -> str",
        "standard": {
            "signature": "data_context_resolver_tool(dataset_path: str, study_metadata_path: str = '', column_roles_path: str = '', source_hint: str = '', user_declared_data_level: str = '') -> str",
            "params": {
                "dataset_path": "数据文件路径（CSV或Excel）",
                "study_metadata_path": "研究元数据路径（可选）",
                "column_roles_path": "列角色JSON路径（可选）",
                "source_hint": "数据来源提示，例如 'MetaboLights'",
                "user_declared_data_level": "用户声明的数据层级，可选 processed_matrix 或 raw_like_matrix_with_qc_metadata"
            },
            "returns": "JSON字符串: {platform_source: str, data_level: str, has_pooled_qc: bool, candidate_branch: str, protected_columns: List[str], ...}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import data_context_resolver_tool
import json

result_json = data_context_resolver_tool(
    dataset_path='data/input.csv',
    study_metadata_path='data/study_metadata.json',
    column_roles_path='data/column_roles.json',
    source_hint='Metabolomics Workbench'
)
result = json.loads(result_json)

if result['success']:
    branch = result['candidate_branch']
    protected = result['protected_columns']
"""
    },

    "zero_semantics_analyzer_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "zero_semantics_analyzer_tool(dataset_path: str, context_report_path: str, study_metadata_path: str = '', protected_columns: List[str] = None) -> str",
        "standard": {
            "signature": "zero_semantics_analyzer_tool(dataset_path: str, context_report_path: str, study_metadata_path: str = '', protected_columns: List[str] = None) -> str",
            "params": {
                "dataset_path": "数据文件路径（CSV或Excel）",
                "context_report_path": "preprocessing_context.json 路径",
                "study_metadata_path": "研究元数据路径（可选）",
                "protected_columns": "额外保护列（可选）"
            },
            "returns": "JSON字符串: {total_zero_count: int, zero_rate_global: float, zero_semantics_status: str, suggested_zero_handling: str, top_zero_features: List[dict], ...}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import zero_semantics_analyzer_tool
import json

result_json = zero_semantics_analyzer_tool(
    dataset_path='data/input.csv',
    context_report_path='output/preprocessing_context.json',
    study_metadata_path='data/study_metadata.json'
)
result = json.loads(result_json)

if result['success']:
    handling = result['suggested_zero_handling']
    evidence = result['evidence_flags']
"""
    },

    "missingness_assessment_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "missingness_assessment_tool(dataset_path: str, context_report_path: str, zero_pattern_report_path: str = '', zero_handling_mode: str = 'auto', protected_columns: List[str] = None) -> str",
        "standard": {
            "signature": "missingness_assessment_tool(dataset_path: str, context_report_path: str, zero_pattern_report_path: str = '', zero_handling_mode: str = 'auto', protected_columns: List[str] = None) -> str",
            "params": {
                "dataset_path": "数据文件路径（CSV或Excel）",
                "context_report_path": "preprocessing_context.json 路径",
                "zero_pattern_report_path": "zero_pattern_report.json 路径（可选）",
                "zero_handling_mode": "零值处理模式: 'keep_zero' | 'convert_zero_to_nan' | 'auto'",
                "protected_columns": "额外保护列（可选）"
            },
            "returns": "JSON字符串: {total_missing_count: int, suspected_mechanism: str, recommended_imputation_family: str, feature_missing_summary: dict, sample_missing_summary: dict, ...}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import missingness_assessment_tool
import json

result_json = missingness_assessment_tool(
    dataset_path='data/input.csv',
    context_report_path='output/preprocessing_context.json',
    zero_pattern_report_path='output/zero_pattern_report.json',
    zero_handling_mode='auto'
)
result = json.loads(result_json)

if result['success']:
    mechanism = result['suspected_mechanism']
    family = result['recommended_imputation_family']
    feature_summary = result['feature_missing_summary']
"""
    },

    "preprocessing_qa_reporter_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "preprocessing_qa_reporter_tool(context_report_path: str, schema_audit_report_path: str = '', eligibility_filter_report_path: str = '', zero_pattern_report_path: str = '', missingness_report_path: str = '', imputation_report_path: str = '', normalization_decision_report_path: str = '', transformation_report_path: str = '', outlier_audit_report_path: str = '', batch_drift_audit_path: str = '', feature_engineering_report_path: str = '', scaling_report_path: str = '') -> str",
        "standard": {
            "signature": "preprocessing_qa_reporter_tool(context_report_path: str, schema_audit_report_path: str = '', eligibility_filter_report_path: str = '', zero_pattern_report_path: str = '', missingness_report_path: str = '', imputation_report_path: str = '', normalization_decision_report_path: str = '', transformation_report_path: str = '', outlier_audit_report_path: str = '', batch_drift_audit_path: str = '', feature_engineering_report_path: str = '', scaling_report_path: str = '') -> str",
            "params": {
                "context_report_path": "preprocessing_context.json 路径",
                "schema_audit_report_path": "schema audit 报告路径（可选）",
                "eligibility_filter_report_path": "eligibility filter 报告路径（可选）",
                "zero_pattern_report_path": "zero pattern 报告路径（可选）",
                "missingness_report_path": "missingness 报告路径（可选）",
                "imputation_report_path": "imputation 报告路径（可选）",
                "normalization_decision_report_path": "normalization 报告路径（可选）",
                "transformation_report_path": "transformation 报告路径（可选）",
                "outlier_audit_report_path": "outlier audit 报告路径（可选）",
                "batch_drift_audit_path": "batch/drift audit 报告路径（可选）",
                "feature_engineering_report_path": "feature engineering 报告路径（可选）",
                "scaling_report_path": "scaling 报告路径（可选）"
            },
            "returns": "JSON字符串: {preprocessing_report_path: str, preprocessing_summary_path: str, preprocessing_decision_pack_path: str, branch_used: str, phase4_reportable_facts: dict}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import preprocessing_qa_reporter_tool
import json

result_json = preprocessing_qa_reporter_tool(
    context_report_path='output/preprocessing_context.json',
    zero_pattern_report_path='output/zero_pattern_report.json',
    missingness_report_path='output/missingness_report.json'
)
result = json.loads(result_json)

if result['success']:
    report_path = result['preprocessing_report_path']
    summary_path = result['preprocessing_summary_path']
    decision_pack_path = result['preprocessing_decision_pack_path']
"""
    },
    
    "delete_columns_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "delete_columns_tool(columns: List[str], data_path: str) -> str",
        "standard": {
            "signature": "delete_columns_tool(columns: List[str], data_path: str) -> str",
            "params": {
                "columns": "要删除的列名列表",
                "data_path": "数据文件路径（会被原地修改）"
            },
            "returns": "JSON字符串: {success: bool, output_path: str, deleted_columns: List[str]}"
        },
        "full": """
**delete_columns_tool** (完整示例)

从数据集中删除指定列。工具会原地修改文件。

```python
from src.tools.analysis.data_analysis_tools import delete_columns_tool
import json

# 删除不需要的列
result_json = delete_columns_tool(
    columns=['bad_col1', 'bad_col2'],  # 要删除的列名列表
    data_path='data/input.csv'  # 数据文件路径（会被原地修改）
)

# 解析返回的 JSON 字符串
result = json.loads(result_json)

# 检查执行结果
if result['success']:
    print(f"Successfully deleted: {result['deleted_columns']}")
    updated_path = result['output_path']  # 通常与输入路径相同（原地修改）
    
    # 使用 Magic Output Protocol 更新状态
    print(json.dumps({
        '__METABO_UPDATE__': {
            'current_data_path': updated_path,
            'columns_deleted': result['deleted_columns']
        }
    }))
else:
    print(f"Error: {result.get('error')}")
    raise ValueError(result['error'])
```

**常见错误**:
- ❌ 传递 DataFrame 对象而不是文件路径
- ❌ 忘记用 json.loads() 解析返回值
- ❌ 使用错误的参数名（如 'cols' 而不是 'columns'）

**最佳实践**:
- ✅ 始终检查 result['success']
- ✅ 使用 try-except 捕获异常
- ✅ 通过 Magic Output Protocol 更新状态
"""
    },
    
    "delete_rows_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "delete_rows_tool(condition: str, data_path: str) -> str",
        "standard": {
            "signature": "delete_rows_tool(condition: str, data_path: str) -> str",
            "params": {
                "condition": "删除条件字符串，如 'missing_rate > 0.35'",
                "data_path": "数据文件路径（会被原地修改）"
            },
            "returns": "JSON字符串: {success: bool, output_path: str, deleted_rows: int}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import delete_rows_tool
import json

result_json = delete_rows_tool(
    condition='missing_rate > 0.35',
    data_path='data/input.csv'
)
result = json.loads(result_json)

if result['success']:
    print(f"Deleted {result['deleted_rows']} rows")
"""
    },
    
    "impute_missing_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "impute_missing_tool(columns: List[str], data_path: str, method: str = 'mean') -> str",
        "standard": {
            "signature": "impute_missing_tool(columns: List[str], data_path: str, method: str = 'mean') -> str",
            "params": {
                "columns": "要填充的列名列表",
                "data_path": "数据文件路径（会被原地修改）",
                "method": "填充方法: 'mean'|'median'|'mode'|'knn'|'qrilc'"
            },
            "returns": "JSON字符串: {success: bool, output_path: str, imputed_columns: List[str], method: str}"
        },
        "full": """
**impute_missing_tool** (完整示例)

使用指定方法填充缺失值。支持多种插补策略。

```python
from src.tools.analysis.data_analysis_tools import impute_missing_tool
import json
import pandas as pd

# 加载数据以识别有缺失值的列
df = pd.read_csv('data/input.csv')
cols_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]

# 选择插补方法
# - 'mean': 均值填充（适用于正态分布）
# - 'median': 中位数填充（适用于偏态分布）
# - 'knn': K近邻填充（考虑特征间关系）
# - 'qrilc': QRILC方法（专门处理低于检测限的MNAR数据）

result_json = impute_missing_tool(
    columns=cols_with_missing,  # 要填充的列名列表
    data_path='data/input.csv',  # 数据文件路径（会被原地修改）
    method='knn'  # 插补方法
)

# 解析返回的 JSON 字符串
result = json.loads(result_json)

# 检查执行结果
if result['success']:
    print(f"Successfully imputed {len(result['imputed_columns'])} columns")
    print(f"Method used: {result['method']}")
    updated_path = result['output_path']
    
    # 使用 Magic Output Protocol 更新状态
    print(json.dumps({
        '__METABO_UPDATE__': {
            'current_data_path': updated_path,
            'imputation_method': result['method'],
            'imputed_columns': result['imputed_columns']
        }
    }))
else:
    print(f"Error: {result.get('error')}")
    # 可以尝试降级到更简单的方法
    print("Trying fallback method: median")
    result_json = impute_missing_tool(columns=cols_with_missing, data_path='data/input.csv', method='median')
    result = json.loads(result_json)
```

**方法选择指南**:
- MNAR数据（低于检测限）→ 'qrilc'
- 多变量相关性强 → 'knn'
- 正态分布 → 'mean'
- 偏态分布 → 'median'

**常见错误**:
- ❌ 传递 DataFrame 而不是文件路径
- ❌ 忘记解析 JSON 返回值
- ❌ 使用错误的 method 参数值

**最佳实践**:
- ✅ 先分析数据分布再选择方法
- ✅ 为 MNAR 数据使用 'qrilc'
- ✅ 大数据集避免使用 'knn'（性能问题）
"""
    },
    
    "pqn_normalization_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "pqn_normalization_tool(data_path: str, sample_column: str = None) -> str",
        "standard": {
            "signature": "pqn_normalization_tool(data_path: str, sample_column: str = None) -> str",
            "params": {
                "data_path": "数据文件路径（会被原地修改）",
                "sample_column": "样本ID列名（默认使用第一列）"
            },
            "returns": "JSON字符串: {success: bool, output_path: str, normalized_columns: List[str]}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import pqn_normalization_tool
import json

result_json = pqn_normalization_tool(
    data_path='data/preprocessed.csv',
    sample_column='Sample_ID'
)
result = json.loads(result_json)

if result['success']:
    normalized_path = result['output_path']  # Same as input (in-place)
"""
    },
    
    "log2_transformation_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "log2_transformation_tool(data_path: str, columns: List[str] = None) -> str",
        "standard": {
            "signature": "log2_transformation_tool(data_path: str, columns: List[str] = None) -> str",
            "params": {
                "data_path": "数据文件路径",
                "columns": "要转换的列（None=所有数值列）"
            },
            "returns": "JSON字符串: {success: bool, output_path: str}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import log2_transformation_tool
import json

result_json = log2_transformation_tool(data_path='data/normalized.csv', columns=None)
result = json.loads(result_json)

if result['success']:
    log2_path = result['output_path']
"""
    },
    
    "cap_outliers_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "cap_outliers_tool(data_path: str, columns: List[str] = None, method: str = 'iqr') -> str",
        "standard": {
            "signature": "cap_outliers_tool(data_path: str, columns: List[str] = None, method: str = 'iqr') -> str",
            "params": {
                "data_path": "数据文件路径",
                "columns": "要处理的列（None=所有数值列）",
                "method": "异常值检测方法（目前仅支持'iqr'）"
            },
            "returns": "JSON字符串: {success: bool, output_path: str, capped_columns: List[str]}"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import cap_outliers_tool
import json

result_json = cap_outliers_tool(data_path='data/log2.csv', columns=None, method='iqr')
result = json.loads(result_json)

if result['success']:
    capped_path = result['output_path']
"""
    },
    
    "auto_scaling_tool": {
        "module": "src.tools.analysis.data_analysis_tools",
        "minimal": "auto_scaling_tool(data_path: str, columns: List[str] = None, method: str = 'standard') -> str",
        "standard": {
            "signature": "auto_scaling_tool(data_path: str, columns: List[str] = None, method: str = 'standard') -> str",
            "params": {
                "data_path": "数据文件路径。对于 Phase 1 Step 1.5.5，应直接使用当前已准备好的 current_data_path，不要自行重新扫描 engineered 文件路径，也不要根据 engineered_features_merged / n_ratio_features / n_sum_features / n_pathway_scores 做额外前置拦截",
                "columns": "要缩放的列（None=所有数值列）",
                "method": "缩放方法: 'standard'(Z-score) | 'minmax'"
            },
            "returns": "JSON字符串: {success: bool, output_path: str}. 注意: 返回值是 JSON 字符串，需要 json.loads(...) 后读取 output_path；不要在 Step 1.5.5 里再定义 has_engineered_markers()/resolve_input_path()/choose_engineered_source() 之类 helper 去重新解析输入路径，也不要自行抛出 'Auto-scaling aborted' 这类前置错误。"
        },
        "full": """
from src.tools.analysis.data_analysis_tools import auto_scaling_tool
import json

# For Phase 1 Step 1.5.5, current_data_path is already the prepared input.
# Do NOT define custom helpers to rediscover engineered files.
result_json = auto_scaling_tool(data_path=current_data_path, columns=None, method='standard')
result = json.loads(result_json)

if result['success']:
    scaled_path = result['output_path']
"""
    },
    
    # ========== Feature Selection Tools ==========
    "run_random_forest_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_random_forest_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_random_forest_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "数据文件路径",
                "group_col": "目标列名（通常是'Group'）",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_scores: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_random_forest_selector
import json

result_json = run_random_forest_selector(
    data_path='data/features.csv',
    group_col='Group',
    n_features=50
)
result = json.loads(result_json)
rf_features = result['selected_features']
"""
    },

    "run_lightgbm_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_lightgbm_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_lightgbm_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_scores: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_lightgbm_selector
import json

result_json = run_lightgbm_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
lightgbm_features = result['selected_features']
"""
    },

    "run_xgboost_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_xgboost_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_xgboost_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_scores: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_xgboost_selector
import json

result_json = run_xgboost_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
xgboost_features = result['selected_features']
"""
    },
    
    "run_lasso_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_lasso_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_lasso_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "数据文件路径",
                "group_col": "目标列名",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_coefficients: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_lasso_selector
import json

result_json = run_lasso_selector(
    data_path='data/features.csv',
    group_col='Group',
    n_features=50
)
result = json.loads(result_json)
lasso_features = result['selected_features']
"""
    },

    "run_elasticnet_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_elasticnet_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_elasticnet_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_coefficients: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_elasticnet_selector
import json

result_json = run_elasticnet_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
elasticnet_features = result['selected_features']
"""
    },

    "run_mutual_info_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_mutual_info_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_mutual_info_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_scores: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_mutual_info_selector
import json

result_json = run_mutual_info_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
mi_features = result['selected_features']
"""
    },

    "run_t_test_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_t_test_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_t_test_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group；t-test 适用于二分类",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], t_scores: dict, p_values: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_t_test_selector
import json

result_json = run_t_test_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
t_test_features = result['selected_features']
"""
    },

    "run_f_statistic_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_f_statistic_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_f_statistic_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_scores: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_f_statistic_selector
import json

result_json = run_f_statistic_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
f_stat_features = result['selected_features']
"""
    },

    "run_anova_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_anova_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_anova_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], f_scores: dict, p_values: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_anova_selector
import json

result_json = run_anova_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
anova_features = result['selected_features']
"""
    },

    "run_statistical_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_statistical_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_statistical_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], statistics: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_statistical_selector
import json

result_json = run_statistical_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
stat_features = result['selected_features']
"""
    },

    "run_fcbf_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_fcbf_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_fcbf_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_scores: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_fcbf_selector
import json

result_json = run_fcbf_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
fcbf_features = result['selected_features']
"""
    },

    "run_cfs_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_cfs_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_cfs_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str]}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_cfs_selector
import json

result_json = run_cfs_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
cfs_features = result['selected_features']
"""
    },

    "run_relief_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_relief_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_relief_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "样本矩阵数据文件路径（必须包含目标列，不要传 feature_pool.csv）",
                "group_col": "目标列名，如 target 或 Group",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str], feature_scores: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_relief_selector
import json

result_json = run_relief_selector(
    data_path='data/features.csv',
    group_col='target',
    n_features=50
)
result = json.loads(result_json)
relief_features = result['selected_features']
"""
    },
    
    "run_mrmr_selector": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_mrmr_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
        "standard": {
            "signature": "run_mrmr_selector(data_path: str, group_col: str, n_features: int = 50) -> str",
            "params": {
                "data_path": "数据文件路径",
                "group_col": "目标列名",
                "n_features": "要选择的特征数量"
            },
            "returns": "JSON字符串: {method: str, selected_features: List[str]}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_mrmr_selector
import json

result_json = run_mrmr_selector(
    data_path='data/features.csv',
    group_col='Group',
    n_features=50
)
result = json.loads(result_json)
mrmr_features = result['selected_features']
"""
    },
    
    "perform_stability_selection": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "perform_stability_selection(data_path: str, group_col: str, iterations: int, methods: List[str], top_percentage: float, temp_dir: str, mandatory_features: List[str] = None) -> str",
        "standard": {
            "signature": "perform_stability_selection(data_path: str, group_col: str, iterations: int, methods: List[str], top_percentage: float, temp_dir: str, mandatory_features: List[str] = None) -> str",
            "params": {
                "data_path": "数据文件路径",
                "group_col": "目标列名",
                "iterations": "重采样迭代次数（推荐100）",
                "methods": "特征选择方法列表，如 ['run_random_forest_selector', 'run_lasso_selector', 'run_mrmr_selector']",
                "top_percentage": "每次迭代选择的特征百分比（如0.3表示30%）",
                "temp_dir": "临时日志目录路径",
                "mandatory_features": "必须保护的特征列表（prior biomarkers），默认为None"
            },
            "returns": "JSON字符串: {success: bool, iterations_completed: int, methods: List[str], temp_dir: str, mandatory_features_protected: int}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import perform_stability_selection, calculate_frequencies_from_logs
import json

# CRITICAL: 参数名必须精确匹配
# Optional: Get mandatory features (prior biomarkers) from context
mandatory_features = context_variables.get('mandatory_features', [])

result_json = perform_stability_selection(
    data_path='data/features.csv',
    group_col='Group',
    iterations=100,  # 必须是 'iterations' 不是 'n_iterations'
    methods=['run_random_forest_selector', 'run_lasso_selector', 'run_mrmr_selector'],
    top_percentage=0.30,
    temp_dir='temp_stability',  # 必须是 'temp_dir' 不是 'output_dir'
    mandatory_features=mandatory_features  # Optional: protected features
)
result = json.loads(result_json)

if result['success']:
    # 计算特征频率
    freq_json = calculate_frequencies_from_logs('temp_stability')
    frequencies = json.loads(freq_json)
"""
    },

    "run_train_only_stability_selection": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "run_train_only_stability_selection(data_path: str, group_col: str, iterations: int, methods: List[str], n_features_per_method: int, temp_dir: str, sample_fraction: float = 0.8, mandatory_features: List[str] = None, random_state: int = 42, data_already_scaled: bool = False) -> str",
        "standard": {
            "signature": "run_train_only_stability_selection(data_path: str, group_col: str, iterations: int, methods: List[str], n_features_per_method: int, temp_dir: str, sample_fraction: float = 0.8, mandatory_features: List[str] = None, random_state: int = 42, data_already_scaled: bool = False) -> str",
            "params": {
                "data_path": "Step 5.0 生成的训练池数据文件路径",
                "group_col": "目标列名",
                "iterations": "训练池内部重采样次数，例如 30",
                "methods": "固定 selector family",
                "n_features_per_method": "每个方法固定保留的 top-k 特征数量",
                "temp_dir": "稳定性日志与中间文件目录",
                "sample_fraction": "每次重采样保留的训练样本比例，默认 0.8",
                "mandatory_features": "受保护的 prior biomarker 列表",
                "random_state": "基础随机种子",
                "data_already_scaled": "如果 Step 5.0 已经执行 train-fit scaling，则设为 True，避免 selector 内部再次 StandardScaler"
            },
            "returns": "JSON字符串: {success: bool, iterations_completed: int, methods_used: List[str], n_features_per_method: int, sample_fraction: float, data_already_scaled: bool, temp_dir: str, summary_path: str}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import run_train_only_stability_selection
import json

selector_family = [
    'run_elasticnet_selector',
    'run_lasso_selector',
    'run_random_forest_selector',
    'run_lightgbm_selector',
    'run_mrmr_selector',
    'run_t_test_selector',
    'run_fdr_effect_size_selector',
]

result_json = run_train_only_stability_selection(
    data_path='output/phase1/final/selected_features_train_pool.csv',
    group_col='target',
    iterations=30,
    methods=selector_family,
    n_features_per_method=15,
    temp_dir='output/phase1/intermediate/latest/stability_logs',
    sample_fraction=0.8,
    mandatory_features=context_variables.get('mandatory_features', []),
    random_state=42,
    data_already_scaled=bool(context_variables.get('train_fit_scaling_applied', False)),
)
result = json.loads(result_json)
if not result['success']:
    raise ValueError(result['error'])
"""
    },

    "build_stable_panel_from_logs": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "build_stable_panel_from_logs(temp_dir: str, iterations: int, methods: List[str], final_panel_size: int, mandatory_features: List[str] = None, selection_frequency_threshold: float = 0.60, method_consensus_threshold: int = 3, data_path: str = '', group_col: str = '', max_pairwise_correlation: float = 0.90, n_features_per_method: int = None) -> str",
        "standard": {
            "signature": "build_stable_panel_from_logs(temp_dir: str, iterations: int, methods: List[str], final_panel_size: int, mandatory_features: List[str] = None, selection_frequency_threshold: float = 0.60, method_consensus_threshold: int = 3, data_path: str = '', group_col: str = '', max_pairwise_correlation: float = 0.90, n_features_per_method: int = None) -> str",
            "params": {
                "temp_dir": "稳定性日志目录",
                "iterations": "训练池重采样次数",
                "methods": "固定 selector family",
                "final_panel_size": "Phase 1 最终 panel 目标宽度",
                "mandatory_features": "受保护的 prior biomarker 列表",
                "selection_frequency_threshold": "stable core 的平均选择频率阈值",
                "method_consensus_threshold": "stable core 的方法共识阈值",
                "data_path": "训练池数据路径，用于 panel completion 冗余控制",
                "group_col": "目标列名",
                "max_pairwise_correlation": "panel completion 时允许的最大成对相关性",
                "n_features_per_method": "每个方法固定保留的 top-k"
            },
            "returns": "JSON字符串: {success: bool, stable_core_features: List[str], final_panel_features: List[str], final_panel_count: int, stability_scores: List[dict], panel_completion: dict}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import (
    run_train_only_stability_selection,
    build_stable_panel_from_logs,
)
import json

selector_family = [
    'run_elasticnet_selector',
    'run_lasso_selector',
    'run_random_forest_selector',
    'run_lightgbm_selector',
    'run_mrmr_selector',
    'run_t_test_selector',
    'run_fdr_effect_size_selector',
]
temp_dir = 'output/phase1/intermediate/latest/stability_logs'

run_result = json.loads(run_train_only_stability_selection(
    data_path='output/phase1/final/selected_features_train_pool.csv',
    group_col='target',
    iterations=30,
    methods=selector_family,
    n_features_per_method=15,
    temp_dir=temp_dir,
    mandatory_features=context_variables.get('mandatory_features', []),
    data_already_scaled=bool(context_variables.get('train_fit_scaling_applied', False)),
))
if not run_result['success']:
    raise ValueError(run_result['error'])

panel_result = json.loads(build_stable_panel_from_logs(
    temp_dir=temp_dir,
    iterations=30,
    methods=selector_family,
    final_panel_size=15,
    mandatory_features=context_variables.get('mandatory_features', []),
    selection_frequency_threshold=0.60,
    method_consensus_threshold=3,
    data_path='output/phase1/final/selected_features_train_pool.csv',
    group_col='target',
    max_pairwise_correlation=0.90,
    n_features_per_method=15,
))
if not panel_result['success']:
    raise ValueError(panel_result['error'])
"""
    },
    
    "calculate_frequencies_from_logs": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "calculate_frequencies_from_logs(log_dir: str) -> str",
        "standard": {
            "signature": "calculate_frequencies_from_logs(log_dir: str) -> str",
            "params": {
                "log_dir": "包含选择日志的目录路径"
            },
            "returns": "JSON字符串: {feature_name: frequency, ...}，频率范围 0.0-1.0"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import calculate_frequencies_from_logs
import json

freq_json = calculate_frequencies_from_logs('temp_stability')
frequencies = json.loads(freq_json)

# 按频率筛选特征
stable_features = [f for f, freq in frequencies.items() if freq >= 0.7]
"""
    },
    
    "get_adaptive_consensus_features": {
        "module": "src.tools.analysis.feature_selection_tools",
        "minimal": "get_adaptive_consensus_features(frequencies_json: str, min_features: int = 10) -> str",
        "standard": {
            "signature": "get_adaptive_consensus_features(frequencies_json: str, min_features: int = 10) -> str",
            "params": {
                "frequencies_json": "JSON字符串，可为单次频率数据，或 threshold_runs 结构（如 30%/35%/40% 多轮尝试）",
                "min_features": "最小特征数量要求（默认10，但当前策略实际要求最终特征数 >10）"
            },
        "returns": "JSON字符串: {success: bool, consensus_features: List[str], strategy_used: str, selection_threshold: float|None, run_mode: str, freq_threshold: int|None, vote_threshold: int, is_fallback: bool}"
        },
        "full": """
from src.tools.analysis.feature_selection_tools import (
    perform_stability_selection,
    calculate_frequencies_from_logs,
    get_adaptive_consensus_features
)
import json

# Step 1: 在 30% -> 35% -> 40% 阈值计划下准备多轮频率数据
mandatory_features = context_variables.get('mandatory_features', [])
threshold_runs = []

for p in [0.30, 0.35, 0.40]:
    result_json = perform_stability_selection(
        data_path='data/features.csv',
        group_col='Group',
        iterations=100,
        methods=[
            'run_random_forest_selector',
            'run_lightgbm_selector',
            'run_xgboost_selector',
            'run_lasso_selector',
            'run_elasticnet_selector',
            'run_mutual_info_selector',
            'run_t_test_selector',
            'run_anova_selector',
            'run_fcbf_selector',
            'run_cfs_selector',
            'run_relief_selector',
            'run_mrmr_selector'
        ],
        top_percentage=p,
        temp_dir=f'temp_stability_{int(p*100)}',
        mandatory_features=mandatory_features
    )
    freq_json = calculate_frequencies_from_logs(f'temp_stability_{int(p*100)}')
    threshold_runs.append({
        "selection_threshold": p,
        "run_mode": "imbalanced",
        "frequencies": json.loads(freq_json)
    })

# Step 2: 自适应共识特征选择（统一现役逻辑）
consensus_json = get_adaptive_consensus_features(
    frequencies_json=json.dumps({"threshold_runs": threshold_runs}),
    min_features=10
)

# Step 4: 解析结果
result = json.loads(consensus_json)
if result['success']:
    features = result['consensus_features']
    strategy = result['strategy_used']
    print(f"Selected {len(features)} features using strategy: {strategy}")
    print(f"Freq threshold: {result['freq_threshold']}, Vote threshold: {result['vote_threshold']}")
    
    if result['is_fallback']:
        print("WARNING: Using fallback strategy (union of all methods)")
else:
    print(f"Error: {result['error']}")
"""
    },
    
    # ========== Domain Tools ==========
    "map_metabolite_fast": {
        "module": "src.tools.domain.json_metabolite_mapper",
        "minimal": "map_metabolite_fast(name: str, json_path: str = 'storage/metabolite_lookup.json') -> Optional[str]",
        "standard": {
            "signature": "map_metabolite_fast(name: str, json_path: str = 'storage/metabolite_lookup.json') -> Optional[str]",
            "params": {
                "name": "代谢物名称",
                "json_path": "查找表JSON文件路径（默认值通常正确）"
            },
            "returns": "HMDB ID字符串（如'HMDB0000064'）或 None（未找到）"
        },
        "full": """
from src.tools.domain.json_metabolite_mapper import map_metabolite_fast

# 快速查找（O(1)复杂度）
hmdb_id = map_metabolite_fast("Creatine")
if hmdb_id:
    print(f"Found: {hmdb_id}")
else:
    print("Not found, keep original name")
"""
    },
    
    "ReactionCheckerTool": {
        "module": "src.tools.domain.reaction_checker",
        "minimal": "checker = ReactionCheckerTool(); checker.verify_pair(id_a: str, id_b: str) -> dict; checker.find_connected_pairs(hmdb_ids: List[str]) -> List[dict]",
        "standard": {
            "signature": "ReactionCheckerTool().verify_pair(id_a: str, id_b: str) -> dict\nReactionCheckerTool().find_connected_pairs(hmdb_ids: List[str]) -> List[dict]",
            "params": {
                "verify_pair.id_a": "第一个代谢物的HMDB ID",
                "verify_pair.id_b": "第二个代谢物的HMDB ID",
                "find_connected_pairs.hmdb_ids": "数据集中所有HMDB ID的列表"
            },
            "returns": "verify_pair: 字典 {is_connected: bool, reaction_type: str, rc_id: str, ...}\nfind_connected_pairs: 列表 [{id_a: str, id_b: str, rc_id: str, reaction_type: str}, ...]\n注意: 这是低层工具。Phase 1 的 1.5.1 主入口应优先使用 generate_reaction_ratios(...), 而不是在 SOP 代码里直接调用本工具。"
        },
        "full": """
from src.tools.domain.reaction_checker import ReactionCheckerTool

# LOW-LEVEL / LEGACY USAGE ONLY:
# In Phase 1 Step 1.5.1, prefer generate_reaction_ratios(...)

# 初始化工具
checker = ReactionCheckerTool()

# 方法1: 验证单个代谢物对（旧方法，不推荐用于批量）
result = checker.verify_pair('HMDB0000064', 'HMDB0000161')
if result['is_connected']:
    print(result['reaction_type'])

# 方法2: 批量查找所有连接对（推荐，O(N)复杂度）
hmdb_cols = [col for col in data.columns if col.startswith('HMDB')]
valid_pairs = checker.find_connected_pairs(hmdb_cols)

print(f"Found {len(valid_pairs)} biochemically connected pairs")

for pair in valid_pairs[:5]:
    print(f"  {pair['id_a']} <-> {pair['id_b']}: {pair['reaction_type']}")
"""
    },
    
    # Method-level contract for AST validation
    "ReactionCheckerTool.verify_pair": {
        "module": "src.tools.domain.reaction_checker",
        "minimal": "checker.verify_pair(id_a: str, id_b: str) -> dict",
        "standard": {
            "signature": "verify_pair(id_a: str, id_b: str) -> dict",
            "params": {
                "id_a": "第一个代谢物的HMDB ID",
                "id_b": "第二个代谢物的HMDB ID"
            },
            "returns": "字典: {is_connected: bool, reaction_type: str, rc_id: str, ...}"
        }
    },
    
    # NEW: Method-level contract for find_connected_pairs
    "ReactionCheckerTool.find_connected_pairs": {
        "module": "src.tools.domain.reaction_checker",
        "minimal": "checker.find_connected_pairs(hmdb_ids: List[str]) -> List[dict]",
        "standard": {
            "signature": "find_connected_pairs(hmdb_ids: List[str]) -> List[dict]",
            "params": {
                "hmdb_ids": "数据集中所有HMDB ID的列表"
            },
            "returns": "列表: [{id_a: str, id_b: str, rc_id: str, reaction_type: str}, ...]，包含所有生化连接对"
        }
    },

    "generate_reaction_ratios": {
        "module": "src.tools.domain.feature_generator",
        "minimal": "generate_reaction_ratios(data_path: str, output_path: str, registry_path: str = None, rules_path: str = None, include_label: bool = False) -> dict",
        "standard": {
            "signature": "generate_reaction_ratios(data_path: str, output_path: str, registry_path: str = None, rules_path: str = None, include_label: bool = False) -> dict",
            "params": {
                "data_path": "输入样本矩阵路径。对于 Phase 1，应优先传 pre_engineering_data_path",
                "output_path": "ratio 临时特征文件输出路径",
                "registry_path": "工程特征注册表输出路径（可选，默认从配置读取）",
                "rules_path": "工程特征规则 JSON 路径（可选，默认从配置读取）",
                "include_label": "是否在输出文件中保留目标列。生成临时特征文件时应设为 False"
            },
            "returns": "字典: {status: str, output_path: str, num_ratios: int, ratio_names: List[str], registry_path: str, n_connected_pairs: int, n_skipped_pairs: int, message: str}. 注意: 这是普通函数，必须直接调用 generate_reaction_ratios(...), 不要写 generate_reaction_ratios.get(...)."
        },
        "full": """
import os
from src.tools.domain.feature_generator import generate_reaction_ratios

# CORRECT: call the function directly, then inspect the returned dict.
result = generate_reaction_ratios(
    data_path=pre_engineering_data_path if pre_engineering_data_path else current_data_path,
    output_path=os.path.join(phase1_intermediate_latest, 'engineered', 'temp_ratios.csv'),
    include_label=False,
)

# WRONG: generate_reaction_ratios.get(...)
if result['status'] != 'success':
    raise ValueError(result['message'])

print(result['num_ratios'])
print(result['registry_path'])
"""
    },

    "generate_taxonomy_sums": {
        "module": "src.tools.domain.feature_generator",
        "minimal": "generate_taxonomy_sums(data_path: str, output_path: str, registry_path: str = None, rules_path: str = None, include_label: bool = False) -> dict",
        "standard": {
            "signature": "generate_taxonomy_sums(data_path: str, output_path: str, registry_path: str = None, rules_path: str = None, include_label: bool = False) -> dict",
            "params": {
                "data_path": "输入样本矩阵路径。对于 Phase 1，应优先传 pre_engineering_data_path",
                "output_path": "sum 临时特征文件输出路径",
                "registry_path": "工程特征注册表输出路径（可选，默认从配置读取）",
                "rules_path": "工程特征规则 JSON 路径（可选，默认从配置读取）",
                "include_label": "是否在输出文件中保留目标列。生成临时特征文件时应设为 False"
            },
            "returns": "字典: {status: str, output_path: str, num_sums: int, sum_names: List[str], registry_path: str, n_group_candidates: int, n_skipped_groups: int, message: str}. 注意: 这是普通函数，必须直接调用 generate_taxonomy_sums(...), 不要写 generate_taxonomy_sums.get(...)."
        },
        "full": """
import os
from src.tools.domain.feature_generator import generate_taxonomy_sums

# CORRECT: call the function directly, then inspect the returned dict.
result = generate_taxonomy_sums(
    data_path=pre_engineering_data_path if pre_engineering_data_path else current_data_path,
    output_path=os.path.join(phase1_intermediate_latest, 'engineered', 'temp_sums.csv'),
    include_label=False,
)

# WRONG: generate_taxonomy_sums.get(...)
if result['status'] != 'success':
    raise ValueError(result['message'])

print(result['num_sums'])
print(result['registry_path'])
"""
    },

    "generate_pathway_scores": {
        "module": "src.tools.domain.feature_generator",
        "minimal": "generate_pathway_scores(data_path: str, output_path: str, pathway_names: List[str] = None, phase0_output: dict = None, registry_path: str = None, rules_path: str = None, include_label: bool = False) -> dict",
        "standard": {
            "signature": "generate_pathway_scores(data_path: str, output_path: str, pathway_names: List[str] = None, phase0_output: dict = None, registry_path: str = None, rules_path: str = None, include_label: bool = False) -> dict",
            "params": {
                "data_path": "输入样本矩阵路径。对于 Phase 1，应优先传 pre_engineering_data_path",
                "output_path": "pathway 临时特征文件输出路径",
                "pathway_names": "显式通路名称列表（可选）。未提供时默认从 phase0_output.feature_definitions.target_pathways 获取",
                "phase0_output": "Phase 0 输出字典（可选）",
                "registry_path": "工程特征注册表输出路径（可选，默认从配置读取）",
                "rules_path": "工程特征规则 JSON 路径（可选，默认从配置读取）",
                "include_label": "是否在输出文件中保留目标列。生成临时特征文件时应设为 False"
            },
            "returns": "字典: {status: str, output_path: str, num_pathways: int, pathway_names: List[str], registry_path: str, n_candidates: int, n_skipped_pathways: int, message: str}. 注意: 这是普通函数，必须直接调用 generate_pathway_scores(...); 返回值是 dict，不要再 json.loads(...)。若没有 pathway candidates 或没有 pathway 通过 coverage gate，也应允许返回 status='success' 且 num_pathways=0 的空文件结果。"
        },
        "full": """
import os
from src.tools.domain.feature_generator import generate_pathway_scores

# CORRECT: call the function directly, then inspect the returned dict.
result = generate_pathway_scores(
    data_path=pre_engineering_data_path if pre_engineering_data_path else current_data_path,
    output_path=os.path.join(phase1_intermediate_latest, 'engineered', 'temp_pathway_scores.csv'),
    phase0_output=context_variables.get('phase0_output'),
    include_label=False,
)

# WRONG: generate_pathway_scores.get(...)
if result['status'] != 'success':
    raise ValueError(result['message'])

print(result['num_pathways'])
print(result['registry_path'])
"""
    },

    "merge_engineered_features": {
        "module": "src.tools.domain.feature_generator",
        "minimal": "merge_engineered_features(base_data_path: str, ratio_path: str = 'data/ratio_features_data.csv', taxonomy_path: str = 'data/taxonomy_sum_features_data.csv', pathway_path: str = 'data/pathway_scores_data.csv', output_path: str = 'data/enriched_data.csv', metadata_path: str = 'data/feature_metadata.json') -> dict",
        "standard": {
            "signature": "merge_engineered_features(base_data_path: str, ratio_path: str = 'data/ratio_features_data.csv', taxonomy_path: str = 'data/taxonomy_sum_features_data.csv', pathway_path: str = 'data/pathway_scores_data.csv', output_path: str = 'data/enriched_data.csv', metadata_path: str = 'data/feature_metadata.json') -> dict",
            "params": {
                "base_data_path": "基础样本矩阵路径。对于 Phase 1，应优先传 pre_engineering_data_path",
                "ratio_path": "ratio 临时特征文件路径",
                "taxonomy_path": "sum 临时特征文件路径",
                "pathway_path": "pathway 临时特征文件路径",
                "output_path": "合并后的完整数据集输出路径",
                "metadata_path": "特征元数据 JSON 输出路径"
            },
            "returns": "字典: {status: str, output_path: str, metadata_path: str, num_features: int, feature_breakdown: {original: int, ratio: int, taxonomy_sum: int, pathway_score: int}, message: str}. 注意: 这是普通函数，必须直接调用 merge_engineered_features(...), 并从 feature_breakdown 读取真实计数，不要用 CSV shape[1] 手工猜。"
        },
        "full": """
import os
from src.tools.domain.feature_generator import merge_engineered_features

base_path = pre_engineering_data_path if pre_engineering_data_path else current_data_path
engineered_dir = os.path.join(phase1_intermediate_latest, 'engineered')

result = merge_engineered_features(
    base_data_path=base_path,
    ratio_path=os.path.join(engineered_dir, 'temp_ratios.csv'),
    taxonomy_path=os.path.join(engineered_dir, 'temp_sums.csv'),
    pathway_path=os.path.join(engineered_dir, 'temp_pathway_scores.csv'),
    output_path=os.path.join(engineered_dir, 'data_with_engineered_features.csv'),
    metadata_path=os.path.join(engineered_dir, 'feature_metadata.json'),
)

if result['status'] != 'success':
    raise ValueError(result['message'])

breakdown = result['feature_breakdown']
print(result['output_path'])
print(breakdown['ratio'], breakdown['taxonomy_sum'], breakdown['pathway_score'])
"""
    },
    
    "TaxonomyGroupTool": {
        "module": "src.tools.domain.taxonomy_group_tool",
        "minimal": "tool = TaxonomyGroupTool(); tool.group_metabolites(metabolite_ids: List[str], level: str = 'sub_class') -> dict",
        "standard": {
            "signature": "TaxonomyGroupTool().group_metabolites(metabolite_ids: List[str], level: str = 'sub_class') -> dict",
            "params": {
                "metabolite_ids": "HMDB ID列表",
                "level": "分类级别: 'class' | 'sub_class'"
            },
            "returns": "字典: {group_name: [hmdb_id1, hmdb_id2, ...], ...}\n注意: 这是低层工具。Phase 1 的 1.5.2 主入口应优先使用 generate_taxonomy_sums(...), 而不是在 SOP 代码里手工对所有 taxonomy group 求和。"
        },
        "full": """
from src.tools.domain.taxonomy_group_tool import TaxonomyGroupTool
import pandas as pd

# LOW-LEVEL / LEGACY USAGE ONLY:
# In Phase 1 Step 1.5.2, prefer generate_taxonomy_sums(...)

# 初始化工具
tool = TaxonomyGroupTool()

# 提取HMDB ID列
hmdb_ids = [col for col in data.columns if col.startswith('HMDB')]

# 按分类学分组
groups = tool.group_metabolites(metabolite_ids=hmdb_ids, level='sub_class')

# 这里只演示如何查看分组，不建议在 SOP 代码里盲目对全部分组直接求和
for group_name, hmdb_list in list(groups.items())[:5]:
    print(group_name, len(hmdb_list))
"""
    },
    
    # Method-level contract for AST validation
    "TaxonomyGroupTool.group_metabolites": {
        "module": "src.tools.domain.taxonomy_group_tool",
        "minimal": "tool.group_metabolites(metabolite_ids: List[str], level: str = 'sub_class') -> dict",
        "standard": {
            "signature": "group_metabolites(metabolite_ids: List[str], level: str = 'sub_class') -> dict",
            "params": {
                "metabolite_ids": "HMDB ID列表",
                "level": "分类级别: 'class' | 'sub_class'"
            },
            "returns": "字典: {group_name: [hmdb_id1, hmdb_id2, ...], ...}"
        }
    },
    
    "calculate_pathway_scores": {
        "module": "src.tools.domain.pathway_scorer",
        "minimal": "calculate_pathway_scores(data_matrix_path: str, pathway_names: List[str], output_path: str = None, include_group: bool = True) -> str",
        "standard": {
            "signature": "calculate_pathway_scores(data_matrix_path: str, pathway_names: List[str], output_path: str = None, include_group: bool = True) -> str",
            "params": {
                "data_matrix_path": "数据矩阵CSV路径（样本×代谢物，自动过滤元数据列）",
                "pathway_names": "通路名称列表",
                "output_path": "输出文件路径（可选）",
                "include_group": "是否在输出中包含Group列（默认True）。生成临时特征文件时设为False以避免合并冲突"
            },
            "returns": "JSON字符串: {success: bool, output_path: str, n_samples: int, n_pathways: int, pathways: List[str], summary: str}\n注意: 这是低层打分函数。Phase 1 的 1.5.3 主入口应优先使用 generate_pathway_scores(...), 由其统一处理 coverage gate 和 Phase 0 target_pathways。"
        },
        "full": """
from src.tools.domain.pathway_scorer import calculate_pathway_scores
import json

# LOW-LEVEL / LEGACY USAGE ONLY:
# In Phase 1 Step 1.5.3, prefer generate_pathway_scores(...)

# 计算通路活性评分（仅适用于低层直接调用场景）
result_json = calculate_pathway_scores(
    data_matrix_path='data/cleaned.csv',
    pathway_names=['Urea Cycle', 'Arginine Metabolism'],
    output_path='data/temp_pathways.csv',
    include_group=False  # CRITICAL: 临时特征文件不包含Group列
)

# 解析JSON返回值
result = json.loads(result_json)

if result['success']:
    print(f"Calculated scores for {result['n_pathways']} pathways")
    print(f"Output saved to: {result['output_path']}")
    print(result['summary'])  # 打印摘要
    
    # 更新状态
    print(json.dumps({
        '__METABO_UPDATE__': {
            'temp_pathways_path': result['output_path']
        }
    }))
else:
    print(f"Error: {result.get('error')}")
"""
    },
    
    # ========== Model Building Tools ==========
    "analyze_data_for_modeling": {
        "module": "src.tools.analysis.model_building_tools",
        "minimal": "analyze_data_for_modeling(data_path: str, target_column: str) -> str",
        "standard": {
            "signature": "analyze_data_for_modeling(data_path: str, target_column: str) -> str",
            "params": {
                "data_path": "数据文件路径",
                "target_column": "目标列名"
            },
            "returns": "JSON字符串: {n_samples: int, n_features: int, class_distribution: dict, imbalance_ratio: float, ...}"
        },
        "full": """
from src.tools.analysis.model_building_tools import analyze_data_for_modeling
import json

result_json = analyze_data_for_modeling(
    data_path='data/selected_features.csv',
    target_column='Group'
)
result = json.loads(result_json)

imbalance_ratio = result['imbalance_ratio']
"""
    },
    
    "train_with_autogluon": {
        "module": "src.tools.analysis.model_building_tools",
        "minimal": "train_with_autogluon(data_path: str, target_column: str, balancing_method: str = 'none', time_limit: int = 600, eval_metric: str = 'roc_auc', presets: str = 'best_quality', test_data_path: str = None) -> str",
        "standard": {
            "signature": "train_with_autogluon(data_path: str, target_column: str, balancing_method: str = 'none', time_limit: int = 600, eval_metric: str = 'roc_auc', presets: str = 'best_quality', test_data_path: str = None) -> str",
            "params": {
                "data_path": "数据文件路径",
                "target_column": "目标列名",
                "balancing_method": "'none' | 'smote' | 'smote_nearmiss'",
                "time_limit": "训练时间限制（秒）",
                "eval_metric": "'accuracy' | 'roc_auc' | 'f1'",
                "presets": "'best_quality' | 'high_quality' | 'medium_quality'",
                "test_data_path": "可选的外部 holdout 测试集路径；如果提供，则不会再对 data_path 做内部切分"
            },
            "returns": "JSON字符串: {success: bool, best_model: str, primary_metric: str, test_set_metrics: {...}, split_strategy: str, ...}"
        },
        "full": """
from src.tools.analysis.model_building_tools import train_with_autogluon
import json

result_json = train_with_autogluon(
    data_path='data/selected_features.csv',
    target_column='Group',
    balancing_method='smote_nearmiss',  # 如果不平衡
    time_limit=180,
    eval_metric='accuracy',
    presets='best_quality',
    test_data_path='data/selected_features_holdout.csv'
)
result = json.loads(result_json)

if result.get('success'):
    print(f"Best model: {result['best_model']}")
    print(f"ROC-AUC: {result['test_set_metrics']['roc_auc']}")
"""
    }
}


def get_tool_contract(tool_name: str, level: str = "standard") -> str:
    """
    Get tool contract at specified detail level.
    
    Args:
        tool_name: Name of the tool
        level: Detail level - "minimal" | "standard" | "full"
    
    Returns:
        Tool contract string
    """
    if tool_name not in TOOL_CONTRACTS:
        return f"# Tool '{tool_name}' not found in registry"
    
    contract = TOOL_CONTRACTS[tool_name]
    
    if level == "minimal":
        return f"from {contract['module']} import {tool_name}\n{contract['minimal']}"
    
    elif level == "standard":
        std = contract["standard"]
        params_str = "\n".join([f"  - {k}: {v}" for k, v in std["params"].items()])
        return f"""
**{tool_name}**
```python
from {contract['module']} import {tool_name}
{std['signature']}
```
参数:
{params_str}

返回: {std['returns']}
"""
    
    elif level == "full":
        return f"""
**{tool_name}** (完整示例)
```python
{contract['full']}
```
"""
    
    else:
        return f"# Invalid level: {level}"


def get_tools_documentation(
    tools_required: List[str],
    level: str = "standard",
    lib_hint: str = ""
) -> str:
    """
    Get documentation for multiple tools at specified detail level.
    
    This is the main function used by the prompt builder.
    
    Args:
        tools_required: List of tool names needed for current step
        level: Detail level - "minimal" | "standard" | "full"
        lib_hint: Library hint from SOP (for fallback)
    
    Returns:
        Concatenated documentation string
    """
    if not tools_required:
        return "\n**Available Tools:** (No specific tools required for this step)\n"
    
    docs = []
    for tool_name in tools_required:
        doc = get_tool_contract(tool_name, level)
        docs.append(doc)
    
    header = f"\n**Available Tools** (Detail Level: {level.upper()}):\n"
    return header + "\n".join(docs)


def get_all_tool_names() -> List[str]:
    """Get list of all registered tool names."""
    return list(TOOL_CONTRACTS.keys())


def get_tool_module(tool_name: str) -> str:
    """Get the module path for a tool."""
    if tool_name in TOOL_CONTRACTS:
        return TOOL_CONTRACTS[tool_name]["module"]
    return ""


def get_tool_signature(tool_name: str) -> str:
    """Get the minimal signature for a tool."""
    if tool_name in TOOL_CONTRACTS:
        return TOOL_CONTRACTS[tool_name]["minimal"]
    return ""
