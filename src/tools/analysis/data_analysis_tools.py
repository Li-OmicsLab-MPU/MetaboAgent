# src/tools/analysis/data_analysis_tools.py
"""
Data Analysis Tools for MetaboAgent

This module contains core data analysis and cleaning tools for metabolomics data processing.
All tools follow a privacy-preserving design pattern to prevent data leakage to LLMs.

Main Features:
1. Data Quality Analysis: analyze_dataset_quality - comprehensive dataset quality assessment
2. Code Knowledge Base: search_cleaning_code - provides data cleaning code examples
3. Data Cleaning Operations: delete_columns, impute_missing, cap_outliers, delete_rows
4. Normalization Tools: PQN normalization, log2 transformation, auto scaling
5. Statistical Power Assessment: check_power_impact

Privacy-Preserving Design:
- All tools accept file paths instead of DataFrame objects
- Data is read locally, processed locally, and saved locally
- Only metadata and statistics are returned (no raw data)
- No sensitive data is exposed to external LLM APIs

主要功能：
1. 数据质量分析：analyze_dataset_quality - 全面分析数据集的质量问题
2. 代码知识库搜索：search_cleaning_code - 提供数据清洗相关的代码示例
3. 数据清洗操作：删除列、填充缺失值、截断异常值、删除行
4. 归一化工具：PQN归一化、log2变换、自动缩放
5. 统计功效评估：check_power_impact

这些工具被设计为LangChain工具，可以在LangGraph中直接调用，
支持复杂的代谢组学数据清洗工作流。

主要特点：
- 支持CSV和Excel文件格式
- 提供详细的统计分析和异常值检测
- 包含丰富的代码示例知识库
- 支持多种清洗策略（缺失值填充、异常值处理等）
- 完整的错误处理和日志记录
- 隐私保护设计，防止数据泄露到LLM
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

from scipy import stats


TECHNICAL_PROTECTED_COLUMNS = {
    "Group",
    "group",
    "target",
    "Factors",
    "factors",
    "FACTOR",
    "factor",
    "ID",
    "id",
    "Sample_ID",
    "sample_id",
    "SampleID",
    "sampleid",
    "ROW_ID",
    "row_id",
    "__row_id__",
}


def _read_table_file(file_path: str) -> pd.DataFrame:
    """Read a CSV/Excel table from disk."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    if file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    raise ValueError("不支持的文件格式，请使用CSV或Excel文件")


def _load_optional_payload(file_path: Optional[str]) -> Any:
    """Load a lightweight JSON/CSV/Excel payload if it exists."""
    if not file_path:
        return None

    path = Path(file_path)
    if not path.exists():
        return None

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path).to_dict(orient="records")

    return path.read_text(encoding="utf-8", errors="ignore")


def _summarize_payload_text(payload: Any) -> str:
    """Convert structured metadata into compact lowercase text for heuristics."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.lower()
    try:
        return json.dumps(payload, ensure_ascii=False).lower()
    except Exception:
        return str(payload).lower()


def _normalize_column_key(value: Any) -> str:
    return str(value).strip().lower()


_TECHNICAL_PROTECTED_KEYS = {_normalize_column_key(item) for item in TECHNICAL_PROTECTED_COLUMNS}


def _is_technical_protected_column(col_name: Any) -> bool:
    return _normalize_column_key(col_name) in _TECHNICAL_PROTECTED_KEYS


def _extract_protected_columns(column_roles_payload: Any) -> List[str]:
    """Extract protected columns from a roles payload without exposing data values."""
    protected: List[str] = []

    if isinstance(column_roles_payload, dict):
        for key in ("protected_columns", "protected_cols"):
            value = column_roles_payload.get(key)
            if isinstance(value, list):
                protected.extend([str(item) for item in value if item])

        for key in ("target_column", "sample_id_column", "sample_column", "group_column"):
            value = column_roles_payload.get(key)
            if isinstance(value, str) and value.strip():
                protected.append(value.strip())

    protected.extend(list(TECHNICAL_PROTECTED_COLUMNS))
    return list(dict.fromkeys(protected))


def _safe_float(value: Any, digits: int = 6) -> Optional[float]:
    """Round finite numeric values and normalize everything else to None."""
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except Exception:
        return None

    if not np.isfinite(numeric_value):
        return None
    return round(numeric_value, digits)


def _infer_platform_source(
    dataset_path: str,
    source_hint: Optional[str],
    metadata_text: str,
) -> str:
    """Infer the most likely public source from hints and metadata."""
    combined = " ".join([
        str(dataset_path or "").lower(),
        str(source_hint or "").lower(),
        metadata_text,
    ])

    if "uk biobank" in combined or "ukb" in combined:
        return "UK Biobank"
    if "metabolights" in combined or "mtbls" in combined:
        return "MetaboLights"
    if "metabolomics workbench" in combined or "workbench" in combined:
        return "Metabolomics Workbench"
    return "unknown"


def _infer_data_level(
    user_declared_data_level: Optional[str],
    metadata_text: str,
    has_pooled_qc: bool,
    has_run_order: bool,
) -> str:
    """Infer whether the dataset behaves like a processed matrix or QC-aware matrix."""
    declared = str(user_declared_data_level or "").strip()
    if declared in {"processed_matrix", "raw_like_matrix_with_qc_metadata"}:
        return declared

    if has_pooled_qc or has_run_order:
        return "raw_like_matrix_with_qc_metadata"

    processed_markers = (
        "normalized abundance",
        "processed matrix",
        "concentration matrix",
        "quantified metabolite",
        "release data",
        "uk biobank",
    )
    if any(marker in metadata_text for marker in processed_markers):
        return "processed_matrix"

    return "unknown"

def get_demo_data_path() -> str:
    """获取demo数据文件路径"""
    # 优先使用环境变量
    demo_file = os.getenv("DEMO_DEPRESSION_FILE")
    if demo_file and Path(demo_file).exists():
        return demo_file
    
    return str(Path("data") / "demo.csv")


def analyze_dataset_quality(file_path: str = "") -> str:
    """
    数据质量分析工具 - 全面评估数据集的质量状况
    
    这是数据清洗工作流的第一步，对输入数据集进行全面的质量分析。
    分析结果将用于生成针对性的数据清洗计划。
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally from the file system
    - Processes data locally without sending raw data to external APIs
    - Returns only metadata and statistics (no raw data values)
    
    分析内容：
    1. 缺失值分析：计算每列的缺失值数量和百分比
    2. 分布统计：对数值列计算均值、中位数、标准差、偏度、峰度等
    3. 异常值检测：使用IQR方法识别潜在的异常值
    4. 数据类型分析：识别数值型和分类型列
    5. 整体质量概览：提供数据集的整体质量评估
    
    Args:
        file_path (str): 数据文件的路径，支持CSV和Excel格式
        
    Returns:
        str: JSON格式的详细质量报告，包含以下结构：
            {
                "file_path": "文件路径",
                "data_shape": {"rows": 行数, "columns": 列数},
                "missing_values": {
                    "列名": {
                        "missing_count": 缺失值数量,
                        "missing_percentage": 缺失值百分比
                    }
                },
                "numeric_columns_stats": {
                    "列名": {
                        "mean": 均值, "median": 中位数, "std": 标准差,
                        "skewness": 偏度, "kurtosis": 峰度, "iqr": 四分位距
                    }
                },
                "potential_outlier_columns": [
                    {
                        "column": "列名",
                        "outlier_count": 异常值数量,
                        "outlier_percentage": 异常值百分比
                    }
                ],
                "group_sample_sizes": {
                    "类别名": 样本数量
                },
                "mnar_likelihood_analysis": {
                    "列名": true
                },
                "data_overview": {
                    "total_missing_values": 总缺失值数,
                    "missing_percentage_overall": 整体缺失值百分比
                }
            }
    
    Raises:
        Exception: 当文件读取失败或格式不支持时抛出异常
    """
    try:
        # 如果没有提供文件路径，使用demo数据
        if not file_path or file_path == "":
            file_path = get_demo_data_path()
        
        # 读取数据文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return json.dumps({"error": "不支持的文件格式，请使用CSV或Excel文件"})
        
        # 初始化质量报告
        quality_report = {
            "file_path": file_path,
            "data_shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "missing_values": {},
            "numeric_columns_stats": {},
            "potential_outlier_columns": [],
            "column_types": {},
            "data_overview": {},
            "group_sample_sizes": {},
            "mnar_likelihood_analysis": {}
        }
        
        # 分析每列的缺失值
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        for col in df.columns:
            quality_report["missing_values"][col] = {
                "missing_count": int(df[col].isnull().sum()),
                "missing_percentage": round(missing_percentages[col], 2)
            }
        
        # 分析数值列的分布统计
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].notna().sum() > 0:  # 确保有非空值
                col_data = df[col].dropna()
                quality_report["numeric_columns_stats"][col] = {
                    "mean": round(col_data.mean(), 4),
                    "median": round(col_data.median(), 4),
                    "std": round(col_data.std(), 4),
                    "min": round(col_data.min(), 4),
                    "max": round(col_data.max(), 4),
                    "skewness": round(stats.skew(col_data), 4),
                    "kurtosis": round(stats.kurtosis(col_data), 4),
                    "iqr": round(col_data.quantile(0.75) - col_data.quantile(0.25), 4)
                }
                
                # 检测潜在异常值（使用IQR方法）
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_percentage = (len(outliers) / len(col_data)) * 100
                
                if outlier_percentage > 5:  # 如果异常值超过5%
                    quality_report["potential_outlier_columns"].append({
                        "column": col,
                        "outlier_count": len(outliers),
                        "outlier_percentage": round(outlier_percentage, 2),
                        "outlier_bounds": {
                            "lower": round(lower_bound, 4),
                            "upper": round(upper_bound, 4)
                        }
                    })
        
        # 记录列类型
        for col in df.columns:
            quality_report["column_types"][col] = str(df[col].dtype)
        
        # 分析group列（默认为第二列）的样本数量
        if len(df.columns) >= 2:
            group_col = df.columns[1]  # 第二列作为group列
            if group_col in df.columns:
                group_counts = df[group_col].value_counts().to_dict()
                quality_report["group_sample_sizes"] = {str(k): int(v) for k, v in group_counts.items()}
        
        # 分析MNAR（Missing Not At Random）可能性
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:  # 只分析有缺失值的列
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    q25 = col_data.quantile(0.25)
                    mean_val = col_data.mean()
                    threshold = mean_val * 0.1  # 均值的10%
                    
                    # 如果25%分位数低于均值的10%，认为可能有MNAR倾向
                    if q25 < threshold:
                        quality_report["mnar_likelihood_analysis"][col] = True
        
        # 数据概览
        quality_report["data_overview"] = {
            "total_missing_values": int(df.isnull().sum().sum()),
            "total_cells": len(df) * len(df.columns),
            "missing_percentage_overall": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            "numeric_columns_count": len(numeric_columns),
            "categorical_columns_count": len(df.columns) - len(numeric_columns)
        }
        
        return json.dumps(quality_report, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"分析数据质量时发生错误: {str(e)}"})


def data_context_resolver_tool(
    dataset_path: str,
    study_metadata_path: str = "",
    column_roles_path: str = "",
    source_hint: str = "",
    user_declared_data_level: str = "",
) -> str:
    """
    Resolve the Phase 1 preprocessing context without exposing raw matrix values.

    This tool inspects the dataset schema plus lightweight metadata to determine:
    - likely public source (Workbench / MetaboLights / UK Biobank / unknown)
    - whether the dataset behaves like a processed matrix or a QC-aware matrix
    - whether pooled QC, blanks, run order, or batch metadata appear to be present
    - protected columns that should be excluded from preprocessing transforms

    Args:
        dataset_path: 数据文件路径（CSV或Excel）
        study_metadata_path: 研究元数据路径（可选，JSON/CSV/Excel）
        column_roles_path: 列角色定义路径（可选，JSON）
        source_hint: 用户或上游提供的数据源提示
        user_declared_data_level: 用户声明的数据层级，支持 processed_matrix / raw_like_matrix_with_qc_metadata

    Returns:
        JSON字符串，包含 preprocessing_context.json 所需的核心字段
    """
    try:
        df = _read_table_file(dataset_path)
        metadata_payload = _load_optional_payload(study_metadata_path)
        column_roles_payload = _load_optional_payload(column_roles_path)
        metadata_text = _summarize_payload_text(metadata_payload)

        protected_columns = _extract_protected_columns(column_roles_payload)
        dataset_columns = [str(col) for col in df.columns]
        dataset_columns_lower = [col.lower() for col in dataset_columns]

        has_pooled_qc = any(
            token in metadata_text for token in ("pooled qc", "quality control", "qc sample")
        ) or any("qc" in col for col in dataset_columns_lower)
        has_blank_samples = any(
            token in metadata_text for token in ("blank sample", "solvent blank", "process blank")
        ) or any("blank" in col for col in dataset_columns_lower)
        has_run_order = any(
            token in metadata_text for token in ("run order", "injection order", "sequence order")
        ) or any(
            token in dataset_columns_lower
            for token in ("run_order", "run order", "injection_order", "injection order", "sequence")
        )
        has_batch_metadata = any(
            token in metadata_text for token in ("batch", "plate", "instrument", "center", "site")
        ) or any(
            any(token in col for token in ("batch", "plate", "instrument", "center", "site"))
            for col in dataset_columns_lower
        )

        platform_source = _infer_platform_source(dataset_path, source_hint, metadata_text)
        data_level = _infer_data_level(
            user_declared_data_level=user_declared_data_level,
            metadata_text=metadata_text,
            has_pooled_qc=has_pooled_qc,
            has_run_order=has_run_order,
        )

        if data_level == "raw_like_matrix_with_qc_metadata" or (
            has_pooled_qc and (has_run_order or has_batch_metadata)
        ):
            candidate_branch = "qc_aware_branch"
        elif data_level == "processed_matrix" or platform_source != "unknown":
            candidate_branch = "public_matrix_branch"
        else:
            candidate_branch = "unknown"

        target_column = None
        if isinstance(column_roles_payload, dict):
            target_column = column_roles_payload.get("target_column") or column_roles_payload.get("group_column")
        if not target_column:
            for preferred in ("Group", "group", "target"):
                if preferred in dataset_columns:
                    target_column = preferred
                    break

        numeric_feature_count = int(
            len([
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in protected_columns
            ])
        )

        warnings: List[str] = []
        if platform_source == "unknown":
            warnings.append("Unable to infer a known public source from the available hints.")
        if data_level == "unknown":
            warnings.append("Dataset level is unknown; branch selection may require LLM confirmation.")
        if not has_batch_metadata:
            warnings.append("No obvious batch metadata detected.")

        report = {
            "success": True,
            "dataset_path": dataset_path,
            "omics_type": "targeted_metabolomics",
            "platform_source": platform_source,
            "data_level": data_level,
            "has_pooled_qc": has_pooled_qc,
            "has_blank_samples": has_blank_samples,
            "has_run_order": has_run_order,
            "has_batch_metadata": has_batch_metadata,
            "candidate_branch": candidate_branch,
            "sample_count": int(len(df)),
            "numeric_feature_count": numeric_feature_count,
            "column_count": int(len(df.columns)),
            "target_column": target_column,
            "protected_columns": protected_columns,
            "available_columns": dataset_columns,
            "metadata_sources": {
                "study_metadata_path": study_metadata_path or None,
                "column_roles_path": column_roles_path or None,
                "source_hint": source_hint or None,
            },
            "warnings": warnings,
        }
        return json.dumps(report, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def zero_semantics_analyzer_tool(
    dataset_path: str,
    context_report_path: str,
    study_metadata_path: str = "",
    protected_columns: Optional[List[str]] = None,
) -> str:
    """
    Analyze whether zero values behave like true zeros or non-detect-like encoded missing values.

    The tool reads the dataset locally, excludes protected columns, and returns only summary
    statistics and heuristic evidence. It never returns raw matrix values.

    Args:
        dataset_path: 数据文件路径（CSV或Excel）
        context_report_path: preprocessing_context.json 路径
        study_metadata_path: 研究元数据路径（可选）
        protected_columns: 额外保护列名（可选）

    Returns:
        JSON字符串，包含 zero_pattern_report.json 所需的核心字段
    """
    try:
        df = _read_table_file(dataset_path)
        context_report = _load_optional_payload(context_report_path) or {}
        metadata_payload = _load_optional_payload(study_metadata_path)
        metadata_text = _summarize_payload_text(metadata_payload)

        combined_protected = list(dict.fromkeys(
            (protected_columns or [])
            + list(context_report.get("protected_columns", []))
            + list(TECHNICAL_PROTECTED_COLUMNS)
        ))

        numeric_columns = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in combined_protected
        ]
        if not numeric_columns:
            return json.dumps({
                "success": False,
                "error": "未找到可用于零值语义分析的数值特征列"
            }, ensure_ascii=False)

        zero_feature_summaries: List[Dict[str, Any]] = []
        total_zero_count = 0
        positive_continuous_zero_features = 0
        binary_like_zero_features = 0
        evidence_flags: List[str] = []

        total_numeric_cells = int(df[numeric_columns].shape[0] * len(numeric_columns))

        for col in numeric_columns:
            series = df[col]
            zero_mask = series == 0
            zero_count = int(zero_mask.sum())
            if zero_count == 0:
                continue

            total_zero_count += zero_count
            nonzero = series[~zero_mask].dropna()
            nonzero_min = _safe_float(nonzero.min()) if len(nonzero) else None
            nonzero_median = _safe_float(nonzero.median()) if len(nonzero) else None
            zero_rate = round(zero_count / max(len(series), 1), 6)

            if len(nonzero) > 0:
                all_positive_nonzero = bool((nonzero > 0).all())
                unique_nonzero = nonzero.nunique(dropna=True)
                is_integer_like = bool(np.allclose(nonzero, np.round(nonzero), equal_nan=True))
                if all_positive_nonzero and (unique_nonzero >= 5 or not is_integer_like):
                    positive_continuous_zero_features += 1
                if is_integer_like and set(pd.unique(series.dropna())).issubset({0, 1}):
                    binary_like_zero_features += 1

            zero_feature_summaries.append({
                "name": col,
                "zero_count": zero_count,
                "zero_rate": zero_rate,
                "nonzero_min": nonzero_min,
                "nonzero_median": nonzero_median,
            })

        zero_feature_summaries.sort(key=lambda item: (-item["zero_rate"], item["name"]))
        zero_rate_global = round(total_zero_count / max(total_numeric_cells, 1), 6)
        features_with_zero_count = len(zero_feature_summaries)

        if total_zero_count == 0:
            zero_semantics_status = "unknown"
            suggested_zero_handling = "keep_zero"
            evidence_flags.append("No zero values detected in numeric feature columns.")
        else:
            non_detect_keywords = (
                "lod",
                "loq",
                "below detection",
                "below limit",
                "non-detect",
                "undetected",
                "not detected",
            )
            metadata_supports_non_detect = any(keyword in metadata_text for keyword in non_detect_keywords)

            if zero_rate_global >= 0.02:
                evidence_flags.append("Global zero rate exceeds 2% across numeric features.")
            if positive_continuous_zero_features >= max(3, features_with_zero_count // 2):
                evidence_flags.append("Zeros are concentrated in positive-valued continuous features.")
            if metadata_supports_non_detect:
                evidence_flags.append("Study metadata contains LOD/LOQ or non-detect wording.")
            if binary_like_zero_features >= max(3, features_with_zero_count // 2):
                evidence_flags.append("A substantial subset of zero-containing features looks binary-like.")

            if metadata_supports_non_detect or (
                zero_rate_global >= 0.02
                and positive_continuous_zero_features >= max(3, features_with_zero_count // 2)
            ):
                zero_semantics_status = "non_detect_like"
                suggested_zero_handling = "convert_zero_to_nan"
            elif binary_like_zero_features >= max(3, features_with_zero_count // 2) and (
                positive_continuous_zero_features < max(2, features_with_zero_count // 3)
            ):
                zero_semantics_status = "true_zero"
                suggested_zero_handling = "keep_zero"
            else:
                zero_semantics_status = "unknown"
                suggested_zero_handling = "require_sensitivity_analysis"

        report = {
            "success": True,
            "dataset_path": dataset_path,
            "context_report_path": context_report_path,
            "features_analyzed": len(numeric_columns),
            "total_zero_count": int(total_zero_count),
            "zero_rate_global": zero_rate_global,
            "features_with_zero_count": int(features_with_zero_count),
            "positive_continuous_zero_features": int(positive_continuous_zero_features),
            "binary_like_zero_features": int(binary_like_zero_features),
            "zero_semantics_status": zero_semantics_status,
            "suggested_zero_handling": suggested_zero_handling,
            "top_zero_features": zero_feature_summaries[:10],
            "evidence_flags": evidence_flags,
            "warnings": [] if zero_semantics_status != "unknown" else [
                "Zero semantics remain ambiguous; consider running a sensitivity analysis."
            ],
        }
        return json.dumps(report, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def missingness_assessment_tool(
    dataset_path: str,
    context_report_path: str,
    zero_pattern_report_path: str = "",
    zero_handling_mode: str = "auto",
    protected_columns: Optional[List[str]] = None,
) -> str:
    """
    Summarize missingness after optional zero-to-NaN conversion and recommend an imputation family.

    This tool is designed for privacy-preserving preprocessing orchestration:
    it reads the matrix locally, applies an optional zero-to-NaN transformation for analysis only,
    and returns structured summary statistics without exposing raw data values.

    Args:
        dataset_path: 数据文件路径（CSV或Excel）
        context_report_path: preprocessing_context.json 路径
        zero_pattern_report_path: zero_pattern_report.json 路径（可选）
        zero_handling_mode: keep_zero | convert_zero_to_nan | auto
        protected_columns: 额外保护列（可选）

    Returns:
        JSON字符串，包含 missingness_report.json 所需的核心字段
    """
    try:
        df = _read_table_file(dataset_path)
        context_report = _load_optional_payload(context_report_path) or {}
        zero_report = _load_optional_payload(zero_pattern_report_path) or {}

        combined_protected = list(dict.fromkeys(
            (protected_columns or [])
            + list(context_report.get("protected_columns", []))
            + list(TECHNICAL_PROTECTED_COLUMNS)
        ))

        numeric_columns = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in combined_protected
        ]
        if not numeric_columns:
            return json.dumps({
                "success": False,
                "error": "未找到可用于缺失机制评估的数值特征列"
            }, ensure_ascii=False)

        analysis_df = df[numeric_columns].copy()
        resolved_zero_mode = zero_handling_mode
        zero_status = str(zero_report.get("zero_semantics_status", "unknown"))
        zero_suggestion = str(zero_report.get("suggested_zero_handling", ""))

        if zero_handling_mode == "auto":
            if zero_suggestion == "convert_zero_to_nan" or zero_status == "non_detect_like":
                resolved_zero_mode = "convert_zero_to_nan"
            else:
                resolved_zero_mode = "keep_zero"

        converted_zero_count = 0
        if resolved_zero_mode == "convert_zero_to_nan":
            zero_mask = analysis_df == 0
            converted_zero_count = int(zero_mask.sum().sum())
            analysis_df = analysis_df.mask(zero_mask, np.nan)

        total_missing_count = int(analysis_df.isnull().sum().sum())
        total_cells = int(analysis_df.shape[0] * analysis_df.shape[1])
        missing_rate_global = round(total_missing_count / max(total_cells, 1), 6)

        feature_missing_rates = analysis_df.isnull().mean(axis=0)
        sample_missing_rates = analysis_df.isnull().mean(axis=1)

        high_missing_features = feature_missing_rates[feature_missing_rates > 0.30]
        high_missing_samples = sample_missing_rates[sample_missing_rates > 0.35]

        left_censoring_features: List[str] = []
        for col in numeric_columns:
            series = analysis_df[col]
            if series.isnull().sum() == 0:
                continue
            observed = series.dropna()
            if observed.empty:
                continue
            q25 = observed.quantile(0.25)
            median = observed.median()
            min_positive = observed[observed > 0].min() if (observed > 0).any() else None
            if min_positive is not None and q25 <= max(min_positive * 2, median * 0.2):
                left_censoring_features.append(col)

        feature_missing_summary = {
            "feature_count_analyzed": int(len(numeric_columns)),
            "features_with_missing": int((feature_missing_rates > 0).sum()),
            "features_above_10pct_missing": int((feature_missing_rates > 0.10).sum()),
            "features_above_30pct_missing": int((feature_missing_rates > 0.30).sum()),
            "max_feature_missing_rate": _safe_float(feature_missing_rates.max()),
            "median_feature_missing_rate": _safe_float(feature_missing_rates.median()),
            "top_missing_features": [
                {
                    "name": str(name),
                    "missing_rate": round(float(rate), 6),
                }
                for name, rate in feature_missing_rates.sort_values(ascending=False).head(10).items()
                if float(rate) > 0
            ],
        }

        sample_missing_summary = {
            "sample_count_analyzed": int(len(analysis_df)),
            "samples_with_missing": int((sample_missing_rates > 0).sum()),
            "samples_above_20pct_missing": int((sample_missing_rates > 0.20).sum()),
            "samples_above_35pct_missing": int((sample_missing_rates > 0.35).sum()),
            "max_sample_missing_rate": _safe_float(sample_missing_rates.max()),
            "median_sample_missing_rate": _safe_float(sample_missing_rates.median()),
        }

        warnings: List[str] = []
        evidence_flags: List[str] = []

        if resolved_zero_mode == "convert_zero_to_nan" and converted_zero_count > 0:
            evidence_flags.append("Zero values were converted to missing for assessment.")
        if missing_rate_global == 0:
            suspected_mechanism = "low_missing"
            recommended_imputation_family = "none"
            evidence_flags.append("No missing values remain after the selected zero handling mode.")
        else:
            left_censoring_ratio = len(left_censoring_features) / max(int((feature_missing_rates > 0).sum()), 1)
            if zero_status == "non_detect_like":
                evidence_flags.append("Upstream zero analysis suggested non-detect-like zeros.")
            if len(left_censoring_features) > 0:
                evidence_flags.append("Several missingness-bearing features show left-censoring-like distributions.")
            if len(high_missing_features) > 0:
                warnings.append("Some features exceed the default 30% missing-rate review threshold.")
            if len(high_missing_samples) > 0:
                warnings.append("Some samples exceed the default 35% missing-rate review threshold.")

            if zero_status == "non_detect_like" or left_censoring_ratio >= 0.30:
                suspected_mechanism = "MNAR"
                recommended_imputation_family = "QRILC"
            elif missing_rate_global <= 0.02 and len(left_censoring_features) == 0:
                suspected_mechanism = "low_missing"
                recommended_imputation_family = "none"
            else:
                suspected_mechanism = "MAR_or_MCAR"
                recommended_imputation_family = "KNN"

        recommended_thresholds = {
            "sample_missing_rate_threshold": 0.35,
            "feature_missing_rate_threshold": 0.30,
            "zero_handling_mode_used": resolved_zero_mode,
            "zero_converted_to_missing_count": int(converted_zero_count),
        }

        if recommended_imputation_family == "none" and total_missing_count > 0:
            recommended_imputation_family = "median_fallback"
            warnings.append("Missingness remains but no strong mechanism was identified; consider conservative fallback.")

        report = {
            "success": True,
            "dataset_path": dataset_path,
            "context_report_path": context_report_path,
            "zero_pattern_report_path": zero_pattern_report_path or None,
            "zero_handling_mode_requested": zero_handling_mode,
            "zero_handling_mode_used": resolved_zero_mode,
            "total_missing_count": total_missing_count,
            "missing_rate_global": missing_rate_global,
            "feature_missing_summary": feature_missing_summary,
            "sample_missing_summary": sample_missing_summary,
            "suspected_mechanism": suspected_mechanism,
            "left_censoring_signals": left_censoring_features[:20],
            "recommended_imputation_family": recommended_imputation_family,
            "recommended_thresholds": recommended_thresholds,
            "evidence_flags": evidence_flags,
            "warnings": warnings,
        }
        return json.dumps(report, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def preprocessing_qa_reporter_tool(
    context_report_path: str,
    schema_audit_report_path: str = "",
    eligibility_filter_report_path: str = "",
    zero_pattern_report_path: str = "",
    missingness_report_path: str = "",
    imputation_report_path: str = "",
    normalization_decision_report_path: str = "",
    transformation_report_path: str = "",
    outlier_audit_report_path: str = "",
    batch_drift_audit_path: str = "",
    feature_engineering_report_path: str = "",
    scaling_report_path: str = "",
) -> str:
    """
    Merge preprocessing stage reports into a single auditable source of truth.

    The reporter collects stage-level artifacts, derives a compact set of Phase 4
    reportable facts, and writes three outputs next to the context report:
    - preprocessing_report.json
    - preprocessing_summary.md
    - preprocessing_decision_pack.json

    Args:
        context_report_path: preprocessing_context.json 路径
        schema_audit_report_path: schema audit 报告路径（可选）
        eligibility_filter_report_path: eligibility filter 报告路径（可选）
        zero_pattern_report_path: zero pattern 报告路径（可选）
        missingness_report_path: missingness 报告路径（可选）
        imputation_report_path: imputation 报告路径（可选）
        normalization_decision_report_path: normalization decision 报告路径（可选）
        transformation_report_path: transformation 报告路径（可选）
        outlier_audit_report_path: outlier audit 报告路径（可选）
        batch_drift_audit_path: batch/drift audit 报告路径（可选）
        feature_engineering_report_path: feature engineering 报告路径（可选）
        scaling_report_path: scaling 报告路径（可选）

    Returns:
        JSON字符串，说明三个输出工件路径和核心摘要字段
    """
    try:
        report_paths = {
            "context": context_report_path,
            "schema_audit": schema_audit_report_path,
            "eligibility_filter": eligibility_filter_report_path,
            "zero_pattern": zero_pattern_report_path,
            "missingness": missingness_report_path,
            "imputation": imputation_report_path,
            "normalization": normalization_decision_report_path,
            "transformation": transformation_report_path,
            "outlier": outlier_audit_report_path,
            "batch_drift": batch_drift_audit_path,
            "feature_engineering": feature_engineering_report_path,
            "scaling": scaling_report_path,
        }
        loaded_reports = {
            key: _load_optional_payload(path)
            for key, path in report_paths.items()
        }

        context = loaded_reports.get("context") or {}
        zero_report = loaded_reports.get("zero_pattern") or {}
        missingness_report = loaded_reports.get("missingness") or {}
        imputation_report = loaded_reports.get("imputation") or {}
        normalization_report = loaded_reports.get("normalization") or {}
        transformation_report = loaded_reports.get("transformation") or {}
        outlier_report = loaded_reports.get("outlier") or {}
        batch_report = loaded_reports.get("batch_drift") or {}
        feature_engineering_report = loaded_reports.get("feature_engineering") or {}
        scaling_report = loaded_reports.get("scaling") or {}

        output_dir = Path(context_report_path).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        branch_used = (
            context.get("candidate_branch")
            or context.get("branch_used")
            or "unknown"
        )
        thresholds = {}
        for source in (
            loaded_reports.get("eligibility_filter") or {},
            missingness_report.get("recommended_thresholds") or {},
            normalization_report.get("recommended_thresholds") or {},
        ):
            if isinstance(source, dict):
                thresholds.update(source)

        zero_handling_strategy = (
            missingness_report.get("zero_handling_mode_used")
            or zero_report.get("suggested_zero_handling")
            or "not_recorded"
        )
        imputation_method = (
            imputation_report.get("method")
            or missingness_report.get("recommended_imputation_family")
            or "not_recorded"
        )
        normalization_method = (
            normalization_report.get("method")
            or normalization_report.get("normalization_method")
            or "not_recorded"
        )
        transformation_method = (
            transformation_report.get("method")
            or transformation_report.get("transformation_method")
            or "not_recorded"
        )
        outlier_method = (
            outlier_report.get("method")
            or outlier_report.get("outlier_method")
            or "not_recorded"
        )

        batch_correction_applied = bool(
            batch_report.get("batch_correction_applied")
            or batch_report.get("correction_applied")
        )
        qc_rsd_filter_applied = bool(
            batch_report.get("qc_rsd_filter_applied")
            or (loaded_reports.get("eligibility_filter") or {}).get("qc_rsd_filter_applied")
        )
        qc_rsd_threshold = (
            batch_report.get("qc_rsd_threshold")
            or (loaded_reports.get("eligibility_filter") or {}).get("qc_rsd_threshold")
        )

        final_feature_count = (
            feature_engineering_report.get("final_feature_count")
            or feature_engineering_report.get("engineered_feature_count")
            or scaling_report.get("scaled_feature_count")
            or context.get("numeric_feature_count")
        )

        warnings: List[str] = []
        for key, report in loaded_reports.items():
            if isinstance(report, dict):
                warnings.extend([str(item) for item in report.get("warnings", []) if item])
        warnings = list(dict.fromkeys(warnings))

        stage_status = {}
        for key, path in report_paths.items():
            stage_status[key] = {
                "path": path or None,
                "present": bool(path and Path(path).exists()),
            }

        phase4_reportable_facts = {
            "data_level": context.get("data_level", "unknown"),
            "platform_source": context.get("platform_source", "unknown"),
            "zero_handling_strategy": zero_handling_strategy,
            "sample_missing_rate_threshold": thresholds.get("sample_missing_rate_threshold"),
            "feature_missing_rate_threshold": thresholds.get("feature_missing_rate_threshold"),
            "imputation_method": imputation_method,
            "normalization_method": normalization_method,
            "transformation_method": transformation_method,
            "outlier_method": outlier_method,
            "batch_correction_applied": batch_correction_applied,
            "qc_rsd_filter_applied": qc_rsd_filter_applied,
            "qc_rsd_threshold": qc_rsd_threshold,
        }

        preprocessing_report = {
            "success": True,
            "branch_used": branch_used,
            "thresholds": thresholds,
            "zero_handling_strategy": zero_handling_strategy,
            "imputation_method": imputation_method,
            "normalization_method": normalization_method,
            "transformation_method": transformation_method,
            "outlier_method": outlier_method,
            "batch_correction_applied": batch_correction_applied,
            "qc_rsd_filter_applied": qc_rsd_filter_applied,
            "qc_rsd_threshold": qc_rsd_threshold,
            "final_feature_count": final_feature_count,
            "warnings": warnings,
            "phase4_reportable_facts": phase4_reportable_facts,
            "stage_status": stage_status,
        }

        decision_pack = {
            "branch_used": branch_used,
            "context": {
                "platform_source": context.get("platform_source"),
                "data_level": context.get("data_level"),
                "has_pooled_qc": context.get("has_pooled_qc"),
                "has_blank_samples": context.get("has_blank_samples"),
                "has_run_order": context.get("has_run_order"),
                "has_batch_metadata": context.get("has_batch_metadata"),
            },
            "decisions": {
                "zero_handling_strategy": zero_handling_strategy,
                "suspected_missingness_mechanism": missingness_report.get("suspected_mechanism"),
                "recommended_imputation_family": missingness_report.get("recommended_imputation_family"),
                "imputation_method": imputation_method,
                "normalization_method": normalization_method,
                "transformation_method": transformation_method,
                "outlier_method": outlier_method,
            },
            "phase4_reportable_facts": phase4_reportable_facts,
            "warnings": warnings,
        }

        summary_lines = [
            "# Preprocessing Summary",
            "",
            f"- Branch used: `{branch_used}`",
            f"- Platform source: `{phase4_reportable_facts['platform_source']}`",
            f"- Data level: `{phase4_reportable_facts['data_level']}`",
            f"- Zero handling: `{zero_handling_strategy}`",
            f"- Imputation method: `{imputation_method}`",
            f"- Normalization method: `{normalization_method}`",
            f"- Transformation method: `{transformation_method}`",
            f"- Outlier method: `{outlier_method}`",
            f"- Batch correction applied: `{batch_correction_applied}`",
            f"- QC-RSD filter applied: `{qc_rsd_filter_applied}`",
            f"- Final feature count: `{final_feature_count}`",
        ]
        if warnings:
            summary_lines.extend([
                "",
                "## Warnings",
            ])
            summary_lines.extend([f"- {warning}" for warning in warnings])

        report_path = output_dir / "preprocessing_report.json"
        summary_path = output_dir / "preprocessing_summary.md"
        decision_pack_path = output_dir / "preprocessing_decision_pack.json"

        report_path.write_text(json.dumps(preprocessing_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        decision_pack_path.write_text(json.dumps(decision_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        return json.dumps({
            "success": True,
            "output_directory": str(output_dir),
            "preprocessing_report_path": str(report_path),
            "preprocessing_summary_path": str(summary_path),
            "preprocessing_decision_pack_path": str(decision_pack_path),
            "branch_used": branch_used,
            "phase4_reportable_facts": phase4_reportable_facts,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def search_cleaning_code(query: str) -> str:
    """
    数据清洗代码知识库搜索工具
    
    这个工具提供了一个专门针对数据清洗任务的Python代码知识库。
    它包含了各种常用的数据清洗技术、算法和最佳实践的代码示例。
    
    知识库内容：
    1. 缺失值处理：均值填充、中位数填充、众数填充、KNN填充
    2. 异常值处理：Z-score检测、IQR检测、异常值截断
    3. 数据标准化：Z-score标准化、Min-Max缩放
    4. 高缺失值处理：列删除、行删除策略
    
    搜索机制：
    - 支持中文关键词搜索
    - 基于关键词匹配返回相关代码片段
    - 提供完整的可执行代码示例
    - 包含详细的使用说明和参数解释
    
    Args:
        query (str): 搜索查询关键词，支持以下类型：
            - 中文关键词：如"缺失值填充"、"异常值处理"、"数据标准化"
            - 英文关键词：如"missing values"、"outliers"、"normalization"
            - 具体方法：如"mean imputation"、"IQR detection"、"KNN imputation"
        
    Returns:
        str: 格式化的代码片段和说明文档，包含：
            - 代码标题和分类
            - 完整的Python代码示例
            - 使用说明和参数解释
            - 适用场景和注意事项
    
    Example:
        >>> result = search_cleaning_code("缺失值填充")
        >>> print(result)  # 返回均值、中位数、众数、KNN填充的代码示例
    """
    # 数据清洗代码知识库
    cleaning_code_kb = {
        "缺失值填充": {
            "mean_imputation": """
# 均值填充 - 适用于正态分布的数据
import pandas as pd
import numpy as np

def mean_imputation(df, column):
    df[column] = df[column].fillna(df[column].mean())
    return df

# 使用示例
df = mean_imputation(df, 'age')
""",
            "median_imputation": """
# 中位数填充 - 适用于偏态分布的数据
import pandas as pd

def median_imputation(df, column):
    df[column] = df[column].fillna(df[column].median())
    return df

# 使用示例
df = median_imputation(df, 'income')
""",
            "mode_imputation": """
# 众数填充 - 适用于分类变量
import pandas as pd

def mode_imputation(df, column):
    mode_value = df[column].mode()[0] if not df[column].mode().empty else None
    if mode_value is not None:
        df[column] = df[column].fillna(mode_value)
    return df

# 使用示例
df = mode_imputation(df, 'category')
""",
            "knn_imputation": """
# KNN填充 - 基于相似性填充
from sklearn.impute import KNNImputer
import pandas as pd

def knn_imputation(df, columns, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[columns] = imputer.fit_transform(df[columns])
    return df

# 使用示例
numeric_columns = df.select_dtypes(include=[np.number]).columns
df = knn_imputation(df, numeric_columns)
"""
        },
        "异常值处理": {
            "zscore_detection": """
# Z-score方法检测异常值
import numpy as np
import pandas as pd

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[column][z_scores > threshold]
    return outliers

# 使用示例
outliers = detect_outliers_zscore(df, 'metabolite_concentration')
""",
            "iqr_detection": """
# IQR方法检测异常值
import pandas as pd

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# 使用示例
outliers = detect_outliers_iqr(df, 'metabolite_concentration')
""",
            "outlier_capping": """
# 异常值截断处理
import pandas as pd

def cap_outliers(df, column, method='iqr'):
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'percentile':
        lower_bound = df[column].quantile(0.05)
        upper_bound = df[column].quantile(0.95)
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

# 使用示例
df = cap_outliers(df, 'metabolite_concentration', method='iqr')
"""
        },
        "数据标准化": {
            "standardization": """
# Z-score标准化
from sklearn.preprocessing import StandardScaler
import pandas as pd

def standardize_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# 使用示例
numeric_columns = df.select_dtypes(include=[np.number]).columns
df, scaler = standardize_data(df, numeric_columns)
""",
            "minmax_scaling": """
# Min-Max标准化
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def minmax_scale_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# 使用示例
numeric_columns = df.select_dtypes(include=[np.number]).columns
df, scaler = minmax_scale_data(df, numeric_columns)
"""
        },
        "高缺失值处理": {
            "column_removal": """
# 删除高缺失值列
import pandas as pd

def remove_high_missing_columns(df, threshold=0.5):
    missing_percentages = df.isnull().sum() / len(df)
    columns_to_remove = missing_percentages[missing_percentages > threshold].index
    df = df.drop(columns=columns_to_remove)
    return df, list(columns_to_remove)

# 使用示例
df, removed_columns = remove_high_missing_columns(df, threshold=0.8)
print(f"删除了以下列: {removed_columns}")
""",
            "row_removal": """
# 删除高缺失值行
import pandas as pd

def remove_high_missing_rows(df, threshold=0.5):
    missing_percentages = df.isnull().sum(axis=1) / len(df.columns)
    rows_to_remove = missing_percentages[missing_percentages > threshold].index
    df = df.drop(index=rows_to_remove)
    return df, len(rows_to_remove)

# 使用示例
df, removed_rows_count = remove_high_missing_rows(df, threshold=0.7)
print(f"删除了 {removed_rows_count} 行")
"""
        }
    }
    
    # 简单的关键词匹配搜索
    query_lower = query.lower()
    results = []
    
    for category, methods in cleaning_code_kb.items():
        if any(keyword in query_lower for keyword in [category.lower(), "缺失", "异常", "标准化", "填充", "处理"]):
            for method_name, code in methods.items():
                if any(keyword in query_lower for keyword in method_name.lower().split('_')):
                    results.append(f"## {category} - {method_name}\n{code}")
    
    # 如果没有找到匹配的结果，返回通用建议
    if not results:
        results.append("""
## 数据清洗通用建议

1. **缺失值处理**:
   - 数值型数据: 考虑均值、中位数或KNN填充
   - 分类数据: 使用众数填充或创建"未知"类别

2. **异常值处理**:
   - 使用IQR或Z-score方法检测
   - 考虑截断、删除或转换处理

3. **数据标准化**:
   - 对于需要距离计算的算法，使用标准化
   - 对于神经网络，考虑Min-Max缩放

请提供更具体的查询以获得详细的代码示例。
        """)
    
    return "\n\n".join(results)


def delete_columns_tool(columns: List[str], data_path: str) -> str:
    """
    删除指定列的工具
    
    从数据集中删除一个或多个指定的列。
    遵循"读取 -> 修改 -> 保存"模式，覆盖输入文件以确保更改被链接。
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally, modifies it, and saves back to the same file
    - Returns only metadata (success status and deleted column names)
    - No raw data is exposed to external APIs
    
    Args:
        columns: 要删除的列名列表
        data_path: 数据文件路径（CSV或Excel）
        
    Returns:
        报告成功和删除列名的JSON字符串
    """
    try:
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "不支持的文件格式"})
        
        original_columns = df.columns.tolist()
        df.drop(columns=columns, inplace=True, errors='ignore')
        
        # 保存修改后的数据
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
        else:
            df.to_excel(data_path, index=False)
        
        deleted = [col for col in columns if col in original_columns and col not in df.columns]
        return json.dumps({"success": True, "output_path": data_path, "deleted_columns": deleted})
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def impute_missing_tool(columns: List[str], data_path: str, method: str = 'mean') -> str:
    """
    填充缺失值的工具
    
    使用指定方法填充指定列中的缺失值。
    遵循"读取 -> 修改 -> 保存"模式，覆盖输入文件以确保更改被链接。
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally, processes it, and saves back to the same file
    - Returns only metadata (success status, imputed columns, and method used)
    - No raw data values are exposed to external APIs
    
    支持的插补方法：
    - 'mean', 'median', 'mode': 基础统计方法
    - 'knn': K近邻插补，适用于多变量缺失
    - 'qrilc': QRILC方法，专门处理低于检测限(MNAR)的数据
    
    Args:
        columns: 要填充的列名列表
        data_path: 数据文件路径
        method: 填充方法。支持: 'mean', 'median', 'mode', 'knn', 'qrilc'
        
    Returns:
        报告成功和填充详情的JSON字符串
    """
    try:
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "不支持的文件格式"})
        
        method_requested = str(method or "").strip().lower() or "median"
        effective_method = method_requested
        warned_knn_fallback = False

        for col in columns:
            if col in df.columns:
                if method_requested == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif method_requested == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif method_requested == 'mode':
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else None
                    if mode_value is not None:
                        df[col] = df[col].fillna(mode_value)
                elif method_requested == 'knn':
                    # 使用KNN方法填充缺失值
                    # ⚠️ 性能警告：KNN插补对大数据集（>10000行）非常慢，复杂度O(n²)
                    try:
                        from sklearn.impute import KNNImputer
                        # 只对数值列进行KNN填充
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if col in numeric_cols and len(numeric_cols) > 1:
                            # 检查数据规模，对于大数据集使用更快的插补方法
                            n_rows = len(df)
                            if n_rows > 10000:
                                # 对于大数据集，KNN太慢（O(n²)复杂度），改用中位数填充
                                import warnings
                                if not warned_knn_fallback:
                                    warnings.warn(
                                        f"数据规模较大（{n_rows}行），KNN插补会非常慢（可能需要数小时）。改用中位数填充以提高性能。"
                                    )
                                    warned_knn_fallback = True
                                # 使用SimpleImputer替代KNN，速度快100倍以上
                                from sklearn.impute import SimpleImputer
                                simple_imputer = SimpleImputer(strategy='median')
                                df[numeric_cols] = simple_imputer.fit_transform(df[numeric_cols])
                                effective_method = "median"
                                break
                            else:
                                # 小数据集（≤10000行），可以使用KNN
                                imputer = KNNImputer(n_neighbors=min(5, n_rows // 1000 + 1))  # 根据数据规模调整邻居数
                                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                                effective_method = "knn"
                        else:
                            # 如果只有一列或不是数值列，回退到均值填充
                            df[col] = df[col].fillna(df[col].mean())
                            effective_method = "mean"
                    except ImportError:
                        # 如果没有sklearn，回退到均值填充
                        df[col] = df[col].fillna(df[col].mean())
                        effective_method = "mean"
                    except Exception as e:
                        # 如果KNN失败（可能因为内存或性能问题），回退到中位数填充
                        import warnings
                        if not warned_knn_fallback:
                            warnings.warn(f"KNN插补失败: {str(e)}，改用中位数填充。")
                            warned_knn_fallback = True
                        df[col] = df[col].fillna(df[col].median())
                        effective_method = "median"
                elif method == 'qrilc':
                    # 使用QRILC方法填充缺失值（专门处理低于检测限的数据）
                    try:
                        from impyute.imputation.cs import qrilc
                        # 只对数值列进行QRILC填充
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if col in numeric_cols and len(numeric_cols) > 1:
                            # 使用所有数值列进行QRILC填充
                            df_numeric = df[numeric_cols].copy()
                            # QRILC需要数值数据，处理无穷大和NaN
                            df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
                            if not df_numeric.isnull().all().all():
                                imputed_data = qrilc(df_numeric.values)
                                df[numeric_cols] = imputed_data
                        else:
                            # 如果只有一列或不是数值列，使用half-minimum方法
                            col_data = df[col].dropna()
                            if len(col_data) > 0:
                                half_min = col_data.min() / 2
                                df[col] = df[col].fillna(half_min)
                            else:
                                df[col] = df[col].fillna(0)
                    except ImportError:
                        # 如果没有impyute库，使用half-minimum方法作为备用方案
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            half_min = col_data.min() / 2
                            df[col] = df[col].fillna(half_min)
                        else:
                            df[col] = df[col].fillna(0)
        
        # 保存修改后的数据
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
        else:
            df.to_excel(data_path, index=False)
        
        return json.dumps({
            "success": True,
            "output_path": data_path,
            "imputed_columns": columns,
            "method": effective_method,
            "method_used": effective_method,
            "method_requested": method_requested,
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def cap_outliers_tool(data_path: str, columns: List[str] = None, method: str = 'iqr') -> str:
    """
    截断异常值的工具
    
    使用IQR方法截断指定列中的异常值。
    遵循"读取 -> 修改 -> 保存"模式，覆盖输入文件以确保更改被链接。
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally, processes it, and saves back to the same file
    - Returns only metadata (success status, capped columns, and method used)
    - No raw data values are exposed to external APIs
    
    Args:
        data_path: 数据文件路径
        columns: 要截断的列名列表。如果为None，则处理所有数值列
        method: 异常值检测方法。目前支持'iqr'
        
    Returns:
        报告成功和截断详情的JSON字符串
    """
    try:
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "不支持的文件格式"})
        
        # 如果没有指定列，则处理所有数值列
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numeric_cols if not _is_technical_protected_column(col)]
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 保存修改后的数据
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
        else:
            df.to_excel(data_path, index=False)
        
        return json.dumps({"success": True, "output_path": data_path, "capped_columns": columns, "method": method})
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def delete_rows_tool(condition: str, data_path: str) -> str:
    """
    删除行的工具
    
    根据指定条件删除行。
    遵循"读取 -> 修改 -> 保存"模式，覆盖输入文件以确保更改被链接。
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally, processes it, and saves back to the same file
    - Returns only metadata (success status, deleted row count, and condition)
    - No raw data values are exposed to external APIs
    
    Args:
        condition: 删除条件（例如："missing_percentage > 0.8"）
        data_path: 数据文件路径
        
    Returns:
        报告成功和删除详情的JSON字符串
    """
    try:
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "不支持的文件格式"})
        
        original_rows = len(df)
        
        # 根据条件删除行
        if "missing_rate" in condition and ">" in condition:
            # 处理 missing_rate > 0.35 这样的条件
            threshold = float(condition.split(">")[1].strip())
            missing_percentage = df.isnull().sum(axis=1) / len(df.columns)
            df = df[missing_percentage <= threshold]
        elif "missing_percentage" in condition and ">" in condition:
            # 处理 missing_percentage > 0.8 这样的条件
            threshold = float(condition.split(">")[1].strip())
            missing_percentage = df.isnull().sum(axis=1) / len(df.columns)
            df = df[missing_percentage <= threshold]
        else:
            # 如果条件不匹配，返回错误
            return json.dumps({"success": False, "error": f"不支持的条件格式: {condition}"})
        
        deleted_rows = original_rows - len(df)
        
        # 保存修改后的数据
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
        else:
            df.to_excel(data_path, index=False)
        
        return json.dumps({"success": True, "output_path": data_path, "deleted_rows": deleted_rows, "condition": condition})
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def check_power_impact(n_case: int, n_control: int, n_to_remove_case: int, n_to_remove_control: int, n_min: int = 20) -> str:
    """
    统计功效影响评估工具
    
    评估删除样本操作对统计功效的潜在影响。这个工具帮助研究人员在删除数据前
    评估操作对研究统计功效的影响，确保删除后仍有足够的样本量进行有效的统计分析。
    
    功能：
    1. 计算删除样本后剩余的病例组和对照组数量
    2. 检查剩余样本量是否满足最小要求
    3. 评估统计功效受损的风险等级
    4. 提供风险评估建议
    
    Args:
        n_case (int): 当前病例组样本数量
        n_control (int): 当前对照组样本数量  
        n_to_remove_case (int): 计划删除的病例组样本数量
        n_to_remove_control (int): 计划删除的对照组样本数量
        n_min (int, optional): 最小样本量要求，默认为20
        
    Returns:
        str: JSON格式的风险评估结果，包含以下结构：
            {
                "risk": "High" | "Low",
                "message": "详细的风险评估信息",
                "remaining_case": 剩余病例组样本数,
                "remaining_control": 剩余对照组样本数,
                "risk_factors": ["风险因素列表"]
            }
    
    Raises:
        Exception: 当参数无效时抛出异常
    """
    try:
        # 计算删除样本后剩余的数量
        remaining_case = n_case - n_to_remove_case
        remaining_control = n_control - n_to_remove_control
        
        # 检查是否有任一组的剩余样本量低于最小要求
        risk_factors = []
        if remaining_case < n_min:
            risk_factors.append(f"病例组剩余样本量({remaining_case})低于最小要求({n_min})")
        if remaining_control < n_min:
            risk_factors.append(f"对照组剩余样本量({remaining_control})低于最小要求({n_min})")
        
        # 评估风险等级
        if len(risk_factors) > 0:
            return json.dumps({
                "risk": "High",
                "message": f"警告：删除操作将导致至少一个组的样本量低于最小要求（{n_min}），统计功效可能严重受损。",
                "remaining_case": remaining_case,
                "remaining_control": remaining_control,
                "risk_factors": risk_factors
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "risk": "Low", 
                "message": "风险评估通过。删除后样本量充足。",
                "remaining_case": remaining_case,
                "remaining_control": remaining_control,
                "risk_factors": []
            }, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({"error": f"评估统计功效影响时发生错误: {str(e)}"})


def pqn_normalization_tool(data_path: str, sample_column: str = None) -> str:
    """
    PQN (Probabilistic Quotient Normalization) 归一化工具
    
    概率商归一化是代谢组学中常用的归一化方法，通过计算每个样本与参考样本（通常是中位数样本）的商来归一化数据。
    这种方法可以有效消除系统性的样本间差异，保留生物学的相关变化。
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally, processes it, and saves back to the same file
    - Returns only metadata (success status, normalized columns, method, and sample column)
    - No raw data values are exposed to external APIs
    
    Args:
        data_path: 数据文件路径（CSV或Excel）
        sample_column: 样本列名（通常是第一列，包含样本ID）。如果为None，将使用第一列作为样本列。
        
    Returns:
        报告成功和归一化详情的JSON字符串
    """
    try:
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "不支持的文件格式"})
        
        def _norm_col(value: Any) -> str:
            return str(value).strip().lower()

        normalized_to_original: Dict[str, str] = {}
        for col in df.columns:
            key = _norm_col(col)
            if key and key not in normalized_to_original:
                normalized_to_original[key] = col

        if sample_column is None:
            candidate_id_cols = (
                "Sample_ID",
                "sample_id",
                "SampleID",
                "sampleid",
                "ROW_ID",
                "row_id",
                "__row_id__",
                "ID",
                "id",
            )
            sample_column = next(
                (
                    c for c in candidate_id_cols
                    if c in df.columns or _norm_col(c) in normalized_to_original
                ),
                None,
            )
            if sample_column is not None and sample_column not in df.columns:
                sample_column = normalized_to_original.get(_norm_col(sample_column), sample_column)
            if sample_column is None:
                sample_column = "__row_id__"
                df[sample_column] = range(len(df))
        elif sample_column not in df.columns:
            resolved = normalized_to_original.get(_norm_col(sample_column))
            if resolved:
                sample_column = resolved
        
        # 分离样本列和特征列
        sample_ids = df[sample_column]
        
        protected_cols_set = {sample_column}
        for technical_col in ("Sample_ID", "sample_id", "SampleID", "sampleid", "ROW_ID", "row_id", "__row_id__", "ID", "id"):
            if technical_col in df.columns:
                protected_cols_set.add(technical_col)
            else:
                resolved = normalized_to_original.get(_norm_col(technical_col))
                if resolved:
                    protected_cols_set.add(resolved)
        for target_col in ("Group", "group", "target"):
            if target_col in df.columns:
                protected_cols_set.add(target_col)
            else:
                resolved = normalized_to_original.get(_norm_col(target_col))
                if resolved:
                    protected_cols_set.add(resolved)
        protected_cols = list(protected_cols_set)
        
        feature_df = df.drop(columns=protected_cols, errors="ignore")
        
        # 只对数值列进行PQN归一化
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return json.dumps({"success": False, "error": "未找到数值型特征列"})
        
        # 提取数值数据
        numeric_data = feature_df[numeric_cols].copy()
        
        # 计算参考谱（中位数谱）- 只使用非缺失值
        reference_spectrum = numeric_data.median(axis=0)
        
        # 避免除零错误，将0替换为很小的正数
        reference_spectrum = reference_spectrum.replace(0, 1e-10)
        
        # 对每个样本计算相对于参考谱的商
        quotient_matrix = numeric_data.div(reference_spectrum, axis=1)
        
        # 计算每个样本的中位数商（只使用非缺失值）
        median_quotients = quotient_matrix.median(axis=1)
        
        # 避免除零错误，将0或缺失值替换为很小的正数
        median_quotients = median_quotients.replace([0, np.nan], 1e-10)
        
        # 用中位数商归一化原始数据
        for col in numeric_cols:
            feature_df[col] = numeric_data[col].div(median_quotients, axis=0)
        
        # 将归一化后的数据合并回原始DataFrame
        df[sample_column] = sample_ids
        df[numeric_cols] = feature_df[numeric_cols]
        # Group column is preserved automatically (not in numeric_cols)
        
        # 保存修改后的数据
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
        else:
            df.to_excel(data_path, index=False)
        
        return json.dumps({
            "success": True,
            "output_path": data_path,  # File path after normalization (in-place)
            "normalized_columns": numeric_cols,
            "method": "PQN",
            "sample_column": sample_column
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def log2_transformation_tool(data_path: str, columns: List[str] = None) -> str:
    """
    Log2变换工具
    
    对指定列进行log2变换，常用于代谢组学数据预处理，可以：
    1. 稳定方差
    2. 使数据分布更接近正态分布
    3. 减少异常值的影响
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally, processes it, and saves back to the same file
    - Returns only metadata (success status, transformed columns, and method)
    - No raw data values are exposed to external APIs
    
    注意：对于包含0或负值的数据，会在变换前加上一个小的常数（通常是1）以避免log(0)错误。
    
    Args:
        data_path: 数据文件路径（CSV或Excel）
        columns: 要进行log2变换的列名列表。如果为None，将对所有数值列进行变换。
        
    Returns:
        报告成功和变换详情的JSON字符串
    """
    try:
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "不支持的文件格式"})
        
        # 如果没有指定列，对所有数值列进行变换
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numeric_cols if not _is_technical_protected_column(col)]
        else:
            # 过滤出实际存在的数值列
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(columns) == 0:
            return json.dumps({"success": False, "error": "未找到有效的数值列"})
        
        # 对每列进行log2变换
        transformed_columns = []
        for col in columns:
            # 检查是否有0或负值
            col_min = df[col].min()
            if col_min <= 0:
                # 添加常数1以避免log(0)或log(负数)
                df[col] = np.log2(df[col] + 1 - col_min)
            else:
                df[col] = np.log2(df[col])
            transformed_columns.append(col)
        
        # 保存修改后的数据
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
        else:
            df.to_excel(data_path, index=False)
        
        return json.dumps({
            "success": True,
            "output_path": data_path,  # File path after transformation (in-place)
            "transformed_columns": transformed_columns,
            "method": "log2"
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def auto_scaling_tool(data_path: str, columns: List[str] = None, method: str = 'standard') -> str:
    """
    自动缩放工具
    
    对指定列进行标准化/归一化处理，常用的缩放方法包括：
    - 'standard': Z-score标准化（均值为0，标准差为1）
    - 'minmax': Min-Max归一化（缩放到0-1范围）
    - 'robust': 稳健缩放（使用中位数和四分位距，对异常值更稳健）
    
    Privacy-Preserving Design:
    - Accepts file path instead of DataFrame to prevent data leakage to LLMs
    - Reads data locally, processes it, and saves back to the same file
    - Returns only metadata (success status, scaled columns, and method)
    - No raw data values are exposed to external APIs
    
    Args:
        data_path: 数据文件路径（CSV或Excel）
        columns: 要进行缩放的列名列表。如果为None，将对所有数值列进行缩放。
        method: 缩放方法，支持 'standard', 'minmax', 'robust'，默认为 'standard'
        
    Returns:
        报告成功和缩放详情的JSON字符串
    """
    try:
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            return json.dumps({"success": False, "error": "不支持的文件格式"})
        
        # 如果没有指定列，对所有数值列进行缩放
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [col for col in numeric_cols if not _is_technical_protected_column(col)]
        else:
            # 过滤出实际存在的数值列
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(columns) == 0:
            return json.dumps({"success": False, "error": "未找到有效的数值列"})
        
        # 根据方法进行缩放
        scaled_columns = []
        for col in columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
                
            if method == 'standard':
                # Z-score标准化
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                scaled_columns.append(col)
                
            elif method == 'minmax':
                # Min-Max归一化
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                scaled_columns.append(col)
                
            elif method == 'robust':
                # 稳健缩放（使用中位数和IQR）
                median_val = col_data.median()
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    df[col] = (df[col] - median_val) / IQR
                scaled_columns.append(col)
            else:
                return json.dumps({"success": False, "error": f"不支持的缩放方法: {method}"})
        
        # 保存修改后的数据
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
        else:
            df.to_excel(data_path, index=False)
        
        return json.dumps({
            "success": True,
            "output_path": data_path,  # File path after scaling (in-place)
            "scaled_columns": scaled_columns,
            "method": method
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ============================================================================
# LangChain Tool Wrappers
# ============================================================================
# These wrappers are created for LangGraph/LangChain integration.
# The raw functions above can be imported and called directly by generated code.

from langchain_core.tools import StructuredTool

# Create tool wrappers for LangChain/LangGraph
analyze_dataset_quality_tool = StructuredTool.from_function(
    func=analyze_dataset_quality,
    name="analyze_dataset_quality",
    description=analyze_dataset_quality.__doc__
)

data_context_resolver_tool_wrapped = StructuredTool.from_function(
    func=data_context_resolver_tool,
    name="data_context_resolver_tool",
    description=data_context_resolver_tool.__doc__
)

zero_semantics_analyzer_tool_wrapped = StructuredTool.from_function(
    func=zero_semantics_analyzer_tool,
    name="zero_semantics_analyzer_tool",
    description=zero_semantics_analyzer_tool.__doc__
)

missingness_assessment_tool_wrapped = StructuredTool.from_function(
    func=missingness_assessment_tool,
    name="missingness_assessment_tool",
    description=missingness_assessment_tool.__doc__
)

preprocessing_qa_reporter_tool_wrapped = StructuredTool.from_function(
    func=preprocessing_qa_reporter_tool,
    name="preprocessing_qa_reporter_tool",
    description=preprocessing_qa_reporter_tool.__doc__
)

search_cleaning_code_tool = StructuredTool.from_function(
    func=search_cleaning_code,
    name="search_cleaning_code",
    description=search_cleaning_code.__doc__
)

delete_columns_tool_wrapped = StructuredTool.from_function(
    func=delete_columns_tool,
    name="delete_columns_tool",
    description=delete_columns_tool.__doc__
)

impute_missing_tool_wrapped = StructuredTool.from_function(
    func=impute_missing_tool,
    name="impute_missing_tool",
    description=impute_missing_tool.__doc__
)

cap_outliers_tool_wrapped = StructuredTool.from_function(
    func=cap_outliers_tool,
    name="cap_outliers_tool",
    description=cap_outliers_tool.__doc__
)

delete_rows_tool_wrapped = StructuredTool.from_function(
    func=delete_rows_tool,
    name="delete_rows_tool",
    description=delete_rows_tool.__doc__
)

check_power_impact_tool = StructuredTool.from_function(
    func=check_power_impact,
    name="check_power_impact",
    description=check_power_impact.__doc__
)

pqn_normalization_tool_wrapped = StructuredTool.from_function(
    func=pqn_normalization_tool,
    name="pqn_normalization_tool",
    description=pqn_normalization_tool.__doc__
)

log2_transformation_tool_wrapped = StructuredTool.from_function(
    func=log2_transformation_tool,
    name="log2_transformation_tool",
    description=log2_transformation_tool.__doc__
)

auto_scaling_tool_wrapped = StructuredTool.from_function(
    func=auto_scaling_tool,
    name="auto_scaling_tool",
    description=auto_scaling_tool.__doc__
)

# 工具列表，用于LangGraph注册（使用包装器）
DATA_CLEANING_TOOLS = [
    # 数据质量分析工具
    analyze_dataset_quality_tool,
    data_context_resolver_tool_wrapped,
    zero_semantics_analyzer_tool_wrapped,
    missingness_assessment_tool_wrapped,
    preprocessing_qa_reporter_tool_wrapped,
    
    # 代码知识库工具
    search_cleaning_code_tool,
    
    # 数据预处理工具（在特征列和行过滤之前执行）
    pqn_normalization_tool_wrapped,
    log2_transformation_tool_wrapped,
    auto_scaling_tool_wrapped,
    
    # 数据清洗执行工具
    delete_columns_tool_wrapped,
    impute_missing_tool_wrapped,
    cap_outliers_tool_wrapped,
    delete_rows_tool_wrapped,
    
    # 风险评估工具
    check_power_impact_tool
]
