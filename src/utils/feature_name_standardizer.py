"""
Feature Name Standardizer - 动态特征名称标准化工具

目标：通过算法自动抹平全宇宙代谢物名字的格式差异，实现 100% 的泛化性

核心逻辑：
1. 统一转换为大写
2. 去除首尾空格
3. 将所有特殊字符（空格、冒号、斜杠、括号、逗号等）全部替换为单个下划线 _
4. 处理连续下划线（合并为单个下划线）
5. 去除首尾下划线

示例转换：
- "CAR 16:0" → "CAR_16_0"
- "LPC 18:0" → "LPC_18_0"
- "LPC P-18:0 or LPC O-18:1" → "LPC_P_18_0_OR_LPC_O_18_1"
- "Sum_Amino acids, peptides, and analogues" → "SUM_AMINO_ACIDS_PEPTIDES_AND_ANALOGUES"
- "3-Hydroxybutyric acid" → "3_HYDROXYBUTYRIC_ACID"

使用场景：
1. Phase 1 Stage 1.5 特征工程输出时标准化特征名
2. Phase 2/3 特征匹配/验证时标准化输入特征名
3. 任何需要特征名称对齐的场景

作者：MetaboAgent Team
日期：2026-04-29
"""

import re
from typing import List, Dict


def standardize_feature_name(feature_name: str) -> str:
    """
    标准化特征名称，使用动态正则表达式清洗。
    
    转换规则：
    1. 去除首尾空格
    2. 转换为大写
    3. 将所有特殊字符替换为下划线
    4. 合并连续下划线
    5. 去除首尾下划线
    
    Args:
        feature_name: 原始特征名称
    
    Returns:
        标准化后的特征名称
    
    Examples:
        >>> standardize_feature_name("CAR 16:0")
        'CAR_16_0'
        
        >>> standardize_feature_name("LPC P-18:0 or LPC O-18:1")
        'LPC_P_18_0_OR_LPC_O_18_1'
        
        >>> standardize_feature_name("Sum_Amino acids, peptides, and analogues")
        'SUM_AMINO_ACIDS_PEPTIDES_AND_ANALOGUES'
        
        >>> standardize_feature_name("3-Hydroxybutyric acid")
        '3_HYDROXYBUTYRIC_ACID'
        
        >>> standardize_feature_name("  HMDB0000123  ")
        'HMDB0000123'
    """
    if not feature_name or not isinstance(feature_name, str):
        return ""
    
    # Step 1: 去除首尾空格
    cleaned = feature_name.strip()
    
    # Step 2: 转换为大写
    cleaned = cleaned.upper()
    
    # Step 3: 将所有特殊字符替换为下划线
    # 保留：字母、数字、下划线
    # 替换：空格、冒号、斜杠、括号、逗号、连字符、点、分号等
    cleaned = re.sub(r'[^A-Z0-9_]+', '_', cleaned)
    
    # Step 4: 合并连续下划线为单个下划线
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Step 5: 去除首尾下划线
    cleaned = cleaned.strip('_')
    
    return cleaned


def standardize_feature_list(feature_names: List[str]) -> List[str]:
    """
    批量标准化特征名称列表。
    
    Args:
        feature_names: 原始特征名称列表
    
    Returns:
        标准化后的特征名称列表
    
    Example:
        >>> features = ["CAR 16:0", "LPC 18:0", "HMDB0000123"]
        >>> standardize_feature_list(features)
        ['CAR_16_0', 'LPC_18_0', 'HMDB0000123']
    """
    return [standardize_feature_name(name) for name in feature_names]


def create_feature_mapping(original_names: List[str]) -> Dict[str, str]:
    """
    创建原始名称到标准化名称的映射字典。
    
    Args:
        original_names: 原始特征名称列表
    
    Returns:
        映射字典 {原始名称: 标准化名称}
    
    Example:
        >>> names = ["CAR 16:0", "LPC 18:0", "HMDB0000123"]
        >>> create_feature_mapping(names)
        {
            'CAR 16:0': 'CAR_16_0',
            'LPC 18:0': 'LPC_18_0',
            'HMDB0000123': 'HMDB0000123'
        }
    """
    return {name: standardize_feature_name(name) for name in original_names}


def reverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """
    创建反向映射（标准化名称 → 原始名称）。
    
    注意：如果多个原始名称映射到同一个标准化名称，只保留最后一个。
    
    Args:
        mapping: 原始映射字典 {原始名称: 标准化名称}
    
    Returns:
        反向映射字典 {标准化名称: 原始名称}
    
    Example:
        >>> mapping = {'CAR 16:0': 'CAR_16_0', 'LPC 18:0': 'LPC_18_0'}
        >>> reverse_mapping(mapping)
        {'CAR_16_0': 'CAR 16:0', 'LPC_18_0': 'LPC 18:0'}
    """
    return {std_name: orig_name for orig_name, std_name in mapping.items()}


def validate_feature_names(
    feature_names: List[str],
    dataset_columns: List[str],
    auto_standardize: bool = True
) -> Dict[str, any]:
    """
    验证特征名称是否在数据集中存在，支持自动标准化匹配。
    
    Args:
        feature_names: 要验证的特征名称列表
        dataset_columns: 数据集的列名列表
        auto_standardize: 是否自动标准化进行匹配（默认True）
    
    Returns:
        验证结果字典：
        {
            'valid': [有效特征列表],
            'missing': [缺失特征列表],
            'mapping': {原始名称: 数据集中的名称},
            'standardized_valid': [标准化后的有效特征],
            'standardized_missing': [标准化后的缺失特征]
        }
    
    Example:
        >>> features = ["CAR 16:0", "LPC 18:0", "HMDB0000123"]
        >>> columns = ["CAR_16_0", "LPC_18_0", "HMDB0000123", "Group"]
        >>> result = validate_feature_names(features, columns)
        >>> result['valid']
        ['CAR 16:0', 'LPC 18:0', 'HMDB0000123']
    """
    valid = []
    missing = []
    mapping = {}
    
    if auto_standardize:
        # 标准化数据集列名
        std_columns = {standardize_feature_name(col): col for col in dataset_columns}
        
        for feature in feature_names:
            std_feature = standardize_feature_name(feature)
            
            # 先尝试直接匹配
            if feature in dataset_columns:
                valid.append(feature)
                mapping[feature] = feature
            # 再尝试标准化匹配
            elif std_feature in std_columns:
                valid.append(feature)
                mapping[feature] = std_columns[std_feature]
            else:
                missing.append(feature)
    else:
        # 不使用标准化，直接匹配
        for feature in feature_names:
            if feature in dataset_columns:
                valid.append(feature)
                mapping[feature] = feature
            else:
                missing.append(feature)
    
    return {
        'valid': valid,
        'missing': missing,
        'mapping': mapping,
        'standardized_valid': [standardize_feature_name(f) for f in valid],
        'standardized_missing': [standardize_feature_name(f) for f in missing]
    }


# ============================================================================
# 测试和示例
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Feature Name Standardizer - 测试")
    print("="*80)
    
    # 测试用例
    test_cases = [
        "CAR 16:0",
        "CAR 14:0",
        "CAR 18:2",
        "CAR 18:1",
        "CAR 18:0",
        "CAR 20:0",
        "LPC 18:0",
        "LPC 18:1",
        "LPC P-18:0 or LPC O-18:1",
        "LPC P-16:1 or LPC O-16:2",
        "Sum_Amines",
        "Sum_Amino acids, peptides, and analogues",
        "Sum_Purines and purine derivatives",
        "Ratio_HMDB0000269_HMDB0004610",
        "3-Hydroxybutyric acid",
        "gamma-Aminobutyric acid",
        "L-Phenylalanine",
        "HMDB0000123",
        "  HMDB0000456  ",
        "SM 18:1;O2/16:0"
    ]
    
    print("\n单个特征标准化测试:")
    print("-" * 80)
    for original in test_cases:
        standardized = standardize_feature_name(original)
        print(f"{original:50s} → {standardized}")
    
    print("\n" + "="*80)
    print("批量标准化测试:")
    print("-" * 80)
    standardized_list = standardize_feature_list(test_cases)
    print(f"原始特征数: {len(test_cases)}")
    print(f"标准化后: {len(standardized_list)}")
    print(f"唯一特征数: {len(set(standardized_list))}")
    
    print("\n" + "="*80)
    print("特征验证测试:")
    print("-" * 80)
    
    # 模拟数据集列名（已标准化）
    dataset_columns = [
        "SAMPLE_ID",
        "GROUP",
        "CAR_16_0",
        "CAR_14_0",
        "LPC_18_0",
        "HMDB0000123",
        "SUM_AMINES",
        "RATIO_HMDB0000269_HMDB0004610"
    ]
    
    # 要验证的特征（原始格式）
    features_to_validate = [
        "CAR 16:0",
        "CAR 14:0",
        "LPC 18:0",
        "LPC 18:1",  # 不存在
        "HMDB0000123",
        "Sum_Amines",
        "Ratio_HMDB0000269_HMDB0004610",
        "CAR 20:0"  # 不存在
    ]
    
    result = validate_feature_names(features_to_validate, dataset_columns)
    
    print(f"\n有效特征 ({len(result['valid'])}):")
    for feature in result['valid']:
        mapped = result['mapping'][feature]
        print(f"  ✅ {feature:40s} → {mapped}")
    
    print(f"\n缺失特征 ({len(result['missing'])}):")
    for feature in result['missing']:
        std = standardize_feature_name(feature)
        print(f"  ❌ {feature:40s} (标准化: {std})")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
