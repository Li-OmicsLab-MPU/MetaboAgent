"""
智能标签编码器 (Smart Label Encoder)

在编码时自动识别正类，并确保正类编码为 1，符合医学约定和 sklearn 默认假设。

核心功能:
1. 关键词检测: 自动识别医学/疾病相关标签
2. 少数类检测: 在不平衡数据中识别少数类
3. 默认规则: 回退到字母顺序（向后兼容）

Author: MetaboAgent Team
Date: 2026-04-24
"""

from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Optional, Dict, Any


class SmartLabelEncoder(LabelEncoder):
    """
    智能标签编码器
    
    在编码时自动识别正类，并确保正类编码为 1。
    
    识别规则（按优先级）：
    1. 关键词检测：医学/疾病关键词（disease, cd, cancer, patient 等）
    2. 少数类检测：样本量较少的类（样本量差异 >= 2:1）
    3. 默认规则：按字母顺序（向后兼容）
    
    Examples:
        >>> # 示例 1: 医学场景
        >>> le = SmartLabelEncoder()
        >>> y = ['CD', 'Control', 'CD', 'Control']
        >>> y_encoded = le.fit_transform(y)
        >>> print(y_encoded)  # [1, 0, 1, 0] - CD 被编码为 1
        >>> print(le.classes_)  # ['Control', 'CD']
        
        >>> # 示例 2: 不平衡数据
        >>> y = ['A'] * 15 + ['B'] * 45  # A 是少数类
        >>> y_encoded = le.fit_transform(y)
        >>> print(le.classes_)  # ['B', 'A'] - A 被放在索引 1
        
        >>> # 示例 3: 获取正类信息
        >>> info = le.get_positive_class_info()
        >>> print(info['positive_class_name'])  # 'CD' 或 'A'
    """
    
    def __init__(self, positive_class_strategy: str = 'auto'):
        """
        初始化智能标签编码器
        
        Args:
            positive_class_strategy: 正类识别策略
                - 'auto': 自动检测（关键词 + 少数类）【推荐】
                - 'minority': 强制使用少数类
                - 'keyword': 只使用关键词检测
                - 'alphabetical': 使用字母顺序（向后兼容）
        """
        super().__init__()
        self.positive_class_strategy = positive_class_strategy
        self.positive_class_name_: Optional[str] = None
        self.detection_method_: Optional[str] = None
        self.original_order_: Optional[np.ndarray] = None
    
    def fit(self, y):
        """
        拟合编码器，自动识别正类
        
        Args:
            y: 标签数组（array-like）
        
        Returns:
            self
        """
        # 先使用父类的 fit 获取唯一类别（按字母顺序）
        super().fit(y)
        
        # 保存原始顺序
        self.original_order_ = self.classes_.copy()
        
        # 如果不是二分类，使用默认行为
        if len(self.classes_) != 2:
            print(f"⚠️  SmartLabelEncoder: {len(self.classes_)} classes detected, "
                  f"using default alphabetical order")
            self.positive_class_name_ = None
            self.detection_method_ = 'default (not binary)'
            return self
        
        # 如果策略是 alphabetical，直接返回
        if self.positive_class_strategy == 'alphabetical':
            print(f"📋 SmartLabelEncoder: Using alphabetical order (strategy='alphabetical')")
            self.positive_class_name_ = self.classes_[1]
            self.detection_method_ = 'alphabetical'
            return self
        
        # 检测正类
        positive_class_idx = self._detect_positive_class(y)
        
        # 如果正类不是 class 1，需要重新排序
        if positive_class_idx == 0:
            # 交换类别顺序，确保正类在索引 1
            self.classes_ = np.array([self.classes_[1], self.classes_[0]])
            print(f"✅ SmartLabelEncoder: Reordered classes to ensure positive class at index 1")
            print(f"   Original order: {self.original_order_}")
            print(f"   New order: {self.classes_} → {self.classes_[1]} is positive class")
        else:
            print(f"✅ SmartLabelEncoder: Positive class already at index 1")
            print(f"   Order: {self.classes_} → {self.classes_[1]} is positive class")
        
        self.positive_class_name_ = self.classes_[1]
        
        return self
    
    def _detect_positive_class(self, y) -> int:
        """
        检测正类
        
        Args:
            y: 标签数组
        
        Returns:
            int: 正类在 self.classes_ 中的索引 (0 or 1)
        """
        # 规则 1: 关键词检测（最高优先级）
        if self.positive_class_strategy in ['auto', 'keyword']:
            disease_keywords = [
                'disease', 'case', 'patient', 'cd', 'cancer', 
                'tumor', 'positive', 'sick', 'ill', 'affected',
                'malignant', 'pathological', 'abnormal', 'disorder',
                'syndrome', 'infection', 'inflammatory'
            ]
            
            for idx, label in enumerate(self.classes_):
                label_lower = str(label).lower()
                if any(keyword in label_lower for keyword in disease_keywords):
                    self.detection_method_ = f'keyword ({label})'
                    print(f"  🎯 Detected positive class by keyword: '{label}' (index {idx})")
                    return idx
        
        # 规则 2: 少数类检测（中等优先级）
        if self.positive_class_strategy in ['auto', 'minority']:
            # 计算每个类别的样本数
            unique, counts = np.unique(y, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            # 获取每个 class 的样本数（按 self.classes_ 的顺序）
            counts_ordered = [class_counts[cls] for cls in self.classes_]
            
            # 找到少数类
            minority_idx = np.argmin(counts_ordered)
            
            # 只有在样本量差异显著时才使用（至少 2:1）
            ratio = max(counts_ordered) / (min(counts_ordered) + 1e-10)
            if ratio >= 2.0:
                self.detection_method_ = f'minority (ratio={ratio:.1f}:1)'
                print(f"  🎯 Detected positive class by minority: '{self.classes_[minority_idx]}' "
                      f"(index {minority_idx}, {counts_ordered[minority_idx]} samples vs "
                      f"{counts_ordered[1-minority_idx]} samples, ratio={ratio:.1f}:1)")
                return minority_idx
        
        # 规则 3: 默认使用 class 1（字母顺序中的第二个）
        self.detection_method_ = 'default (alphabetical)'
        print(f"  ⚠️  Cannot determine positive class, using default: "
              f"class 1 ('{self.classes_[1]}')")
        return 1
    
    def transform(self, y):
        """
        转换标签
        
        Args:
            y: 标签数组
        
        Returns:
            编码后的数组
        """
        return super().transform(y)
    
    def fit_transform(self, y):
        """
        拟合并转换标签
        
        Args:
            y: 标签数组
        
        Returns:
            编码后的数组
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        """
        反转换标签
        
        Args:
            y: 编码数组
        
        Returns:
            原始标签数组
        """
        return super().inverse_transform(y)
    
    def get_positive_class_info(self) -> Dict[str, Any]:
        """
        获取正类信息
        
        Returns:
            dict: 包含正类名称、索引、检测方法等信息
                - positive_class_name: 正类名称
                - positive_class_index: 正类索引（总是 1）
                - detection_method: 检测方法
                - classes: 类别列表
                - original_order: 原始字母顺序
                - strategy: 使用的策略
        """
        return {
            'positive_class_name': self.positive_class_name_,
            'positive_class_index': 1 if self.positive_class_name_ is not None else None,
            'detection_method': self.detection_method_,
            'classes': self.classes_.tolist(),
            'original_order': self.original_order_.tolist() if self.original_order_ is not None else None,
            'strategy': self.positive_class_strategy
        }


def smart_label_encode(y, strategy: str = 'auto'):
    """
    智能标签编码函数（便捷接口）
    
    Args:
        y: 标签数组
        strategy: 正类识别策略
            - 'auto': 自动检测（关键词 + 少数类）【推荐】
            - 'minority': 强制使用少数类
            - 'keyword': 只使用关键词检测
            - 'alphabetical': 使用字母顺序（向后兼容）
    
    Returns:
        tuple: (y_encoded, label_encoder, positive_class_info)
            - y_encoded: 编码后的数组
            - label_encoder: SmartLabelEncoder 对象
            - positive_class_info: 正类信息字典
    
    Examples:
        >>> y = ['CD', 'Control', 'CD', 'Control']
        >>> y_encoded, le, info = smart_label_encode(y)
        >>> print(y_encoded)  # [1, 0, 1, 0]
        >>> print(info['positive_class_name'])  # 'CD'
    """
    le = SmartLabelEncoder(positive_class_strategy=strategy)
    y_encoded = le.fit_transform(y)
    positive_class_info = le.get_positive_class_info()
    
    return y_encoded, le, positive_class_info
