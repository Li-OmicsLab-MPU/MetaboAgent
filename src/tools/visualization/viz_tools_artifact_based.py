"""
Phase 3: Artifact-Based Visualization Functions

这些函数完全依赖 Phase 1/2 导出的 Artifacts，不进行任何模型训练和预测计算。
确保 100% 数值一致性。

核心设计理念:
- Phase 1/2: 计算 + 导出 Artifacts
- Phase 3: 只读取 Artifacts + 画图
- 目标: 100% 数值一致性

Author: MetaboAgent Team
Date: 2026-04-24
"""

import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


# ============================================================================
# Artifact-Based Visualization Functions
# ============================================================================

def plot_final_roc_from_artifact(
    predictions_artifact_path: str,
    save_path: str = "output/figures/fig4a_final_roc.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None
) -> Tuple[str, Dict]:
    """
    从 Artifact 绘制最终 ROC 曲线（不训练模型！）
    
    这个函数直接读取 Phase 2 导出的交叉验证预测概率，绘制 ROC 曲线。
    完全不进行任何模型训练和预测计算，确保数值与 Phase 2 100% 一致。
    
    核心视觉元素:
    1. ROC 曲线: 基于 OOF 预测概率计算的 ROC（深红色粗实线）
    2. AUC 标注: 显著标注 AUC 值
    3. 对角线参考: 随机猜测的基线 (AUC=0.5)
    
    Args:
        predictions_artifact_path: CV 预测概率 Artifact 文件路径
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> path, kwargs = plot_final_roc_from_artifact(
        ...     predictions_artifact_path='output/artifacts/winner_cv_predictions.csv',
        ...     save_path='output/figures/fig4a_final_roc.pdf'
        ... )
        >>> print(f"Saved to: {path}")
    """
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'legend_loc': 'lower right',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11,
        'bbox_to_anchor': None,
        'auc_text_x': 0.6,
        'auc_text_y': 0.2
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_final_roc_from_artifact using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 1. 读取 Artifact（不训练模型！）
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: Final ROC Curve (Artifact-Based, No Model Training)")
    print("="*80)
    print(f"Reading artifact from: {predictions_artifact_path}")
    
    if not os.path.exists(predictions_artifact_path):
        raise FileNotFoundError(f"Artifact not found: {predictions_artifact_path}")
    
    df = pd.read_csv(predictions_artifact_path)
    
    print(f"✅ Artifact loaded successfully")
    print(f"   Total samples: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # ========================================================================
    # 2. 计算 ROC 曲线（不训练模型！）
    # ========================================================================
    print(f"\n[Step 1] Computing ROC curve from artifact predictions...")
    
    # 提取真实标签和预测概率
    # LabelEncoder sorts labels alphabetically: ['CD', 'Control']
    # So class0 = 'CD', class1 = 'Control'
    # We want CD as the positive class, so we use pred_proba_class0
    
    y_true = (df['true_label'] == 'CD').astype(int)
    y_pred_proba = df['pred_proba_class0'].values
    
    print(f"   Positive class: CD")
    print(f"   Using pred_proba_class0 for CD predictions")
    
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"✅ ROC curve computed from artifact (no model training!)")
    print(f"   AUC: {roc_auc:.4f}")
    print(f"   FPR points: {len(fpr)}")
    print(f"   TPR points: {len(tpr)}")
    
    # ========================================================================
    # 3. 绘制 ROC 曲线
    # ========================================================================
    print(f"\n[Step 2] Plotting ROC curve...")
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制 ROC 曲线
    ax.plot(
        fpr, tpr,
        color='darkred',
        linewidth=2.5,
        label=f'ROC Curve (AUC = {roc_auc:.3f})'
    )
    
    # 绘制对角线参考
    ax.plot(
        [0, 1], [0, 1],
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Random Guess (AUC = 0.5)'
    )
    
    # 设置坐标轴
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel(
        'False Positive Rate (1 - Specificity)',
        fontsize=final_layout_kwargs['label_fontsize'],
        fontweight='bold'
    )
    ax.set_ylabel(
        'True Positive Rate (Sensitivity)',
        fontsize=final_layout_kwargs['label_fontsize'],
        fontweight='bold'
    )
    ax.set_title(
        'ROC Curve - Winner Panel Performance',
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        pad=20
    )
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # 添加图例
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        shadow=True,
        fancybox=True
    )
    
    # 添加 AUC 文本标注
    ax.text(
        final_layout_kwargs['auc_text_x'],
        final_layout_kwargs['auc_text_y'],
        f'AUC = {roc_auc:.4f}',
        fontsize=final_layout_kwargs['fontsize'] + 2,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3)
    )
    
    # 添加水印（标注这是基于 Artifact 的图表）
    ax.text(
        0.98, 0.02,
        'Artifact-Based (No Model Training)',
        fontsize=8,
        color='gray',
        alpha=0.5,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure saved to: {save_path}")
    print(f"   File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


def plot_dca_from_artifact(
    predictions_artifact_path: str,
    save_path: str = "output/figures/fig4b_dca.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None
) -> Tuple[str, Dict]:
    """
    从 Artifact 绘制决策曲线分析（不训练模型！）
    
    这个函数直接读取 Phase 2 导出的交叉验证预测概率，绘制 DCA 曲线。
    完全不进行任何模型训练和预测计算，确保数值与 Phase 2 100% 一致。
    
    核心视觉元素:
    1. 模型净收益曲线: 基于 OOF 预测概率计算的净收益
    2. 全部治疗基线: 假设所有患者都接受治疗
    3. 全部不治疗基线: 假设所有患者都不接受治疗
    
    Args:
        predictions_artifact_path: CV 预测概率 Artifact 文件路径
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> path, kwargs = plot_dca_from_artifact(
        ...     predictions_artifact_path='output/artifacts/winner_cv_predictions.csv',
        ...     save_path='output/figures/fig4b_dca.pdf'
        ... )
        >>> print(f"Saved to: {path}")
    """
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'legend_loc': 'upper right',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_dca_from_artifact using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 1. 读取 Artifact（不训练模型！）
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: Decision Curve Analysis (Artifact-Based, No Model Training)")
    print("="*80)
    print(f"Reading artifact from: {predictions_artifact_path}")
    
    if not os.path.exists(predictions_artifact_path):
        raise FileNotFoundError(f"Artifact not found: {predictions_artifact_path}")
    
    df = pd.read_csv(predictions_artifact_path)
    
    print(f"✅ Artifact loaded successfully")
    print(f"   Total samples: {len(df)}")
    
    # ========================================================================
    # 2. 计算 DCA（不训练模型！）
    # ========================================================================
    print(f"\n[Step 1] Computing DCA from artifact predictions...")
    
    # 提取真实标签和预测概率
    # LabelEncoder sorts labels alphabetically: ['CD', 'Control']
    # So class0 = 'CD', class1 = 'Control'
    # We want CD as the positive class, so we use pred_proba_class0
    
    y_true = (df['true_label'] == 'CD').astype(int)
    y_pred_proba = df['pred_proba_class0'].values
    
    print(f"   Positive class: CD")
    print(f"   Using pred_proba_class0 for CD predictions")
    
    # 阈值概率范围
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # 计算净收益
    net_benefits_model = []
    net_benefits_all = []
    net_benefits_none = []
    
    n = len(y_true)
    
    for pt in thresholds:
        # 模型策略: 根据预测概率决策
        y_pred = (y_pred_proba >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        net_benefit_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits_model.append(net_benefit_model)
        
        # 全部治疗策略
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        net_benefit_all = (tp_all / n) - (fp_all / n) * (pt / (1 - pt))
        net_benefits_all.append(net_benefit_all)
        
        # 全部不治疗策略
        net_benefits_none.append(0)
    
    print(f"✅ DCA computed from artifact (no model training!)")
    print(f"   Threshold range: [{thresholds[0]:.2f}, {thresholds[-1]:.2f}]")
    print(f"   Net benefit range: [{min(net_benefits_model):.4f}, {max(net_benefits_model):.4f}]")
    
    # ========================================================================
    # 3. 绘制 DCA 曲线
    # ========================================================================
    print(f"\n[Step 2] Plotting DCA curve...")
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制曲线
    ax.plot(
        thresholds, net_benefits_model,
        color='darkred',
        linewidth=2.5,
        label='Winner Panel Model'
    )
    ax.plot(
        thresholds, net_benefits_all,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Treat All'
    )
    ax.plot(
        thresholds, net_benefits_none,
        color='black',
        linestyle=':',
        linewidth=1.5,
        label='Treat None'
    )
    
    # 设置坐标轴
    ax.set_xlim([0, 1])
    ax.set_ylim([min(net_benefits_model + net_benefits_all) - 0.05, max(net_benefits_model) + 0.05])
    ax.set_xlabel(
        'Threshold Probability',
        fontsize=final_layout_kwargs['label_fontsize'],
        fontweight='bold'
    )
    ax.set_ylabel(
        'Net Benefit',
        fontsize=final_layout_kwargs['label_fontsize'],
        fontweight='bold'
    )
    ax.set_title(
        'Decision Curve Analysis - Clinical Utility',
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        pad=20
    )
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # 添加图例
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        shadow=True,
        fancybox=True
    )
    
    # 添加水印
    ax.text(
        0.98, 0.02,
        'Artifact-Based (No Model Training)',
        fontsize=8,
        color='gray',
        alpha=0.5,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure saved to: {save_path}")
    print(f"   File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


def plot_stats_scatter_from_artifact(
    stats_artifact_path: str,
    method: str = 'pca',
    save_path: str = "output/figures/fig1a_scatter.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None
) -> Tuple[str, Dict]:
    """
    从 Artifact 绘制统计降维散点图（不重新计算降维！）
    
    这个函数直接读取 Phase 1 导出的降维坐标，绘制散点图。
    完全不进行任何降维计算，确保数值与 Phase 1 100% 一致。
    
    Args:
        stats_artifact_path: 统计降维 Artifact 文件路径
        method: 降维方法 ('pca' 或 'plsda')
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> path, kwargs = plot_stats_scatter_from_artifact(
        ...     stats_artifact_path='output/artifacts/phase1_stats_artifacts.npz',
        ...     method='pca',
        ...     save_path='output/figures/fig1a_scatter.pdf'
        ... )
        >>> print(f"Saved to: {path}")
    """
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'legend_loc': 'best',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_stats_scatter_from_artifact using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 1. 读取 Artifact（不重新计算降维！）
    # ========================================================================
    print("\n" + "="*80)
    print(f"Phase 3: {method.upper()} Scatter Plot (Artifact-Based, No Recomputation)")
    print("="*80)
    print(f"Reading artifact from: {stats_artifact_path}")
    
    if not os.path.exists(stats_artifact_path):
        raise FileNotFoundError(f"Artifact not found: {stats_artifact_path}")
    
    artifact = np.load(stats_artifact_path)
    
    # 根据方法选择坐标
    if method.lower() == 'pca':
        coords = artifact['pca_coords']
        variance_ratio = artifact['pca_variance_ratio']
        comp1_label = f'PC1 ({variance_ratio[0]:.1%})'
        comp2_label = f'PC2 ({variance_ratio[1]:.1%})'
        title = 'PCA Score Plot'
    elif method.lower() == 'plsda':
        coords = artifact['plsda_coords']
        variance_ratio = artifact['plsda_variance_ratio']
        comp1_label = f'LV1 ({variance_ratio[0]:.1%})'
        comp2_label = f'LV2 ({variance_ratio[1]:.1%})'
        title = 'PLS-DA Score Plot'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'plsda'")
    
    labels = artifact['labels']
    
    print(f"✅ Artifact loaded successfully")
    print(f"   Method: {method.upper()}")
    print(f"   Coordinates shape: {coords.shape}")
    print(f"   Variance explained: {variance_ratio}")
    
    # ========================================================================
    # 2. 绘制散点图
    # ========================================================================
    print(f"\n[Step 1] Plotting scatter plot...")
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 为每个类别绘制散点
    unique_labels = np.unique(labels)
    colors = ['#E74C3C', '#3498DB']  # 红色和蓝色
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors[i],
            label=label,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    # 设置坐标轴
    ax.set_xlabel(comp1_label, fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold')
    ax.set_ylabel(comp2_label, fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold')
    ax.set_title(title, fontsize=final_layout_kwargs['title_fontsize'], fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # 添加图例
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        shadow=True,
        fancybox=True
    )
    
    # 添加水印
    ax.text(
        0.98, 0.02,
        'Artifact-Based (No Recomputation)',
        fontsize=8,
        color='gray',
        alpha=0.5,
        ha='right',
        va='bottom',
        transform=ax.transAxes
    )
    
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Figure saved to: {save_path}")
    print(f"   File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


if __name__ == '__main__':
    print("="*80)
    print("Artifact-Based Visualization Functions")
    print("="*80)
    print()
    print("This module provides visualization functions that read from artifacts")
    print("instead of recomputing predictions. This ensures 100% numerical consistency.")
    print()
    print("Available functions:")
    print("  1. plot_final_roc_from_artifact() - ROC curve from CV predictions")
    print("  2. plot_dca_from_artifact() - DCA from CV predictions")
    print("  3. plot_stats_scatter_from_artifact() - PCA/PLS-DA from Phase 1 artifacts")
    print()
    print("="*80)
