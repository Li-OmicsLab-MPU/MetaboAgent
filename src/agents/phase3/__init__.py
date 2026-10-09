"""
Phase 3 统一导出入口。

这里集中导出 Phase 3 主流程、路由器、审查器和评估桥接工具，减少业务脚本
直接跨文件引用内部实现细节。
"""

from .evaluator_bridge import ClinicalEvaluator, normalize_scores_for_radar

__all__ = [
    'ClinicalEvaluator',
    'normalize_scores_for_radar',
    'Phase3Pipeline',
    'VisualAuditor',
    'FigureRouter',
]


def __getattr__(name):
    if name == 'Phase3Pipeline':
        from .generate_final_report import Phase3Pipeline
        return Phase3Pipeline
    if name == 'VisualAuditor':
        from .layout_agent import VisualAuditor
        return VisualAuditor
    if name == 'FigureRouter':
        from .router import FigureRouter
        return FigureRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
