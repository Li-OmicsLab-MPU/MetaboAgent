"""
Phase 3 可视化工具统一导出入口。

这个模块负责收口 Phase 3 会直接调用的绘图函数，避免业务代码在
`viz_tools.py` 和 `viz_tools_artifact_based.py` 之间来回跳转查找入口。
"""

from .viz_tools import (
    plot_calibration,
    plot_dca,
    plot_final_roc,
    plot_pareto_trajectory,
    plot_radar_comparison,
    plot_rcs_curves as plot_rcs_curves_legacy,
    plot_shap,
    plot_stats_scatter,
    plot_threshold_performance,
    plot_upset,
)
from .autogluon_roc_r import plot_autogluon_roc
from .phase0_prior_atlas_r import plot_phase0_prior_evidence_atlas
from .phase1_stability_r import plot_stability_landscape
from .phase1_method_support_r import plot_method_feature_heatmap
from .phase3_clinical_curves_r import (
    plot_calibration_r,
    plot_dca_r,
    plot_final_holdout_dca_r,
    plot_final_holdout_roc_r,
    plot_final_roc_r,
)
from .phase3_rcs_r import plot_rcs_curves_r
from .phase1_qc import (
    plot_phase1_final_panel_correlation_heatmap,
)
from .phase3_selection_baseline_composite import (
    plot_phase1_selection_baseline_composite,
    plot_phase2_clinical_validation_composite,
    plot_phase2_radar_validation_composite,
    plot_phase3_shap_interpretation_composite,
)
from .phase2_objective_shift_r import plot_phase2_objective_shift_summary
from .phase2_incremental_value_r import plot_phase2_incremental_value_summary
from .viz_tools_artifact_based import (
    plot_dca_from_artifact,
    plot_final_roc_from_artifact,
    plot_stats_scatter_from_artifact,
)

plot_rcs_curves = plot_rcs_curves_r


PLOT_FUNCTIONS = {
    'plot_stats_scatter': plot_stats_scatter,
    'plot_upset': plot_upset,
    'plot_phase0_prior_evidence_atlas': plot_phase0_prior_evidence_atlas,
    'plot_stability_landscape': plot_stability_landscape,
    'plot_method_feature_heatmap': plot_method_feature_heatmap,
    'plot_phase1_final_panel_correlation_heatmap': plot_phase1_final_panel_correlation_heatmap,
    'plot_autogluon_roc': plot_autogluon_roc,
    'plot_phase1_selection_baseline_composite': plot_phase1_selection_baseline_composite,
    'plot_phase2_clinical_validation_composite': plot_phase2_clinical_validation_composite,
    'plot_phase2_radar_validation_composite': plot_phase2_radar_validation_composite,
    'plot_phase3_shap_interpretation_composite': plot_phase3_shap_interpretation_composite,
    'plot_phase2_objective_shift_summary': plot_phase2_objective_shift_summary,
    'plot_phase2_incremental_value_summary': plot_phase2_incremental_value_summary,
    'plot_calibration': plot_calibration_r,
    'plot_pareto_trajectory': plot_pareto_trajectory,
    'plot_radar_comparison': plot_radar_comparison,
    'plot_final_roc': plot_final_roc_r,
    'plot_final_holdout_roc': plot_final_holdout_roc_r,
    'plot_final_holdout_dca': plot_final_holdout_dca_r,
    'plot_dca': plot_dca_r,
    'plot_threshold_performance': plot_threshold_performance,
    'plot_shap': plot_shap,
    'plot_rcs_curves': plot_rcs_curves_r,
}

ARTIFACT_PLOT_FUNCTIONS = {
    'plot_stats_scatter_from_artifact': plot_stats_scatter_from_artifact,
    'plot_final_roc_from_artifact': plot_final_roc_from_artifact,
    'plot_dca_from_artifact': plot_dca_from_artifact,
}

__all__ = [
    'PLOT_FUNCTIONS',
    'ARTIFACT_PLOT_FUNCTIONS',
    'plot_stats_scatter',
    'plot_upset',
    'plot_phase0_prior_evidence_atlas',
    'plot_stability_landscape',
    'plot_method_feature_heatmap',
    'plot_phase1_final_panel_correlation_heatmap',
    'plot_autogluon_roc',
    'plot_phase1_selection_baseline_composite',
    'plot_phase2_clinical_validation_composite',
    'plot_phase2_radar_validation_composite',
    'plot_phase3_shap_interpretation_composite',
    'plot_phase2_objective_shift_summary',
    'plot_phase2_incremental_value_summary',
    'plot_calibration',
    'plot_calibration_r',
    'plot_pareto_trajectory',
    'plot_radar_comparison',
    'plot_final_roc',
    'plot_final_roc_r',
    'plot_final_holdout_roc_r',
    'plot_final_holdout_dca_r',
    'plot_dca',
    'plot_dca_r',
    'plot_threshold_performance',
    'plot_shap',
    'plot_rcs_curves',
    'plot_rcs_curves_r',
    'plot_rcs_curves_legacy',
    'plot_stats_scatter_from_artifact',
    'plot_final_roc_from_artifact',
    'plot_dca_from_artifact',
]
