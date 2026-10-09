from __future__ import annotations

from typing import Any, Dict

from src.agents.phase3.styles.journal_registry import get_journal_style
from src.agents.phase3.styles.style_types import RenderConfig


def _decide_label_strategy(n_labels: int) -> str:
    if n_labels <= 8:
        return "show_all"
    if n_labels <= 15:
        return "show_final_panel_only"
    return "show_top_n_by_score"


def _scale_font_size(n_rows: int, max_label_length: int) -> float:
    row_penalty = max(0, n_rows - 12) * 0.08
    label_penalty = max(0, max_label_length - 20) * 0.03
    return max(5.8, 8.6 - row_penalty - label_penalty)


def _figure_height_for_heatmap(n_rows: int) -> float:
    return max(7.2, min(0.40 * n_rows + 2.6, 13.8))


def build_render_config(
    *,
    journal: str,
    figure_type: str,
    data_summary: Dict[str, Any],
    policy: Dict[str, Any] | None = None,
) -> RenderConfig:
    style = get_journal_style(journal)
    policy = dict(policy or {})
    n_rows = int(data_summary.get("n_rows", 0) or 0)
    n_labels = int(data_summary.get("n_labels", n_rows) or n_rows)
    max_label_length = int(data_summary.get("max_label_length", 20) or 20)
    internal_retry_limit = int(policy.get("internal_retry_limit", 1) or 1)

    if figure_type == "stability_landscape":
        return RenderConfig(
            style=style,
            figure_type=figure_type,
            figure_width=12.0,
            figure_height=8.2,
            label_strategy=_decide_label_strategy(n_labels),
            annotate_top_n=min(int(policy.get("annotate_top_n_max", 15) or 15), max(n_labels, 1)),
            font_size_feature=_scale_font_size(n_rows=max(n_rows, n_labels), max_label_length=max_label_length),
            legend_position="outside_right",
            legend_outside=True,
            bubble_size_min=70.0,
            bubble_size_max=480.0,
            margin_left=0.10,
            margin_right=0.22,
            margin_top=0.08,
            margin_bottom=0.10,
            internal_retry_limit=internal_retry_limit,
            extras={
                "zone_label_alpha": 0.70,
                "show_size_legend": bool(policy.get("show_size_legend", True)),
            },
        )

    if figure_type == "method_feature_heatmap":
        focused_threshold = int(policy.get("focused_mode_threshold", 18) or 18)
        focused_rejected_n = int(policy.get("focused_rejected_count", 6) or 6)
        return RenderConfig(
            style=style,
            figure_type=figure_type,
            figure_width=11.8,
            figure_height=_figure_height_for_heatmap(n_rows),
            label_strategy="focused_mode" if n_rows > focused_threshold else "show_all",
            annotate_top_n=0,
            font_size_feature=_scale_font_size(n_rows=n_rows, max_label_length=max_label_length),
            xtick_rotation=35.0,
            ytick_max_len=int(policy.get("max_feature_label_length", 28) or 28),
            legend_position="outside_right",
            legend_outside=True,
            colorbar_shrink=0.72,
            margin_left=0.24,
            margin_right=0.14,
            margin_top=0.08,
            margin_bottom=0.10,
            focused_mode=n_rows > focused_threshold,
            focused_rejected_n=focused_rejected_n,
            internal_retry_limit=internal_retry_limit,
            extras={
                "show_method_group_strip": True,
                "show_status_strip": True,
            },
        )

    return RenderConfig(
        style=style,
        figure_type=figure_type,
        figure_width=10.0,
        figure_height=7.0,
        internal_retry_limit=internal_retry_limit,
    )
