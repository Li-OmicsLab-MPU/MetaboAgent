from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List


@dataclass(frozen=True)
class StylePreset:
    key: str
    palette: List[str]
    status_colors: Dict[str, str]
    font_family: str = "Arial"
    base_font_size: float = 8.5
    title_font_size: float = 11.0
    label_font_size: float = 9.0
    tick_font_size: float = 8.0
    legend_font_size: float = 8.0
    line_width: float = 1.2
    marker_edge_width: float = 0.8
    heatmap_cmap: str = "custom_blues"
    zero_value_color: str = "#FFFFFF"
    grid_policy: str = "light"
    spine_policy: str = "minimal"
    threshold_line_style: str = "dashed"
    panel_label_style: str = "bold_caps"
    export_dpi: int = 300


@dataclass(frozen=True)
class RenderConfig:
    style: StylePreset
    figure_type: str
    figure_width: float
    figure_height: float
    label_strategy: str = "show_top_n_by_score"
    annotate_top_n: int = 12
    font_size_feature: float = 8.0
    xtick_rotation: float = 0.0
    ytick_max_len: int = 28
    legend_position: str = "outside_right"
    legend_outside: bool = True
    bubble_size_min: float = 70.0
    bubble_size_max: float = 420.0
    colorbar_shrink: float = 0.72
    margin_left: float = 0.18
    margin_right: float = 0.16
    margin_top: float = 0.10
    margin_bottom: float = 0.08
    focused_mode: bool = False
    focused_rejected_n: int = 6
    internal_retry_limit: int = 1
    extras: Dict[str, Any] = field(default_factory=dict)

    def patch(self, **updates: Any) -> "RenderConfig":
        current = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        extras = dict(current.get("extras") or {})
        recognized = {}
        for key, value in updates.items():
            if key in current:
                recognized[key] = value
            else:
                extras[key] = value
        recognized["extras"] = extras
        return replace(self, **recognized)


@dataclass
class RenderResult:
    save_path: str
    figure_type: str
    render_config: RenderConfig
    self_check_passed: bool
    self_check_issues: List[str]
    retry_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
