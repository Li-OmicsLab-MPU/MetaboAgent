from __future__ import annotations

from typing import Dict

from .style_types import StylePreset


GUT_PALETTE = ["#D94442", "#62B0E5", "#35BCB8", "#947EC4", "#F5A8B8", "#E2C372", "#BFC4CC"]
SCIENCE_ADVANCES_PALETTE = ["#EC5555", "#F8941D", "#FAC63D", "#63BF7B", "#38B6C3", "#5186F0", "#AA84DF", "#B3B3B3"]
CELL_METABOLISM_PALETTE = ["#6E3B33", "#6D9185", "#7B8F5A", "#8A7C95", "#B46857", "#9C7450", "#C8B27B", "#A39B95"]
NATURE_PALETTE = ["#202754", "#3D85C6", "#19AA9D", "#89C94A", "#E96362", "#FCC248", "#9B7FC4", "#BBBDC0"]


JOURNAL_STYLE_REGISTRY: Dict[str, StylePreset] = {
    "gut": StylePreset(
        key="gut",
        palette=GUT_PALETTE,
        status_colors={
            "mandatory": "#E2C372",
            "stable_core": "#62B0E5",
            "completed": "#35BCB8",
            "rejected": "#BFC4CC",
        },
        font_family="Arial",
        heatmap_cmap="gut_diverging",
        grid_policy="light",
        spine_policy="minimal",
        export_dpi=300,
    ),
    "science_advances": StylePreset(
        key="science_advances",
        palette=SCIENCE_ADVANCES_PALETTE,
        status_colors={
            "mandatory": "#FAC63D",
            "stable_core": "#5186F0",
            "completed": "#63BF7B",
            "rejected": "#B3B3B3",
        },
        font_family="Arial",
        line_width=1.25,
        heatmap_cmap="science_advances_diverging",
        grid_policy="light",
        spine_policy="minimal",
        export_dpi=300,
    ),
    "cell_metabolism": StylePreset(
        key="cell_metabolism",
        palette=CELL_METABOLISM_PALETTE,
        status_colors={
            "mandatory": "#C8B27B",
            "stable_core": "#6D9185",
            "completed": "#7B8F5A",
            "rejected": "#A39B95",
        },
        font_family="Arial",
        heatmap_cmap="cell_metabolism_diverging",
        grid_policy="light",
        spine_policy="minimal",
        export_dpi=300,
    ),
    "nature": StylePreset(
        key="nature",
        palette=NATURE_PALETTE,
        status_colors={
            "mandatory": "#FCC248",
            "stable_core": "#3D85C6",
            "completed": "#19AA9D",
            "rejected": "#BBBDC0",
        },
        font_family="Times New Roman",
        line_width=1.2,
        heatmap_cmap="nature_diverging",
        grid_policy="light",
        spine_policy="minimal",
        export_dpi=300,
    ),
    "nature_minimal": StylePreset(
        key="nature",
        palette=NATURE_PALETTE,
        status_colors={
            "mandatory": "#FCC248",
            "stable_core": "#3D85C6",
            "completed": "#19AA9D",
            "rejected": "#BBBDC0",
        },
        font_family="Times New Roman",
        base_font_size=8.5,
        title_font_size=11.0,
        label_font_size=9.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        line_width=1.2,
        marker_edge_width=0.8,
        heatmap_cmap="nature_diverging",
        zero_value_color="#FFFFFF",
        grid_policy="light",
        spine_policy="minimal",
        threshold_line_style="dashed",
        panel_label_style="bold_caps",
        export_dpi=300,
    ),
    "lancet_clinical": StylePreset(
        key="lancet_clinical",
        palette=["#153B6D", "#3F6FB5", "#8FB9E3", "#C64E4E", "#6A994E"],
        status_colors={
            "mandatory": "#B7791F",
            "stable_core": "#245FA6",
            "completed": "#5BAE8B",
            "rejected": "#D8DDE6",
        },
        font_family="Arial",
        base_font_size=8.5,
        title_font_size=11.0,
        label_font_size=9.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        line_width=1.25,
        marker_edge_width=0.8,
        heatmap_cmap="custom_blues",
        zero_value_color="#FFFFFF",
        grid_policy="clinical",
        spine_policy="minimal",
        threshold_line_style="dashed",
        panel_label_style="bold_caps",
        export_dpi=300,
    ),
    "cell_systems": StylePreset(
        key="cell_systems",
        palette=["#3B4CC0", "#688AE8", "#8DB0FE", "#F7B267", "#D1495B"],
        status_colors={
            "mandatory": "#D97706",
            "stable_core": "#355FB8",
            "completed": "#4FB286",
            "rejected": "#D4D8DF",
        },
        font_family="Arial",
        base_font_size=8.5,
        title_font_size=11.0,
        label_font_size=9.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        line_width=1.2,
        marker_edge_width=0.8,
        heatmap_cmap="custom_blues",
        zero_value_color="#FFFFFF",
        grid_policy="light",
        spine_policy="minimal",
        threshold_line_style="dashed",
        panel_label_style="bold_caps",
        export_dpi=300,
    ),
    "nejm_mono": StylePreset(
        key="nejm_mono",
        palette=["#111827", "#4B5563", "#9CA3AF", "#D1D5DB", "#374151"],
        status_colors={
            "mandatory": "#6B7280",
            "stable_core": "#1F2937",
            "completed": "#4B5563",
            "rejected": "#D1D5DB",
        },
        font_family="Arial",
        base_font_size=8.5,
        title_font_size=11.0,
        label_font_size=9.0,
        tick_font_size=8.0,
        legend_font_size=8.0,
        line_width=1.2,
        marker_edge_width=0.8,
        heatmap_cmap="custom_blues",
        zero_value_color="#FFFFFF",
        grid_policy="mono",
        spine_policy="minimal",
        threshold_line_style="dashed",
        panel_label_style="bold_caps",
        export_dpi=300,
    ),
}


def get_journal_style(style_key: str | None = None) -> StylePreset:
    aliases = {
        "science": "science_advances",
        "science advances": "science_advances",
        "science-advances": "science_advances",
        "cell": "cell_metabolism",
        "cell metabolism": "cell_metabolism",
        "cell-metabolism": "cell_metabolism",
        "nature_minimal": "nature",
    }
    key = str(style_key or "nature").strip().lower()
    key = aliases.get(key, key)
    return JOURNAL_STYLE_REGISTRY.get(key, JOURNAL_STYLE_REGISTRY["nature"])
