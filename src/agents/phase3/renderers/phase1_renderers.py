from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from src.agents.phase3.renderers.base_renderer import BaseSmartRenderer
from src.agents.phase3.styles.style_types import RenderConfig
from src.tools.visualization.viz_tools import (
    ADJUST_TEXT_AVAILABLE,
    _STABILITY_METHOD_LABELS,
    _build_stability_plot_dataframe,
    _format_feature_axis_label,
    _load_json_report,
    _select_focused_heatmap_rows,
)

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional dependency
    adjust_text = None


def load_phase1_stability_plot_data(
    *,
    stability_scores_path: str,
    stability_summary_path: str,
    feature_provenance_path: str = "",
) -> Dict[str, Any]:
    stability_summary = _load_json_report(stability_summary_path)
    stability_scores_payload = _load_json_report(stability_scores_path)
    if not stability_summary or not stability_scores_payload:
        raise FileNotFoundError("Stability summary or stability score artifact is missing.")

    df = _build_stability_plot_dataframe(
        stability_summary=stability_summary,
        stability_scores_payload=stability_scores_payload,
        feature_provenance_path=feature_provenance_path,
    )
    plot_df = _select_focused_heatmap_rows(df, rejected_n=6)
    data_summary = {
        "n_rows": int(len(df)),
        "n_labels": int(df["is_final_panel"].sum()),
        "max_label_length": int(df["display_name"].astype(str).map(len).max()),
        "n_methods": int(sum(1 for method in _STABILITY_METHOD_LABELS if method in df.columns)),
        "n_focused_rows": int(len(plot_df)),
    }
    return {
        "stability_df": df,
        "focused_heatmap_df": plot_df,
        "stability_summary": stability_summary,
        "data_summary": data_summary,
    }


def _apply_style(render_config: RenderConfig, grid: str = "whitegrid") -> None:
    sns.set_style(grid)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [render_config.style.font_family, "DejaVu Sans", "Liberation Sans", "Arial"]
    plt.rcParams["font.size"] = render_config.style.base_font_size


def _build_custom_blues(zero_color: str) -> LinearSegmentedColormap:
    colors = [zero_color, "#DEEBF7", "#9ECAE1", "#3182BD", "#08306B"]
    return LinearSegmentedColormap.from_list("phase3_custom_blues", colors)


class StabilityLandscapeRenderer(BaseSmartRenderer):
    figure_type = "stability_landscape"

    def _render_once(self, plot_data: Dict[str, Any], render_config: RenderConfig):
        _apply_style(render_config, grid="whitegrid")
        df = plot_data["stability_df"]
        style = render_config.style
        fig, ax = plt.subplots(figsize=(render_config.figure_width, render_config.figure_height), dpi=style.export_dpi)

        fig.subplots_adjust(
            left=render_config.margin_left,
            right=1 - render_config.margin_right,
            top=1 - render_config.margin_top,
            bottom=render_config.margin_bottom,
        )

        ax.axvspan(0.60, 1.02, ymin=(3 - 1) / (7.4 - 1), ymax=1, color="#EEF6FF", alpha=0.90, zorder=0)
        ax.axvspan(0.0, 0.60, ymin=0, ymax=(3 - 1) / (7.4 - 1), color="#F8FAFC", alpha=0.95, zorder=0)
        ax.axvline(0.60, color="#6B7280", linestyle="--", linewidth=style.line_width)
        ax.axhline(3, color="#6B7280", linestyle="--", linewidth=style.line_width)

        panel_scores = df["panel_score"].to_numpy(dtype=float)
        if np.allclose(panel_scores.max(), panel_scores.min()):
            sizes = np.full(len(df), (render_config.bubble_size_min + render_config.bubble_size_max) / 2.0)
        else:
            sizes = render_config.bubble_size_min + (
                (panel_scores - panel_scores.min()) / (panel_scores.max() - panel_scores.min())
            ) * (render_config.bubble_size_max - render_config.bubble_size_min)

        scatter_collections = []
        for status, color in style.status_colors.items():
            subset = df[df["status"] == status]
            if subset.empty:
                continue
            collection = ax.scatter(
                subset["selection_frequency"],
                subset["method_consensus"],
                s=sizes[subset.index],
                c=color,
                edgecolors="white",
                linewidths=style.marker_edge_width,
                alpha=0.92 if status != "rejected" else 0.70,
                label=status.replace("_", " ").title(),
                zorder=3 if status != "rejected" else 2,
            )
            scatter_collections.append(collection)

        label_df = df[df["is_final_panel"]].head(render_config.annotate_top_n)
        text_artists = []
        offsets = [(8, 8), (8, -10), (-10, 8), (-10, -10), (10, 2), (2, 10), (-12, 3), (10, -2)]
        for idx, (_, row) in enumerate(label_df.iterrows()):
            dx, dy = offsets[idx % len(offsets)]
            text_artists.append(
                ax.annotate(
                    _format_feature_axis_label(row["display_name"], max_len=render_config.ytick_max_len),
                    (row["selection_frequency"], row["method_consensus"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=render_config.font_size_feature,
                    color="#1F2937",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.82},
                )
            )
        if ADJUST_TEXT_AVAILABLE and adjust_text is not None and text_artists:
            adjust_text(
                text_artists,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.5, alpha=0.6),
                only_move={"points": "y", "texts": "xy"},
            )

        zone_alpha = float(render_config.extras.get("zone_label_alpha", 0.70))
        ax.text(0.79, 6.7, "Stable Core Zone", fontsize=style.tick_font_size, color="#6B7280", alpha=zone_alpha, ha="center", style="italic")
        ax.text(0.18, 1.45, "Low Frequency\nLow Consensus", fontsize=style.tick_font_size, color="#6B7280", ha="center")
        ax.text(0.83, 1.45, "High Frequency\nLow Consensus", fontsize=style.tick_font_size, color="#6B7280", ha="center")
        ax.text(0.18, 5.7, "Low Frequency\nHigh Consensus", fontsize=style.tick_font_size, color="#6B7280", ha="center")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.7, 7.4)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(range(1, 8))
        ax.set_xlabel("Selection Frequency", fontsize=style.label_font_size)
        ax.set_ylabel("Method Consensus", fontsize=style.label_font_size)
        ax.set_title("Phase 1 Stability Landscape", fontsize=style.title_font_size, fontweight="bold", pad=12)
        ax.tick_params(labelsize=style.tick_font_size)
        ax.grid(color="#E5E7EB", linewidth=0.8)

        legend = ax.legend(
            frameon=True,
            loc="center left" if render_config.legend_outside else "lower right",
            bbox_to_anchor=(1.02, 0.5) if render_config.legend_outside else None,
            title="Feature Status",
            fontsize=style.legend_font_size,
            title_fontsize=style.legend_font_size,
        )

        size_values = [0.3, 0.6, 0.9]
        size_handles = [
            plt.scatter([], [], s=render_config.bubble_size_min + value * (render_config.bubble_size_max - render_config.bubble_size_min), color="#9CA3AF", alpha=0.5)
            for value in size_values
        ]
        size_legend = ax.legend(
            size_handles,
            [f"{value:.1f}" for value in size_values],
            title="Panel score",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            fontsize=style.legend_font_size,
            title_fontsize=style.legend_font_size,
        )
        ax.add_artist(legend)

        fig.text(
            0.01,
            0.01,
            "Bubble size encodes composite panel score. Dashed lines mark the stable-core thresholds "
            "(selection frequency >= 0.60; method consensus >= 3).",
            fontsize=style.tick_font_size,
            color="#4B5563",
        )

        metadata = {
            "managed_texts": text_artists,
            "managed_legends": [legend, size_legend],
            "data_artists": scatter_collections,
            "annotate_top_n": render_config.annotate_top_n,
            "font_size_feature": render_config.font_size_feature,
            "margin_right": render_config.margin_right,
            "figure_height": render_config.figure_height,
        }
        return fig, metadata


class MethodFeatureHeatmapRenderer(BaseSmartRenderer):
    figure_type = "method_feature_heatmap"

    def _render_once(self, plot_data: Dict[str, Any], render_config: RenderConfig):
        _apply_style(render_config, grid="white")
        plot_df = plot_data["focused_heatmap_df"]
        if render_config.focused_mode:
            plot_df = plot_data["stability_df"]
            plot_df = _select_focused_heatmap_rows(plot_df, rejected_n=render_config.focused_rejected_n)

        method_cols = [method for method in _STABILITY_METHOD_LABELS if method in plot_df.columns]
        heatmap_df = plot_df[["display_name"] + method_cols].copy()
        heatmap_df = heatmap_df.rename(columns=_STABILITY_METHOD_LABELS)
        heatmap_df["display_name"] = heatmap_df["display_name"].apply(
            lambda value: _format_feature_axis_label(value, max_len=render_config.ytick_max_len)
        )
        heatmap_df = heatmap_df.set_index("display_name")

        style = render_config.style
        fig, (ax, ax_status) = plt.subplots(
            1,
            2,
            figsize=(render_config.figure_width, render_config.figure_height),
            dpi=style.export_dpi,
            gridspec_kw={"width_ratios": [18, 0.7], "wspace": 0.04},
        )
        fig.subplots_adjust(
            left=render_config.margin_left,
            right=1 - render_config.margin_right,
            top=1 - render_config.margin_top,
            bottom=render_config.margin_bottom,
        )

        cmap = _build_custom_blues(style.zero_value_color)
        status_codes = plot_df["status"].map(
            {status: idx for idx, status in enumerate(["rejected", "completed", "stable_core", "mandatory"])}
        ).to_numpy().reshape(-1, 1)
        status_cmap = ListedColormap([
            style.status_colors["rejected"],
            style.status_colors["completed"],
            style.status_colors["stable_core"],
            style.status_colors["mandatory"],
        ])

        sns.heatmap(
            heatmap_df,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0.35,
            linecolor="#F3F4F6",
            cbar_kws={"label": "Per-method frequency", "shrink": render_config.colorbar_shrink},
        )
        ax.set_title("Focused Method-Feature Frequency Heatmap", fontsize=style.title_font_size, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=render_config.xtick_rotation, labelsize=style.tick_font_size)
        ax.tick_params(axis="y", labelsize=render_config.font_size_feature)

        stable_core_idx = plot_df.index[plot_df["is_stable_core"]].tolist()
        final_panel_idx = plot_df.index[plot_df["is_final_panel"]].tolist()
        if stable_core_idx:
            ax.hlines(max(stable_core_idx) + 1, *ax.get_xlim(), colors=style.status_colors["stable_core"], linewidth=2.0)
        if final_panel_idx:
            ax.hlines(max(final_panel_idx) + 1, *ax.get_xlim(), colors=style.status_colors["completed"], linewidth=1.6, linestyles="--")
            ax.text(-0.12, (max(final_panel_idx) + 0.5) / max(len(plot_df), 1), "Final panel", transform=ax.transAxes, ha="right", va="center", fontsize=style.tick_font_size, color=style.status_colors["completed"], weight="bold")
        if len(plot_df) > len(final_panel_idx):
            ax.text(-0.12, (len(final_panel_idx) + (len(plot_df) - len(final_panel_idx)) / 2) / max(len(plot_df), 1), "Near-miss\nrejected", transform=ax.transAxes, ha="right", va="center", fontsize=style.tick_font_size, color="#6B7280", weight="bold")

        for tick, status in zip(ax.get_yticklabels(), plot_df["status"]):
            if status == "stable_core":
                tick.set_color(style.status_colors["stable_core"])
                tick.set_fontweight("bold")
            elif status == "completed":
                tick.set_color(style.status_colors["completed"])
            elif status == "mandatory":
                tick.set_color(style.status_colors["mandatory"])
                tick.set_fontweight("bold")
            else:
                tick.set_color("#4B5563")

        ax_status.imshow(status_codes, aspect="auto", cmap=status_cmap, interpolation="nearest")
        ax_status.set_xticks([])
        ax_status.set_yticks([])
        ax_status.set_title("Status", fontsize=style.tick_font_size, pad=10)
        for spine in ax_status.spines.values():
            spine.set_visible(False)

        if render_config.extras.get("show_method_group_strip", True):
            method_groups = [
                ("Regularized", 0, 2, style.palette[0]),
                ("Tree-based", 2, 4, style.palette[2]),
                ("Filter", 4, 7, style.palette[3]),
            ]
            for label, start, end, color in method_groups:
                ax.annotate("", xy=(end, -0.55), xytext=(start, -0.55), xycoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color=color, lw=2))
                ax.text((start + end) / 2, -0.60, label, ha="center", va="top", fontsize=style.tick_font_size, color=color, transform=ax.get_xaxis_transform())

        handles = [
            mlines.Line2D([0], [0], marker="s", color="w", markerfacecolor=style.status_colors[key], markersize=9, label=key.replace("_", " ").title())
            for key in ["mandatory", "stable_core", "completed", "rejected"]
        ]
        legend = ax.legend(
            handles=handles,
            frameon=True,
            fontsize=style.legend_font_size,
            title="Row status",
            title_fontsize=style.legend_font_size,
            loc="upper left",
            bbox_to_anchor=(1.15, 1.0),
        )

        fig.text(
            0.01,
            0.01,
            "Shows all final-panel features plus a small set of highest-priority near-miss rejected candidates. "
            "Solid blue line marks the stable core; dashed green line marks the final panel boundary.",
            fontsize=style.tick_font_size,
            color="#4B5563",
        )

        metadata = {
            "managed_texts": [],
            "managed_legends": [legend],
            "data_artists": list(ax.collections),
            "tracked_yticklabels": list(ax.get_yticklabels()),
            "font_size_feature": render_config.font_size_feature,
            "figure_height": render_config.figure_height,
        }
        return fig, metadata
