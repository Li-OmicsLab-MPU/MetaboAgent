from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from src.agents.phase3.styles.style_types import RenderConfig, RenderResult


def _bbox_intersects(box_a, box_b) -> bool:
    return not (
        box_a.x1 < box_b.x0
        or box_a.x0 > box_b.x1
        or box_a.y1 < box_b.y0
        or box_a.y0 > box_b.y1
    )


class BaseSmartRenderer:
    figure_type: str = "generic"

    def render(self, plot_data: Dict[str, Any], render_config: RenderConfig, save_path: str) -> RenderResult:
        current_config = render_config
        issues: List[str] = []
        retry_count = 0
        final_metadata: Dict[str, Any] = {}
        final_fig = None

        for attempt in range(current_config.internal_retry_limit + 1):
            fig, metadata = self._render_once(plot_data, current_config)
            qa_result = self._self_check(fig=fig, metadata=metadata)
            final_fig = fig
            final_metadata = metadata
            issues = list(qa_result.get("issues", []))

            if qa_result.get("pass", True):
                break

            if attempt >= current_config.internal_retry_limit:
                break

            retry_count += 1
            current_config = self._patch_render_config(current_config, qa_result)
            plt.close(fig)

        if final_fig is None:
            raise RuntimeError("Renderer failed to create a matplotlib figure.")

        self._save_figure(final_fig, save_path, current_config)
        plt.close(final_fig)
        return RenderResult(
            save_path=save_path,
            figure_type=self.figure_type,
            render_config=current_config,
            self_check_passed=not issues,
            self_check_issues=issues,
            retry_count=retry_count,
            metadata=final_metadata,
        )

    def _render_once(self, plot_data: Dict[str, Any], render_config: RenderConfig):
        raise NotImplementedError

    def _self_check(self, fig, metadata: Dict[str, Any]) -> Dict[str, Any]:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        issues: List[str] = []
        suggested_actions: Dict[str, Any] = {}

        managed_texts = [text for text in metadata.get("managed_texts", []) if getattr(text, "get_visible", lambda: True)()]
        text_boxes = [text.get_window_extent(renderer=renderer) for text in managed_texts if text.get_text().strip()]
        overlap_count = 0
        for idx in range(len(text_boxes)):
            for jdx in range(idx + 1, len(text_boxes)):
                if _bbox_intersects(text_boxes[idx], text_boxes[jdx]):
                    overlap_count += 1
        if overlap_count > 0:
            issues.append("label_overlap")
            suggested_actions["annotate_top_n"] = max(6, int(metadata.get("annotate_top_n", 12)) - 2)
            suggested_actions["font_size_feature"] = max(5.8, float(metadata.get("font_size_feature", 8.0)) - 0.4)

        managed_legends = [legend for legend in metadata.get("managed_legends", []) if legend is not None]
        data_bboxes = [artist.get_window_extent(renderer=renderer) for artist in metadata.get("data_artists", []) if hasattr(artist, "get_window_extent")]
        for legend in managed_legends:
            legend_box = legend.get_window_extent(renderer=renderer)
            for data_box in data_bboxes:
                if _bbox_intersects(legend_box, data_box):
                    issues.append("legend_occlusion")
                    suggested_actions["legend_outside"] = True
                    suggested_actions["margin_right"] = max(0.18, float(metadata.get("margin_right", 0.14)))
                    break

        tracked_yticklabels = [tick for tick in metadata.get("tracked_yticklabels", []) if tick.get_text().strip()]
        tick_boxes = [tick.get_window_extent(renderer=renderer) for tick in tracked_yticklabels]
        tick_overlap = 0
        for idx in range(len(tick_boxes)):
            for jdx in range(idx + 1, len(tick_boxes)):
                if _bbox_intersects(tick_boxes[idx], tick_boxes[jdx]):
                    tick_overlap += 1
        if tick_overlap > 0:
            issues.append("tick_overlap")
            suggested_actions["figure_height"] = min(14.5, float(metadata.get("figure_height", 8.0)) + 0.8)
            suggested_actions["font_size_feature"] = max(5.8, float(metadata.get("font_size_feature", 8.0)) - 0.2)

        return {"pass": not issues, "issues": issues, "suggested_actions": suggested_actions}

    def _patch_render_config(self, render_config: RenderConfig, qa_result: Dict[str, Any]) -> RenderConfig:
        updates = dict(qa_result.get("suggested_actions", {}) or {})
        if not updates:
            return render_config
        return render_config.patch(**updates)

    def _save_figure(self, fig, save_path: str, render_config: RenderConfig) -> None:
        fig.savefig(save_path, dpi=render_config.style.export_dpi, bbox_inches="tight", facecolor="white")
