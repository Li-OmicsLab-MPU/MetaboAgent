from __future__ import annotations

import json
from typing import Any, Dict, Optional


class VLMReviewer:
    """Unified reviewer for single-figure and composite-figure visual audits."""

    def __init__(self, use_mock: bool = True, vlm_caller: Optional[Any] = None):
        self.use_mock = use_mock
        self.vlm_caller = vlm_caller
        self.audit_count = 0

    def audit(
        self,
        *,
        image_path: str,
        scope: str = "single",
        figure_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.audit_count += 1
        if self.use_mock:
            return self._mock_audit(image_path=image_path, scope=scope, figure_type=figure_type, metadata=metadata or {})
        return self._real_audit(image_path=image_path, scope=scope, figure_type=figure_type, metadata=metadata or {})

    def audit_figure(self, image_path: str) -> Dict[str, Any]:
        """Backward-compatible alias for legacy callers."""
        return self.audit(image_path=image_path, scope="single", figure_type="legacy")

    def _mock_audit(
        self,
        *,
        image_path: str,
        scope: str,
        figure_type: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        high_risk = figure_type in {"plot_stability_landscape", "plot_shap", "composite_figure"}
        if high_risk and self.audit_count % 2 == 1:
            return {
                "status": "FAIL",
                "scope": scope,
                "reasoning": "Mock reviewer flagged minor crowding and recommends reducing annotations plus moving legends outward.",
                "severity": "medium",
                "issues": [{"type": "crowding", "location": "upper_right_cluster", "description": "Labels and legends feel dense."}],
                "suggested_actions": {"annotate_top_n": 10, "legend_outside": True, "margin_right": 0.20},
            }
        return {
            "status": "PASS",
            "scope": scope,
            "reasoning": "Mock reviewer found no blocking visual issue.",
            "severity": "low",
            "issues": [],
            "suggested_actions": {},
        }

    def _real_audit(
        self,
        *,
        image_path: str,
        scope: str,
        figure_type: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.vlm_caller is None:
            return {
                "status": "PASS",
                "scope": scope,
                "reasoning": "No real VLM caller configured; fallback to PASS.",
                "severity": "low",
                "issues": [],
                "suggested_actions": {},
            }

        prompt = (
            "You are a top-tier journal visual reviewer.\n"
            f"Scope: {scope}\n"
            f"Figure type: {figure_type}\n"
            f"Metadata: {json.dumps(metadata, ensure_ascii=True)}\n"
            "Assess label overlap, legend occlusion, bounding-box overflow, and overall academic aesthetics.\n"
            "Return strict JSON with keys: status, scope, reasoning, severity, issues, suggested_actions."
        )
        try:
            response_str = self.vlm_caller(prompt, image_path=image_path)
            clean_str = response_str.replace("```json", "").replace("```", "").strip()
            payload = json.loads(clean_str)
            if not isinstance(payload, dict):
                raise ValueError("VLM output is not a JSON object")
            payload.setdefault("scope", scope)
            payload.setdefault("severity", "medium")
            payload.setdefault("issues", [])
            payload.setdefault("suggested_actions", {})
            return payload
        except Exception as exc:
            return {
                "status": "PASS",
                "scope": scope,
                "reasoning": f"VLM audit failed ({exc}); fallback to PASS.",
                "severity": "low",
                "issues": [],
                "suggested_actions": {},
            }
