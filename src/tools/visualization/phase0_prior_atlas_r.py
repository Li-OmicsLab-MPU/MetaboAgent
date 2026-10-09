from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.tools.visualization.journal_theme import write_theme_json
from src.tools.visualization.viz_tools import _normalize_feature_token, _resolve_original_feature_name_map


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _run_r_prior_atlas(
    summary_csv: Path,
    segment_csv: Path,
    output_stem: Path,
    title: str,
    threshold_value: float,
    theme_json_path: str = "",
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_phase0_prior_evidence_atlas.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(project_root / ".r_libs")
    command = [
        "Rscript",
        str(r_script),
        str(summary_csv),
        str(segment_csv),
        str(output_stem),
        title,
        f"{threshold_value:.12f}",
        theme_json_path,
    ]
    subprocess.run(command, check=True, cwd=str(project_root), env=env)


def _wrap_label(name: str, width: int = 22) -> str:
    text = str(name).replace("_", " ").strip()
    if len(text) <= width:
        return text
    parts = text.split(" ")
    lines: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = part
    if current:
        lines.append(current)
    return "\n".join(lines[:2]) if lines else text


def _normalize_phase0_payload(phase0_output: Dict[str, Any] | str) -> Dict[str, Any]:
    if isinstance(phase0_output, dict):
        return phase0_output
    with open(str(phase0_output), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_phase0_display_name(record: Dict[str, Any], original_name_map: Dict[str, str]) -> str:
    candidates: List[str] = []
    for key in [
        "original_name",
        "raw_feature_name",
        "source_feature",
        "matched_feature",
        "mapped_feature",
        "display_name",
        "canonical_name",
        "standardized_name",
        "name",
        "id",
    ]:
        value = str(record.get(key, "") or "").strip()
        if value:
            candidates.append(value)
    for item in record.get("mapped_metabolites") or []:
        value = str(item).strip()
        if value:
            candidates.append(value)

    normalized_map = {
        _normalize_feature_token(key): value
        for key, value in original_name_map.items()
        if str(key).strip() and str(value).strip()
    }
    for candidate in candidates:
        mapped = original_name_map.get(candidate)
        if mapped:
            return str(mapped)
        mapped = normalized_map.get(_normalize_feature_token(candidate))
        if mapped:
            return str(mapped)
    return ""


def plot_phase0_prior_evidence_atlas(
    phase0_output: Dict[str, Any] | str,
    save_path: str = "output/figures/fig1c_prior_evidence_atlas.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    payload = _normalize_phase0_payload(phase0_output)
    candidate_summary = payload.get("candidate_scores_summary") or payload.get("confirmed_biomarkers") or []
    if not isinstance(candidate_summary, list):
        candidate_summary = []
    candidate_summary = [
        item if isinstance(item, dict) else {"name": str(item), "bio_prior_raw": 0.0}
        for item in candidate_summary
    ]
    final_priors = [str(item).strip() for item in (payload.get("final_priors") or []) if str(item).strip()]
    if not candidate_summary and final_priors:
        candidate_summary = [
            {"name": item, "bio_prior_raw": 0.0, "selected_after_threshold": True}
            for item in final_priors
        ]
    if not candidate_summary:
        candidate_summary = [
            {
                "name": "No prior evidence retained",
                "original_name": "No prior evidence retained",
                "bio_prior_raw": 0.0,
                "selected_after_threshold": False,
                "_placeholder": True,
            }
        ]
    threshold_info = payload.get("selection_threshold") or {}
    if "threshold" not in threshold_info and final_priors:
        retained_scores = [
            float(item.get("bio_prior_raw", item.get("confidence_score", 0.0)) or 0.0)
            for item in candidate_summary
            if str(item.get("name", "")) in set(final_priors)
        ]
        if retained_scores:
            threshold_info = {"threshold": min(retained_scores)}
    disease_name = str(payload.get("disease_name") or "disease").strip()
    if "threshold" not in threshold_info:
        threshold_info = {"threshold": 0.0}

    summary_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    original_name_map = _resolve_original_feature_name_map()
    segment_specs = [
        ("mechanistic_contribution", "Mechanistic plausibility"),
        ("specificity_contribution", "Disease specificity"),
        ("clinical_contribution", "Clinical evidence"),
        ("consistency_contribution", "Consistency bonus"),
    ]

    ordered = sorted(candidate_summary, key=lambda item: float(item.get("bio_prior_raw", 0.0) or 0.0), reverse=True)
    row_rank = 0
    for record in ordered:
        total_score = float(record.get("bio_prior_raw", record.get("confidence_score", 0.0)) or 0.0)
        row_rank += 1
        resolved_name = _resolve_phase0_display_name(record, original_name_map)
        fallback_name = str(record.get("original_name") or record.get("name") or f"candidate_{row_rank}")
        display_name = resolved_name or fallback_name
        display_label = _wrap_label(display_name)
        selected = bool(record.get("selected_after_threshold", str(record.get("name", "")) in set(payload.get("final_priors") or [])))
        status_label = "Retained" if selected else "Excluded"
        summary_rows.append(
            {
                "rank": row_rank,
                "name": str(record.get("name", "")),
                "display_name": display_name,
                "display_label": display_label,
                "total_score": total_score,
                "selected_after_threshold": selected,
                "status_label": status_label,
                "pubmed_hit_count": int(record.get("pubmed_hit_count", 0) or 0),
                "pathway_overlap_count": int(record.get("pathway_overlap_count", 0) or 0),
            }
        )
        for segment_key, segment_label in segment_specs:
            contribution = float(record.get(segment_key, record.get(segment_key.replace("_contribution", "_score"), 0.0)) or 0.0)
            if segment_key == "consistency_contribution":
                contribution = float(
                    record.get(
                        "consistency_contribution",
                        record.get("consistency_bonus", record.get("consistency", contribution)),
                    )
                    or 0.0
                )
            segment_rows.append(
                {
                    "rank": row_rank,
                    "display_label": display_label,
                    "segment_label": segment_label,
                    "contribution": contribution,
                    "status_label": status_label,
                    "total_score": total_score,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    segment_df = pd.DataFrame(segment_rows)
    if summary_df.empty or segment_df.empty:
        raise ValueError("Unable to build Prior Evidence Atlas rows from the phase0 payload.")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    summary_csv = output_path.parent / f"{output_stem.name}_summary.csv"
    segment_csv = output_path.parent / f"{output_stem.name}_segments.csv"
    summary_df.to_csv(summary_csv, index=False)
    segment_df.to_csv(segment_csv, index=False)

    threshold_value = float(threshold_info["threshold"])
    theme_json_path = (
        write_theme_json(
            journal_theme,
            output_stem.with_name(f"{output_stem.name}_theme.json"),
            task_name="phase0_prior_evidence_atlas",
        )
        if journal_theme else ""
    )
    _run_r_prior_atlas(
        summary_csv=summary_csv,
        segment_csv=segment_csv,
        output_stem=output_stem,
        title=f"Prior Evidence Atlas: {disease_name}",
        threshold_value=threshold_value,
        theme_json_path=theme_json_path,
    )

    saved_paths = _save_r_figure_bundle(save_path)
    metadata = {
        "figure_title": "Prior Evidence Atlas",
        "disease_name": disease_name,
        "threshold": threshold_value,
        "candidate_count": int(len(summary_df)),
        "retained_count": int(summary_df["selected_after_threshold"].sum()),
        "placeholder_used": bool(summary_df.get("name", pd.Series(dtype=str)).eq("No prior evidence retained").any()),
        "summary_csv": str(summary_csv),
        "segment_csv": str(segment_csv),
        "theme_json_path": theme_json_path,
        "auxiliary_outputs": saved_paths[1:],
    }
    return saved_paths[0], metadata
