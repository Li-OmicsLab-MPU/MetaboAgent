from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _run_r_prior_profile(
    segment_csv: Path,
    summary_csv: Path,
    output_stem: Path,
    title: str,
    subtitle: str,
    threshold_label: str,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_phase0_prior_evidence_profile.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(project_root / ".r_libs")
    command = [
        "Rscript",
        str(r_script),
        str(segment_csv),
        str(summary_csv),
        str(output_stem),
        title,
        subtitle,
        threshold_label,
    ]
    subprocess.run(command, check=True, cwd=str(project_root), env=env)


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _wrap_label(name: str, width: int = 30) -> str:
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


def plot_phase0_prior_evidence_profile(
    phase0_output_path: str,
    save_path: str = "output/phase0/artifacts/prior_profile_preview_figures/phase0_prior_evidence_profile.pdf",
) -> Tuple[str, Dict]:
    payload = _read_json(phase0_output_path)
    candidate_summary = payload.get("candidate_scores_summary") or []
    threshold_info = payload.get("selection_threshold") or {}
    disease_name = str(payload.get("disease_name") or "disease").strip()
    if not candidate_summary:
        raise ValueError("candidate_scores_summary is empty; cannot build Prior Evidence Profile.")
    if "threshold" not in threshold_info:
        raise ValueError("selection_threshold.threshold is missing; cannot build Prior Evidence Profile.")

    summary_rows: List[Dict] = []
    segment_rows: List[Dict] = []
    # Keep the segment export order aligned with the desired left-to-right stack order
    # after coord_flip() in ggplot: mechanistic block should anchor the left side.
    segment_specs = [
        ("clinical_contribution", "Clinical evidence"),
        ("specificity_contribution", "Disease specificity"),
        ("consistency_contribution", "Consistency bonus"),
        ("mechanistic_contribution", "Mechanistic plausibility"),
    ]

    ordered = sorted(candidate_summary, key=lambda item: float(item.get("bio_prior_raw", 0.0) or 0.0), reverse=True)
    for rank, record in enumerate(ordered, start=1):
        display_label = _wrap_label(str(record.get("name", f"candidate_{rank}")))
        total_score = float(record.get("bio_prior_raw", 0.0) or 0.0)
        if total_score <= 0:
            continue
        selected = bool(record.get("selected_after_threshold", False))
        status_label = "Retained" if selected else "Excluded"
        pubmed_hits = int(record.get("pubmed_hit_count", 0) or 0)
        pathway_overlap = int(record.get("pathway_overlap_count", 0) or 0)
        summary_rows.append(
            {
                "rank": rank,
                "name": str(record.get("name", "")),
                "display_label": display_label,
                "total_score": total_score,
                "selected_after_threshold": selected,
                "status_label": status_label,
                "pubmed_hit_count": pubmed_hits,
                "pathway_overlap_count": pathway_overlap,
                "evidence_zone": str(record.get("evidence_zone", "")),
                "retain_reason": str(record.get("retain_reason", "") or ""),
                "drop_reason": str(record.get("drop_reason", "") or ""),
                "score_confidence": record.get("score_confidence"),
                "selected_flag_int": 1 if selected else 0,
            }
        )
        for segment_key, segment_label in segment_specs:
            segment_rows.append(
                {
                    "rank": rank,
                    "display_label": display_label,
                    "segment_key": segment_key,
                    "segment_label": segment_label,
                    "contribution": float(record.get(segment_key, 0.0) or 0.0),
                    "total_score": total_score,
                    "status_label": status_label,
                    "selected_after_threshold": selected,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    segment_df = pd.DataFrame(segment_rows)
    if summary_df.empty or segment_df.empty:
        raise ValueError("No non-zero prior evidence rows remain after filtering.")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    segment_csv = output_path.parent / "phase0_prior_evidence_segments.csv"
    summary_csv = output_path.parent / "phase0_prior_evidence_summary.csv"
    segment_df.to_csv(segment_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    threshold_value = float(threshold_info["threshold"])
    threshold_rule = str(threshold_info.get("rule", "adaptive_threshold"))
    retained_count = int(summary_df["selected_after_threshold"].sum())
    subtitle = (
        f"{disease_name.title()} candidates ranked by adaptive prior score; "
        f"{retained_count}/{len(summary_df)} retained after thresholding"
    )
    threshold_label = f"Adaptive threshold = {threshold_value:.2f} ({threshold_rule})"

    _run_r_prior_profile(
        segment_csv=segment_csv,
        summary_csv=summary_csv,
        output_stem=output_stem,
        title="Prior Evidence Profile",
        subtitle=subtitle,
        threshold_label=threshold_label,
    )
    saved_paths = _save_r_figure_bundle(save_path)
    metadata = {
        "figure_title": "Prior Evidence Profile",
        "phase0_output_path": phase0_output_path,
        "segment_csv": str(segment_csv),
        "summary_csv": str(summary_csv),
        "threshold": threshold_value,
        "threshold_rule": threshold_rule,
        "candidate_count": int(len(summary_df)),
        "retained_count": retained_count,
        "auxiliary_outputs": saved_paths[1:],
    }
    return saved_paths[0], metadata
