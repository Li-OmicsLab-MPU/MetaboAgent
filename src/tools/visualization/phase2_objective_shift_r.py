from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.tools.visualization.journal_theme import write_theme_json
from src.tools.visualization.viz_tools import plot_radar_comparison

from Evaluate.build_phase2_offline_repair import (
    PROJECT_ROOT,
    _load_taxonomy_map,
    _read_json,
    _repair_winner_feature_contributions,
    _resolve_existing_path,
)


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _normalize_features(features: Sequence[str]) -> List[str]:
    seen = set()
    values: List[str] = []
    for feature in features:
        item = str(feature or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        values.append(item)
    return values


def _run_r_script(radar_csv: Path, contrib_csv: Path, output_stem: Path, theme_json_path: str = "") -> None:
    r_script = PROJECT_ROOT / "src" / "tools" / "visualization" / "r" / "plot_phase2_phase1_to_phase2_summary.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(PROJECT_ROOT / ".r_libs")
    subprocess.run(
        ["Rscript", str(r_script), str(radar_csv), str(contrib_csv), str(output_stem), theme_json_path],
        check=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _resolve_phase0_context(
    winner_scores: Dict[str, Any],
    search_summary: Dict[str, Any],
) -> Dict[str, Any]:
    for payload in (
        winner_scores.get("phase0_output"),
        search_summary.get("phase0_output"),
    ):
        if isinstance(payload, dict) and payload:
            return payload

    disease_name = str(
        winner_scores.get("disease_name")
        or search_summary.get("disease_name")
        or ""
    ).strip()
    if not disease_name:
        return {}
    return {"disease_name": disease_name}


def _extract_phase2_score_map(winner_scores: Dict[str, Any]) -> Dict[str, float]:
    """Extract Phase 2 winner 4D scores from both canonical and test-result schemas."""
    final_result = winner_scores.get("final_result") if isinstance(winner_scores.get("final_result"), dict) else {}
    nested_scores = winner_scores.get("scores") if isinstance(winner_scores.get("scores"), dict) else {}

    def _first_float(*values: Any) -> float:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    comprehensive_metrics = final_result.get("comprehensive_metrics") if isinstance(final_result.get("comprehensive_metrics"), dict) else {}
    auc = _first_float(
        comprehensive_metrics.get("roc_auc"),
        final_result.get("roc_auc"),
        winner_scores.get("roc_auc"),
        nested_scores.get("roc_auc"),
        final_result.get("perf"),
        nested_scores.get("f_perf"),
    )
    return {
        "f_perf": auc,
        "f_bio": _first_float(final_result.get("bio"), nested_scores.get("f_bio"), winner_scores.get("bio")),
        "f_cost": _first_float(final_result.get("cost"), nested_scores.get("f_cost"), winner_scores.get("cost")),
        "f_corr": _first_float(final_result.get("corr"), nested_scores.get("f_corr"), winner_scores.get("corr")),
    }


def _build_radar_csv(
    *,
    phase1_scores: Dict[str, Any],
    winner_scores: Dict[str, Any],
    output_dir: Path,
    output_stem: str,
) -> Path:
    baseline_scores = phase1_scores.get("scores", {}) or {}
    winner_stored_scores = _extract_phase2_score_map(winner_scores)
    radar_rows: List[Dict[str, Any]] = []
    for panel_name, score_map in [
        ("Phase 1 baseline", baseline_scores),
        ("Phase 2 winner", winner_stored_scores),
    ]:
        radar_rows.extend(
            [
                {"panel": panel_name, "metric": "AUC", "value": float(score_map.get("f_perf", 0.0) or 0.0)},
                {"panel": panel_name, "metric": "Biology", "value": float(score_map.get("f_bio", 0.0) or 0.0)},
                {"panel": panel_name, "metric": "Parsimony", "value": 1.0 - float(score_map.get("f_cost", 0.0) or 0.0)},
                {"panel": panel_name, "metric": "Independence", "value": 1.0 - float(score_map.get("f_corr", 0.0) or 0.0)},
            ]
        )

    radar_csv = output_dir / f"{output_stem}_radar_profiles.csv"
    pd.DataFrame(radar_rows).to_csv(radar_csv, index=False)
    return radar_csv


def _build_legacy_radar_profiles(
    *,
    phase1_scores: Dict[str, Any],
    winner_scores: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    baseline_scores = phase1_scores.get("scores", {}) or {}
    winner_stored_scores = _extract_phase2_score_map(winner_scores)
    return {
        "Phase 1 Baseline": {
            "AUC": float(baseline_scores.get("f_perf", 0.0) or 0.0),
            "f_bio": float(baseline_scores.get("f_bio", 0.0) or 0.0),
            "1-f_cost": 1.0 - float(baseline_scores.get("f_cost", 0.0) or 0.0),
            "1-f_corr": 1.0 - float(baseline_scores.get("f_corr", 0.0) or 0.0),
        },
        "PToT Winner": {
            "AUC": float(winner_stored_scores.get("f_perf", 0.0) or 0.0),
            "f_bio": float(winner_stored_scores.get("f_bio", 0.0) or 0.0),
            "1-f_cost": 1.0 - float(winner_stored_scores.get("f_cost", 0.0) or 0.0),
            "1-f_corr": 1.0 - float(winner_stored_scores.get("f_corr", 0.0) or 0.0),
        },
    }


def _write_legacy_radar_4d(
    *,
    phase1_scores: Dict[str, Any],
    winner_scores: Dict[str, Any],
    save_path: Path,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> List[str]:
    radar_pdf = save_path.with_name("phase2_4d_radar_profile.pdf")
    profiles = _build_legacy_radar_profiles(
        phase1_scores=phase1_scores,
        winner_scores=winner_scores,
    )
    saved_paths = _save_r_figure_bundle(str(radar_pdf))
    for path in saved_paths:
        plot_radar_comparison(
            profiles_dict=profiles,
            save_path=path,
            title="4D Panel Performance Profile",
            journal_theme=journal_theme,
        )
    return saved_paths


def _count_feature_hits(data_path: Path, features: Sequence[str], target_column: str) -> int:
    if not data_path.exists():
        return -1
    try:
        columns = list(pd.read_csv(data_path, nrows=0).columns)
    except Exception:
        return -1
    usable_features = {str(feature).strip() for feature in features if str(feature).strip() and str(feature).strip() != target_column}
    if not usable_features:
        return 0
    return sum(1 for column in columns if column in usable_features)


def _load_existing_contribution_summary(
    *,
    candidate_dirs: Sequence[Path],
    winner_features: Sequence[str],
) -> Optional[Dict[str, Any]]:
    normalized_winner_features = {str(value).strip() for value in winner_features if str(value).strip()}
    seen_dirs: set[str] = set()
    for candidate_dir in candidate_dirs:
        dir_key = str(candidate_dir)
        if not dir_key or dir_key in seen_dirs:
            continue
        seen_dirs.add(dir_key)

        contrib_csv = candidate_dir / "phase2_winner_feature_contributions.csv"
        summary_json = candidate_dir / "phase2_winner_feature_contributions_summary.json"
        if not contrib_csv.exists():
            continue
        try:
            frame = pd.read_csv(contrib_csv)
        except Exception:
            continue
        existing_features = {str(value).strip() for value in frame.get("feature", []) if str(value).strip()}
        if len(frame.index) != len(normalized_winner_features) or existing_features != normalized_winner_features:
            continue
        summary: Dict[str, Any] = {}
        if summary_json.exists():
            try:
                loaded = json.loads(summary_json.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
            except Exception:
                summary = {}
        summary["contribution_csv"] = str(contrib_csv)
        summary["winner_feature_count"] = len(frame.index)
        summary["reused_existing"] = True
        summary["reuse_source_dir"] = str(candidate_dir)
        return summary
    return None


def _resolve_best_data_path(
    *,
    winner_scores: Dict[str, Any],
    phase1_scores: Dict[str, Any],
) -> Path:
    target_column = str(winner_scores.get("target_column", "group") or "group")
    winner_features = _normalize_features(winner_scores.get("selected_features", []) or [])
    candidate_paths: List[Path] = []

    for raw_path in [
        winner_scores.get("data_path"),
        winner_scores.get("evaluation_data_path"),
        winner_scores.get("training_data_path"),
        phase1_scores.get("data_path"),
        phase1_scores.get("evaluation_data_path"),
        phase1_scores.get("training_data_path"),
        PROJECT_ROOT / "output" / "phase1" / "final" / "selected_features_final.csv",
    ]:
        if not raw_path:
            continue
        try:
            resolved = _resolve_existing_path(raw_path)
        except FileNotFoundError:
            continue
        if resolved not in candidate_paths:
            candidate_paths.append(resolved)

    if not candidate_paths:
        raise FileNotFoundError("No candidate data path found for Phase 2 objective-shift summary.")

    best_path = candidate_paths[0]
    best_hits = -1
    for candidate in candidate_paths:
        hits = _count_feature_hits(candidate, winner_features, target_column)
        if hits > best_hits:
            best_hits = hits
            best_path = candidate

    return best_path


def plot_phase2_objective_shift_summary(
    phase2_result_path: str,
    artifact_dir: str = "output/artifacts",
    ag_results_path: str = "output/phase1/artifacts/autogluon_training_results.json",
    feature_provenance_path: str = "",
    phase1_panel_scores_path: str = "",
    save_path: str = "output/figures/fig3_phase2_objective_shift_summary.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir_path = _resolve_existing_path(artifact_dir)
    winner_scores_path = _resolve_existing_path(
        phase2_result_path,
        artifact_dir_path / "phase2_winner_scores.json",
    )
    phase1_scores_path = _resolve_existing_path(
        phase1_panel_scores_path,
        artifact_dir_path / "phase1_panel_scores.json",
    )
    search_summary_path = _resolve_existing_path(artifact_dir_path / "phase2_search_summary.json")
    feature_provenance_resolved = _resolve_existing_path(
        feature_provenance_path,
        artifact_dir_path / "feature_provenance.json",
    )
    ag_results_resolved = _resolve_existing_path(
        ag_results_path,
        PROJECT_ROOT / "output" / "phase1" / "artifacts" / "autogluon_training_results.json",
        artifact_dir_path / "autogluon_training_results.json",
    )
    taxonomy_map_path = _resolve_existing_path(PROJECT_ROOT / "storage" / "taxonomy_map.json")

    phase1_scores = _read_json(phase1_scores_path)
    winner_scores = _read_json(winner_scores_path)
    legacy_radar_paths = _write_legacy_radar_4d(
        phase1_scores=phase1_scores,
        winner_scores=winner_scores,
        save_path=output_path,
        journal_theme=journal_theme,
    )
    search_summary = _read_json(search_summary_path)
    provenance_payload = _read_json(feature_provenance_resolved)
    taxonomy_map = _load_taxonomy_map(taxonomy_map_path)
    phase0_context = _resolve_phase0_context(winner_scores, search_summary)

    phase1_data_path = _resolve_best_data_path(
        winner_scores=winner_scores,
        phase1_scores=phase1_scores,
    )

    radar_csv = _build_radar_csv(
        phase1_scores=phase1_scores,
        winner_scores=winner_scores,
        output_dir=asset_dir,
        output_stem=output_stem.name,
    )
    contribution_summary = _load_existing_contribution_summary(
        candidate_dirs=[artifact_dir_path, asset_dir],
        winner_features=_normalize_features(winner_scores.get("selected_features", []) or []),
    )
    if contribution_summary is None:
        contribution_summary = _repair_winner_feature_contributions(
            winner_scores=winner_scores,
            search_summary=search_summary,
            phase0_output=phase0_context,
            provenance_payload=provenance_payload,
            taxonomy_map=taxonomy_map,
            data_path=phase1_data_path,
            ag_results_path=ag_results_resolved,
            output_dir=artifact_dir_path,
        )
    contrib_csv = Path(contribution_summary["contribution_csv"])

    # Some valid Phase 2 runs do not have per-feature contribution rows (for
    # example, when every candidate has zero biological support).  The 4D
    # panel profile is still well-defined from the stored objective scores;
    # do not discard that figure merely because the optional side track is
    # empty.  Render the radar-only output and let the report record that the
    # contribution track was unavailable.
    if not contrib_csv.exists() or contrib_csv.stat().st_size <= 1:
        saved_paths = _save_r_figure_bundle(str(output_path))
        profiles = _build_legacy_radar_profiles(
            phase1_scores=phase1_scores,
            winner_scores=winner_scores,
        )
        for path in saved_paths:
            plot_radar_comparison(
                profiles_dict=profiles,
                save_path=path,
                title="4D Panel Performance Profile",
                journal_theme=journal_theme,
            )
        metadata = {
            "figure_title": "Phase 2 objective shift and winner feature contribution",
            "phase2_result_path": str(winner_scores_path),
            "phase2_search_summary_path": str(search_summary_path),
            "phase1_panel_scores_path": str(phase1_scores_path),
            "feature_provenance_path": str(feature_provenance_resolved),
            "ag_results_path": str(ag_results_resolved),
            "evaluation_data_path": str(phase1_data_path),
            "radar_csv": str(radar_csv),
            "contrib_csv": str(contrib_csv),
            "contribution_track_available": False,
            "asset_dir": str(asset_dir),
            "winner_feature_count": int(len(_normalize_features(winner_scores.get("selected_features", []) or []))),
            "auxiliary_outputs": saved_paths[1:] + [str(radar_csv), str(contrib_csv)] + legacy_radar_paths,
            "legacy_radar_4d_paths": legacy_radar_paths,
        }
        return saved_paths[0], metadata

    theme_json_path = (
        write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json"))
        if journal_theme else ""
    )
    _run_r_script(radar_csv=radar_csv, contrib_csv=contrib_csv, output_stem=output_stem, theme_json_path=theme_json_path)
    saved_paths = _save_r_figure_bundle(save_path)

    metadata = {
        "figure_title": "Phase 2 objective shift and winner feature contribution",
        "phase2_result_path": str(winner_scores_path),
        "phase2_search_summary_path": str(search_summary_path),
        "phase1_panel_scores_path": str(phase1_scores_path),
        "feature_provenance_path": str(feature_provenance_resolved),
        "ag_results_path": str(ag_results_resolved),
        "evaluation_data_path": str(phase1_data_path),
        "radar_csv": str(radar_csv),
        "contrib_csv": str(contrib_csv),
        "asset_dir": str(asset_dir),
        "winner_feature_count": int(len(_normalize_features(winner_scores.get("selected_features", []) or []))),
        "auxiliary_outputs": saved_paths[1:] + [str(radar_csv), str(contrib_csv)] + legacy_radar_paths,
        "legacy_radar_4d_paths": legacy_radar_paths,
    }
    return saved_paths[0], metadata
