from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve

from src.tools.visualization.journal_theme import write_theme_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _absolute_output_stem(output_stem: Path) -> str:
    """Keep R outputs in the caller's run directory, not PROJECT_ROOT."""
    return str(output_stem.resolve())


def _resolve_existing_path(*candidates: Any) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        raw = Path(str(candidate))
        probe_paths = [raw]
        if not raw.is_absolute():
            probe_paths.append(PROJECT_ROOT / raw)
        for probe in probe_paths:
            if probe.exists():
                return probe.resolve()
    raise FileNotFoundError(f"Unable to resolve existing path from candidates: {candidates}")


def _read_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_phase2_payload(phase2_result_path: str) -> Tuple[Path, Dict[str, Any]]:
    resolved = _resolve_existing_path(phase2_result_path)
    return resolved, _read_json(resolved)


def _candidate_runtime_roots(phase2_result_path: Path) -> List[Path]:
    """Return run-scoped roots before falling back to the project root.

    Phase 2 winner JSON files are mirrored in canonical and legacy artifact
    directories.  The evaluation prediction CSVs, however, are written under
    the run's ``output/audit`` directory.  Resolving that directory explicitly
    prevents plots from silently using a stale global/legacy payload.
    """
    roots: List[Path] = []
    for env_name in ("METABOAGENT_RUNTIME_ROOT", "METABOAGENT_RUN_ROOT"):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            roots.append(Path(raw).expanduser())
    resolved = Path(phase2_result_path).resolve()
    roots.extend([resolved.parent, *resolved.parents])
    roots.extend([Path.cwd(), PROJECT_ROOT])
    unique: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(Path(key))
    return unique


def _canonical_prediction_path(phase2_result_path: Path, scope: str) -> Optional[Path]:
    filename = {
        "cv": "evaluation_predictions_oof.csv",
        "holdout": "evaluation_predictions_holdout.csv",
    }.get(scope)
    if not filename:
        return None
    for root in _candidate_runtime_roots(phase2_result_path):
        candidate = root / "output" / "audit" / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def _prediction_frame_from_canonical_csv(path: Path) -> Tuple[pd.DataFrame, str]:
    """Normalize the audit prediction schema used by all P0 figures."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Canonical prediction artifact is empty: {path}")
    true_column = "y_true" if "y_true" in frame.columns else "true_label_encoded"
    probability_column = "pred_probability"
    if probability_column not in frame.columns:
        probability_column = "pred_positive" if "pred_positive" in frame.columns else "pred_proba_class1"
    if true_column not in frame.columns or probability_column not in frame.columns:
        raise ValueError(
            f"Canonical prediction artifact lacks y_true/pred_probability columns: {path}"
        )
    output = frame.copy()
    output["y_true_positive"] = pd.to_numeric(output[true_column], errors="raise").astype(int)
    output["pred_positive"] = pd.to_numeric(output[probability_column], errors="raise").astype(float)
    if "fold" in output.columns:
        output["fold"] = output["fold"].astype(str)
    return output, "canonical_audit_prediction"


def _canonical_prediction_frame(phase2_result_path: Path, scope: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    path = _canonical_prediction_path(phase2_result_path, scope)
    if path is None:
        return None, None
    frame, _ = _prediction_frame_from_canonical_csv(path)
    return frame, path


def _load_phase2_clinical_utility_artifact(phase2_result_path: Path) -> Dict[str, Any]:
    artifact_path = phase2_result_path.parent / "phase2_clinical_utility.json"
    if not artifact_path.exists():
        return {}
    return _read_json(artifact_path)


def _extract_cv_predictions(phase2_payload: Dict[str, Any]) -> Dict[str, Any]:
    cv_predictions = (
        phase2_payload.get("final_result", {}).get("cv_predictions")
        or phase2_payload.get("cv_predictions")
        or {}
    )
    if not isinstance(cv_predictions, dict) or not cv_predictions.get("predictions"):
        raise ValueError("cv_predictions are missing from the Phase 2 artifact.")
    return cv_predictions


def _extract_prediction_frame_from_payload(
    prediction_payload: Dict[str, Any],
    payload_name: str,
) -> Tuple[pd.DataFrame, int, str]:
    if not isinstance(prediction_payload, dict) or not prediction_payload.get("predictions"):
        raise ValueError(f"{payload_name} are missing prediction records.")
    predictions_df = pd.DataFrame(prediction_payload["predictions"])
    if "true_label_encoded" not in predictions_df.columns:
        raise ValueError(f"{payload_name} do not contain true_label_encoded.")
    label_names = [str(item) for item in (prediction_payload.get("label_names") or [])]
    y_encoded = predictions_df["true_label_encoded"].astype(int).to_numpy()
    positive_class_idx, positive_class_name = _detect_positive_class(label_names, y_encoded)
    probability_column = "pred_proba_class0" if positive_class_idx == 0 else "pred_proba_class1"
    if probability_column not in predictions_df.columns:
        raise ValueError(f"{payload_name} do not contain {probability_column}.")
    output_df = predictions_df.copy()
    output_df["y_true_positive"] = (output_df["true_label_encoded"].astype(int) == positive_class_idx).astype(int)
    output_df["pred_positive"] = output_df[probability_column].astype(float)
    return output_df, positive_class_idx, positive_class_name


def _resolve_phase2_external_artifacts(phase2_result_path: Path) -> Tuple[Path, Path]:
    artifact_dir = phase2_result_path.parent
    return (
        artifact_dir / "phase2_external_validation.json",
        artifact_dir / "phase2_external_winner_predictions.csv",
    )


def _load_phase2_external_prediction_frame(phase2_result_path: Path) -> Tuple[pd.DataFrame, Path, str]:
    external_json_path, external_csv_path = _resolve_phase2_external_artifacts(phase2_result_path)

    if external_json_path.exists():
        external_payload = _read_json(external_json_path)
        winner_prediction_payload = external_payload.get("winner_prediction_payload") or {}
        predictions_df, _, positive_class_name = _extract_prediction_frame_from_payload(
            winner_prediction_payload,
            "winner_prediction_payload from phase2_external_validation.json",
        )
        return predictions_df, external_json_path, positive_class_name

    if external_csv_path.exists():
        predictions_df = pd.read_csv(external_csv_path)
        if predictions_df.empty:
            raise ValueError(
                f"External holdout prediction CSV is empty: {external_csv_path}"
            )
        if "true_label_encoded" not in predictions_df.columns:
            raise ValueError(
                "phase2_external_winner_predictions.csv does not contain true_label_encoded."
            )
        label_names: List[str] = []
        y_encoded = predictions_df["true_label_encoded"].astype(int).to_numpy()
        positive_class_idx, positive_class_name = _detect_positive_class(label_names, y_encoded)
        probability_column = "pred_proba_class0" if positive_class_idx == 0 else "pred_proba_class1"
        if probability_column not in predictions_df.columns:
            raise ValueError(
                f"phase2_external_winner_predictions.csv does not contain {probability_column}."
            )
        output_df = predictions_df.copy()
        output_df["y_true_positive"] = (output_df["true_label_encoded"].astype(int) == positive_class_idx).astype(int)
        output_df["pred_positive"] = output_df[probability_column].astype(float)
        return output_df, external_csv_path, positive_class_name

    raise FileNotFoundError(
        "External holdout ROC artifacts were not found alongside the Phase 2 result. "
        f"Expected either {external_json_path.name} or {external_csv_path.name} in {phase2_result_path.parent}."
    )


def _load_internal_holdout_prediction_frame(
    phase2_result_path: Path,
    phase2_payload: Dict[str, Any],
) -> Tuple[pd.DataFrame, Path, str, str]:
    """Load the one fixed internal-holdout probability set used by all plots.

    Archived predictions always take precedence over model refitting.  This
    keeps ROC, calibration and DCA on the identical patients and probabilities
    and makes historical runs reproducible even when the original audit CSV
    predates the canonical ``output/audit`` layout.
    """
    # The Phase 2 JSON is the authoritative source for the locked panel
    # prediction set. The audit CSV is a derived mirror and can be produced by
    # a later export step. Prefer the Phase 2 payload so all Phase 3 figures
    # consume the exact probabilities emitted by panel evaluation.
    try:
        predictions_df, prediction_path, positive_class_name = _load_phase2_external_prediction_frame(
            phase2_result_path
        )
        return (
            predictions_df,
            prediction_path,
            positive_class_name,
            "archived_phase2_holdout_prediction",
        )
    except FileNotFoundError:
        canonical_predictions, canonical_path = _canonical_prediction_frame(
            phase2_result_path,
            "holdout",
        )
        if canonical_predictions is not None and canonical_path is not None:
            return canonical_predictions, canonical_path, "1", "canonical_audit_prediction"
        predictions_df, prediction_path, positive_class_name = _recompute_phase2_external_prediction_frame(
            phase2_result_path,
            phase2_payload,
        )
        return predictions_df, prediction_path, positive_class_name, "phase2_recomputed_prediction"



def _resolve_winner_model_family(phase2_payload: Dict[str, Any]) -> str:
    candidates = [
        phase2_payload.get("selected_model"),
        phase2_payload.get("final_result", {}).get("selected_model"),
        phase2_payload.get("champion_model_family"),
    ]
    for candidate in candidates:
        raw = str(candidate or "").strip()
        if not raw:
            continue
        lowered = raw.lower()
        if "weighteden" in lowered:
            continue
        if "lightgbm" in lowered or "lgbm" in lowered:
            return "LightGBM"
        if "xgboost" in lowered or "xgb" in lowered:
            return "XGBoost"
        if "randomforest" in lowered or "random forest" in lowered:
            return "RandomForest"
        if "catboost" in lowered:
            return "CatBoost"
        return raw
    return "RandomForest"


def _recompute_phase2_external_prediction_frame(
    phase2_result_path: Path,
    phase2_payload: Dict[str, Any],
) -> Tuple[pd.DataFrame, Path, str]:
    from src.tools.analysis.pareto_evaluator import calculate_f_perf

    features = (
        phase2_payload.get("final_result", {}).get("features")
        or phase2_payload.get("selected_features")
        or []
    )
    if not features:
        raise ValueError("Phase 2 payload does not contain selected_features/final_result.features for holdout recomputation.")

    training_data_path = _resolve_existing_path(phase2_payload.get("data_path"))
    target_column = str(phase2_payload.get("target_column") or "group").strip() or "group"
    holdout_data_path = training_data_path.parent / "selected_features_holdout.csv"
    if not holdout_data_path.exists():
        raise FileNotFoundError(
            f"selected_features_holdout.csv not found next to Phase 2 training matrix: {holdout_data_path}"
        )

    champion_model_family = _resolve_winner_model_family(phase2_payload)
    ag_results_candidates = [
        phase2_result_path.parent / "autogluon_training_results.json",
        PROJECT_ROOT / "output" / "phase1" / "artifacts" / "autogluon_training_results.json",
        PROJECT_ROOT / "data" / "autogluon_training_results.json",
    ]
    ag_results_path = str(_resolve_existing_path(*ag_results_candidates))

    _, predictions_payload = calculate_f_perf(
        data_path=str(training_data_path),
        target_column=target_column,
        features_list=[str(feature) for feature in features],
        champion_model_family=champion_model_family,
        ag_results_path=ag_results_path,
        metric="roc_auc",
        use_phase1_config=True,
        return_predictions=True,
        holdout_data_path=str(holdout_data_path),
        evaluation_mode="holdout_only",
    )
    predictions_df, _, positive_class_name = _extract_prediction_frame_from_payload(
        predictions_payload,
        "recomputed_holdout_predictions",
    )
    return predictions_df, holdout_data_path, positive_class_name


def _detect_positive_class(label_names: List[str], y_true_encoded: np.ndarray) -> Tuple[int, str]:
    if not label_names or len(label_names) != 2:
        class_counts = np.bincount(y_true_encoded.astype(int))
        if len(class_counts) == 2:
            minority_class = int(np.argmin(class_counts))
            return minority_class, str(minority_class)
        return 1, "1"

    disease_keywords = [
        "disease",
        "case",
        "patient",
        "cd",
        "cancer",
        "tumor",
        "positive",
        "sick",
        "ill",
        "affected",
    ]
    for idx, label in enumerate(label_names):
        if any(keyword in str(label).lower() for keyword in disease_keywords):
            return idx, str(label)

    class_counts = np.bincount(y_true_encoded.astype(int))
    if len(class_counts) == 2:
        minority_class = int(np.argmin(class_counts))
        ratio = class_counts.max() / (class_counts.min() + 1e-10)
        if ratio >= 2.0:
            return minority_class, str(label_names[minority_class])

    return 1, str(label_names[1])



def _extract_prediction_frame(phase2_payload: Dict[str, Any]) -> Tuple[pd.DataFrame, int, str]:
    cv_predictions = _extract_cv_predictions(phase2_payload)
    return _extract_prediction_frame_from_payload(cv_predictions, "cv_predictions")


def _resolve_feature_count(phase2_payload: Dict[str, Any]) -> int:
    features = (
        phase2_payload.get("final_result", {}).get("features")
        or phase2_payload.get("selected_features")
        or []
    )
    return len(features)


def _english_scenario_label(raw_value: Any) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return "Primary care screening"
    if "(" in raw and ")" in raw:
        inner = raw.split("(", 1)[1].rsplit(")", 1)[0].strip()
        if inner and any("a" <= char.lower() <= "z" for char in inner):
            return inner
    if all(ord(char) < 128 for char in raw):
        return raw
    return "Primary care screening"


def _write_csv(rows: List[Dict[str, Any]], file_path: Path) -> Path:
    pd.DataFrame(rows).to_csv(file_path, index=False)
    return file_path


def _write_summary_csv(summary: Dict[str, Any], file_path: Path) -> Path:
    pd.DataFrame([summary]).to_csv(file_path, index=False)
    return file_path


def _run_r_script(script_name: str, args: List[str]) -> None:
    r_script = PROJECT_ROOT / "src" / "tools" / "visualization" / "r" / script_name
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(PROJECT_ROOT / ".r_libs")
    subprocess.run(
        ["Rscript", str(r_script), *args],
        check=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _safe_quantile(values: List[float], quantile: float, fallback: float) -> float:
    if not values:
        return fallback
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _build_stratified_bootstrap_index(
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)
    if len(positive_idx) == 0 or len(negative_idx) == 0:
        return np.arange(len(y_true), dtype=int)
    boot_positive = rng.choice(positive_idx, size=len(positive_idx), replace=True)
    boot_negative = rng.choice(negative_idx, size=len(negative_idx), replace=True)
    bootstrap_index = np.concatenate([boot_positive, boot_negative]).astype(int)
    rng.shuffle(bootstrap_index)
    return bootstrap_index


def _build_roc_exports_from_predictions(
    predictions_df: pd.DataFrame,
    feature_count: int,
    positive_class_name: str,
    output_dir: Path,
    output_stem: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    y_true = predictions_df["y_true_positive"].to_numpy(dtype=int)
    y_prob = predictions_df["pred_positive"].to_numpy(dtype=float)
    pooled_fpr, pooled_tpr, _ = roc_curve(y_true, y_prob)
    pooled_auc = float(auc(pooled_fpr, pooled_tpr))
    fpr_grid = np.linspace(0.0, 1.0, 201)
    pooled_tpr_grid = np.interp(fpr_grid, pooled_fpr, pooled_tpr)

    fold_aucs: List[float] = []
    if "fold" in predictions_df.columns:
        fold_series = predictions_df["fold"].dropna()
        for fold_id in sorted(fold_series.unique()):
            fold_df = predictions_df[predictions_df["fold"] == fold_id].copy()
            if fold_df["y_true_positive"].nunique() < 2:
                continue
            fold_fpr, fold_tpr, _ = roc_curve(
                fold_df["y_true_positive"].to_numpy(dtype=int),
                fold_df["pred_positive"].to_numpy(dtype=float),
            )
            fold_auc = float(auc(fold_fpr, fold_tpr))
            fold_aucs.append(fold_auc)

    rng = np.random.default_rng(42)
    bootstrap_aucs: List[float] = []
    bootstrap_tprs: List[np.ndarray] = []
    n_bootstrap = 500
    for _ in range(n_bootstrap):
        bootstrap_index = _build_stratified_bootstrap_index(y_true, rng)
        boot_y_true = y_true[bootstrap_index]
        boot_y_prob = y_prob[bootstrap_index]
        if np.unique(boot_y_true).size < 2:
            continue
        boot_fpr, boot_tpr, _ = roc_curve(boot_y_true, boot_y_prob)
        bootstrap_tprs.append(np.interp(fpr_grid, boot_fpr, boot_tpr))
        bootstrap_aucs.append(float(auc(boot_fpr, boot_tpr)))

    tpr_lower = np.quantile(np.vstack(bootstrap_tprs), 0.025, axis=0) if bootstrap_tprs else pooled_tpr_grid
    tpr_upper = np.quantile(np.vstack(bootstrap_tprs), 0.975, axis=0) if bootstrap_tprs else pooled_tpr_grid

    roc_rows: List[Dict[str, Any]] = [
        {
            "fpr": float(fpr_value),
            "tpr": float(tpr_value),
            "tpr_lower": float(lower_value),
            "tpr_upper": float(upper_value),
        }
        for fpr_value, tpr_value, lower_value, upper_value in zip(
            fpr_grid,
            pooled_tpr_grid,
            tpr_lower,
            tpr_upper,
        )
    ]

    roc_csv = _write_csv(roc_rows, output_dir / f"{output_stem}_roc_curve.csv")
    summary_csv = _write_summary_csv(
        {
            "feature_count": feature_count,
            "positive_class_name": positive_class_name,
            "pooled_auc": pooled_auc,
            "pooled_auc_ci_low": _safe_quantile(bootstrap_aucs, 0.025, pooled_auc),
            "pooled_auc_ci_high": _safe_quantile(bootstrap_aucs, 0.975, pooled_auc),
            "cv_auc": float(np.mean(fold_aucs)) if fold_aucs else pooled_auc,
            "cv_auc_sd": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
            "n_bootstrap": int(len(bootstrap_aucs)),
            "n_samples": int(len(predictions_df)),
        },
        output_dir / f"{output_stem}_roc_summary.csv",
    )
    metadata = {
        "roc_csv": str(roc_csv),
        "summary_csv": str(summary_csv),
        "pooled_auc": pooled_auc,
        "positive_class_name": positive_class_name,
        "feature_count": feature_count,
    }
    return roc_csv, summary_csv, metadata


def _align_roc_summary_to_audit(summary_csv: Path, phase2_result_path: Path, scope: str) -> None:
    """Use the canonical performance-uncertainty CI in the plotted label.

    Recomputing a second bootstrap inside the figure renderer can differ by a
    few thousandths from the archived 2,000-resample audit estimate.  The
    report and figure must expose exactly the same point estimate and CI.
    """
    audit_path = None
    for root in _candidate_runtime_roots(phase2_result_path):
        candidate = root / "output" / "audit" / "performance_uncertainty.json"
        if candidate.exists():
            audit_path = candidate
            break
    if audit_path is None:
        return
    payload = _read_json(audit_path)
    block = payload.get("development_oof" if scope == "cv" else "internal_holdout") or {}
    point = block.get("point_estimate") or {}
    ci = block.get("bootstrap_ci") or {}
    auc_ci = ci.get("roc_auc_ci95") or []
    if len(auc_ci) < 2 or point.get("roc_auc") is None:
        return
    frame = pd.read_csv(summary_csv)
    if frame.empty:
        return
    frame.loc[0, "pooled_auc"] = float(point["roc_auc"])
    frame.loc[0, "pooled_auc_ci_low"] = float(auc_ci[0])
    frame.loc[0, "pooled_auc_ci_high"] = float(auc_ci[1])
    frame.to_csv(summary_csv, index=False)




def _build_roc_exports(
    phase2_payload: Dict[str, Any],
    output_dir: Path,
    output_stem: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    predictions_df, _, positive_class_name = _extract_prediction_frame(phase2_payload)
    feature_count = _resolve_feature_count(phase2_payload)
    return _build_roc_exports_from_predictions(
        predictions_df=predictions_df,
        feature_count=feature_count,
        positive_class_name=positive_class_name,
        output_dir=output_dir,
        output_stem=output_stem,
    )


def _compute_net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    n_samples = float(len(y_true))
    if n_samples <= 0:
        return 0.0, 0.0, 0.0

    pred_positive = (y_prob >= threshold).astype(int)
    tp = float(np.sum((pred_positive == 1) & (y_true == 1)))
    fp = float(np.sum((pred_positive == 1) & (y_true == 0)))
    prevalence = float(np.mean(y_true))
    odds = threshold / max(1.0 - threshold, 1e-12)

    model_nb = (tp / n_samples) - (fp / n_samples) * odds
    treat_all_nb = prevalence - (1.0 - prevalence) * odds
    treat_none_nb = 0.0
    return model_nb, treat_all_nb, treat_none_nb


def _build_dca_exports(
    phase2_payload: Dict[str, Any],
    clinical_payload: Dict[str, Any],
    output_dir: Path,
    output_stem: str,
    predictions_df: Optional[pd.DataFrame] = None,
    evaluation_scope: str = "internal_cv",
    prediction_source: str = "phase2_payload",
) -> Tuple[Path, Path, Dict[str, Any]]:
    if predictions_df is None:
        predictions_df, _, _ = _extract_prediction_frame(phase2_payload)
    feature_count = _resolve_feature_count(phase2_payload)
    y_true = predictions_df["y_true_positive"].to_numpy(dtype=int)
    y_prob = predictions_df["pred_positive"].to_numpy(dtype=float)

    dca_summary = clinical_payload.get("decision_curve_summary", {}) or {}
    scenario = clinical_payload.get("scenario", {}) or {}
    clinical_threshold_min = float(
        dca_summary.get(
            "threshold_grid_min",
            scenario.get("risk_thresholds", [0.10])[0] if scenario.get("risk_thresholds") else 0.10,
        )
    )
    clinical_threshold_max = float(
        dca_summary.get(
            "threshold_grid_max",
            scenario.get("risk_thresholds", [0.25])[-1] if scenario.get("risk_thresholds") else 0.25,
        )
    )
    if clinical_threshold_max <= clinical_threshold_min:
        clinical_threshold_min, clinical_threshold_max = 0.10, 0.25

    full_threshold_min, full_threshold_max = 0.0, 1.00
    thresholds = np.linspace(full_threshold_min, full_threshold_max, 501)
    curve_rows: List[Dict[str, Any]] = []
    model_values: List[float] = []
    for threshold in thresholds:
        if threshold <= 0.0 or threshold >= 1.0:
            continue
        model_nb, treat_all_nb, treat_none_nb = _compute_net_benefit(y_true, y_prob, float(threshold))
        model_values.append(model_nb)
        curve_rows.append(
            {
                "threshold": float(threshold),
                "winner_panel": float(model_nb),
                "treat_all": float(treat_all_nb),
                "treat_none": float(treat_none_nb),
            }
        )

    default_threshold = (
        (clinical_payload.get("recommended_threshold_summary", {}) or {}).get("selected_threshold")
        or scenario.get("default_action_threshold")
    )
    companion_threshold = (clinical_payload.get("data_driven_companion_threshold_summary", {}) or {}).get("selected_threshold")

    curve_csv = _write_csv(curve_rows, output_dir / f"{output_stem}_dca_curve.csv")
    clinical_rows = [
        row for row in curve_rows
        if clinical_threshold_min <= float(row["threshold"]) <= clinical_threshold_max
    ]
    # Report the peak that is relevant to the prespecified clinical decision
    # window. The global curve peak is retained for auditability, but it is
    # often driven by thresholds near zero and is not clinically interpretable
    # for a screening workflow.
    global_peak_row = max(curve_rows, key=lambda row: float(row["winner_panel"])) if curve_rows else {}
    clinical_peak_row = max(clinical_rows, key=lambda row: float(row["winner_panel"])) if clinical_rows else global_peak_row
    beneficial_rows = [
        row for row in clinical_rows
        if float(row["winner_panel"]) > max(float(row["treat_all"]), float(row["treat_none"]))
    ]
    better_none_rows = [
        row for row in clinical_rows
        if float(row["winner_panel"]) > float(row["treat_none"])
    ]
    better_all_rows = [
        row for row in clinical_rows
        if float(row["winner_panel"]) > float(row["treat_all"])
    ]

    def _contiguous_ranges(rows: List[Dict[str, Any]]) -> List[List[float]]:
        values = sorted(float(row["threshold"]) for row in rows)
        if not values:
            return []
        ranges: List[List[float]] = []
        start = previous = values[0]
        grid_step = float(thresholds[1] - thresholds[0]) if len(thresholds) > 1 else 0.002
        for value in values[1:]:
            if value - previous > grid_step * 1.5:
                ranges.append([start, previous])
                start = value
            previous = value
        ranges.append([start, previous])
        return ranges

    dca_evidence = {
        "evaluation_scope": evaluation_scope,
        "prediction_source": prediction_source,
        "n_samples": int(len(predictions_df)),
        "event_prevalence": float(np.mean(y_true)),
        "peak_net_benefit": float(clinical_peak_row.get("winner_panel", 0.0)),
        "peak_net_benefit_threshold": float(clinical_peak_row.get("threshold", np.nan)),
        "clinical_window_peak_net_benefit": float(clinical_peak_row.get("winner_panel", 0.0)),
        "clinical_window_peak_net_benefit_threshold": float(clinical_peak_row.get("threshold", np.nan)),
        "global_peak_net_benefit": float(global_peak_row.get("winner_panel", 0.0)),
        "global_peak_net_benefit_threshold": float(global_peak_row.get("threshold", np.nan)),
        "clinical_threshold_min": clinical_threshold_min,
        "clinical_threshold_max": clinical_threshold_max,
        "benefit_over_both_min_threshold": (
            float(min(row["threshold"] for row in beneficial_rows)) if beneficial_rows else np.nan
        ),
        "benefit_over_both_max_threshold": (
            float(max(row["threshold"] for row in beneficial_rows)) if beneficial_rows else np.nan
        ),
        "winner_better_than_treat_none_ranges": _contiguous_ranges(better_none_rows),
        "winner_better_than_treat_all_ranges": _contiguous_ranges(better_all_rows),
        "winner_better_than_both_ranges": _contiguous_ranges(beneficial_rows),
        "summary_type": "fixed_internal_holdout",
    }
    summary_csv = _write_summary_csv(
        {
            "feature_count": feature_count,
            **dca_evidence,
            "scenario_name": _english_scenario_label(scenario.get("name", "Primary care screening")),
            "threshold_min": full_threshold_min,
            "threshold_max": full_threshold_max,
            "clinical_threshold_min": clinical_threshold_min,
            "clinical_threshold_max": clinical_threshold_max,
            "default_threshold": float(default_threshold) if default_threshold is not None else np.nan,
            "companion_threshold": float(companion_threshold) if companion_threshold is not None else np.nan,
        },
        output_dir / f"{output_stem}_dca_summary.csv",
    )
    metadata = {
        "curve_csv": str(curve_csv),
        "summary_csv": str(summary_csv),
        "feature_count": feature_count,
        "threshold_min": full_threshold_min,
        "threshold_max": full_threshold_max,
        "evaluation_scope": evaluation_scope,
        "prediction_source": prediction_source,
        "dca_summary": dca_evidence,
    }
    return curve_csv, summary_csv, metadata


def _build_calibration_bin_rows(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[Dict[str, Any]], float]:
    calibration_df = pd.DataFrame({"y_true": y_true.astype(int), "y_prob": y_prob.astype(float)})
    quantile_bins = min(max(int(n_bins), 2), max(int(calibration_df["y_prob"].nunique()), 2))
    calibration_df["bin"] = pd.qcut(calibration_df["y_prob"], q=quantile_bins, duplicates="drop")
    grouped = calibration_df.groupby("bin", observed=False)

    bin_rows: List[Dict[str, Any]] = []
    absolute_errors: List[float] = []
    weighted_errors: List[float] = []

    for bin_index, (_, frame) in enumerate(grouped):
        if frame.empty:
            continue
        mean_pred = float(frame["y_prob"].mean())
        observed = float(frame["y_true"].mean())
        sample_count = int(len(frame))
        abs_error = abs(observed - mean_pred)
        left = float(frame["y_prob"].min())
        right = float(frame["y_prob"].max())
        bin_rows.append(
            {
                "bin_index": bin_index,
                "lower_bound": left,
                "upper_bound": right,
                "sample_count": sample_count,
                "mean_predicted_probability": mean_pred,
                "observed_event_rate": observed,
                "absolute_calibration_error": abs_error,
            }
        )
        absolute_errors.append(abs_error)
        weighted_errors.append(abs_error * sample_count)

    total_samples = max(int(len(calibration_df)), 1)
    ece = float(sum(weighted_errors) / total_samples) if weighted_errors else 0.0
    return bin_rows, ece


def _build_calibration_exports(
    phase2_payload: Dict[str, Any],
    clinical_payload: Dict[str, Any],
    output_dir: Path,
    output_stem: str,
    predictions_df: Optional[pd.DataFrame] = None,
    assessment_label_override: Optional[str] = None,
    evaluation_scope: str = "internal_cv",
    prediction_source: str = "phase2_payload",
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    canonical_prediction_source = predictions_df is not None
    if predictions_df is None:
        predictions_df, _, _ = _extract_prediction_frame(phase2_payload)
    feature_count = _resolve_feature_count(phase2_payload)
    prediction_export = predictions_df.loc[:, ["y_true_positive", "pred_positive"]].copy()
    prediction_export = prediction_export.rename(
        columns={
            "y_true_positive": "observed_label",
            "pred_positive": "predicted_probability",
        }
    )

    preferred_adjusted_summary = clinical_payload.get("probability_calibration_adjusted") or {}
    preferred_raw_summary = clinical_payload.get("probability_calibration_raw") or {}
    embedded_summary = clinical_payload.get("probability_calibration") or {}
    prevalence_context = clinical_payload.get("prevalence_context") or {}

    # When the audit prediction CSV is available, recompute every displayed
    # calibration metric from it.  This prevents a legacy Phase 2 clinical
    # utility payload from overwriting the current run's Brier score/curve.
    using_adjusted_summary = isinstance(preferred_adjusted_summary, dict) and bool(preferred_adjusted_summary.get("available"))
    if canonical_prediction_source:
        bin_rows, ece = _build_calibration_bin_rows(
            prediction_export["observed_label"].to_numpy(dtype=int),
            prediction_export["predicted_probability"].to_numpy(dtype=float),
            n_bins=10,
        )
        y_values = prediction_export["observed_label"].to_numpy(dtype=float)
        p_values = prediction_export["predicted_probability"].to_numpy(dtype=float)
        clipped_probabilities = np.clip(p_values, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped_probabilities / (1.0 - clipped_probabilities)).reshape(-1, 1)
        calibration_intercept = np.nan
        calibration_slope = np.nan
        if np.unique(y_values).size == 2:
            try:
                calibration_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
                calibration_model.fit(logits, y_values.astype(int))
                calibration_intercept = float(calibration_model.intercept_[0])
                calibration_slope = float(calibration_model.coef_[0][0])
            except (TypeError, ValueError):
                # Compatibility with older scikit-learn releases.
                calibration_model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
                calibration_model.fit(logits, y_values.astype(int))
                calibration_intercept = float(calibration_model.intercept_[0])
                calibration_slope = float(calibration_model.coef_[0][0])
        calibration_summary = {
            "available": True,
            "n_samples": int(len(prediction_export)),
            "n_bins": len(bin_rows),
            "binning_strategy": "quantile",
            "brier_score": float(np.mean((y_values - p_values) ** 2)),
            "expected_calibration_error": ece,
            "integrated_calibration_index": ece,
            "max_calibration_error": float(max((row["absolute_calibration_error"] for row in bin_rows), default=0.0)),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "calibration_bins": bin_rows,
        }
        assessment_label = assessment_label_override or "Internal OOF probability assessment"
        using_adjusted_summary = False
    elif using_adjusted_summary:
        calibration_summary = dict(preferred_adjusted_summary)
        assessment_label = "Target-population prevalence-adjusted assessment"
    elif isinstance(preferred_raw_summary, dict) and preferred_raw_summary:
        calibration_summary = dict(preferred_raw_summary)
        assessment_label = "Internal study-prevalence diagnostic"
    elif isinstance(embedded_summary, dict) and embedded_summary:
        calibration_summary = dict(embedded_summary)
        assessment_label = "Internal study-prevalence diagnostic"
    else:
        bin_rows, ece = _build_calibration_bin_rows(
            prediction_export["observed_label"].to_numpy(dtype=int),
            prediction_export["predicted_probability"].to_numpy(dtype=float),
            n_bins=10,
        )
        calibration_summary = {
            "available": True,
            "n_samples": int(len(prediction_export)),
            "n_bins": len(bin_rows),
            "binning_strategy": "quantile",
            "brier_score": float(
                np.mean(
                    (
                        prediction_export["observed_label"].to_numpy(dtype=float)
                        - prediction_export["predicted_probability"].to_numpy(dtype=float)
                    )
                    ** 2
                )
            ),
            "expected_calibration_error": ece,
            "integrated_calibration_index": ece,
            "max_calibration_error": float(max((row["absolute_calibration_error"] for row in bin_rows), default=0.0)),
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "calibration_bins": bin_rows,
        }
        assessment_label = "Internal study-prevalence diagnostic"

    bin_rows = calibration_summary.get("calibration_bins", []) or []
    if assessment_label_override:
        assessment_label = assessment_label_override
    if evaluation_scope == "internal_holdout":
        reported_prevalence = float(prediction_export["observed_label"].mean())
        prevalence_label = "Holdout prevalence"
    else:
        reported_prevalence = prevalence_context.get("target_prevalence", np.nan)
        prevalence_label = "Target prevalence"
    calibration_summary = {
        **calibration_summary,
        "evaluation_scope": evaluation_scope,
        "prediction_source": prediction_source,
        "reported_prevalence": reported_prevalence,
        "prevalence_label": prevalence_label,
    }
    prediction_csv = output_dir / f"{output_stem}_calibration_predictions.csv"
    prediction_export.to_csv(prediction_csv, index=False)
    bins_csv = _write_csv(bin_rows, output_dir / f"{output_stem}_calibration_bins.csv")
    summary_csv = _write_summary_csv(
        {
            "feature_count": feature_count,
            "n_samples": int(len(prediction_export)),
            "evaluation_scope": evaluation_scope,
            "prediction_source": prediction_source,
            "assessment_label": assessment_label,
            "brier_score": calibration_summary.get("brier_score", np.nan),
            "ici": calibration_summary.get(
                "integrated_calibration_index",
                calibration_summary.get("expected_calibration_error", np.nan),
            ),
            "slope": calibration_summary.get("calibration_slope", np.nan),
            "intercept": calibration_summary.get("calibration_intercept", np.nan),
            "target_prevalence": reported_prevalence,
            "reported_prevalence": reported_prevalence,
            "prevalence_label": prevalence_label,
            "n_positive": int(prediction_export["observed_label"].sum()),
            "n_negative": int(len(prediction_export) - prediction_export["observed_label"].sum()),
            "using_adjusted_summary": bool(using_adjusted_summary),
        },
        output_dir / f"{output_stem}_calibration_summary.csv",
    )
    metadata = {
        "prediction_csv": str(prediction_csv),
        "bins_csv": str(bins_csv),
        "summary_csv": str(summary_csv),
        "feature_count": feature_count,
        "assessment_label": assessment_label,
        "evaluation_scope": evaluation_scope,
        "prediction_source": prediction_source,
        "calibration_summary": calibration_summary,
    }
    return prediction_csv, bins_csv, summary_csv, metadata


def plot_final_roc_r(
    phase2_result_path: str,
    save_path: str = "output/figures/fig4a_final_roc.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    phase2_path, phase2_payload = _load_phase2_payload(phase2_result_path)
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    canonical_predictions, canonical_path = _canonical_prediction_frame(phase2_path, "cv")
    if canonical_predictions is not None:
        roc_csv, summary_csv, metadata = _build_roc_exports_from_predictions(
            canonical_predictions,
            _resolve_feature_count(phase2_payload),
            "1",
            asset_dir,
            output_stem.name,
        )
        metadata["prediction_source"] = str(canonical_path)
    else:
        roc_csv, summary_csv, metadata = _build_roc_exports(phase2_payload, asset_dir, output_stem.name)
        metadata["prediction_source"] = "phase2_payload"
    _align_roc_summary_to_audit(summary_csv, phase2_path, "cv")
    _run_r_script(
        "plot_phase3_final_roc.R",
        [
            str(roc_csv),
            str(summary_csv),
            _absolute_output_stem(output_stem),
            theme_json_path,
            "cv",
            "Winner Panel ROC Curve (Internal CV)",
        ],
    )
    saved_paths = _save_r_figure_bundle(save_path)
    return saved_paths[0], {
        "figure_title": "Final ROC Curve",
        "phase2_result_path": str(phase2_path),
        "theme_json_path": theme_json_path,
        "auxiliary_outputs": saved_paths[1:],
        **metadata,
    }


def plot_final_holdout_roc_r(
    phase2_result_path: str,
    save_path: str = "output/figures/single_panels/phase2_winner_holdout_roc.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    phase2_path, phase2_payload = _load_phase2_payload(phase2_result_path)
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    predictions_df, prediction_source_path, positive_class_name, prediction_source_kind = (
        _load_internal_holdout_prediction_frame(phase2_path, phase2_payload)
    )
    feature_count = _resolve_feature_count(phase2_payload)
    roc_csv, summary_csv, metadata = _build_roc_exports_from_predictions(
        predictions_df=predictions_df,
        feature_count=feature_count,
        positive_class_name=positive_class_name,
        output_dir=asset_dir,
        output_stem=output_stem.name,
    )
    _align_roc_summary_to_audit(summary_csv, phase2_path, "holdout")
    _run_r_script(
        "plot_phase3_final_roc.R",
        [
            str(roc_csv),
            str(summary_csv),
            _absolute_output_stem(output_stem),
            theme_json_path,
            "holdout",
            "Winner Panel ROC Curve (Internal Holdout)",
        ],
    )
    saved_paths = _save_r_figure_bundle(save_path)
    return saved_paths[0], {
        "figure_title": "Final Holdout ROC Curve",
        "phase2_result_path": str(phase2_path),
        "internal_holdout_prediction_source_path": str(prediction_source_path),
        "prediction_source": prediction_source_kind,
        "evaluation_scope": "internal_holdout",
        "theme_json_path": theme_json_path,
        "auxiliary_outputs": saved_paths[1:],
        **metadata,
    }


def plot_dca_r(
    phase2_result_path: str,
    save_path: str = "output/figures/fig4b_dca.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    phase2_path, phase2_payload = _load_phase2_payload(phase2_result_path)
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    clinical_payload = _load_phase2_clinical_utility_artifact(phase2_path)
    canonical_predictions, canonical_path = _canonical_prediction_frame(phase2_path, "cv")
    curve_csv, summary_csv, metadata = _build_dca_exports(
        phase2_payload,
        clinical_payload,
        asset_dir,
        output_stem.name,
        predictions_df=canonical_predictions,
        evaluation_scope="internal_cv",
        prediction_source=str(canonical_path) if canonical_path else "phase2_payload",
    )
    metadata["prediction_source"] = str(canonical_path) if canonical_path else "phase2_payload"
    _run_r_script(
        "plot_phase3_dca.R",
        [
            str(curve_csv),
            str(summary_csv),
            _absolute_output_stem(output_stem),
            theme_json_path,
            "Winner Panel Decision Curve (Internal CV)",
        ],
    )
    saved_paths = _save_r_figure_bundle(save_path)
    return saved_paths[0], {
        "figure_title": "Winner Panel Decision Curve (Internal CV)",
        "phase2_result_path": str(phase2_path),
        "theme_json_path": theme_json_path,
        "clinical_utility_available": bool(clinical_payload),
        "auxiliary_outputs": saved_paths[1:],
        **metadata,
    }


def plot_calibration_r(
    phase2_result_path: str,
    save_path: str = "output/figures/fig4e_calibration.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    phase2_path, phase2_payload = _load_phase2_payload(phase2_result_path)
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    clinical_payload = _load_phase2_clinical_utility_artifact(phase2_path)
    canonical_predictions, canonical_path = _canonical_prediction_frame(phase2_path, "cv")
    prediction_csv, bins_csv, summary_csv, metadata = _build_calibration_exports(
        phase2_payload,
        clinical_payload,
        asset_dir,
        output_stem.name,
        predictions_df=canonical_predictions,
        assessment_label_override="Internal OOF probability assessment",
        evaluation_scope="internal_cv",
        prediction_source=str(canonical_path) if canonical_path else "phase2_payload",
    )
    metadata["prediction_source"] = str(canonical_path) if canonical_path else "phase2_payload"
    _run_r_script(
        "plot_phase3_calibration.R",
        [
            str(prediction_csv),
            str(bins_csv),
            str(summary_csv),
            _absolute_output_stem(output_stem),
            theme_json_path,
            "Winner Panel Calibration Curve (Internal CV)",
        ],
    )
    saved_paths = _save_r_figure_bundle(save_path)
    return saved_paths[0], {
        "figure_title": "Winner Panel Calibration Curve (Internal CV)",
        "phase2_result_path": str(phase2_path),
        "theme_json_path": theme_json_path,
        "clinical_utility_available": bool(clinical_payload),
        "auxiliary_outputs": saved_paths[1:],
        "prediction_csv": str(prediction_csv),
        "bins_csv": str(bins_csv),
        "summary_csv": str(summary_csv),
        **metadata,
    }


def plot_final_holdout_dca_r(
    phase2_result_path: str,
    save_path: str = "output/figures/single_panels/phase2_winner_holdout_decision_curve.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Render DCA from the archived fixed internal-holdout probabilities."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    phase2_path, phase2_payload = _load_phase2_payload(phase2_result_path)
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    clinical_payload = _load_phase2_clinical_utility_artifact(phase2_path)
    predictions_df, source_path, _, source_kind = _load_internal_holdout_prediction_frame(
        phase2_path,
        phase2_payload,
    )
    curve_csv, summary_csv, metadata = _build_dca_exports(
        phase2_payload,
        clinical_payload,
        asset_dir,
        output_stem.name,
        predictions_df=predictions_df,
        evaluation_scope="internal_holdout",
        prediction_source=str(source_path),
    )
    _run_r_script(
        "plot_phase3_dca.R",
        [
            str(curve_csv),
            str(summary_csv),
            _absolute_output_stem(output_stem),
            theme_json_path,
            "Winner Panel Decision Curve (Internal Holdout)",
        ],
    )
    saved_paths = _save_r_figure_bundle(save_path)
    return saved_paths[0], {
        "figure_title": "Winner Panel Decision Curve (Internal Holdout)",
        "phase2_result_path": str(phase2_path),
        "internal_holdout_prediction_source_path": str(source_path),
        "prediction_source_kind": source_kind,
        "theme_json_path": theme_json_path,
        "clinical_utility_available": bool(clinical_payload),
        "auxiliary_outputs": saved_paths[1:],
        **metadata,
    }


def plot_final_holdout_calibration_r(
    phase2_result_path: str,
    save_path: str = "output/figures/single_panels/phase2_winner_holdout_calibration.pdf",
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Render calibration from the archived fixed internal-holdout probabilities."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    asset_root = output_path.parent.parent / "assets" if output_path.parent.name == "single_panels" else output_path.parent
    asset_dir = asset_root / f"{output_stem.name}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    phase2_path, phase2_payload = _load_phase2_payload(phase2_result_path)
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    clinical_payload = _load_phase2_clinical_utility_artifact(phase2_path)
    predictions_df, source_path, _, source_kind = _load_internal_holdout_prediction_frame(
        phase2_path,
        phase2_payload,
    )
    prediction_csv, bins_csv, summary_csv, metadata = _build_calibration_exports(
        phase2_payload,
        clinical_payload,
        asset_dir,
        output_stem.name,
        predictions_df=predictions_df,
        assessment_label_override="Fixed internal holdout probability assessment",
        evaluation_scope="internal_holdout",
        prediction_source=str(source_path),
    )
    _run_r_script(
        "plot_phase3_calibration.R",
        [
            str(prediction_csv),
            str(bins_csv),
            str(summary_csv),
            _absolute_output_stem(output_stem),
            theme_json_path,
            "Winner Panel Calibration Curve (Internal Holdout)",
        ],
    )
    saved_paths = _save_r_figure_bundle(save_path)
    return saved_paths[0], {
        "figure_title": "Winner Panel Calibration Curve (Internal Holdout)",
        "phase2_result_path": str(phase2_path),
        "internal_holdout_prediction_source_path": str(source_path),
        "prediction_source_kind": source_kind,
        "theme_json_path": theme_json_path,
        "clinical_utility_available": bool(clinical_payload),
        "auxiliary_outputs": saved_paths[1:],
        **metadata,
    }
