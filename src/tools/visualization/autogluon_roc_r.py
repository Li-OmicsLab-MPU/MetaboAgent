from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.tools.visualization.journal_theme import write_theme_json


def _save_r_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _run_r_autogluon_roc(
    roc_csv: Path,
    summary_csv: Path,
    output_stem: Path,
    title: str,
    subtitle: str,
    mode: str,
    message: str = "",
    theme_json_path: str = "",
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_autogluon_roc.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plotting script not found: {r_script}")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(project_root / ".r_libs")
    command = [
        "Rscript",
        str(r_script),
        mode,
        str(roc_csv),
        str(summary_csv),
        str(output_stem.resolve()),
        title,
        subtitle,
        message,
        theme_json_path,
    ]
    subprocess.run(command, check=True, cwd=str(project_root), env=env)


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_label_vector(ag_results: Dict, data_path: str, target_column: str) -> np.ndarray:
    data_path_resolved = Path(data_path)
    if not data_path_resolved.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path_resolved)
    split_strategy = str(ag_results.get("split_strategy", ""))
    test_set_role = str(ag_results.get("test_set_role", ""))
    y: Optional[np.ndarray] = None

    if test_set_role == "external_holdout" and split_strategy.startswith("external_holdout:"):
        holdout_path = split_strategy.split(":", 1)[1].strip()
        if holdout_path and Path(holdout_path).exists():
            holdout_df = pd.read_csv(holdout_path)
            holdout_target_col = target_column
            if holdout_target_col not in holdout_df.columns:
                if "target" in holdout_df.columns:
                    holdout_target_col = "target"
                elif "Group" in holdout_df.columns:
                    holdout_target_col = "Group"
                elif "group" in holdout_df.columns:
                    holdout_target_col = "group"
                else:
                    holdout_target_col = None
            if holdout_target_col is not None:
                y = holdout_df[holdout_target_col].to_numpy()

    if y is None:
        test_set_size = ag_results.get("test_set_size", None)
        df_target_col = target_column
        if df_target_col not in df.columns:
            if "target" in df.columns:
                df_target_col = "target"
            elif "Group" in df.columns:
                df_target_col = "Group"
            elif "group" in df.columns:
                df_target_col = "group"
            else:
                raise ValueError(f"Target column not found in dataframe: {target_column}")

        if test_set_size is not None and test_set_size < len(df):
            x_df = df.drop(columns=[df_target_col])
            y_full = df[df_target_col]
            _, _, _, y_test = train_test_split(
                x_df, y_full, test_size=test_set_size, stratify=y_full, random_state=42
            )
            y = y_test.to_numpy()
        else:
            y = df[df_target_col].to_numpy()

    test_set_size = ag_results.get("test_set_size", None)
    if isinstance(test_set_size, int) and len(y) != test_set_size and len(y) > test_set_size:
        y = np.array(y[-test_set_size:])
    return np.asarray(y)


def _extract_models_data(ag_results: Dict) -> List[Dict]:
    models_data: List[Dict] = []

    if "models" in ag_results:
        for model_name, model_info in ag_results["models"].items():
            if "predictions" in model_info and "auc" in model_info:
                models_data.append(
                    {
                        "name": model_name,
                        "predictions": model_info["predictions"],
                        "auc": float(model_info["auc"]),
                    }
                )
    elif "leaderboard" in ag_results:
        leaderboard = ag_results["leaderboard"]
        if isinstance(leaderboard, list):
            for model_info in leaderboard[:5]:
                if "model" in model_info and "score_val" in model_info:
                    models_data.append(
                        {
                            "name": model_info["model"],
                            "predictions": model_info.get("predictions"),
                            "auc": float(model_info["score_val"]),
                        }
                    )
    elif "leaderboard_top5" in ag_results:
        leaderboard = ag_results["leaderboard_top5"]
        if isinstance(leaderboard, list):
            for model_info in leaderboard[:5]:
                auc_score = (
                    model_info.get("score_test")
                    or model_info.get("score_holdout")
                    or model_info.get("score_val")
                )
                if "model" in model_info and auc_score is not None:
                    models_data.append(
                        {
                            "name": model_info["model"],
                            "predictions": model_info.get("predictions"),
                            "auc": float(auc_score),
                        }
                    )

    models_data = sorted(models_data, key=lambda item: item["auc"], reverse=True)
    return models_data[:5]


def _build_placeholder(save_path: str, title: str, message: str, journal_theme: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict]:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    roc_csv = output_path.parent / "autogluon_roc_curves.csv"
    summary_csv = output_path.parent / "autogluon_roc_summary.csv"
    pd.DataFrame(columns=["fpr", "tpr", "model_name", "legend_label", "auc", "rank", "is_best"]).to_csv(
        roc_csv, index=False
    )
    pd.DataFrame(columns=["model_name", "legend_label", "auc", "rank", "is_best"]).to_csv(summary_csv, index=False)
    _run_r_autogluon_roc(
        roc_csv=roc_csv,
        summary_csv=summary_csv,
        output_stem=output_stem,
        title=title,
        subtitle="",
        mode="placeholder",
        message=message,
        theme_json_path=theme_json_path,
    )
    saved_paths = _save_r_figure_bundle(save_path)
    metadata = {
        "figure_title": title,
        "mode": "placeholder",
        "message": message,
        "data_csv": str(roc_csv),
        "summary_csv": str(summary_csv),
        "theme_json_path": theme_json_path,
        "auxiliary_outputs": saved_paths[1:],
    }
    return saved_paths[0], metadata


def plot_autogluon_roc(
    ag_results_path: str = "data/autogluon_training_results.json",
    data_path: str = "data/cleaned_data.csv",
    target_column: str = "Group",
    save_path: str = "output/figures/fig2a_autogluon_roc.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    del dpi, figsize, layout_kwargs

    figure_title = "AutoGluon Multi-Model ROC Comparison"
    figure_subtitle = "Phase 1 feature matrix evaluated across top baseline learners"

    if not os.path.exists(ag_results_path):
        return _build_placeholder(
            save_path=save_path,
            title=figure_title,
            message=f"Results file not found: {ag_results_path}",
            journal_theme=journal_theme,
        )

    ag_results = _load_json(ag_results_path)
    models_data = _extract_models_data(ag_results)
    if not models_data:
        return _build_placeholder(
            save_path=save_path,
            title=figure_title,
            message="No valid model data found in AutoGluon results.",
            journal_theme=journal_theme,
        )

    if not all(model["predictions"] is not None for model in models_data):
        return _build_placeholder(
            save_path=save_path,
            title=figure_title,
            message="Prediction probabilities are missing; please re-run Phase 1 AutoGluon training.",
            journal_theme=journal_theme,
        )

    y = _resolve_label_vector(ag_results=ag_results, data_path=data_path, target_column=target_column)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    roc_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for rank, model in enumerate(models_data, start=1):
        predictions = np.asarray(model["predictions"], dtype=float)
        fpr, tpr, _ = roc_curve(y_encoded, predictions)
        roc_auc = float(auc(fpr, tpr))
        legend_label = f"{model['name']} (AUC = {roc_auc:.3f})"
        for x_val, y_val in zip(fpr, tpr):
            roc_rows.append(
                {
                    "fpr": float(x_val),
                    "tpr": float(y_val),
                    "model_name": model["name"],
                    "legend_label": legend_label,
                    "auc": roc_auc,
                    "rank": rank,
                    "is_best": rank == 1,
                }
            )
        summary_rows.append(
            {
                "model_name": model["name"],
                "legend_label": legend_label,
                "auc": roc_auc,
                "rank": rank,
                "is_best": rank == 1,
            }
        )

    roc_df = pd.DataFrame(roc_rows)
    summary_df = pd.DataFrame(summary_rows)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    theme_json_path = write_theme_json(journal_theme, output_stem.with_name(f"{output_stem.name}_theme.json")) if journal_theme else ""
    roc_csv = output_path.parent / "autogluon_roc_curves.csv"
    summary_csv = output_path.parent / "autogluon_roc_summary.csv"
    roc_df.to_csv(roc_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    _run_r_autogluon_roc(
        roc_csv=roc_csv,
        summary_csv=summary_csv,
        output_stem=output_stem,
        title=figure_title,
        subtitle=figure_subtitle,
        mode="success",
        theme_json_path=theme_json_path,
    )
    saved_paths = _save_r_figure_bundle(save_path)
    metadata = {
        "figure_title": figure_title,
        "ag_results_path": ag_results_path,
        "data_path": data_path,
        "target_column": target_column,
        "roc_csv": str(roc_csv),
        "summary_csv": str(summary_csv),
        "theme_json_path": theme_json_path,
        "best_model": summary_df.iloc[0]["model_name"] if not summary_df.empty else "",
        "n_models": int(len(summary_df)),
        "class_labels": [str(item) for item in encoder.classes_.tolist()],
        "auxiliary_outputs": saved_paths[1:],
    }
    return saved_paths[0], metadata
