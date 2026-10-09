"""
Phase 3: 临床报告与可视化生成工具 (Actor-Critic Visual Auditing Loop)

核心功能:
1. plot_pareto_trajectory: 帕累托进化轨迹散点图
2. plot_radar_comparison: 多模型 4D 雷达对比图
3. 未来扩展: 其他可视化图表

ACVAL 架构 (Actor-Critic Visual Auditing Loop):
- Actor: 所有图表函数支持 layout_kwargs 参数化
- Critic: VisualAuditor 类审查图片并提供调整建议
- Orchestrator: 重试循环直到图表排版完美

Author: MetaboAgent Team
Date: 2026-04-24
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from src.tools.analysis import (
    build_threshold_metrics_table,
    build_probability_calibration_summary,
    summarize_decision_curve_ranges,
)
from src.tools.visualization.journal_theme import resolve_theme_color, write_theme_json
from src.utils.config_manager import get_config

# 引入排版神器：自动推开重叠的文字
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    print("Warning: adjustText not installed. Text overlap adjustment will be disabled.")
    print("Install with: pip install adjustText")
    ADJUST_TEXT_AVAILABLE = False


def _load_phase2_clinical_utility_artifact(phase2_result_path: Optional[str]) -> Dict:
    if not phase2_result_path:
        return {}
    artifact_dir = os.path.dirname(phase2_result_path)
    clinical_utility_path = os.path.join(artifact_dir, "phase2_clinical_utility.json")
    if not os.path.exists(clinical_utility_path):
        return {}
    try:
        with open(clinical_utility_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_canonical_audit_oof_predictions(phase2_result_path: Optional[str]):
    """Load the run-scoped OOF audit predictions for threshold plots."""
    roots = []
    for env_name in ("METABOAGENT_RUNTIME_ROOT", "METABOAGENT_RUN_ROOT"):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            roots.append(Path(raw))
    if phase2_result_path:
        resolved = Path(phase2_result_path).expanduser()
        roots.extend([resolved.parent, *resolved.parents])
    roots.extend([Path.cwd()])
    seen = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            pass
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        path = root / "output" / "audit" / "evaluation_predictions_oof.csv"
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
            if {"y_true", "pred_probability"}.issubset(frame.columns) and not frame.empty:
                return frame, path
        except Exception:
            continue
    return None, None


def _normalize_feature_token(name: str) -> str:
    """Normalize feature token for robust fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _is_standardized_feature_name(name: str) -> bool:
    """Heuristic to detect standardized/internal feature naming style."""
    token = str(name).strip()
    if not token:
        return False
    if re.match(r"^HMDB\d+(?:[_\.]\d+)?$", token, flags=re.IGNORECASE):
        return True
    upper = token.upper()
    standardized_prefixes = (
        "RATIO_",
        "SUM_",
        "TAXSUM_",
        "HMDB",
    )
    if any(upper.startswith(prefix) for prefix in standardized_prefixes):
        return True
    if "_" in token and token == upper:
        return True
    return False


def _register_mapping_pair(
    mapping: Dict[str, str],
    src: str,
    dst: str,
) -> None:
    """Register a candidate mapping pair with direction auto-inference."""
    src = str(src).strip()
    dst = str(dst).strip()
    if not src or not dst or src.lower() == "nan" or dst.lower() == "nan":
        return

    src_std = _is_standardized_feature_name(src)
    dst_std = _is_standardized_feature_name(dst)

    # Prefer standardized -> original
    if src_std and not dst_std:
        mapping.setdefault(src, dst)
    elif dst_std and not src_std:
        mapping.setdefault(dst, src)
    else:
        # Ambiguous direction: keep both directions as fallback
        mapping.setdefault(src, dst)
        mapping.setdefault(dst, src)


def _resolve_original_feature_name_map() -> Dict[str, str]:
    """
    构建用于可视化展示的特征名映射：
    当前特征名(标准化/HMDB) -> 原始数据集特征名
    """
    mapping: Dict[str, str] = {}
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / 'data'
    latest_engineered_dir = project_root / 'output' / 'phase1' / 'intermediate' / 'latest' / 'engineered'
    latest_phase1_root = project_root / 'output' / 'phase1' / 'intermediate' / 'latest'

    std_map_csv = data_dir / 'feature_name_mapping.csv'
    if std_map_csv.exists():
        try:
            df_map = pd.read_csv(std_map_csv)
            if {'original_name', 'standardized_name'}.issubset(df_map.columns):
                for _, row in df_map.iterrows():
                    original = str(row['original_name']).strip()
                    standardized = str(row['standardized_name']).strip()
                    _register_mapping_pair(mapping, standardized, original)
        except Exception as exc:
            print(f"  ⚠️ Failed loading {std_map_csv.name}: {exc}")

    hmdb_map_csv = data_dir / 'hmdb_mappings.csv'
    if hmdb_map_csv.exists():
        try:
            df_hmdb = pd.read_csv(hmdb_map_csv)
            if {'Metabolite', 'HMDB_ID'}.issubset(df_hmdb.columns):
                for _, row in df_hmdb.iterrows():
                    metabolite = str(row['Metabolite']).strip()
                    hmdb_id = str(row['HMDB_ID']).strip()
                    _register_mapping_pair(mapping, hmdb_id, metabolite)
        except Exception as exc:
            print(f"  ⚠️ Failed loading {hmdb_map_csv.name}: {exc}")

    name_map_json = data_dir / 'name_mapping.json'
    if name_map_json.exists():
        try:
            with open(name_map_json, 'r', encoding='utf-8') as f:
                jmap = json.load(f)
            if isinstance(jmap, dict):
                for key, val in jmap.items():
                    _register_mapping_pair(mapping, key, val)
        except Exception as exc:
            print(f"  ⚠️ Failed loading {name_map_json.name}: {exc}")

    test_map_json = data_dir / 'test_data2_for_agent_name_mapping.json'
    if test_map_json.exists():
        try:
            with open(test_map_json, 'r', encoding='utf-8') as f:
                jmap = json.load(f)
            if isinstance(jmap, dict):
                for original, mapped in jmap.items():
                    _register_mapping_pair(mapping, mapped, original)
        except Exception as exc:
            print(f"  ⚠️ Failed loading {test_map_json.name}: {exc}")

    # NEW: Load latest canonical engineered mapping produced by Phase 1.
    candidate_json_maps = [
        latest_engineered_dir / 'column_name_mapping.json',
        latest_phase1_root / 'test_lung_cancer_name_mapping.json',
    ]
    # Also include any name-mapping json generated under latest intermediate folder.
    for path_str in glob.glob(str(latest_phase1_root / '**' / '*name_mapping*.json'), recursive=True):
        candidate_json_maps.append(Path(path_str))

    seen_paths = set()
    for map_path in candidate_json_maps:
        map_path = Path(map_path)
        if map_path in seen_paths or not map_path.exists():
            continue
        seen_paths.add(map_path)
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                jmap = json.load(f)
            if isinstance(jmap, dict):
                for key, val in jmap.items():
                    _register_mapping_pair(mapping, key, val)
        except Exception as exc:
            print(f"  ⚠️ Failed loading {map_path.name}: {exc}")

    return mapping


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_shap_direction(
    feature_values: np.ndarray,
    shap_feature_values: np.ndarray,
) -> Tuple[str, Optional[float]]:
    if feature_values.size == 0 or shap_feature_values.size == 0:
        return 'mixed_or_weak', None
    if np.allclose(np.nanstd(feature_values), 0.0) or np.allclose(np.nanstd(shap_feature_values), 0.0):
        return 'mixed_or_weak', None

    correlation = np.corrcoef(feature_values, shap_feature_values)[0, 1]
    corr_float = _safe_float(correlation)
    if corr_float is None or np.isnan(corr_float):
        return 'mixed_or_weak', None
    if corr_float >= 0.10:
        return 'higher_values_increase_risk', corr_float
    if corr_float <= -0.10:
        return 'higher_values_decrease_risk', corr_float
    return 'mixed_or_weak', corr_float


def _build_shap_summary_payload(
    *,
    feature_names: List[str],
    feature_matrix: pd.DataFrame,
    shap_values: np.ndarray,
    patient_row_idx: int,
    expected_value: float,
    predicted_probability: Optional[float],
) -> Dict[str, object]:
    ranked_features: List[Dict[str, object]] = []
    for idx, feature_name in enumerate(feature_names):
        feature_series = feature_matrix.iloc[:, idx].to_numpy(dtype=float, copy=False)
        feature_shap = np.asarray(shap_values[:, idx], dtype=float)
        direction_label, corr_value = _summarize_shap_direction(feature_series, feature_shap)
        ranked_features.append(
            {
                'feature': feature_name,
                'mean_abs_shap': round(float(np.mean(np.abs(feature_shap))), 6),
                'mean_shap': round(float(np.mean(feature_shap)), 6),
                'median_abs_shap': round(float(np.median(np.abs(feature_shap))), 6),
                'direction': direction_label,
                'direction_correlation': round(corr_value, 6) if corr_value is not None else None,
                'positive_share': round(float(np.mean(feature_shap > 0)), 6),
                'negative_share': round(float(np.mean(feature_shap < 0)), 6),
            }
        )

    ranked_features.sort(key=lambda item: float(item.get('mean_abs_shap') or 0.0), reverse=True)
    for rank, item in enumerate(ranked_features, start=1):
        item['rank'] = rank

    positive_risk_features = [
        {
            'feature': item['feature'],
            'rank': item['rank'],
            'mean_abs_shap': item['mean_abs_shap'],
            'direction_correlation': item['direction_correlation'],
        }
        for item in ranked_features
        if item.get('direction') == 'higher_values_increase_risk'
    ]
    negative_risk_features = [
        {
            'feature': item['feature'],
            'rank': item['rank'],
            'mean_abs_shap': item['mean_abs_shap'],
            'direction_correlation': item['direction_correlation'],
        }
        for item in ranked_features
        if item.get('direction') == 'higher_values_decrease_risk'
    ]

    local_feature_values = feature_matrix.iloc[patient_row_idx].to_numpy(dtype=float, copy=False)
    local_shap_values = np.asarray(shap_values[patient_row_idx], dtype=float)
    local_contributors: List[Dict[str, object]] = []
    for idx, feature_name in enumerate(feature_names):
        contribution = float(local_shap_values[idx])
        local_contributors.append(
            {
                'feature': feature_name,
                'feature_value': round(float(local_feature_values[idx]), 6),
                'shap_value': round(contribution, 6),
                'direction': 'increase_prediction' if contribution > 0 else 'decrease_prediction' if contribution < 0 else 'neutral',
                'abs_shap': round(abs(contribution), 6),
            }
        )
    local_contributors.sort(key=lambda item: float(item.get('abs_shap') or 0.0), reverse=True)

    return {
        'schema_version': 'phase3.shap_summary.v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'feature_count': len(feature_names),
        'patient_index_used': patient_row_idx + 1,
        'expected_value': round(float(expected_value), 6),
        'predicted_probability': round(predicted_probability, 6) if predicted_probability is not None else None,
        'ranked_features': ranked_features,
        'top_positive_risk_features': positive_risk_features[:5],
        'top_negative_risk_features': negative_risk_features[:5],
        'local_example': {
            'patient_index': patient_row_idx + 1,
            'top_contributors': local_contributors[:5],
        },
    }


def _resolve_shap_display_feature_names(
    features: List[str],
    phase2_result_path: Optional[str] = None,
) -> List[str]:
    """Prefer Phase 1/2 provenance display names; fallback to legacy original-name mapping."""
    display_map: Dict[str, str] = {}
    if phase2_result_path:
        artifact_dir = Path(phase2_result_path).resolve().parent
        candidate_paths = [
            artifact_dir / "feature_provenance.json",
            artifact_dir.parent / "feature_provenance.json",
        ]
        for candidate_path in candidate_paths:
            if not candidate_path.exists():
                continue
            try:
                payload = json.loads(candidate_path.read_text(encoding="utf-8"))
                for item in payload.get("features", []):
                    feature_name = str(item.get("feature", "")).strip()
                    display_name = str(item.get("display_name", "")).strip()
                    if feature_name and display_name:
                        display_map.setdefault(feature_name, display_name)
            except Exception as exc:
                print(f"  ⚠️ Failed loading SHAP feature provenance from {candidate_path.name}: {exc}")
            if display_map:
                break

    if display_map:
        return [display_map.get(feature, feature) for feature in features]
    return _to_display_feature_names(features)


def _write_shap_global_assets(
    *,
    output_dir: Path,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    feature_names: List[str],
    display_feature_names: List[str],
    shap_values: np.ndarray,
    class_labels: List[str],
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    long_frames = []
    for feature_idx, feature_name in enumerate(feature_names):
        feature_values = X.iloc[:, feature_idx].to_numpy(dtype=float, copy=False)
        shap_column = np.asarray(shap_values[:, feature_idx], dtype=float)
        feature_min = np.nanmin(feature_values)
        feature_max = np.nanmax(feature_values)
        if np.isfinite(feature_min) and np.isfinite(feature_max) and not np.isclose(feature_min, feature_max):
            feature_value_norm = (feature_values - feature_min) / (feature_max - feature_min)
        else:
            feature_value_norm = np.full(feature_values.shape, 0.5, dtype=float)

        long_frames.append(
            pd.DataFrame(
                {
                    "sample_id": np.arange(1, len(X) + 1, dtype=int),
                    "actual_class_id": y_encoded.astype(int),
                    "actual_class_label": [class_labels[int(v)] for v in y_encoded],
                    "feature": feature_name,
                    "feature_display_name": display_feature_names[feature_idx],
                    "feature_value": feature_values,
                    "feature_value_norm": feature_value_norm.astype(float),
                    "shap_value": shap_column,
                    "abs_shap": np.abs(shap_column),
                    "direction": np.where(shap_column >= 0, "push_class_1", "push_class_0"),
                }
            )
        )

    shap_long_df = pd.concat(long_frames, ignore_index=True)
    mean_abs_shap = np.mean(np.abs(np.asarray(shap_values, dtype=float)), axis=0)
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "feature_display_name": display_feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = np.arange(1, len(importance_df) + 1, dtype=int)
    importance_df["proportion"] = importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum()

    metadata = {
        "schema_version": "phase3.shap_global_assets.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": list(importance_df["feature"]),
        "feature_display_order": list(importance_df["feature_display_name"]),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }

    shap_long_path = output_dir / "shap_long.csv"
    feature_importance_path = output_dir / "feature_importance.csv"
    metadata_path = output_dir / "metadata.json"
    shap_long_df.to_csv(shap_long_path, index=False)
    importance_df.to_csv(feature_importance_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return shap_long_path, feature_importance_path, metadata_path


def _render_shap_global_importance_r(
    *,
    shap_long_path: Path,
    feature_importance_path: Path,
    save_path: str,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_shap_global_importance_plot2_style.R"
    if not r_script.exists():
        raise FileNotFoundError(f"SHAP R script not found: {r_script}")

    rscript_path = shutil.which("Rscript")
    if not rscript_path:
        raise RuntimeError("Rscript is not available on PATH")

    output_stem = str(Path(save_path).with_suffix(""))
    theme_json_path = write_theme_json(journal_theme, Path(output_stem).with_name(f"{Path(output_stem).name}_theme.json")) if journal_theme else ""
    command = [
        rscript_path,
        str(r_script),
        str(shap_long_path),
        str(feature_importance_path),
        output_stem,
    ]
    if theme_json_path:
        command.append(theme_json_path)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to render SHAP global importance plot with R.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    pdf_path = f"{output_stem}.pdf"
    png_path = f"{output_stem}.png"
    svg_path = f"{output_stem}.svg"
    return pdf_path, png_path, svg_path


def _to_display_feature_names(features: List[str]) -> List[str]:
    feature_name_map = _resolve_original_feature_name_map()
    normalized_index = {
        _normalize_feature_token(k): v
        for k, v in feature_name_map.items()
        if _normalize_feature_token(k)
    }

    display_names: List[str] = []
    for feat in features:
        if feat in feature_name_map:
            display_names.append(feature_name_map[feat])
            continue
        norm_feat = _normalize_feature_token(feat)
        mapped = normalized_index.get(norm_feat)
        display_names.append(mapped if mapped else feat)
    return display_names


def _format_dependence_panel_feature_name(feature_name: str) -> str:
    display_name = _to_display_feature_names([feature_name])[0]
    if display_name != feature_name:
        return str(display_name)

    if feature_name.startswith("RATIO_"):
        parts = feature_name.split("_")
        if len(parts) == 3:
            left = _to_display_feature_names([parts[1]])[0]
            right = _to_display_feature_names([parts[2]])[0]
            return f"{left} / {right} ratio"

    return str(feature_name).replace("_", " ")


def _wrap_dependence_axis_label(label: str, width: int = 18) -> str:
    wrapped = textwrap.wrap(str(label), width=width, break_long_words=False)
    return "\n".join(wrapped) if wrapped else str(label)


def _compute_dependence_zero_crossings(
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    tolerance: float,
) -> List[float]:
    crossings: List[float] = []
    for idx in range(len(curve_x) - 1):
        x0, x1 = float(curve_x[idx]), float(curve_x[idx + 1])
        y0, y1 = float(curve_y[idx]), float(curve_y[idx + 1])
        if abs(y0) < 1e-8:
            x_cross = x0
        elif abs(y1) < 1e-8:
            x_cross = x1
        elif y0 * y1 < 0:
            x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
        else:
            continue
        if not crossings or abs(crossings[-1] - x_cross) > tolerance:
            crossings.append(x_cross)
    return crossings


def _save_shap_dependence_panels(
    *,
    feature_matrix: pd.DataFrame,
    shap_values: np.ndarray,
    save_stem: Path,
    dpi: int = 300,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    feature_names = list(feature_matrix.columns)
    if not feature_names:
        return {}

    n_panels = len(feature_names)
    n_cols = min(3, max(1, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))

    theme = journal_theme or {}
    curve_color = resolve_theme_color(theme, "baseline", "#5186F0")
    ci_color = resolve_theme_color(theme, "uncertainty_fill", "#D6E4FF")
    positive_color = resolve_theme_color(theme, "positive", "#63BF7B")
    negative_color = resolve_theme_color(theme, "negative", "#EC5555")
    high_color = resolve_theme_color(theme, "winner", "#EC5555")
    low_color = resolve_theme_color(theme, "phase1", "#5186F0")
    threshold_color = resolve_theme_color(theme, "highlight", "#AA84DF")
    reference_color = resolve_theme_color(theme, "reference", "#B3B3B3")
    text_color = str(theme.get("text_color") or "#1F2937")
    axis_text_color = str(theme.get("axis_text_color") or "#374151")
    grid_color = str(theme.get("grid_color") or "#E9EDF2")
    panel_border_color = str(theme.get("panel_border_color") or "#D7DEE8")
    cmap = LinearSegmentedColormap.from_list(
        "journal_shap_dependence",
        [low_color, "#F8FAFC", high_color],
        N=256,
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.3 * n_cols, 5.3 * n_rows),
        dpi=dpi,
        squeeze=False,
    )
    flat_axes = axes.flatten()
    manifest_rows: List[Dict[str, object]] = []

    for idx, feature_name in enumerate(feature_names):
        ax = flat_axes[idx]
        x_vals = pd.to_numeric(feature_matrix.iloc[:, idx], errors="coerce").to_numpy(dtype=float)
        y_vals = np.asarray(shap_values[:, idx], dtype=float)
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if valid_mask.sum() < 8:
            ax.axis("off")
            continue

        x_valid = x_vals[valid_mask]
        y_valid = y_vals[valid_mask]
        sort_idx = np.argsort(x_valid)
        x_sorted = x_valid[sort_idx]
        y_sorted = y_valid[sort_idx]
        lowess = sm.nonparametric.lowess(y_sorted, x_sorted, frac=0.3, return_sorted=True)
        curve_x = lowess[:, 0]
        curve_y = lowess[:, 1]
        band = (
            pd.Series(y_sorted - curve_y)
            .rolling(window=max(5, int(len(y_sorted) * 0.1)), min_periods=1, center=True)
            .std()
            .fillna(0.0)
            .to_numpy()
        )
        upper = curve_y + band
        lower = curve_y - band

        x_pad = max((x_valid.max() - x_valid.min()) * 0.03, 1e-6)
        y_all = np.concatenate([y_valid, upper, lower, np.array([0.0])])
        y_pad = max((y_all.max() - y_all.min()) * 0.06, 1e-6)
        x_min, x_max = x_valid.min() - x_pad, x_valid.max() + x_pad
        y_min, y_max = y_all.min() - y_pad, y_all.max() + y_pad
        zero_crossings = _compute_dependence_zero_crossings(
            curve_x,
            curve_y,
            tolerance=max((x_max - x_min) * 0.01, 1e-6),
        )

        scatter = ax.scatter(
            x_valid,
            y_valid,
            c=y_valid,
            cmap=cmap,
            s=24,
            alpha=0.9,
            edgecolors="none",
            zorder=4,
        )

        bounds = [x_min, *zero_crossings, x_max]
        for left, right in zip(bounds[:-1], bounds[1:]):
            mid_x = (left + right) / 2.0
            mid_y = curve_y[np.abs(curve_x - mid_x).argmin()]
            if mid_y >= 0:
                ax.fill_between([left, right], 0, y_max, color=positive_color, alpha=0.12, zorder=1)
            else:
                ax.fill_between([left, right], y_min, 0, color=negative_color, alpha=0.12, zorder=1)

        ax.fill_between(curve_x, lower, upper, color=ci_color, alpha=0.30, zorder=2)
        ax.plot(curve_x, curve_y, color=curve_color, linewidth=2.2, alpha=0.95, zorder=5)

        y_range = max(y_max - y_min, 1e-6)
        for cross_idx, x_cross in enumerate(zero_crossings):
            ax.axvline(x_cross, color=threshold_color, linestyle="--", linewidth=1.0, alpha=0.85, zorder=3)
            ax.scatter([x_cross], [0], color=threshold_color, s=18, zorder=6)
            y_offset = 0.08 * y_range if cross_idx % 2 == 0 else -0.08 * y_range
            va = "bottom" if cross_idx % 2 == 0 else "top"
            ax.text(
                x_cross,
                y_offset,
                f"{x_cross:.2f}",
                fontsize=9,
                color=threshold_color,
                ha="center",
                va=va,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
                zorder=6,
            )

        ax.axhline(0, color=reference_color, linestyle="--", linewidth=0.9, alpha=0.8, zorder=3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(
            _wrap_dependence_axis_label(_format_dependence_panel_feature_name(feature_name)),
            fontsize=12,
        )
        ax.set_ylabel("SHAP value", fontsize=12, color=text_color)
        ax.set_title("")
        ax.grid(True, linestyle="--", alpha=0.25, color=grid_color)
        ax.tick_params(axis="both", colors=axis_text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_color(panel_border_color)
        ax.legend(
            handles=[
                Line2D([0], [0], color=curve_color, linewidth=2.2, label="Lowess curve"),
                mpatches.Patch(facecolor=positive_color, edgecolor="none", alpha=0.12, label="Positive"),
                mpatches.Patch(facecolor=negative_color, edgecolor="none", alpha=0.12, label="Negative"),
            ],
            loc="best",
            frameon=False,
            fontsize=10,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_ticks([float(np.nanmin(y_valid)), float(np.nanmax(y_valid))])
        cbar.set_ticklabels(["Low", "High"])
        cbar.set_label("SHAP value", rotation=270, labelpad=14)

        manifest_rows.append(
            {
                "feature": str(feature_name),
                "display_name": _format_dependence_panel_feature_name(feature_name),
                "zero_crossings": [float(v) for v in zero_crossings],
            }
        )

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    png_path = save_stem.with_suffix(".png")
    pdf_path = save_stem.with_suffix(".pdf")
    manifest_path = save_stem.with_suffix(".json")
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    manifest_payload = {
        "schema_version": "phase3.shap_dependence_panels.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(manifest_rows),
        "rows": n_rows,
        "cols": n_cols,
        "files": manifest_rows,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "dependence_panels_png_path": str(png_path),
        "dependence_panels_pdf_path": str(pdf_path),
        "dependence_panels_manifest_path": str(manifest_path),
    }


def _load_json_report(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_phase3_feature_display_map(feature_provenance_path: str = '') -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    candidate_paths: List[str] = []
    if feature_provenance_path:
        candidate_paths.append(feature_provenance_path)
    candidate_paths.extend([
        'output/phase1/artifacts/feature_provenance.json',
        'data/feature_provenance.json',
    ])

    seen = set()
    for path in candidate_paths:
        normalized_path = str(path).strip()
        if not normalized_path or normalized_path in seen or not os.path.exists(normalized_path):
            continue
        seen.add(normalized_path)
        payload = _load_json_report(normalized_path)
        records = payload.get('features', []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            feature = str(record.get('feature', '') or '').strip()
            standardized = str(record.get('standardized_name', '') or '').strip()
            display_name = str(record.get('display_name', '') or '').strip()
            mapped_metabolites = [
                str(item).strip()
                for item in (record.get('mapped_metabolites') or [])
                if str(item).strip()
            ]
            origin_type = str(record.get('origin_type', '') or '').strip()

            if origin_type == 'engineered_ratio' and len(mapped_metabolites) >= 2:
                display_name = f"{mapped_metabolites[0]} / {mapped_metabolites[1]}"
            elif not display_name and mapped_metabolites:
                display_name = mapped_metabolites[0]

            if not display_name:
                continue
            if feature:
                mapping.setdefault(feature, display_name)
            if standardized:
                mapping.setdefault(standardized, display_name)
    return mapping


def _to_phase3_display_feature_names(
    features: List[str],
    feature_provenance_path: str = '',
) -> List[str]:
    provenance_map = _build_phase3_feature_display_map(feature_provenance_path)
    fallback_map = _resolve_original_feature_name_map()
    fallback_normalized = {
        _normalize_feature_token(k): v
        for k, v in fallback_map.items()
        if _normalize_feature_token(k)
    }

    display_names: List[str] = []
    for feature in features:
        raw_feature = str(feature).strip()
        if raw_feature in provenance_map:
            display_names.append(provenance_map[raw_feature])
            continue
        normalized = _normalize_feature_token(raw_feature)
        if normalized:
            provenance_hit = next(
                (value for key, value in provenance_map.items() if _normalize_feature_token(key) == normalized),
                None,
            )
            if provenance_hit:
                display_names.append(provenance_hit)
                continue
        if raw_feature in fallback_map:
            display_names.append(fallback_map[raw_feature])
            continue
        fallback_hit = fallback_normalized.get(normalized)
        display_names.append(fallback_hit if fallback_hit else raw_feature.replace('_', ' '))
    return display_names


_STABILITY_METHOD_LABELS = {
    'run_elasticnet_selector': 'Elastic Net',
    'run_lasso_selector': 'Lasso',
    'run_random_forest_selector': 'Random Forest',
    'run_lightgbm_selector': 'LightGBM',
    'run_mrmr_selector': 'mRMR',
    'run_t_test_selector': 'Welch t-test',
    'run_fdr_effect_size_selector': 'FDR/effect-size',
}

_STABILITY_STATUS_ORDER = ['rejected', 'completed', 'stable_core', 'mandatory']
_STABILITY_STATUS_LABELS = {
    'rejected': 'Rejected',
    'completed': 'Greedy Completed',
    'stable_core': 'Stable Core',
    'mandatory': 'Mandatory',
}
_STABILITY_STATUS_COLORS = {
    'rejected': '#D9DCE3',
    'completed': '#78C6A3',
    'stable_core': '#2B6CB0',
    'mandatory': '#C68A00',
}


def _build_stability_plot_dataframe(
    stability_summary: Dict[str, Any],
    stability_scores_payload: Dict[str, Any],
    feature_provenance_path: str = '',
) -> pd.DataFrame:
    stable_core = set(stability_summary.get('stable_core_features', []) or [])
    final_panel = set(stability_summary.get('final_panel_features', []) or [])
    mandatory = set(stability_summary.get('mandatory_features', []) or [])
    display_map = _build_phase3_feature_display_map(feature_provenance_path)
    stability_rows = stability_scores_payload.get('stability_scores', []) or []

    rows: List[Dict[str, Any]] = []
    for row in stability_rows:
        if not isinstance(row, dict):
            continue
        feature = str(row.get('feature', '') or '').strip()
        if not feature:
            continue

        if feature in mandatory or bool(row.get('is_prior_protected')):
            status = 'mandatory'
        elif feature in stable_core:
            status = 'stable_core'
        elif feature in final_panel:
            status = 'completed'
        else:
            status = 'rejected'

        display_name = display_map.get(feature)
        if not display_name:
            display_name = _to_phase3_display_feature_names([feature], feature_provenance_path=feature_provenance_path)[0]

        clean_row: Dict[str, Any] = {
            'feature': feature,
            'display_name': display_name,
            'selection_frequency': float(row.get('selection_frequency', 0.0) or 0.0),
            'method_consensus': int(row.get('method_consensus', 0) or 0),
            'panel_score': float(row.get('panel_score', 0.0) or 0.0),
            'status': status,
            'is_final_panel': feature in final_panel,
            'is_stable_core': feature in stable_core,
            'is_mandatory': feature in mandatory or bool(row.get('is_prior_protected')),
        }
        for method, freq in (row.get('per_method_frequency') or {}).items():
            clean_row[str(method)] = float(freq or 0.0)
        rows.append(clean_row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No stability rows available for plotting.')

    df['status'] = pd.Categorical(df['status'], categories=_STABILITY_STATUS_ORDER, ordered=True)
    df = df.sort_values(
        ['panel_score', 'selection_frequency', 'method_consensus', 'feature'],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return df


def _select_focused_heatmap_rows(df: pd.DataFrame, rejected_n: int = 6) -> pd.DataFrame:
    final_panel = df[df['is_final_panel']].copy()
    rejected = df[~df['is_final_panel']].copy()
    if rejected.empty:
        return final_panel.reset_index(drop=True)

    rejected = rejected.assign(
        stable_core_gap=np.maximum(0.60 - rejected['selection_frequency'], 0.0),
        consensus_gap=np.maximum(3 - rejected['method_consensus'], 0.0),
    )
    rejected = rejected.sort_values(
        ['stable_core_gap', 'consensus_gap', 'panel_score', 'selection_frequency', 'method_consensus'],
        ascending=[True, True, False, False, False],
    )
    near_miss = rejected.head(rejected_n).copy()
    subset = pd.concat([final_panel, near_miss], axis=0, ignore_index=True)
    return subset


def _format_feature_axis_label(label: str, max_len: int = 28) -> str:
    text = str(label).strip()
    return text if len(text) <= max_len else text[: max_len - 1] + '…'


def _sanitize_filename_token(text: str) -> str:
    token = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')
    return token[:80] if token else 'feature'


def _run_r_rcs_plotter(
    data_path: str,
    target_column: str,
    features: List[str],
    display_feature_names: List[str],
    save_path: str,
    n_grid: int,
    dpi: int,
    figsize: Tuple[int, int],
    final_layout_kwargs: Dict,
) -> Tuple[str, Dict]:
    project_root = Path(__file__).resolve().parents[3]
    r_script_path = Path(__file__).resolve().parent / "r" / "plot_rcs_panels.R"
    if not r_script_path.exists():
        raise FileNotFoundError(f"R RCS plotting script not found: {r_script_path}")

    rscript_path = shutil.which("Rscript")
    if not rscript_path:
        raise RuntimeError("Rscript is not available on PATH")

    summary_path = str(Path(save_path).with_suffix(".json"))
    payload = {
        "data_path": str(Path(data_path).resolve()),
        "target_column": target_column,
        "features": features,
        "display_features": display_feature_names,
        "save_path": str(Path(save_path).resolve()),
        "summary_path": str(Path(summary_path).resolve()),
        "n_grid": int(n_grid),
        "dpi": int(dpi),
        "figsize": [float(figsize[0]), float(figsize[1])],
        "line_color": final_layout_kwargs.get("line_color", "#8B0000"),
        "scatter_alpha": float(final_layout_kwargs.get("scatter_alpha", 0.18)),
        "fontsize": int(final_layout_kwargs.get("fontsize", 10)),
        "title_fontsize": int(final_layout_kwargs.get("title_fontsize", 14)),
        "label_fontsize": int(final_layout_kwargs.get("label_fontsize", 12)),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rcs_r_backend_") as tmpdir:
        payload_path = Path(tmpdir) / "rcs_payload.json"
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        cmd = [rscript_path, str(r_script_path), str(payload_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown R plotting error"
        raise RuntimeError(f"R RCS plotting failed: {stderr}")
    if result.stderr.strip():
        print(result.stderr.strip())
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"R RCS plotting completed but output PDF is missing: {save_path}")

    result_kwargs = dict(final_layout_kwargs)
    result_kwargs["rcs_summary_path"] = summary_path
    result_kwargs["rcs_backend"] = "r"
    result_kwargs["rcs_engine"] = "plotRCS"
    return save_path, result_kwargs


def plot_pareto_trajectory(
    history_log_path: str,
    best_panel_path: str,
    save_path: str = "output/figures/fig2b_pareto_trajectory.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制帕累托进化轨迹散点图（Pareto Trajectory Scatter Plot）
    
    这张图向审稿人直观展示 PToT 引擎在"成本"与"效能"之间的寻优过程，
    并凸显最终大模型的临床裁决。
    
    核心视觉元素:
    1. 全量探索节点: 所有历史候选节点（带透明度）
    2. 帕累托前沿线: 非支配排序 Rank 1 的点集（红色虚线）
    3. LLM 裁决高亮: 最终获胜的散点（金色五角星 + 红色边框）
    4. LLM Reasoning 注释: 截断后的前两句话
    
    视觉映射:
    - X轴: f_cost (临床成本，越小越好)
    - Y轴: f_perf (预测性能/AUC，越大越好)
    - 颜色: f_bio (生物学机制分，连续色谱 viridis)
    - 大小: f_corr (冗余度，转换为气泡大小)
    
    ACVAL 支持:
    - layout_kwargs: 排版参数字典，支持动态调整图例位置、字体大小等
    - 返回值: (save_path, final_layout_kwargs) 用于 Critic 审查和重试
    
    Args:
        history_log_path: Phase 2 生成的 JSON 文件路径（包含搜索历史）
        best_panel_path: LLM 最终裁决的 JSON 文件路径（同一个文件）
        save_path: 输出图片路径（默认 output/figures/fig1_pareto_trajectory.pdf）
        dpi: 图片分辨率（默认 300，符合顶级医学期刊标准）
        figsize: 图片尺寸（默认 12x8 英寸）
        layout_kwargs: 排版参数字典（可选），支持的键:
            - legend_loc: 主图例位置 (默认 'lower right')
            - legend2_loc: 气泡大小图例位置 (默认 'upper right')
            - fontsize: 全局字体大小 (默认 11)
            - title_fontsize: 标题字体大小 (默认 16)
            - label_fontsize: 坐标轴标签字体大小 (默认 14)
            - legend_fontsize: 图例字体大小 (默认 11)
            - bbox_to_anchor: 图例锚点位置 (默认 None)
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> path, kwargs = plot_pareto_trajectory(
        ...     history_log_path="test_results/phase2_test.json",
        ...     best_panel_path="test_results/phase2_test.json",
        ...     save_path="output/figures/fig1_pareto_trajectory.pdf",
        ...     layout_kwargs={'legend_loc': 'upper left', 'fontsize': 12}
        ... )
        >>> print(f"Saved to: {path}")
        >>> print(f"Final kwargs: {kwargs}")
    """
    # ========================================================================
    # 0. 初始化排版参数 (Layout Parameters Initialization)
    # ========================================================================
    # 默认排版参数
    default_layout_kwargs = {
        'legend_loc': 'lower right',
        'legend2_loc': 'upper right',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11,
        'legend2_fontsize': 8,
        'bbox_to_anchor': None,
        'bbox_to_anchor2': None
    }
    
    # 合并用户提供的参数
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    theme = journal_theme or {}
    palette = theme.get('palette', []) if isinstance(theme, dict) else []
    semantic_colors = theme.get('semantic_colors', {}) if isinstance(theme, dict) else {}
    bio_colormap = final_layout_kwargs.get('bio_colormap', 'cividis')
    pareto_line_color = resolve_theme_color(theme, 'winner', palette[0] if palette else '#8A4F4A')
    winner_edge_color = resolve_theme_color(theme, 'clinical', pareto_line_color)
    winner_marker_color = resolve_theme_color(theme, 'highlight', palette[6] if len(palette) > 6 else '#C6B27A')
    neutral_edge_color = resolve_theme_color(theme, 'reference', palette[-1] if palette else '#A7A19A')
    depth_label_color = resolve_theme_color(theme, 'cost', palette[5] if len(palette) > 5 else '#9B7A4F')
    winner_label_color = resolve_theme_color(theme, 'positive', palette[1] if len(palette) > 1 else '#5F8F8A')
    grid_color = theme.get('grid_color', '#E9EDF2') if isinstance(theme, dict) else '#E9EDF2'
    axis_text_color = theme.get('axis_text_color', '#374151') if isinstance(theme, dict) else '#374151'
    text_color = theme.get('text_color', '#1F2937') if isinstance(theme, dict) else '#1F2937'
    plt.rcParams['font.family'] = theme.get('font_family', 'Arial') if isinstance(theme, dict) else 'Arial'
    
    print(f"\n[ACVAL Actor] Using layout parameters: {final_layout_kwargs}")
    # ========================================================================
    # 1. 数据解析逻辑 (Data Parsing)
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: Pareto Trajectory Visualization")
    print("="*80)
    print(f"Loading data from: {history_log_path}")
    
    # 读取历史轨迹 JSON 数据
    with open(history_log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    winner_source = data
    if best_panel_path and os.path.exists(best_panel_path):
        try:
            with open(best_panel_path, 'r', encoding='utf-8') as f:
                winner_source = json.load(f)
        except Exception:
            winner_source = data
    
    # 提取所有历史候选节点
    all_candidates = []
    pareto_front_candidates = []
    
    # 兼容两种 search_details 的位置：final_result 内部或根目录
    search_details = None
    if 'final_result' in data and 'search_details' in data['final_result']:
        search_details = data['final_result']['search_details']
    elif 'search_details' in data:
        search_details = data['search_details']
        
    if search_details:
        for layer in search_details.get('layers', []):
            # 兼容两种提取评估候选的方式：evaluation.candidates 或 candidate_trace
            candidates = []
            if 'evaluation' in layer and 'candidates' in layer['evaluation']:
                candidates = layer['evaluation']['candidates']
            elif 'candidate_trace' in layer:
                candidates = layer['candidate_trace']
                
            for cand in candidates:
                cand_data = {
                    'features': cand['features'],
                    'perf': cand['perf'],
                    'bio': cand['bio'],
                    'corr': cand['corr'],
                    'cost': cand['cost'],
                    'depth': layer['depth']
                }
                all_candidates.append(cand_data)
                
                # 提取帕累托前沿候选 (in_pareto_front 标记)
                if cand.get('in_pareto_front', False):
                    pareto_front_candidates.append(cand_data)
            
            # 如果 candidate_trace 里没标 in_pareto_front，尝试从 pareto_front.candidates 提取
            if not pareto_front_candidates and 'pareto_front' in layer and 'candidates' in layer['pareto_front']:
                for cand in layer['pareto_front']['candidates']:
                    pareto_front_candidates.append({
                        'features': cand['features'],
                        'perf': cand['perf'],
                        'bio': cand['bio'],
                        'corr': cand['corr'],
                        'cost': cand['cost'],
                        'depth': layer['depth']
                    })
    
    # 提取最终获胜者
    final_result = winner_source.get('final_result', {})
    if not final_result:
        final_result = {
            'features': winner_source.get('selected_features', []),
            'perf': winner_source.get('scores', {}).get('f_perf', 0.0),
            'bio': winner_source.get('scores', {}).get('f_bio', 0.0),
            'corr': winner_source.get('scores', {}).get('f_corr', 0.0),
            'cost': winner_source.get('scores', {}).get('f_cost', 0.0),
            'cv_predictions': winner_source.get('cv_predictions'),
        }
    winner = {
        'features': final_result.get('features', []),
        'perf': final_result.get('perf', 0.0),
        'bio': final_result.get('bio', 0.0),
        'corr': final_result.get('corr', 0.0),
        'cost': final_result.get('cost', 0.0)
    }
    
    # 🚨 紧急核查：打印 Winner 的真实四维数据
    print("\n" + "="*80)
    print("🚨 WINNER DATA VERIFICATION (紧急核查)")
    print("="*80)
    print(f"Winner f_perf (AUC):        {winner['perf']:.6f}")
    print(f"Winner f_cost (Cost):       {winner['cost']:.6f}")
    print(f"Winner f_bio (Biology):     {winner['bio']:.6f}  ⚠️ 如果接近 0，说明生物学分数极低！")
    print(f"Winner f_corr (Redundancy): {winner['corr']:.6f}")
    print(f"Winner features count:      {len(winner['features'])}")
    print("="*80 + "\n")
    
    # 提取 LLM reasoning（从最后一层的 llm_adjudication）
    llm_reasoning = "LLM Chosen Panel"
    if 'search_details' in final_result:
        layers = final_result['search_details'].get('layers', [])
        if layers:
            last_layer = layers[-1]
            # 注意：reasoning 可能不在 JSON 中，因为它在 LLM 调用时生成但未保存
            # 我们使用默认文本
            llm_reasoning = "LLM Chosen Panel\n(Best balance of performance,\nbiology, and cost)"
    
    print(f"Total candidates evaluated: {len(all_candidates)}")
    print(f"Pareto front candidates: {len(pareto_front_candidates)}")
    print(f"Final winner features: {len(winner['features'])}")
    
    # 🚨 核查：打印 f_bio 的数据范围
    if all_candidates:
        all_bio_values = [c['bio'] for c in all_candidates]
        print(f"\n📊 f_bio Data Range Analysis:")
        print(f"   Min f_bio: {min(all_bio_values):.6f}")
        print(f"   Max f_bio: {max(all_bio_values):.6f}")
        print(f"   Mean f_bio: {sum(all_bio_values)/len(all_bio_values):.6f}")
        print(f"   Winner f_bio: {winner['bio']:.6f}")
        print(f"   Winner percentile: {sum(1 for v in all_bio_values if v <= winner['bio']) / len(all_bio_values) * 100:.1f}%")
        print(f"   ⚠️  If Winner is in lower percentile, it will appear BLUE in coolwarm!")
    
    # ========================================================================
    # 2. 数据准备
    # ========================================================================
    # 转换为 DataFrame
    df_all = pd.DataFrame(all_candidates)
    df_pareto = pd.DataFrame(pareto_front_candidates)
    
    # 去重帕累托前沿（按 cost 和 perf 去重）
    df_pareto = df_pareto.drop_duplicates(subset=['cost', 'perf'])
    
    # 🚨 修复 1: 将 Winner 加入帕累托前沿（确保轨迹连贯）
    # Winner 必须是轨迹的终点，不能孤立
    winner_in_pareto = False
    for idx, row in df_pareto.iterrows():
        if (abs(row['cost'] - winner['cost']) < 1e-6 and 
            abs(row['perf'] - winner['perf']) < 1e-6):
            winner_in_pareto = True
            break
    
    if not winner_in_pareto:
        # Winner 不在帕累托前沿中，需要添加
        print("⚠️  Warning: Winner not in Pareto front, adding it now...")
        winner_row = pd.DataFrame([{
            'features': winner['features'],
            'perf': winner['perf'],
            'bio': winner['bio'],
            'corr': winner['corr'],
            'cost': winner['cost'],
            'depth': final_result.get('search_details', {}).get('layers', [{}])[-1].get('depth', 0)
        }])
        df_pareto = pd.concat([df_pareto, winner_row], ignore_index=True)
        print(f"✅ Winner added to Pareto front with depth={winner_row['depth'].values[0]}")
    
    # 按 cost 排序（用于绘制帕累托前沿线）
    df_pareto = df_pareto.sort_values('cost')
    
    # ========================================================================
    # 3. 视觉映射规则 (Visual Aesthetics)
    # ========================================================================
    # 设置学术风格 - 顶刊标准：使用 serif 字体（类 Times New Roman）
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = theme.get('font_family', 'Arial') if isinstance(theme, dict) else 'Arial'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']  # 使用参数化字体大小
    
    # 创建画布
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # ========================================================================
    # 气泡大小映射（修复 Bug：使用 Min-Max 归一化，缩小差异）
    # ========================================================================
    # 问题 1：f_corr 值太小（0.0-0.2），导致视觉差异不明显
    # 问题 2：差异太大（100-800），视觉上过于夸张
    # 解决：使用 Min-Max Scaling 映射到 [200, 500] 范围（更温和的差异）
    
    all_corr_values = df_all['corr'].values
    min_corr = all_corr_values.min()
    max_corr = all_corr_values.max()
    
    # 避免除以 0
    if max_corr > min_corr:
        # 归一化到 [0, 1]，然后映射到 [200, 500]（2.5倍差异，更温和）
        df_all['bubble_size'] = 200 + 300 * ((df_all['corr'] - min_corr) / (max_corr - min_corr))
        df_pareto['bubble_size'] = 200 + 300 * ((df_pareto['corr'] - min_corr) / (max_corr - min_corr))
    else:
        # 所有值相同，使用默认大小
        df_all['bubble_size'] = 350
        df_pareto['bubble_size'] = 350
    
    print(f"Bubble size range: {df_all['bubble_size'].min():.1f} - {df_all['bubble_size'].max():.1f}")
    print(f"Corr range: {min_corr:.4f} - {max_corr:.4f}")
    
    # ========================================================================
    # 4. 核心绘制元素 (Key Elements)
    # ========================================================================
    
    # 4.1 全量探索节点（带透明度）
    scatter_all = ax.scatter(
        df_all['cost'],
        df_all['perf'],
        s=df_all['bubble_size'],
        c=df_all['bio'],
        cmap=bio_colormap,
        alpha=0.4,
        edgecolors=neutral_edge_color,
        linewidths=0.5,
        label='All Explored Candidates',
        zorder=1
    )
    
    # 4.2 帕累托前沿线（红色虚线）
    if len(df_pareto) > 1:
        ax.plot(
            df_pareto['cost'],
            df_pareto['perf'],
            color=pareto_line_color,
            linestyle='--',
            linewidth=2.5,
            alpha=0.8,
            label='Pareto Front',
            zorder=2
        )
    
    # 4.3 帕累托前沿节点（高亮显示）
    ax.scatter(
        df_pareto['cost'],
        df_pareto['perf'],
        s=df_pareto['bubble_size'] * 1.2,
        c=df_pareto['bio'],
        cmap=bio_colormap,
        alpha=0.9,
        edgecolors=pareto_line_color,
        linewidths=2,
        zorder=3
    )
    
    # 4.4 LLM 裁决高亮（核心重构：Winner 作为数据气泡 + 黄色小星星标记）
    # ========================================================================
    # 绝对原则：Winner 必须是一个数据气泡，严禁被单纯的星星代替
    # ========================================================================
    
    # 4.4.1 计算 Winner 的气泡大小（基于其 f_corr 值）
    if max_corr > min_corr:
        winner_bubble_size = 200 + 300 * ((winner['corr'] - min_corr) / (max_corr - min_corr))
    else:
        winner_bubble_size = 350
    
    # 4.4.2 绘制 Winner 数据气泡（颜色和大小严格依据其 f_bio 和 f_corr）
    # 🚨 核查 B: 确保 c 参数正确传入 f_bio，没有取反
    print(f"🔍 Color mapping check: Winner f_bio = {winner['bio']:.6f}")
    print(f"   Colormap: coolwarm (low=blue, high=red)")
    print(f"   Expected color: {'BLUE (LOW BIO!)' if winner['bio'] < 0.3 else 'RED (HIGH BIO)'}")
    
    ax.scatter(
        winner['cost'],
        winner['perf'],
        s=winner_bubble_size * 1.3,  # 稍微放大以突出显示
        c=[winner['bio']],
        cmap=bio_colormap,
        alpha=1.0,  # 完全不透明，突出显示
        edgecolors=winner_edge_color,
        linewidths=3,
        zorder=9  # 确保在普通节点之上
    )
    
    # 4.4.3 在 Winner 气泡右上方叠加黄色小五角星（身份标签）
    # 计算星星的偏移位置（右上方，确保不遮挡气泡）
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    star_x_offset = x_range * 0.015  # 向右偏移 1.5%（增加偏移避免遮挡）
    star_y_offset = y_range * 0.020  # 向上偏移 2.0%（增加偏移避免遮挡）
    
    ax.scatter(
        winner['cost'] + star_x_offset,
        winner['perf'] + star_y_offset,
        s=250,  # 缩小星星大小（从 400 减小到 250）
        marker='*',
        c=winner_marker_color,
        edgecolors=depth_label_color,
        linewidths=1.5,
        zorder=10  # 确保星星在最顶层
    )
    
    # 4.5 搜索深度标注（标注所有帕累托前沿节点，包括 Winner）
    # 为每个帕累托前沿点添加深度标签
    for idx, row in df_pareto.iterrows():
        # 计算偏移量（避免文字重叠）
        x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.008
        
        # 检查是否是 Winner 节点
        is_winner = (abs(row['cost'] - winner['cost']) < 1e-6 and 
                    abs(row['perf'] - winner['perf']) < 1e-6)
        
        # Winner 使用特殊颜色和样式
        text_color_local = depth_label_color if not is_winner else winner_label_color
        text_weight = 'bold' if not is_winner else 'heavy'
        
        ax.text(
            row['cost'] + x_offset,
            row['perf'] - y_offset,
            f"D{row['depth']}",
            fontsize=9,
            color=text_color_local,
            fontstyle='italic',
            fontweight=text_weight,
            zorder=4
        )
    
    # ========================================================================
    # 5. 图表装饰
    # ========================================================================
    
    # 5.1 坐标轴标签
    ax.set_xlabel('Clinical Cost (f_cost)', fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold', color=axis_text_color)
    ax.set_ylabel('Predictive Performance (f_perf / AUC)', fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold', color=axis_text_color)
    
    # 5.2 标题
    ax.set_title(
        'Pareto Trajectory of LLM-PToT Search Engine\n'
        'Multi-Objective Optimization: Performance vs. Cost',
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        pad=20
    )
    
    # 5.3 颜色条（Colorbar）
    cbar = plt.colorbar(scatter_all, ax=ax, pad=0.02)
    cbar.set_label('Biological Mechanism Score (f_bio)', fontsize=12, fontweight='bold', color=axis_text_color)
    
    # 5.4 图例（主图例 - 只包含主要元素，不包含 redundancy）
    # 注意：这里先创建主图例，稍后会添加 Bubble Size 图例
    legend1 = ax.legend(
        loc=final_layout_kwargs['legend_loc'],  # 使用参数化位置
        fontsize=final_layout_kwargs['legend_fontsize'],  # 使用参数化字体大小
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        borderpad=1.0,  # 防止出框
        bbox_to_anchor=final_layout_kwargs['bbox_to_anchor']  # 使用参数化锚点
    )
    
    # 5.5 去除顶部和右侧边框（符合顶级医学期刊标准）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 5.6 网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ========================================================================
    # 精修 2: 提高 Y 轴天花板（解决图例遮挡数据）
    # ========================================================================
    # 在绘制完所有元素后，动态拔高 Y 轴上限，为右上角图例腾出空间
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    # 将 Y 轴上限再拔高 12%，为右上角的图例和星星腾出空间
    ax.set_ylim(ymin, ymax + y_range * 0.12)
    
    print(f"Y-axis adjusted: {ymin:.4f} - {ymax + y_range * 0.12:.4f} (headroom: +{y_range * 0.12:.4f})")
    
    # 5.7 添加气泡大小图例（人类可读性优化）
    # ========================================================================
    # 顶刊标准：Clean Legend Formatting
    # ========================================================================
    # 问题：自动等分会产生 0.076, 0.153 这样的"机械感"数值
    # 解决：手动构建干净的、有意义的刻度（0.00, 0.05, 0.10, 0.15）
    
    # 智能刻度吸附算法
    def snap_to_clean_ticks(min_val, max_val, num_ticks=3):
        """
        将数据范围吸附到干净的刻度值
        
        Args:
            min_val: 数据最小值
            max_val: 数据最大值
            num_ticks: 期望的刻度数量
        
        Returns:
            List[float]: 干净的刻度值列表
        """
        # 计算数据跨度
        span = max_val - min_val
        
        # 定义干净的刻度候选（0.01 的倍数）
        clean_increments = [0.01, 0.02, 0.025, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.50]
        
        # 选择最合适的增量
        target_increment = span / (num_ticks - 1)
        best_increment = min(clean_increments, key=lambda x: abs(x - target_increment))
        
        # 特殊处理：如果数据范围接近 0.15，优先使用 [0.00, 0.08, 0.15]
        if 0.14 <= max_val <= 0.16 and min_val < 0.01:
            return [0.00, 0.08, 0.15]
        
        # 生成刻度
        # 起点：向下取整到最近的增量倍数
        start = (min_val // best_increment) * best_increment
        if start < min_val:
            start += best_increment
        
        # 生成刻度序列
        ticks = []
        current = start
        while current <= max_val and len(ticks) < num_ticks:
            ticks.append(current)
            current += best_increment
        
        # 确保至少有 num_ticks 个刻度
        if len(ticks) < num_ticks:
            # 如果不够，添加最大值
            if max_val not in ticks:
                ticks.append(max_val)
        
        # 如果还是不够，在中间插值
        while len(ticks) < num_ticks:
            mid = (ticks[0] + ticks[-1]) / 2
            ticks.insert(len(ticks) // 2, mid)
        
        return ticks[:num_ticks]
    
    # 生成干净的刻度值
    clean_corr_ticks = snap_to_clean_ticks(min_corr, max_corr, num_ticks=3)
    
    # 使用相同的归一化公式计算示例大小（新范围：200-500）
    if max_corr > min_corr:
        size_examples = [200 + 300 * ((c - min_corr) / (max_corr - min_corr)) for c in clean_corr_ticks]
    else:
        size_examples = [350, 350, 350]
    
    # 在右上角添加气泡大小说明（使用干净的刻度）
    legend_elements = [
        plt.scatter([], [], s=size_examples[0], c=neutral_edge_color, alpha=0.6, 
                   label=f'Low Redundancy (corr={clean_corr_ticks[0]:.2f})'),
        plt.scatter([], [], s=size_examples[1], c=neutral_edge_color, alpha=0.6, 
                   label=f'Medium Redundancy (corr={clean_corr_ticks[1]:.2f})'),
        plt.scatter([], [], s=size_examples[2], c=neutral_edge_color, alpha=0.6, 
                   label=f'High Redundancy (corr={clean_corr_ticks[2]:.2f})')
    ]
    
    print(f"Clean legend ticks: {[f'{t:.2f}' for t in clean_corr_ticks]}")
    
    # ========================================================================
    # 精修 1: 增加图例内部间距（解决溢出与重叠）
    # 精修 2: 调整位置避免遮挡底部内容
    # 精修 3: 删除底部图例中重复的 redundancy 气泡（已在上方创建主图例）
    # 精修 4: 缩小 Bubble Size panel，确保内容不超出边界
    # ========================================================================
    # 添加第二个图例（Bubble Size）- 使用 add_artist 保留第一个图例
    legend2 = ax.legend(
        handles=legend_elements,
        loc=final_layout_kwargs['legend2_loc'],  # 使用参数化位置
        fontsize=final_layout_kwargs['legend2_fontsize'],  # 使用参数化字体大小
        title='Bubble Size (Redundancy)',
        title_fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        # 核心修复参数：
        labelspacing=1.2,
        borderpad=0.8,
        handletextpad=0.8,
        bbox_to_anchor=final_layout_kwargs['bbox_to_anchor2']  # 使用参数化锚点
    )
    
    # 将主图例添加回来（因为创建 legend2 会覆盖 legend1）
    ax.add_artist(legend1)
    
    # ========================================================================
    # 6. 保存图片
    # ========================================================================
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Pareto trajectory plot saved to: {save_path}")
    print(f"[ACVAL Actor] Final layout kwargs: {final_layout_kwargs}")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


def _alpha_color(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgba = list(to_rgba(color))
    rgba[3] = alpha
    return tuple(rgba)


def plot_radar_comparison(
    profiles_dict: dict,
    save_path: str = "output/figures/fig3_radar_4d.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "4D Panel Performance Profile",
    journal_theme: Optional[Dict[str, Any]] = None
) -> str:
    """
    绘制多模型 4D 雷达对比图 (Multi-Model 4D Radar Comparison)
    
    这张图向审稿人直观展示基线模型与 PToT 冠军面板在四个维度的综合表现对比。
    雷达图的面积越大、越均衡，代表面板的综合临床转化价值越高。
    
    核心视觉元素:
    1. 四个维度轴: AUC (预测性能), f_bio (机制价值), 1-f_cost (简约性), 1-f_corr (独立性)
    2. 多模型对比: Phase 1 Baseline, Phase 0 Baseline, PToT Winner
    3. 配色方案: 基线模型使用浅色 + 低透明度，Winner 使用深红/金色 + 高透明度
    
    Args:
        profiles_dict: 模型得分字典，支持两种格式:
            格式1: { 'Model Name': [score1, score2, score3, score4] }
            格式2: { 'Model Name': {'AUC': x, 'f_bio': y, '1-f_cost': z, '1-f_corr': w} }
        save_path: 输出图片路径（默认 output/figures/fig3_radar_4d.pdf）
        dpi: 图片分辨率（默认 300）
        figsize: 图片尺寸（默认 8x8 英寸，正方形）
        title: 雷达图标题（默认使用更简洁的 4D panel profile 标题）
    
    Returns:
        str: 生成的图片路径
    
    Examples:
        >>> profiles = {
        ...     'Phase 1 Baseline': {'AUC': 0.85, 'f_bio': 0.60, '1-f_cost': 0.40, '1-f_corr': 0.70},
        ...     'Phase 0 Baseline': {'AUC': 0.78, 'f_bio': 0.75, '1-f_cost': 0.90, '1-f_corr': 0.65},
        ...     'PToT Winner': {'AUC': 0.88, 'f_bio': 0.82, '1-f_cost': 0.85, '1-f_corr': 0.88}
        ... }
        >>> plot_radar_comparison(profiles, save_path="output/figures/fig3_radar_4d.pdf")
        'output/figures/fig3_radar_4d.pdf'
    """
    # ========================================================================
    # 1. 数据解析与标准化
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: 4D Radar Comparison Visualization")
    print("="*80)
    
    # 定义四个维度的标签（带换行符，避免标签重叠）
    categories = [
        "Predictive Performance\n(AUC)",
        "Mechanistic Value\n(f_bio)",
        "Parsimony\n(1 - f_cost)",
        "Independence\n(1 - f_corr)"
    ]
    
    # 标准化数据格式（统一转换为列表格式）
    standardized_data = {}
    
    for model_name, scores in profiles_dict.items():
        if isinstance(scores, dict):
            # 格式2: 字典格式 -> 按固定顺序提取
            # 支持多种键名变体
            auc = scores.get('AUC') or scores.get('auc') or scores.get('perf') or scores.get('f_perf', 0.0)
            bio = scores.get('f_bio') or scores.get('bio') or scores.get('Mechanistic', 0.0)
            cost = scores.get('1-f_cost') or scores.get('parsimony') or scores.get('Parsimony', 0.0)
            corr = scores.get('1-f_corr') or scores.get('independence') or scores.get('Independence', 0.0)
            
            standardized_data[model_name] = [auc, bio, cost, corr]
        elif isinstance(scores, (list, tuple)) and len(scores) == 4:
            # 格式1: 列表格式 -> 直接使用
            standardized_data[model_name] = list(scores)
        else:
            raise ValueError(f"Invalid data format for model '{model_name}': {scores}")
    
    print(f"Models to compare: {list(standardized_data.keys())}")
    
    # ========================================================================
    # 2. 雷达图坐标系设置
    # ========================================================================
    # 计算角度（4个维度 + 闭合点）
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合圆圈
    
    # 设置学术风格
    theme = journal_theme or {}
    text_color = resolve_theme_color(theme, 'text_color', '#1F2937')
    axis_text_color = resolve_theme_color(theme, 'axis_text_color', '#374151')
    grid_color = resolve_theme_color(theme, 'grid_color', '#E9EDF2')
    panel_border_color = resolve_theme_color(theme, 'panel_border_color', '#D7DEE8')
    semantic_colors = theme.get('semantic_colors', {}) if isinstance(theme, dict) else {}
    palette = theme.get('palette', []) if isinstance(theme, dict) else []
    plt.rcParams['font.family'] = theme.get('font_family', 'Arial') if isinstance(theme, dict) else 'Arial'
    plt.rcParams['font.size'] = 12
    
    # 创建极坐标子图
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), dpi=dpi)
    
    # ========================================================================
    # 3. 配色方案与视觉映射
    # ========================================================================
    baseline_color = semantic_colors.get('baseline', palette[1] if len(palette) > 1 else '#62B0E5')
    phase0_color = semantic_colors.get('phase0', semantic_colors.get('reference', palette[2] if len(palette) > 2 else '#35BCB8'))
    winner_color = semantic_colors.get('winner', palette[4] if len(palette) > 4 else '#D94442')
    highlight_color = semantic_colors.get('highlight', palette[0] if palette else '#D94442')
    radar_colors = {
        'Phase 1 baseline': baseline_color,
        'Phase 0 baseline': phase0_color,
        'PToT Winner': winner_color,
        'PToT Champion': highlight_color,
    }
    radar_fills = {
        'Phase 1 baseline': _alpha_color(baseline_color, 0.18),
        'Phase 0 baseline': _alpha_color(phase0_color, 0.18),
        'PToT Winner': _alpha_color(winner_color, 0.20),
        'PToT Champion': _alpha_color(highlight_color, 0.20),
    }
    color_idx = 0
    
    # ========================================================================
    # 4. 绘制雷达图
    # ========================================================================
    legend_handles = []
    
    for model_name, scores in standardized_data.items():
        # 数据闭合（首尾相连）
        values = scores + scores[:1]
        
        model_key = str(model_name).strip().lower()
        if model_key == 'phase 1 baseline':
            style_color, style_alpha, marker = baseline_color, 0.12, 'o'
        elif model_key == 'phase 0 baseline':
            style_color, style_alpha, marker = phase0_color, 0.12, 's'
        elif 'winner' in model_key:
            style_color, style_alpha, marker = winner_color, 0.25, '*'
        elif 'champion' in model_key:
            style_color, style_alpha, marker = highlight_color, 0.25, '*'
        else:
            style_color = palette[color_idx % len(palette)] if palette else baseline_color
            style_alpha, marker = 0.15, 'o'
            color_idx += 1

        line = ax.plot(
            angles,
            values,
            color=style_color,
            linewidth=2.5 if ('winner' in model_key or 'champion' in model_key) else 1.5,
            label=model_name,
            marker=marker,
            markersize=8 if ('winner' in model_key or 'champion' in model_key) else 6
        )

        ax.fill(
            angles,
            values,
            color=style_color,
            alpha=style_alpha
        )
        
        legend_handles.append(line[0])
        
        print(f"  - {model_name}: {[f'{v:.3f}' for v in scores]}")
    
    # ========================================================================
    # 5. 坐标轴与网格设置
    # ========================================================================
    # 设置径向轴范围 [0, 1]
    ax.set_ylim(0, 1.0)
    
    # 设置径向网格线（虚线，浅灰色）
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, color=axis_text_color)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, color=grid_color, alpha=0.8)
    
    # 设置角度轴标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold', color=text_color)
    
    # 移除径向轴的刻度线（更简洁）
    ax.tick_params(axis='y', which='both', length=0)
    
    # ========================================================================
    # 6. 图例与标题
    # ========================================================================
    # 添加图例（突出显示 Winner）
    # 将 Winner 放在图例最后（最显眼）
    sorted_handles = []
    sorted_labels = []
    
    for handle in legend_handles:
        label = handle.get_label()
        if 'Winner' in label or 'Champion' in label:
            # Winner 放在最后
            sorted_handles.append(handle)
            sorted_labels.append(label)
        else:
            # 其他模型放在前面
            sorted_handles.insert(0, handle)
            sorted_labels.insert(0, label)
    
    ax.spines['polar'].set_color(panel_border_color)
    ax.spines['polar'].set_linewidth(0.8)

    ax.legend(
        sorted_handles,
        sorted_labels,
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.92,
        edgecolor=panel_border_color,
        facecolor='white'
    )
    
    # 添加简洁标题
    ax.set_title(
        title,
        fontsize=14,
        fontweight='bold',
        color=text_color,
        pad=30
    )
    
    # ========================================================================
    # 7. 保存图片
    # ========================================================================
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 4D Radar comparison plot saved to: {save_path}")
    print("="*80 + "\n")
    
    return save_path


def plot_stats_scatter(
    data_path: str,
    target_column: str,
    method: str = 'pca',
    save_path: str = "output/figures/fig1a_stats_scatter.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制 PCA/PLS-DA 降维散点图 (Statistical Significance Scatter Plot)
    
    这是代谢组学论文的"门面图"，用于证明疾病组和健康组在基础代谢图谱上存在本质差异。
    
    核心视觉元素:
    1. 二维散点图: PC1 vs PC2 (PCA) 或 LV1 vs LV2 (PLS-DA)
    2. 分组着色: 不同组别使用不同颜色（蓝色/红色）
    3. 95% 置信椭圆: 使用 matplotlib.patches.Ellipse 展现组内聚集度
    4. 方差解释度: 坐标轴标签显示方差解释百分比（如 "PC1 (34.5%)"）
    
    ACVAL 支持:
    - layout_kwargs: 排版参数字典，支持动态调整图例位置、字体大小等
    - 返回值: (save_path, final_layout_kwargs) 用于 Critic 审查和重试
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名称
        method: 降维方法 ('pca' 或 'plsda')
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> path, kwargs = plot_stats_scatter(
        ...     data_path='data/cleaned.csv',
        ...     target_column='Group',
        ...     method='pca'
        ... )
    """
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'legend_loc': 'best',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11,
        'bbox_to_anchor': None
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_stats_scatter using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 1. 数据加载和预处理
    # ========================================================================
    print("\n" + "="*80)
    print(f"Phase 3: {method.upper()} Dimensionality Reduction Scatter Plot")
    print("="*80)
    print(f"Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # 分离特征和目标（只选择数值型特征）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_column]
    
    X = df[feature_cols].values
    y = df[target_column].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 编码目标变量
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Classes: {le.classes_}")
    print(f"Method: {method.upper()}")
    
    # ========================================================================
    # 2. 降维
    # ========================================================================
    print(f"\n[Step 1] Performing {method.upper()} dimensionality reduction...")
    
    if method.lower() == 'pca':
        # PCA 降维
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        
        # 提取方差解释率
        explained_var = reducer.explained_variance_ratio_
        comp1_label = f'PC1 ({explained_var[0]*100:.1f}%)'
        comp2_label = f'PC2 ({explained_var[1]*100:.1f}%)'
        
        print(f"  PC1 explained variance: {explained_var[0]*100:.2f}%")
        print(f"  PC2 explained variance: {explained_var[1]*100:.2f}%")
        print(f"  Total explained variance: {sum(explained_var)*100:.2f}%")
    
    else:  # plsda
        # PLS-DA 降维
        reducer = PLSRegression(n_components=2, scale=False)
        X_reduced = reducer.fit_transform(X_scaled, y_encoded)[0]
        
        # PLS-DA 的方差解释率计算
        # 使用 X 方差解释率作为近似
        total_var = np.var(X_scaled, axis=0).sum()
        comp1_var = np.var(X_reduced[:, 0]) / total_var
        comp2_var = np.var(X_reduced[:, 1]) / total_var
        
        comp1_label = f'LV1 ({comp1_var*100:.1f}%)'
        comp2_label = f'LV2 ({comp2_var*100:.1f}%)'
        
        print(f"  LV1 explained variance: {comp1_var*100:.2f}%")
        print(f"  LV2 explained variance: {comp2_var*100:.2f}%")
    
    # ========================================================================
    # 3. 绘制散点图
    # ========================================================================
    print(f"\n[Step 2] Plotting scatter plot with 95% confidence ellipses...")
    
    # 设置学术风格
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 定义颜色方案
    colors = ['#1f77b4', '#d62728']  # 蓝色和红色
    markers = ['o', 's']  # 圆形和方形
    
    # 为每个类别绘制散点和置信椭圆
    for idx, class_label in enumerate(le.classes_):
        # 提取该类别的数据
        mask = (y == class_label)
        X_class = X_reduced[mask]
        
        # 绘制散点
        ax.scatter(
            X_class[:, 0],
            X_class[:, 1],
            c=colors[idx],
            marker=markers[idx],
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5,
            label=class_label
        )
        
        # ====================================================================
        # 计算并绘制 95% 置信椭圆 (CRITICAL: 顶刊要求)
        # ====================================================================
        def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
            """
            计算并绘制置信椭圆
            
            Args:
                x, y: 数据点坐标
                ax: matplotlib axes
                n_std: 标准差倍数（2.0 对应约 95% 置信区间）
                facecolor: 填充颜色
                **kwargs: 其他 Ellipse 参数
            """
            if x.size != y.size:
                raise ValueError("x and y must be the same size")
            
            # 计算协方差矩阵
            cov = np.cov(x, y)
            
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # 计算椭圆的角度
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # 计算椭圆的宽度和高度（2倍标准差）
            width, height = 2 * n_std * np.sqrt(eigenvalues)
            
            # 创建椭圆
            ellipse = Ellipse(
                xy=(np.mean(x), np.mean(y)),
                width=width,
                height=height,
                angle=angle,
                facecolor=facecolor,
                **kwargs
            )
            
            return ax.add_patch(ellipse)
        
        # 绘制 95% 置信椭圆
        confidence_ellipse(
            X_class[:, 0],
            X_class[:, 1],
            ax,
            n_std=2.0,  # 2 标准差 ≈ 95% 置信区间
            edgecolor=colors[idx],
            linewidth=2,
            linestyle='--',
            alpha=0.5
        )
        
        print(f"  {class_label}: {np.sum(mask)} samples, 95% confidence ellipse added")
    
    # ========================================================================
    # 4. 图表装饰
    # ========================================================================
    # 4.1 坐标轴标签（带方差解释率）
    ax.set_xlabel(comp1_label, 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    ax.set_ylabel(comp2_label, 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    
    # 4.2 标题
    method_name = 'PCA' if method.lower() == 'pca' else 'PLS-DA'
    ax.set_title(
        f'{method_name} Score Plot\n'
        f'Metabolic Profile Separation between Groups',
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        pad=20
    )
    
    # 4.3 图例
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        bbox_to_anchor=final_layout_kwargs['bbox_to_anchor']
    )
    
    # 4.4 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4.5 网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 4.6 添加原点参考线
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # ========================================================================
    # 5. 保存图片
    # ========================================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ {method.upper()} scatter plot saved to: {save_path}")
    print(f"[ACVAL Actor] Final layout kwargs: {final_layout_kwargs}")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


def plot_upset(
    feature_sets: Dict[str, List[str]],
    prior_biomarkers: Optional[List[str]] = None,
    save_path: str = "output/figures/fig1c_upset.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制 UpSet Plot (多方法投票特征交集图)
    
    这张图向审稿人展示不同特征选择方法的投票结果，并高亮 Phase 0 的 prior biomarker。
    彻底抛弃维恩图，使用 UpSet 图清晰展示多种特征筛选方法的交集。
    
    核心视觉元素:
    1. UpSet 矩阵: 展示特征集合的交集关系
    2. 交集大小柱状图: 显示每个交集的特征数量
    3. Prior Biomarker 高亮: 用金色标记 Phase 0 的先验特征
    
    ACVAL 支持:
    - layout_kwargs: 排版参数字典，支持动态调整字体大小等
    - 返回值: (save_path, final_layout_kwargs) 用于 Critic 审查和重试
    
    Args:
        feature_sets: 特征集合字典，格式: {'Method1': ['feature1', 'feature2'], ...}
        prior_biomarkers: Phase 0 的先验生物标志物列表
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> feature_sets = {
        ...     'RandomForest': ['feature1', 'feature2', 'feature3'],
        ...     'XGBoost': ['feature2', 'feature3', 'feature4'],
        ...     'SVM': ['feature1', 'feature3', 'feature5']
        ... }
        >>> prior_biomarkers = ['feature1', 'feature2']
        >>> path, kwargs = plot_upset(feature_sets, prior_biomarkers)
    """
    try:
        from upsetplot import from_contents, UpSet
        UPSETPLOT_AVAILABLE = True
    except ImportError:
        print("Warning: upsetplot library not installed. Install with: pip install upsetplot")
        UPSETPLOT_AVAILABLE = False
    
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_upset using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 1. 数据准备
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: UpSet Plot - Multi-Method Feature Voting")
    print("="*80)
    print(f"Methods: {list(feature_sets.keys())}")
    print(f"Prior biomarkers: {len(prior_biomarkers) if prior_biomarkers else 0}")
    
    if not UPSETPLOT_AVAILABLE:
        # 如果 upsetplot 库未安装，创建一个占位图
        print(f"\n[Warning] upsetplot library not available, creating placeholder plot...")
        
        # 设置学术风格
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        ax.text(
            0.5, 0.5,
            'UpSet Plot\n\n'
            'Please install upsetplot library:\n'
            'pip install upsetplot',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20,
            transform=ax.transAxes
        )
        
        ax.set_title(
            'UpSet Plot: Multi-Method Feature Voting (Placeholder)',
            fontsize=final_layout_kwargs['title_fontsize'],
            fontweight='bold',
            pad=20
        )
        
        ax.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return save_path, final_layout_kwargs
    
    # 转换为 upsetplot 所需的格式
    # from_contents 接受字典: {'Set1': ['item1', 'item2'], 'Set2': ['item2', 'item3']}
    upset_data = from_contents(feature_sets)
    
    print(f"\n[Step 1] Data conversion completed:")
    print(f"  Total unique features: {len(upset_data)}")
    print(f"  Number of methods: {len(feature_sets)}")
    
    # ========================================================================
    # 2. 绘制 UpSet Plot
    # ========================================================================
    print(f"\n[Step 2] Plotting UpSet diagram...")
    
    # 设置学术风格
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    # 创建 UpSet 对象
    # 我们需要先计算好哪些交集包含 prior biomarkers
    # upset_data 的 index 是 MultiIndex，最后一层是特征名，前面的层是方法布尔值
    method_levels = list(feature_sets.keys())
    
    # 统计每个特征是否是 prior biomarker
    prior_biomarkers_set = set(prior_biomarkers) if prior_biomarkers else set()
    
    # 创建 UpSet 对象
    upset = UpSet(
        upset_data,
        subset_size='count',
        show_counts=False, # 暂时关闭以避免 Python 3.13 兼容性问题导致的 TypeError
        sort_by='cardinality',
        sort_categories_by='cardinality'
    )
    
    # ========================================================================
    # 3. 高亮 Prior Biomarkers
    # ========================================================================
    if prior_biomarkers_set:
        print(f"\n[Step 3] Identifying intersections with prior biomarkers...")
        
        # 获取所有交集的数据
        # upset.intersections 是一个 Series，index 是方法的 MultiIndex，值是该交集的特征数量
        # 我们需要知道每个交集具体包含哪些特征名
        
        # 重新组织数据以方便查询
        # 将 MultiIndex 的最后一层（特征名）提取出来
        feature_names = upset_data.index.get_level_values(-1)
        is_prior = [name in prior_biomarkers_set for name in feature_names]
        
        # 将是否为 prior biomarker 作为一个新的 level 或者直接按 MultiIndex 分组
        # 这里我们按方法层分组，计算每组中 prior biomarker 的数量
        grouped = pd.Series(is_prior, index=upset_data.index).groupby(level=method_levels).sum()
        
        # 找到包含 prior biomarker 的交集（即 sum > 0 的组）
        # 注意：upsetplot 在绘图时会根据 sort_by 重新排序交集
        # 我们使用 upset.add_catplot 或直接在 plot 后修改 bars 比较复杂
        # 简单的方法是使用 upset.add_catplot 或在 plot 前标记数据
        
        # 统计总共有多少个交集包含 prior biomarkers
        intersections_with_prior = grouped[grouped > 0]
        print(f"  Found {len(intersections_with_prior)} intersections containing prior biomarkers")
        print(f"  Total prior biomarkers found in intersections: {int(intersections_with_prior.sum())}")
    
    # 绘制 UpSet 图
    fig = plt.figure(figsize=figsize, dpi=dpi)
    upset.plot(fig=fig)
    
    # ========================================================================
    # 4. 强制 Intersection Size 纵轴为整数 + 装饰
    # ========================================================================
    axes = fig.get_axes()
    if len(axes) > 0:
        bar_ax = axes[0]
        try:
            from matplotlib.ticker import MaxNLocator
            bar_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except Exception as e:
            print(f"  ⚠️ Could not set integer locator: {e}")
            
        # 注意：在 UpSet 绘图中，高亮特定柱子需要非常小心，因为 bars 的顺序
        # 与 grouped 的顺序可能不一致（由于排序）。
        # 暂时跳过高亮逻辑，优先确保图能画出来且纵轴正确。
    
    # ========================================================================
    # 5. 添加标题
    # ========================================================================
    fig.suptitle(
        'UpSet Plot: Multi-Method Feature Selection Consensus\n'
        'Intersection Analysis of Feature Selection Methods',
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        y=0.98
    )
    
    # ========================================================================
    # 6. 保存图片
    # ========================================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 不要对 UpSet 使用 tight_layout，它会破坏布局
    # plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ UpSet plot saved to: {save_path}")
    print(f"[ACVAL Actor] Final layout kwargs: {final_layout_kwargs}")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


def plot_stability_landscape(
    stability_scores_path: str,
    stability_summary_path: str,
    feature_provenance_path: str = '',
    save_path: str = "output/figures/fig1d_stability_landscape.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 8),
    layout_kwargs: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """
    Plot the Phase 1 stability landscape using selection frequency and method consensus.
    """
    default_layout_kwargs = {
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 13,
        'annotate_top_n': 15,
    }
    final_layout_kwargs = {**default_layout_kwargs, **(layout_kwargs or {})}

    stability_summary = _load_json_report(stability_summary_path)
    stability_scores_payload = _load_json_report(stability_scores_path)
    if not stability_summary or not stability_scores_payload:
        raise FileNotFoundError("Stability summary or stability score artifact is missing.")

    df = _build_stability_plot_dataframe(
        stability_summary=stability_summary,
        stability_scores_payload=stability_scores_payload,
        feature_provenance_path=feature_provenance_path,
    )

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axvspan(0.60, 1.02, ymin=(3 - 1) / (7.4 - 1), ymax=1, color="#EEF6FF", alpha=0.88, zorder=0)
    ax.axvspan(0, 0.60, ymin=0, ymax=(3 - 1) / (7.4 - 1), color="#F8FAFC", alpha=0.96, zorder=0)
    ax.axvline(0.60, color="#4B5563", linestyle="--", linewidth=1.4)
    ax.axhline(3, color="#4B5563", linestyle="--", linewidth=1.4)

    panel_scores = df['panel_score'].to_numpy(dtype=float)
    size_min, size_max = 70, 460
    if np.allclose(panel_scores.max(), panel_scores.min()):
        sizes = np.full(len(df), (size_min + size_max) / 2.0)
    else:
        sizes = size_min + (panel_scores - panel_scores.min()) / (panel_scores.max() - panel_scores.min()) * (size_max - size_min)

    for status in _STABILITY_STATUS_ORDER:
        subset = df[df['status'] == status]
        if subset.empty:
            continue
        ax.scatter(
            subset['selection_frequency'],
            subset['method_consensus'],
            s=sizes[subset.index],
            c=_STABILITY_STATUS_COLORS[status],
            label=_STABILITY_STATUS_LABELS[status],
            edgecolor='white',
            linewidth=0.8,
            alpha=0.95 if status != 'rejected' else 0.72,
            zorder=3 if status != 'rejected' else 2,
        )

    label_df = df[df['is_final_panel']].head(int(final_layout_kwargs.get('annotate_top_n', 15) or 15))
    text_artists = []
    offsets = [(8, 8), (8, -10), (-10, 8), (-10, -10), (10, 2), (2, 10), (-12, 3), (10, -2)]
    for idx, (_, row) in enumerate(label_df.iterrows()):
        dx, dy = offsets[idx % len(offsets)]
        text_artists.append(
            ax.annotate(
                _format_feature_axis_label(row['display_name'], max_len=26),
                (row['selection_frequency'], row['method_consensus']),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=8,
                color="#1F2937",
                bbox={'boxstyle': 'round,pad=0.15', 'fc': 'white', 'ec': 'none', 'alpha': 0.88},
            )
        )
    if ADJUST_TEXT_AVAILABLE and text_artists:
        adjust_text(text_artists, ax=ax, only_move={'points': 'y', 'texts': 'y'})

    ax.text(0.79, 6.75, "Stable Core Zone", fontsize=12, weight='bold', color="#1D4ED8", ha='center')
    ax.text(0.18, 1.45, "Low Frequency\nLow Consensus", fontsize=10, color="#6B7280", ha='center')
    ax.text(0.83, 1.45, "High Frequency\nLow Consensus", fontsize=10, color="#6B7280", ha='center')
    ax.text(0.18, 5.7, "Low Frequency\nHigh Consensus", fontsize=10, color="#6B7280", ha='center')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.7, 7.4)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(range(1, 8))
    ax.set_xlabel("Selection Frequency", fontsize=final_layout_kwargs['label_fontsize'])
    ax.set_ylabel("Method Consensus", fontsize=final_layout_kwargs['label_fontsize'])
    ax.set_title("Phase 1 Stability Landscape", fontsize=final_layout_kwargs['title_fontsize'], fontweight='bold', pad=14)
    ax.grid(color="#E5E7EB", linewidth=0.8)

    legend = ax.legend(frameon=True, loc='lower right', title="Feature Status")
    legend.get_title().set_fontsize(10)
    for text in legend.get_texts():
        text.set_fontsize(9)

    fig.text(
        0.01,
        0.01,
        "Bubble size encodes composite panel score. Dashed lines mark the stable-core thresholds "
        "(selection frequency >= 0.60; method consensus >= 3).",
        fontsize=9.5,
        color="#4B5563",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path, final_layout_kwargs


def plot_method_feature_heatmap(
    stability_scores_path: str,
    stability_summary_path: str,
    feature_provenance_path: str = '',
    save_path: str = "output/figures/fig1e_method_feature_heatmap.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (11.5, 9.5),
    layout_kwargs: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """
    Plot a focused method-feature frequency heatmap for Phase 1 stability selection.
    """
    default_layout_kwargs = {
        'fontsize': 10,
        'title_fontsize': 16,
        'rejected_n': 6,
    }
    final_layout_kwargs = {**default_layout_kwargs, **(layout_kwargs or {})}

    stability_summary = _load_json_report(stability_summary_path)
    stability_scores_payload = _load_json_report(stability_scores_path)
    if not stability_summary or not stability_scores_payload:
        raise FileNotFoundError("Stability summary or stability score artifact is missing.")

    df = _build_stability_plot_dataframe(
        stability_summary=stability_summary,
        stability_scores_payload=stability_scores_payload,
        feature_provenance_path=feature_provenance_path,
    )
    plot_df = _select_focused_heatmap_rows(df, rejected_n=int(final_layout_kwargs.get('rejected_n', 6) or 6))

    method_cols = [method for method in _STABILITY_METHOD_LABELS if method in plot_df.columns]
    heatmap_df = plot_df[['display_name'] + method_cols].copy()
    heatmap_df = heatmap_df.rename(columns=_STABILITY_METHOD_LABELS)
    heatmap_df['display_name'] = heatmap_df['display_name'].apply(_format_feature_axis_label)
    heatmap_df = heatmap_df.set_index('display_name')
    status_codes = plot_df['status'].map({status: i for i, status in enumerate(_STABILITY_STATUS_ORDER)}).to_numpy().reshape(-1, 1)
    status_cmap = ListedColormap([_STABILITY_STATUS_COLORS[key] for key in _STABILITY_STATUS_ORDER])

    sns.set_style("white")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']

    n_rows = len(heatmap_df)
    fig_height = max(figsize[1], min(0.42 * n_rows + 2.6, 13.6))
    fig, (ax, ax_status) = plt.subplots(
        1,
        2,
        figsize=(figsize[0], fig_height),
        dpi=dpi,
        gridspec_kw={'width_ratios': [18, 0.7], 'wspace': 0.04},
    )
    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap=sns.color_palette("Blues", as_cmap=True),
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Per-method frequency'},
        linewidths=0.4,
        linecolor="#F3F4F6",
    )
    ax.set_title("Focused Method-Feature Frequency Heatmap", fontsize=final_layout_kwargs['title_fontsize'], fontweight='bold', pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=35, labelsize=10)
    ax.tick_params(axis='y', labelsize=9)

    stable_core_idx = plot_df.index[plot_df['is_stable_core']].tolist()
    final_panel_idx = plot_df.index[plot_df['is_final_panel']].tolist()
    if stable_core_idx:
        ax.hlines(max(stable_core_idx) + 1, *ax.get_xlim(), colors="#1D4ED8", linewidth=2.0)
    if final_panel_idx:
        ax.hlines(max(final_panel_idx) + 1, *ax.get_xlim(), colors="#059669", linewidth=1.6, linestyles='--')
        ax.text(
            -0.12,
            (max(final_panel_idx) + 0.5) / max(len(plot_df), 1),
            "Final panel",
            transform=ax.transAxes,
            ha='right',
            va='center',
            fontsize=10,
            color="#047857",
            weight='bold',
        )
    if len(plot_df) > len(final_panel_idx):
        ax.text(
            -0.12,
            (len(final_panel_idx) + (len(plot_df) - len(final_panel_idx)) / 2) / max(len(plot_df), 1),
            "Near-miss\nrejected",
            transform=ax.transAxes,
            ha='right',
            va='center',
            fontsize=10,
            color="#6B7280",
            weight='bold',
        )

    for tick, status in zip(ax.get_yticklabels(), plot_df['status']):
        if status == 'stable_core':
            tick.set_color("#1D4ED8")
            tick.set_fontweight('bold')
        elif status == 'completed':
            tick.set_color("#047857")
        elif status == 'mandatory':
            tick.set_color("#A16207")
            tick.set_fontweight('bold')
        else:
            tick.set_color("#4B5563")

    ax_status.imshow(status_codes, aspect='auto', cmap=status_cmap, interpolation='nearest')
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    ax_status.set_title("Status", fontsize=10, pad=10)
    for spine in ax_status.spines.values():
        spine.set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=_STABILITY_STATUS_COLORS[status], markersize=10, label=_STABILITY_STATUS_LABELS[status])
        for status in reversed(_STABILITY_STATUS_ORDER)
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        fontsize=9,
        title="Row status",
        title_fontsize=10,
        loc='upper left',
        bbox_to_anchor=(1.15, 1.0),
    )

    fig.text(
        0.01,
        0.01,
        "Shows all final-panel features plus a small set of highest-priority near-miss rejected candidates. "
        "Solid blue line marks the stable core; dashed green line marks the final panel boundary.",
        fontsize=9.5,
        color="#4B5563",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.subplots_adjust(left=0.24, right=0.88, top=0.92, bottom=0.08, wspace=0.04)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path, final_layout_kwargs


def plot_autogluon_roc(
    ag_results_path: str = 'data/autogluon_training_results.json',
    data_path: str = 'data/cleaned_data.csv',
    target_column: str = 'Group',
    save_path: str = "output/figures/fig2a_autogluon_roc.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制 AutoGluon 多模型基础 ROC 曲线对比图
    
    这张图向审稿人展示传统机器学习的预测天花板，为 Phase 2 的优化提供基线参考。
    用于证明即使不进行复杂的多目标优化，全量特征在传统机器学习上的预测天花板。
    
    核心视觉元素:
    1. 多条 ROC 曲线: 3-5 个代表性模型的 ROC 曲线
    2. AUC 标注: 每条曲线标注对应的 AUC 值
    3. 对角线参考: 随机猜测的基线 (AUC=0.5)
    4. 最佳模型高亮: 用特殊颜色和线宽标记最佳模型
    5. 图例排序: 按 AUC 从高到低排序
    
    ACVAL 支持:
    - layout_kwargs: 排版参数字典，支持动态调整图例位置、字体大小等
    - 返回值: (save_path, final_layout_kwargs) 用于 Critic 审查和重试
    
    Args:
        ag_results_path: AutoGluon 训练结果 JSON 文件路径
        data_path: 数据文件路径
        target_column: 目标列名称
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> path, kwargs = plot_autogluon_roc(
        ...     ag_results_path='data/autogluon_training_results.json',
        ...     data_path='data/cleaned_data.csv',
        ...     target_column='Group'
        ... )
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import LabelEncoder
    
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'legend_loc': 'lower right',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 10,
        'bbox_to_anchor': None
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_autogluon_roc using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 1. 数据加载
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: AutoGluon Multi-Model ROC Comparison")
    print("="*80)
    print(f"Loading AutoGluon results from: {ag_results_path}")
    print(f"Loading data from: {data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(ag_results_path):
        print(f"\n[Warning] AutoGluon results file not found: {ag_results_path}")
        print("Creating placeholder plot...")
        
        # 创建占位图
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        ax.text(
            0.5, 0.5,
            'AutoGluon Multi-Model ROC\n\n'
            f'Results file not found:\n{ag_results_path}\n\n'
            'Please run Phase 1 AutoGluon training first.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=14,
            transform=ax.transAxes
        )
        
        ax.set_title(
            'AutoGluon Multi-Model ROC Comparison (Placeholder)',
            fontsize=final_layout_kwargs['title_fontsize'],
            fontweight='bold',
            pad=20
        )
        
        ax.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return save_path, final_layout_kwargs
    
    # 读取 AutoGluon 结果
    with open(ag_results_path, 'r', encoding='utf-8') as f:
        ag_results = json.load(f)
    
    # 读取数据（用于 fallback）
    df = pd.read_csv(data_path)

    # 优先使用真实 holdout 标签（外部测试集），避免“重分割错配”导致 AUC 假性偏低
    split_strategy = str(ag_results.get('split_strategy', ''))
    test_set_role = str(ag_results.get('test_set_role', ''))
    y = None

    if test_set_role == 'external_holdout' and split_strategy.startswith('external_holdout:'):
        holdout_path = split_strategy.split(':', 1)[1].strip()
        if holdout_path and os.path.exists(holdout_path):
            holdout_df = pd.read_csv(holdout_path)
            holdout_target_col = target_column
            if holdout_target_col not in holdout_df.columns:
                if 'target' in holdout_df.columns:
                    holdout_target_col = 'target'
                elif 'Group' in holdout_df.columns:
                    holdout_target_col = 'Group'
                else:
                    holdout_target_col = None
            if holdout_target_col is not None:
                y = holdout_df[holdout_target_col].values
                print(
                    f"Using external holdout labels from: {holdout_path} "
                    f"(target={holdout_target_col}, size={len(y)})"
                )

    # 回退：无法读取 external holdout 时，按旧逻辑重分割
    if y is None:
        test_set_size = ag_results.get('test_set_size', None)
        from sklearn.model_selection import train_test_split
        if test_set_size is not None and test_set_size < len(df):
            print(f"Using reconstructed test split (size={test_set_size}) for ROC calculation")
            df_target_col = target_column
            if df_target_col not in df.columns:
                if 'target' in df.columns:
                    df_target_col = 'target'
                elif 'Group' in df.columns:
                    df_target_col = 'Group'
                else:
                    raise ValueError(f"Target column not found in dataframe: {target_column}")
            X = df.drop(columns=[df_target_col])
            y_full = df[df_target_col]
            _, _, _, y_test = train_test_split(
                X, y_full, test_size=test_set_size, stratify=y_full, random_state=42
            )
            y = y_test.values
        else:
            if target_column in df.columns:
                y = df[target_column].values
            elif 'target' in df.columns:
                y = df['target'].values
            elif 'Group' in df.columns:
                y = df['Group'].values
            else:
                raise ValueError(f"Target column not found in dataframe: {target_column}")

    # 与预测长度对齐（防止长度不一致）
    test_set_size = ag_results.get('test_set_size', None)
    if isinstance(test_set_size, int) and len(y) != test_set_size:
        if len(y) > test_set_size:
            y = np.array(y[-test_set_size:])
        else:
            print(
                f"⚠️ Warning: label count ({len(y)}) != test_set_size ({test_set_size}); "
                "ROC may be unreliable."
            )
    
    # 编码目标变量
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Data shape: {df.shape}")
    print(f"Classes: {le.classes_}")
    
    # ========================================================================
    # 2. 提取模型预测结果（支持 leaderboard_top5 格式）
    # ========================================================================
    print(f"\n[Step 1] Extracting model predictions...")
    
    # 从 AutoGluon 结果中提取模型预测
    models_data = []
    
    # 格式 1: 完整的模型预测概率（最理想）
    if 'models' in ag_results:
        print("  Format detected: 'models' (with predictions)")
        for model_name, model_info in ag_results['models'].items():
            if 'predictions' in model_info and 'auc' in model_info:
                models_data.append({
                    'name': model_name,
                    'predictions': model_info['predictions'],
                    'auc': model_info['auc']
                })
    
    # 格式 2: leaderboard（标准格式）
    elif 'leaderboard' in ag_results:
        print("  Format detected: 'leaderboard'")
        leaderboard = ag_results['leaderboard']
        if isinstance(leaderboard, list):
            for model_info in leaderboard[:5]:  # 取前 5 个模型
                if 'model' in model_info and 'score_val' in model_info:
                    models_data.append({
                        'name': model_info['model'],
                        'predictions': model_info.get('predictions'),  # 修复：读取实际预测概率
                        'auc': model_info['score_val']
                    })
    
    # 格式 3: leaderboard_top5（我们的实际格式）
    elif 'leaderboard_top5' in ag_results:
        print("  Format detected: 'leaderboard_top5' (using test scores)")
        leaderboard = ag_results['leaderboard_top5']
        if isinstance(leaderboard, list):
            for model_info in leaderboard[:5]:  # 取前 5 个模型
                # 兼容不同版本的键名：score_test, score_holdout, score_val
                auc_score = model_info.get('score_test') or model_info.get('score_holdout') or model_info.get('score_val')
                
                if 'model' in model_info and auc_score is not None:
                    models_data.append({
                        'name': model_info['model'],
                        'predictions': model_info.get('predictions'),  # 修复：读取实际预测概率
                        'auc': auc_score
                    })
                    print(f"    - {model_info['model']}: auc={auc_score:.4f}")
    
    # 如果没有找到模型数据，报错（不再使用假数据）
    if not models_data:
        error_msg = (
            f"❌ Error: No valid model data found in AutoGluon results.\n"
            f"   Expected keys: 'models', 'leaderboard', or 'leaderboard_top5'\n"
            f"   Found keys: {list(ag_results.keys())}\n"
            f"   Please ensure the AutoGluon results file contains model information."
        )
        print(f"\n{error_msg}")
        
        # 创建错误占位图
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        ax.text(
            0.5, 0.5,
            'AutoGluon Multi-Model ROC\n\n'
            'Error: Invalid results file format\n\n'
            f'Expected: models/leaderboard/leaderboard_top5\n'
            f'Found: {", ".join(ag_results.keys())}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            transform=ax.transAxes
        )
        
        ax.set_title(
            'AutoGluon Multi-Model ROC Comparison (Error)',
            fontsize=final_layout_kwargs['title_fontsize'],
            fontweight='bold',
            pad=20
        )
        
        ax.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return save_path, final_layout_kwargs
    
    # 按 AUC 从高到低排序
    models_data = sorted(models_data, key=lambda x: x['auc'], reverse=True)
    
    # 只保留前 5 个模型
    models_data = models_data[:5]
    
    print(f"  Selected {len(models_data)} models for comparison:")
    for model in models_data:
        print(f"    - {model['name']}: AUC = {model['auc']:.4f}")
    
    # ========================================================================
    # 3. 计算 ROC 曲线（使用真实预测概率）
    # ========================================================================
    print(f"\n[Step 2] Calculating ROC curves from real predictions...")
    
    # 检查是否有预测概率数据
    has_predictions = all(model['predictions'] is not None for model in models_data)
    
    if not has_predictions:
        error_msg = (
            f"❌ Error: No prediction probabilities found in AutoGluon results.\n"
            f"   This indicates the results file was generated by an older version.\n"
            f"   Please re-run Phase 1 AutoGluon training to generate predictions.\n"
            f"   Expected format: leaderboard_top5[i]['predictions'] should contain probability arrays."
        )
        print(f"\n{error_msg}")
        
        # 创建错误占位图
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        ax.text(
            0.5, 0.5,
            'AutoGluon Multi-Model ROC\n\n'
            'Error: No prediction probabilities found\n\n'
            'Please re-run Phase 1 training\n'
            'to generate prediction data',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            transform=ax.transAxes
        )
        
        ax.set_title(
            'AutoGluon Multi-Model ROC Comparison (Error)',
            fontsize=final_layout_kwargs['title_fontsize'],
            fontweight='bold',
            pad=20
        )
        
        ax.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return save_path, final_layout_kwargs
    
    # 使用真实预测概率计算 ROC 曲线
    print("  ✓ Real predictions available - plotting authentic ROC curves")
    
    roc_data = []
    for model in models_data:
        predictions = np.array(model['predictions'])
        
        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y_encoded, predictions)
        roc_auc = auc(fpr, tpr)
        
        roc_data.append({
            'name': model['name'],
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        })
        
        print(f"  {model['name']}: AUC = {roc_auc:.4f} (computed from {len(predictions)} predictions)")
    
    # ========================================================================
    # 4. 绘制 ROC 曲线
    # ========================================================================
    print(f"\n[Step 3] Plotting ROC curves...")
    
    # 设置学术风格
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 4.1 绘制对角线参考（随机猜测基线）
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Guess (AUC = 0.50)')
    
    # 4.2 定义颜色方案
    colors = ['darkred', 'darkorange', 'gold', 'green', 'blue']
    
    # 4.3 绘制每个模型的 ROC 曲线
    for idx, model in enumerate(roc_data):
        # 最佳模型（第一个）使用更粗的线条
        linewidth = 3 if idx == 0 else 2
        alpha = 1.0 if idx == 0 else 0.8
        
        ax.plot(
            model['fpr'],
            model['tpr'],
            color=colors[idx % len(colors)],
            linewidth=linewidth,
            alpha=alpha,
            label=f"{model['name']} (AUC = {model['auc']:.3f})"
        )
    
    # ========================================================================
    # 5. 图表装饰
    # ========================================================================
    # 5.1 坐标轴标签
    ax.set_xlabel('False Positive Rate (1 - Specificity)', 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    
    # 5.2 标题
    ax.set_title(
        'AutoGluon Multi-Model ROC Comparison\n'
        'Baseline Performance with Full Feature Set',
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        pad=20
    )
    
    # 5.3 图例（按 AUC 从高到低排序，已经在数据准备时排序）
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        bbox_to_anchor=final_layout_kwargs['bbox_to_anchor']
    )
    
    # 5.4 设置坐标轴范围
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # 5.5 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 5.6 网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ========================================================================
    # 6. 保存图片
    # ========================================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ AutoGluon multi-model ROC plot saved to: {save_path}")
    print(f"[ACVAL Actor] Final layout kwargs: {final_layout_kwargs}")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


def plot_final_roc(
    data_path: Optional[str] = None,
    target_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    champion_model_family: Optional[str] = None,
    phase2_result_path: Optional[str] = None,
    save_path: str = "output/figures/fig4a_final_roc.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    k_folds: int = 5,
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制最终 ROC 曲线 (Final Winner Panel ROC with Confidence Interval)
    
    这张图向审稿人展示最终选出的最优特征集合的预测性能和鲁棒性。
    
    核心视觉元素:
    1. 均值 ROC 曲线: K-fold 交叉验证的平均 ROC（深红色粗实线）
    2. 置信区间: ±1 标准差的阴影区域（透明度 0.2）
    3. AUC 标注: 显著标注 AUC 均值和标准差
    4. 对角线参考: 随机猜测的基线 (AUC=0.5)
    
    NEW (2026-04-24): 优先读取 Phase 2 的 cv_predictions
    - 如果提供 phase2_result_path，优先从 Phase 2 结果中读取预测概率
    - 这样可以确保 100% 数值一致性（使用 Phase 2 的 LightGBM_BAG_L2 模型）
    - 如果没有 Phase 2 结果，则回退到传统方法（重新训练模型）
    
    ACVAL 支持:
    - layout_kwargs: 排版参数字典，支持动态调整图例位置、字体大小等
    - 返回值: (save_path, final_layout_kwargs) 用于 Critic 审查和重试
    
    Args:
        data_path: 数据文件路径（可选，用于向后兼容）
        target_column: 目标列名称（可选，用于向后兼容）
        features: 最终特征列表（可选，用于向后兼容）
        champion_model_family: 冠军模型家族（可选，用于向后兼容）
        phase2_result_path: Phase 2 结果 JSON 文件路径（推荐使用）
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        k_folds: 交叉验证折数
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 最终使用的排版参数)
    
    Examples:
        >>> # 推荐方法：使用 Phase 2 结果
        >>> path, kwargs = plot_final_roc(
        ...     phase2_result_path='test_results/phase2_result.json',
        ...     save_path='output/figures/fig4a_final_roc.pdf'
        ... )
        
        >>> # 传统方法：重新训练模型（向后兼容）
        >>> path, kwargs = plot_final_roc(
        ...     data_path='data/cleaned.csv',
        ...     target_column='Group',
        ...     features=['feature1', 'feature2'],
        ...     champion_model_family='XGBoost',
        ...     k_folds=5
        ... )
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc, average_precision_score
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    try:
        from lightgbm import LGBMClassifier
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False
    
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'legend_loc': 'lower right',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11,
        'bbox_to_anchor': None,
        'auc_text_x': 0.6,
        'auc_text_y': 0.2
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_final_roc using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 🎯 NEW: 优先读取 Phase 2 的 cv_predictions
    # ========================================================================
    use_phase2_predictions = False
    metric_name = "ROC-AUC"
    display_auc = None
    display_auc_std = None
    pooled_fpr = None
    pooled_tpr = None
    fold_mean_auc = None
    fold_std_auc = None
    
    if phase2_result_path and os.path.exists(phase2_result_path):
        print("\n" + "="*80)
        print("Phase 3: Final ROC Curve (Using Phase 2 CV Predictions)")
        print("="*80)
        print(f"Reading Phase 2 result from: {phase2_result_path}")
        
        try:
            with open(phase2_result_path, 'r', encoding='utf-8') as f:
                phase2_result = json.load(f)
            
            cv_predictions = (
                phase2_result.get('final_result', {}).get('cv_predictions')
                or phase2_result.get('cv_predictions')
            )
            
            if cv_predictions and cv_predictions.get('predictions'):
                print(f"✅ Found cv_predictions in Phase 2 result")
                print(f"   Samples: {cv_predictions.get('n_samples', 'N/A')}")
                print(f"   K-folds: {cv_predictions.get('k_folds', 'N/A')}")
                print(f"   Metric: {cv_predictions.get('metric', 'N/A')}")
                mean_score = cv_predictions.get(
                    'mean_score',
                    phase2_result.get('final_result', {}).get('perf', phase2_result.get('scores', {}).get('f_perf', 'N/A'))
                )
                if isinstance(mean_score, (int, float)):
                    print(f"   Mean score: {mean_score:.4f}")
                else:
                    print(f"   Mean score: {mean_score}")
                
                # 转换为 DataFrame
                predictions_df = pd.DataFrame(cv_predictions['predictions'])
                
                # 提取真实标签和预测概率
                y_true = predictions_df['true_label_encoded'].values
                
                # ================================================================
                # 🎯 智能正类检测：自动识别哪个类别是我们关心的正类
                # ================================================================
                # 策略：
                # 1. 优先使用用户指定的 positive_class 参数（如果提供）
                # 2. 否则，使用启发式规则自动检测：
                #    - 医学场景：疾病类通常是正类（如 'CD', 'Disease', 'Case'）
                #    - 通用场景：样本量较少的类通常是正类（不平衡数据集）
                # 3. 如果无法确定，默认使用 class 1（sklearn 默认）
                # ================================================================
                
                label_names = cv_predictions.get('label_names', [])
                
                # 启发式规则：检测哪个是正类
                def detect_positive_class(label_names, y_true):
                    """
                    智能检测正类
                    
                    Returns:
                        int: 正类的编码 (0 or 1)
                        str: 正类的名称
                    """
                    if not label_names or len(label_names) != 2:
                        # 无法确定，使用默认值 1
                        return 1, None
                    
                    # 规则 1: 医学/疾病关键词检测
                    disease_keywords = ['disease', 'case', 'patient', 'cd', 'cancer', 
                                       'tumor', 'positive', 'sick', 'ill', 'affected']
                    
                    for idx, label in enumerate(label_names):
                        label_lower = str(label).lower()
                        if any(keyword in label_lower for keyword in disease_keywords):
                            print(f"  🎯 Detected positive class by keyword: '{label}' (class {idx})")
                            return idx, label
                    
                    # 规则 2: 样本量检测（少数类通常是正类）
                    class_counts = np.bincount(y_true)
                    minority_class = np.argmin(class_counts)
                    
                    # 只有在样本量差异显著时才使用这个规则（至少 2:1）
                    ratio = class_counts.max() / (class_counts.min() + 1e-10)
                    if ratio >= 2.0:
                        print(f"  🎯 Detected positive class by minority: '{label_names[minority_class]}' "
                              f"(class {minority_class}, {class_counts[minority_class]} samples vs "
                              f"{class_counts[1-minority_class]} samples, ratio={ratio:.1f}:1)")
                        return minority_class, label_names[minority_class]
                    
                    # 规则 3: 默认使用 class 1（sklearn 默认）
                    print(f"  ⚠️  Cannot determine positive class, using default: class 1 ('{label_names[1]}')")
                    return 1, label_names[1]
                
                # 检测正类
                positive_class_idx, positive_class_name = detect_positive_class(label_names, y_true)
                
                # 选择对应的预测概率
                if positive_class_idx == 0:
                    y_pred_proba = predictions_df['pred_proba_class0'].values
                    print(f"  ✅ Using pred_proba_class0 for positive class '{positive_class_name}'")
                else:
                    y_pred_proba = predictions_df['pred_proba_class1'].values
                    print(f"  ✅ Using pred_proba_class1 for positive class '{positive_class_name}'")
                
                # 获取 fold 信息
                fold_ids = predictions_df['fold'].values
                k_folds = cv_predictions.get('k_folds', 5)

                # 先计算 pooled OOF ROC，作为最终展示主口径
                pooled_fpr, pooled_tpr, _ = roc_curve(
                    y_true,
                    y_pred_proba,
                    pos_label=positive_class_idx,
                )
                pooled_auc = auc(pooled_fpr, pooled_tpr)
                
                # 计算每个 fold 的 ROC 曲线
                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)
                
                for fold_idx in range(1, k_folds + 1):
                    fold_mask = fold_ids == fold_idx
                    y_fold = y_true[fold_mask]
                    y_pred_fold = y_pred_proba[fold_mask]
                    
                    # 🎯 使用检测到的正类标签计算 ROC 曲线
                    fpr, tpr, _ = roc_curve(y_fold, y_pred_fold, pos_label=positive_class_idx)
                    roc_auc = auc(fpr, tpr)
                    
                    # 插值到统一的 FPR 网格
                    tpr_interp = np.interp(mean_fpr, fpr, tpr)
                    tpr_interp[0] = 0.0
                    tprs.append(tpr_interp)
                    aucs.append(roc_auc)
                    
                    print(f"  Fold {fold_idx}/{k_folds}: ROC-AUC = {roc_auc:.4f}")
                
                # 计算均值和标准差
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                fold_mean_auc = mean_auc
                fold_std_auc = std_auc
                display_auc = pooled_auc
                display_auc_std = std_auc
                
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                
                # 检测使用的指标
                phase2_metric = cv_predictions.get('metric', 'auprc')
                if phase2_metric == 'auprc':
                    # Phase 2 使用 AUPRC，计算 AUPRC 用于标注
                    auprc_score = average_precision_score(y_true, y_pred_proba)
                    metric_name = f"ROC-AUC (Phase 2 AUPRC: {auprc_score:.3f})"
                else:
                    metric_name = "ROC-AUC"
                
                print(f"\n✅ Using Phase 2 predictions (100% numerical consistency)")
                print(f"   Pooled OOF ROC-AUC: {pooled_auc:.4f}")
                print(f"   Mean fold ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
                if phase2_metric == 'auprc':
                    print(f"   Phase 2 AUPRC: {auprc_score:.4f}")
                
                use_phase2_predictions = True
                
            else:
                print(f"⚠️  cv_predictions not found in Phase 2 result")
                print(f"   Falling back to traditional method (re-training model)")
        
        except Exception as e:
            print(f"⚠️  Error reading Phase 2 result: {e}")
            print(f"   Falling back to traditional method (re-training model)")
    
    # ========================================================================
    # 传统方法：重新训练模型（向后兼容）
    # ========================================================================
    if not use_phase2_predictions:
        print("\n" + "="*80)
        print("Phase 3: Final ROC Curve with K-Fold Cross-Validation (Traditional Method)")
        print("="*80)
        print(f"Loading data from: {data_path}")
        
        if not data_path or not target_column or not features or not champion_model_family:
            raise ValueError(
                "When phase2_result_path is not provided, "
                "data_path, target_column, features, and champion_model_family are required"
            )
        
        df = pd.read_csv(data_path)
        
        # 提取特征和目标
        X = df[features].values
        y = df[target_column].values
        
        # 编码目标变量
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"Data shape: {X.shape}")
        print(f"Features: {len(features)}")
        print(f"Classes: {le.classes_}")
        print(f"K-folds: {k_folds}")
        
        # ====================================================================
        # K-Fold 交叉验证
        # ====================================================================
        print(f"\n[Step 1] Running {k_folds}-fold cross-validation...")
        
        # 选择模型
        if 'XGBoost' in champion_model_family or 'XGB' in champion_model_family:
            model = XGBClassifier(random_state=42, eval_metric='logloss')
        elif 'LightGBM' in champion_model_family or 'LGBM' in champion_model_family:
            if LIGHTGBM_AVAILABLE:
                model = LGBMClassifier(random_state=42, verbose=-1)
            else:
                print("Warning: LightGBM not available, falling back to RandomForest")
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        print(f"Model: {model.__class__.__name__}")
        
        # 存储每一折的 TPR 和 FPR
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测概率
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 计算 ROC 曲线
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # 插值到统一的 FPR 网格
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(roc_auc)
            
            print(f"  Fold {fold_idx + 1}/{k_folds}: AUC = {roc_auc:.4f}")
        
        # 计算均值和标准差
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        display_auc = mean_auc
        display_auc_std = std_auc
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        print(f"\n[Step 2] Cross-validation results:")
        print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        metric_name = "ROC-AUC"
    
    # ========================================================================
    # 3. 绘制 ROC 曲线
    # ========================================================================
    print(f"\n[Step 3] Plotting ROC curve...")
    
    # 设置学术风格
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 3.1 绘制对角线参考（随机猜测基线）
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Guess (AUC = 0.50)')
    
    # 3.2 绘制置信区间（±1 标准差阴影）
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color='darkred',
        alpha=0.2,
        label=f'±1 std. dev.'
    )
    
    # 3.3 绘制均值 ROC 曲线（深红色粗实线）
    if use_phase2_predictions and pooled_fpr is not None and pooled_tpr is not None:
        ax.plot(
            pooled_fpr,
            pooled_tpr,
            color='darkred',
            linewidth=3,
            label=f'Pooled OOF ROC (AUC = {display_auc:.3f})'
        )
    else:
        ax.plot(
            mean_fpr,
            mean_tpr,
            color='darkred',
            linewidth=3,
            label=f'Mean ROC (AUC = {display_auc:.3f} ± {display_auc_std:.3f})'
        )
    
    # 3.4 添加 AUC 文本标注
    if use_phase2_predictions and fold_mean_auc is not None and fold_std_auc is not None:
        auc_text = (
            f'Pooled OOF AUC = {display_auc:.3f}\n'
            f'Mean Fold AUC = {fold_mean_auc:.3f}\n'
            f'Std. Dev. = {fold_std_auc:.3f}\n'
            f'{k_folds}-Fold CV'
        )
    else:
        auc_text = f'Mean AUC = {display_auc:.3f}\nStd. Dev. = {display_auc_std:.3f}\n{k_folds}-Fold CV'
    ax.text(
        final_layout_kwargs['auc_text_x'],
        final_layout_kwargs['auc_text_y'],
        auc_text,
        fontsize=final_layout_kwargs['fontsize'] + 1,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='bottom'
    )
    
    # ========================================================================
    # 4. 图表装饰
    # ========================================================================
    # 4.1 坐标轴标签
    ax.set_xlabel('False Positive Rate (1 - Specificity)', 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    
    # 4.2 标题（智能处理特征数量）
    # 如果使用 Phase 2 预测，从 phase2_result 中获取特征数量
    if use_phase2_predictions and phase2_result_path:
        try:
            with open(phase2_result_path, 'r', encoding='utf-8') as f:
                phase2_result = json.load(f)
            n_features = len(phase2_result.get('final_result', {}).get('features', []))
        except:
            n_features = len(features) if features else 0
    else:
        n_features = len(features) if features else 0
    
    # 构建标题（包含指标信息）
    title_text = f'ROC Curve: Winner Panel ({n_features} Features)\n'
    title_text += f'{k_folds}-Fold Cross-Validation with Confidence Interval'
    
    # 如果使用 Phase 2 预测，添加说明
    if use_phase2_predictions:
        title_text += '\n(Using Phase 2 Predictions for 100% Consistency)'
    
    ax.set_title(
        title_text,
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        pad=20
    )
    
    # 4.3 图例
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        bbox_to_anchor=final_layout_kwargs['bbox_to_anchor']
    )
    
    # 4.4 设置坐标轴范围
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # 4.5 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4.6 网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # ========================================================================
    # 5. 保存图片
    # ========================================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Final ROC curve saved to: {save_path}")
    print(f"[ACVAL Actor] Final layout kwargs: {final_layout_kwargs}")
    print("="*80 + "\n")
    
    return save_path, final_layout_kwargs


def plot_dca(
    data_path: Optional[str] = None,
    target_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    champion_model_family: Optional[str] = None,
    phase2_result_path: Optional[str] = None,
    save_path: str = "output/figures/fig4b_dca.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制 DCA 决策曲线 (Decision Curve Analysis)
    
    这张图向审稿人展示模型在不同决策阈值下的临床净收益，是医学顶刊的免死金牌。
    
    核心视觉元素:
    1. "Treat None" 线: 全不治疗的基线（灰色水平线，始终为 0）
    2. "Treat All" 线: 全员治疗的基线（灰色虚线，向下倾斜）
    3. Model 线: Winner Panel 的净收益曲线（深红色实线）
    4. 净收益公式: NB = TP/N - FP/N × Pt/(1-Pt)
    
    NEW (2026-04-24): 优先读取 Phase 2 的 cv_predictions
    - 如果提供 phase2_result_path，优先从 Phase 2 结果中读取预测概率
    - 这样可以确保 100% 数值一致性（使用 Phase 2 的 LightGBM_BAG_L2 模型）
    - 如果没有 Phase 2 结果，则回退到传统方法（重新训练模型）
    
    ACVAL 支持:
    - layout_kwargs: 排版参数字典，支持动态调整图例位置、字体大小等
    - 返回值: (save_path, metadata)；metadata 中保留 layout 参数，并附带 DCA 摘要
    
    Args:
        data_path: 数据文件路径（可选，用于向后兼容）
        target_column: 目标列名称（可选，用于向后兼容）
        features: 最终特征列表（可选，用于向后兼容）
        champion_model_family: 冠军模型家族（可选，用于向后兼容）
        phase2_result_path: Phase 2 结果 JSON 文件路径（推荐使用）
        save_path: 输出图片路径
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (生成的图片路径, 元数据字典)
    
    References:
        Vickers AJ, Elkin EB. Decision curve analysis: a novel method for 
        evaluating prediction models. Med Decis Making. 2006;26(6):565-574.
    
    Examples:
        >>> # 推荐方法：使用 Phase 2 结果
        >>> path, kwargs = plot_dca(
        ...     phase2_result_path='test_results/phase2_result.json',
        ...     save_path='output/figures/fig4b_dca.pdf'
        ... )
        
        >>> # 传统方法：重新训练模型（向后兼容）
        >>> path, kwargs = plot_dca(
        ...     data_path='data/cleaned.csv',
        ...     target_column='Group',
        ...     features=['feature1', 'feature2'],
        ...     champion_model_family='XGBoost'
        ... )
    """
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'legend_loc': 'upper right',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11,
        'bbox_to_anchor': None,
        # 医学 DCA 常用阈值区间，避免高阈值段导致 Treat-All 数值爆炸
        'threshold_min': 0.01,
        'threshold_max': 0.99,
        'threshold_step': 0.01,
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_dca using layout parameters: {final_layout_kwargs}")
    
    # ========================================================================
    # 🎯 NEW: 优先读取 Phase 2 的 cv_predictions
    # ========================================================================
    def detect_positive_class(label_names, y_true_encoded):
        """
        智能检测正类索引（与 final ROC 保持一致策略）
        """
        if not label_names or len(label_names) != 2:
            class_counts = np.bincount(y_true_encoded)
            if len(class_counts) == 2:
                minority_class = int(np.argmin(class_counts))
                return minority_class, str(minority_class)
            return 1, '1'

        disease_keywords = [
            'disease', 'case', 'patient', 'cd', 'cancer',
            'tumor', 'positive', 'sick', 'ill', 'affected'
        ]
        for idx, label in enumerate(label_names):
            if any(k in str(label).lower() for k in disease_keywords):
                print(f"  🎯 Detected positive class by keyword: '{label}' (class {idx})")
                return idx, label

        class_counts = np.bincount(y_true_encoded)
        if len(class_counts) == 2:
            minority_class = int(np.argmin(class_counts))
            ratio = class_counts.max() / (class_counts.min() + 1e-10)
            if ratio >= 2.0:
                print(
                    f"  🎯 Detected positive class by minority: '{label_names[minority_class]}' "
                    f"(class {minority_class}, ratio={ratio:.1f}:1)"
                )
                return minority_class, label_names[minority_class]

        print(f"  ⚠️  Cannot determine positive class, using default: class 1 ('{label_names[1]}')")
        return 1, label_names[1]

    from sklearn.metrics import roc_auc_score

    use_phase2_predictions = False
    y_encoded = None
    y_pred_proba_cv = None
    positive_class_idx = 1
    positive_class_name = '1'
    n_features = 0
    
    if phase2_result_path and os.path.exists(phase2_result_path):
        print("\n" + "="*80)
        print("Phase 3: Decision Curve Analysis (Using Phase 2 CV Predictions)")
        print("="*80)
        print(f"Reading Phase 2 result from: {phase2_result_path}")
        
        try:
            with open(phase2_result_path, 'r', encoding='utf-8') as f:
                phase2_result = json.load(f)
            
            cv_predictions = (
                phase2_result.get('final_result', {}).get('cv_predictions')
                or phase2_result.get('cv_predictions')
            )
            
            if cv_predictions and cv_predictions.get('predictions'):
                print(f"✅ Found cv_predictions in Phase 2 result")
                print(f"   Samples: {cv_predictions.get('n_samples', 'N/A')}")
                print(f"   K-folds: {cv_predictions.get('k_folds', 'N/A')}")
                print(f"   Metric: {cv_predictions.get('metric', 'N/A')}")
                
                # 转换为 DataFrame
                predictions_df = pd.DataFrame(cv_predictions['predictions'])
                
                # 提取真实标签和预测概率
                y_encoded = predictions_df['true_label_encoded'].values
                
                # 智能检测正类，并选择对应概率列
                label_names = cv_predictions.get('label_names', ['CD', 'Control'])
                positive_class_idx, positive_class_name = detect_positive_class(label_names, y_encoded)
                if positive_class_idx == 0:
                    y_pred_proba_cv = predictions_df['pred_proba_class0'].values
                    print(f"  ✅ Using pred_proba_class0 for positive class '{positive_class_name}'")
                else:
                    y_pred_proba_cv = predictions_df['pred_proba_class1'].values
                    print(f"  ✅ Using pred_proba_class1 for positive class '{positive_class_name}'")

                # 标签反转体检：若 AUC 显著低于 0.5，尝试切换另一列概率
                y_true_positive = (y_encoded == positive_class_idx).astype(int)
                auc_selected = roc_auc_score(y_true_positive, y_pred_proba_cv)
                alt_idx = 1 - positive_class_idx
                alt_col = 'pred_proba_class0' if alt_idx == 0 else 'pred_proba_class1'
                y_pred_alt = predictions_df[alt_col].values
                auc_alt = roc_auc_score(y_true_positive, y_pred_alt)
                if auc_selected < 0.5 and auc_alt > auc_selected + 0.05:
                    print(
                        f"  ⚠️ Potential label-probability mismatch detected: "
                        f"AUC(selected)={auc_selected:.4f}, AUC(alt)={auc_alt:.4f}. "
                        f"Switching to {alt_col}."
                    )
                    positive_class_idx = alt_idx
                    positive_class_name = str(label_names[alt_idx]) if label_names else str(alt_idx)
                    y_pred_proba_cv = y_pred_alt
                
                # 获取特征数量，兼容旧格式 final_result.features 和 canonical selected_features
                n_features = len(
                    phase2_result.get('final_result', {}).get('features', [])
                    or phase2_result.get('selected_features', [])
                )
                
                print(f"\n✅ Using Phase 2 predictions (100% numerical consistency)")
                print(f"   Samples: {len(y_encoded)}")
                print(f"   Features: {n_features}")
                print(f"   Prediction probability range: [{y_pred_proba_cv.min():.4f}, {y_pred_proba_cv.max():.4f}]")
                
                use_phase2_predictions = True
            else:
                print(f"⚠️  cv_predictions not found in Phase 2 result")
                print(f"   Falling back to traditional method (re-training model)")
        
        except Exception as e:
            print(f"⚠️  Error reading Phase 2 result: {e}")
            print(f"   Falling back to traditional method (re-training model)")
    
    # ========================================================================
    # 传统方法：重新训练模型（向后兼容）
    # ========================================================================
    if not use_phase2_predictions:
        print("\n" + "="*80)
        print("Phase 3: Decision Curve Analysis (DCA) - Cross-Validated")
        print("="*80)
        print(f"Loading data from: {data_path}")
        
        if not data_path or not target_column or not features or not champion_model_family:
            raise ValueError(
                "When phase2_result_path is not provided, "
                "data_path, target_column, features, and champion_model_family are required"
            )
        
        df = pd.read_csv(data_path)
        
        # 提取特征和目标
        X = df[features].values
        y = df[target_column].values
        
        # 编码目标变量
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        positive_class_idx, positive_class_name = detect_positive_class(list(le.classes_), y_encoded)
        
        n_features = len(features)
        
        print(f"Data shape: {X.shape}")
        print(f"Features: {n_features}")
        print(f"Classes: {le.classes_}")
        
        # 选择模型
        if 'XGBoost' in champion_model_family or 'XGB' in champion_model_family:
            model = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        print(f"Model: {model.__class__.__name__}")
        
        # ====================================================================
        # 2. 使用交叉验证获取袋外预测概率（避免过拟合）
        # ====================================================================
        print(f"\n[Step 1] Obtaining out-of-fold predictions via 5-fold stratified cross-validation...")
        print("  ⚠️  Using cross_val_predict to avoid overfitting (fit and predict on same data)")
        print("  ✅ This ensures DCA curve reflects true generalization performance")
        
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        
        # 使用 StratifiedKFold 确保每折中类别分布一致
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 使用 cross_val_predict 获取全量样本的袋外预测概率
        # cv: 5 折分层交叉验证
        # method='predict_proba': 返回预测概率
        # 每个样本的预测概率来自它在测试集中的那一折，避免了数据泄露
        y_pred_proba_all = cross_val_predict(
            model, 
            X, 
            y_encoded, 
            cv=cv,  # 使用 StratifiedKFold
            method='predict_proba'
        )
        y_pred_proba_cv = y_pred_proba_all[:, positive_class_idx]

        y_true_positive = (y_encoded == positive_class_idx).astype(int)
        auc_selected = roc_auc_score(y_true_positive, y_pred_proba_cv)
        alt_idx = 1 - positive_class_idx
        y_pred_alt = y_pred_proba_all[:, alt_idx]
        auc_alt = roc_auc_score(y_true_positive, y_pred_alt)
        if auc_selected < 0.5 and auc_alt > auc_selected + 0.05:
            print(
                f"  ⚠️ Potential label-probability mismatch detected: "
                f"AUC(selected)={auc_selected:.4f}, AUC(alt)={auc_alt:.4f}. "
                f"Switching to class {alt_idx} probability."
            )
            positive_class_idx = alt_idx
            positive_class_name = str(le.classes_[alt_idx]) if len(le.classes_) > alt_idx else str(alt_idx)
            y_pred_proba_cv = y_pred_alt
        
        print(f"  Cross-validated predictions obtained for {len(y_pred_proba_cv)} samples")
        print(f"  Prediction probability range: [{y_pred_proba_cv.min():.4f}, {y_pred_proba_cv.max():.4f}]")
    
    # ========================================================================
    # 3. 计算 DCA 曲线（统一路径）
    # ========================================================================
    print(f"\n[Step 2] Calculating net benefit across threshold probabilities...")
    
    # ========================================================================
    # 3. 计算 DCA 曲线
    # ========================================================================
    # 设定阈值概率区间（开区间），默认聚焦临床可解释区间
    threshold_min = float(final_layout_kwargs.get('threshold_min', 0.01))
    threshold_max = float(final_layout_kwargs.get('threshold_max', 0.99))
    threshold_step = float(final_layout_kwargs.get('threshold_step', 0.01))
    if not (0 < threshold_min < threshold_max < 1):
        threshold_min, threshold_max = 0.01, 0.99
    if threshold_step <= 0:
        threshold_step = 0.01
    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    
    # 样本总数
    N = len(y_encoded)
    y_true_positive = (y_encoded == positive_class_idx).astype(int)
    
    # 存储净收益
    net_benefits_model = []
    net_benefits_treat_all = []
    net_benefits_treat_none = []
    
    for pt in thresholds:
        # ====================================================================
        # Model Net Benefit（使用交叉验证的预测概率）
        # ====================================================================
        # 根据阈值进行预测
        y_pred = (y_pred_proba_cv >= pt).astype(int)
        
        # 计算 TP 和 FP
        TP = np.sum((y_pred == 1) & (y_true_positive == 1))
        FP = np.sum((y_pred == 1) & (y_true_positive == 0))
        
        # 计算净收益: NB = TP/N - FP/N × Pt/(1-Pt)
        nb_model = (TP / N) - (FP / N) * (pt / (1 - pt))
        
        net_benefits_model.append(nb_model)
        
        # ====================================================================
        # Treat All Net Benefit
        # ====================================================================
        # 假设所有人都治疗
        TP_all = np.sum(y_true_positive == 1)
        FP_all = np.sum(y_true_positive == 0)
        
        nb_treat_all = (TP_all / N) - (FP_all / N) * (pt / (1 - pt))
        
        net_benefits_treat_all.append(nb_treat_all)
        
        # ====================================================================
        # Treat None Net Benefit
        # ====================================================================
        # 假设所有人都不治疗，净收益始终为 0
        net_benefits_treat_none.append(0.0)
    
    print(f"  Calculated net benefit for {len(thresholds)} threshold points")
    print(f"  Threshold range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    print(f"  Positive class: {positive_class_name} (idx={positive_class_idx})")
    print(f"  Prediction std: {float(np.std(y_pred_proba_cv)):.6f}")

    dca_summary = summarize_decision_curve_ranges(
        y_true=y_true_positive,
        y_score=y_pred_proba_cv,
        thresholds=thresholds.tolist(),
        positive_label=1,
    )
    clinical_utility_payload = _load_phase2_clinical_utility_artifact(phase2_result_path)
    preferred_relative_summary = clinical_utility_payload.get('decision_curve_relative') or {}
    preferred_adjusted_summary = clinical_utility_payload.get('decision_curve_adjusted') or {}
    prevalence_context = clinical_utility_payload.get('prevalence_context') or {}
    using_relative_summary = isinstance(preferred_relative_summary, dict) and bool(preferred_relative_summary)
    using_adjusted_summary = (
        isinstance(preferred_adjusted_summary, dict)
        and bool(preferred_adjusted_summary)
        and bool(prevalence_context.get('prevalence_transport_applied', False))
    )
    if using_relative_summary:
        dca_summary = dict(preferred_relative_summary)
    dca_summary.update(
        {
            'positive_class_name': str(positive_class_name),
            'positive_class_idx': int(positive_class_idx),
            'n_samples': int(N),
            'prevalence': float(np.mean(y_true_positive)),
            'use_phase2_predictions': bool(use_phase2_predictions),
            'prediction_probability_min': float(np.min(y_pred_proba_cv)),
            'prediction_probability_max': float(np.max(y_pred_proba_cv)),
            'prevalence_transport_applied': bool(prevalence_context.get('prevalence_transport_applied', False)),
            'target_prevalence': prevalence_context.get('target_prevalence'),
        }
    )
    print(f"  Peak net benefit threshold: {dca_summary.get('peak_net_benefit_threshold')}")
    print(
        "  Winner net-benefit ranges "
        f"vs treat-all: {dca_summary.get('winner_better_than_treat_all_ranges', [])}; "
        f"vs treat-none: {dca_summary.get('winner_better_than_treat_none_ranges', [])}"
    )
    
    # ========================================================================
    # 3. 绘制 DCA 曲线
    # ========================================================================
    print(f"\n[Step 3] Plotting DCA curve...")
    
    # 设置学术风格
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 3.1 绘制 "Treat None" 线（灰色水平线）
    ax.plot(
        thresholds,
        net_benefits_treat_none,
        'k-',
        linewidth=2,
        alpha=0.5,
        label='Treat None'
    )
    
    # 3.2 绘制 "Treat All" 线（灰色虚线）
    ax.plot(
        thresholds,
        net_benefits_treat_all,
        'k--',
        linewidth=2,
        alpha=0.5,
        label='Treat All'
    )
    
    # 3.3 绘制 Model 线（深红色实线）
    ax.plot(
        thresholds,
        net_benefits_model,
        color='darkred',
        linewidth=3,
        label=f'Winner Panel ({n_features} Features)'
    )
    
    # ========================================================================
    # 4. 图表装饰
    # ========================================================================
    # 4.1 坐标轴标签
    ax.set_xlabel('Threshold Probability (Pt)', 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    ax.set_ylabel('Net Benefit', 
                   fontsize=final_layout_kwargs['label_fontsize'], 
                   fontweight='bold')
    
    # 4.2 标题
    title_text = 'Decision Curve Analysis (DCA)\n'
    if using_adjusted_summary:
        title_text += 'Prevalence-Adjusted Clinical Net Benefit'
    else:
        title_text += 'Relative Utility Under Case-Control Sampling'
    
    # 如果使用 Phase 2 预测，添加说明
    if use_phase2_predictions:
        title_text += '\n(Using Phase 2 Predictions for 100% Consistency)'
    
    ax.set_title(
        title_text,
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        pad=20
    )
    ax.text(
        0.02,
        0.02,
        (
            f"Positive class: {positive_class_name} (idx={positive_class_idx})"
            + (
                f"\nTarget prevalence: {float(prevalence_context.get('target_prevalence')):.4f}"
                if prevalence_context.get('prevalence_transport_applied', False)
                and prevalence_context.get('target_prevalence') is not None
                else "\nRelative DCA is primary for case-control evaluation"
            )
        ),
        transform=ax.transAxes,
        fontsize=max(final_layout_kwargs['fontsize'] - 1, 8),
        alpha=0.7,
        va='bottom',
        ha='left'
    )
    
    # 4.3 图例
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        bbox_to_anchor=final_layout_kwargs['bbox_to_anchor']
    )
    
    # 4.4 设置坐标轴范围
    ax.set_xlim([0, 1])

    # 使用患病率设定 Y 轴范围，避免高阈值段 Treat-All 下坠拉坏画布
    prevalence = float(np.mean(y_true_positive))
    ax.set_ylim([-0.05, prevalence + 0.05])
    
    # 4.5 去除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4.6 网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 4.7 添加参考线（Y=0）
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # ========================================================================
    # 5. 保存图片
    # ========================================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ DCA curve saved to: {save_path}")
    print(f"[ACVAL Actor] Final layout kwargs: {final_layout_kwargs}")
    print("="*80 + "\n")

    return save_path, {
        **final_layout_kwargs,
        'dca_summary': dca_summary,
    }


def plot_calibration(
    data_path: Optional[str] = None,
    target_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    champion_model_family: Optional[str] = None,
    phase2_result_path: Optional[str] = None,
    save_path: str = "output/figures/fig4e_calibration.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """
    绘制 Winner Panel 的概率校准图（reliability / calibration plot）。

    优先使用 Phase 2 的 OOF 预测概率；如果不可用，则回退到基于交叉验证的
    `predict_proba` 结果重新评估，以保持与 DCA 的调用方式一致。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    def detect_positive_class(label_names, y_true_encoded):
        if not label_names or len(label_names) != 2:
            class_counts = np.bincount(y_true_encoded)
            if len(class_counts) == 2:
                minority_class = int(np.argmin(class_counts))
                return minority_class, str(minority_class)
            return 1, '1'

        disease_keywords = [
            'disease', 'case', 'patient', 'cd', 'cancer',
            'tumor', 'positive', 'sick', 'ill', 'affected'
        ]
        for idx, label in enumerate(label_names):
            if any(k in str(label).lower() for k in disease_keywords):
                return idx, label
        class_counts = np.bincount(y_true_encoded)
        if len(class_counts) == 2:
            minority_class = int(np.argmin(class_counts))
            return minority_class, label_names[minority_class]
        return 1, label_names[1] if len(label_names) > 1 else '1'

    default_layout_kwargs = {
        'legend_loc': 'upper left',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11,
        'n_bins': 10,
        'binning_strategy': 'quantile',
    }
    if layout_kwargs is None:
        layout_kwargs = {}
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}

    print(f"\n[ACVAL Actor] plot_calibration using layout parameters: {final_layout_kwargs}")

    use_phase2_predictions = False
    y_true_positive = None
    y_pred_proba_cv = None
    positive_class_idx = 1
    positive_class_name = '1'
    n_features = 0

    if phase2_result_path and os.path.exists(phase2_result_path):
        print("\n" + "=" * 80)
        print("Phase 3: Calibration Plot (Using Phase 2 CV Predictions)")
        print("=" * 80)
        print(f"Reading Phase 2 result from: {phase2_result_path}")
        try:
            with open(phase2_result_path, 'r', encoding='utf-8') as f:
                phase2_result = json.load(f)
            cv_predictions = (
                phase2_result.get('final_result', {}).get('cv_predictions')
                or phase2_result.get('cv_predictions')
            )
            if cv_predictions and cv_predictions.get('predictions'):
                predictions_df = pd.DataFrame(cv_predictions['predictions'])
                y_encoded = predictions_df['true_label_encoded'].values
                label_names = cv_predictions.get('label_names', ['CD', 'Control'])
                positive_class_idx, positive_class_name = detect_positive_class(list(label_names), y_encoded)
                if positive_class_idx == 0:
                    y_pred_proba_cv = predictions_df['pred_proba_class0'].values
                else:
                    y_pred_proba_cv = predictions_df['pred_proba_class1'].values
                y_true_positive = (y_encoded == positive_class_idx).astype(int)
                n_features = len(
                    phase2_result.get('final_result', {}).get('features', [])
                    or phase2_result.get('selected_features', [])
                )
                use_phase2_predictions = True
        except Exception as exc:
            print(f"⚠️  Error reading Phase 2 result for calibration plot: {exc}")

    if not use_phase2_predictions:
        print("\n" + "=" * 80)
        print("Phase 3: Calibration Plot (Cross-Validated Fallback)")
        print("=" * 80)
        if not data_path or not target_column or not features or not champion_model_family:
            raise ValueError(
                "When phase2_result_path is not provided, "
                "data_path, target_column, features, and champion_model_family are required"
            )

        df = pd.read_csv(data_path)
        X = df[features].values
        y = df[target_column].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        positive_class_idx, positive_class_name = detect_positive_class(list(le.classes_), y_encoded)
        n_features = len(features)
        if 'XGBoost' in champion_model_family or 'XGB' in champion_model_family:
            model = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_proba_all = cross_val_predict(
            model,
            X,
            y_encoded,
            cv=cv,
            method='predict_proba',
        )
        y_pred_proba_cv = y_pred_proba_all[:, positive_class_idx]
        y_true_positive = (y_encoded == positive_class_idx).astype(int)

    calibration_summary = build_probability_calibration_summary(
        y_true_positive,
        y_pred_proba_cv,
        n_bins=int(final_layout_kwargs.get('n_bins', 10)),
        strategy=str(final_layout_kwargs.get('binning_strategy', 'quantile')),
    )
    clinical_utility_payload = _load_phase2_clinical_utility_artifact(phase2_result_path)
    preferred_adjusted_summary = clinical_utility_payload.get('probability_calibration_adjusted') or {}
    preferred_raw_summary = clinical_utility_payload.get('probability_calibration_raw') or {}
    prevalence_context = clinical_utility_payload.get('prevalence_context') or {}
    using_adjusted_summary = isinstance(preferred_adjusted_summary, dict) and bool(preferred_adjusted_summary.get('available'))
    if using_adjusted_summary:
        calibration_summary = dict(preferred_adjusted_summary)
    elif isinstance(preferred_raw_summary, dict) and preferred_raw_summary:
        calibration_summary = dict(preferred_raw_summary)
    print(
        "  Calibration summary: "
        f"Brier={calibration_summary.get('brier_score', 0.0):.4f}, "
        f"ICI={calibration_summary.get('integrated_calibration_index', 0.0):.4f}, "
        f"Slope={calibration_summary.get('calibration_slope', 0.0):.4f}"
    )

    curve_points = calibration_summary.get('calibration_curve', []) or []
    mean_pred = [float(row.get('mean_predicted_probability', 0.0)) for row in curve_points]
    observed = [float(row.get('observed_event_rate', 0.0)) for row in curve_points]

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Perfect Calibration')
    ax.plot(
        mean_pred,
        observed,
        color='darkblue',
        linewidth=2.5,
        marker='o',
        markersize=6,
        label=f'Winner Panel ({n_features} Features)',
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability', fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold')
    ax.set_ylabel('Observed Event Rate', fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold')
    title = 'Calibration Plot\n'
    title += (
        'Target-Population Prevalence-Adjusted Assessment'
        if using_adjusted_summary
        else 'Internal Study-Prevalence Diagnostic'
    )
    if use_phase2_predictions:
        title += '\n(Using Phase 2 OOF Predictions)'
    ax.set_title(title, fontsize=final_layout_kwargs['title_fontsize'], fontweight='bold', pad=20)
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.text(
        0.02,
        0.02,
        (
            f"Brier={calibration_summary.get('brier_score', 0.0):.3f}\n"
            f"ICI={calibration_summary.get('integrated_calibration_index', calibration_summary.get('expected_calibration_error', 0.0)):.3f}\n"
            f"Slope={calibration_summary.get('calibration_slope', 0.0):.3f}\n"
            f"Intercept={calibration_summary.get('calibration_intercept', 0.0):.3f}"
            + (
                f"\nTarget Prev={float(prevalence_context.get('target_prevalence')):.3f}"
                if using_adjusted_summary and prevalence_context.get('target_prevalence') is not None
                else "\nECE suppressed in main report"
            )
        ),
        transform=ax.transAxes,
        fontsize=max(final_layout_kwargs['fontsize'] - 1, 8),
        va='bottom',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'),
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Calibration plot saved to: {save_path}")
    return save_path, {
        **final_layout_kwargs,
        'calibration_summary': {
            **calibration_summary,
            'positive_class_name': str(positive_class_name),
            'positive_class_idx': int(positive_class_idx),
            'use_phase2_predictions': bool(use_phase2_predictions),
        },
    }


def plot_threshold_performance(
    data_path: Optional[str] = None,
    target_column: Optional[str] = None,
    features: Optional[List[str]] = None,
    champion_model_family: Optional[str] = None,
    phase2_result_path: Optional[str] = None,
    save_path: str = "output/figures/fig4f_threshold_performance.pdf",
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制多阈值 operating characteristics 曲线。

    横轴为阈值，纵轴展示 sensitivity / specificity / PPV / NPV，
    以便直观看到不同临床动作阈值下的性能权衡。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    def _safe_float_local(value):
        try:
            if value is None or value == '':
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_threshold_candidates(values):
        normalized = []
        for value in values or []:
            numeric = _safe_float_local(value)
            if numeric is not None and 0.0 < numeric < 1.0:
                normalized.append(float(numeric))
        return sorted(set(normalized))

    def _resolve_threshold_configuration(phase2_result, phase2_result_path_local):
        scenario_thresholds = []
        resolved_default_threshold = None
        resolved_companion_threshold = None
        scenario_name = None

        candidate_paths = []
        if phase2_result_path_local:
            artifact_dir = os.path.dirname(phase2_result_path_local)
            candidate_paths.append(os.path.join(artifact_dir, 'phase2_clinical_utility.json'))

        final_result_payload = phase2_result.get('final_result', {}) if isinstance(phase2_result, dict) else {}
        if isinstance(final_result_payload, dict):
            embedded_clinical_utility = final_result_payload.get('clinical_utility') or phase2_result.get('clinical_utility')
            if isinstance(embedded_clinical_utility, dict):
                scenario = embedded_clinical_utility.get('scenario', {})
                scenario_thresholds = _normalize_threshold_candidates(scenario.get('risk_thresholds'))
                resolved_default_threshold = _safe_float_local(scenario.get('default_action_threshold'))
                companion_summary = embedded_clinical_utility.get('data_driven_companion_threshold_summary', {})
                if isinstance(companion_summary, dict):
                    resolved_companion_threshold = _safe_float_local(companion_summary.get('selected_threshold'))
                scenario_name = scenario.get('key') or scenario.get('name')

        for clinical_utility_path in candidate_paths:
            if not os.path.exists(clinical_utility_path):
                continue
            try:
                with open(clinical_utility_path, 'r', encoding='utf-8') as f:
                    clinical_utility_payload = json.load(f)
                scenario = clinical_utility_payload.get('scenario', {})
                scenario_thresholds = _normalize_threshold_candidates(
                    scenario.get('risk_thresholds', scenario_thresholds)
                ) or scenario_thresholds
                resolved_default_threshold = _safe_float_local(
                    scenario.get('default_action_threshold', resolved_default_threshold)
                ) or resolved_default_threshold
                companion_summary = clinical_utility_payload.get('data_driven_companion_threshold_summary', {})
                if isinstance(companion_summary, dict):
                    resolved_companion_threshold = _safe_float_local(
                        companion_summary.get('selected_threshold', resolved_companion_threshold)
                    ) or resolved_companion_threshold
                scenario_name = scenario.get('key') or scenario.get('name') or scenario_name
                break
            except Exception as exc:
                print(f"⚠️  Error reading clinical utility artifact for threshold plot: {exc}")

        if not scenario_thresholds or resolved_default_threshold is None:
            try:
                config = get_config()
                config_scenario_name = str(
                    phase2_result.get('clinical_scenario')
                    or phase2_result.get('final_result', {}).get('clinical_scenario')
                    or scenario_name
                    or config.get_active_scenario()
                ).strip()
                scenario_definition = config.get_scenario_definition(config_scenario_name)
                scenario_thresholds = scenario_thresholds or _normalize_threshold_candidates(
                    scenario_definition.get('risk_thresholds')
                )
                if resolved_default_threshold is None:
                    resolved_default_threshold = _safe_float_local(
                        scenario_definition.get('default_action_threshold')
                    )
                scenario_name = scenario_definition.get('key') or scenario_definition.get('name') or scenario_name
            except Exception as exc:
                print(f"⚠️  Error loading scenario thresholds from config: {exc}")

        return {
            'scenario_thresholds': scenario_thresholds,
            'default_threshold': resolved_default_threshold,
            'companion_threshold': resolved_companion_threshold,
            'scenario_name': scenario_name,
        }

    def detect_positive_class(label_names, y_true_encoded):
        if not label_names or len(label_names) != 2:
            class_counts = np.bincount(y_true_encoded)
            if len(class_counts) == 2:
                minority_class = int(np.argmin(class_counts))
                return minority_class, str(minority_class)
            return 1, '1'
        disease_keywords = [
            'disease', 'case', 'patient', 'cd', 'cancer',
            'tumor', 'positive', 'sick', 'ill', 'affected'
        ]
        for idx, label in enumerate(label_names):
            if any(k in str(label).lower() for k in disease_keywords):
                return idx, label
        class_counts = np.bincount(y_true_encoded)
        if len(class_counts) == 2:
            minority_class = int(np.argmin(class_counts))
            return minority_class, label_names[minority_class]
        return 1, label_names[1] if len(label_names) > 1 else '1'

    default_layout_kwargs = {
        'legend_loc': 'lower left',
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14,
        'legend_fontsize': 11,
        'threshold_min': 0.05,
        'threshold_max': 0.60,
        'threshold_step': 0.01,
        'highlight_default_threshold': True,
    }
    if layout_kwargs is None:
        layout_kwargs = {}
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    theme = journal_theme or {}
    palette = theme.get('palette', []) if isinstance(theme, dict) else []
    sensitivity_color = resolve_theme_color(theme, 'winner', palette[0] if palette else '#8A4F4A')
    specificity_color = resolve_theme_color(theme, 'positive', palette[1] if len(palette) > 1 else '#5F8F8A')
    ppv_color = resolve_theme_color(theme, 'cost', palette[5] if len(palette) > 5 else '#9B7A4F')
    npv_color = resolve_theme_color(theme, 'redundancy', palette[3] if len(palette) > 3 else '#8D8396')
    default_threshold_color = resolve_theme_color(theme, 'reference', palette[-1] if palette else '#A7A19A')
    companion_threshold_color = resolve_theme_color(theme, 'clinical', palette[0] if palette else '#8A4F4A')
    axis_text_color = theme.get('axis_text_color', '#374151') if isinstance(theme, dict) else '#374151'
    grid_color = theme.get('grid_color', '#E9EDF2') if isinstance(theme, dict) else '#E9EDF2'
    plt.rcParams['font.family'] = theme.get('font_family', 'Arial') if isinstance(theme, dict) else 'Arial'
    print(f"\n[ACVAL Actor] plot_threshold_performance using layout parameters: {final_layout_kwargs}")

    use_phase2_predictions = False
    y_true_positive = None
    y_pred_proba_cv = None
    positive_class_idx = 1
    positive_class_name = '1'
    n_features = 0
    default_threshold = None
    companion_threshold = None
    scenario_thresholds = []
    scenario_name = None

    if phase2_result_path and os.path.exists(phase2_result_path):
        print("\n" + "=" * 80)
        print("Phase 3: Threshold Performance Plot (Using Phase 2 CV Predictions)")
        print("=" * 80)
        print(f"Reading Phase 2 result from: {phase2_result_path}")
        try:
            with open(phase2_result_path, 'r', encoding='utf-8') as f:
                phase2_result = json.load(f)

            canonical_df, canonical_path = _load_canonical_audit_oof_predictions(phase2_result_path)
            if canonical_df is not None:
                y_true_positive = pd.to_numeric(canonical_df['y_true'], errors='raise').astype(int).to_numpy()
                y_pred_proba_cv = pd.to_numeric(canonical_df['pred_probability'], errors='raise').astype(float).to_numpy()
                positive_class_idx, positive_class_name = 1, '1'
                print(f"Using canonical audit OOF predictions: {canonical_path}")
            else:
                cv_predictions = (
                    phase2_result.get('final_result', {}).get('cv_predictions')
                    or phase2_result.get('cv_predictions')
                )
                if cv_predictions and cv_predictions.get('predictions'):
                    predictions_df = pd.DataFrame(cv_predictions['predictions'])
                    y_encoded = predictions_df['true_label_encoded'].values
                    label_names = cv_predictions.get('label_names', ['CD', 'Control'])
                    positive_class_idx, positive_class_name = detect_positive_class(list(label_names), y_encoded)
                    if positive_class_idx == 0:
                        y_pred_proba_cv = predictions_df['pred_proba_class0'].values
                    else:
                        y_pred_proba_cv = predictions_df['pred_proba_class1'].values
                    y_true_positive = (y_encoded == positive_class_idx).astype(int)
            if y_true_positive is not None and y_pred_proba_cv is not None:
                n_features = len(
                    phase2_result.get('final_result', {}).get('features', [])
                    or phase2_result.get('selected_features', [])
                )
                use_phase2_predictions = True

            threshold_config = _resolve_threshold_configuration(phase2_result, phase2_result_path)
            default_threshold = threshold_config.get('default_threshold')
            companion_threshold = threshold_config.get('companion_threshold')
            scenario_thresholds = list(threshold_config.get('scenario_thresholds', []))
            scenario_name = threshold_config.get('scenario_name')
        except Exception as exc:
            print(f"⚠️  Error reading Phase 2 result for threshold-performance plot: {exc}")

    if not use_phase2_predictions:
        print("\n" + "=" * 80)
        print("Phase 3: Threshold Performance Plot (Cross-Validated Fallback)")
        print("=" * 80)
        if not data_path or not target_column or not features or not champion_model_family:
            raise ValueError(
                "When phase2_result_path is not provided, "
                "data_path, target_column, features, and champion_model_family are required"
            )

        df = pd.read_csv(data_path)
        X = df[features].values
        y = df[target_column].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        positive_class_idx, positive_class_name = detect_positive_class(list(le.classes_), y_encoded)
        n_features = len(features)
        if 'XGBoost' in champion_model_family or 'XGB' in champion_model_family:
            model = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_proba_all = cross_val_predict(
            model,
            X,
            y_encoded,
            cv=cv,
            method='predict_proba',
        )
        y_pred_proba_cv = y_pred_proba_all[:, positive_class_idx]
        y_true_positive = (y_encoded == positive_class_idx).astype(int)

    threshold_min = float(final_layout_kwargs.get('threshold_min', 0.05))
    threshold_max = float(final_layout_kwargs.get('threshold_max', 0.60))
    threshold_step = float(final_layout_kwargs.get('threshold_step', 0.01))
    if not (0 < threshold_min < threshold_max < 1):
        threshold_min, threshold_max = 0.05, 0.60
    if threshold_step <= 0:
        threshold_step = 0.01
    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    scenario_thresholds = [
        threshold for threshold in _normalize_threshold_candidates(scenario_thresholds)
        if threshold_min <= threshold <= threshold_max
    ]
    if default_threshold is not None and threshold_min <= float(default_threshold) <= threshold_max:
        scenario_thresholds = sorted(set(scenario_thresholds + [float(default_threshold)]))
    if companion_threshold is not None and threshold_min <= float(companion_threshold) <= threshold_max:
        scenario_thresholds = sorted(set(scenario_thresholds + [float(companion_threshold)]))
    if not scenario_thresholds:
        fallback_candidates = [0.10, 0.15, 0.20, 0.30]
        scenario_thresholds = [
            threshold for threshold in fallback_candidates
            if threshold_min <= threshold <= threshold_max
        ]

    metrics_table = build_threshold_metrics_table(
        y_true_positive,
        y_pred_proba_cv,
        thresholds.tolist(),
        positive_label=1,
    )

    sensitivity = [float(row.get('sensitivity', 0.0)) for row in metrics_table]
    specificity = [float(row.get('specificity', 0.0)) for row in metrics_table]
    ppv = [float(row.get('ppv', 0.0)) for row in metrics_table]
    npv = [float(row.get('npv', 0.0)) for row in metrics_table]
    metrics_by_threshold = {
        round(float(row.get('threshold', 0.0)), 6): row
        for row in metrics_table
        if isinstance(row, dict)
    }

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = theme.get('font_family', 'Arial') if isinstance(theme, dict) else 'Arial'
    plt.rcParams['font.size'] = final_layout_kwargs['fontsize']

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(thresholds, sensitivity, label='Sensitivity', linewidth=2.2, color=sensitivity_color)
    ax.plot(thresholds, specificity, label='Specificity', linewidth=2.2, color=specificity_color)
    ax.plot(thresholds, ppv, label='PPV', linewidth=2.2, color=ppv_color)
    ax.plot(thresholds, npv, label='NPV', linewidth=2.2, color=npv_color)

    if final_layout_kwargs.get('highlight_default_threshold', True) and default_threshold is not None:
        ax.axvline(float(default_threshold), linestyle='--', linewidth=1.5, color=default_threshold_color, alpha=0.85)
        ax.text(
            float(default_threshold) + 0.005,
            0.04,
            f'Default threshold={float(default_threshold):.2f}',
            rotation=90,
            va='bottom',
            ha='left',
            fontsize=max(final_layout_kwargs['fontsize'] - 1, 8),
            color=default_threshold_color,
        )
    if companion_threshold is not None:
        ax.axvline(float(companion_threshold), linestyle='-.', linewidth=1.8, color=companion_threshold_color, alpha=0.9)
        ax.text(
            float(companion_threshold) + 0.005,
            0.50,
            f'Companion threshold={float(companion_threshold):.2f}',
            rotation=90,
            va='bottom',
            ha='left',
            fontsize=max(final_layout_kwargs['fontsize'] - 1, 8),
            color=companion_threshold_color,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor=grid_color),
        )

    threshold_palette = [sensitivity_color, specificity_color, ppv_color, npv_color]
    threshold_points = []
    line_specs = [
        ('Sensitivity', 'sensitivity', sensitivity_color),
        ('Specificity', 'specificity', specificity_color),
        ('PPV', 'ppv', ppv_color),
        ('NPV', 'npv', npv_color),
    ]
    for idx, threshold_value in enumerate(scenario_thresholds[:4]):
        marker_color = threshold_palette[idx % len(threshold_palette)]
        if (
            (default_threshold is not None and abs(float(threshold_value) - float(default_threshold)) < 1e-9)
            or (companion_threshold is not None and abs(float(threshold_value) - float(companion_threshold)) < 1e-9)
        ):
            ax.axvline(threshold_value, linestyle=':', linewidth=1.1, color=marker_color, alpha=0.25)
        else:
            ax.axvline(threshold_value, linestyle=':', linewidth=1.1, color=marker_color, alpha=0.45)
        row = metrics_by_threshold.get(round(float(threshold_value), 6))
        if row is None:
            continue
        for _, metric_key, line_color in line_specs:
            metric_value = row.get(metric_key)
            if metric_value is None:
                continue
            ax.scatter(
                [threshold_value],
                [float(metric_value)],
                color=line_color,
                s=22,
                zorder=5,
                edgecolors='white',
                linewidths=0.5,
            )
        threshold_points.append(
            (
                float(threshold_value),
                (
                    f'{"Default " if default_threshold is not None and abs(float(threshold_value) - float(default_threshold)) < 1e-9 else ""}'
                    f'{"Companion " if companion_threshold is not None and abs(float(threshold_value) - float(companion_threshold)) < 1e-9 else ""}'
                    f'T={float(threshold_value):.2f}\n'
                    f'Se={float(row.get("sensitivity", 0.0)):.2f}, '
                    f'Sp={float(row.get("specificity", 0.0)):.2f}\n'
                    f'PPV={float(row.get("ppv", 0.0)):.2f}, '
                    f'NPV={float(row.get("npv", 0.0)):.2f}'
                ),
            )
        )

    ax.set_xlim([thresholds.min(), thresholds.max()])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('Decision Threshold', fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold', color=axis_text_color)
    ax.set_ylabel('Operating Characteristic', fontsize=final_layout_kwargs['label_fontsize'], fontweight='bold', color=axis_text_color)
    title = 'Threshold Performance Plot\nSensitivity / Specificity / PPV / NPV Across Thresholds'
    if use_phase2_predictions:
        title += '\n(Using Phase 2 OOF Predictions)'
    ax.set_title(title, fontsize=final_layout_kwargs['title_fontsize'], fontweight='bold', pad=20)
    ax.legend(
        loc=final_layout_kwargs['legend_loc'],
        fontsize=final_layout_kwargs['legend_fontsize'],
        frameon=True,
        fancybox=True,
        shadow=True,
    )
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=grid_color)
    if threshold_points:
        annotation_box = '\n\n'.join(text for _, text in threshold_points)
        ax.text(
            0.985,
            0.02,
            annotation_box,
            transform=ax.transAxes,
            fontsize=max(final_layout_kwargs['fontsize'] - 2, 8),
            va='bottom',
            ha='right',
            color=axis_text_color,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, edgecolor=grid_color),
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    summary = {
        'n_thresholds': int(len(metrics_table)),
        'default_threshold': float(default_threshold) if default_threshold is not None else None,
        'companion_threshold': float(companion_threshold) if companion_threshold is not None else None,
        'threshold_range': [float(thresholds.min()), float(thresholds.max())],
        'use_phase2_predictions': bool(use_phase2_predictions),
        'positive_class_name': str(positive_class_name),
        'positive_class_idx': int(positive_class_idx),
        'feature_count': int(n_features),
        'scenario_name': str(scenario_name or ''),
        'scenario_thresholds': [float(value) for value in scenario_thresholds[:4]],
        'sample_threshold_metrics': metrics_table[:5],
    }
    print(f"\n✅ Threshold performance plot saved to: {save_path}")
    return save_path, {
        **final_layout_kwargs,
        'threshold_performance_summary': summary,
    }


def _compute_curve_zero_crossings(x_vals: np.ndarray, y_vals: np.ndarray, tolerance: float) -> List[float]:
    crossings: List[float] = []
    for idx in range(len(x_vals) - 1):
        x0, x1 = float(x_vals[idx]), float(x_vals[idx + 1])
        y0, y1 = float(y_vals[idx]), float(y_vals[idx + 1])
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
            continue
        if abs(y0) < 1e-8:
            x_cross = x0
        elif abs(y1) < 1e-8:
            x_cross = x1
        elif y0 * y1 < 0:
            x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
        else:
            continue
        if not crossings or abs(crossings[-1] - x_cross) > tolerance:
            crossings.append(x_cross)
    return crossings


def _export_reference_style_dependence_plots(feature_matrix: pd.DataFrame, shap_values: np.ndarray, output_dir: str, dpi: int = 300) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"schema_version": "phase3.shap_dependence_reference.v1", "generated_at": datetime.now(timezone.utc).isoformat(), "files": []}
    for idx, feature_name in enumerate(feature_matrix.columns):
        x_vals = pd.to_numeric(feature_matrix.iloc[:, idx], errors='coerce').to_numpy(dtype=float)
        y_vals = np.asarray(shap_values[:, idx], dtype=float)
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        if valid.sum() < 8:
            continue
        x_valid, y_valid = x_vals[valid], y_vals[valid]
        order = np.argsort(x_valid)
        x_sorted, y_sorted = x_valid[order], y_valid[order]
        lowess = sm.nonparametric.lowess(y_sorted, x_sorted, frac=0.3, return_sorted=True)
        curve_x, curve_y = lowess[:, 0], lowess[:, 1]
        band = pd.Series(y_sorted - curve_y).rolling(window=max(5, int(len(y_sorted) * 0.1)), min_periods=1, center=True).std().fillna(0).to_numpy()
        upper, lower = curve_y + band, curve_y - band
        x_pad = max((x_valid.max() - x_valid.min()) * 0.03, 1e-6)
        y_all = np.concatenate([y_valid, upper, lower, np.array([0.0])])
        y_pad = max((y_all.max() - y_all.min()) * 0.06, 1e-6)
        x_min, x_max = x_valid.min() - x_pad, x_valid.max() + x_pad
        y_min, y_max = y_all.min() - y_pad, y_all.max() + y_pad
        crossings = _compute_curve_zero_crossings(curve_x, curve_y, tolerance=max((x_max - x_min) * 0.01, 1e-6))
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        scatter = ax.scatter(x_valid, y_valid, c=y_valid, cmap='coolwarm', s=24, alpha=0.85, edgecolors='none', zorder=4)
        bounds = [x_min, *crossings, x_max]
        for left, right in zip(bounds[:-1], bounds[1:]):
            mid_y = curve_y[np.abs(curve_x - ((left + right) / 2.0)).argmin()]
            if mid_y >= 0:
                ax.fill_between([left, right], 0, y_max, color='#cfe8cf', alpha=0.28, zorder=1)
            else:
                ax.fill_between([left, right], y_min, 0, color='#f5d0d6', alpha=0.28, zorder=1)
        ax.fill_between(curve_x, lower, upper, color='#4c72b0', alpha=0.14, zorder=2)
        ax.plot(curve_x, curve_y, color='#2f5597', linewidth=2.2, zorder=5, label='Lowess curve')
        for x_cross in crossings:
            ax.axvline(x_cross, color='#d32f2f', linestyle='--', linewidth=1.0, alpha=0.85, zorder=3)
            ax.scatter([x_cross], [0], color='#d32f2f', s=18, zorder=6)
            ax.text(x_cross, y_max - (y_max - y_min) * 0.06, f'{x_cross:.2f}', color='#d32f2f', fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5), zorder=6)
        ax.axhline(0, color='#7a8594', linestyle='--', linewidth=0.9, alpha=0.8, zorder=3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Feature value')
        ax.set_ylabel('SHAP value')
        ax.set_title(str(feature_name))
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.legend(loc='best', frameon=False)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_ticks([float(np.nanmin(y_valid)), float(np.nanmax(y_valid))])
        cbar.set_ticklabels(['Low', 'High'])
        cbar.set_label('SHAP value', rotation=270, labelpad=14)
        feature_slug = re.sub(r"[^\w]+", "_", str(feature_name)).strip("_") or f"feature_{idx + 1}"
        out_path = os.path.join(output_dir, f'{idx + 1:02d}_{feature_slug}_dependence_reference.pdf')
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        manifest['files'].append({"feature": str(feature_name), "path": out_path, "zero_crossings": [float(v) for v in crossings]})
    manifest_path = os.path.join(output_dir, 'dependence_reference_manifest.json')
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest_path


def plot_shap(
    data_path: str,
    target_column: str,
    features: List[str],
    champion_model_family: str,
    phase2_result_path: Optional[str] = None,
    save_path: str = "output/figures/fig4c_shap.pdf",
    patient_index: int = 1,
    dpi: int = 300,
    figsize: Tuple[int, int] = (12, 8),
    layout_kwargs: Optional[Dict] = None,
    journal_theme: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict]:
    """
    绘制 SHAP 可解释性分析图：
    1) 全局解释图：SHAP Summary Plot (Bee Swarm)
    2) 单特征依赖图：final panel 中各特征的 SHAP dependence profiles
    
    这张图向审稿人展示模型的可解释性，揭示每个特征对预测结果的贡献。
    
    核心视觉元素:
    1. SHAP Summary Plot: 特征重要性排序（按平均 |SHAP| 值）
    2. SHAP Dependence Panels: 各特征的单特征依赖关系与非线性模式
    3. 颜色映射: 特征值高低用颜色区分（红色=高值，蓝色=低值）
    4. 全局重要性 + 单特征响应关系: 同时满足论文展示与机制解释
    
    ACVAL 支持:
    - layout_kwargs: 排版参数字典，支持动态调整图例位置、字体大小等
    - 返回值: (save_path, final_layout_kwargs) 用于 Critic 审查和重试
    
    CRITICAL: shap 库会强制接管 Matplotlib 画布，必须传入 show=False，
    然后使用 plt.gca() 获取当前坐标轴，才能继续使用 layout_kwargs 调整。
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名称
        features: 最终特征列表
        champion_model_family: 冠军模型家族
        save_path: 输出图片路径
        patient_index: 患者序号（从 1 开始，默认 1）
        dpi: 图片分辨率
        figsize: 图片尺寸
        layout_kwargs: 排版参数字典（可选）
    
    Returns:
        Tuple[str, Dict]: (summary 图路径, 最终使用的排版参数)
    
    References:
        Lundberg SM, Lee SI. A unified approach to interpreting model predictions. 
        Advances in neural information processing systems. 2017;30.
    
    Examples:
        >>> path, kwargs = plot_shap(
        ...     data_path='data/cleaned.csv',
        ...     target_column='Group',
        ...     features=['feature1', 'feature2'],
        ...     champion_model_family='XGBoost'
        ... )
    """
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        print("Warning: shap library not installed. Install with: pip install shap")
        SHAP_AVAILABLE = False
    
    # ========================================================================
    # 0. 初始化排版参数
    # ========================================================================
    default_layout_kwargs = {
        'fontsize': 11,
        'title_fontsize': 16,
        'label_fontsize': 14
    }
    
    if layout_kwargs is None:
        layout_kwargs = {}
    
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    
    print(f"\n[ACVAL Actor] plot_shap using layout parameters: {final_layout_kwargs}")

    # ========================================================================
    # 1. 数据加载和模型训练
    # ========================================================================
    print("\n" + "="*80)
    print("Phase 3: SHAP Interpretability Analysis")
    print("="*80)
    print(f"Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # 提取特征和目标
    X = df[features]
    y = df[target_column].values

    # 仅用于图上显示：优先使用 provenance 中的原始展示名
    display_feature_names = _resolve_shap_display_feature_names(features, phase2_result_path=phase2_result_path)
    renamed_count = sum(1 for old, new in zip(features, display_feature_names) if old != new)
    if renamed_count:
        print(f"  ✓ Renamed {renamed_count}/{len(features)} SHAP feature labels to original dataset names")
    X_display = X.copy()
    X_display.columns = display_feature_names
    
    # 编码目标变量
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {len(features)}")
    print(f"Classes: {le.classes_}")
    
    # 选择模型
    if 'XGBoost' in champion_model_family or 'XGB' in champion_model_family:
        model = XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print(f"Model: {model.__class__.__name__}")
    
    # 训练模型
    print(f"\n[Step 1] Training model on full dataset...")
    model.fit(X, y_encoded)
    
    # ========================================================================
    # 2. 计算 SHAP 值
    # ========================================================================
    if not SHAP_AVAILABLE:
        # 如果 shap 库未安装，创建一个占位图
        print(f"\n[Warning] SHAP library not available, creating placeholder plot...")
        
        # 设置学术风格
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['font.size'] = final_layout_kwargs['fontsize']
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        ax.text(
            0.5, 0.5,
            'SHAP Analysis\n\n'
            'Please install shap library:\n'
            'pip install shap',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=20,
            transform=ax.transAxes
        )
        
        ax.set_title(
            'SHAP Interpretability Analysis (Placeholder)',
            fontsize=final_layout_kwargs['title_fontsize'],
            fontweight='bold',
            pad=20
        )
        
        ax.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        shap_summary_path = str(Path(save_path).with_name(f"{Path(save_path).stem}_summary.json"))
        summary_payload = {
            'schema_version': 'phase3.shap_summary.v1',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'feature_count': len(display_feature_names),
            'patient_index_used': max(1, int(patient_index)),
            'shap_available': False,
            'ranked_features': [],
            'top_positive_risk_features': [],
            'top_negative_risk_features': [],
            'local_example': {
                'patient_index': max(1, int(patient_index)),
                'top_contributors': [],
            },
        }
        Path(shap_summary_path).write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        final_layout_kwargs['shap_summary_path'] = shap_summary_path
        final_layout_kwargs['shap_summary_available'] = False
        return save_path, final_layout_kwargs
    
    print(f"\n[Step 2] Calculating SHAP values...")
    
    # 创建 TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # 计算 SHAP 值，并统一为二分类正类的一维输出（n_samples, n_features）
    shap_values_raw = explainer.shap_values(X)

    if isinstance(shap_values_raw, list):
        # 老版本/部分模型：list[class_idx]
        shap_values = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        # 新版本常见格式：(n_samples, n_features, n_classes)
        class_idx = 1 if shap_values_raw.shape[2] > 1 else 0
        shap_values = shap_values_raw[:, :, class_idx]
    else:
        shap_values = shap_values_raw

    expected_value_raw = explainer.expected_value
    if isinstance(expected_value_raw, (list, np.ndarray)):
        expected_value_arr = np.array(expected_value_raw).reshape(-1)
        expected_value = float(expected_value_arr[1] if expected_value_arr.size > 1 else expected_value_arr[0])
    else:
        expected_value = float(expected_value_raw)

    print(f"  SHAP values shape (normalized): {shap_values.shape}")
    
    # ========================================================================
    # 3. 绘制 SHAP Summary Plot（正式版：导出资产后调用 R 风格脚本）
    # ========================================================================
    print(f"\n[Step 3] Plotting SHAP summary plot...")
    save_path_obj = Path(save_path)
    os.makedirs(save_path_obj.parent, exist_ok=True)
    shap_assets_root = save_path_obj.parent.parent / "assets" if save_path_obj.parent.name == "single_panels" else save_path_obj.parent
    shap_assets_dir = shap_assets_root / f"{save_path_obj.stem}_assets"
    class_label_map = {
        int(encoded): f"Class {original_label}"
        for encoded, original_label in enumerate(le.classes_)
    }
    shap_long_path, feature_importance_path, shap_assets_metadata_path = _write_shap_global_assets(
        output_dir=shap_assets_dir,
        X=X,
        y_encoded=y_encoded,
        feature_names=list(features),
        display_feature_names=list(display_feature_names),
        shap_values=np.asarray(shap_values, dtype=float),
        class_labels=[class_label_map[idx] for idx in sorted(class_label_map)],
    )
    _, summary_png_path, summary_svg_path = _render_shap_global_importance_r(
        shap_long_path=shap_long_path,
        feature_importance_path=feature_importance_path,
        save_path=save_path,
        journal_theme=journal_theme,
    )

    safe_patient_index = max(1, int(patient_index))
    patient_row_idx = min(safe_patient_index - 1, len(X) - 1)
    if patient_row_idx != safe_patient_index - 1:
        print(f"  ⚠️ patient_index={patient_index} 超出范围，自动使用最后一个样本（#{len(X)}）")

    predicted_probability = None
    if hasattr(model, 'predict_proba'):
        try:
            predicted_probability = float(model.predict_proba(X.iloc[[patient_row_idx]])[0, 1])
        except Exception:
            predicted_probability = None
    shap_summary_path = str(Path(save_path).with_name(f"{Path(save_path).stem}_summary.json"))
    shap_summary_payload = _build_shap_summary_payload(
        feature_names=list(X_display.columns),
        feature_matrix=X_display,
        shap_values=shap_values,
        patient_row_idx=patient_row_idx,
        expected_value=expected_value,
        predicted_probability=predicted_probability,
    )
    dependence_panel_paths = _save_shap_dependence_panels(
        feature_matrix=X_display,
        shap_values=shap_values,
        save_stem=Path(save_path).with_name(f"{Path(save_path).stem}_dependence_panels"),
        dpi=dpi,
        journal_theme=journal_theme,
    )
    if dependence_panel_paths:
        shap_summary_payload['dependence_panels'] = dependence_panel_paths
    Path(shap_summary_path).write_text(
        json.dumps(shap_summary_payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    print(f"\n✅ SHAP summary plot saved to: {save_path}")
    print(f"✅ SHAP summary JSON saved to: {shap_summary_path}")
    print(f"[ACVAL Actor] Final layout kwargs: {final_layout_kwargs}")
    print("="*80 + "\n")

    final_layout_kwargs['shap_summary_path'] = shap_summary_path
    final_layout_kwargs['shap_summary_available'] = True
    final_layout_kwargs['patient_index_used'] = patient_row_idx + 1
    final_layout_kwargs['summary_png_path'] = summary_png_path
    final_layout_kwargs['summary_svg_path'] = summary_svg_path
    final_layout_kwargs['shap_assets_dir'] = str(shap_assets_dir)
    final_layout_kwargs['shap_long_path'] = str(shap_long_path)
    final_layout_kwargs['feature_importance_path'] = str(feature_importance_path)
    final_layout_kwargs['shap_assets_metadata_path'] = str(shap_assets_metadata_path)
    final_layout_kwargs.update(dependence_panel_paths)
    return save_path, final_layout_kwargs


def plot_rcs_curves(
    data_path: str,
    target_column: str,
    features: List[str],
    save_path: str = "output/figures/fig4d_rcs_panels.pdf",
    n_knots: int = 3,
    n_grid: int = 200,
    dpi: int = 300,
    figsize: Tuple[int, int] = (14, 10),
    backend: str = "r",
    layout_kwargs: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """
    为 Winner Panel 的每个特征绘制 Restricted Cubic Spline (RCS) 风险曲线。

    输出为多面板图：每个子图对应一个特征，展示特征浓度与结局概率的非线性关系。
    """
    import warnings
    import statsmodels.api as sm
    from patsy import build_design_matrices, dmatrix
    from scipy.stats import chi2
    from sklearn.preprocessing import LabelEncoder

    default_layout_kwargs = {
        'fontsize': 10,
        'title_fontsize': 14,
        'label_fontsize': 12,
        'line_color': '#8B0000',
        'scatter_alpha': 0.18,
    }
    if layout_kwargs is None:
        layout_kwargs = {}
    final_layout_kwargs = {**default_layout_kwargs, **layout_kwargs}
    effective_n_knots = 3
    requested_backend = (backend or "r").strip().lower()

    print(f"\n[ACVAL Actor] plot_rcs_curves using layout parameters: {final_layout_kwargs}")
    print("\n" + "=" * 80)
    print("Phase 3: Restricted Cubic Spline (RCS) Curves")
    print("=" * 80)
    print(f"Loading data from: {data_path}")
    print(f"  ✓ Preferred RCS backend: {requested_backend}")
    if n_knots != effective_n_knots:
        print(f"  ⚠️ Overriding requested n_knots={n_knots} -> {effective_n_knots} for RCS stability")
    print("  ✓ Enforcing natural cubic spline with boundary knots at P10/P90 and internal knot at P50")

    df = pd.read_csv(data_path)
    missing = [col for col in features + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_column].values)
    class_counts = np.bincount(y_encoded)
    positive_class_idx = int(np.argmin(class_counts)) if len(class_counts) == 2 else 1
    y_pos = (y_encoded == positive_class_idx).astype(int)
    prevalence = float(y_pos.mean())

    display_feature_names = _to_display_feature_names(features)
    renamed_count = sum(1 for old, new in zip(features, display_feature_names) if old != new)
    if renamed_count:
        print(f"  ✓ Renamed {renamed_count}/{len(features)} RCS feature labels to original dataset names")

    if requested_backend == "r":
        try:
            return _run_r_rcs_plotter(
                data_path=data_path,
                target_column=target_column,
                features=features,
                display_feature_names=display_feature_names,
                save_path=save_path,
                n_grid=n_grid,
                dpi=dpi,
                figsize=figsize,
                final_layout_kwargs=final_layout_kwargs,
            )
        except Exception as exc:
            print(f"  ⚠️ R backend failed, falling back to Python backend: {exc}")
    elif requested_backend != "python":
        raise ValueError(f"Unsupported RCS backend: {backend}")

    print("  ℹ️ Using Python fallback backend for RCS plotting")

    n_feat = len(features)
    n_cols = 2 if n_feat > 1 else 1
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
    axes_flat = axes.flatten()

    summary_rows = []
    def _format_p_value(p_value: float) -> str:
        if p_value < 0.001:
            return "P < 0.001"
        if np.isclose(p_value, 1.0):
            return "P = 1.000"
        return f"P = {p_value:.3f}"

    for idx, (feature, display_name) in enumerate(zip(features, display_feature_names)):
        ax = axes_flat[idx]
        x_raw = pd.to_numeric(df[feature], errors='coerce').values
        mask = np.isfinite(x_raw) & np.isfinite(y_pos)
        x = x_raw[mask]
        y = y_pos[mask]

        if len(np.unique(x)) < 4:
            ax.text(0.5, 0.5, f"{display_name}\nInsufficient unique values",
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        # 使用 5th~95th percentile 限制核心作图区间，避免尾部稀疏区引起喇叭口效应
        lo = float(np.nanquantile(x, 0.05))
        hi = float(np.nanquantile(x, 0.95))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            ax.text(0.5, 0.5, f"{display_name}\nInsufficient spread for core-range prediction",
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        core_mask = (x >= lo) & (x <= hi)
        x_core = x[core_mask]
        y_core = y[core_mask]
        x2d = x.reshape(-1, 1)

        try:
            knot_positions = np.quantile(x, [0.10, 0.50, 0.90]).astype(float)
            if np.any(~np.isfinite(knot_positions)) or np.any(np.diff(knot_positions) <= 0):
                unique_x = np.unique(x)
                fallback_indices = [
                    max(0, int(np.floor(0.10 * (len(unique_x) - 1)))),
                    int(round(0.50 * (len(unique_x) - 1))),
                    min(len(unique_x) - 1, int(np.ceil(0.90 * (len(unique_x) - 1)))),
                ]
                knot_positions = unique_x[fallback_indices].astype(float)
            if np.any(~np.isfinite(knot_positions)) or np.any(np.diff(knot_positions) <= 0):
                ax.text(0.5, 0.5, f"{display_name}\nUnable to place 3 stable RCS knots",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            boundary_lo = float(knot_positions[0])
            internal_knot = float(knot_positions[1])
            boundary_hi = float(knot_positions[2])
            if not (boundary_lo < internal_knot < boundary_hi):
                ax.text(0.5, 0.5, f"{display_name}\nUnable to place ordered boundary/internal knots",
                        ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            spline_formula = "cr(x, knots=internal_knots, lower_bound=lower_bound, upper_bound=upper_bound) - 1"
            spline_data = {
                'x': x,
                'internal_knots': [internal_knot],
                'lower_bound': boundary_lo,
                'upper_bound': boundary_hi,
            }
            X_spline = dmatrix(spline_formula, spline_data, return_type='dataframe')
            X_spline_sm = sm.add_constant(X_spline, has_constant='add')
            X_linear_sm = sm.add_constant(x2d, has_constant='add')

            fit_warning_msgs: List[str] = []
            with warnings.catch_warnings(record=True) as w_records:
                warnings.simplefilter("always")
                model_spline = sm.GLM(y, X_spline_sm, family=sm.families.Binomial()).fit()
                model_linear = sm.GLM(y, X_linear_sm, family=sm.families.Binomial()).fit()
                fit_warning_msgs = [str(w.message) for w in w_records]

            lr_stat = max(2.0 * (model_spline.llf - model_linear.llf), 0.0)
            df_diff = max(int(round(model_spline.df_model - model_linear.df_model)), 1)
            p_nonlin = float(chi2.sf(lr_stat, df=df_diff))

            x_plot = np.linspace(lo, hi, int(n_grid)).reshape(-1, 1)
            X_grid_spline = build_design_matrices(
                [X_spline.design_info],
                {
                    'x': x_plot[:, 0],
                    'internal_knots': [internal_knot],
                    'lower_bound': boundary_lo,
                    'upper_bound': boundary_hi,
                },
                return_type='dataframe',
            )[0]
            X_grid_spline = sm.add_constant(X_grid_spline, has_constant='add')
            pred_grid = model_spline.get_prediction(X_grid_spline).summary_frame(alpha=0.05)
            y_grid = pred_grid['mean'].values
            lower_ci = pred_grid['mean_ci_lower'].values
            upper_ci = pred_grid['mean_ci_upper'].values

            baseline_risk = float(np.mean(model_spline.predict(X_spline_sm)))

            ax.scatter(x_core, y_core, s=10, alpha=final_layout_kwargs['scatter_alpha'], color='gray', edgecolors='none')
            ax.plot(x_plot[:, 0], y_grid, color=final_layout_kwargs['line_color'], linewidth=2.2)
            ax.fill_between(x_plot[:, 0], lower_ci, upper_ci, color='red', alpha=0.2)
            ax.axhline(y=baseline_risk, color='gray', linestyle='--', alpha=0.7, linewidth=1.2)
            ax.set_xlim([lo, hi])
            ax.set_ylim([-0.05, 0.8])
            ax.set_xlabel(display_name, fontsize=final_layout_kwargs['label_fontsize'])
            ax.set_ylabel('Predicted Risk', fontsize=final_layout_kwargs['label_fontsize'])
            ax.set_title(
                f"RCS: {display_name}\n{_format_p_value(p_nonlin)} (non-linearity)",
                fontsize=final_layout_kwargs['fontsize'],
                fontweight='bold',
            )
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            summary_rows.append({
                'feature': feature,
                'display_feature': display_name,
                'p_nonlinearity': p_nonlin,
                'n_samples': int(len(x)),
                'effective_n_knots': effective_n_knots,
                'boundary_knot_p10': boundary_lo,
                'internal_knot_p50': internal_knot,
                'boundary_knot_p90': boundary_hi,
                'x_plot_min_p05': lo,
                'x_plot_max_p95': hi,
                'baseline_risk': baseline_risk,
                'fit_warning_count': len(fit_warning_msgs),
            })
            if np.isclose(p_nonlin, 1.0):
                if fit_warning_msgs:
                    print(f"  ⚠️ {display_name}: P=1.000 and model warnings detected")
                    for msg in fit_warning_msgs[:2]:
                        print(f"     - {msg}")
                else:
                    print(f"  ℹ️ {display_name}: P=1.000 with no model warnings")
        except Exception as exc:
            ax.text(0.5, 0.5, f"{display_name}\nRCS fit failed:\n{exc}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(
        "Restricted Cubic Spline Risk Curves of Winner-Panel Metabolites",
        fontsize=final_layout_kwargs['title_fontsize'],
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    summary_path = str(Path(save_path).with_suffix('.json'))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'positive_class_idx': int(positive_class_idx),
                'prevalence': prevalence,
                'n_features': n_feat,
                'features': summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✅ RCS panel plot saved to: {save_path}")
    print(f"✅ RCS summary saved to: {summary_path}")
    final_layout_kwargs['rcs_summary_path'] = summary_path
    final_layout_kwargs['rcs_backend'] = 'python'
    final_layout_kwargs['rcs_engine'] = 'patsy_statsmodels'
    return save_path, final_layout_kwargs


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("Testing Phase 3 Visualization Tools")
    print("="*80)

    # ========================================================================
    # Test 1: Pareto Trajectory Plot
    # ========================================================================
    print("\n" + "="*80)
    print("Test 1: Pareto Trajectory Plot")
    print("="*80)

    test_data_path = "test_results/phase2_correct_architecture_test_20260421_164705.json"

    if os.path.exists(test_data_path):
        print(f"\n✅ Found test data: {test_data_path}")

        # 生成帕累托轨迹图
        output_path = plot_pareto_trajectory(
            history_log_path=test_data_path,
            best_panel_path=test_data_path,
            save_path="output/figures/fig1_pareto_trajectory.pdf"
        )

        print(f"\n✅ Test 1 completed successfully!")
        print(f"   Output: {output_path}")
    else:
        print(f"\n❌ Test data not found: {test_data_path}")
        print("   Please run Phase 2 tests first to generate test data.")

    # ========================================================================
    # Test 2: 4D Radar Comparison Plot (使用假数据)
    # ========================================================================
    print("\n" + "="*80)
    print("Test 2: 4D Radar Comparison Plot (Dummy Data)")
    print("="*80)

    # 构造假数据：展示 PToT Winner 的"均衡碾压"优势
    dummy_profiles = {
        'Phase 1 Baseline': {
            'AUC': 0.85,           # 高预测性能
            'f_bio': 0.45,         # 低生物学价值（文献特征少）
            '1-f_cost': 0.30,      # 低简约性（特征多，成本高）
            '1-f_corr': 0.60       # 中等独立性（有冗余）
        },
        'Phase 0 Baseline': {
            'AUC': 0.78,           # 中等预测性能
            'f_bio': 0.88,         # 高生物学价值（全是文献特征）
            '1-f_cost': 0.92,      # 高简约性（特征少）
            '1-f_corr': 0.75       # 较高独立性
        },
        'PToT Winner': {
            'AUC': 0.90,           # 最高预测性能
            'f_bio': 0.85,         # 高生物学价值（平衡）
            '1-f_cost': 0.82,      # 高简约性（平衡）
            '1-f_corr': 0.88       # 高独立性（平衡）
        }
    }

    print("\nDummy data profiles:")
    for model, scores in dummy_profiles.items():
        print(f"  {model}:")
        for dim, val in scores.items():
            print(f"    {dim}: {val:.2f}")

    # 生成雷达图
    radar_output = plot_radar_comparison(
        profiles_dict=dummy_profiles,
        save_path="output/figures/fig3_radar_4d.pdf"
    )

    print(f"\n✅ Test 2 completed successfully!")
    print(f"   Output: {radar_output}")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "="*80)
    print("All Tests Completed!")
    print("="*80)
    print("\nGenerated figures:")
    print("  1. output/figures/fig2b_pareto_trajectory.pdf")
    print("  2. output/figures/fig3_radar_4d.pdf")
    print("\n" + "="*80)
