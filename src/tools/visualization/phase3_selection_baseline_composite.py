from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _resolve_font_family(journal_theme: Dict[str, Any] | None = None) -> str:
    if isinstance(journal_theme, dict):
        font_family = str(journal_theme.get("font_family") or "").strip()
        if font_family:
            return font_family
    return "Arial"


def _resolve_text_color(journal_theme: Dict[str, Any] | None = None) -> str:
    if isinstance(journal_theme, dict):
        text_color = str(journal_theme.get("text_color") or "").strip()
        if text_color:
            return text_color
    return "#111827"


def _save_figure_bundle(save_path: str) -> List[str]:
    output_path = Path(save_path)
    stem = output_path.with_suffix("")
    requested_suffix = output_path.suffix.lower() or ".pdf"
    ordered_exts = [requested_suffix] + [ext for ext in [".png", ".svg", ".pdf"] if ext != requested_suffix]
    return [str(stem.with_suffix(ext)) for ext in ordered_exts]


def _rasterize_pdf_to_png(pdf_path: Path, png_path: Path | None = None, zoom: float = 2.6) -> Path:
    import fitz

    resolved_png = png_path or pdf_path.with_suffix(".png")
    if resolved_png.exists() and resolved_png.stat().st_mtime >= pdf_path.stat().st_mtime:
        return resolved_png

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(resolved_png))
    finally:
        doc.close()
    return resolved_png


def _resolve_raster_source(path: str) -> Path:
    candidate = Path(path)
    if candidate.suffix.lower() == ".png" and candidate.exists():
        return candidate

    png_candidate = candidate.with_suffix(".png")
    if png_candidate.exists():
        return png_candidate

    if candidate.suffix.lower() == ".pdf" and candidate.exists():
        return _rasterize_pdf_to_png(candidate)

    pdf_candidate = candidate.with_suffix(".pdf")
    if pdf_candidate.exists():
        return _rasterize_pdf_to_png(pdf_candidate, png_candidate)

    if candidate.exists():
        raise ValueError(
            f"Composite figure currently requires PNG or PDF inputs. Received unsupported source: {candidate}"
        )
    raise FileNotFoundError(f"Composite figure source not found: {candidate}")


def _load_png(path: str) -> np.ndarray:
    resolved = _resolve_raster_source(path)
    with Image.open(resolved) as img:
        return np.asarray(img.convert("RGBA"))


def _content_mask(image: np.ndarray) -> np.ndarray:
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]
    return ~((rgb > 245).all(axis=2) & (alpha > 0)) & (alpha > 0)


def _trim_image_to_content(image: np.ndarray, pad: int = 24) -> np.ndarray:
    mask = _content_mask(image)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def _trim_image_horizontally_to_content(image: np.ndarray, pad: int = 24) -> np.ndarray:
    mask = _content_mask(image)
    _, xs = np.where(mask)
    if len(xs) == 0:
        return image
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.shape[1])
    return image[:, x0:x1]


def _pad_image(
    image: np.ndarray,
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
) -> np.ndarray:
    if top == bottom == left == right == 0:
        return image
    return np.pad(
        image,
        ((top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=255,
    )


def _find_split_point(col_signal: np.ndarray, start: int, end: int, fallback: int) -> int:
    start = max(0, min(start, len(col_signal) - 1))
    end = max(start + 1, min(end, len(col_signal)))
    if end - start < 10:
        return fallback

    low_mask = col_signal[start:end] < 0.02
    best_center = fallback
    best_width = -1
    run_start = None
    for idx, is_low in enumerate(low_mask):
        if is_low and run_start is None:
            run_start = idx
        elif (not is_low) and run_start is not None:
            width = idx - run_start
            if width > best_width:
                best_width = width
                best_center = start + (run_start + idx - 1) // 2
            run_start = None
    if run_start is not None:
        width = len(low_mask) - run_start
        if width > best_width:
            best_center = start + (run_start + len(low_mask) - 1) // 2
    return int(best_center)


def _split_objective_shift_triptych(image: np.ndarray) -> List[np.ndarray]:
    # Match the original R/cowplot layout in `plot_phase2_phase1_to_phase2_summary.R`:
    # 1) global title band with rel_heights = c(0.08, 1)
    # 2) body split into radar_panel and contrib_panel with rel_widths = c(1.00, 1.90)
    # 3) contrib_panel split into main_track_panel and track_panel with rel_widths = c(2.05, 1.35)
    trimmed = _trim_image_to_content(image, pad=10)
    total_h, total_w = trimmed.shape[:2]
    if total_h < 20 or total_w < 20:
        return [trimmed.copy(), trimmed.copy(), trimmed.copy()]

    title_frac = 0.08 / 1.08
    body_y0 = min(max(int(round(total_h * title_frac)) + 2, 0), total_h - 1)
    body = trimmed[body_y0:, :, :].copy()
    body_h, body_w = body.shape[:2]

    left_frac = 1.0 / (1.0 + 1.90)
    right_inner_frac = 2.05 / (2.05 + 1.35)
    split1 = int(round(body_w * left_frac))
    split2 = split1 + int(round((body_w - split1) * right_inner_frac))

    gutter = max(6, int(round(body_w * 0.006)))
    left_panel = body[:, : max(split1 - gutter, 1), :].copy()
    middle_panel = body[:, max(split1 + gutter, 0) : max(split2 - gutter, split1 + gutter + 1), :].copy()
    right_panel = body[:, min(split2 + gutter, body_w - 1) :, :].copy()

    return [
        _trim_image_horizontally_to_content(left_panel, pad=10),
        _trim_image_horizontally_to_content(middle_panel, pad=10),
        _trim_image_horizontally_to_content(right_panel, pad=10),
    ]


def _generate_objective_shift_subpanels(objective_shift_path: str) -> List[Path]:
    objective_path = _resolve_raster_source(objective_shift_path)
    asset_dir = objective_path.parent / f"{objective_path.stem}_assets"
    radar_csv = asset_dir / f"{objective_path.stem}_radar_profiles.csv"
    contrib_csv = asset_dir / "phase2_winner_feature_contributions.csv"
    panel_prefix = asset_dir / f"{objective_path.stem}_subpanel"
    panel_paths = [
        panel_prefix.with_name(f"{panel_prefix.name}_a.png"),
        panel_prefix.with_name(f"{panel_prefix.name}_b.png"),
        panel_prefix.with_name(f"{panel_prefix.name}_c.png"),
    ]

    if all(path.exists() for path in panel_paths):
        return panel_paths
    if not (radar_csv.exists() and contrib_csv.exists()):
        return []

    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "src" / "tools" / "visualization" / "r" / "plot_phase2_phase1_to_phase2_subpanels.R"
    if not r_script.exists():
        return []

    env = os.environ.copy()
    env["METABOAGENT_R_LIB"] = str(project_root / ".r_libs")
    subprocess.run(
        ["Rscript", str(r_script), str(radar_csv), str(contrib_csv), str(panel_prefix)],
        check=True,
        cwd=str(project_root),
        env=env,
    )
    return [path for path in panel_paths if path.exists()]


def _load_objective_shift_triptych(objective_shift_path: str) -> List[np.ndarray]:
    try:
        panel_paths = _generate_objective_shift_subpanels(objective_shift_path)
    except Exception:
        panel_paths = []
    if len(panel_paths) == 3:
        return [_load_png(str(path)) for path in panel_paths]

    objective_shift_image = _load_png(objective_shift_path)
    return _split_objective_shift_triptych(objective_shift_image)


def _draw_panel_image(ax: plt.Axes, label: str, image: np.ndarray, label_color: str = "#111827") -> None:
    ax.imshow(image)
    ax.set_axis_off()
    ax.text(
        -0.03,
        1.02,
        str(label),
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=label_color,
    )


def plot_phase1_selection_baseline_composite(
    stability_landscape_path: str,
    method_support_path: str,
    panel_correlation_path: str,
    autogluon_roc_path: str,
    save_path: str,
    title: str = "",
    panel_labels: Sequence[str] = ("A", "B", "C", "D"),
    width_ratios: Sequence[float] = (1.27, 1.0),
    height_ratios: Sequence[float] = (0.95, 1.05),
    journal_theme: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compose fig1d/fig1e/fig1g/fig2a into a publication-style 2x2 panel."""

    output_paths = _save_figure_bundle(save_path)
    primary_output = output_paths[0]
    output_dir = Path(primary_output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: Iterable[str] = (
        stability_landscape_path,
        method_support_path,
        panel_correlation_path,
        autogluon_roc_path,
    )
    images = [_load_png(path) for path in image_paths]

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    title_fontfamily = _resolve_font_family(journal_theme)
    text_color = _resolve_text_color(journal_theme)
    grid = fig.add_gridspec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    for ax, label, image in zip(axes, panel_labels, images):
        _draw_panel_image(ax, str(label), image, text_color)

    if title.strip():
        fig.suptitle(title.strip(), fontsize=18, fontweight="bold", y=0.995, fontfamily=title_fontfamily, color=text_color)
        top_margin = 0.955
    else:
        top_margin = 0.985

    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.025, top=top_margin, wspace=0.045, hspace=0.055)

    for path_str in output_paths:
        ext = Path(path_str).suffix.lower()
        save_kwargs: Dict[str, Any] = {
            "facecolor": "white",
            "bbox_inches": "tight",
        }
        if ext == ".png":
            save_kwargs["dpi"] = 300
        fig.savefig(path_str, **save_kwargs)

    plt.close(fig)

    metadata = {
        "generated_paths": output_paths,
        "source_paths": {
            "fig1d_stability_landscape": str(_resolve_raster_source(stability_landscape_path)),
            "fig1e_method_support_dot_matrix": str(_resolve_raster_source(method_support_path)),
            "fig1g_final_panel_correlation_heatmap": str(_resolve_raster_source(panel_correlation_path)),
            "fig2a_autogluon_roc": str(_resolve_raster_source(autogluon_roc_path)),
        },
        "panel_labels": list(panel_labels),
        "width_ratios": list(width_ratios),
        "height_ratios": list(height_ratios),
        "title": title,
    }
    return primary_output, metadata


def plot_phase2_clinical_validation_composite(
    objective_shift_path: str,
    final_roc_path: str,
    dca_path: str,
    calibration_path: str,
    save_path: str,
    title: str = "",
    panel_labels: Sequence[str] = ("A", "B", "C", "D", "E", "F"),
    width_ratios: Sequence[float] = (1.12, 0.96, 0.92),
    height_ratios: Sequence[float] = (1.0, 1.0),
    journal_theme: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compose split fig3 + fig4a/fig4b/fig4e into a 2x3 publication panel."""

    output_paths = _save_figure_bundle(save_path)
    primary_output = output_paths[0]
    Path(primary_output).parent.mkdir(parents=True, exist_ok=True)

    top_images = _load_objective_shift_triptych(objective_shift_path)
    top_images = [
        _pad_image(top_images[0], top=24, left=18, right=18),
        _pad_image(top_images[1], top=24, left=12, right=12),
        _pad_image(top_images[2], top=24, left=12, right=12),
    ]
    bottom_images = [_load_png(path) for path in (final_roc_path, dca_path, calibration_path)]
    images = top_images + bottom_images

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    title_fontfamily = _resolve_font_family(journal_theme)
    text_color = _resolve_text_color(journal_theme)
    grid = fig.add_gridspec(2, 3, width_ratios=width_ratios, height_ratios=height_ratios)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]

    for ax, label, image in zip(axes, panel_labels, images):
        _draw_panel_image(ax, str(label), image, text_color)

    if title.strip():
        fig.suptitle(title.strip(), fontsize=18, fontweight="bold", y=0.995, fontfamily=title_fontfamily, color=text_color)
        top_margin = 0.955
    else:
        top_margin = 0.985

    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.025, top=top_margin, wspace=0.035, hspace=0.06)

    for path_str in output_paths:
        ext = Path(path_str).suffix.lower()
        save_kwargs: Dict[str, Any] = {
            "facecolor": "white",
            "bbox_inches": "tight",
        }
        if ext == ".png":
            save_kwargs["dpi"] = 300
        fig.savefig(path_str, **save_kwargs)

    plt.close(fig)

    metadata = {
        "generated_paths": output_paths,
        "source_paths": {
            "fig3_phase2_objective_shift_summary": str(_resolve_raster_source(objective_shift_path)),
            "fig4a_final_roc": str(_resolve_raster_source(final_roc_path)),
            "fig4b_dca": str(_resolve_raster_source(dca_path)),
            "fig4e_calibration": str(_resolve_raster_source(calibration_path)),
        },
        "layout": "2x3_split_fig3",
        "panel_labels": list(panel_labels),
        "width_ratios": list(width_ratios),
        "height_ratios": list(height_ratios),
        "title": title,
    }
    return primary_output, metadata


def plot_phase3_shap_interpretation_composite(
    shap_summary_path: str,
    shap_dependence_path: str,
    save_path: str,
    title: str = "",
    panel_labels: Sequence[str] = ("A", "B"),
    width_ratios: Sequence[float] = (0.9, 1.55),
    journal_theme: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compose Phase 2 winner SHAP summary and dependence panels into one interpretation panel."""

    output_paths = _save_figure_bundle(save_path)
    primary_output = output_paths[0]
    Path(primary_output).parent.mkdir(parents=True, exist_ok=True)

    dependence_candidate = Path(shap_dependence_path)
    if dependence_candidate.name.startswith("phase2_winner_shap_dependence_panels"):
        canonical_candidate = dependence_candidate.with_name(
            dependence_candidate.name.replace(
                "phase2_winner_shap_dependence_panels",
                "phase2_winner_shap_summary_dependence_panels",
                1,
            )
        )
        if canonical_candidate.with_suffix(".png").exists() or canonical_candidate.with_suffix(".pdf").exists():
            shap_dependence_path = str(canonical_candidate)

    shap_summary_image = _pad_image(_load_png(shap_summary_path), top=16, left=10, right=10)
    shap_dependence_image = _pad_image(_load_png(shap_dependence_path), top=16, left=10, right=10)
    images = [shap_summary_image, shap_dependence_image]

    fig = plt.figure(figsize=(18, 8.6), facecolor="white")
    title_fontfamily = _resolve_font_family(journal_theme)
    grid = fig.add_gridspec(1, 2, width_ratios=width_ratios)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
    ]

    for ax, label, image in zip(axes, panel_labels, images):
        _draw_panel_image(ax, str(label), image)

    if title.strip():
        fig.suptitle(title.strip(), fontsize=18, fontweight="bold", y=0.995, fontfamily=title_fontfamily)
        top_margin = 0.955
    else:
        top_margin = 0.985

    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=top_margin, wspace=0.045)

    for path_str in output_paths:
        ext = Path(path_str).suffix.lower()
        save_kwargs: Dict[str, Any] = {
            "facecolor": "white",
            "bbox_inches": "tight",
        }
        if ext == ".png":
            save_kwargs["dpi"] = 300
        fig.savefig(path_str, **save_kwargs)

    plt.close(fig)

    metadata = {
        "generated_paths": output_paths,
        "source_paths": {
            "fig4c_shap_summary": str(_resolve_raster_source(shap_summary_path)),
            "fig4c_shap_dependence_panels": str(_resolve_raster_source(shap_dependence_path)),
        },
        "layout": "1x2_horizontal",
        "panel_labels": list(panel_labels),
        "width_ratios": list(width_ratios),
        "title": title,
    }
    return primary_output, metadata


def plot_phase2_radar_validation_composite(
    radar_4d_path: str,
    final_roc_path: str,
    dca_path: str,
    calibration_path: str,
    save_path: str,
    title: str = "",
    panel_labels: Sequence[str] = ("A", "B", "C", "D"),
    width_ratios: Sequence[float] = (0.98, 1.02),
    height_ratios: Sequence[float] = (0.96, 1.04),
    journal_theme: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compose whole fig3_radar_4d + fig4a/fig4b/fig4e into a 2x2 publication panel."""

    output_paths = _save_figure_bundle(save_path)
    primary_output = output_paths[0]
    Path(primary_output).parent.mkdir(parents=True, exist_ok=True)

    images = [
        _pad_image(_load_png(radar_4d_path), top=18, left=12, right=12),
        _pad_image(_load_png(final_roc_path), top=18, left=8, right=8),
        _pad_image(_load_png(dca_path), top=18, left=8, right=8),
        _pad_image(_load_png(calibration_path), top=18, left=8, right=8),
    ]

    fig = plt.figure(figsize=(16.2, 12.2), facecolor="white")
    title_fontfamily = _resolve_font_family(journal_theme)
    grid = fig.add_gridspec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    for ax, label, image in zip(axes, panel_labels, images):
        _draw_panel_image(ax, str(label), image)

    if title.strip():
        fig.suptitle(title.strip(), fontsize=18, fontweight="bold", y=0.995, fontfamily=title_fontfamily)
        top_margin = 0.955
    else:
        top_margin = 0.985

    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=top_margin, wspace=0.04, hspace=0.05)

    for path_str in output_paths:
        ext = Path(path_str).suffix.lower()
        save_kwargs: Dict[str, Any] = {
            "facecolor": "white",
            "bbox_inches": "tight",
        }
        if ext == ".png":
            save_kwargs["dpi"] = 300
        fig.savefig(path_str, **save_kwargs)

    plt.close(fig)

    metadata = {
        "generated_paths": output_paths,
        "source_paths": {
            "fig3_radar_4d": str(_resolve_raster_source(radar_4d_path)),
            "fig4a_final_roc": str(_resolve_raster_source(final_roc_path)),
            "fig4b_dca": str(_resolve_raster_source(dca_path)),
            "fig4e_calibration": str(_resolve_raster_source(calibration_path)),
        },
        "layout": "2x2_whole_fig3_radar",
        "panel_labels": list(panel_labels),
        "width_ratios": list(width_ratios),
        "height_ratios": list(height_ratios),
        "title": title,
    }
    return primary_output, metadata
