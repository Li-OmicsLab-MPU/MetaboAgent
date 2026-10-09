from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from src.agents.phase3.styles.journal_registry import get_journal_style
from src.agents.phase3.styles.style_types import StylePreset


FONT_FAMILY_ALIASES = {
    "times": "Times New Roman",
    "times roman": "Times New Roman",
    "times new roman": "Times New Roman",
    "times new roman ps": "Times New Roman",
}


def normalize_font_family(font_family: str | None) -> str:
    raw = str(font_family or "").strip()
    if not raw:
        return raw
    return FONT_FAMILY_ALIASES.get(raw.lower(), raw)


JOURNAL_STYLE_ALIASES = {
    "gut": "gut",
    "science": "science_advances",
    "science_advances": "science_advances",
    "science-advances": "science_advances",
    "science advances": "science_advances",
    "cell": "cell_metabolism",
    "cell_metabolism": "cell_metabolism",
    "cell-metabolism": "cell_metabolism",
    "cell metabolism": "cell_metabolism",
    "nature": "nature",
    "nature_minimal": "nature",
}

STYLE_ROLE_OVERRIDES: Dict[Tuple[str, str], Dict[str, str]] = {
    ("nature", "phase0_prior_evidence_atlas"): {
        "mechanistic": "#19AA9D",
        "phase1": "#3D85C6",
        "clinical": "#E96362",
        "redundancy": "#9B7FC4",
        "positive": "#19AA9D",
        "uncertainty_fill": "#D8EEF0",
        "primary_fill": "#D8EEF0",
        "winner": "#202754",
        "baseline": "#3D85C6",
        "reference": "#BBBDC0",
    },
}


SEMANTIC_COLOR_KEYS = (
    "phase0",
    "phase1",
    "phase2",
    "winner",
    "baseline",
    "mechanistic",
    "clinical",
    "cost",
    "redundancy",
    "highlight",
    "reference",
    "neutral",
    "negative",
    "positive",
    "uncertainty_fill",
)


def normalize_journal_style(style_key: str | None = None) -> str:
    raw_key = str(style_key or "nature").strip().lower()
    return JOURNAL_STYLE_ALIASES.get(raw_key, raw_key)


def _lighten_hex(hex_color: str, amount: float = 0.70) -> str:
    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        return hex_color
    rgb = [int(color[i:i + 2], 16) for i in (0, 2, 4)]
    lightened = [round(channel + (255 - channel) * amount) for channel in rgb]
    return "#" + "".join(f"{channel:02X}" for channel in lightened)


def _semantic_colors(palette: List[str], journal_style: str | None = None) -> Dict[str, str]:
    colors = list(palette)
    while len(colors) < 8:
        colors.append(colors[-1] if colors else "#9E9E9E")

    style_key = normalize_journal_style(journal_style)
    if style_key == "science_advances":
        positive_color = colors[4]
        negative_color = colors[0]
        return {
            "phase0": colors[2],
            "phase1": colors[1],
            "phase2": colors[4],
            "winner": colors[0],
            "baseline": colors[1],
            "mechanistic": colors[2],
            "clinical": colors[4],
            "cost": colors[5],
            "redundancy": colors[3],
            "highlight": colors[6],
            "reference": colors[-1],
            "neutral": colors[-1],
            "negative": negative_color,
            "positive": positive_color,
            "uncertainty_fill": _lighten_hex(positive_color, 0.68),
        }

    return {
        "phase0": colors[2],
        "phase1": colors[1],
        "phase2": colors[4],
        "winner": colors[0],
        "baseline": colors[1],
        "mechanistic": colors[2],
        "clinical": colors[0],
        "cost": colors[5],
        "redundancy": colors[3],
        "highlight": colors[6],
        "reference": colors[-1],
        "neutral": colors[-1],
        "negative": colors[0],
        "positive": colors[1],
        "uncertainty_fill": _lighten_hex(colors[1], 0.68),
    }


def _apply_role_overrides(theme: Dict[str, Any], task_name: str | None = None) -> Dict[str, Any]:
    if not task_name:
        return theme
    style_key = normalize_journal_style(str(theme.get("journal_style") or theme.get("key") or "nature"))
    overrides = STYLE_ROLE_OVERRIDES.get((style_key, str(task_name).strip()))
    if not overrides:
        return theme

    semantic = dict(theme.get("semantic_colors") or {})
    for key, value in overrides.items():
        if key in SEMANTIC_COLOR_KEYS:
            semantic[key] = value
        else:
            theme[key] = value
    theme["semantic_colors"] = semantic
    if semantic.get("winner"):
        theme["primary_color"] = semantic["winner"]
    if semantic.get("uncertainty_fill"):
        theme["primary_fill"] = overrides.get("primary_fill", semantic["uncertainty_fill"])
    if semantic.get("baseline"):
        theme["baseline_color"] = semantic["baseline"]
    if semantic.get("reference"):
        theme["reference_color"] = semantic["reference"]
    return theme


def style_to_theme(
    style: StylePreset | Mapping[str, Any] | str | None = None,
    task_name: str | None = None,
) -> Dict[str, Any]:
    if isinstance(style, StylePreset):
        preset = style
        palette = list(preset.palette)
        semantic_colors = _semantic_colors(palette, preset.key)
        theme = {
            **asdict(preset),
            "journal_style": preset.key,
            "semantic_colors": semantic_colors,
            "primary_color": semantic_colors["winner"],
            "primary_fill": semantic_colors["uncertainty_fill"],
            "baseline_color": semantic_colors["baseline"],
            "reference_color": semantic_colors["reference"],
            "text_color": "#1F2937",
            "subtitle_color": "#64748B",
            "axis_text_color": "#374151",
            "grid_color": "#E9EDF2",
            "panel_border_color": "#D7DEE8",
            "background_color": "#FFFFFF",
        }
    elif isinstance(style, Mapping):
        if style.get("semantic_colors"):
            theme = dict(style)
            theme["journal_style"] = normalize_journal_style(
                str(theme.get("journal_style") or theme.get("key") or theme.get("name") or "nature")
            )
        else:
            style_key = style.get("key") or style.get("journal_style") or style.get("name")
            preset = get_journal_style(str(style_key or "nature"))
            palette = list(preset.palette)
            semantic_colors = _semantic_colors(palette, preset.key)
            theme = {
                **asdict(preset),
                "journal_style": preset.key,
                "semantic_colors": semantic_colors,
                "primary_color": semantic_colors["winner"],
                "primary_fill": semantic_colors["uncertainty_fill"],
                "baseline_color": semantic_colors["baseline"],
                "reference_color": semantic_colors["reference"],
                "text_color": "#1F2937",
                "subtitle_color": "#64748B",
                "axis_text_color": "#374151",
                "grid_color": "#E9EDF2",
                "panel_border_color": "#D7DEE8",
                "background_color": "#FFFFFF",
            }
        override_font = normalize_font_family(style.get("font_family"))
        if override_font:
            theme["font_family"] = override_font
        elif theme.get("font_family"):
            theme["font_family"] = normalize_font_family(theme.get("font_family"))
    else:
        preset = get_journal_style(str(style or "nature"))
        palette = list(preset.palette)
        semantic_colors = _semantic_colors(palette, preset.key)
        theme = {
            **asdict(preset),
            "journal_style": preset.key,
            "semantic_colors": semantic_colors,
            "primary_color": semantic_colors["winner"],
            "primary_fill": semantic_colors["uncertainty_fill"],
            "baseline_color": semantic_colors["baseline"],
            "reference_color": semantic_colors["reference"],
            "text_color": "#1F2937",
            "subtitle_color": "#64748B",
            "axis_text_color": "#374151",
            "grid_color": "#E9EDF2",
            "panel_border_color": "#D7DEE8",
            "background_color": "#FFFFFF",
        }
        theme["font_family"] = normalize_font_family(theme.get("font_family"))
    return _apply_role_overrides(theme, task_name)


def write_theme_json(
    style: StylePreset | Mapping[str, Any] | str | None,
    output_path: str | Path,
    task_name: str | None = None,
) -> str:
    theme = style_to_theme(style, task_name=task_name)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(theme, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def resolve_theme_color(theme: Mapping[str, Any] | None, key: str, fallback: str) -> str:
    if not theme:
        return fallback
    semantic = theme.get("semantic_colors") or {}
    if isinstance(semantic, Mapping) and semantic.get(key):
        return str(semantic[key])
    if theme.get(key):
        return str(theme[key])
    return fallback


def palette_for_roles(theme: Mapping[str, Any] | None, roles: Iterable[str]) -> List[str]:
    if not theme:
        return []
    return [resolve_theme_color(theme, role, "#9E9E9E") for role in roles]
