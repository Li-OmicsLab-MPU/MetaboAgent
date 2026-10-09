"""
Phase 4 renderers.

The initial renderer produces a report-like Markdown/HTML document from
`report_context.json` and `llm_sections.json` without introducing heavy PDF
or template dependencies. PDF rendering will be added in a later step.
"""

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


SECTION_HEADINGS = {
    'executive_summary': 'Executive Summary',
    'methodology_workflow': 'Methodology & Workflow',
    'results': 'Results',
    'clinical_utility_decision_support': 'Clinical Utility & Decision Support',
    'discussion_mechanistic_insights': 'Discussion & Evidence-scoped Biological Interpretation',
}


def _safe_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_str(value: Any) -> str:
    if value is None:
        return ''
    return str(value)


def _relative_asset_path(report_dir: str, asset_path: str) -> str:
    normalized_path = _safe_str(asset_path).strip()
    if not normalized_path:
        return ''
    try:
        return os.path.relpath(normalized_path, start=report_dir)
    except ValueError:
        return normalized_path


def _figure_caption(figure: Dict[str, Any]) -> str:
    return _safe_str(figure.get('caption_seed')).strip()


def _figure_section_matches(figure: Dict[str, Any], section_name: str) -> bool:
    return _safe_str(figure.get('section')) == section_name


def _section_figures(report_context: Dict[str, Any], section_name: str) -> List[Dict[str, Any]]:
    figures = _safe_list(_safe_dict(report_context.get('phase3')).get('figures'))
    return [
        figure
        for figure in figures
        if isinstance(figure, dict) and _figure_section_matches(figure, section_name)
    ]


def _render_asset_links_markdown(outputs: List[Dict[str, Any]], report_dir: str) -> List[str]:
    lines: List[str] = []
    for output in outputs:
        path = _safe_str(output.get('path'))
        if not path:
            continue
        relative_path = _relative_asset_path(report_dir, path)
        label = Path(path).name
        role = _safe_str(output.get('role')) or 'asset'
        lines.append(f"- {role.title()}: [{label}]({relative_path})")
    return lines


def _render_embedded_figures_markdown(figures: List[Dict[str, Any]], report_dir: str) -> List[str]:
    lines = ['### Figures', '']
    if not figures:
        lines.append('- No section-aligned figures available.')
        lines.append('')
        return lines

    for figure in figures:
        primary = _safe_dict(figure.get('primary_output'))
        figure_id = _safe_str(figure.get('figure_id')) or 'figure'
        title = _safe_str(figure.get('title')) or figure_id
        path = _safe_str(primary.get('path'))
        relative_path = _relative_asset_path(report_dir, path)
        kind = _safe_str(primary.get('kind')).lower()
        caption = _figure_caption(figure)
        all_outputs = _safe_list(figure.get('all_outputs'))
        auxiliary_outputs = _safe_list(figure.get('auxiliary_outputs'))
        metadata_output = _safe_dict(figure.get('metadata_output'))

        lines.append(f'#### Figure `{figure_id}`: {title}')
        lines.append('')
        if caption:
            lines.append(f'*{caption}*')
            lines.append('')

        if relative_path:
            lines.append(f'- Primary asset: [{Path(path).name}]({relative_path})')
        else:
            lines.append('- Primary asset: unavailable')

        if kind == 'image' and relative_path:
            lines.append('')
            lines.append(f'![{title}]({relative_path})')
        elif kind == 'pdf' and relative_path:
            lines.append('')
            lines.append(f'<object data="{relative_path}" type="application/pdf" width="100%" height="520">')
            lines.append(f'  <p>PDF preview unavailable. Open <a href="{relative_path}">{html.escape(Path(path).name)}</a>.</p>')
            lines.append('</object>')

        extra_outputs: List[Dict[str, Any]] = []
        extra_outputs.extend(item for item in auxiliary_outputs if isinstance(item, dict))
        if metadata_output:
            extra_outputs.append(metadata_output)
        if extra_outputs:
            lines.append('')
            lines.append('Additional assets:')
            lines.extend(_render_asset_links_markdown(extra_outputs, report_dir))
        elif len(all_outputs) > 1:
            additional = [
                item for item in all_outputs
                if isinstance(item, dict) and _safe_str(item.get('path')) != path
            ]
            if additional:
                lines.append('')
                lines.append('Additional assets:')
                lines.extend(_render_asset_links_markdown(additional, report_dir))
        lines.append('')
    return lines


def _section_content(llm_sections: Dict[str, Any], key: str) -> str:
    sections = llm_sections.get('sections', {}) if isinstance(llm_sections, dict) else {}
    section = sections.get(key, {}) if isinstance(sections, dict) else {}
    content = section.get('content', '') if isinstance(section, dict) else ''
    return str(content).strip()


def _section_payload(llm_sections: Dict[str, Any], key: str) -> Dict[str, Any]:
    sections = llm_sections.get('sections', {}) if isinstance(llm_sections, dict) else {}
    section = sections.get(key, {}) if isinstance(sections, dict) else {}
    return section if isinstance(section, dict) else {}


def _render_bullet_list(items: List[Any], empty_message: str) -> List[str]:
    normalized = [item for item in items if item not in (None, '', [], {})]
    if not normalized:
        return [f'- {empty_message}']
    return [f'- `{item}`' if isinstance(item, str) else f'- {item}' for item in normalized]


def _render_scalar(value: Any) -> str:
    if isinstance(value, str):
        return f'`{value}`' if value else '`N/A`'
    if isinstance(value, bool):
        return '`true`' if value else '`false`'
    if value is None:
        return '`N/A`'
    return f'`{value}`'


def _render_structured_block(data: Any, heading: str) -> List[str]:
    lines = [f'### {heading}', '']
    if isinstance(data, dict):
        if not data:
            lines.append('- No structured facts available.')
            lines.append('')
            return lines
        for key, value in data.items():
            label = key.replace('_', ' ').strip().title()
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, ensure_ascii=False, indent=2)
                lines.append(f'- {label}:')
                lines.append('')
                lines.append('```json')
                lines.append(serialized)
                lines.append('```')
            else:
                lines.append(f'- {label}: {_render_scalar(value)}')
        lines.append('')
        return lines

    if isinstance(data, list):
        if not data:
            lines.append('- No structured facts available.')
            lines.append('')
            return lines
        simple_scalars = all(not isinstance(item, (dict, list)) for item in data)
        if simple_scalars:
            lines.extend(_render_bullet_list(data, 'No items available.'))
            lines.append('')
            return lines
        lines.append('```json')
        lines.append(json.dumps(data, ensure_ascii=False, indent=2))
        lines.append('```')
        lines.append('')
        return lines

    lines.append(f'- {_render_scalar(data)}')
    lines.append('')
    return lines


def _render_feature_records(feature_records: List[Dict[str, Any]]) -> List[str]:
    lines = ['### Winner Panel Composition', '']
    if not feature_records:
        lines.append('- No structured feature records available.')
        lines.append('')
        return lines

    for record in feature_records:
        feature_name = _safe_str(record.get('composition_label') or record.get('report_label') or record.get('feature_name')) or 'unknown_feature'
        origin_type = _safe_str(record.get('origin_type')) or 'unknown'
        origin_subtype = _safe_str(record.get('origin_subtype')) or 'unknown'
        matched_pathway = _safe_str(record.get('matched_pathway'))
        prior_supported = bool(record.get('prior_supported'))
        note_parts = [origin_type, origin_subtype]
        hmdb_ids = _safe_list(record.get('hmdb_ids'))
        if hmdb_ids:
            note_parts.append(f'HMDB={",".join(str(item) for item in hmdb_ids)}')
        if matched_pathway:
            note_parts.append(f'pathway={matched_pathway}')
        if prior_supported:
            note_parts.append('prior_supported=true')
        lines.append(f"- `{feature_name}` | {' | '.join(note_parts)}")
    lines.append('')
    return lines


def _render_figures(figures: List[Dict[str, Any]], report_dir: str) -> List[str]:
    lines = ['### Figure Index', '']
    if not figures:
        lines.append('- No figures were registered in `figure_manifest.json`.')
        lines.append('')
        return lines

    for figure in figures:
        primary = _safe_dict(figure.get('primary_output'))
        figure_id = _safe_str(figure.get('figure_id'))
        title = _safe_str(figure.get('title'))
        path = _safe_str(primary.get('path'))
        section = _safe_str(figure.get('section'))
        relative_path = _relative_asset_path(report_dir, path)
        if relative_path:
            lines.append(f'- `{figure_id}` | {title} | section: `{section}` | [open asset]({relative_path})')
        else:
            lines.append(f'- `{figure_id}` | {title} | section: `{section}` | path unavailable')
    lines.append('')
    return lines


def _render_recommended_figures(figures: List[Dict[str, Any]], report_dir: str) -> List[str]:
    lines = ['### Figure References', '']
    if not figures:
        lines.append('- No recommended figures for this section.')
        lines.append('')
        return lines
    for figure in figures:
        path = _safe_str(figure.get('path'))
        relative_path = _relative_asset_path(report_dir, path)
        link_text = Path(path).name if path else 'asset unavailable'
        if relative_path:
            asset_text = f'[{link_text}]({relative_path})'
        else:
            asset_text = '`asset unavailable`'
        lines.append(
            f"- `{_safe_str(figure.get('figure_id'))}` | {_safe_str(figure.get('title'))} | {asset_text}"
        )
    lines.append('')
    return lines


def _placeholder_text(section_payload: Dict[str, Any]) -> str:
    objective = _safe_str(section_payload.get('objective'))
    if objective:
        return f'Pending LLM narrative. Objective: {objective}'
    return 'Pending LLM narrative.'


def _is_generated_section(section_payload: Dict[str, Any]) -> bool:
    return _safe_str(section_payload.get('status')) in {'generated', 'deterministic'}


def _strip_duplicate_heading(content: str, heading: str) -> str:
    cleaned = content.strip()
    if not cleaned:
        return cleaned

    lines = cleaned.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ''

    first_line = lines[0].strip()
    normalized_first = re.sub(r'^[#\s]+', '', first_line).strip().lower()
    normalized_heading = heading.strip().lower()
    if normalized_first == normalized_heading:
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)

    return '\n'.join(lines).strip()


def _normalize_section_content(content: str, heading: str) -> str:
    cleaned = _strip_duplicate_heading(content, heading)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def _render_section(
    key: str,
    report_context: Dict[str, Any],
    llm_sections: Dict[str, Any],
    report_dir: str,
) -> List[str]:
    section_payload = _section_payload(llm_sections, key)
    heading = _safe_str(section_payload.get('title')) or SECTION_HEADINGS.get(key, key)
    content = _section_content(llm_sections, key)
    content = _normalize_section_content(content, heading)
    generated = _is_generated_section(section_payload)
    lines = [f'## {heading}', '']
    lines.append(content or _placeholder_text(section_payload))
    lines.append('')

    style_constraints = _safe_list(section_payload.get('style_constraints'))
    if style_constraints and not generated:
        lines.append('### Writing Constraints')
        lines.append('')
        lines.extend([f'- {item}' for item in style_constraints])
        lines.append('')

    required_facts = section_payload.get('required_facts')
    if not generated:
        lines.extend(_render_structured_block(required_facts, 'Structured Facts'))

    missing_evidence = _safe_list(section_payload.get('missing_evidence'))
    if missing_evidence:
        lines.append('### Missing Evidence')
        lines.append('')
        lines.extend([f'- `{item}`' for item in missing_evidence])
        lines.append('')

    recommended_figures = _safe_list(section_payload.get('recommended_figures'))
    if recommended_figures:
        lines.extend(_render_recommended_figures(recommended_figures, report_dir))

    prompt_scaffold = _safe_str(section_payload.get('prompt_scaffold'))
    if prompt_scaffold and not generated:
        lines.append('### Prompt Scaffold')
        lines.append('')
        lines.append(prompt_scaffold)
        lines.append('')

    return lines


def render_markdown(
    report_context: Dict[str, Any],
    llm_sections: Dict[str, Any],
    report_dir: str,
) -> str:
    """Render a structured Markdown report from Phase 4 inputs."""

    report_path = Path(report_dir) / 'report.md'
    run = _safe_dict(report_context.get('run'))
    quality_checks = _safe_dict(report_context.get('quality_checks'))
    winner_features = _safe_list(_safe_dict(report_context.get('phase2')).get('winner_features'))
    winner_feature_records = _safe_list(_safe_dict(report_context.get('phase2')).get('winner_feature_dictionary')) or _safe_list(_safe_dict(report_context.get('phase2')).get('winner_feature_records'))
    figures = _safe_list(_safe_dict(report_context.get('phase3')).get('figures'))
    global_guidance = _safe_dict(llm_sections.get('global_guidance'))
    summary = _safe_dict(llm_sections.get('summary'))
    section_order = _safe_list(llm_sections.get('section_order')) or list(SECTION_HEADINGS.keys())

    lines = [
        '# Phase 4 Report',
        '',
        f"- Run ID: `{run.get('run_id', '')}`",
        f"- Generated at: `{report_context.get('generated_at', '')}`",
        f"- LLM Generated: `{global_guidance.get('llm_generation_succeeded', False)}`",
        f"- Generated Sections: `{summary.get('generated_section_count', 0)}`",
        '',
        '## Report Status',
        '',
        f"- Missing required inputs: `{_safe_list(quality_checks.get('missing_required_inputs'))}`",
        f"- Missing source files: `{_safe_list(quality_checks.get('missing_source_files'))}`",
    ]
    lines.append('')

    for key in section_order:
        lines.extend(_render_section(key, report_context, llm_sections, report_dir))
        if key == 'results':
            lines.extend(_render_feature_records(winner_feature_records))
            if not winner_feature_records:
                lines.extend(_render_bullet_list(winner_features, 'No winner features available yet.'))
                lines.append('')

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return str(report_path)


def _render_html_text_block(content: str) -> str:
    normalized = content.strip()
    if not normalized:
        return ''

    blocks: List[str] = []
    paragraph_lines: List[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = ' '.join(line.strip() for line in paragraph_lines if line.strip())
        if paragraph:
            blocks.append(f'<p>{html.escape(paragraph)}</p>')
        paragraph_lines = []

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if line.startswith('#### '):
            flush_paragraph()
            blocks.append(f'<h4>{html.escape(line[5:].strip())}</h4>')
            continue
        if line.startswith('### '):
            flush_paragraph()
            blocks.append(f'<h3>{html.escape(line[4:].strip())}</h3>')
            continue
        paragraph_lines.append(line.rstrip())

    flush_paragraph()
    return '\n'.join(blocks)


def _render_html_simple_list(items: List[str]) -> str:
    if not items:
        return '<ul><li>None</li></ul>'
    return '<ul>' + ''.join(f'<li>{html.escape(item)}</li>' for item in items) + '</ul>'


def _render_html_asset_links(outputs: List[Dict[str, Any]], report_dir: str) -> str:
    if not outputs:
        return ''
    items: List[str] = []
    for output in outputs:
        path = _safe_str(output.get('path'))
        if not path:
            continue
        relative_path = _relative_asset_path(report_dir, path)
        label = html.escape(Path(path).name)
        role = html.escape((_safe_str(output.get('role')) or 'asset').title())
        items.append(f'<li>{role}: <a href="{html.escape(relative_path)}">{label}</a></li>')
    if not items:
        return ''
    return '<ul class="asset-list">' + ''.join(items) + '</ul>'


def _render_html_figure_cards(figures: List[Dict[str, Any]], report_dir: str) -> str:
    if not figures:
        return '<h3>Figures</h3><p>No section-aligned figures available.</p>'

    cards: List[str] = ['<h3>Figures</h3>']
    for figure in figures:
        primary = _safe_dict(figure.get('primary_output'))
        path = _safe_str(primary.get('path'))
        relative_path = _relative_asset_path(report_dir, path)
        kind = _safe_str(primary.get('kind')).lower()
        figure_id = html.escape(_safe_str(figure.get('figure_id')) or 'figure')
        title = html.escape(_safe_str(figure.get('title')) or figure_id)
        caption = html.escape(_figure_caption(figure))

        preview_html = ''
        if relative_path and kind == 'image':
            preview_html = f'<img src="{html.escape(relative_path)}" alt="{title}" />'
        elif relative_path and kind == 'pdf':
            preview_html = (
                f'<object data="{html.escape(relative_path)}" type="application/pdf" width="100%" height="560">'
                f'<p>PDF preview unavailable. <a href="{html.escape(relative_path)}">Open asset</a>.</p>'
                '</object>'
            )
        elif relative_path:
            preview_html = f'<p><a href="{html.escape(relative_path)}">Open asset</a></p>'
        else:
            preview_html = '<p>Primary asset unavailable.</p>'

        extra_outputs: List[Dict[str, Any]] = []
        extra_outputs.extend(item for item in _safe_list(figure.get('auxiliary_outputs')) if isinstance(item, dict))
        metadata_output = _safe_dict(figure.get('metadata_output'))
        if metadata_output:
            extra_outputs.append(metadata_output)

        cards.append('<div class="figure-card">')
        cards.append(f'<h4>Figure {figure_id}: {title}</h4>')
        if caption:
            cards.append(f'<p class="caption">{caption}</p>')
        if relative_path:
            cards.append(f'<p class="asset-link"><a href="{html.escape(relative_path)}">{html.escape(Path(path).name)}</a></p>')
        cards.append(preview_html)
        if extra_outputs:
            cards.append('<div class="extra-assets"><strong>Additional assets</strong>')
            cards.append(_render_html_asset_links(extra_outputs, report_dir))
            cards.append('</div>')
        cards.append('</div>')
    return '\n'.join(cards)


def _render_html_recommended_figures(figures: List[Dict[str, Any]], report_dir: str) -> str:
    if not figures:
        return '<h3>Figure References</h3><p>No recommended figures for this section.</p>'
    items: List[str] = []
    for figure in figures:
        path = _safe_str(figure.get('path'))
        relative_path = _relative_asset_path(report_dir, path)
        title = html.escape(_safe_str(figure.get('title')))
        figure_id = html.escape(_safe_str(figure.get('figure_id')))
        if relative_path:
            items.append(f'<li><code>{figure_id}</code> | {title} | <a href="{html.escape(relative_path)}">{html.escape(Path(path).name)}</a></li>')
        else:
            items.append(f'<li><code>{figure_id}</code> | {title}</li>')
    return '<h3>Figure References</h3><ul>' + ''.join(items) + '</ul>'


def _render_html_feature_records(feature_records: List[Dict[str, Any]]) -> str:
    if not feature_records:
        return '<h3>Winner Panel Composition</h3><p>No structured feature records available.</p>'
    items: List[str] = []
    for record in feature_records:
        feature_name = html.escape(_safe_str(record.get('composition_label') or record.get('report_label') or record.get('feature_name')) or 'unknown_feature')
        origin_type = html.escape(_safe_str(record.get('origin_type')) or 'unknown')
        origin_subtype = html.escape(_safe_str(record.get('origin_subtype')) or 'unknown')
        matched_pathway = _safe_str(record.get('matched_pathway'))
        prior_supported = bool(record.get('prior_supported'))
        note_parts = [origin_type, origin_subtype]
        hmdb_ids = _safe_list(record.get('hmdb_ids'))
        if hmdb_ids:
            note_parts.append(f'HMDB={",".join(str(item) for item in hmdb_ids)}')
        if matched_pathway:
            note_parts.append(f'pathway={matched_pathway}')
        if prior_supported:
            note_parts.append('prior_supported=true')
        items.append(f'<li><code>{feature_name}</code> | {" | ".join(html.escape(part) for part in note_parts)}</li>')
    return '<h3>Winner Panel Composition</h3><ul>' + ''.join(items) + '</ul>'


def _render_html_structured_block(data: Any, heading: str) -> str:
    if isinstance(data, (dict, list)):
        serialized = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
        return f'<h3>{html.escape(heading)}</h3><pre>{serialized}</pre>'
    return f'<h3>{html.escape(heading)}</h3><p>{html.escape(_safe_str(data) or "N/A")}</p>'


def render_html(
    report_context: Dict[str, Any],
    llm_sections: Dict[str, Any],
    report_dir: str,
) -> str:
    """Render a lightweight HTML report with figure references only."""

    html_path = Path(report_dir) / 'report.html'
    run = _safe_dict(report_context.get('run'))
    quality_checks = _safe_dict(report_context.get('quality_checks'))
    winner_feature_records = _safe_list(_safe_dict(report_context.get('phase2')).get('winner_feature_dictionary')) or _safe_list(_safe_dict(report_context.get('phase2')).get('winner_feature_records'))
    global_guidance = _safe_dict(llm_sections.get('global_guidance'))
    summary = _safe_dict(llm_sections.get('summary'))
    section_order = _safe_list(llm_sections.get('section_order')) or list(SECTION_HEADINGS.keys())

    parts = [
        '<html><head><meta charset="utf-8"><title>Phase 4 Report</title>',
        '<style>'
        'body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.65;color:#1f2937;}'
        'h1,h2,h3,h4{color:#111827;}'
        '.meta, .status{background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:16px 20px;margin:18px 0;}'
        '.figure-card{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:16px 0;background:#fff;}'
        '.figure-card img,.figure-card object{display:block;width:100%;margin-top:12px;border-radius:8px;background:#f8fafc;}'
        '.caption{color:#4b5563;font-style:italic;}'
        '.asset-link,.extra-assets{margin-top:10px;}'
        'pre{white-space:pre-wrap;word-wrap:break-word;background:#f8fafc;padding:16px;border-radius:10px;border:1px solid #e5e7eb;}'
        'code{background:#f3f4f6;padding:2px 5px;border-radius:4px;}'
        'ul{padding-left:22px;}'
        '</style></head><body>',
        '<h1>Phase 4 Report</h1>',
        '<div class="meta">',
        f'<p><strong>Run ID:</strong> <code>{html.escape(_safe_str(run.get("run_id")))}</code></p>',
        f'<p><strong>Generated at:</strong> <code>{html.escape(_safe_str(report_context.get("generated_at")))}</code></p>',
        f'<p><strong>LLM Generated:</strong> <code>{html.escape(_safe_str(global_guidance.get("llm_generation_succeeded", False)))}</code></p>',
        f'<p><strong>Generated Sections:</strong> <code>{html.escape(_safe_str(summary.get("generated_section_count", 0)))}</code></p>',
        '</div>',
        '<h2>Report Status</h2>',
        '<div class="status">',
        f'<p><strong>Missing required inputs:</strong> <code>{html.escape(json.dumps(_safe_list(quality_checks.get("missing_required_inputs")), ensure_ascii=False))}</code></p>',
        f'<p><strong>Missing source files:</strong> <code>{html.escape(json.dumps(_safe_list(quality_checks.get("missing_source_files")), ensure_ascii=False))}</code></p>',
        '</div>',
    ]

    for key in section_order:
        section_payload = _section_payload(llm_sections, key)
        heading = _safe_str(section_payload.get('title')) or SECTION_HEADINGS.get(key, key)
        content = _normalize_section_content(_section_content(llm_sections, key), heading) or _placeholder_text(section_payload)
        recommended_figures = _safe_list(section_payload.get('recommended_figures'))

        parts.append(f'<section><h2>{html.escape(heading)}</h2>')
        parts.append(_render_html_text_block(content))

        missing_evidence = _safe_list(section_payload.get('missing_evidence'))
        if missing_evidence:
            parts.append('<h3>Missing Evidence</h3>')
            parts.append(_render_html_simple_list([_safe_str(item) for item in missing_evidence]))

        if recommended_figures:
            parts.append(_render_html_recommended_figures(recommended_figures, report_dir))

        if key == 'results':
            parts.append(_render_html_feature_records(winner_feature_records))
        parts.append('</section>')

    parts.append('</body></html>\n')

    html_path.write_text(''.join(parts), encoding='utf-8')
    return str(html_path)


def render_pdf(
    report_context: Dict[str, Any],
    llm_sections: Dict[str, Any],
    report_dir: str,
) -> Optional[str]:
    """Render a PDF report using ReportLab when available."""

    if not REPORTLAB_AVAILABLE:
        return None

    pdf_path = Path(report_dir) / 'report.pdf'
    run = _safe_dict(report_context.get('run'))
    quality_checks = _safe_dict(report_context.get('quality_checks'))
    winner_features = _safe_list(_safe_dict(report_context.get('phase2')).get('winner_features'))
    winner_feature_records = _safe_list(_safe_dict(report_context.get('phase2')).get('winner_feature_dictionary')) or _safe_list(_safe_dict(report_context.get('phase2')).get('winner_feature_records'))
    global_guidance = _safe_dict(llm_sections.get('global_guidance'))
    summary = _safe_dict(llm_sections.get('summary'))
    section_order = _safe_list(llm_sections.get('section_order')) or list(SECTION_HEADINGS.keys())

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        'BulletBody',
        parent=body_style,
        leftIndent=14,
        bulletIndent=0,
    )
    code_style = ParagraphStyle(
        'CodeBlock',
        parent=body_style,
        fontName='Courier',
        fontSize=8,
        leading=10,
        backColor=colors.HexColor('#f8fafc'),
        borderColor=colors.HexColor('#e5e7eb'),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8,
    )

    def add_paragraphs(story: List[Any], content: str) -> None:
        normalized = _normalize_section_content(content, '')
        if not normalized:
            return
        for block in normalized.split('\n\n'):
            text = block.strip()
            if not text:
                continue
            if text.startswith('### '):
                story.append(Paragraph(html.escape(text[4:].strip()), subheading_style))
                continue
            if text.startswith('#### '):
                story.append(Paragraph(html.escape(text[5:].strip()), subheading_style))
                continue
            joined = ' '.join(line.strip() for line in text.splitlines() if line.strip())
            if joined:
                story.append(Paragraph(html.escape(joined), body_style))

    def add_bullets(story: List[Any], items: List[str], empty_text: Optional[str] = None) -> None:
        if not items:
            if empty_text:
                story.append(Paragraph(html.escape(empty_text), body_style))
            return
        for item in items:
            story.append(Paragraph(html.escape(_safe_str(item)), bullet_style, bulletText='-'))

    def add_structured_block(story: List[Any], heading: str, data: Any) -> None:
        story.append(Paragraph(html.escape(heading), subheading_style))
        if isinstance(data, (dict, list)):
            rendered = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            rendered = _safe_str(data) or 'N/A'
        story.append(Preformatted(rendered, code_style))

    def add_figure_refs(story: List[Any], refs: List[Dict[str, Any]], empty_text: str) -> None:
        if not refs:
            story.append(Paragraph(html.escape(empty_text), body_style))
            return
        for figure in refs:
            figure_id = _safe_str(figure.get('figure_id')) or 'figure'
            title = _safe_str(figure.get('title')) or figure_id
            path = _safe_str(figure.get('path'))
            text = f'{figure_id} | {title}'
            if path:
                text += f' | {_relative_asset_path(report_dir, path)}'
            story.append(Paragraph(html.escape(text), bullet_style, bulletText='-'))

    story: List[Any] = []
    story.append(Paragraph('Phase 4 Report', title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f'Run ID: {html.escape(_safe_str(run.get("run_id")))}', body_style))
    story.append(Paragraph(f'Generated at: {html.escape(_safe_str(report_context.get("generated_at")))}', body_style))
    story.append(Paragraph(f'LLM Generated: {html.escape(_safe_str(global_guidance.get("llm_generation_succeeded", False)))}', body_style))
    story.append(Paragraph(f'Generated Sections: {html.escape(_safe_str(summary.get("generated_section_count", 0)))}', body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph('Report Status', heading_style))
    add_bullets(
        story,
        [
            f'Missing required inputs: {_safe_list(quality_checks.get("missing_required_inputs"))}',
            f'Missing source files: {_safe_list(quality_checks.get("missing_source_files"))}',
        ],
    )
    story.append(Spacer(1, 0.12 * inch))

    for key in section_order:
        section_payload = _section_payload(llm_sections, key)
        heading = _safe_str(section_payload.get('title')) or SECTION_HEADINGS.get(key, key)
        content = _normalize_section_content(_section_content(llm_sections, key), heading) or _placeholder_text(section_payload)
        recommended_figures = _safe_list(section_payload.get('recommended_figures'))

        story.append(Paragraph(html.escape(heading), heading_style))
        add_paragraphs(story, content)

        missing_evidence = [f'Missing evidence: {_safe_str(item)}' for item in _safe_list(section_payload.get('missing_evidence'))]
        add_bullets(story, missing_evidence)

        if recommended_figures:
            story.append(Paragraph('Figure References', subheading_style))
            add_figure_refs(story, recommended_figures, 'No recommended figures for this section.')

        if key == 'results':
            story.append(Paragraph('Winner Panel Composition', subheading_style))
            if winner_feature_records:
                for record in winner_feature_records:
                    feature_name = _safe_str(record.get('composition_label') or record.get('report_label') or record.get('feature_name')) or 'unknown_feature'
                    origin_type = _safe_str(record.get('origin_type')) or 'unknown'
                    origin_subtype = _safe_str(record.get('origin_subtype')) or 'unknown'
                    matched_pathway = _safe_str(record.get('matched_pathway'))
                    prior_supported = bool(record.get('prior_supported'))
                    text = f'{feature_name} | {origin_type} | {origin_subtype}'
                    hmdb_ids = _safe_list(record.get('hmdb_ids'))
                    if hmdb_ids:
                        text += f' | HMDB={",".join(str(item) for item in hmdb_ids)}'
                    if matched_pathway:
                        text += f' | pathway={matched_pathway}'
                    if prior_supported:
                        text += ' | prior_supported=true'
                    story.append(Paragraph(html.escape(text), bullet_style, bulletText='-'))
            else:
                add_bullets(story, [_safe_str(item) for item in winner_features], 'No winner features available yet.')

        story.append(Spacer(1, 0.12 * inch))

    add_structured_block(story, 'Generation Constraints', _safe_list(global_guidance.get('constraints')))
    add_structured_block(story, 'Collection Summary', _safe_dict(report_context.get('collection_summary')))
    add_structured_block(story, 'Missing Required Inputs', _safe_list(quality_checks.get('missing_required_inputs')))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title='Phase 4 Report',
        author='MetaboAgent',
    )
    doc.build(story)
    return str(pdf_path)
