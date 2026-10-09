"""
Phase 4 prompt templates.

These prompts turn the structured `llm_sections.json` contract into grounded
section-writing instructions for an OpenAI-compatible chat model.
"""

import json
from typing import Any, Dict


PHASE4_REPORT_SYSTEM_PROMPT = """You are a biomedical manuscript writer for a metabolomics biomarker discovery pipeline.

Your job is to write one report section at a time using ONLY the supplied structured facts and section contract.

Writing philosophy:
- Treat the section contract as a manuscript blueprint, not as optional guidance.
- Use the seed summary as a factual outline and coverage checklist, but improve the prose substantially; do not merely paraphrase line by line.
- Prefer fluent academic paragraphs over debug-style narration, while preserving every mandatory quantitative fact and figure-backed claim.

Hard rules:
1. Write exclusively in polished academic English suitable for a biomedical manuscript.
2. Do not invent numbers, p-values, confidence intervals, pathways, figures, patient examples, or model behavior that are not explicitly supported.
3. If evidence is missing, acknowledge the limitation briefly and neutrally instead of filling gaps with generic prose.
4. Keep Methodology factual, reproducible, and operational; reserve downstream interpretation for Results or Discussion.
5. Keep Discussion hypothesis-generating rather than causal when direct validation is absent.
6. In narrative prose, use human-readable metabolite names and do not append HMDB identifiers unless the contract explicitly asks for a dedicated composition list.
7. Preserve every fixed subsection title required by the contract; do not rename, merge, reorder, or drop them.
8. When stating empirical findings in Results or Discussion, end the sentence with the supporting figure citation in parentheses whenever figure-backed evidence is available.
9. Return plain text only. Do not return JSON, markdown fences, bullet commentary about the instructions, or meta-explanations.
10. Do not omit central quantitative facts already required by the contract, especially AUC, thresholds, NRI/IDI, confidence intervals, feature counts, provenance counts, and indexed figure-linked examples.
11. Do not replace concrete quantitative reporting with vague claims such as "better performance", "meaningful utility", or "improved reclassification" unless the sentence already contains the supporting numbers.
12. Do not expose internal optimization/debug variables such as f_perf, f_bio, f_cost, or f_corr in manuscript prose unless the contract explicitly requests them.
13. For SHAP-style interpretation, describe direction as model-based attribution rather than direct biological causality when the contract indicates such caution.
14. Follow section-specific boundaries, mandatory mentions, forbidden phrasings, and figure-writing instructions when they are provided in the contract.
15. For Methodology sections, prioritize explicit description of prior-evidence curation, preprocessing, optimization, and statistical evaluation when those components are present in the contract.
16. When the contract includes engineered-feature counts or names, explain the feature-engineering rationale before reporting which engineered candidates survived or were discarded.
17. Candidate-pool sizes must be taken from the current run facts in the contract; never reuse counts from older runs or infer them from figure overlap summaries.
18. If the contract indicates engineered_count > 0, explicitly acknowledge engineered feature retention and name the surviving engineered feature when available.
19. If the contract includes imbalance-aware stability-search facts, describe them cautiously as run-aligned methodological facts; do not overclaim unsupported execution details beyond the supplied evidence.
20. If the supplied facts indicate that the selected AutoGluon baseline learner is a WeightedEnsemble, explain once that it is an automatically learned weighted combination of multiple high-ranking base learners selected according to validation performance, rather than presenting it as a single standalone learner.
"""


_BULKY_KEYS = {
    'bio_debug_artifact',
    'bio_debug',
    'bio_context',
    'search_model_bio_debug',
    'phase1_model_bio_debug',
    'feature_records',
    'winner_feature_records',
    'figures',
    'task_records',
    'calibration_curve',
    'calibration_bins',
    'layers',
    'pareto_front',
    'cv_predictions',
    'top_candidate_beams',
}


def _shrink_value(value: Any, *, key: str = '', depth: int = 0) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:497] + '...'

    if isinstance(value, list):
        if key in _BULKY_KEYS:
            sample = [_shrink_value(item, depth=depth + 1) for item in value[:3]]
            return {
                '_summary': f'truncated list for `{key}`',
                'item_count': len(value),
                'sample_items': sample,
            }
        if not value:
            return []
        if depth >= 3:
            return {
                '_summary': 'list truncated by depth limit',
                'item_count': len(value),
            }
        if all(not isinstance(item, (dict, list)) for item in value):
            if len(value) <= 10:
                return value
            return value[:8] + [f'... ({len(value) - 8} more items)']
        sample = [_shrink_value(item, depth=depth + 1) for item in value[:4]]
        if len(value) <= 4:
            return sample
        return {
            '_summary': 'list truncated',
            'item_count': len(value),
            'sample_items': sample,
        }

    if isinstance(value, dict):
        if key in _BULKY_KEYS:
            keys = list(value.keys())
            return {
                '_summary': f'truncated dict for `{key}`',
                'key_count': len(keys),
                'available_keys': keys[:12],
            }
        if depth >= 3:
            keys = list(value.keys())
            return {
                '_summary': 'dict truncated by depth limit',
                'key_count': len(keys),
                'available_keys': keys[:10],
            }
        compact: Dict[str, Any] = {}
        for idx, (child_key, child_value) in enumerate(value.items()):
            if idx >= 18:
                compact['_truncated_keys'] = len(value) - 18
                break
            compact[child_key] = _shrink_value(child_value, key=child_key, depth=depth + 1)
        return compact

    return str(value)


def _build_prompt_ready_payload(
    section_key: str,
    section_payload: Dict[str, Any],
    global_guidance: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_payload = {
        'section_key': section_key,
        'title': section_payload.get('title', section_key),
        'objective': section_payload.get('objective', ''),
        'seed_summary': section_payload.get('content', ''),
        'style_constraints': section_payload.get('style_constraints', []),
        'section_blueprint': section_payload.get('section_blueprint', []),
        'section_boundaries': section_payload.get('section_boundaries', []),
        'required_quantitative_mentions': section_payload.get('required_quantitative_mentions', []),
        'required_qualitative_mentions': section_payload.get('required_qualitative_mentions', []),
        'forbidden_phrasings': section_payload.get('forbidden_phrasings', []),
        'figure_writing_instructions': section_payload.get('figure_writing_instructions', []),
        'required_facts': _shrink_value(section_payload.get('required_facts', {}), key='required_facts'),
        'recommended_figures': _shrink_value(section_payload.get('recommended_figures', []), key='recommended_figures'),
        'missing_evidence': section_payload.get('missing_evidence', []),
        'prompt_scaffold': section_payload.get('prompt_scaffold', ''),
        'global_constraints': global_guidance.get('constraints', []),
    }

    serialized = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    if len(serialized) <= 45000:
        return prompt_payload

    prompt_payload['required_facts'] = {
        '_summary': 'required_facts truncated aggressively to stay within model context',
        'available_top_level_keys': list((section_payload.get('required_facts') or {}).keys())[:20],
    }
    prompt_payload['recommended_figures'] = [
        {
            'figure_id': item.get('figure_id'),
            'title': item.get('title'),
            'path': item.get('path'),
        }
        for item in (section_payload.get('recommended_figures') or [])[:8]
        if isinstance(item, dict)
    ]
    return prompt_payload


def build_section_generation_prompt(
    section_key: str,
    section_payload: Dict[str, Any],
    global_guidance: Dict[str, Any],
) -> str:
    """Build a section-level generation prompt from the writer contract."""

    compact_payload = _build_prompt_ready_payload(
        section_key=section_key,
        section_payload=section_payload,
        global_guidance=global_guidance,
    )

    return (
        "[Task]\n"
        f"Write the `{section_key}` section.\n\n"
        "[Section Contract]\n"
        f"{json.dumps(compact_payload, ensure_ascii=False, indent=2)}\n\n"
        "[Output Requirements]\n"
        "- Return plain section prose only.\n"
        "- Write in English only.\n"
        "- Do not include headings unless the section contract clearly requires substructure.\n"
        "- Keep the response grounded in the provided facts.\n"
        "- Treat the seed summary as a factual outline and coverage checklist; improve the prose instead of copying its wording mechanically.\n"
        "- Preserve all central numeric facts already present in the seed summary unless a factual correction is required by the supplied facts.\n"
        "- If supplied facts correct an outdated seed statement, follow the supplied facts and not the seed wording.\n"
        "- Satisfy every section blueprint item, quantitative mention, qualitative mention, boundary condition, and figure-writing instruction when they are present in the contract.\n"
        "- For methodology_workflow, write the section as a manuscript methods workflow rather than an internal phase-by-phase execution log.\n"
        "- Avoid forbidden phrasings or vague summary language if the contract flags them.\n"
        "- If a fact is unavailable, state the limitation briefly instead of fabricating detail.\n"
        "- Outside dedicated composition lists, do not add HMDB identifiers after metabolite names.\n"
        "- Do not rename or reorder fixed subsection titles provided in the section contract.\n"
    )
