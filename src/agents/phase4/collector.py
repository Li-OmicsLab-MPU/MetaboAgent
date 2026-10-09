"""
Phase 4 artifact collection.

This module resolves authoritative Phase 0-3 inputs and records exactly where
each payload came from, so downstream normalization and LLM writing can remain
fact-centric and source-aware.
"""

import json
import os
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config_manager import ConfigManager


PHASE2_PATTERN = 'output/runs/phase2_result_*.json'
PHASE01_PATTERNS = [
    'output/runs/phase0_phase1_*.json',
    'output/runs/phase0_phase1*.json',
]


def _dedupe_paths(paths: List[str]) -> List[str]:
    resolved: List[str] = []
    seen = set()
    for path in paths:
        if not path:
            continue
        normalized = os.path.normpath(str(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def _read_text_if_exists(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    return Path(path).read_text(encoding='utf-8')


def _find_latest_phase2_result() -> str:
    candidates = [path for path in glob(PHASE2_PATTERN) if os.path.exists(path)]
    if not candidates:
        return ''
    candidates.sort(key=lambda item: os.path.getmtime(item), reverse=True)
    return candidates[0]


def _get_runtime_root() -> str:
    return str(os.environ.get('METABOAGENT_RUNTIME_ROOT', '') or '').strip()


def _runtime_path(*parts: str) -> str:
    """Resolve a path inside the explicit run root when Phase 4 is run-scoped."""
    runtime_root = _get_runtime_root()
    if not runtime_root:
        return ''
    return str(Path(runtime_root, *parts))


def _runtime_phase2_candidates() -> List[str]:
    """Return run-local Phase 2 mirrors before consulting global test outputs."""
    names = ('phase2_winner_scores.json',)
    candidates: List[str] = []
    for base in (
        _runtime_path('phase2', 'artifacts'),
        _runtime_path('phase1', 'legacy', 'artifacts'),
        _runtime_path('phase1', 'artifacts'),
        _runtime_path('output', 'phase2', 'artifacts'),
        _runtime_path('output', 'phase1', 'artifacts'),
        _runtime_path('output', 'artifacts'),
    ):
        if base:
            candidates.extend(str(Path(base, name)) for name in names)
    return _dedupe_paths(candidates)


def _find_phase01_candidates() -> List[str]:
    candidates: List[str] = []
    for pattern in PHASE01_PATTERNS:
        candidates.extend(path for path in glob(pattern) if os.path.exists(path))
    return _dedupe_paths(candidates)


def _extract_phase01_run_id(phase2_payload: Dict[str, Any]) -> str:
    text_candidates = [
        str(phase2_payload.get('data_path') or ''),
        str(phase2_payload.get('phase0_output_path') or ''),
        str((phase2_payload.get('lineage') or {}).get('phase0_output_path') or ''),
    ]
    for text in text_candidates:
        normalized = str(text or '').strip()
        if not normalized:
            continue
        match = re.search(r'(phase0_phase1_[A-Za-z0-9_]+)(?=(?:/|\\|_test\.json|\.json|$))', normalized)
        if match:
            return match.group(1)
    return ''


def _preferred_phase01_candidates(phase2_payload: Dict[str, Any]) -> List[str]:
    preferred: List[str] = []

    run_id = _extract_phase01_run_id(phase2_payload)
    if run_id:
        preferred.extend(
            [
                f'output/runs/{run_id}_test.json',
                f'output/runs/{run_id}.json',
            ]
        )

    runtime_root = _get_runtime_root()
    if runtime_root:
        preferred.extend(
            sorted(
                (
                    path
                    for path in glob(
                        os.path.join('output', 'runs', f'*{Path(runtime_root).name}*.json')
                    )
                    if os.path.exists(path)
                ),
                key=lambda item: os.path.getmtime(item),
                reverse=True,
            )
        )

    return _dedupe_paths(preferred)


def _runtime_phase0_candidates() -> List[str]:
    runtime_root = _get_runtime_root()
    if not runtime_root:
        return []

    phase0_root = Path(runtime_root) / 'phase0'
    candidates = [
        str(phase0_root / 'phase0_output_latest.json'),
    ]
    outputs_dir = phase0_root / 'outputs'
    if outputs_dir.exists():
        candidates.extend(
            str(path)
            for path in sorted(
                outputs_dir.glob('*.json'),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        )
    return _dedupe_paths(candidates)


def _phase0_phase1_match_score(
    payload: Dict[str, Any],
    *,
    expected_disease: str,
    expected_target: str,
    expected_data_path: str,
) -> int:
    test_config = payload.get('test_config') if isinstance(payload, dict) else {}
    if not isinstance(test_config, dict):
        return 0

    score = 0
    disease_name = str(test_config.get('disease_name') or '').strip().lower()
    target_column = str(test_config.get('target_column') or '').strip()
    data_path = str(test_config.get('data_path') or '').strip()

    if expected_disease and disease_name == expected_disease.strip().lower():
        score += 4
    if expected_target and target_column == expected_target:
        score += 3
    if expected_data_path and data_path:
        normalized_expected = os.path.normpath(expected_data_path)
        normalized_actual = os.path.normpath(data_path)
        if normalized_actual == normalized_expected:
            score += 5
        elif os.path.basename(normalized_actual) == os.path.basename(normalized_expected):
            score += 2
    return score


def _ordered_phase0_phase1_candidates(
    config: ConfigManager,
    phase2_payload: Dict[str, Any],
) -> List[str]:
    candidates = _find_phase01_candidates()
    if not candidates:
        return _preferred_phase01_candidates(phase2_payload)

    expected_disease = str(
        phase2_payload.get('phase0_output', {}).get('disease_name')
        or phase2_payload.get('disease_name')
        or config.get_test_disease_name()
        or ''
    ).strip()
    expected_target = str(
        phase2_payload.get('target_column')
        or phase2_payload.get('dataset_fingerprint', {}).get('target_column')
        or config.get_target_column()
        or ''
    ).strip()
    expected_data_path = str(
        phase2_payload.get('data_path')
        or config.get_test_data_path()
        or ''
    ).strip()

    preferred = _preferred_phase01_candidates(phase2_payload)
    scored: List[Tuple[int, float, str]] = []
    for candidate in candidates:
        payload = _read_json_if_exists(candidate) or {}
        score = _phase0_phase1_match_score(
            payload,
            expected_disease=expected_disease,
            expected_target=expected_target,
            expected_data_path=expected_data_path,
        )
        scored.append((score, os.path.getmtime(candidate), candidate))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return _dedupe_paths(preferred + [path for _, _, path in scored])


def _is_compact_phase2_result(path: str) -> bool:
    return Path(path).name == 'phase2_winner_scores.json'


def _is_full_phase2_result(path: str) -> bool:
    return bool(path) and not _is_compact_phase2_result(path)


def _ordered_phase2_candidates(
    *,
    explicit_path: str,
    env_path: str,
    latest_phase2_path: str,
    config_phase2_path: str,
) -> List[str]:
    if explicit_path:
        return _dedupe_paths([
            explicit_path,
            env_path,
            latest_phase2_path,
            config_phase2_path,
            *_runtime_phase2_candidates(),
            'output/phase1/artifacts/phase2_winner_scores.json',
            'output/artifacts/phase2_winner_scores.json',
        ])

    preferred_full = []
    compact_fallback = []
    for candidate in [
        env_path,
        latest_phase2_path,
        config_phase2_path,
        *_runtime_phase2_candidates(),
        'output/phase1/artifacts/phase2_winner_scores.json',
        'output/artifacts/phase2_winner_scores.json',
    ]:
        normalized = os.path.normpath(str(candidate)) if candidate else ''
        if not normalized:
            continue
        if _is_compact_phase2_result(normalized):
            compact_fallback.append(normalized)
        else:
            preferred_full.append(normalized)
    return _dedupe_paths(preferred_full + compact_fallback)


def _extract_run_id_from_phase2_path(phase2_path: str) -> str:
    if not phase2_path:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    stem = Path(phase2_path).stem
    for marker in ('phase2_correct_architecture_test_', 'phase2_', 'test_'):
        if marker in stem:
            suffix = stem.split(marker, 1)[-1]
            if suffix:
                return suffix
    return stem or datetime.now().strftime('%Y%m%d_%H%M%S')


def _resolve_phase2_result_path(config: ConfigManager, phase2_result_path: Optional[str]) -> str:
    env_path = os.environ.get('METABOAGENT_PHASE2_RESULT_PATH', '').strip()
    candidates = _ordered_phase2_candidates(
        explicit_path=phase2_result_path or '',
        env_path=env_path,
        latest_phase2_path=_find_latest_phase2_result(),
        config_phase2_path=config.get_path('phase2_winner_panel'),
    )
    # Phase 4 runs from the explicit run root.  An explicit path may still be
    # relative to the project root (the way CLI/config callers commonly pass
    # it), so probe both the current run-root cwd and the original project
    # working directory exposed by the config path.  Keep the returned path
    # absolute once found so the run-root guard and provenance registry agree.
    config_root = Path(str(config.config_path)).resolve().parent if getattr(config, 'config_path', None) else Path.cwd()
    project_candidates = [config_root, *config_root.parents]
    for candidate in candidates:
        if not candidate:
            continue
        raw = Path(candidate).expanduser()
        variants = [raw]
        if not raw.is_absolute():
            variants.extend([Path.cwd() / raw, *[root / raw for root in project_candidates]])
        for variant in variants:
            if variant.exists() and variant.is_file():
                return str(variant.resolve())
    return candidates[0] if candidates else ''


def _candidate_phase1_artifact_dirs(config: ConfigManager) -> List[str]:
    phase1_paths = config.get_phase1_paths()
    return _dedupe_paths([
        _runtime_path('phase1', 'artifacts'),
        _runtime_path('phase1', 'legacy', 'artifacts'),
        phase1_paths.get('artifacts_dir', ''),
        phase1_paths.get('legacy_artifacts_dir', ''),
        'phase1/artifacts',
        'phase1/legacy/artifacts',
        'output/phase1/artifacts',
        'output/artifacts',
    ])


def _candidate_phase1_intermediate_dirs(config: ConfigManager) -> List[str]:
    phase1_paths = config.get_phase1_paths()
    return _dedupe_paths([
        _runtime_path('phase1', 'intermediate', 'latest'),
        phase1_paths.get('intermediate_latest_dir', ''),
        'phase1/intermediate/latest',
        'output/phase1/intermediate/latest',
    ])


def _resolve_existing_path(candidates: List[str]) -> str:
    for candidate in _dedupe_paths(candidates):
        if candidate and os.path.exists(candidate):
            return candidate
    return _dedupe_paths(candidates)[0] if _dedupe_paths(candidates) else ''


def _load_json_source(
    name: str,
    candidates: List[str],
    authority_note: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    deduped = _dedupe_paths(candidates)
    selected_path = ''
    payload: Dict[str, Any] = {}
    for candidate in deduped:
        if not candidate or not os.path.exists(candidate):
            continue
        candidate_payload = _read_json_if_exists(candidate)
        if candidate_payload is None:
            continue
        selected_path = candidate
        payload = candidate_payload
        break
    if not selected_path:
        selected_path = _resolve_existing_path(deduped)
    return payload, {
        'artifact_name': name,
        'format': 'json',
        'selected_path': selected_path,
        'exists': bool(selected_path) and os.path.exists(selected_path),
        'candidates': deduped,
        'authority_note': authority_note,
    }


def _load_text_source(
    name: str,
    candidates: List[str],
    authority_note: str,
) -> Tuple[str, Dict[str, Any]]:
    deduped = _dedupe_paths(candidates)
    selected_path = _resolve_existing_path(deduped)
    payload = _read_text_if_exists(selected_path) or ''
    return payload, {
        'artifact_name': name,
        'format': 'text',
        'selected_path': selected_path,
        'exists': bool(selected_path) and os.path.exists(selected_path),
        'candidates': deduped,
        'authority_note': authority_note,
    }


_AUDIT_FIGURE_TASK_MAP = {
    'plot_phase0_prior_evidence_atlas': ('fig1c', 'Prior Evidence Atlas', 'Methodology & Workflow'),
    'plot_stability_landscape': ('fig1d', 'Phase 1 Stability Landscape', 'Methodology & Workflow'),
    'plot_method_feature_heatmap': ('fig1e', 'Method Support Dot Matrix', 'Methodology & Workflow'),
    'plot_phase1_final_panel_correlation_heatmap': ('fig1g', 'Final Panel Correlation Heatmap', 'Methodology & Workflow'),
    'plot_autogluon_roc': ('fig2a', 'AutoGluon Multi-Model ROC Comparison', 'Results'),
    'plot_phase1_selection_baseline_composite': ('fig2b', 'Stable Panel Selection and Baseline Model Performance', 'Results'),
    'plot_radar_4d': ('fig3', 'Phase 2 Objective Shift and Winner Feature Contribution', 'Results'),
    'plot_phase2_clinical_validation_composite': ('fig4h', 'Phase 2 Decision and Clinical Validation Composite', 'Results'),
    'plot_phase2_radar_validation_composite': ('fig4h_alt', 'Phase 2 Radar and Clinical Validation Composite', 'Results'),
    'plot_phase3_shap_interpretation_composite': ('fig4i', 'SHAP Summary and Dependence Composite', 'Results'),
    'plot_final_roc': ('fig4a', 'Final ROC Curve', 'Results'),
    'plot_final_holdout_roc': ('fig4a_holdout', 'Final Holdout ROC Curve', 'Results'),
    'plot_final_holdout_dca': ('fig4b_holdout', 'Internal Holdout Decision Curve Analysis', 'Results'),
    'plot_dca': ('fig4b', 'Decision Curve Analysis', 'Results'),
    'plot_shap': ('fig4c', 'SHAP Summary Plot', 'Results'),
    'plot_rcs': ('fig4d', 'Restricted Cubic Spline Curves', 'Results'),
    'plot_calibration': ('fig4e', 'Calibration Plot', 'Results'),
    'plot_threshold_performance': ('fig4f', 'Threshold Performance Plot', 'Results'),
    'plot_incremental_value_summary': ('fig4j', 'Incremental Value of the Phase 2 Panel', 'Results'),
}


def _is_retired_holdout_calibration_record(record: Dict[str, Any]) -> bool:
    """Keep the retired internal-holdout calibration figure out of reports."""
    if not isinstance(record, dict):
        return False
    task = str(record.get('task') or record.get('source_task') or '').strip().lower()
    figure_id = str(record.get('figure_id') or '').strip().lower()
    title = str(record.get('title') or '').strip().lower()
    return (
        'plot_final_holdout_calibration' in task
        or 'final_holdout_calibration' in figure_id
        or title == 'internal holdout calibration plot'
    )


def _filter_retired_figure_manifest(figure_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Remove retired figure records from both new and historical manifests."""
    if not isinstance(figure_manifest, dict):
        return figure_manifest
    filtered_tasks = [
        item for item in figure_manifest.get('task_records', [])
        if not _is_retired_holdout_calibration_record(item)
    ]
    filtered_figures = [
        item for item in figure_manifest.get('figures', [])
        if not _is_retired_holdout_calibration_record(item)
    ]
    filtered = dict(figure_manifest)
    filtered['task_records'] = filtered_tasks
    filtered['figures'] = filtered_figures
    summary = dict(filtered.get('summary') or {})
    summary['required_task_count'] = len(filtered_tasks)
    summary['completed_task_count'] = sum(
        1 for item in filtered_tasks if isinstance(item, dict) and item.get('status') == 'completed'
    )
    summary['figure_count'] = len(filtered_figures)
    filtered['summary'] = summary
    return filtered


def _resolve_run_figure_asset(run_root: Path, run_id: str, asset_name: str) -> str:
    """Resolve a figure only from the explicit run root."""
    candidates: List[Path] = []
    if run_id:
        candidates.append(run_root / 'output' / 'final_delivery' / run_id / 'figures' / asset_name)
    candidates.extend([
        run_root / 'output' / 'figures' / asset_name,
        run_root / 'output' / 'figures' / 'single_panels' / asset_name,
        run_root / 'output' / 'figures' / 'composite_panels' / asset_name,
    ])
    if not any(path.exists() for path in candidates):
        candidates.extend(sorted(run_root.glob(f'output/final_delivery/*/figures/{asset_name}')))
    for path in candidates:
        if path.exists() and path.is_file():
            return str(path)
    return ''


def _build_figure_manifest_from_audit(
    audit_payload: Dict[str, Any],
    *,
    run_root: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Create a run-local, report-compatible manifest from the audit inventory.

    This is a historical-run fallback only.  New Phase 3 runs should write the
    canonical figure manifest directly into the run's report directory.
    """
    figures: List[Dict[str, Any]] = []
    task_records: List[Dict[str, Any]] = []
    generated = audit_payload.get('generated_files') if isinstance(audit_payload, dict) else {}
    if not isinstance(generated, dict):
        generated = {}
    for task, descriptor in generated.items():
        if not isinstance(descriptor, dict):
            continue
        raw_path = str(descriptor.get('path') or '').strip()
        asset_name = Path(raw_path).name if raw_path else ''
        if not asset_name:
            continue
        figure_id, title, section = _AUDIT_FIGURE_TASK_MAP.get(
            task,
            (task, task, 'Results'),
        )
        selected = _resolve_run_figure_asset(run_root, run_id, asset_name)
        output_descriptor = {
            'path': selected,
            'kind': 'primary',
            'exists': bool(selected),
            'asset_name': asset_name,
            'source': 'audit_inventory_fallback',
        }
        task_records.append({
            'task': task,
            'plot_key': task,
            'renderer_name': 'audit_inventory_fallback',
            'status': 'completed' if selected else 'missing_primary_output',
            'outputs': [selected] if selected else [],
            'renderer_metadata': {},
            'skip_reason': '',
            'error': '',
        })
        figures.append({
            'figure_id': figure_id,
            'title': title,
            'section': section,
            'source_task': task,
            'plot_key': task,
            'renderer_name': 'audit_inventory_fallback',
            'status': 'completed' if selected else 'missing_primary_output',
            'caption_seed': 'Historical figure inventory fallback; inspect the linked asset and audit manifest for scope.',
            'primary_output': output_descriptor,
            'auxiliary_outputs': [],
            'metadata_output': None,
            'renderer_metadata': {},
            'all_outputs': [output_descriptor],
        })
    return {
        'schema_version': 'phase4.figure_manifest.audit_fallback.v1',
        'generated_at': datetime.now().isoformat(),
        'run_id': run_id,
        'phase2_run_source': '',
        'phase2_result_path': '',
        'phase2_history_path': '',
        'figures_dir': str(run_root / 'output' / 'final_delivery' / run_id / 'figures'),
        'report_root': str(run_root / 'output' / 'reports'),
        'run_report_dir': str(run_root / 'output' / 'reports' / run_id),
        'required_tasks': list(generated),
        'task_records': task_records,
        'figures': figures,
        'summary': {
            'required_task_count': len(task_records),
            'completed_task_count': sum(1 for item in task_records if item.get('status') == 'completed'),
            'figure_count': len(figures),
        },
        'notes': ['Generated from output/audit/figure_table_manifest.json; canonical Phase 3 manifest should be used for new runs.'],
    }


def _build_phase0_source_registry(phase2_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    embedded_payload = phase2_payload.get('phase0_output')
    explicit_phase0_path = str(phase2_payload.get('phase0_output_path', '') or '').strip()
    latest_path = 'storage/phase0_output_latest.json'
    runtime_candidates = _runtime_phase0_candidates()
    if isinstance(embedded_payload, dict) and embedded_payload:
        return embedded_payload, {
            'artifact_name': 'phase0_output',
            'format': 'json',
            'selected_path': 'embedded://phase2_result.phase0_output',
            'exists': True,
            'candidates': [
                'embedded://phase2_result.phase0_output',
                *runtime_candidates,
                latest_path,
            ],
            'authority_note': 'Prefer Phase 2-embedded Phase 0 output when available so all downstream facts stay run-aligned.',
        }
    if explicit_phase0_path and os.path.exists(explicit_phase0_path):
        payload = _read_json_if_exists(explicit_phase0_path) or {}
        return payload, {
            'artifact_name': 'phase0_output',
            'format': 'json',
            'selected_path': explicit_phase0_path,
            'exists': True,
            'candidates': [
                explicit_phase0_path,
                *runtime_candidates,
                latest_path,
            ],
            'authority_note': 'Prefer the Phase 0 output path recorded by the current Phase 2 test artifact before consulting any global latest pointer.',
        }
    for candidate in runtime_candidates + [latest_path]:
        if not candidate or not os.path.exists(candidate):
            continue
        payload = _read_json_if_exists(candidate) or {}
        return payload, {
            'artifact_name': 'phase0_output',
            'format': 'json',
            'selected_path': candidate,
            'exists': True,
            'candidates': runtime_candidates + [latest_path],
            'authority_note': 'Prefer runtime-scoped Phase 0 outputs before falling back to the global latest pointer, so Phase 4 stays bound to the current run.',
        }
    payload = _read_json_if_exists(latest_path) or {}
    return payload, {
        'artifact_name': 'phase0_output',
        'format': 'json',
        'selected_path': latest_path,
        'exists': os.path.exists(latest_path),
        'candidates': runtime_candidates + [latest_path],
        'authority_note': 'Fallback to stable Phase 0 latest output when the current Phase 2 result does not embed phase0_output.',
    }


def collect_phase_artifacts(
    config_path: str = 'config.yaml',
    phase2_result_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect authoritative Phase 0-3 inputs for Phase 4."""

    config = ConfigManager(config_path)
    artifact_dirs = _candidate_phase1_artifact_dirs(config)
    intermediate_dirs = _candidate_phase1_intermediate_dirs(config)

    phase2_path = _resolve_phase2_result_path(config, phase2_result_path)
    phase2_payload = _read_json_if_exists(phase2_path) or {}
    resolved_run_id = str(run_id or os.environ.get('METABOAGENT_RUN_TAG', '') or '').strip()
    resolved_run_id = resolved_run_id or _extract_run_id_from_phase2_path(phase2_path)
    report_dir = os.path.join('output', 'reports', resolved_run_id)

    phase0_payload, phase0_meta = _build_phase0_source_registry(phase2_payload)

    phase1_panel_scores, phase1_panel_scores_meta = _load_json_source(
        name='phase1_panel_scores',
        candidates=[os.path.join(base, 'phase1_panel_scores.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 1 artifact directory first, then fall back to legacy artifact mirrors.',
    )
    feature_provenance, feature_provenance_meta = _load_json_source(
        name='feature_provenance',
        candidates=[os.path.join(base, 'feature_provenance.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 1 artifact directory first, then fall back to legacy artifact mirrors.',
    )
    phase2_search_summary, phase2_search_summary_meta = _load_json_source(
        name='phase2_search_summary',
        candidates=[os.path.join(base, 'phase2_search_summary.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 1 artifact directory first, then fall back to legacy artifact mirrors.',
    )
    phase2_bio_debug, phase2_bio_debug_meta = _load_json_source(
        name='phase2_bio_debug',
        candidates=[os.path.join(base, 'phase2_bio_debug.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 2 biological debug artifact first, then fall back to legacy artifact mirrors.',
    )
    phase2_clinical_utility, phase2_clinical_utility_meta = _load_json_source(
        name='phase2_clinical_utility',
        candidates=[os.path.join(base, 'phase2_clinical_utility.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 2 clinical utility artifact first, then fall back to legacy artifact mirrors.',
    )
    phase2_probability_recalibration, phase2_probability_recalibration_meta = _load_json_source(
        name='phase2_probability_recalibration',
        candidates=[os.path.join(base, 'phase2_probability_recalibration.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 2 probability recalibration artifact first, then fall back to legacy artifact mirrors.',
    )
    phase2_bio_calibration, phase2_bio_calibration_meta = _load_json_source(
        name='phase2_bio_calibration',
        candidates=[os.path.join(base, 'phase2_bio_calibration.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 2 biological calibration artifact first, then fall back to legacy artifact mirrors.',
    )
    phase2_external_validation, phase2_external_validation_meta = _load_json_source(
        name='phase2_external_validation',
        candidates=[os.path.join(base, 'phase2_external_validation.json') for base in artifact_dirs],
        authority_note='Prefer canonical Phase 2 external-validation artifact first, then fall back to legacy artifact mirrors.',
    )
    phase0_phase1_test_result, phase0_phase1_test_result_meta = _load_json_source(
        name='phase0_phase1_test_result',
        candidates=_ordered_phase0_phase1_candidates(config, phase2_payload),
        authority_note='Prefer the Phase 0/1 test artifact whose disease, target, and data path best match the current Phase 2 run, so methodology text can stay aligned with the latest full-pipeline test configuration.',
    )
    feature_selection_summary, feature_selection_summary_meta = _load_json_source(
        name='feature_selection_summary',
        candidates=[
            os.path.join(base, 'feature_selection', 'stability_selection_summary.json')
            for base in intermediate_dirs
        ] + [
            os.path.join(base, 'feature_selection', 'feature_selection_summary.json')
            for base in intermediate_dirs
        ],
        authority_note='Use latest Phase 1 intermediate feature-selection summary from the canonical latest directory.',
    )
    final_selection_summary, final_selection_summary_meta = _load_json_source(
        name='final_selection_summary',
        candidates=[
            os.path.join(base, 'feature_selection', 'final_selection_summary.json')
            for base in intermediate_dirs
        ],
        authority_note='Use latest Phase 1 final selection summary from the canonical latest directory.',
    )
    adaptive_selection_report, adaptive_selection_report_meta = _load_json_source(
        name='adaptive_selection_report',
        candidates=[
            os.path.join(base, 'feature_selection', 'adaptive_selection_report.json')
            for base in intermediate_dirs
        ],
        authority_note='Use latest Phase 1 adaptive selection report from the canonical latest directory.',
    )
    missing_value_metadata, missing_value_metadata_meta = _load_json_source(
        name='missing_value_metadata',
        candidates=[
            os.path.join(base, 'missing_value_metadata.json')
            for base in intermediate_dirs
        ] + [
            os.path.join(base, 'test_lung_cancer_imputed_missing_value_metadata.json')
            for base in intermediate_dirs
        ],
        authority_note='Use latest Phase 1 missing-value audit metadata to describe whether imputation was required in the current run.',
    )
    preprocessing_report, preprocessing_report_meta = _load_json_source(
        name='preprocessing_report',
        candidates=[
            os.path.join(base, 'preprocessing_report.json')
            for base in intermediate_dirs
        ],
        authority_note='Use the unified Phase 1 preprocessing report to describe zero handling, imputation, normalization, transformation, outlier handling, and QC-related decisions.',
    )
    preprocessing_decision_pack, preprocessing_decision_pack_meta = _load_json_source(
        name='preprocessing_decision_pack',
        candidates=[
            os.path.join(base, 'preprocessing_decision_pack.json')
            for base in intermediate_dirs
        ],
        authority_note='Use the unified Phase 1 preprocessing decision pack to recover structured preprocessing rationale for report writing.',
    )
    modeling_readiness_report, modeling_readiness_report_meta = _load_json_source(
        name='modeling_readiness_report',
        candidates=[
            os.path.join(base, 'model_building', 'modeling_readiness_report.json')
            for base in intermediate_dirs
        ],
        authority_note='Use latest Phase 1 modeling readiness report for data-quality and missingness summary.',
    )
    data_analysis_for_modeling, data_analysis_for_modeling_meta = _load_json_source(
        name='data_analysis_for_modeling',
        candidates=[
            os.path.join(base, 'model_analysis', 'data_analysis_for_modeling.json')
            for base in intermediate_dirs
        ],
        authority_note='Use latest Phase 1 model-analysis data quality report for class balance and missingness summary.',
    )
    figure_manifest, figure_manifest_meta = _load_json_source(
        name='figure_manifest',
        candidates=[
            os.path.join(report_dir, 'figure_manifest.json'),
            os.path.join('output', 'final_delivery', resolved_run_id, 'report', 'figure_manifest.json'),
            os.path.join('phase3', 'figure_manifest.json'),
            os.path.join('output', 'reports', 'figure_manifest.json'),
        ],
        authority_note='Prefer run-specific figure manifest first, then fall back to the latest mirror under output/reports.',
    )
    if not figure_manifest:
        audit_manifest, audit_manifest_meta = _load_json_source(
            name='figure_table_audit_manifest',
            candidates=[
                os.path.join('output', 'audit', 'figure_table_manifest.json'),
            ],
            authority_note='Historical-run fallback: derive report figure records only from the current run audit inventory and current run delivery assets.',
        )
        if audit_manifest:
            figure_manifest = _build_figure_manifest_from_audit(
                audit_manifest,
                run_root=Path.cwd(),
                run_id=resolved_run_id,
            )
            figure_manifest_meta = {
                'artifact_name': 'figure_manifest',
                'format': 'json',
                'selected_path': audit_manifest_meta.get('selected_path', ''),
                'exists': True,
                'candidates': audit_manifest_meta.get('candidates', []),
                'authority_note': audit_manifest_meta.get('authority_note', ''),
                'fallback_from_audit_manifest': True,
            }
    figure_manifest = _filter_retired_figure_manifest(figure_manifest)
    phase3_summary_markdown, phase3_summary_meta = _load_text_source(
        name='phase3_pipeline_summary',
        candidates=[
            os.path.join(report_dir, 'phase3_pipeline_summary.md'),
            os.path.join('output', 'final_delivery', resolved_run_id, 'report', 'phase3_pipeline_summary.md'),
            os.path.join('phase3', 'phase3_pipeline_summary.md'),
            os.path.join('output', 'reports', 'phase3_pipeline_summary.md'),
        ],
        authority_note='Use the Phase 3 pipeline summary as a human-readable audit trace, not as the primary fact source.',
    )

    phase2_result_meta = {
        'artifact_name': 'phase2_result',
        'format': 'json',
        'selected_path': phase2_path,
        'exists': bool(phase2_path) and os.path.exists(phase2_path),
        'candidates': _ordered_phase2_candidates(
            explicit_path=phase2_result_path or '',
            env_path=os.environ.get('METABOAGENT_PHASE2_RESULT_PATH', '').strip(),
            latest_phase2_path=_find_latest_phase2_result(),
            config_phase2_path=config.get_path('phase2_winner_panel'),
        ),
        'authority_note': 'Prefer run-aligned full Phase 2 result JSON over compact winner-only artifacts.',
    }

    source_registry = {
        'phase0_output': phase0_meta,
        'phase1_panel_scores': phase1_panel_scores_meta,
        'feature_provenance': feature_provenance_meta,
        'phase2_result': phase2_result_meta,
        'phase2_search_summary': phase2_search_summary_meta,
        'phase2_bio_debug': phase2_bio_debug_meta,
        'phase2_clinical_utility': phase2_clinical_utility_meta,
        'phase2_probability_recalibration': phase2_probability_recalibration_meta,
        'phase2_bio_calibration': phase2_bio_calibration_meta,
        'phase2_external_validation': phase2_external_validation_meta,
        'phase0_phase1_test_result': phase0_phase1_test_result_meta,
        'feature_selection_summary': feature_selection_summary_meta,
        'final_selection_summary': final_selection_summary_meta,
        'adaptive_selection_report': adaptive_selection_report_meta,
        'missing_value_metadata': missing_value_metadata_meta,
        'preprocessing_report': preprocessing_report_meta,
        'preprocessing_decision_pack': preprocessing_decision_pack_meta,
        'modeling_readiness_report': modeling_readiness_report_meta,
        'data_analysis_for_modeling': data_analysis_for_modeling_meta,
        'figure_manifest': figure_manifest_meta,
        'phase3_pipeline_summary': phase3_summary_meta,
    }

    selected_paths = {
        name: meta.get('selected_path', '')
        for name, meta in source_registry.items()
    }

    missing_sources = sorted(
        name for name, meta in source_registry.items()
        if not meta.get('exists', False)
    )

    return {
        'config_path': config_path,
        'run_id': resolved_run_id,
        'report_dir': report_dir,
        'paths': selected_paths,
        'source_registry': source_registry,
        'collection_summary': {
            'total_sources': len(source_registry),
            'available_sources': sum(1 for meta in source_registry.values() if meta.get('exists')),
            'missing_sources': missing_sources,
            'phase1_artifact_dirs_considered': artifact_dirs,
            'phase1_intermediate_dirs_considered': intermediate_dirs,
        },
        'phase0': phase0_payload,
        'phase1': {
            'panel_scores': phase1_panel_scores,
            'feature_provenance': feature_provenance,
            'feature_selection_summary': feature_selection_summary,
            'final_selection_summary': final_selection_summary,
            'adaptive_selection_report': adaptive_selection_report,
            'phase0_phase1_test_result': phase0_phase1_test_result,
            'missing_value_metadata': missing_value_metadata,
            'preprocessing_report': preprocessing_report,
            'preprocessing_decision_pack': preprocessing_decision_pack,
            'modeling_readiness_report': modeling_readiness_report,
            'data_analysis_for_modeling': data_analysis_for_modeling,
        },
        'phase2': {
            'result': phase2_payload,
            'search_summary': phase2_search_summary,
            'bio_debug': phase2_bio_debug,
            'clinical_utility': phase2_clinical_utility,
            'probability_recalibration': phase2_probability_recalibration,
            'bio_calibration': phase2_bio_calibration,
            'external_validation': phase2_external_validation,
        },
        'phase3': {
            'figure_manifest': figure_manifest,
            'pipeline_summary_markdown': phase3_summary_markdown,
        },
    }
