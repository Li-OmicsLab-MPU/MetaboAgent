"""Phase 4 reporting pipeline.

Phase 4 is a read-only consumer of Phase 0-3 artifacts.  It produces the legacy
structured report outputs plus the v2 Evidence-to-Decision report, quality gate
and final-delivery refresh without recomputing scientific results.
"""

import os
import re
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from .collector import collect_phase_artifacts
from .evidence_report import (
    build_evidence_bundle,
    refresh_final_delivery,
    render_evidence_html,
    render_evidence_pdf,
    write_evidence_bundle,
)
from .normalizer import build_report_context
from .renderer import render_markdown
from .state import Phase4State, create_initial_phase4_state
from .writer import generate_llm_sections, write_llm_sections, write_report_context


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _enforce_run_root_sources(collected: Dict[str, Any], run_root: Path) -> None:
    """Reject accidental fallback to artifacts belonging to another run."""

    payload_locations = {
        'phase0_output': ('', 'phase0'),
        'phase1_panel_scores': ('phase1', 'panel_scores'),
        'feature_provenance': ('phase1', 'feature_provenance'),
        'feature_selection_summary': ('phase1', 'feature_selection_summary'),
        'final_selection_summary': ('phase1', 'final_selection_summary'),
        'adaptive_selection_report': ('phase1', 'adaptive_selection_report'),
        'missing_value_metadata': ('phase1', 'missing_value_metadata'),
        'preprocessing_report': ('phase1', 'preprocessing_report'),
        'preprocessing_decision_pack': ('phase1', 'preprocessing_decision_pack'),
        'modeling_readiness_report': ('phase1', 'modeling_readiness_report'),
        'data_analysis_for_modeling': ('phase1', 'data_analysis_for_modeling'),
        'phase2_result': ('phase2', 'result'),
        'phase2_search_summary': ('phase2', 'search_summary'),
        'phase2_bio_debug': ('phase2', 'bio_debug'),
        'phase2_clinical_utility': ('phase2', 'clinical_utility'),
        'phase2_probability_recalibration': ('phase2', 'probability_recalibration'),
        'phase2_bio_calibration': ('phase2', 'bio_calibration'),
        'phase2_external_validation': ('phase2', 'external_validation'),
        'figure_manifest': ('phase3', 'figure_manifest'),
        'phase3_pipeline_summary': ('phase3', 'pipeline_summary_markdown'),
        'phase0_phase1_test_result': ('phase1', 'phase0_phase1_test_result'),
    }
    rejected: Dict[str, str] = {}
    registry = collected.get('source_registry', {})
    for name, meta_value in list(registry.items()):
        meta = meta_value if isinstance(meta_value, dict) else {}
        selected = str(meta.get('selected_path') or '').strip()
        if not selected or selected.startswith('embedded://'):
            continue
        candidate = Path(selected)
        candidate = candidate if candidate.is_absolute() else run_root / candidate
        if _is_within(candidate, run_root):
            continue
        rejected[name] = str(candidate)
        meta['exists'] = False
        meta['rejected_by_run_root_guard'] = True
        meta['authority_note'] = (
            str(meta.get('authority_note') or '')
            + ' Rejected because the selected artifact is outside the explicit Phase 4 run root.'
        ).strip()
        registry[name] = meta
        location = payload_locations.get(name)
        if location:
            section, key = location
            if not section:
                collected[key] = {}
            elif isinstance(collected.get(section), dict):
                collected[section][key] = '' if name == 'phase3_pipeline_summary' else {}
        if isinstance(collected.get('paths'), dict):
            collected['paths'][name] = ''
    summary = collected.setdefault('collection_summary', {})
    summary['strict_run_root'] = str(run_root)
    summary['rejected_out_of_root_sources'] = rejected
    summary['missing_sources'] = sorted(
        name for name, meta in registry.items()
        if not isinstance(meta, dict) or not meta.get('exists', False)
    )


class Phase4Pipeline:
    """Orchestrates Phase 4 collection, normalization, writing, and rendering."""

    def __init__(
        self,
        config_path: str = 'config.yaml',
        phase2_result_path: Optional[str] = None,
        run_root: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        # Preserve the caller's project root before Phase 4 changes cwd to the
        # run root.  User/config paths are often project-relative, whereas
        # run-local artifacts are naturally run-root-relative.
        self.project_root = Path.cwd().resolve()
        self.config_path = str(Path(config_path).resolve())
        self.phase2_result_path = phase2_result_path
        self.run_root = Path(
            run_root
            or os.environ.get('METABOAGENT_RUNTIME_ROOT', '').strip()
            or Path.cwd()
        ).resolve()
        self.run_id = str(run_id or os.environ.get('METABOAGENT_RUN_TAG', '') or '').strip()
        self.state: Optional[Phase4State] = None

    def _resolve_phase2_result(self) -> Optional[str]:
        candidates = [
            self.phase2_result_path,
            os.environ.get('METABOAGENT_PHASE2_RESULT_PATH', '').strip(),
            str(self.run_root / 'output' / 'phase1' / 'artifacts' / 'phase2_winner_scores.json'),
            str(self.run_root / 'phase1' / 'artifacts' / 'phase2_winner_scores.json'),
            str(self.run_root / 'phase1' / 'legacy' / 'artifacts' / 'phase2_winner_scores.json'),
            str(self.run_root / 'output' / 'artifacts' / 'phase2_winner_scores.json'),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            raw = Path(candidate).expanduser()
            variants = [raw]
            if not raw.is_absolute():
                # The first two variants cover calls made before and after
                # entering the run root.  The project-root variant is needed
                # for paths such as Evaluate/<run>/phase1/legacy/artifacts/.
                variants.extend([
                    self.project_root / raw,
                    self.run_root / raw,
                ])
            for variant in variants:
                if variant.exists() and variant.is_file():
                    return str(variant.resolve())
        return self.phase2_result_path

    def run(
        self,
        render_html_output: bool = True,
        render_pdf_output: bool = True,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        if not self.run_root.exists():
            raise FileNotFoundError(f'Phase 4 run root does not exist: {self.run_root}')

        with _working_directory(self.run_root):
            # Collector runtime candidates (especially Phase 0) are resolved
            # from this explicit run root.  Older callers did not export the
            # environment variable, causing the collector to skip
            # ``<run_root>/phase0/phase0_output_latest.json`` and silently
            # normalize an empty Phase 0 payload even when the atlas existed.
            previous_runtime_root = os.environ.get('METABOAGENT_RUNTIME_ROOT')
            os.environ['METABOAGENT_RUNTIME_ROOT'] = str(self.run_root)
            try:
                collected = collect_phase_artifacts(
                    config_path=self.config_path,
                    phase2_result_path=self._resolve_phase2_result(),
                    run_id=self.run_id,
                )
                _enforce_run_root_sources(collected, self.run_root)
            finally:
                if previous_runtime_root is None:
                    os.environ.pop('METABOAGENT_RUNTIME_ROOT', None)
                else:
                    os.environ['METABOAGENT_RUNTIME_ROOT'] = previous_runtime_root

            # The scientific run identity is immutable.  Report rebuilds are
            # releases of that run, not new scientific runs.  Rendering first
            # into a private staging directory and then replacing ``current``
            # prevents schema/date suffixes from creating an ever-growing set
            # of apparently competing reports.
            run_id = self.run_id or str(collected.get('run_id') or '').strip() or 'phase4-report'
            report_channel = re.sub(
                r'[^A-Za-z0-9_.-]+', '-',
                os.environ.get('METABOAGENT_REPORT_CHANNEL', 'current').strip() or 'current',
            ).strip('.-') or 'current'
            report_build_id = f"report-{uuid.uuid4().hex[:12]}"
            reports_root = self.run_root / 'output' / 'reports'
            canonical_report_dir = reports_root / report_channel
            staging_report_dir = reports_root / '.staging' / report_build_id
            report_dir = str(staging_report_dir)
            collected['run_id'] = run_id
            collected['report_dir'] = report_dir

            self.state = create_initial_phase4_state(
                config_path=self.config_path,
                run_id=run_id,
                report_dir=report_dir,
            )
            self.state['collected_artifacts'] = collected

            report_context = build_report_context(collected)
            report_context['schema_version'] = 'phase4.report_context.v2'
            report_context.setdefault('run', {})['run_id'] = run_id
            report_context['run']['report_build_id'] = report_build_id
            report_context['run']['report_channel'] = report_channel
            report_context['run']['report_dir'] = str(canonical_report_dir)
            report_context['run']['run_root'] = str(self.run_root)
            llm_sections = generate_llm_sections(
                report_context=report_context,
                config_path=self.config_path,
                use_llm=use_llm,
            )

            report_context_path = write_report_context(report_context, report_dir)
            llm_sections_path = write_llm_sections(report_context, report_dir, llm_sections)
            markdown_path = render_markdown(report_context, llm_sections, report_dir)

            evidence_bundle = build_evidence_bundle(
                report_context=report_context,
                collected=collected,
                run_root=str(self.run_root),
            )
            evidence_paths = write_evidence_bundle(evidence_bundle, report_context, report_dir)

            rendered_outputs = {
                'report_context': report_context_path,
                'llm_sections': llm_sections_path,
                'markdown': markdown_path,
                **evidence_paths,
            }

            if render_html_output:
                rendered_outputs['html'] = render_evidence_html(
                    report_context=report_context,
                    llm_sections=llm_sections,
                    bundle=evidence_bundle,
                    report_dir=report_dir,
                    run_root=str(self.run_root),
                )

            if render_pdf_output:
                pdf_path = render_evidence_pdf(report_context, evidence_bundle, report_dir)
                if pdf_path:
                    rendered_outputs['pdf'] = pdf_path
                    # Publish the PDF parity manifest alongside the PDF.  It
                    # is a root-level report artifact, so exposing it through
                    # rendered_outputs ensures refresh_final_delivery copies
                    # it into the fixed current/report/ directory as well.
                    pdf_manifest_path = Path(pdf_path).with_name('pdf_content_manifest.json')
                    if pdf_manifest_path.exists():
                        rendered_outputs['pdf_content_manifest'] = str(pdf_manifest_path)

            canonical_report_dir.parent.mkdir(parents=True, exist_ok=True)
            previous_report_dir = reports_root / '.previous-current'
            if previous_report_dir.exists():
                shutil.rmtree(previous_report_dir)
            if canonical_report_dir.exists():
                canonical_report_dir.replace(previous_report_dir)
            staging_report_dir.replace(canonical_report_dir)
            if previous_report_dir.exists():
                shutil.rmtree(previous_report_dir)
            rendered_outputs = {
                key: str(canonical_report_dir / Path(value).relative_to(staging_report_dir))
                if Path(value).is_relative_to(staging_report_dir) else value
                for key, value in rendered_outputs.items()
            }

            delivery = refresh_final_delivery(
                run_root=str(self.run_root),
                run_id=run_id,
                delivery_id=report_channel,
                report_build_id=report_build_id,
                report_outputs=rendered_outputs,
                report_context=report_context,
            )
            rendered_outputs.update(delivery)

        self.state['report_context'] = report_context
        self.state['llm_sections'] = llm_sections
        self.state['rendered_outputs'] = rendered_outputs
        self.state['completed'] = True
        return rendered_outputs


def generate_phase4_report(
    config_path: str = 'config.yaml',
    phase2_result_path: Optional[str] = None,
    run_root: Optional[str] = None,
    run_id: Optional[str] = None,
    render_html_output: bool = True,
    render_pdf_output: bool = True,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper for running the Phase 4 pipeline."""

    pipeline = Phase4Pipeline(
        config_path=config_path,
        phase2_result_path=phase2_result_path,
        run_root=run_root,
        run_id=run_id,
    )
    return pipeline.run(
        render_html_output=render_html_output,
        render_pdf_output=render_pdf_output,
        use_llm=use_llm,
    )
