"""
Phase 4 reporting package.

This package assembles Phase 0-3 artifacts into a report-oriented context,
persists Phase 4 JSON outputs, and renders lightweight report documents.
"""

from .collector import collect_phase_artifacts
from .evidence_report import (
    build_evidence_bundle,
    refresh_final_delivery,
    render_evidence_html,
    render_evidence_pdf,
    write_evidence_bundle,
)
from .normalizer import build_report_context
from .pipeline import Phase4Pipeline, generate_phase4_report
from .state import Phase4State, create_initial_phase4_state

__all__ = [
    'collect_phase_artifacts',
    'build_report_context',
    'build_evidence_bundle',
    'write_evidence_bundle',
    'render_evidence_html',
    'render_evidence_pdf',
    'refresh_final_delivery',
    'Phase4Pipeline',
    'generate_phase4_report',
    'Phase4State',
    'create_initial_phase4_state',
]
