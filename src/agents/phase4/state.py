"""
Phase 4 state definitions.

Phase 4 is intentionally artifact-centric: it collects structured outputs from
Phase 0-3, normalizes them into a stable report context, then renders report
artifacts without recomputing scientific facts.
"""

from typing import Any, Dict, List, Optional, TypedDict


class Phase4State(TypedDict):
    """Typed state for the Phase 4 reporting pipeline."""

    config_path: str
    run_id: str
    report_dir: str
    collected_artifacts: Dict[str, Any]
    report_context: Optional[Dict[str, Any]]
    llm_sections: Optional[Dict[str, Any]]
    rendered_outputs: Dict[str, str]
    error: Optional[str]
    completed: bool
    messages: List[str]


def create_initial_phase4_state(
    config_path: str,
    run_id: str,
    report_dir: str,
) -> Phase4State:
    """Create an empty Phase 4 state container."""

    return {
        'config_path': config_path,
        'run_id': run_id,
        'report_dir': report_dir,
        'collected_artifacts': {},
        'report_context': None,
        'llm_sections': None,
        'rendered_outputs': {},
        'error': None,
        'completed': False,
        'messages': [],
    }
