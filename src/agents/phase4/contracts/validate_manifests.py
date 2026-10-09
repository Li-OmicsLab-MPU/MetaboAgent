#!/usr/bin/env python3
"""Validate Phase 4 report contracts without requiring third-party packages.

The JSON files in this directory are JSON Schema instances.  This validator adds
the cross-file rules that a generic JSON Schema validator cannot express easily:
artifact references, phase coverage, figure-to-claim bindings and template slots.
If ``jsonschema`` is installed, callers may also run their preferred Draft 2020-12
validator against the schema files; this script remains usable in the minimal
MetaboAgent runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


PHASES = {"phase0", "phase1", "phase2", "phase3", "phase4"}
SCOPES = {"raw", "train_only", "development_cv", "internal_holdout", "external", "mixed"}
FIGURE_MAIN_ROLES = {"main_figure"}


class ContractError(ValueError):
    pass


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"root must be an object: {path}")
    return value


def require(value: Any, field: str, errors: List[str], path: str) -> None:
    if value is None or value == "" or value == []:
        errors.append(f"{path}: missing required field '{field}'")


def unique_ids(items: Iterable[Dict[str, Any]], key: str, label: str, errors: List[str]) -> Set[str]:
    seen: Set[str] = set()
    for index, item in enumerate(items):
        identifier = str(item.get(key, ""))
        if not identifier:
            errors.append(f"{label}[{index}]: missing {key}")
        elif identifier in seen:
            errors.append(f"{label}[{index}]: duplicate {key} '{identifier}'")
        seen.add(identifier)
    return seen


def validate_flow(flow: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if flow.get("schema_version") != "metaboagent.phase_flow.v1":
        errors.append("phase_flow: unsupported schema_version")
    for field in ("run_id", "study", "artifacts", "nodes", "edges", "policy"):
        require(flow.get(field), field, errors, "phase_flow")

    artifacts = flow.get("artifacts") if isinstance(flow.get("artifacts"), list) else []
    nodes = flow.get("nodes") if isinstance(flow.get("nodes"), list) else []
    edges = flow.get("edges") if isinstance(flow.get("edges"), list) else []
    artifact_ids = unique_ids(artifacts, "id", "phase_flow.artifacts", errors)
    node_ids = unique_ids(nodes, "id", "phase_flow.nodes", errors)

    for index, artifact in enumerate(artifacts):
        scope = artifact.get("scope")
        if scope not in SCOPES:
            errors.append(f"phase_flow.artifacts[{index}]: invalid scope '{scope}'")
        if artifact.get("status") != "not_archived" and not artifact.get("path"):
            errors.append(f"phase_flow.artifacts[{index}]: path required unless status=not_archived")

    seen_phases = set()
    for index, node in enumerate(nodes):
        phase = node.get("phase")
        if phase not in PHASES:
            errors.append(f"phase_flow.nodes[{index}]: invalid phase '{phase}'")
        else:
            seen_phases.add(phase)
        for field in ("inputs", "outputs"):
            refs = node.get(field) if isinstance(node.get(field), list) else []
            for ref in refs:
                if ref not in artifact_ids:
                    errors.append(f"phase_flow.nodes[{index}].{field}: unknown artifact '{ref}'")
        if node.get("sample_scope") not in SCOPES:
            errors.append(f"phase_flow.nodes[{index}]: invalid sample_scope")
        if node.get("fit_scope") not in {"none", "train_only", "within_training_fold", "full_development", "external_only", "mixed"}:
            errors.append(f"phase_flow.nodes[{index}]: invalid fit_scope")
        summary = node.get("stage_summary")
        if summary is not None:
            if not isinstance(summary, dict) or not summary.get("headline") or not isinstance(summary.get("items"), list) or not summary.get("items"):
                errors.append(f"phase_flow.nodes[{index}]: stage_summary requires headline and at least one item")
            else:
                for item_index, item in enumerate(summary.get("items", [])):
                    if not isinstance(item, dict) or not item.get("label") or not item.get("value"):
                        errors.append(f"phase_flow.nodes[{index}].stage_summary.items[{item_index}]: label and value are required")
    missing_phases = PHASES - seen_phases
    if missing_phases:
        errors.append(f"phase_flow: missing phase nodes: {', '.join(sorted(missing_phases))}")

    for index, edge in enumerate(edges):
        if edge.get("from") not in node_ids:
            errors.append(f"phase_flow.edges[{index}]: unknown from node '{edge.get('from')}'")
        if edge.get("to") not in node_ids:
            errors.append(f"phase_flow.edges[{index}]: unknown to node '{edge.get('to')}'")
        if edge.get("artifact_id") not in artifact_ids:
            errors.append(f"phase_flow.edges[{index}]: unknown artifact '{edge.get('artifact_id')}'")

    policy = flow.get("policy") if isinstance(flow.get("policy"), dict) else {}
    if policy.get("phase_order") != ["phase0", "phase1", "phase2", "phase3", "phase4"]:
        errors.append("phase_flow.policy.phase_order must be phase0 through phase4")
    return errors


def validate_template(template: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if template.get("schema_version") != "metaboagent.report_template.v1":
        errors.append("report_template: unsupported schema_version")
    sections = template.get("sections") if isinstance(template.get("sections"), list) else []
    section_ids = unique_ids(sections, "section_id", "report_template.sections", errors)
    orders = [section.get("order") for section in sections]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        errors.append("report_template.sections: order values must be unique and ascending")

    claims: Set[str] = set()
    figure_slots: Dict[str, str] = {}
    for index, section in enumerate(sections):
        for claim_id in section.get("required_claim_ids", []) or []:
            claims.add(str(claim_id))
        for slot in section.get("figure_slots", []) or []:
            slot_id = str(slot.get("slot_id", ""))
            if slot_id in figure_slots:
                errors.append(f"report_template.sections[{index}]: duplicate slot_id '{slot_id}'")
            figure_slots[slot_id] = str(slot.get("figure_id", ""))
            if not slot.get("figure_id"):
                errors.append(f"report_template.sections[{index}]: figure slot missing figure_id")
    if "study_overview" not in section_ids:
        errors.append("report_template: study_overview section is required for the Phase 0-4 flow figure")
    if "limitations" not in section_ids:
        errors.append("report_template: limitations section is required")
    return errors


def validate_figures(figures: Dict[str, Any], flow: Dict[str, Any], template: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if figures.get("schema_version") != "metaboagent.figure_manifest.v1":
        errors.append("figure_manifest: unsupported schema_version")
    items = figures.get("figures") if isinstance(figures.get("figures"), list) else []
    figure_ids = unique_ids(items, "figure_id", "figure_manifest.figures", errors)
    artifact_ids = {str(item.get("id")) for item in flow.get("artifacts", []) if isinstance(item, dict)}
    sections = {str(item.get("section_id")): item for item in template.get("sections", []) if isinstance(item, dict)}
    template_claims = {
        str(claim)
        for section in template.get("sections", [])
        if isinstance(section, dict)
        for claim in section.get("required_claim_ids", []) or []
    }
    template_figures = {
        str(slot.get("figure_id"))
        for section in template.get("sections", [])
        if isinstance(section, dict)
        for slot in section.get("figure_slots", []) or []
    }

    for index, item in enumerate(items):
        prefix = f"figure_manifest.figures[{index}]"
        for artifact_id in item.get("source_artifacts", []) or []:
            if artifact_id not in artifact_ids:
                errors.append(f"{prefix}: unknown source artifact '{artifact_id}'")
        for claim_id in item.get("claim_ids", []) or []:
            if claim_id not in template_claims:
                errors.append(f"{prefix}: claim '{claim_id}' is not declared by report template")
        placement = item.get("placement") if isinstance(item.get("placement"), dict) else {}
        section_id = placement.get("section_id")
        if section_id not in sections:
            errors.append(f"{prefix}: placement references unknown section '{section_id}'")
        if item.get("figure_id") not in template_figures:
            errors.append(f"{prefix}: figure is not bound to a template figure slot")
        anchor = placement.get("anchor_after_claim")
        if item.get("publication_role") in FIGURE_MAIN_ROLES:
            if not anchor:
                errors.append(f"{prefix}: main figure requires anchor_after_claim")
            elif anchor not in template_claims:
                errors.append(f"{prefix}: anchor claim '{anchor}' is not declared by report template")
        if item.get("cohort_scope") == "external":
            external_sources = [
                artifact_id for artifact_id in item.get("source_artifacts", []) or []
                if any(
                    isinstance(artifact, dict)
                    and artifact.get("id") == artifact_id
                    and artifact.get("scope") == "external"
                    for artifact in flow.get("artifacts", [])
                )
            ]
            if not external_sources:
                errors.append(f"{prefix}: external figure has no explicitly external source artifact")
        caption = item.get("caption_contract") if isinstance(item.get("caption_contract"), dict) else {}
        if len(caption.get("required_fields", []) or []) < 3:
            errors.append(f"{prefix}: caption contract must require at least three fields")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MetaboAgent Phase 4 report contracts")
    parser.add_argument("--flow", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    args = parser.parse_args()

    try:
        flow = load_json(args.flow)
        figures = load_json(args.figures)
        template = load_json(args.template)
    except ContractError as exc:
        print(f"FAIL: {exc}")
        return 1

    errors = []
    errors.extend(validate_flow(flow))
    errors.extend(validate_template(template))
    errors.extend(validate_figures(figures, flow, template))
    if errors:
        print(f"FAIL: {len(errors)} contract error(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: phase flow, figure manifest and report template contracts are consistent")
    print(f"  phases: {len({node.get('phase') for node in flow.get('nodes', [])})}/5")
    print(f"  artifacts: {len(flow.get('artifacts', []))}")
    print(f"  figures: {len(figures.get('figures', []))}")
    print(f"  sections: {len(template.get('sections', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
