# Phase 4 report contracts

These contracts define the stable boundary between archived Phase 0-3 evidence and the reader-facing Phase 4 report.

## Files

- `schemas/phase_flow_manifest.schema.json`: phase-level data lineage, artifact references, sample scope and fit scope.
- `schemas/figure_manifest.schema.json`: figure provenance, claim binding, caption requirements and inline placement.
- `schemas/report_template.schema.json`: cross-dataset section order, figure slots, LLM permissions and release gates.
- `examples/`: a minimal liver-cancer instance showing the expected cross-file references.
- `validate_manifests.py`: dependency-light cross-file validation.

## Contract boundaries

The manifests do not duplicate scientific results. Numeric values remain in the canonical `report_context`/`metric_registry`; manifests describe where evidence comes from, what it can support and where it is rendered.

The LLM should receive validated claims and figure bindings as read-only context. It may write narrative prose, but it must not invent metrics, change evidence scope or create figure IDs.

## Validate the example

```bash
python3 server_worktree/phase4/contracts/validate_manifests.py \
  --flow server_worktree/phase4/contracts/examples/liver_cancer_phase_flow_manifest.json \
  --figures server_worktree/phase4/contracts/examples/liver_cancer_figure_manifest.json \
  --template server_worktree/phase4/contracts/examples/liver_cancer_report_template.json
```

The next integration step is to generate these instances from `report_context` and replace hard-coded figure placement with `figure_id`/`anchor_after_claim` lookup.
