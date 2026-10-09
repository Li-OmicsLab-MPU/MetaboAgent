# MetaboAgent publication report architecture

## Single source of truth

- `report_context_v2.json` contains archived scientific facts only.
- `phase_flow_manifest.json` defines Phase 0–4 lineage and stage counts.
- `figure_manifest.json` defines figure provenance, scope, claim binding, and placement.
- `report_template.json` defines the cross-dataset narrative order and release gates.
- `metric_registry.json` is the only source for reader-facing performance values.
- `statistical_claim_validation.json` verifies CI/null semantics and threshold metrics before publication.
- `feature_identity_registry.json` is the single identity authority for narrative, tables, SHAP, RCS, and figures.
- `evidence_maturity.json` separates validation maturity from artifact integrity.
- `prevalence_scenarios.json` contains deterministic projections, explicitly distinct from observed validation.
- `reporting_readiness.json` is an internal TRIPOD+AI/PROBAST+AI readiness screen, not a formal appraisal.
- `upstream_evidence_requirements.json` records gaps that cannot be repaired by Phase 4 prose or layout.
- LLM prose may explain these records, but must not create metrics, validation scope, intended use, identities, or causal claims.

## Release model

- The scientific `run_id` is immutable and is never changed by report regeneration.
- Every render receives a separate ephemeral `report_build_id`.
- Rendering occurs in `output/reports/.staging/<report_build_id>`.
- A successful render atomically replaces `output/reports/current`.
- The delivery atomically replaces `output/final_delivery/current`.
- `output/final_delivery/latest.json` is the canonical machine-readable pointer.
- Historical schema/date suffixes are legacy outputs; new Phase 4 runs do not create them.

## Reader-facing narrative order

1. Executive evidence summary
2. Study design and Phase 0–4 flow
3. Cohort, preprocessing, and QC
4. Prior evidence and final-panel biomarker evidence
5. Model development and selection
6. Performance, calibration, and uncertainty
7. Exploratory clinical utility, when archived
8. Evidence-scoped biological interpretation
9. Limitations
10. Collapsed technical appendix (claim graph, supplementary figures, audit, artifacts)

Main figures appear once beside their scientific argument. Supplementary figures remain available in the collapsed appendix and are not repeated in a gallery.

## Changes that Phase 4 can make directly

- Reorder, condense, and style the report.
- Enforce terminology, validation-scope, identifier, and metric consistency.
- Join archived HMDB identity, selection frequency, method consensus, and train-only effect-size evidence.
- Build the Phase 0–4 flow from run-scoped counts.
- Bind and place archived figures through the figure manifest.
- Suppress unsupported LLM prose and expose unavailable evidence.
- Package portable evidence and enforce the deterministic quality gate.
- Project PPV, NPV, positive tests, and false positives across declared prevalence scenarios without refitting the model.
- Prevent a configured optimization threshold from being described as a recommended or validated clinical threshold.

## Release-gate semantics

- `PASS` means the report package satisfies deterministic artifact, contract, identity, and numerical-claim checks.
- It is displayed as **Artifact integrity**, never as an overall scientific-quality verdict.
- Evidence maturity is reported separately as Level I (internal resampling), Level II (locked internal holdout), or Level III (independent external evaluation).
- A report may pass artifact integrity while remaining development-only, analytically partial, or clinically exploratory.

## Evidence that requires upstream analysis changes

Phase 4 must not manufacture the following. Each item requires a Phase 0–3 artifact and, where applicable, reanalysis:

- Independent external-cohort validation and transportability claims.
- A fully documented nested-resampling proof for preprocessing, feature selection, tuning, and model selection.
- End-to-end repeated-seed panel stability (Phase 1 frequency alone is insufficient).
- Calibration slope/intercept confidence intervals and optimism-corrected calibration.
- A prespecified clinical-only comparator with delta-AUROC/DeLong and clinically justified incremental-value analyses.
- Target-population prevalence, recalibration, and prospective decision-curve utility.
- Batch-effect diagnostics and before/after correction figures when batch metadata were not archived.
- Prespecified univariate tests, multiplicity correction, effect-size confidence intervals, and assay-level analytical validation.
- Verified biological citations, pathway enrichment inputs, MSI identification confidence, and causal/mechanistic validation.
- A clinically authored intended-use statement and target population.

These items should be represented as explicit `not_archived`, `not_assessed`, or `external_validation_required` states until the upstream workflow supplies the necessary evidence.
