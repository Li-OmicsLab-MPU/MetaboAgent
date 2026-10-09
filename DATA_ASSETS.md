# Local knowledge assets and demo data

MetaboAgent uses generated reference maps and optional vector databases. They are
not included in this source package because several are large, generated from
third-party databases, or unsuitable for normal Git history.

The repository includes one small, de-identified reviewer demo at
`demo_data/liver_cancer_demo.csv`. It is used by the frontend's **Load demo
dataset** action and is not a replacement for the reference assets below.

Place the following generated files under `storage/` when enabling the complete
Phase 0 and knowledge-driven feature workflow:

```text
storage/disease_map.json
storage/pathbank_pathway_map.json
storage/pathbank_pathway_map_reverse.json
storage/taxonomy_map.json
storage/hmdb_reaction_graph.json
storage/reaction_class_info.json
storage/metabolite_lookup.json
storage/shorthand_map.json
storage/disease_pathway_map.json
storage/metabolite_context_map.json
storage/engineered_feature_rules.json
```

Optional local vector stores are expected under:

```text
storage/hmdb_chroma/
storage/pathbank_chroma/
storage/mapping_chroma/
storage/sop_chroma/
```

The builders in `src/data_ops/` can regenerate many of these assets from locally
obtained HMDB, PathBank, KEGG, or Reactome source data. Review the upstream data
licenses before redistributing derived assets.

For a private deployment, these assets may also be copied from an existing
MetaboAgent installation. Keep caches, run registries, memory stores, uploaded
datasets, and prior analysis outputs out of the public repository.
