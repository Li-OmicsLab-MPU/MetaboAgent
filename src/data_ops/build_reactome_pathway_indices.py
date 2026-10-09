#!/usr/bin/env python3
"""
Build Reactome pathway indices for downstream disease-pathway retrieval.

Inputs
- storage/ReactomePathways.txt
- storage/ReactomePathwaysRelation.txt
- storage/ChEBI2Reactome_All_Levels.txt

Outputs
- storage/external_pathways/reactome/reactome_human_pathways.json
- storage/external_pathways/reactome/reactome_pathway_hierarchy.json
- storage/external_pathways/reactome/reactome_small_molecule_support.json
- storage/external_pathways/reactome/reactome_processing_manifest.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "storage"
OUTPUT_DIR = PROJECT_ROOT / "storage" / "external_pathways" / "reactome"

PATHWAYS_TXT = RAW_DIR / "ReactomePathways.txt"
RELATIONS_TXT = RAW_DIR / "ReactomePathwaysRelation.txt"
CHEBI_MAP_TXT = RAW_DIR / "ChEBI2Reactome_All_Levels.txt"

HUMAN_PATHWAYS_JSON = OUTPUT_DIR / "reactome_human_pathways.json"
HIERARCHY_JSON = OUTPUT_DIR / "reactome_pathway_hierarchy.json"
SMALL_MOLECULE_JSON = OUTPUT_DIR / "reactome_small_molecule_support.json"
MANIFEST_JSON = OUTPUT_DIR / "reactome_processing_manifest.json"

HOMO_SAPIENS = "Homo sapiens"


def _read_tsv(path: Path) -> Iterable[List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def load_human_pathways(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    ReactomePathways.txt columns:
    pathway_id, pathway_name, species
    """
    pathways: Dict[str, Dict[str, Any]] = {}
    for row in _read_tsv(path):
        if len(row) < 3:
            continue
        pathway_id, pathway_name, species = row[0].strip(), _normalize_text(row[1]), _normalize_text(row[2])
        if species != HOMO_SAPIENS:
            continue
        pathways[pathway_id] = {
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
            "species": species,
            "source_db": "Reactome",
            "source_url": f"https://reactome.org/PathwayBrowser/#/{pathway_id}",
        }
    return pathways


def load_human_hierarchy(path: Path, valid_pathway_ids: Set[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Any]]:
    """
    ReactomePathwaysRelation.txt columns:
    parent_pathway_id, child_pathway_id
    """
    parent_to_children: Dict[str, Set[str]] = defaultdict(set)
    child_to_parents: Dict[str, Set[str]] = defaultdict(set)
    skipped_non_human = 0

    for row in _read_tsv(path):
        if len(row) < 2:
            continue
        parent_id, child_id = row[0].strip(), row[1].strip()
        if parent_id not in valid_pathway_ids or child_id not in valid_pathway_ids:
            skipped_non_human += 1
            continue
        parent_to_children[parent_id].add(child_id)
        child_to_parents[child_id].add(parent_id)

    parent_to_children_sorted = {
        pathway_id: sorted(children)
        for pathway_id, children in sorted(parent_to_children.items())
    }
    child_to_parents_sorted = {
        pathway_id: sorted(parents)
        for pathway_id, parents in sorted(child_to_parents.items())
    }
    stats = {
        "parent_node_count": len(parent_to_children_sorted),
        "child_node_count": len(child_to_parents_sorted),
        "edge_count": sum(len(children) for children in parent_to_children_sorted.values()),
        "skipped_non_human_edges": skipped_non_human,
    }
    return parent_to_children_sorted, child_to_parents_sorted, stats


def load_small_molecule_support(path: Path, valid_pathway_ids: Set[str], pathway_name_map: Dict[str, str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    ChEBI2Reactome_All_Levels.txt columns:
    chebi_id, pathway_id, pathway_url, pathway_name, evidence_code, species
    """
    pathway_to_chebi_ids: Dict[str, Set[str]] = defaultdict(set)
    pathway_to_evidence: Dict[str, Set[str]] = defaultdict(set)
    skipped_non_human = 0

    for row in _read_tsv(path):
        if len(row) < 6:
            continue
        chebi_id, pathway_id, _url, pathway_name, evidence_code, species = (
            row[0].strip(),
            row[1].strip(),
            row[2].strip(),
            _normalize_text(row[3]),
            row[4].strip(),
            _normalize_text(row[5]),
        )
        if species != HOMO_SAPIENS or pathway_id not in valid_pathway_ids:
            skipped_non_human += 1
            continue
        pathway_to_chebi_ids[pathway_id].add(chebi_id)
        if evidence_code:
            pathway_to_evidence[pathway_id].add(evidence_code)
        if pathway_id not in pathway_name_map and pathway_name:
            pathway_name_map[pathway_id] = pathway_name

    support_map: Dict[str, Dict[str, Any]] = {}
    for pathway_id in sorted(valid_pathway_ids):
        chebi_ids = sorted(pathway_to_chebi_ids.get(pathway_id, set()))
        evidence_codes = sorted(pathway_to_evidence.get(pathway_id, set()))
        support_map[pathway_id] = {
            "pathway_id": pathway_id,
            "pathway_name": pathway_name_map.get(pathway_id),
            "has_small_molecule_support": bool(chebi_ids),
            "small_molecule_count": len(chebi_ids),
            "chebi_ids": chebi_ids,
            "evidence_codes": evidence_codes,
            "species": HOMO_SAPIENS,
            "source_db": "Reactome",
        }

    stats = {
        "pathways_with_small_molecule_support": sum(
            1 for item in support_map.values() if item["has_small_molecule_support"]
        ),
        "total_supported_pathways": len(support_map),
        "total_unique_chebi_ids": len({chebi_id for item in support_map.values() for chebi_id in item["chebi_ids"]}),
        "skipped_non_human_rows": skipped_non_human,
    }
    return support_map, stats


def build_reactome_indices() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    human_pathways = load_human_pathways(PATHWAYS_TXT)
    valid_pathway_ids = set(human_pathways.keys())
    pathway_name_map = {pathway_id: item["pathway_name"] for pathway_id, item in human_pathways.items()}

    parent_to_children, child_to_parents, hierarchy_stats = load_human_hierarchy(RELATIONS_TXT, valid_pathway_ids)
    small_molecule_support, small_molecule_stats = load_small_molecule_support(CHEBI_MAP_TXT, valid_pathway_ids, pathway_name_map)

    hierarchy_payload = {
        "parent_to_children": parent_to_children,
        "child_to_parents": child_to_parents,
        "stats": hierarchy_stats,
    }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "reactome_pathways_txt": str(PATHWAYS_TXT),
            "reactome_relations_txt": str(RELATIONS_TXT),
            "chebi2reactome_all_levels_txt": str(CHEBI_MAP_TXT),
        },
        "outputs": {
            "reactome_human_pathways_json": str(HUMAN_PATHWAYS_JSON),
            "reactome_pathway_hierarchy_json": str(HIERARCHY_JSON),
            "reactome_small_molecule_support_json": str(SMALL_MOLECULE_JSON),
        },
        "stats": {
            "human_pathway_count": len(human_pathways),
            **hierarchy_stats,
            **small_molecule_stats,
        },
        "notes": [
            "Hierarchy only keeps edges where both parent and child are Homo sapiens Reactome pathways.",
            "Small molecule support is derived from ChEBI2Reactome_All_Levels and filtered to Homo sapiens.",
        ],
    }

    with open(HUMAN_PATHWAYS_JSON, "w", encoding="utf-8") as handle:
        json.dump(human_pathways, handle, indent=2, ensure_ascii=False)

    with open(HIERARCHY_JSON, "w", encoding="utf-8") as handle:
        json.dump(hierarchy_payload, handle, indent=2, ensure_ascii=False)

    with open(SMALL_MOLECULE_JSON, "w", encoding="utf-8") as handle:
        json.dump(small_molecule_support, handle, indent=2, ensure_ascii=False)

    with open(MANIFEST_JSON, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    manifest = build_reactome_indices()
    stats = manifest["stats"]
    print("=" * 72)
    print("Reactome Pathway Indices Builder")
    print("=" * 72)
    print(f"Human pathways: {stats['human_pathway_count']}")
    print(f"Hierarchy edges: {stats['edge_count']}")
    print(f"Pathways with small-molecule support: {stats['pathways_with_small_molecule_support']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 72)
