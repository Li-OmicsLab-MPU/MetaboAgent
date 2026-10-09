#!/usr/bin/env python3
"""
Build structured KEGG human disease / cancer pathway candidates from local raw files.

Inputs
- storage/kegg/kegg_brite_human_diseases_raw.txt
- storage/kegg/kegg_hsa_pathway_list.tsv
- storage/kegg/kegg_reaction_graph_raw.json (optional, summarized only)

Outputs
- storage/external_pathways/kegg/kegg_processing_manifest.json
- storage/external_pathways/kegg/kegg_human_disease_pathway_candidates.json
- storage/external_pathways/kegg/kegg_human_disease_pathway_details.json
- storage/external_pathways/kegg/kegg_lung_cancer_pathway_candidates.json

This is the KEGG "step 2" processor:
1. Parse the BRITE hierarchy and extract the `AHuman Diseases` subtree
2. Normalize pathway IDs and names with `list/pathway/hsa`
3. Emit structured pathway candidate records for downstream filtering
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "storage" / "kegg"
OUTPUT_DIR = PROJECT_ROOT / "storage" / "external_pathways" / "kegg"

BRITE_RAW_PATH = RAW_DIR / "kegg_brite_human_diseases_raw.txt"
HSA_PATHWAY_LIST_PATH = RAW_DIR / "kegg_hsa_pathway_list.tsv"
REACTION_GRAPH_RAW_PATH = RAW_DIR / "kegg_reaction_graph_raw.json"

MANIFEST_PATH = OUTPUT_DIR / "kegg_processing_manifest.json"
CANDIDATES_PATH = OUTPUT_DIR / "kegg_human_disease_pathway_candidates.json"
DETAILS_PATH = OUTPUT_DIR / "kegg_human_disease_pathway_details.json"
LUNG_CANCER_PATH = OUTPUT_DIR / "kegg_lung_cancer_pathway_candidates.json"


def _read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _clean_hsa_name(raw_name: str) -> str:
    cleaned = _normalize_text(raw_name)
    cleaned = re.sub(r"\s*-\s*Homo sapiens \(human\)\s*$", "", cleaned)
    return cleaned.strip()


def _load_hsa_pathway_list(path: Path) -> Dict[str, str]:
    """Return {hsa05200: 'Pathways in cancer'}."""
    mapping: Dict[str, str] = {}
    for line in _read_lines(path):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        pathway_id = parts[0].strip()
        pathway_name = _clean_hsa_name(parts[1])
        if pathway_id and pathway_name:
            mapping[pathway_id] = pathway_name
    return mapping


def _parse_brite_human_diseases(lines: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Parse the `AHuman Diseases` subtree from the KEGG BRITE text file.

    Example lines:
    AHuman Diseases
    B  Cancer: overview
    C    05200  Pathways in cancer
    """
    in_human_diseases = False
    level_a: Optional[str] = None
    level_b: Optional[str] = None
    records: List[Dict[str, Any]] = []

    pattern_c = re.compile(r"^C\s+(\d{5})\s+(.+?)\s*$")

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue

        if line.startswith("A"):
            label = _normalize_text(line[1:])
            if label == "Human Diseases":
                in_human_diseases = True
                level_a = label
                level_b = None
                continue

            if in_human_diseases:
                break
            continue

        if not in_human_diseases:
            continue

        if line.startswith("B"):
            level_b = _normalize_text(line[1:])
            continue

        if not line.startswith("C"):
            continue

        match = pattern_c.match(line)
        if not match:
            continue

        numeric_id, pathway_name = match.groups()
        pathway_name = _normalize_text(pathway_name)
        records.append(
            {
                "pathway_id": f"hsa{numeric_id}",
                "brite_level_1": level_a,
                "brite_level_2": level_b,
                "brite_level_3": pathway_name,
                "brite_numeric_id": numeric_id,
                "brite_name": pathway_name,
            }
        )

    return records


def _is_cancer_related(pathway_name: str, class_labels: Iterable[str]) -> bool:
    text = " ".join([pathway_name, *class_labels]).lower()
    keywords = (
        "cancer",
        "carcinoma",
        "tumor",
        "tumour",
        "leukemia",
        "lymphoma",
        "melanoma",
        "carcinogenesis",
        "oncogenic",
    )
    return any(keyword in text for keyword in keywords)


def _is_lung_cancer_direct(pathway_name: str) -> bool:
    lowered = pathway_name.lower()
    return "lung cancer" in lowered or "small cell lung cancer" in lowered or "non-small cell lung cancer" in lowered


def _is_generic_cancer(pathway_name: str) -> bool:
    lowered = pathway_name.lower()
    if "pathways in cancer" in lowered:
        return True
    return "cancer" in lowered and "lung cancer" not in lowered


def _build_candidate_records(
    brite_records: List[Dict[str, Any]],
    hsa_pathway_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    for record in brite_records:
        pathway_id = record["pathway_id"]
        if pathway_id in seen:
            continue
        seen.add(pathway_id)

        standardized_name = hsa_pathway_map.get(pathway_id, record["brite_name"])
        class_labels = [record["brite_level_1"], record["brite_level_2"]]
        is_cancer = _is_cancer_related(standardized_name, class_labels)
        is_lung_cancer = _is_lung_cancer_direct(standardized_name)

        merged.append(
            {
                "pathway_id": pathway_id,
                "pathway_name": standardized_name,
                "source_db": "KEGG",
                "source_url": f"https://www.kegg.jp/entry/{pathway_id}",
                "brite_level_1": record["brite_level_1"],
                "brite_level_2": record["brite_level_2"],
                "brite_level_3": record["brite_level_3"],
                "class_labels": [label for label in class_labels if label],
                "is_human_disease_pathway": True,
                "is_cancer_related": is_cancer,
                "is_lung_cancer_direct": is_lung_cancer,
                "is_generic_cancer": _is_generic_cancer(standardized_name),
                "raw_entry_available": False,
            }
        )

    return sorted(merged, key=lambda item: (item["pathway_id"], item["pathway_name"]))


def _build_detail_records(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    details: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        details[item["pathway_id"]] = {
            "pathway_id": item["pathway_id"],
            "pathway_name": item["pathway_name"],
            "class": item["class_labels"],
            "brite_level_1": item["brite_level_1"],
            "brite_level_2": item["brite_level_2"],
            "brite_level_3": item["brite_level_3"],
            "source_db": "KEGG",
            "source_url": item["source_url"],
            "is_human_disease_pathway": item["is_human_disease_pathway"],
            "is_cancer_related": item["is_cancer_related"],
            "is_lung_cancer_direct": item["is_lung_cancer_direct"],
            "is_generic_cancer": item["is_generic_cancer"],
            "raw_entry_available": item["raw_entry_available"],
        }
    return details


def _reaction_graph_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False}
    with open(path, "r", encoding="utf-8") as handle:
        graph = json.load(handle)
    node_count = len(graph)
    edge_count = 0
    sample_nodes = []
    for compound_id, neighbors in graph.items():
        if isinstance(neighbors, list):
            edge_count += len(neighbors)
        if len(sample_nodes) < 5:
            sample_nodes.append(compound_id)
    return {
        "available": True,
        "node_count": node_count,
        "edge_count": edge_count,
        "sample_nodes": sample_nodes,
    }


def _manifest(
    brite_records: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    reaction_summary: Dict[str, Any],
) -> Dict[str, Any]:
    level2_counter = Counter(item["brite_level_2"] for item in candidates if item.get("brite_level_2"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step1_check": {
            "brite_raw_present": BRITE_RAW_PATH.exists(),
            "hsa_pathway_list_present": HSA_PATHWAY_LIST_PATH.exists(),
            "reaction_graph_raw_present": REACTION_GRAPH_RAW_PATH.exists(),
            "brite_human_disease_records": len(brite_records),
        },
        "step2_outputs": {
            "candidate_count": len(candidates),
            "cancer_related_count": sum(1 for item in candidates if item.get("is_cancer_related")),
            "lung_cancer_direct_count": sum(1 for item in candidates if item.get("is_lung_cancer_direct")),
            "output_dir": str(OUTPUT_DIR),
        },
        "brite_level_2_counts": dict(level2_counter),
        "reaction_graph_summary": reaction_summary,
        "notes": [
            "This step extracts the KEGG Human Diseases candidate pathway universe from the BRITE hierarchy.",
            "Per-pathway REST `get/hsaXXXXX` raw entry dumps have not been added yet, so `raw_entry_available` is false.",
        ],
    }


def build_kegg_candidates() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    brite_lines = _read_lines(BRITE_RAW_PATH)
    hsa_pathway_map = _load_hsa_pathway_list(HSA_PATHWAY_LIST_PATH)
    brite_records = _parse_brite_human_diseases(brite_lines)
    candidates = _build_candidate_records(brite_records, hsa_pathway_map)
    details = _build_detail_records(candidates)
    reaction_summary = _reaction_graph_summary(REACTION_GRAPH_RAW_PATH)
    manifest = _manifest(brite_records, candidates, reaction_summary)

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, ensure_ascii=False)

    with open(DETAILS_PATH, "w", encoding="utf-8") as handle:
        json.dump(details, handle, indent=2, ensure_ascii=False)

    lung_cancer = [item for item in candidates if item["is_lung_cancer_direct"] or item["is_generic_cancer"]]
    with open(LUNG_CANCER_PATH, "w", encoding="utf-8") as handle:
        json.dump(lung_cancer, handle, indent=2, ensure_ascii=False)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return candidates, details, manifest


if __name__ == "__main__":
    candidates, _, manifest = build_kegg_candidates()
    print("=" * 72)
    print("KEGG Human Disease / Cancer Pathway Candidate Builder")
    print("=" * 72)
    print(f"Candidates extracted: {len(candidates)}")
    print(f"Cancer-related candidates: {sum(1 for item in candidates if item.get('is_cancer_related'))}")
    print(f"Lung-cancer direct candidates: {sum(1 for item in candidates if item.get('is_lung_cancer_direct'))}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("=" * 72)
