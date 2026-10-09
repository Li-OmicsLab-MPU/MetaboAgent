#!/usr/bin/env python3
"""
Build a generic disease -> pathway map from external Reactome / KEGG indices.

Design principles
- Use disease names from `storage/disease_map.json` only as query labels.
- Do NOT use disease-specific candidate metabolites to define pathways.
- Combine external Reactome + KEGG knowledge into:
  - external_disease_pathways_broad
  - disease_core_pathways
  - top_pathways
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "storage"

DISEASE_MAP_JSON = STORAGE_DIR / "disease_map.json"
REACTOME_HUMAN_JSON = STORAGE_DIR / "external_pathways" / "reactome" / "reactome_human_pathways.json"
REACTOME_SMALL_MOLECULE_JSON = STORAGE_DIR / "external_pathways" / "reactome" / "reactome_small_molecule_support.json"
KEGG_CANDIDATES_JSON = STORAGE_DIR / "external_pathways" / "kegg" / "kegg_human_disease_pathway_candidates.json"

OUTPUT_JSON = STORAGE_DIR / "disease_pathway_map.json"
MANIFEST_JSON = STORAGE_DIR / "external_pathways" / "disease_pathway_map_manifest.json"


STOPWORDS = {
    "disease",
    "disorder",
    "syndrome",
    "type",
    "with",
    "without",
    "and",
    "or",
    "of",
    "the",
    "a",
    "an",
    "due",
    "to",
    "former",
    "formerly",
    "late",
    "early",
    "onset",
    "idiopathic",
    "familial",
    "hereditary",
    "inherited",
    "dependent",
    "x",
    "linked",
}

LOW_SIGNAL_TOKENS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "defective",
    "deficiency",
    "deficiencies",
    "variant",
    "variants",
    "cause",
    "causes",
    "caused",
    "mediated",
    "signaling",
    "signalling",
    "response",
    "pathway",
    "pathways",
    "factor",
    "factors",
}

ORGAN_SITE_TOKENS = {
    "lung",
    "breast",
    "prostate",
    "colorectal",
    "colon",
    "pancreatic",
    "pancreas",
    "gastric",
    "stomach",
    "thyroid",
    "renal",
    "kidney",
    "bladder",
    "endometrial",
    "ovarian",
    "ovary",
    "hepatic",
    "hepatocellular",
    "liver",
    "brain",
    "skin",
}

INFECTIOUS_KEYWORDS = {
    "infection",
    "infectious",
    "viral",
    "bacterial",
    "parasite",
    "parasitic",
    "influenza",
    "covid",
    "hepatitis",
    "leishmania",
    "salmonella",
    "shigella",
    "vibrio",
    "coronavirus",
}

CORE_NOISE_TERMS = {
    "drug resistance",
    "antimicrobial",
    "infection with",
    "action of antimicrobials",
    "antimicrobial resistance",
}

METABOLIC_THEME_KEYWORDS = {
    "metabolism",
    "metabolic",
    "glycolysis",
    "gluconeogenesis",
    "citrate",
    "tca",
    "carbon",
    "choline",
    "glutamine",
    "glutaminolysis",
    "serine",
    "glycine",
    "methionine",
    "one-carbon",
    "lipid",
    "fatty acid",
    "sphingolipid",
    "purine",
    "pyrimidine",
    "oxidative",
    "redox",
    "amino acid",
    "biosynthesis",
    "degradation",
}

FAMILY_RULES = {
    "cancer": {
        "disease_keywords": {
            "cancer",
            "carcinoma",
            "tumor",
            "tumour",
            "neoplasm",
            "glioma",
            "melanoma",
            "leukemia",
            "lymphoma",
            "sarcoma",
        },
        "kegg_classes": {"Cancer: overview", "Cancer: specific types", "Drug resistance: antineoplastic"},
        "pathway_keywords": {
            "cancer",
            "carcinoma",
            "carcinogenesis",
            "oncogenic",
            "tumor",
            "tumour",
            "cell cycle",
            "apoptosis",
            "pd-l1",
            "microRNAs",
        },
    },
    "endocrine_metabolic": {
        "disease_keywords": {
            "diabetes",
            "obesity",
            "hyperglycemia",
            "hypoglycemia",
            "insulin",
            "metabolic",
            "lipidemia",
            "dyslipidemia",
        },
        "kegg_classes": {"Endocrine and metabolic disease"},
        "pathway_keywords": {
            "insulin",
            "glucagon",
            "glucose",
            "metabolism",
            "lipid",
            "fatty acid",
            "amino acid",
            "carbon",
        },
    },
    "immune_inflammatory": {
        "disease_keywords": {
            "crohn",
            "colitis",
            "arthritis",
            "psoriasis",
            "lupus",
            "autoimmune",
            "immune",
            "inflammatory",
            "asthma",
            "allergy",
        },
        "kegg_classes": {"Immune disease"},
        "pathway_keywords": {
            "immune",
            "interleukin",
            "cytokine",
            "inflammatory",
            "innate",
            "adaptive",
            "antigen",
        },
    },
    "cardiovascular": {
        "disease_keywords": {
            "heart",
            "cardio",
            "myocardial",
            "atherosclerosis",
            "hypertension",
            "stroke",
            "vascular",
            "coronary",
        },
        "kegg_classes": {"Cardiovascular disease"},
        "pathway_keywords": {
            "hemostasis",
            "platelet",
            "vascular",
            "angiotensin",
            "coagulation",
            "lipoprotein",
        },
    },
    "neurodegenerative": {
        "disease_keywords": {
            "alzheimer",
            "parkinson",
            "dementia",
            "huntington",
            "neurodegenerative",
            "ataxia",
            "frontotemporal",
            "lewy",
        },
        "kegg_classes": {"Neurodegenerative disease"},
        "pathway_keywords": {
            "neuronal",
            "synaptic",
            "neuro",
            "amyloid",
            "tau",
            "neurotransmitter",
        },
    },
    "mental_health": {
        "disease_keywords": {
            "schizophrenia",
            "depression",
            "bipolar",
            "anxiety",
            "autism",
            "adhd",
            "psychiatric",
            "psychosis",
            "obsessive",
            "compulsive",
        },
        "kegg_classes": set(),
        "pathway_keywords": {
            "neuronal",
            "synaptic",
            "neurotransmitter",
            "dopamine",
            "serotonin",
            "glutamate",
            "gaba",
            "axon",
            "postsynaptic",
            "presynaptic",
            "chemical synapse",
            "nmda",
            "ampar",
            "glutamatergic",
            "serotonergic",
            "dopaminergic",
            "neuron",
        },
    },
    "renal": {
        "disease_keywords": {
            "kidney",
            "renal",
            "nephro",
            "glomerular",
            "nephritis",
            "nephropathy",
        },
        "kegg_classes": set(),
        "pathway_keywords": {
            "renal",
            "kidney",
            "electrolyte",
            "solute",
            "transport",
            "hemostasis",
        },
    },
    "hepatic": {
        "disease_keywords": {
            "liver",
            "hepatic",
            "hepatitis",
            "cirrhosis",
            "steato",
            "hepatocellular",
        },
        "kegg_classes": set(),
        "pathway_keywords": {
            "liver",
            "hepatic",
            "bile",
            "lipid",
            "cholesterol",
            "fatty acid",
            "drug adme",
        },
    },
    "infectious": {
        "disease_keywords": {
            "infection",
            "infectious",
            "viral",
            "bacterial",
            "parasitic",
            "hepatitis",
            "covid",
            "influenza",
            "virus",
            "pathogen",
        },
        "kegg_classes": {"Infectious disease: viral", "Infectious disease: bacterial", "Infectious disease: parasitic"},
        "pathway_keywords": {
            "infection",
            "infectious",
            "viral",
            "bacterial",
            "pathogen",
        },
    },
    "deficiency_disorder": {
        "disease_keywords": {
            "deficiency",
            "deficiencies",
            "acidemia",
            "aciduria",
            "carboxylase",
            "hydroxylase",
            "dehydrogenase",
            "oxidase",
            "lyase",
            "synthetase",
            "ketothiolase",
            "cps1",
            "glu1ds1",
            "mody",
        },
        "kegg_classes": {"Endocrine and metabolic disease"},
        "pathway_keywords": {
            "deficiency",
            "metabolism",
            "biosynthesis",
            "degradation",
            "oxidation",
            "fatty acid",
            "amino acid",
            "cofactor",
            "mitochondrial",
        },
    },
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_text(value: str) -> str:
    text = (value or "").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(value: str) -> List[str]:
    return [token for token in _normalize_text(value).replace("-", " ").split() if token and token not in STOPWORDS]


def _make_ngrams(tokens: Sequence[str], max_n: int = 4) -> List[str]:
    ngrams: List[str] = []
    for n in range(1, min(max_n, len(tokens)) + 1):
        for idx in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[idx: idx + n]).strip()
            if phrase:
                ngrams.append(phrase)
    return ngrams


def build_query_terms(disease_name: str) -> Dict[str, Any]:
    normalized = _normalize_text(disease_name)
    tokens = _tokenize(disease_name)
    informative_tokens = [
        token
        for token in tokens
        if token not in LOW_SIGNAL_TOKENS and not token.isdigit() and len(token) >= 4
    ]

    phrases: Set[str] = {normalized} if normalized else set()
    if tokens:
        phrases.add(" ".join(tokens))
        phrases.update(phrase for phrase in _make_ngrams(tokens) if len(phrase) >= 4)
    informative_phrases: Set[str] = set()
    if informative_tokens:
        informative_phrases.add(" ".join(informative_tokens))
        informative_phrases.update(
            phrase for phrase in _make_ngrams(informative_tokens, max_n=3) if len(phrase) >= 5
        )

    if normalized.endswith(" disease"):
        phrases.add(normalized[:-8].strip())
    if normalized.endswith(" syndrome"):
        phrases.add(normalized[:-8].strip())
    if " mellitus" in normalized:
        phrases.add(normalized.replace(" mellitus", ""))
    if " type 2" in normalized:
        phrases.add(normalized.replace(" type 2", ""))
    if " type 1" in normalized:
        phrases.add(normalized.replace(" type 1", ""))

    cleaned_phrases = {phrase.strip() for phrase in phrases if phrase and len(phrase.strip()) >= 4}
    return {
        "normalized_name": normalized,
        "tokens": tokens,
        "informative_tokens": informative_tokens,
        "phrases": sorted(cleaned_phrases, key=lambda item: (-len(item), item)),
        "informative_phrases": sorted(
            {phrase.strip() for phrase in informative_phrases if phrase and len(phrase.strip()) >= 5},
            key=lambda item: (-len(item), item),
        ),
        "organ_sites": sorted(token for token in informative_tokens if token in ORGAN_SITE_TOKENS),
    }


def infer_disease_families(query_terms: Dict[str, Any]) -> List[str]:
    text = " ".join(query_terms["phrases"])
    families: List[str] = []
    for family_name, rule in FAMILY_RULES.items():
        if any(keyword in text for keyword in rule["disease_keywords"]):
            families.append(family_name)
    return families


def _pathway_token_overlap(query_tokens: Sequence[str], pathway_name: str) -> int:
    pathway_tokens = set(_tokenize(pathway_name))
    return sum(1 for token in query_tokens if len(token) >= 4 and token in pathway_tokens)


def _phrase_match_score(query_phrases: Sequence[str], pathway_name: str) -> int:
    normalized_pathway = _normalize_text(pathway_name)
    score = 0
    for phrase in query_phrases:
        if phrase and phrase in normalized_pathway:
            score = max(score, 8 if len(phrase.split()) >= 2 else 5)
    return score


def _exact_disease_match_score(query_terms: Dict[str, Any], pathway_name: str) -> int:
    normalized_pathway = _normalize_text(pathway_name)
    exact_score = 0
    for phrase in query_terms["phrases"]:
        if phrase == query_terms["normalized_name"] and phrase and phrase in normalized_pathway:
            exact_score = max(exact_score, 10)
        elif len(phrase.split()) >= 2 and phrase in normalized_pathway:
            exact_score = max(exact_score, 7)
    return exact_score


def _informative_phrase_match_score(query_terms: Dict[str, Any], pathway_name: str) -> int:
    normalized_pathway = _normalize_text(pathway_name)
    score = 0
    for phrase in query_terms.get("informative_phrases", []):
        if phrase and phrase in normalized_pathway:
            score = max(score, 6 if len(phrase.split()) >= 2 else 3)
    return score


def _disease_mismatch_penalty(query_terms: Dict[str, Any], pathway_name: str, families: Sequence[str]) -> int:
    normalized_pathway = _normalize_text(pathway_name)
    penalty = 0

    disease_sites = set(query_terms.get("organ_sites", []))
    pathway_sites = {token for token in _tokenize(pathway_name) if token in ORGAN_SITE_TOKENS}
    if disease_sites and pathway_sites and "cancer" in families:
        if not (disease_sites & pathway_sites):
            if "cancer" in normalized_pathway or "carcinoma" in normalized_pathway or "tumor" in normalized_pathway:
                penalty += 8

    if "infectious" not in families and any(keyword in normalized_pathway for keyword in INFECTIOUS_KEYWORDS):
        penalty += 6

    if query_terms["informative_tokens"]:
        if all(token not in normalized_pathway for token in query_terms["informative_tokens"]):
            if "defective" in normalized_pathway or "deficiency" in normalized_pathway:
                penalty += 4

    return penalty


def _family_score(families: Sequence[str], pathway_name: str, class_labels: Sequence[str]) -> int:
    text = " ".join([_normalize_text(pathway_name), *[_normalize_text(label) for label in class_labels]])
    score = 0
    for family in families:
        rule = FAMILY_RULES[family]
        if any(label in rule["kegg_classes"] for label in class_labels):
            score += 4
        if any(keyword in text for keyword in rule["pathway_keywords"]):
            score += 3
    return score


def _metabolic_theme_score(pathway_name: str) -> int:
    text = _normalize_text(pathway_name)
    return sum(1 for keyword in METABOLIC_THEME_KEYWORDS if keyword in text)


def score_kegg_candidate(candidate: Dict[str, Any], query_terms: Dict[str, Any], families: Sequence[str]) -> Tuple[int, int, Dict[str, Any]]:
    name = candidate["pathway_name"]
    class_labels = candidate.get("class_labels", [])
    exact_match_score = _exact_disease_match_score(query_terms, name)
    phrase_score = _phrase_match_score(query_terms["phrases"], name)
    informative_phrase_score = _informative_phrase_match_score(query_terms, name)
    token_score = min(_pathway_token_overlap(query_terms["informative_tokens"], name), 4)
    family_score = _family_score(families, name, class_labels)
    cancer_bonus = 2 if "cancer" in families and candidate.get("is_cancer_related") else 0
    direct_bonus = 3 if exact_match_score >= 10 else 0
    mismatch_penalty = _disease_mismatch_penalty(query_terms, name, families)
    broad_score = (
        exact_match_score
        + informative_phrase_score
        + min(phrase_score, 5)
        + token_score
        + family_score
        + cancer_bonus
        + direct_bonus
        - mismatch_penalty
    )

    interpretability = _metabolic_theme_score(name)
    if "metabolism" in _normalize_text(name):
        interpretability += 2

    metadata = {
        "db": "KEGG",
        "pathway_id": candidate["pathway_id"],
        "pathway_name": name,
        "score": broad_score,
        "interpretability_score": interpretability,
        "class_labels": class_labels,
        "match_reasons": {
            "exact_match_score": exact_match_score,
            "phrase_score": phrase_score,
            "informative_phrase_score": informative_phrase_score,
            "token_score": token_score,
            "family_score": family_score,
            "cancer_bonus": cancer_bonus,
            "direct_bonus": direct_bonus,
            "mismatch_penalty": mismatch_penalty,
        },
        "source_url": candidate.get("source_url"),
    }
    return broad_score, interpretability, metadata


def score_reactome_candidate(
    pathway_id: str,
    pathway: Dict[str, Any],
    small_molecule: Dict[str, Any],
    query_terms: Dict[str, Any],
    families: Sequence[str],
) -> Tuple[int, int, Dict[str, Any]]:
    name = pathway["pathway_name"]
    exact_match_score = _exact_disease_match_score(query_terms, name)
    phrase_score = _phrase_match_score(query_terms["phrases"], name)
    informative_phrase_score = _informative_phrase_match_score(query_terms, name)
    token_score = min(_pathway_token_overlap(query_terms["informative_tokens"], name), 4)
    family_score = _family_score(families, name, [])
    mismatch_penalty = _disease_mismatch_penalty(query_terms, name, families)
    broad_score = (
        exact_match_score
        + informative_phrase_score
        + min(phrase_score, 5)
        + token_score
        + family_score
        - mismatch_penalty
    )

    if broad_score <= 0:
        return 0, 0, {}

    interpretability = _metabolic_theme_score(name)
    if small_molecule.get("has_small_molecule_support"):
        interpretability += 3
    if "metabolism" in _normalize_text(name):
        interpretability += 2

    metadata = {
        "db": "Reactome",
        "pathway_id": pathway_id,
        "pathway_name": name,
        "score": broad_score,
        "interpretability_score": interpretability,
        "class_labels": [],
        "match_reasons": {
            "exact_match_score": exact_match_score,
            "phrase_score": phrase_score,
            "informative_phrase_score": informative_phrase_score,
            "token_score": token_score,
            "family_score": family_score,
            "small_molecule_support": bool(small_molecule.get("has_small_molecule_support")),
            "mismatch_penalty": mismatch_penalty,
        },
        "source_url": pathway.get("source_url"),
    }
    return broad_score, interpretability, metadata


def deduplicate_ranked(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in records:
        key = (item["db"], _normalize_text(item["pathway_name"]))
        existing = best_by_key.get(key)
        if existing is None or (
            item["score"], item["interpretability_score"], item["pathway_name"]
        ) > (
            existing["score"], existing["interpretability_score"], existing["pathway_name"]
        ):
            best_by_key[key] = item
    return sorted(
        best_by_key.values(),
        key=lambda item: (-item["score"], -item["interpretability_score"], item["pathway_name"], item["db"]),
    )


def _is_core_candidate(item: Dict[str, Any], families: Sequence[str]) -> bool:
    name = _normalize_text(item["pathway_name"])
    reasons = item.get("match_reasons", {})
    exact_match_score = int(reasons.get("exact_match_score", 0) or 0)
    family_score = int(reasons.get("family_score", 0) or 0)
    mismatch_penalty = int(reasons.get("mismatch_penalty", 0) or 0)
    interpretability = int(item.get("interpretability_score", 0) or 0)
    broad_score = int(item.get("score", 0) or 0)

    if mismatch_penalty >= 6:
        return False

    if any(noise_term in name for noise_term in CORE_NOISE_TERMS) and "infectious" not in families:
        return False

    if broad_score < 4:
        return False

    if exact_match_score >= 7:
        return True

    if interpretability >= 4 and broad_score >= 5:
        return True

    if interpretability >= 3 and family_score >= 3 and broad_score >= 6:
        return True

    if "immune_inflammatory" in families:
        if family_score >= 3 and broad_score >= 5 and mismatch_penalty <= 2:
            return True

    if "mental_health" in families:
        if family_score >= 3 and broad_score >= 5 and mismatch_penalty <= 2:
            return True

    if "deficiency_disorder" in families:
        if exact_match_score >= 7:
            return True
        if interpretability >= 4 and family_score >= 3 and broad_score >= 5:
            return True

    return False


def build_disease_record(
    disease_name: str,
    reactome_pathways: Dict[str, Dict[str, Any]],
    reactome_small_molecule: Dict[str, Dict[str, Any]],
    kegg_candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    query_terms = build_query_terms(disease_name)
    families = infer_disease_families(query_terms)

    ranked_records: List[Dict[str, Any]] = []

    for candidate in kegg_candidates:
        broad_score, interpretability, metadata = score_kegg_candidate(candidate, query_terms, families)
        if broad_score <= 0:
            continue
        ranked_records.append(metadata)

    for pathway_id, pathway in reactome_pathways.items():
        small_molecule = reactome_small_molecule.get(pathway_id, {})
        broad_score, interpretability, metadata = score_reactome_candidate(
            pathway_id,
            pathway,
            small_molecule,
            query_terms,
            families,
        )
        if broad_score <= 0:
            continue
        ranked_records.append(metadata)

    ranked_records = deduplicate_ranked(ranked_records)
    broad_records = ranked_records[:50]

    core_records = [
        item
        for item in ranked_records
        if _is_core_candidate(item, families)
    ][:20]

    if not core_records:
        core_records = [
            item for item in broad_records
            if item["interpretability_score"] >= 3 and int(item.get("score", 0)) >= 4
        ][:10]

    disease_core_pathways = [item["pathway_name"] for item in core_records]
    top_pathways = disease_core_pathways[:]
    broad_names = [item["pathway_name"] for item in broad_records]

    return {
        "disease_core_pathways": disease_core_pathways,
        "top_pathways": top_pathways,
        "external_disease_pathways_broad": broad_names,
        "all_pathways_ranked": broad_records,
        "retrieval_metadata": {
            "normalized_name": query_terms["normalized_name"],
            "query_phrases": query_terms["phrases"],
            "query_tokens": query_terms["tokens"],
            "disease_families": families,
            "broad_candidate_count": len(broad_records),
            "core_candidate_count": len(core_records),
        },
    }


def build_disease_pathway_map() -> Dict[str, Any]:
    disease_map = _load_json(DISEASE_MAP_JSON)
    reactome_pathways = _load_json(REACTOME_HUMAN_JSON)
    reactome_small_molecule = _load_json(REACTOME_SMALL_MOLECULE_JSON)
    kegg_candidates = _load_json(KEGG_CANDIDATES_JSON)

    output: Dict[str, Any] = {}
    family_counter = Counter()
    empty_core_count = 0

    for disease_name in sorted(disease_map.keys()):
        record = build_disease_record(
            disease_name=disease_name,
            reactome_pathways=reactome_pathways,
            reactome_small_molecule=reactome_small_molecule,
            kegg_candidates=kegg_candidates,
        )
        output[disease_name] = record
        family_counter.update(record["retrieval_metadata"]["disease_families"])
        if not record["disease_core_pathways"]:
            empty_core_count += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "disease_map_json": str(DISEASE_MAP_JSON),
            "reactome_human_pathways_json": str(REACTOME_HUMAN_JSON),
            "reactome_small_molecule_support_json": str(REACTOME_SMALL_MOLECULE_JSON),
            "kegg_human_disease_pathway_candidates_json": str(KEGG_CANDIDATES_JSON),
        },
        "outputs": {
            "disease_pathway_map_json": str(OUTPUT_JSON),
        },
        "stats": {
            "disease_count": len(output),
            "empty_core_count": empty_core_count,
            "family_counts": dict(family_counter),
            "sample_diseases": {
                key: {
                    "top_pathways": output[key]["top_pathways"][:5],
                    "disease_families": output[key]["retrieval_metadata"]["disease_families"],
                }
                for key in (
                    "lung cancer",
                    "crohn's disease",
                    "diabetes mellitus type 2",
                    "breast cancer",
                    "schizophrenia",
                )
                if key in output
            },
        },
        "notes": [
            "This map is built from external Reactome/KEGG indices using disease names as retrieval queries only.",
            "It does not use disease-linked candidate metabolites to define disease pathways.",
            "Reactome ranking uses pathway names plus small-molecule interpretability support.",
        ],
    }

    with open(MANIFEST_JSON, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    manifest = build_disease_pathway_map()
    stats = manifest["stats"]
    print("=" * 72)
    print("Disease Pathway Map Builder")
    print("=" * 72)
    print(f"Diseases processed: {stats['disease_count']}")
    print(f"Empty core pathway records: {stats['empty_core_count']}")
    print(f"Output: {OUTPUT_JSON}")
    print("=" * 72)
