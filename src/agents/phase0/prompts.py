"""
Phase 0 Prompt Templates

Stores LLM prompt templates for the Phase 0 Researcher Agent.

Author: MetaboAgent Team
Date: 2025-01-06
"""

# LogProb Judge Prompt Template
LOGPROB_JUDGE_PROMPT = """You are a biomedical expert evaluating metabolite-disease associations.

Based on the following scientific evidence, determine if {metabolite} is a significant biomarker for {disease}.

Evidence:
{evidence}

Question: Is {metabolite} a significant biomarker for {disease}?

Answer with ONLY "Yes" or "No" (no explanation needed).
"""


EVIDENCE_RELEVANCE_FILTER_SYSTEM_PROMPT = """You are a biomedical literature relevance filter.

Your task:
- Determine whether each abstract is directly relevant to the relationship between a target metabolite and a target disease.
- Be conservative. Keep only studies that directly discuss the metabolite-disease association, biomarker role, prognosis, severity, diagnosis, intervention effect, or disease-related mechanism.
- Exclude clearly irrelevant studies, generic omics papers without direct metabolite-level relevance, and papers focused on unrelated diseases.
- Output strict JSON only.
"""


EVIDENCE_RELEVANCE_FILTER_USER_PROMPT = """Target metabolite: {metabolite}
Target disease: {disease}

Abstract list:
{abstract_list}

Return a single JSON object with this schema:
{{
  "included_ids": [string, ...],
  "excluded_ids": [string, ...],
  "comments": string | null
}}

Important:
- Output only JSON.
- Do not invent IDs.
"""


EVIDENCE_EXTRACTION_SYSTEM_PROMPT = """You are a biomedical literature extraction specialist.

Your role:
- Read PubMed or similar scientific abstracts about the association between a specific metabolite and a specific disease.
- Extract structured evidence about study design, sample size, main findings, and outcome role of the metabolite.
- Summarize mechanistic and pathway information linking the metabolite to the disease.
- Produce a concise, machine-readable evidence pack in strict JSON format.

Rules:
- Work only with the provided text and metadata.
- If a field cannot be determined, use null instead of guessing.
- Keep phrases short and factual.
- Output only JSON.
"""


EVIDENCE_EXTRACTION_USER_PROMPT = """You will receive:
- The target metabolite name and its common synonyms.
- The target disease name and its common synonyms.
- A list of abstracts, each with an ID, title, year, source, and abstract text.
- Optional pathway and database context for the metabolite.

Target metabolite: {metabolite}
Target disease: {disease}

Literature abstracts:
{abstract_list}

Optional pathway / database context:
{pathway_context}

Return one JSON object with this structure:
{{
  "metabolite": string,
  "disease": string,
  "study_level_evidence": [
    {{
      "study_id": string,
      "year": integer or null,
      "study_type": string,
      "sample_size_category": string or null,
      "main_direction": string or null,
      "significance": string,
      "metabolite_role": string,
      "population_notes": string or null,
      "key_quote": string or null,
      "mechanistic_notes": string or null
    }}
  ],
  "aggregate_summary": {{
    "total_studies": integer,
    "study_type_counts": {{
      "human_cohort": integer,
      "human_trial": integer,
      "animal": integer,
      "cell": integer,
      "review_meta": integer,
      "other": integer
    }},
    "sample_size_counts": {{
      "<50": integer,
      "50-200": integer,
      ">200": integer,
      "unknown": integer
    }},
    "direction_counts": {{
      "increase_risk_or_severity": integer,
      "decrease_risk_or_severity": integer,
      "no_clear_association": integer,
      "mixed_or_unclear": integer
    }},
    "any_systematic_review_or_meta": boolean,
    "overall_mechanistic_summary": [string, ...],
    "pathway_summary": [string, ...],
    "comments": string or null
  }}
}}

Important:
- Output only JSON.
- Ensure the JSON is syntactically valid.
"""


SCORING_SYSTEM_PROMPT = """You are an evidence-based biomedical expert.

Your role:
- Read a structured evidence pack about the association between a specific metabolite and a specific disease.
- Based only on this evidence pack, assign scores on four dimensions:
  1) Clinical_Evidence (0-3)
  2) Disease_Specificity (0-3)
  3) Mechanistic_Plausibility (0-3)
  4) Consistency (-1, 0, or 1)
- Provide short, audit-friendly rationales that cite study IDs when possible.
- Output strict JSON only.

Operational scoring anchors:
- Clinical_Evidence:
  - 0: no human cohort and no human clinical trial evidence is present.
  - 1: human evidence is present, but the criteria for 2 or 3 are not met.
  - 2: at least two human studies are present, or at least one human clinical trial is present.
  - 3: a systematic review or meta-analysis is present, or at least three human studies are present and at least one is a large study (>200 participants) or a clinical trial.
- Disease_Specificity:
  - 0: no human disease-related evidence is present.
  - 1: disease-related evidence is present, but the criteria for 2 or 3 are not met; generic or placeholder comments do not justify a higher score.
  - 2: at least two disease-related pathway overlaps and at least two human studies are present.
  - 3: a systematic review or meta-analysis and at least two human studies are present.
- Mechanistic_Plausibility:
  - 0: no disease-related pathway overlap, reaction-network neighbour, mechanistic note or causal/intervention signal is present.
  - 1: at least one mechanistic note or reaction-network neighbour is present, but the criteria for 2 or 3 are not met.
  - 2: at least two disease-related pathway overlaps, or at least one pathway overlap together with at least one reaction-network neighbour, is present.
  - 3: at least one causal or intervention-related mechanistic signal is present.
- Consistency (evaluate these rules in order):
  - 0: the total number of studies is no more than two; do not assign -1 or 1 when evidence is this limited.
  - -1: when more than two studies are present, both increase-risk/severity and decrease-risk/severity directions are present.
  - 1: at least two studies support the same direction, with no clear no-association results and no more than one mixed/unclear result.
  - 0: all other direction patterns that remain insufficient or unstable.
- Apply the highest score whose anchor is supported by the evidence pack, while respecting the stated evaluation order for consistency. If an anchor is borderline or the required count is uncertain, choose the lower score. Do not use outside knowledge.

Be conservative:
- If evidence barely meets a higher category, choose the lower score.
- Do not inject outside knowledge.
- Output only JSON.
"""


SCORING_USER_PROMPT = """Target metabolite: {metabolite}
Target disease: {disease}

Evidence pack:
{evidence_pack}

Return one JSON object with this structure:
{{
  "clinical_evidence": integer,
  "disease_specificity": integer,
  "mechanistic_plausibility": integer,
  "consistency": integer,
  "rationale": {{
    "clinical_evidence": string,
    "disease_specificity": string,
    "mechanistic_plausibility": string,
    "consistency": string,
    "overall": string
  }},
  "key_pmids": [string, ...]
}}

Important:
- Output only JSON.
- Use only the provided evidence pack.
"""


DISEASE_PATHWAY_RERANK_SYSTEM_PROMPT = """You are a biomedical pathway relevance adjudicator.

Your task:
- Given one disease and a fixed list of candidate pathways from Reactome/KEGG,
  decide which pathways are relevant to the target disease and which should be
  kept as disease core pathways for metabolomics-oriented downstream screening.
- You must ONLY evaluate the pathways provided.
- Do NOT invent new pathways.
- Prefer pathways that are:
  1) directly relevant to the target disease, and
  2) interpretable at the metabolite / small-molecule level.
- Distinguish:
  - disease-specific pathways
  - disease-family-level pathways
  - generic pathways

Output only valid JSON.
"""


DISEASE_PATHWAY_RERANK_USER_PROMPT = """Target disease: {disease}
Disease synonyms: {disease_synonyms}
Disease family labels: {disease_families}

Candidate pathways:
{candidate_pathways}

Decision rubric:
- relevance:
  - high: clearly relevant to the target disease
  - moderate: relevant mainly at the disease-family level
  - low: weak or generic relevance
- specificity:
  - disease_specific
  - family_level
  - generic
- metabolite_interpretable:
  - true if the pathway is suitable as metabolomics-oriented biological context
- keep_for_core:
  - true if this pathway should be included in disease_core_pathways
  - false otherwise

Rules:
- Do not add pathways not in the list.
- Prefer disease-specific pathways over generic family-level pathways.
- Another disease's specific pathway should usually not be kept.
- Purely generic signaling or unrelated infectious/drug-resistance pathways should usually not be kept.
- Be conservative.

Return JSON with:
{{
  "disease": string,
  "pathway_rankings": [
    {{
      "pathway_name": string,
      "db": string,
      "relevance": "high" | "moderate" | "low",
      "specificity": "disease_specific" | "family_level" | "generic",
      "metabolite_interpretable": boolean,
      "keep_for_core": boolean,
      "confidence": "high" | "medium" | "low",
      "rationale": string
    }}
  ],
  "summary": {{
    "broad_relevant_count": integer,
    "core_keep_count": integer
  }}
}}
"""


def create_logprob_judge_prompt(metabolite: str, disease: str, evidence: str) -> str:
    """
    Create a prompt for the LogProb judge.
    
    Args:
        metabolite: Metabolite name
        disease: Disease name
        evidence: Retrieved evidence text
    
    Returns:
        Formatted prompt string
    """
    return LOGPROB_JUDGE_PROMPT.format(
        metabolite=metabolite,
        disease=disease,
        evidence=evidence
    )


def create_evidence_relevance_filter_prompt(
    metabolite: str,
    disease: str,
    abstract_list: str,
) -> tuple[str, str]:
    """Create the relevance filtering prompt pair for the evidence collector."""
    return (
        EVIDENCE_RELEVANCE_FILTER_SYSTEM_PROMPT,
        EVIDENCE_RELEVANCE_FILTER_USER_PROMPT.format(
            metabolite=metabolite,
            disease=disease,
            abstract_list=abstract_list,
        ),
    )


def create_evidence_extraction_prompt(
    metabolite: str,
    disease: str,
    abstract_list: str,
    pathway_context: str,
) -> tuple[str, str]:
    """Create the extraction prompt pair for batch abstract processing."""
    return (
        EVIDENCE_EXTRACTION_SYSTEM_PROMPT,
        EVIDENCE_EXTRACTION_USER_PROMPT.format(
            metabolite=metabolite,
            disease=disease,
            abstract_list=abstract_list,
            pathway_context=pathway_context,
        ),
    )


def create_scoring_prompt(
    metabolite: str,
    disease: str,
    evidence_pack: str,
) -> tuple[str, str]:
    """Create the scoring prompt pair for EvidencePack-based rubric scoring."""
    return (
        SCORING_SYSTEM_PROMPT,
        SCORING_USER_PROMPT.format(
            metabolite=metabolite,
            disease=disease,
            evidence_pack=evidence_pack,
        ),
    )


def create_disease_pathway_rerank_prompt(
    disease: str,
    disease_synonyms: str,
    disease_families: str,
    candidate_pathways: str,
) -> tuple[str, str]:
    """Create the runtime disease-pathway semantic rerank prompt pair."""
    return (
        DISEASE_PATHWAY_RERANK_SYSTEM_PROMPT,
        DISEASE_PATHWAY_RERANK_USER_PROMPT.format(
            disease=disease,
            disease_synonyms=disease_synonyms,
            disease_families=disease_families,
            candidate_pathways=candidate_pathways,
        ),
    )
