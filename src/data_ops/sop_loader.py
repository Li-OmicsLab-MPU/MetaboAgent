from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def _as_nonempty_str(value: Any) -> Optional[str]:
    # Normalize unknown JSON values into a trimmed string. Returning None for empty values
    # makes downstream "if field exists" logic easier.
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _format_mapping(value: Any) -> Optional[str]:
    # Convert nested JSON-like structures (dict/list/scalars) into a compact human-readable
    # representation. This intentionally avoids dumping raw JSON so the text remains semantic.
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        s = _as_nonempty_str(value)
        return s
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            if v is None:
                continue
            items.append(f"{k}={v}")
        return ", ".join(items) if items else None
    if isinstance(value, list):
        parts = []
        for v in value:
            s = _as_nonempty_str(v)
            if s:
                parts.append(s)
        return ", ".join(parts) if parts else None
    return _as_nonempty_str(value)


def _safe_phase_num(step_id: Any) -> str:
    # The SOP encodes phase information inside step_id in the form "Phase.Step" (e.g. "6.2").
    # We derive a coarse phase_num used for metadata filtering/grouping during retrieval.
    s = _as_nonempty_str(step_id)
    if not s:
        return ""
    if "." not in s:
        return s
    return s.split(".", 1)[0]


def _format_branch_target(branch: Any) -> str:
    if not isinstance(branch, dict):
        return _format_mapping(branch) or ""
    return _as_nonempty_str(branch.get("step_id")) or _as_nonempty_str(branch.get("action")) or ""


def _format_step_page_content(stage: Dict[str, Any], step: Dict[str, Any]) -> str:
    # Render a single step into a natural-language, retrieval-friendly block.
    # Key requirement: include Stage context so a retrieved step also carries its
    # stage-level constraint (e.g., Stage Condition).
    parts: List[str] = []

    stage_id = _as_nonempty_str(stage.get("stage_id")) or ""
    stage_name = _as_nonempty_str(stage.get("name")) or ""
    stage_condition = _as_nonempty_str(stage.get("condition"))

    parts.append("[Context]")
    parts.append(f"Stage: {stage_id} - {stage_name}".strip())
    if stage_condition:
        parts.append(f"Stage Condition: {stage_condition} (Only execute this stage if true)")

    step_id = _as_nonempty_str(step.get("step_id")) or ""
    action = _as_nonempty_str(step.get("action")) or ""
    reasoning = _as_nonempty_str(step.get("reasoning")) or ""

    parts.append("")
    parts.append("[Step Details]")
    parts.append(f"Step ID: {step_id}")
    parts.append(f"Action: {action}")
    if reasoning:
        parts.append(f"Goal/Reasoning: {reasoning}")

    parts.append("")
    parts.append("[Configuration]")
    inputs = step.get("inputs")
    inputs_str = _format_mapping(inputs) if inputs is not None else None
    if inputs_str:
        parts.append("Inputs: " + inputs_str)
    params = step.get("params")
    params_str = _format_mapping(params) if params is not None else None
    if params_str:
        parts.append("Params: " + params_str)

    parts.append("")
    parts.append("[Flow Logic]")
    step_condition = _as_nonempty_str(step.get("condition"))
    if step_condition:
        parts.append(f"Condition: {step_condition}")

    if "true_branch" in step or "false_branch" in step:
        true_target = _format_branch_target(step.get("true_branch"))
        false_target = _format_branch_target(step.get("false_branch"))
        parts.append(f"Branching: If true go to {true_target}, else {false_target}.")

    heuristics = step.get("heuristics")
    if heuristics is None:
        heuristics = step.get("guidelines")
    heuristics_str = _format_mapping(heuristics) if heuristics is not None else None
    if heuristics_str:
        parts.append("Heuristics/Guidelines: " + heuristics_str)

    outputs = step.get("outputs")
    outputs_str = _format_mapping(outputs) if outputs is not None else None
    if outputs_str:
        parts.append("")
        parts.append("[Outputs]")
        parts.append("Produces: " + outputs_str)

    return "\n".join(parts).strip() + "\n"


def build_sop_vector_store(
    json_path: str | Path | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str = "sops",
) -> str:
    # Build a Chroma collection containing:
    # - One SOP summary doc (top-level description + applicability assumptions/shape)
    # - One doc per step (with stage context inherited into each step)
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_json_path = (
        Path(json_path) if json_path is not None else project_root / "data" / "SOP-代谢物分析流程.json"
    )
    resolved_persist_dir = (
        Path(persist_dir) if persist_dir is not None else project_root / "storage" / "sop_chroma"
    )

    with open(resolved_json_path, "r", encoding="utf-8") as f:
        sop = json.load(f)

    sop_id = _as_nonempty_str(sop.get("sop_id")) or ""
    sop_name = _as_nonempty_str(sop.get("sop_name")) or ""
    sop_version = _as_nonempty_str(sop.get("sop_version")) or ""
    description = _as_nonempty_str(sop.get("description")) or ""

    applicability = sop.get("applicability") if isinstance(sop.get("applicability"), dict) else {}
    assumptions = applicability.get("assumptions")
    typical_shape = applicability.get("typical_shape")

    assumptions_str = _format_mapping(assumptions) or ""
    typical_shape_str = _format_mapping(typical_shape) or ""

    summary_parts: List[str] = []
    if sop_name:
        summary_parts.append(f"SOP Name: {sop_name}")
    if description:
        summary_parts.append(f"Description: {description}")
    if typical_shape_str:
        summary_parts.append(f"Typical Shape: {typical_shape_str}")
    if assumptions_str:
        summary_parts.append(f"Assumptions: {assumptions_str}")

    docs: List[Document] = []

    docs.append(
        Document(
            page_content="\n".join(summary_parts).strip() + "\n",
            metadata={"type": "sop_summary", "sop_id": sop_id, "version": sop_version},
        )
    )

    step_docs = 0
    stages = sop.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue

            stage_id = _as_nonempty_str(stage.get("stage_id")) or ""
            stage_name = _as_nonempty_str(stage.get("name")) or ""
            stage_condition = _as_nonempty_str(stage.get("condition"))
            requires_routing = bool(stage_condition)

            steps = stage.get("steps")
            if not isinstance(steps, list):
                continue

            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_id = _as_nonempty_str(step.get("step_id"))
                action = _as_nonempty_str(step.get("action"))
                if not step_id or not action:
                    continue

                phase_num = _safe_phase_num(step_id)
                is_conditional = bool(
                    stage_condition
                    or _as_nonempty_str(step.get("condition"))
                    or ("true_branch" in step)
                    or ("false_branch" in step)
                )

                docs.append(
                    Document(
                        page_content=_format_step_page_content(stage=stage, step=step),
                        metadata={
                            "type": "sop_step",
                            "sop_id": sop_id,
                            "stage_id": stage_id,
                            "stage_name": stage_name,
                            "step_id": step_id,
                            "phase_num": phase_num,
                            "action": action,
                            "is_conditional": is_conditional,
                            "requires_routing": requires_routing,
                        },
                    )
                )
                step_docs += 1

    print(
        f"[sop_loader] Loaded SOP id={sop_id} version={sop_version} -> docs: summary=1, steps={step_docs}, total={len(docs)}"
    )

    resolved_persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings_kwargs: Dict[str, Any] = {"model": "text-embedding-3-small"}
    openai_api_base = os.getenv("OPENAI_API_BASE")
    if openai_api_base:
        embeddings_kwargs["openai_api_base"] = openai_api_base

    embeddings = OpenAIEmbeddings(**embeddings_kwargs)

    # Rebuild strategy: delete the existing collection (if present) before inserting.
    # This avoids mixing old/new SOP schemas in the same collection.
    client = chromadb.PersistentClient(path=str(resolved_persist_dir))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    # Persist the rebuilt collection.
    _ = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(resolved_persist_dir),
        collection_name=collection_name,
    )

    print(f"[sop_loader] Persisted Chroma vector store to: {resolved_persist_dir}")
    return str(resolved_persist_dir)


def query_sop_step(
    query: str,
    persist_dir: str | Path | None = None,
    collection_name: str = "sops",
    k: int = 5,
) -> List[Document]:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_persist_dir = (
        Path(persist_dir) if persist_dir is not None else project_root / "storage" / "sop_chroma"
    )

    embeddings_kwargs: Dict[str, Any] = {"model": "text-embedding-3-small"}
    openai_api_base = os.getenv("OPENAI_API_BASE")
    if openai_api_base:
        embeddings_kwargs["openai_api_base"] = openai_api_base

    embeddings = OpenAIEmbeddings(**embeddings_kwargs)

    vs = Chroma(
        collection_name=collection_name,
        persist_directory=str(resolved_persist_dir),
        embedding_function=embeddings,
    )

    return vs.similarity_search(query, k=k)


if __name__ == "__main__":
    build_sop_vector_store()

    results = query_sop_step("univariate analysis", k=5)
    if not results:
        print("[sop_loader] No results found for query: univariate analysis")
    else:
        print("[sop_loader] Top result for query: univariate analysis")
        print(results[0].page_content)
