"""
SOP Loader V2 - Code Interpreter Pattern Support

This module loads and vectorizes the v2.0.0 SOP format which uses the Code Interpreter pattern.
Key differences from v1:
- All actions are 'coding_task' type
- Steps include 'instruction', 'details', 'lib_hint' fields
- Explicit 'context_files' and 'outputs' tracking
- Stage 1.5 for knowledge-driven feature engineering
"""

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
    """Normalize unknown JSON values into a trimmed string."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _format_mapping(value: Any) -> Optional[str]:
    """Convert nested JSON structures into compact human-readable representation."""
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


def _safe_stage_num(stage_id: Any) -> str:
    """Extract stage number from stage_id (e.g., 1.5 -> '1.5', 0 -> '0')."""
    s = _as_nonempty_str(stage_id)
    if not s:
        return ""
    return s


def _format_step_page_content_v2(stage: Dict[str, Any], step: Dict[str, Any]) -> str:
    """
    Render a v2 Code Interpreter step into a retrieval-friendly block.
    
    V2 format includes:
    - instruction: High-level task description
    - details: Detailed implementation guidance
    - lib_hint: Tool library to import
    - context_files: Required input data/state
    - outputs: Expected output variables
    """
    parts: List[str] = []

    # Stage context
    stage_id = _as_nonempty_str(stage.get("stage_id")) or ""
    stage_name = _as_nonempty_str(stage.get("name")) or ""
    stage_description = _as_nonempty_str(stage.get("description"))
    stage_condition = _as_nonempty_str(stage.get("condition"))

    parts.append("[Stage Context]")
    parts.append(f"Stage: {stage_id} - {stage_name}".strip())
    if stage_description:
        parts.append(f"Description: {stage_description}")
    if stage_condition:
        parts.append(f"Stage Condition: {stage_condition} (Only execute if true)")

    # Step identification
    step_id = _as_nonempty_str(step.get("step_id")) or ""
    action = _as_nonempty_str(step.get("action")) or ""
    instruction = _as_nonempty_str(step.get("instruction")) or ""
    
    parts.append("")
    parts.append("[Step Identification]")
    parts.append(f"Step ID: {step_id}")
    parts.append(f"Action Type: {action}")
    if instruction:
        parts.append(f"Instruction: {instruction}")

    # Implementation details (core of v2 format)
    details = _as_nonempty_str(step.get("details"))
    if details:
        parts.append("")
        parts.append("[Implementation Details]")
        parts.append(details)

    # Library hint (critical for code generation)
    lib_hint = _as_nonempty_str(step.get("lib_hint"))
    if lib_hint:
        parts.append("")
        parts.append("[Tool Library]")
        parts.append(f"Import from: {lib_hint}")

    # Context files (inputs)
    context_files = step.get("context_files")
    context_files_str = _format_mapping(context_files) if context_files is not None else None
    if context_files_str:
        parts.append("")
        parts.append("[Required Context]")
        parts.append(f"Context Files/Variables: {context_files_str}")

    # Outputs
    outputs = step.get("outputs")
    outputs_str = _format_mapping(outputs) if outputs is not None else None
    if outputs_str:
        parts.append("")
        parts.append("[Expected Outputs]")
        parts.append(f"Produces: {outputs_str}")

    # Reasoning (biological/methodological rationale)
    reasoning = _as_nonempty_str(step.get("reasoning"))
    if reasoning:
        parts.append("")
        parts.append("[Rationale]")
        parts.append(reasoning)

    return "\n".join(parts).strip() + "\n"


def build_sop_vector_store_v2(
    json_path: str | Path | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str = "sops_v2",
) -> str:
    """
    Build a Chroma collection for v2 Code Interpreter SOP format.
    
    Creates:
    - One SOP summary doc (metadata + tool libraries)
    - One doc per step (with stage context and implementation details)
    
    Args:
        json_path: Path to v2 SOP JSON file
        persist_dir: Directory to persist Chroma DB
        collection_name: Name of the Chroma collection
        
    Returns:
        Path to persisted vector store
    """
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_json_path = (
        Path(json_path) if json_path is not None 
        else project_root / "data" / "SOP-代谢物分析流程_v2_code_interpreter.json"
    )
    resolved_persist_dir = (
        Path(persist_dir) if persist_dir is not None 
        else project_root / "storage" / "sop_chroma_v2"
    )

    with open(resolved_json_path, "r", encoding="utf-8") as f:
        sop = json.load(f)

    # Extract SOP metadata
    sop_id = _as_nonempty_str(sop.get("sop_id")) or ""
    sop_name = _as_nonempty_str(sop.get("sop_name")) or ""
    sop_version = _as_nonempty_str(sop.get("sop_version")) or ""
    description = _as_nonempty_str(sop.get("description")) or ""
    architecture_pattern = _as_nonempty_str(sop.get("architecture_pattern")) or ""

    # Extract applicability
    applicability = sop.get("applicability") if isinstance(sop.get("applicability"), dict) else {}
    assumptions = applicability.get("assumptions")
    typical_shape = applicability.get("typical_shape")

    assumptions_str = _format_mapping(assumptions) or ""
    typical_shape_str = _format_mapping(typical_shape) or ""

    # Extract tool libraries (v2 specific)
    tool_libraries = sop.get("tool_libraries") if isinstance(sop.get("tool_libraries"), dict) else {}
    domain_tools = tool_libraries.get("domain_tools")
    analysis_tools = tool_libraries.get("analysis_tools")
    
    domain_tools_str = _format_mapping(domain_tools) or ""
    analysis_tools_str = _format_mapping(analysis_tools) or ""

    # Build summary document
    summary_parts: List[str] = []
    if sop_name:
        summary_parts.append(f"SOP Name: {sop_name}")
    if description:
        summary_parts.append(f"Description: {description}")
    if architecture_pattern:
        summary_parts.append(f"Architecture Pattern: {architecture_pattern}")
    if typical_shape_str:
        summary_parts.append(f"Typical Data Shape: {typical_shape_str}")
    if assumptions_str:
        summary_parts.append(f"Assumptions: {assumptions_str}")
    if domain_tools_str:
        summary_parts.append(f"Domain Tools: {domain_tools_str}")
    if analysis_tools_str:
        summary_parts.append(f"Analysis Tools: {analysis_tools_str}")

    docs: List[Document] = []

    # Add summary document
    docs.append(
        Document(
            page_content="\n".join(summary_parts).strip() + "\n",
            metadata={
                "type": "sop_summary",
                "sop_id": sop_id,
                "version": sop_version,
                "architecture_pattern": architecture_pattern,
            },
        )
    )

    # Process stages and steps
    step_docs = 0
    stages = sop.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue

            stage_id = _as_nonempty_str(stage.get("stage_id")) or ""
            stage_name = _as_nonempty_str(stage.get("name")) or ""
            stage_description = _as_nonempty_str(stage.get("description"))
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
                instruction = _as_nonempty_str(step.get("instruction"))
                lib_hint = _as_nonempty_str(step.get("lib_hint"))
                
                if not step_id or not action:
                    continue

                stage_num = _safe_stage_num(stage_id)
                
                # Check if step has conditional logic
                is_conditional = bool(
                    stage_condition
                    or _as_nonempty_str(step.get("condition"))
                )

                # Extract context files and outputs for metadata
                context_files = step.get("context_files")
                outputs = step.get("outputs")
                context_files_str = _format_mapping(context_files) if context_files else ""
                outputs_str = _format_mapping(outputs) if outputs else ""

                docs.append(
                    Document(
                        page_content=_format_step_page_content_v2(stage=stage, step=step),
                        metadata={
                            "type": "sop_step",
                            "sop_id": sop_id,
                            "version": sop_version,
                            "stage_id": stage_id,
                            "stage_num": stage_num,
                            "stage_name": stage_name,
                            "step_id": step_id,
                            "action": action,
                            "instruction": instruction or "",
                            "lib_hint": lib_hint or "",
                            "is_conditional": is_conditional,
                            "requires_routing": requires_routing,
                            "has_context_files": bool(context_files),
                            "has_outputs": bool(outputs),
                        },
                    )
                )
                step_docs += 1

    print(
        f"[sop_loader_v2] Loaded SOP id={sop_id} version={sop_version} "
        f"-> docs: summary=1, steps={step_docs}, total={len(docs)}"
    )

    # Create persist directory
    resolved_persist_dir.mkdir(parents=True, exist_ok=True)

    # Setup embeddings
    embeddings_kwargs: Dict[str, Any] = {"model": "text-embedding-3-small"}
    openai_api_base = os.getenv("OPENAI_API_BASE")
    if openai_api_base:
        embeddings_kwargs["openai_api_base"] = openai_api_base

    embeddings = OpenAIEmbeddings(**embeddings_kwargs)

    # Rebuild strategy: delete existing collection before inserting
    client = chromadb.PersistentClient(path=str(resolved_persist_dir))
    try:
        client.delete_collection(name=collection_name)
        print(f"[sop_loader_v2] Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Persist the rebuilt collection
    _ = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(resolved_persist_dir),
        collection_name=collection_name,
    )

    print(f"[sop_loader_v2] Persisted Chroma vector store to: {resolved_persist_dir}")
    return str(resolved_persist_dir)


def query_sop_step_v2(
    query: str,
    persist_dir: str | Path | None = None,
    collection_name: str = "sops_v2",
    k: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Query the v2 SOP vector store.
    
    Args:
        query: Natural language query
        persist_dir: Directory where Chroma DB is persisted
        collection_name: Name of the Chroma collection
        k: Number of results to return
        filter_metadata: Optional metadata filters (e.g., {"stage_id": "1.5"})
        
    Returns:
        List of matching documents
    """
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_persist_dir = (
        Path(persist_dir) if persist_dir is not None 
        else project_root / "storage" / "sop_chroma_v2"
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

    if filter_metadata:
        return vs.similarity_search(query, k=k, filter=filter_metadata)
    else:
        return vs.similarity_search(query, k=k)


if __name__ == "__main__":
    # Build the v2 vector store
    print("=" * 80)
    print("Building SOP v2 Vector Store")
    print("=" * 80)
    build_sop_vector_store_v2()

    print("\n" + "=" * 80)
    print("Testing Queries")
    print("=" * 80)

    # Test query 1: Knowledge-driven feature engineering
    print("\n[Query 1] Knowledge-driven feature engineering")
    results = query_sop_step_v2("knowledge-driven feature engineering pathway scores", k=3)
    if results:
        print(f"Found {len(results)} results")
        print("\nTop result:")
        print(results[0].page_content[:500] + "...")
        print(f"\nMetadata: {results[0].metadata}")
    else:
        print("No results found")

    # Test query 2: Reaction ratios
    print("\n" + "-" * 80)
    print("[Query 2] Reaction-based ratio features")
    results = query_sop_step_v2("generate reaction ratios substrate product", k=2)
    if results:
        print(f"Found {len(results)} results")
        print(f"\nTop result metadata: {results[0].metadata}")
    else:
        print("No results found")

    # Test query 3: Feature selection with filter
    print("\n" + "-" * 80)
    print("[Query 3] Feature selection (Stage 5 only)")
    results = query_sop_step_v2(
        "feature selection consensus methods",
        k=3,
        filter_metadata={"stage_num": "5"}
    )
    if results:
        print(f"Found {len(results)} results in Stage 5")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}: {doc.metadata.get('instruction', 'N/A')}")
    else:
        print("No results found")

    print("\n" + "=" * 80)
    print("Vector store build and test complete!")
    print("=" * 80)
