from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def _as_nonempty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    return s or None


def _normalize_hmdb_id(raw: Any) -> Optional[str]:
    s = _as_nonempty_str(raw)
    if not s:
        return None

    s_up = s.upper()

    if s_up.startswith("HMDB"):
        digits = "".join(ch for ch in s_up[4:] if ch.isdigit())
    else:
        digits = "".join(ch for ch in s_up if ch.isdigit())

    if not digits:
        return None

    if len(digits) < 7:
        digits = digits.zfill(7)
    elif len(digits) > 7:
        digits = digits[-7:]

    return f"HMDB{digits}"


def _build_metadata(row: pd.Series) -> Dict[str, Any]:
    md: Dict[str, Any] = {}

    hmdb_id = _normalize_hmdb_id(row.get("HMDB"))
    if hmdb_id:
        md["hmdb_id"] = hmdb_id

    mapping = {
        "KEGG": "kegg_id",
        "ChEBI": "chebi_id",
        "CAS": "cas_id",
        "Drugbank": "drugbank_id",
        "CID": "pubchem_cid",
        "SID": "pubchem_sid",
        "DTXSID": "dtxsid",
        "DTXCID": "dtxcid",
    }

    for col, key in mapping.items():
        v = _as_nonempty_str(row.get(col))
        if v:
            md[key] = v

    return md


def build_mapping_vector_store(
    csv_path: str | Path | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str = "id_mapping",
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_csv_path = (
        Path(csv_path) if csv_path is not None else project_root / "data" / "metabolitIDmapping.csv"
    )
    resolved_persist_dir = (
        Path(persist_dir) if persist_dir is not None else project_root / "storage" / "mapping_chroma"
    )

    id_cols = ["HMDB", "KEGG", "ChEBI", "CAS", "Drugbank", "CID", "SID", "DTXSID", "DTXCID", "Name"]
    dtype = {c: "string" for c in id_cols}

    df = pd.read_csv(resolved_csv_path, dtype=dtype, keep_default_na=True)

    docs: List[Document] = []
    kept = 0
    skipped_no_name = 0
    skipped_no_ids = 0

    for _, row in df.iterrows():
        name = _as_nonempty_str(row.get("Name"))
        if not name:
            skipped_no_name += 1
            continue

        md = _build_metadata(row)
        if not md:
            skipped_no_ids += 1
            continue

        docs.append(Document(page_content=f"Name: {name}", metadata=md))
        kept += 1

    print(
        f"[mapping_loader] Loaded {len(df)} rows, kept {kept}, skipped_no_name={skipped_no_name}, skipped_no_ids={skipped_no_ids}"
    )

    resolved_persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings_kwargs: Dict[str, Any] = {"model": "text-embedding-3-small"}
    openai_api_base = os.getenv("OPENAI_API_BASE")
    if openai_api_base:
        embeddings_kwargs["openai_api_base"] = openai_api_base

    embeddings = OpenAIEmbeddings(**embeddings_kwargs)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(resolved_persist_dir),
        collection_name=collection_name,
    )

    print(f"[mapping_loader] Persisted Chroma vector store to: {resolved_persist_dir}")
    return str(resolved_persist_dir)


def search_ids_by_name(
    query: str,
    persist_dir: str | Path | None = None,
    collection_name: str = "id_mapping",
    k: int = 5,
) -> List[Document]:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_persist_dir = (
        Path(persist_dir) if persist_dir is not None else project_root / "storage" / "mapping_chroma"
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
    build_mapping_vector_store()

    results = search_ids_by_name("Glucose", k=5)
    if not results:
        print("[mapping_loader] No results found for query: Glucose")
    else:
        print("[mapping_loader] First result metadata for query: Glucose")
        print(results[0].metadata)
