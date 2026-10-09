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
    if pd.isna(value):
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


def _join_unique(values: List[Optional[str]]) -> str:
    uniq = sorted({v for v in values if v})
    return "; ".join(uniq)


def build_pathbank_vector_store(
    csv_path: str | Path | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str = "pathways",
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_csv_path = (
        Path(csv_path)
        if csv_path is not None
        else project_root / "data" / "PathBank_all_metabolites_merge.csv"
    )
    resolved_persist_dir = (
        Path(persist_dir)
        if persist_dir is not None
        else project_root / "storage" / "pathbank_chroma"
    )

    usecols = [
        "PathBank ID",
        "Pathway Name",
        "Pathway Subject",
        "Species",
        "Description",
        "Metabolite Name",
        "HMDB ID",
    ]

    df = pd.read_csv(resolved_csv_path, usecols=usecols, dtype="string", keep_default_na=True)

    def _aggregate_group(g: pd.DataFrame) -> pd.Series:
        pathway_name = _as_nonempty_str(g["Pathway Name"].iloc[0])
        subject = _as_nonempty_str(g["Pathway Subject"].iloc[0])
        species = _as_nonempty_str(g["Species"].iloc[0])
        description = _as_nonempty_str(g["Description"].iloc[0])

        metabolite_names = [
            _as_nonempty_str(v) for v in g["Metabolite Name"].tolist()  # type: ignore[arg-type]
        ]
        hmdb_ids = [_normalize_hmdb_id(v) for v in g["HMDB ID"].tolist()]  # type: ignore[arg-type]

        return pd.Series(
            {
                "Pathway Name": pathway_name or "",
                "Pathway Subject": subject or "",
                "Species": species or "",
                "Description": description or "",
                "Metabolite Names": _join_unique(metabolite_names),
                "HMDB IDs": _join_unique(hmdb_ids),
            }
        )

    grouped = df.groupby("PathBank ID", dropna=False).apply(_aggregate_group)
    grouped = grouped.reset_index().rename(columns={"PathBank ID": "pathbank_id"})

    docs: List[Document] = []
    for _, row in grouped.iterrows():
        pathbank_id = _as_nonempty_str(row.get("pathbank_id"))
        if not pathbank_id:
            continue

        pathway_name = _as_nonempty_str(row.get("Pathway Name")) or ""
        subject = _as_nonempty_str(row.get("Pathway Subject")) or ""
        description = _as_nonempty_str(row.get("Description")) or ""
        involved_metabolites = _as_nonempty_str(row.get("Metabolite Names")) or ""

        page_content = (
            f"Pathway Name: {pathway_name}\n"
            f"Subject: {subject}\n"
            f"Description: {description}\n"
            f"Involved Metabolites: {involved_metabolites}"
        )

        md: Dict[str, Any] = {
            "pathbank_id": pathbank_id,
            "species": _as_nonempty_str(row.get("Species")) or "",
            "hmdb_ids": _as_nonempty_str(row.get("HMDB IDs")) or "",
        }

        docs.append(Document(page_content=page_content, metadata=md))

    print(f"[pathbank_loader] Loaded {len(df)} rows -> {len(grouped)} pathways -> {len(docs)} docs")

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

    print(f"[pathbank_loader] Persisted Chroma vector store to: {resolved_persist_dir}")
    return str(resolved_persist_dir)


def query_pathway_by_metabolite(
    metabolite_name: str,
    persist_dir: str | Path | None = None,
    collection_name: str = "pathways",
    k: int = 5,
) -> List[Document]:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or add it to the project's .env file."
        )

    resolved_persist_dir = (
        Path(persist_dir)
        if persist_dir is not None
        else project_root / "storage" / "pathbank_chroma"
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

    return vs.similarity_search(metabolite_name, k=k)


if __name__ == "__main__":
    build_pathbank_vector_store()

    results = query_pathway_by_metabolite("L-Glutamine", k=5)
    if not results:
        print("[pathbank_loader] No results found for metabolite: L-Glutamine")
    else:
        first = results[0]
        pathway_name = ""
        for line in first.page_content.splitlines():
            if line.startswith("Pathway Name:"):
                pathway_name = line.split(":", 1)[1].strip()
                break
        print("[pathbank_loader] First result Pathway Name for query: L-Glutamine")
        print(pathway_name)
