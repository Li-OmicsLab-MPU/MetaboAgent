import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REGISTRY_SCHEMA_VERSION = "metaboagent.run_registry.v1"
REGISTRY_ROOT = Path("storage/run_registry")
GLOBAL_REGISTRY_PATH = REGISTRY_ROOT / "run_registry.jsonl"
DISEASE_INDEX_DIR = REGISTRY_ROOT / "diseases"
ARTIFACT_ARCHIVE_DIR = REGISTRY_ROOT / "artifacts"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_") or "unknown"


def normalize_path(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return ""
    normalized = os.path.normpath(str(path))
    if os.path.isabs(normalized):
        try:
            return os.path.relpath(normalized, os.getcwd())
        except ValueError:
            return normalized
    return normalized


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return normalize_path(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _copy_path(src_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir():
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)
    else:
        shutil.copy2(src_path, dest_path)


def archive_artifacts(
    *,
    run_id: str,
    disease_name: str,
    stage: str,
    artifact_paths: Dict[str, Any],
) -> Dict[str, Any]:
    disease_slug = slugify(disease_name)
    archived: Dict[str, Any] = {}
    stage_root = ARTIFACT_ARCHIVE_DIR / disease_slug / run_id / stage
    stage_root.mkdir(parents=True, exist_ok=True)

    for label, raw_value in (artifact_paths or {}).items():
        if isinstance(raw_value, (list, tuple)):
            archived_items = []
            for index, item in enumerate(raw_value):
                archived_item = archive_artifacts(
                    run_id=run_id,
                    disease_name=disease_name,
                    stage=stage,
                    artifact_paths={f"{label}_{index}": item},
                ).get(f"{label}_{index}", {})
                archived_items.append(archived_item)
            archived[label] = archived_items
            continue

        source = normalize_path(raw_value)
        if not source:
            archived[label] = {"source": "", "archived_path": "", "exists": False}
            continue

        src_path = Path(source)
        if not src_path.exists():
            archived[label] = {"source": source, "archived_path": "", "exists": False}
            continue

        safe_label = slugify(label)
        destination = stage_root / safe_label / src_path.name
        if src_path.resolve() != destination.resolve():
            _copy_path(src_path, destination)

        archived[label] = {
            "source": source,
            "archived_path": normalize_path(destination),
            "exists": True,
        }

    return archived


def record_stage_run(
    *,
    run_id: str,
    disease_name: str,
    stage: str,
    parameters: Dict[str, Any] | None = None,
    inputs: Dict[str, Any] | None = None,
    outputs: Dict[str, Any] | None = None,
    archived_outputs: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
    lineage: Dict[str, Any] | None = None,
    notes: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    disease_slug = slugify(disease_name)
    recorded_at = datetime.now(timezone.utc).isoformat()
    entry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "recorded_at": recorded_at,
        "run_id": run_id,
        "disease_name": disease_name,
        "disease_slug": disease_slug,
        "stage": stage,
        "parameters": _json_safe(parameters or {}),
        "inputs": _json_safe(inputs or {}),
        "outputs": _json_safe(outputs or {}),
        "archived_outputs": _json_safe(archived_outputs or {}),
        "metrics": _json_safe(metrics or {}),
        "lineage": _json_safe(lineage or {}),
        "notes": _json_safe(notes or {}),
    }

    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    DISEASE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(GLOBAL_REGISTRY_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    disease_index_path = DISEASE_INDEX_DIR / f"{disease_slug}.json"
    if disease_index_path.exists():
        try:
            disease_index = json.loads(disease_index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            disease_index = {}
    else:
        disease_index = {}

    entries = disease_index.get("entries", [])
    entries.append(entry)
    disease_index = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "disease_name": disease_name,
        "disease_slug": disease_slug,
        "updated_at": recorded_at,
        "entry_count": len(entries),
        "entries": entries,
    }
    disease_index_path.write_text(
        json.dumps(disease_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return entry
