"""Deterministic, sidecar audit utilities for MetaboAgent.

This module is deliberately observational: it records files, parameters and
declared execution scopes, but must never alter data, models or decisions.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


AUDIT_SCHEMA_VERSION = "metaboagent.evaluation_audit.v1"
AUDIT_STATUSES = ("PASS", "PARTIAL", "NOT_ASSESSED", "FAIL")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def sha256_file(path: str | os.PathLike[str] | None) -> Optional[str]:
    """Return a content digest, or None for a missing/non-file path."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_path(path: str | os.PathLike[str] | None) -> Dict[str, Any]:
    candidate = Path(path) if path else None
    exists = False
    is_file = False
    size_bytes = None
    if candidate:
        try:
            exists = candidate.exists()
            is_file = candidate.is_file()
            if is_file:
                size_bytes = candidate.stat().st_size
        except OSError:
            # Broken links and transient NFS entries remain explicit omissions.
            exists = False
            is_file = False
    return {
        "path": str(candidate) if candidate else "",
        "exists": bool(exists),
        "is_file": bool(is_file),
        "sha256": sha256_file(candidate) if candidate else None,
        "size_bytes": size_bytes,
    }


def resolve_output_root(base_dir: str | os.PathLike[str] = "output") -> Path:
    """Resolve default audit/output paths into an isolated runtime workspace."""
    if str(base_dir) == "output":
        runtime_root = str(os.environ.get("METABOAGENT_RUNTIME_ROOT", "") or "").strip()
        if runtime_root:
            return Path(runtime_root) / "output"
    return Path(base_dir)


def resolve_audit_dir(base_dir: str | os.PathLike[str] = "output") -> Path:
    path = resolve_output_root(base_dir) / "audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_audit_json(
    name: str,
    payload: Mapping[str, Any],
    base_dir: str | os.PathLike[str] = "output",
) -> str:
    """Atomically write a versioned audit document and return its path."""
    audit_dir = resolve_audit_dir(base_dir)
    destination = audit_dir / name
    body = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **_json_safe(dict(payload)),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return str(destination)


def environment_snapshot(code_paths: Iterable[str] = ()) -> Dict[str, Any]:
    """Capture environment/provenance without exposing secrets or raw data."""
    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL, timeout=30
        ).splitlines()
    except Exception as exc:
        packages = [f"unavailable: {type(exc).__name__}"]
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "packages": sorted(packages),
        "code_files": {str(path): describe_path(path) for path in code_paths},
    }


def _iter_manifest_files(root_path: Path):
    """Yield files below *root_path*, following isolated-run directory links safely.

    ``Path.rglob`` does not recurse through directory symlinks.  Isolated runs
    intentionally expose large Phase 0/1 trees through symlinks, so a manifest
    built with ``rglob`` can incorrectly report zero artifacts.  This walker
    follows links, records visited directory inodes to prevent cycles, and
    yields the logical path under the manifest root.
    """
    if not root_path.exists() or not root_path.is_dir():
        return
    stack = [(root_path, Path(""), {(root_path.stat().st_dev, root_path.stat().st_ino)})]
    while stack:
        current, rel_dir, ancestors = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name)
        except OSError:
            continue
        for entry in entries:
            rel_path = rel_dir / entry.name
            if "audit" in rel_path.parts:
                continue
            try:
                if entry.is_dir(follow_symlinks=True):
                    target_stat = entry.stat(follow_symlinks=True)
                    target_key = (target_stat.st_dev, target_stat.st_ino)
                    if target_key in ancestors:
                        continue
                    stack.append((Path(entry.path), rel_path, ancestors | {target_key}))
                elif entry.is_file(follow_symlinks=True):
                    yield Path(entry.path), rel_path
            except OSError:
                # Broken links, permission errors and transient NFS entries are
                # retained as omissions rather than aborting the whole audit.
                continue


def build_artifact_manifest(root: str | os.PathLike[str] = "output") -> Dict[str, Any]:
    """Index non-audit output files, including files under directory symlinks."""
    root_path = resolve_output_root(root)
    records = []
    for path, relative_path in _iter_manifest_files(root_path) or ():
        record = {"relative_path": str(relative_path), **describe_path(path)}
        record["is_symlink"] = path.is_symlink()
        if path.is_symlink():
            try:
                record["symlink_target"] = str(path.resolve(strict=False))
            except OSError:
                record["symlink_target"] = None
        records.append(record)
    records.sort(key=lambda item: item["relative_path"])
    return {
        "artifact_root": str(root_path),
        "artifact_count": len(records),
        "manifest_status": "PASS" if records else "PARTIAL",
        "followed_directory_symlinks": True,
        "excluded_relative_prefixes": ["audit"],
        "artifacts": records,
    }


def write_pipeline_lock(
    *,
    run_id: str = "",
    lock_created_at_utc: str = "",
    lock_stage: str = "phase2_model_selection_locked",
    status: str = "LOCKED_BEFORE_HOLDOUT",
    evaluation_scope: str = "locked_internal_holdout",
    selected_model: str = "",
    selected_features: Iterable[str] = (),
    train_data_path: str = "",
    holdout_data_path: str = "",
    winner_artifact_path: str = "",
    holdout_artifact_path: str = "",
    holdout_evaluation_started_at_utc: str | None = None,
    holdout_evaluation_completed_at_utc: str | None = None,
    lock_before_holdout: bool = True,
    base_dir: str | os.PathLike[str] = "output",
) -> str:
    """Write explicit model-lock chronology before and after holdout access."""
    features = sorted(dict.fromkeys(str(feature) for feature in selected_features))
    feature_digest = hashlib.sha256(
        json.dumps(features, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return write_audit_json(
        "pipeline_lock.json",
        {
            "assessment_domain": "lock_chronology",
            "run_id": run_id or str(os.environ.get("METABOAGENT_RUN_TAG", "") or ""),
            "lock_stage": lock_stage,
            "status": status,
            "evaluation_scope": evaluation_scope,
            "lock_before_holdout": bool(lock_before_holdout),
            "lock_created_at_utc": lock_created_at_utc or datetime.now(timezone.utc).isoformat(),
            "holdout_evaluation_started_at_utc": holdout_evaluation_started_at_utc,
            "holdout_evaluation_completed_at_utc": holdout_evaluation_completed_at_utc,
            "selected_model": selected_model,
            "selected_features": features,
            "selected_features_sha256": feature_digest,
            "source_artifacts": {
                "train_data": describe_path(train_data_path),
                "holdout_data": describe_path(holdout_data_path),
                "phase2_winner": describe_path(winner_artifact_path),
                "holdout_evaluation": describe_path(holdout_artifact_path),
            },
            "interpretation": (
                "The model and feature panel are fixed before the locked internal holdout is accessed. "
                "This is an internal held-out development cohort, not independent external validation."
            ),
        },
        base_dir=base_dir,
    )


def _path_time(path: str | os.PathLike[str] | None, *, which: str) -> Optional[str]:
    if not path:
        return None
    try:
        stamp = Path(path).stat().st_mtime if which == "modified" else Path(path).stat().st_ctime
        return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()
    except OSError:
        return None


def write_execution_trace(
    *,
    run_id: str = "",
    stage_paths: Optional[Mapping[str, str]] = None,
    status: str = "PASS",
    base_dir: str | os.PathLike[str] = "output",
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write a run-scoped phase execution trace."""
    root = resolve_output_root(base_dir)
    phases = stage_paths or {
        "phase0": str(root.parent / "phase0"),
        "phase1": str(root.parent / "phase1"),
        "phase2": str(root / "artifacts" / "phase2_winner_scores.json"),
        "phase3_report": str(root / "reports" / "phase3_pipeline_summary.md"),
    }
    stages = []
    for stage, raw_path in phases.items():
        descriptor = describe_path(raw_path)
        stages.append({
            "stage": stage,
            "path": descriptor,
            "status": "PASS" if descriptor.get("exists") else "PARTIAL",
            "started_at_utc": _path_time(raw_path, which="created"),
            "completed_at_utc": _path_time(raw_path, which="modified"),
        })
    safe_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("METABOAGENT_") and "KEY" not in key and "TOKEN" not in key and "PASSWORD" not in key
    }
    payload: Dict[str, Any] = {
        "assessment_domain": "execution_trace",
        "run_id": run_id or str(os.environ.get("METABOAGENT_RUN_TAG", "") or ""),
        "status": status,
        "runtime_root": str(root.parent),
        "command_line": list(sys.argv),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "environment": safe_env,
        "stages": stages,
        "interpretation": "Stage timestamps are filesystem-level provenance; they do not replace a process log.",
    }
    if extra:
        payload.update(dict(extra))
    return write_audit_json("execution_trace.json", payload, base_dir=base_dir)


def _status_for(condition: bool, unavailable: bool = False) -> str:
    if unavailable:
        return "NOT_ASSESSED"
    return "PASS" if condition else "PARTIAL"


def build_audit_summary(base_dir: str | os.PathLike[str] = "output") -> Dict[str, Any]:
    """Produce fixed-rule assessment summary. No LLM interpretation is involved."""
    audit_dir = resolve_audit_dir(base_dir)
    required = {
        "data_split": "data_split_audit.json",
        "preprocessing": "preprocessing_audit.json",
        "feature_engineering": "feature_engineering_audit.json",
        "feature_selection": "feature_selection_audit.json",
        "model_selection": "model_selection_audit.json",
        "performance": "performance_audit.json",
        "interpretability": "interpretability_audit.json",
        "figures_and_tables": "figure_table_manifest.json",
        "reproducibility": "environment_manifest.json",
        "lock_chronology": "pipeline_lock.json",
        "heldout_performance": "heldout_performance.json",
        "execution_trace": "execution_trace.json",
        "artifact_inventory": "artifact_manifest.json",
    }
    domains = []
    for domain, filename in required.items():
        path = audit_dir / filename
        available = path.is_file()
        domains.append({
            "domain": domain,
            "status": _status_for(available, unavailable=not available),
            "evidence_file": str(path),
            "evidence_sha256": sha256_file(path),
            "rule": "required deterministic audit artifact is present",
        })
    return {
        "assessment_policy": "artifact_presence_only; scientific validity requires human review",
        "domains": domains,
        "overall_status": "PASS" if all(row["status"] == "PASS" for row in domains) else "PARTIAL",
    }


def package_final_delivery(
    run_id: str,
    report_paths: Iterable[str] = (),
    figure_paths: Iterable[str] = (),
    table_paths: Iterable[str] = (),
    base_dir: str | os.PathLike[str] = "output",
) -> Dict[str, Any]:
    """Copy final artifacts into a user-facing package without altering sources."""
    root = resolve_output_root(base_dir)
    delivery_root = root / "final_delivery" / str(run_id)
    sections = {
        "report": list(report_paths),
        "figures": list(figure_paths),
        "tables": list(table_paths),
        "audit": [str(path) for path in resolve_audit_dir(root).glob("*") if path.is_file()],
    }
    copied: Dict[str, list[Dict[str, Any]]] = {}
    for section, paths in sections.items():
        destination_dir = delivery_root / section
        destination_dir.mkdir(parents=True, exist_ok=True)
        copied[section] = []
        for raw_path in paths:
            source = Path(raw_path)
            if not source.is_file():
                copied[section].append({"source": str(source), "copied": False})
                continue
            destination = destination_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            copied[section].append({
                "source": str(source), "destination": str(destination), "copied": True,
                "sha256": sha256_file(destination),
            })
    readme = delivery_root / "README_AUDIT.md"
    readme.write_text(
        "# MetaboAgent Audit Delivery\n\n"
        "The `audit/` directory is deterministic provenance evidence. Audit status indicates "
        "artifact availability and does not replace scientific or clinical review.\n",
        encoding="utf-8",
    )
    delivery_manifest = delivery_root / "delivery_manifest.json"
    delivery_manifest.write_text(
        json.dumps(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": str(run_id),
                "sections": copied,
                "readme": describe_path(readme),
                "file_count": sum(len(items) for items in copied.values()),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "delivery_root": str(delivery_root),
        "sections": copied,
        "readme": str(readme),
        "delivery_manifest": str(delivery_manifest),
    }
