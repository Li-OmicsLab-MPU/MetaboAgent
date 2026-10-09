"""
MetaboAgent Backend - 真实 Phase0/Phase1 集成
============================================

功能：
- 提供 WebSocket 接口用于前后端实时通信
- 集成真实的 Phase0 和 Phase1 LangGraph Agent
- 实时推送执行进度和日志

作者：MetaboAgent Team
日期：2026-04-14
"""

import asyncio
import io
import json
import sys
import os
import copy
import shutil
import time
import zipfile
from collections import Counter
from datetime import datetime
from statistics import median
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 以当前文件位置确定项目根目录，避免依赖后端启动时的工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.phase0.graph import create_phase0_workflow
from src.agents.phase0.state import create_initial_phase0_state
from src.agents.phase1.graph import build_phase1_graph, load_sop
from src.agents.phase1.state import create_initial_state
from src.agents.phase1.nodes import build_phase1_summary, get_data_summary
from src.memory import MetaboMemoryOrchestrator
from src.tools.analysis.pareto_evaluator import _load_pathway_map, _load_priors_dict, _load_taxonomy_map
from src.tools.domain.reaction_checker import ReactionCheckerTool
from src.utils.config_manager import get_config

# 显式加载项目根目录配置，避免被 backend/.env 或进程工作目录覆盖。
load_dotenv(PROJECT_ROOT / ".env", override=True)

BACKEND_DIR = Path(__file__).resolve().parent
BACKEND_STORAGE_DIR = BACKEND_DIR / "storage"
UPLOADS_DIR = BACKEND_STORAGE_DIR / "uploads"
JOBS_DIR = BACKEND_STORAGE_DIR / "jobs"
# Public, versioned examples live outside the runtime storage directory.  This
# keeps demo inputs reviewable in Git while uploads, jobs, and generated outputs
# remain disposable runtime state.
DEMO_DATASETS_DIR = PROJECT_ROOT / "demo_data"
ALLOWED_UPLOAD_SUFFIXES = {".csv", ".xlsx", ".xls"}

for directory in (BACKEND_STORAGE_DIR, UPLOADS_DIR, JOBS_DIR, DEMO_DATASETS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEMO_DATASETS: Dict[str, Dict[str, Any]] = {
    "liver_cancer": {
        "id": "liver_cancer",
        "name": "Liver cancer example",
        "description": "A small, self-contained liver-cancer metabolomics example for reviewers to try the complete workflow.",
        "filename": "liver_cancer_demo.csv",
        "path": DEMO_DATASETS_DIR / "liver_cancer_demo.csv",
        "defaults": {
            "id_column": "id",
            "group_column": "group",
            "positive_class": "1",
            "disease_name": "liver cancer",
        },
    },
}

# Jobs must not depend on a browser WebSocket staying open.  The public
# deployment is behind a reverse proxy that may reset WebSocket upgrades, so
# the analysis is scheduled by the API process itself and the WebSocket is
# treated as an optional live transport only.
_JOB_TASKS: Dict[str, asyncio.Task] = {}

# ============================================================================
# FastAPI 应用实例化
# ============================================================================

app = FastAPI(
    title="MetaboAgent API",
    description="代谢组学智能分析平台后端服务（真实 Agent 集成）",
    version="2.0.0",
)

# ============================================================================
# CORS 配置
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 辅助函数
# ============================================================================

def get_timestamp() -> str:
    """生成 ISO 格式时间戳"""
    return datetime.now().isoformat()


def get_configured_llm_model() -> str:
    """读取项目 .env 中的统一 LLM 模型配置。"""
    return (
        os.getenv("METABOAGENT_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.5"
    ).strip()


def _safe_name(name: str) -> str:
    """将用户提供的文件名或标识符转换为安全片段。"""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name or ""))
    return safe.strip("._") or "unnamed"


def _resolve_binary_label_metadata(
    data_path: str,
    target_column: str,
    requested_positive_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve the user-facing positive class to a stable 0/1 mapping.

    The UI only asks for the positive class.  The complementary class is
    derived from the uploaded target column here so the same mapping can be
    reused by Phase 1, Phase 2, and downstream reporting.
    """
    import pandas as pd

    if str(data_path).lower().endswith(".csv"):
        target_frame = pd.read_csv(data_path, usecols=[target_column])
    elif str(data_path).lower().endswith((".xlsx", ".xls")):
        target_frame = pd.read_excel(data_path, usecols=[target_column])
    else:
        raise ValueError("不支持的文件格式，请使用 .csv 或 .xlsx")

    raw_values = target_frame[target_column].dropna().tolist()
    unique_values: List[Any] = []
    seen = set()
    for value in raw_values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_values.append(value)

    metadata: Dict[str, Any] = {
        "labels": [str(value).strip() for value in unique_values],
        "positive_class": None,
        "negative_class": None,
        "label_mapping": {},
        "is_binary": len(unique_values) == 2,
    }
    if len(unique_values) != 2:
        if requested_positive_class:
            raise ValueError(
                f"Positive-class selection requires a binary outcome; column '{target_column}' contains {len(unique_values)} non-empty classes"
            )
        return metadata

    normalized = {str(value).strip(): value for value in unique_values}
    requested = str(requested_positive_class or "").strip()
    if requested:
        positive_raw = normalized.get(requested)
        if positive_raw is None:
            raise ValueError(
                f"Positive class '{requested_positive_class}' is not present in outcome column '{target_column}': "
                f"{metadata['labels']}"
            )
    else:
        preferred = {"1", "case", "disease", "positive", "yes", "true"}
        positive_raw = next(
            (value for value in unique_values if str(value).strip().lower() in preferred),
            sorted(unique_values, key=lambda value: str(value).strip())[1],
        )

    negative_raw = next(value for value in unique_values if str(value).strip() != str(positive_raw).strip())
    positive_name = str(positive_raw).strip()
    negative_name = str(negative_raw).strip()
    metadata.update({
        "positive_class": positive_name,
        "negative_class": negative_name,
        "label_mapping": {negative_name: 0, positive_name: 1},
    })
    return metadata


def _json_safe(value: Any) -> Any:
    """递归转换为可 JSON 序列化的数据。"""
    try:
        import math
        import numpy as np
        import pandas as pd
    except Exception:
        math = None
        np = None
        pd = None

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if np is not None and isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return str(value)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


_PATHWAY_INDEX: Optional[Dict[str, List[str]]] = None


def _pathway_index() -> Dict[str, List[str]]:
    """Load the project-local HMDB-to-PathBank index once per API process."""
    global _PATHWAY_INDEX
    if _PATHWAY_INDEX is None:
        payload = _read_json(PROJECT_ROOT / "storage" / "pathbank_pathway_map.json", {})
        _PATHWAY_INDEX = payload if isinstance(payload, dict) else {}
    return _PATHWAY_INDEX


def _feature_display_name_map(job_id: str) -> Dict[str, str]:
    """Return internal feature identifiers mapped back to uploaded column names."""
    mapping: Dict[str, str] = {}
    job_root = _job_dir(job_id)
    candidates = sorted(
        job_root.glob("runtime/phase1/**/*name_mapping*.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for mapping_path in candidates:
        payload = _read_json(mapping_path, {})
        if not isinstance(payload, dict):
            continue
        for source, target in payload.items():
            source_name = str(source or "").strip()
            target_name = str(target or "").strip()
            if not source_name or not target_name:
                continue
            if target_name.upper().startswith("HMDB"):
                mapping.setdefault(target_name.upper(), source_name)
            elif source_name.upper().startswith("HMDB"):
                mapping.setdefault(source_name.upper(), target_name)
    return mapping


def _decorate_result_payload(job_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Add display metadata and repair legacy Phase 0 pathway omissions."""
    display_names = _feature_display_name_map(job_id)
    result["feature_display_names"] = display_names

    phase3 = result.get("phase3")
    if isinstance(phase3, dict):
        excluded_ids = {"fig4h", "fig4h_alt", "objective_shift", "objective_shift_radar"}
        phase3["figures"] = [
            figure for figure in (phase3.get("figures") or [])
            if isinstance(figure, dict)
            and str(figure.get("figure_id") or "") not in excluded_ids
            and "Clinical Validation Composite" not in str(figure.get("title") or "")
            and "plot_final_holdout_calibration" not in str(figure.get("source_task") or "")
            and "final_holdout_calibration" not in str(figure.get("figure_id") or "")
            and "Internal Holdout Calibration Plot" not in str(figure.get("title") or "")
        ]
        phase3["figure_count"] = len(phase3["figures"])
        phase3["final_reports"] = [
            report for report in (phase3.get("final_reports") or [])
            if isinstance(report, dict) and str(report.get("format") or "").lower() in {"html", "pdf"}
        ]

    phase0 = result.get("phase0")
    if not isinstance(phase0, dict):
        return result
    feature_definitions = phase0.get("feature_definitions")
    if not isinstance(feature_definitions, dict):
        feature_definitions = {}
        phase0["feature_definitions"] = feature_definitions
    existing_pathways = feature_definitions.get("target_pathways", []) or phase0.get("top_pathways", [])
    if existing_pathways:
        if not phase0.get("top_pathways"):
            phase0["top_pathways"] = [
                {"name": str(pathway), "supporting_biomarker_count": None, "supporting_biomarkers": []}
                for pathway in existing_pathways
                if str(pathway).strip()
            ]
        phase0.setdefault("pathway_summary", {
            "source": "phase0_feature_rules",
            "message": "Pathways retained by the Phase 0 evidence workflow.",
        })
        return result

    biomarker_ids: Dict[str, str] = {}
    reverse_names = {name.casefold(): hmdb_id for hmdb_id, name in display_names.items()}
    for biomarker in phase0.get("confirmed_biomarkers", []) or []:
        if not isinstance(biomarker, dict):
            continue
        name = str(biomarker.get("name") or biomarker.get("metabolite") or "").strip()
        hmdb_id = str(biomarker.get("hmdb_id") or biomarker.get("hmdb") or biomarker.get("id") or "").strip().upper()
        if not hmdb_id.startswith("HMDB") and name:
            hmdb_id = reverse_names.get(name.casefold(), "")
        if hmdb_id.startswith("HMDB"):
            biomarker_ids[hmdb_id] = name or display_names.get(hmdb_id, hmdb_id)

    counts: Counter[str] = Counter()
    supporting: Dict[str, List[str]] = {}
    pathway_map = _pathway_index()
    for hmdb_id, display_name in biomarker_ids.items():
        for pathway in pathway_map.get(hmdb_id, []) or []:
            pathway_name = str(pathway).strip()
            if not pathway_name:
                continue
            counts[pathway_name] += 1
            supporting.setdefault(pathway_name, []).append(display_name)

    ranked = sorted(counts, key=lambda name: (-counts[name], name.casefold()))[:20]
    if ranked:
        feature_definitions["target_pathways"] = ranked
        phase0["top_pathways"] = [
            {
                "name": pathway,
                "supporting_biomarker_count": counts[pathway],
                "supporting_biomarkers": supporting[pathway],
            }
            for pathway in ranked
        ]
        phase0["pathway_summary"] = {
            "source": "pathbank_confirmed_biomarker_mapping",
            "matched_biomarker_count": len(biomarker_ids),
            "pathway_count": len(ranked),
            "message": "Recovered from PathBank mappings for confirmed Phase 0 biomarkers because the cached result did not contain feature rules.",
        }
    else:
        phase0["pathway_summary"] = {
            "source": "unavailable",
            "matched_biomarker_count": len(biomarker_ids),
            "pathway_count": 0,
            "message": "No PathBank mapping was found for the confirmed biomarkers in this result.",
        }
    return result


def _preview_dataframe(file_path: Path, sheet_name: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    import pandas as pd

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    columns = []
    lowered = {col: str(col).strip().lower() for col in df.columns}
    for col in df.columns:
        series = df[col]
        col_lower = lowered[col]
        if col_lower in {"id", "sampleid", "sample_id", "sample", "subject_id"}:
            role = "id"
        elif col_lower in {"group", "target", "label", "class", "y"}:
            role = "target"
        elif str(series.dtype).startswith(("float", "int")):
            role = "feature"
        else:
            role = "metadata"
        columns.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()) if len(series) else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
            "unique_values": _json_safe(series.dropna().unique().tolist()) if series.nunique(dropna=True) <= 50 else [],
            "suggested_role": role,
        })

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": columns,
        "head": _json_safe(df.head(limit).to_dict(orient="records")),
    }


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / _safe_name(job_id)


def _file_record_path(file_id: str) -> Path:
    return UPLOADS_DIR / _safe_name(file_id) / "file_record.json"


def _load_file_record(file_id: str) -> Dict[str, Any]:
    record = _read_json(_file_record_path(file_id), {})
    if not record:
        raise HTTPException(status_code=404, detail="文件不存在")
    return record


def _load_job(job_id: str) -> Dict[str, Any]:
    job = _read_json(_job_dir(job_id) / "job_status.json", {})
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


def _save_job(job: Dict[str, Any]) -> None:
    _write_json(_job_dir(job["job_id"]) / "job_status.json", job)


def _append_job_log(job_id: str, level: str, phase: str, content: str) -> None:
    job_path = _job_dir(job_id)
    logs_path = job_path / "logs.jsonl"
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": f"log_{datetime.now().timestamp()}",
        "timestamp": get_timestamp(),
        "level": level,
        "phase": phase,
        "content": content,
    }
    with logs_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(entry), ensure_ascii=False) + "\n")


_DEFAULT_PHASE_ESTIMATES: Dict[str, Dict[str, int]] = {
    "phase0": {"median_seconds": 20, "low_seconds": 10, "high_seconds": 45},
    "phase1": {"median_seconds": 1200, "low_seconds": 720, "high_seconds": 2400},
    "phase2": {"median_seconds": 650, "low_seconds": 420, "high_seconds": 1200},
    "phase3": {"median_seconds": 45, "low_seconds": 20, "high_seconds": 120},
}


def _percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _build_runtime_estimate(selected_phases: List[str]) -> Dict[str, Any]:
    """Build a robust ETA profile from recent successful runs.

    Estimates are deliberately returned as ranges. Extremely short legacy or
    mock runs are excluded so that a real analysis is not presented with an
    unrealistically optimistic completion time.
    """
    phase_keys = [phase for phase in ("phase0", "phase1", "phase2", "phase3") if phase in selected_phases]
    samples: Dict[str, List[float]] = {phase: [] for phase in phase_keys}
    minimum_duration = {"phase0": 2, "phase1": 120, "phase2": 120, "phase3": 5}

    completed_jobs = sorted(
        (path for path in JOBS_DIR.glob("job_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:30]
    for job_path in completed_jobs:
        status = _read_json(job_path / "job_status.json", {})
        logs_path = job_path / "logs.jsonl"
        if status.get("status") != "completed" or not logs_path.exists():
            continue
        phase_timestamps: Dict[str, List[datetime]] = {phase: [] for phase in phase_keys}
        try:
            for line in logs_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                phase = str(entry.get("phase", ""))
                if phase not in phase_timestamps or not entry.get("timestamp"):
                    continue
                phase_timestamps[phase].append(datetime.fromisoformat(str(entry["timestamp"])))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for phase, timestamps in phase_timestamps.items():
            if len(timestamps) < 2:
                continue
            duration = (max(timestamps) - min(timestamps)).total_seconds()
            if minimum_duration[phase] <= duration <= 4 * 60 * 60:
                samples[phase].append(duration)

    phase_profile: Dict[str, Dict[str, Any]] = {}
    for phase in phase_keys:
        values = samples[phase]
        fallback = _DEFAULT_PHASE_ESTIMATES[phase]
        if len(values) >= 3:
            center = float(median(values))
            low = min(_percentile(values, 0.25), center * 0.85)
            high = max(_percentile(values, 0.75), center * 1.20)
            source = "historical"
        else:
            center = float(fallback["median_seconds"])
            low = float(fallback["low_seconds"])
            high = float(fallback["high_seconds"])
            source = "baseline"
        phase_profile[phase] = {
            "median_seconds": max(1, round(center)),
            "low_seconds": max(1, round(low)),
            "high_seconds": max(1, round(high)),
            "sample_size": len(values),
            "source": source,
        }

    return {
        "phases": phase_profile,
        "median_seconds": sum(item["median_seconds"] for item in phase_profile.values()),
        "low_seconds": sum(item["low_seconds"] for item in phase_profile.values()),
        "high_seconds": sum(item["high_seconds"] for item in phase_profile.values()),
        "completed_run_count": max((len(values) for values in samples.values()), default=0),
        "method": "recent_completed_runs",
    }


def _list_job_artifacts(job_id: str) -> List[Dict[str, Any]]:
    root = _job_dir(job_id)
    artifacts = []
    if not root.exists():
        return artifacts
    excluded = {"job_status.json", "job_config.json", "result_summary.json", "logs.jsonl"}
    canonical_report_dir = root / "runtime" / "output" / "final_delivery" / "current" / "report"
    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.name in excluded:
            continue
        rel = file_path.relative_to(root).as_posix()
        if "holdout_calibration" in rel.lower():
            continue
        # The Phase 4 report is first rendered below output/reports/current
        # and then copied into final_delivery/current/report.  Expose only the
        # canonical delivery copy in the UI to avoid duplicate report links.
        if rel.startswith("runtime/output/reports/current/") and file_path.name in {"report.html", "report.md", "report.pdf"}:
            if (canonical_report_dir / file_path.name).is_file():
                continue
        # Phase 3 writes its final report, rendered figures, and figure-source
        # data below runtime/output.  These paths do not contain a literal
        # ``phase3`` segment, so classify them explicitly for the UI filter.
        if rel.startswith(("runtime/output/", "runtime/final_delivery/")):
            phase = "phase3"
        else:
            phase = next((part for part in rel.split("/") if part.startswith("phase")), "runtime")
        suffix = file_path.suffix.lower().lstrip(".") or "file"
        if suffix in {"png", "pdf", "svg", "jpg", "jpeg", "webp"} and "figures" in rel:
            artifact_type = "figure"
        elif suffix in {"md", "html", "htm", "docx", "pdf"} and ("reports" in rel or "/report/" in rel):
            artifact_type = "report"
        elif suffix in {"json", "csv", "tsv", "xlsx", "xls"}:
            artifact_type = "data"
        else:
            artifact_type = "file"
        artifacts.append({
            "name": rel,
            "phase": phase,
            "type": artifact_type,
            "format": suffix,
            "size_bytes": file_path.stat().st_size,
            "download_url": f"/api/v1/jobs/{job_id}/download/{rel}",
        })
    return artifacts


class DatasetConfig(BaseModel):
    sheet_name: Optional[str] = None
    id_column: str
    group_column: str
    positive_class: Optional[str] = None
    negative_class: Optional[str] = None


class AnalysisConfig(BaseModel):
    disease_name: str = ""
    clinical_scenario: Optional[str] = None
    run_phase0_prior_search: bool = True
    phases: List[str] = Field(default_factory=lambda: ["phase0", "phase1", "phase2"])


class Phase0Config(BaseModel):
    max_candidates: int = 50
    use_cache: bool = True


class Phase1Config(BaseModel):
    max_steps: int = 100


class Phase2Config(BaseModel):
    beam_width: int = 1
    k_folds: int = 5
    epsilon: float = 0.005
    patience: int = 5


class Phase3Config(BaseModel):
    figure_style: str = "nature"
    language: str = "zh_CN"
    output_formats: List[str] = Field(default_factory=lambda: ["pdf", "png", "json", "html"])
    figures: Dict[str, bool] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    memory_enabled: bool = True
    random_seed: int = 42
    save_intermediate: bool = True


class JobCreateRequest(BaseModel):
    file_id: str
    dataset: DatasetConfig
    analysis: AnalysisConfig
    phase0: Phase0Config = Field(default_factory=Phase0Config)
    phase1: Phase1Config = Field(default_factory=Phase1Config)
    phase2: Phase2Config = Field(default_factory=Phase2Config)
    phase3: Phase3Config = Field(default_factory=Phase3Config)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


async def send_log(
    websocket: WebSocket,
    log_type: str,
    content: str,
    delay: float = 0.1
) -> None:
    """发送日志消息到前端"""
    await asyncio.sleep(delay)
    
    message = {
        "type": "log_message",
        "log": {
            "id": f"log-{datetime.now().timestamp()}",
            "timestamp": get_timestamp(),
            "type": log_type,
            "content": content,
        }
    }
    
    await websocket.send_json(message)
    print(f"[LOG] {log_type.upper()}: {content}")


async def send_state(
    websocket: WebSocket,
    state: str,
    delay: float = 0.1
) -> None:
    """发送 Agent 状态更新到前端"""
    await asyncio.sleep(delay)
    
    message = {
        "type": "agent_state",
        "state": state,
        "timestamp": get_timestamp(),
    }
    
    await websocket.send_json(message)
    print(f"[STATE] {state}")


async def send_activity_update(
    websocket: WebSocket,
    phase: str,
    title: str,
    detail: str,
    *,
    status: str = "running",
    phase_progress: Optional[int] = None,
    step_id: Optional[str] = None,
    step_index: Optional[int] = None,
    step_total: Optional[int] = None,
    elapsed_seconds: Optional[int] = None,
    delay: float = 0.05,
) -> None:
    """Send a concise, structured activity event for the execution trace UI."""
    await asyncio.sleep(delay)
    message: Dict[str, Any] = {
        "type": "activity_update",
        "phase": phase,
        "title": title,
        "detail": detail,
        "status": status,
        "timestamp": get_timestamp(),
    }
    optional_values = {
        "phase_progress": phase_progress,
        "step_id": step_id,
        "step_index": step_index,
        "step_total": step_total,
        "elapsed_seconds": elapsed_seconds,
    }
    message.update({key: value for key, value in optional_values.items() if value is not None})
    await websocket.send_json(message)


async def send_step_progress(
    websocket: WebSocket,
    stage_id: str,
    step_id: str,
    step_name: str,
    status: str,
    *,
    phase_progress: Optional[int] = None,
    step_index: Optional[int] = None,
    step_total: Optional[int] = None,
    delay: float = 0.1
) -> None:
    """Send a Phase 1 step event with enough context for readable progress."""
    await asyncio.sleep(delay)
    
    message = {
        "type": "step_progress",
        "phase": "phase1",
        "stage_id": stage_id,
        "step_id": step_id,
        "step_name": step_name,
        "status": status,
        "timestamp": get_timestamp(),
    }
    if phase_progress is not None:
        message["phase_progress"] = max(0, min(100, int(phase_progress)))
    if step_index is not None:
        message["step_index"] = int(step_index)
    if step_total is not None:
        message["step_total"] = int(step_total)
    
    await websocket.send_json(message)


async def send_context_update(
    websocket: WebSocket,
    variables: Dict[str, Any],
    delay: float = 0.1
) -> None:
    """发送 Context 变量更新到前端"""
    await asyncio.sleep(delay)
    
    message = {
        "type": "context_update",
        "variables": variables,
        "timestamp": get_timestamp(),
    }
    
    await websocket.send_json(message)


async def send_phase0_result(
    websocket: WebSocket,
    result: Dict[str, Any],
    delay: float = 0.1
) -> None:
    """发送 Phase0 结果到前端"""
    await asyncio.sleep(delay)
    
    # 转换为前端格式
    phase0_data = {
        "disease_name": result.get("disease_name", ""),
        "confirmed_biomarkers": [
            {
                "name": name,
                "hmdb_id": f"HMDB{i:07d}",  # Mock HMDB ID
                "confidence_score": 0.85,
            }
            for i, name in enumerate(result.get("final_priors", []), 1)
        ],
        "target_pathways": result.get("feature_definitions", {}).get("target_pathways", []),
        "differential_metabolites_count": len(result.get("candidates", [])),
        "execution_time": result.get("execution_time", 0),
    }
    
    message = {
        "type": "phase0_result",
        "result": phase0_data,
        "timestamp": get_timestamp(),
    }
    
    await websocket.send_json(message)


async def send_phase1_result(
    websocket: WebSocket,
    result: Dict[str, Any],
    delay: float = 0.1
) -> None:
    """发送 Phase1 结果到前端"""
    await asyncio.sleep(delay)
    
    # 提取特征选择和模型训练结果
    context_vars = result.get("context_variables", {})
    
    phase1_data = {
        "feature_selection": {
            "strategy": context_vars.get("feature_selection_method", "unknown"),
            "n_features_selected": len(context_vars.get("selected_features", [])),
            "selected_features": context_vars.get("selected_features", []),
        },
        "training": {
            "best_model": context_vars.get("best_model", "Unknown"),
            "accuracy": context_vars.get("best_model_score", 0.0),
            "roc_auc": context_vars.get("best_model_score", 0.0),
            "training_time": 0,
            "leaderboard": [],
        },
    }
    
    message = {
        "type": "phase1_result",
        "result": phase1_data,
        "timestamp": get_timestamp(),
    }
    
    await websocket.send_json(message)


async def send_phase2_result(
    websocket: WebSocket,
    result: Dict[str, Any],
    delay: float = 0.1
) -> None:
    """发送 Phase2 结果到前端"""
    await asyncio.sleep(delay)

    phase2_data = {
        "winner_features": result.get("features", []),
        "winner_feature_count": len(result.get("features", []) or []),
        "scores": {
            "f_perf": result.get("perf", 0.0),
            "f_bio": result.get("bio", 0.0),
            "f_corr": result.get("corr", 0.0),
            "f_cost": result.get("cost", 0.0),
        },
        "metric": result.get("metric"),
        "stop_reason": result.get("stop_reason"),
        "memory_case_id": result.get("memory_case_id", ""),
        "phase2_summary": result.get("phase2_summary", {}),
    }

    message = {
        "type": "phase2_result",
        "result": phase2_data,
        "timestamp": get_timestamp(),
    }

    await websocket.send_json(message)


# ============================================================================
# Phase 0 执行函数
# ============================================================================

async def run_phase0_real(
    websocket: WebSocket,
    disease_name: str,
    max_candidates: int = 50,
    use_cache: bool = True
) -> Dict[str, Any]:
    """执行真实的 Phase 0 流程"""
    
    await send_log(websocket, "system", f"Phase 0 initialized. Disease context: {disease_name}")
    await send_state(websocket, "phase0_running")
    await send_activity_update(
        websocket,
        "phase0",
        "Preparing prior-evidence search",
        "Loading the disease context and retrieval policy.",
        phase_progress=5,
    )
    
    try:
        # 初始化 LLM
        llm = ChatOpenAI(model=get_configured_llm_model(), temperature=0)
        orchestrator = MetaboMemoryOrchestrator()
        phase0_memory_pack = orchestrator.prepare_phase0_memory_pack(disease_name)
        phase0_memory_debug = orchestrator.summarize_retrieval_trace(
            orchestrator.get_last_retrieval_trace("phase0")
        )
        
        # 创建初始状态
        phase0_input = create_initial_phase0_state(
            disease_name=disease_name,
            use_cache=use_cache,
            max_candidates=max_candidates,
            phase0_memory_pack=phase0_memory_pack,
        )
        
        # 创建并运行 workflow
        await send_log(websocket, "system", "Building the Phase 0 evidence workflow.")
        await send_activity_update(
            websocket,
            "phase0",
            "Building evidence workflow",
            "Preparing literature, pathway, and memory-assisted retrieval.",
            phase_progress=20,
        )
        phase0_workflow = create_phase0_workflow()
        
        await send_log(websocket, "system", "Running Phase 0 evidence retrieval.")
        await send_activity_update(
            websocket,
            "phase0",
            "Retrieving prior evidence",
            "Searching candidate metabolites, confirmed biomarkers, and target pathways.",
            phase_progress=35,
        )
        
        # 在后台线程中运行（因为 LangGraph 是同步的）
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(phase0_workflow.invoke, phase0_input)
            phase0_result = await asyncio.get_event_loop().run_in_executor(None, future.result)
        
        # 发送进度日志
        await send_activity_update(
            websocket,
            "phase0",
            "Consolidating evidence",
            "Ranking retrieved biomarkers and mapping them to metabolic pathways.",
            phase_progress=85,
        )
        await send_log(websocket, "stdout", f"✓ Candidate metabolites: {len(phase0_result.get('candidates', []))}")
        await send_log(websocket, "stdout", f"✓ Confirmed biomarkers: {len(phase0_result.get('final_priors', []))}")
        await send_log(websocket, "stdout", f"✓ Target pathways: {len(phase0_result.get('feature_definitions', {}).get('target_pathways', []))}")
        await send_log(
            websocket,
            "stdout",
            f"✓ Phase 0 memory matches: {len(phase0_memory_pack.get('provenance_cases', []))}"
        )
        await send_log(
            websocket,
            "stdout",
            "  - Phase 0 memory debug: "
            f"considered={phase0_memory_debug.get('considered_case_count', 0)}, "
            f"filtered={phase0_memory_debug.get('filtered_case_count', 0)}, "
            f"selected={phase0_memory_debug.get('selected_case_ids', [])}"
        )
        phase0_summary = orchestrator.build_phase0_summary(phase0_result)
        phase0_result["phase0_summary"] = phase0_summary
        phase0_memory_case = orchestrator.build_phase0_memory_case(
            disease_name=disease_name,
            clinical_scenario=get_config().get_active_scenario(),
            phase0_summary=phase0_summary,
            tags=["phase0", "episodic"],
        )
        phase0_result["memory_case_id"] = orchestrator.write_memory_case(phase0_memory_case)
        if phase0_result.get("memory_case_id"):
            await send_log(
                websocket,
                "stdout",
                f"✓ Phase 0 episodic memory saved: {phase0_result['memory_case_id']}"
            )
        
        # 发送 Phase0 结果
        await send_phase0_result(websocket, phase0_result)
        
        await send_activity_update(
            websocket,
            "phase0",
            "Prior evidence ready",
            "Biomarker and pathway priors are ready for data-driven analysis.",
            status="completed",
            phase_progress=100,
        )
        await send_log(websocket, "system", "Phase 0 completed.")
        
        return phase0_result
        
    except Exception as e:
        await send_log(websocket, "stderr", f"Phase 0 failed: {str(e)}")
        raise


# ============================================================================
# Phase 1 执行函数
# ============================================================================

def _sync_phase1_outputs_to_runtime(result: Dict[str, Any]) -> Dict[str, Any]:
    """将 Phase1 legacy 产物同步到当前 Job runtime 的标准读取路径。"""
    runtime_root = os.environ.get("METABOAGENT_RUNTIME_ROOT")
    if not runtime_root:
        return {"synced": False, "reason": "METABOAGENT_RUNTIME_ROOT 未设置"}

    source_candidates = []
    current_data_path = result.get("current_data_path") if isinstance(result, dict) else None
    if current_data_path:
        source_candidates.append(current_data_path)
    source_candidates.extend([
        "output/phase1/final/selected_features_final.csv",
        "data/selected_features_final.csv",
    ])

    existing_source = next((Path(path) for path in source_candidates if path and Path(path).exists()), None)
    if existing_source is None:
        return {
            "synced": False,
            "reason": "未找到 Phase1 legacy selected_features_final.csv",
            "checked": [str(path) for path in source_candidates if path],
        }

    runtime_root_path = Path(runtime_root)
    final_target = runtime_root_path / "phase1" / "final" / "selected_features_final.csv"
    legacy_target = runtime_root_path / "phase1" / "legacy" / "selected_features_final.csv"
    final_target.parent.mkdir(parents=True, exist_ok=True)
    legacy_target.parent.mkdir(parents=True, exist_ok=True)

    # current_data_path 可能已经指向 runtime/phase1/final 下的标准文件。
    # 这种情况下不能再次 copy 到自身，否则会触发 shutil.SameFileError。
    source_resolved = existing_source.resolve()
    skipped_targets = []
    copied_targets = []
    for target in (final_target, legacy_target):
        if source_resolved == target.resolve():
            skipped_targets.append(str(target))
            continue
        shutil.copy2(existing_source, target)
        copied_targets.append(str(target))

    return {
        "synced": True,
        "source": str(existing_source),
        "targets": [str(final_target), str(legacy_target)],
        "copied_targets": copied_targets,
        "skipped_same_file_targets": skipped_targets,
    }


def _validate_phase1_completion(result: Dict[str, Any], sync_report: Dict[str, Any]) -> Dict[str, Any]:
    """判定 Phase1 是否完整完成，防止非完整 END 进入 Phase2。"""
    runtime_root = Path(os.environ.get("METABOAGENT_RUNTIME_ROOT", ""))
    final_dataset = runtime_root / "phase1" / "final" / "selected_features_final.csv"
    legacy_dataset = runtime_root / "phase1" / "legacy" / "selected_features_final.csv"
    summary = result.get("phase1_summary", {}) if isinstance(result, dict) else {}
    current_data_path = result.get("current_data_path", "") if isinstance(result, dict) else ""
    final_features = (
        result.get("context_variables", {}).get("final_selected_features", [])
        if isinstance(result, dict) and isinstance(result.get("context_variables"), dict)
        else []
    )
    completed = bool(result.get("completed")) if isinstance(result, dict) else False
    completed_effectively = bool(summary.get("completed_effectively"))
    had_error = bool(result.get("error") or summary.get("had_error")) if isinstance(result, dict) else True
    final_dataset_exists = final_dataset.exists() or legacy_dataset.exists()
    current_data_exists = bool(current_data_path and Path(current_data_path).exists())
    valid = (
        not had_error
        and (completed or completed_effectively)
        and final_dataset_exists
        and (bool(final_features) or current_data_exists)
        and bool(sync_report.get("synced"))
    )
    return {
        "valid": valid,
        "completed": completed,
        "completed_effectively": completed_effectively,
        "had_error": had_error,
        "final_dataset_exists": final_dataset_exists,
        "current_data_exists": current_data_exists,
        "final_feature_count": len(final_features or []),
        "sync_report": sync_report,
        "final_dataset": str(final_dataset),
        "legacy_dataset": str(legacy_dataset),
        "current_data_path": current_data_path,
        "last_error": result.get("last_error") or result.get("error") or summary.get("last_error") if isinstance(result, dict) else "Invalid Phase1 result",
    }


async def run_phase1_real(
    websocket: WebSocket,
    data_path: str,
    target_column: str,
    phase0_result: Dict[str, Any],
    max_steps: int = 100,
    id_column: Optional[str] = None,
    positive_class: Optional[str] = None,
    negative_class: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """执行真实的 Phase 1 流程"""
    
    await send_log(websocket, "system", f"Phase 1 initialized. Dataset: {data_path}")
    await send_state(websocket, "phase1_running")
    await send_activity_update(
        websocket,
        "phase1",
        "Inspecting the dataset",
        "Reading the schema, sample count, feature count, and outcome column.",
        phase_progress=3,
    )
    
    try:
        # 初始化 LLM
        llm = ChatOpenAI(model=get_configured_llm_model(), temperature=0)
        
        # 加载 SOP
        sop_path = PROJECT_ROOT / "data" / "phase1_targeted_preprocessing_v2.json"
        sop_config = load_sop(str(sop_path))
        
        # 生成数据摘要
        await send_log(websocket, "system", "Loading and validating the dataset.")
        data_summary = get_data_summary(data_path, target_column)
        
        await send_log(websocket, "stdout", f"✓ Dataset shape: {data_summary['n_rows']} rows × {data_summary['n_cols']} columns")
        await send_activity_update(
            websocket,
            "phase1",
            "Dataset contract validated",
            f"Detected {data_summary['n_rows']} samples and {data_summary['n_cols']} columns. Preparing analysis context.",
            phase_progress=8,
        )

        # 先构建 Phase 0 输出，再注入初始状态和记忆上下文
        feature_definitions = phase0_result.get("feature_definitions", {}) or {}
        phase0_output = {
            "enriched_pathways": feature_definitions.get("target_pathways", []),
            "confirmed_biomarkers": phase0_result.get("confirmed_biomarkers", []) or [],
            "final_priors": phase0_result.get("final_priors", []) or [],
            "feature_definitions": {
                "target_pathways": feature_definitions.get("target_pathways", []),
                "target_metabolites": feature_definitions.get(
                    "target_metabolites",
                    phase0_result.get("final_priors", []) or [],
                ),
            },
            "disease_name": phase0_result.get("disease_name", ""),
            "phase0_completed": True,
        }

        initial_context_variables = {
            "phase0_output": phase0_output,
            "id_column": id_column,
            "sample_id_column": id_column,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "label_mapping": dict(label_mapping or {}),
            "label_encoding": "positive_class_as_1" if positive_class and negative_class else "default",
        }
        orchestrator = MetaboMemoryOrchestrator()
        active_scenario = get_config().get_active_scenario()
        dataset_fingerprint = orchestrator.build_dataset_fingerprint(
            data_path=data_path,
            target_column=target_column,
            data_summary=data_summary,
            context_variables=initial_context_variables,
            disease_name=phase0_output.get("disease_name", ""),
            clinical_scenario=active_scenario,
        )
        phase1_memory_pack = orchestrator.prepare_phase1_memory_pack(dataset_fingerprint)
        phase1_memory_debug = orchestrator.summarize_retrieval_trace(
            orchestrator.get_last_retrieval_trace("phase1")
        )
        memory_context = {
            "memory_enabled": orchestrator.is_enabled(),
            "dataset_fingerprint": dataset_fingerprint,
            "phase1_memory_pack": phase1_memory_pack,
            "retrieval_trace": {
                "phase1_provenance_cases": phase1_memory_pack.get("provenance_cases", []),
                "phase1_debug": phase1_memory_debug,
            },
        }

        # 创建初始状态
        initial_state = create_initial_state(
            sop_config=sop_config,
            initial_data_path=data_path,
            initial_data_summary=data_summary,
            target_column=target_column,
            max_retries=5,
            llm=llm,
            initial_context_variables=initial_context_variables,
            dataset_fingerprint=dataset_fingerprint,
            memory_context=memory_context,
            phase1_memory_pack=phase1_memory_pack,
        )
        
        await send_log(websocket, "system", f"✓ Phase 0 evidence integrated: {len(phase0_output['confirmed_biomarkers'])} confirmed biomarkers")
        await send_log(
            websocket,
            "stdout",
            f"✓ Phase 1 memory initialized: {len(phase1_memory_pack.get('provenance_cases', []))} matched cases"
        )
        await send_log(
            websocket,
            "stdout",
            "  - Phase 1 memory debug: "
            f"considered={phase1_memory_debug.get('considered_case_count', 0)}, "
            f"filtered={phase1_memory_debug.get('filtered_case_count', 0)}, "
            f"selected={phase1_memory_debug.get('selected_case_ids', [])}"
        )
        if phase1_memory_pack.get("semantic_entry_id"):
            await send_log(
                websocket,
                "stdout",
                "  - Phase 1 strategy semantic: "
                f"entry={phase1_memory_pack.get('semantic_entry_id', '')}, "
                f"confidence={float(phase1_memory_pack.get('semantic_confidence', 0.0) or 0.0):.2f}, "
                f"key={phase1_memory_pack.get('semantic_strategy_key', '')}"
            )
        
        await send_activity_update(
            websocket,
            "phase1",
            "Preparing the analysis plan",
            "Combining the fixed SOP, prior evidence, and matched analysis memory.",
            phase_progress=12,
        )

        # 创建 graph
        graph = build_phase1_graph(llm)
        
        await send_log(websocket, "system", "Running the Phase 1 preprocessing and modeling workflow.")
        
        # 在后台线程中运行
        import concurrent.futures
        
        result = None
        step_count = 0
        current_step_id: Optional[str] = None
        current_step_title = "Preparing the fixed analysis protocol"
        current_step_ordinal: Optional[int] = None
        latest_phase1_progress = 12

        phase1_step_catalog: List[Dict[str, str]] = []
        for stage in sop_config.get("stages", []) or []:
            for step in stage.get("steps", []) or []:
                step_id = str(step.get("step_id", "") or "")
                if not step_id:
                    continue
                phase1_step_catalog.append({
                    "stage_id": str(stage.get("stage_id", "") or ""),
                    "step_id": step_id,
                    "step_name": str(
                        step.get("instruction")
                        or step.get("step_instruction")
                        or f"Phase 1 step {step_id}"
                    ),
                })
        phase1_step_lookup = {
            item["step_id"]: {**item, "index": index + 1}
            for index, item in enumerate(phase1_step_catalog)
        }
        phase1_step_total = len(phase1_step_catalog)
        
        # 获取当前事件循环（在主线程中）
        loop = asyncio.get_event_loop()
        
        def run_graph():
            """Run the graph and emit reliable running/completed step events."""
            nonlocal result, step_count, current_step_id, current_step_title
            nonlocal current_step_ordinal, latest_phase1_progress

            reported_history_count = 0
            
            for chunk in graph.stream(initial_state, config={"recursion_limit": max_steps + 10}):
                step_count += 1
                
                for node_name, node_output in chunk.items():
                    result = node_output
                    print(f"[DEBUG] Node: {node_name}")

                    if isinstance(node_output, dict):
                        stage_id = node_output.get('current_stage_id')
                        step_index = node_output.get('current_step_index')
                        step_id = None
                        step_name = None

                        if stage_id is not None and step_index is not None:
                            sop_stages = node_output.get('sop_config', {}).get('stages', [])
                            current_stage = next(
                                (stage for stage in sop_stages if str(stage.get('stage_id')) == str(stage_id)),
                                None,
                            )
                            if current_stage:
                                steps_in_stage = current_stage.get('steps', [])
                                if 0 <= int(step_index) < len(steps_in_stage):
                                    current_step = steps_in_stage[int(step_index)]
                                    step_id = str(current_step.get('step_id', f"{stage_id}.{step_index}"))
                                    step_name = str(
                                        current_step.get('instruction')
                                        or current_step.get('step_instruction')
                                        or f"Phase 1 step {step_id}"
                                    )

                        if step_id and step_id != current_step_id:
                            current_step_id = step_id
                            metadata = phase1_step_lookup.get(step_id, {})
                            ordinal = int(metadata.get("index", 1) or 1)
                            total = max(phase1_step_total, ordinal)
                            running_progress = 12 + int(((ordinal - 1) / max(total, 1)) * 80)
                            current_step_title = step_name or str(metadata.get("step_name") or f"Phase 1 step {step_id}")
                            current_step_ordinal = ordinal
                            latest_phase1_progress = running_progress
                            asyncio.run_coroutine_threadsafe(
                                send_step_progress(
                                    websocket,
                                    str(stage_id or metadata.get("stage_id", "")),
                                    step_id,
                                    current_step_title,
                                    "running",
                                    phase_progress=running_progress,
                                    step_index=ordinal,
                                    step_total=total,
                                ),
                                loop
                            ).result()

                        execution_history = list(node_output.get('execution_history', []) or [])
                        if len(execution_history) > reported_history_count:
                            for execution in execution_history[reported_history_count:]:
                                executed_step_id = str(execution.get('step_id', 'unknown'))
                                metadata = phase1_step_lookup.get(executed_step_id, {})
                                ordinal = int(metadata.get("index", reported_history_count + 1) or 1)
                                total = max(phase1_step_total, ordinal)
                                success = bool(execution.get('success'))
                                status = "completed" if success else "retrying"
                                completed_progress = 12 + int((ordinal / max(total, 1)) * 80)
                                latest_phase1_progress = completed_progress
                                asyncio.run_coroutine_threadsafe(
                                    send_step_progress(
                                        websocket,
                                        str(metadata.get("stage_id", stage_id or "")),
                                        executed_step_id,
                                        str(metadata.get("step_name") or f"Phase 1 step {executed_step_id}"),
                                        status,
                                        phase_progress=completed_progress,
                                        step_index=ordinal,
                                        step_total=total,
                                    ),
                                    loop,
                                ).result()
                                elapsed = float(execution.get('execution_time', 0) or 0)
                                log_text = (
                                    f"✓ Step {executed_step_id} completed in {elapsed:.2f}s"
                                    if success
                                    else f"Step {executed_step_id} requires a retry after {elapsed:.2f}s"
                                )
                                asyncio.run_coroutine_threadsafe(
                                    send_log(websocket, "stdout" if success else "warning", log_text),
                                    loop,
                                ).result()
                            reported_history_count = len(execution_history)
                    
                    # 检查是否完成
                    if result and result.get('completed'):
                        break
                
                if step_count >= max_steps or (result and result.get('completed')):
                    break
        
        phase1_started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, run_graph)
            while not future.done():
                done, _ = await asyncio.wait({future}, timeout=20)
                if done:
                    break
                elapsed_seconds = int(time.monotonic() - phase1_started)
                try:
                    await send_activity_update(
                        websocket,
                        "phase1",
                        current_step_title,
                        "This operation is still running. Long model-training and resampling steps can take several minutes; no action is required.",
                        phase_progress=latest_phase1_progress,
                        step_id=current_step_id,
                        step_index=current_step_ordinal,
                        step_total=phase1_step_total or None,
                        elapsed_seconds=elapsed_seconds,
                    )
                except Exception:
                    # Keep the analysis alive if the browser disconnects mid-run.
                    pass
            await future
        
        # 发送 Phase1 结果
        if result:
            await send_activity_update(
                websocket,
                "phase1",
                "Validating Phase 1 outputs",
                "Checking the selected-feature dataset, model artifacts, and handoff contract.",
                phase_progress=94,
            )
            if isinstance(result, dict):
                result["phase1_summary"] = build_phase1_summary(result)
                phase0_summary = orchestrator.build_phase0_summary(phase0_result)
                memory_case = orchestrator.build_phase1_memory_case(
                    disease_name=phase0_output.get("disease_name", ""),
                    clinical_scenario=active_scenario,
                    dataset_fingerprint=dataset_fingerprint,
                    phase0_summary=phase0_summary,
                    phase1_summary=result.get("phase1_summary", {}),
                    tags=["phase1", "episodic"],
                )
                memory_case_id = orchestrator.write_memory_case(memory_case)
                result["memory_case_id"] = memory_case_id
            sync_report = _sync_phase1_outputs_to_runtime(result if isinstance(result, dict) else {})
            validation = _validate_phase1_completion(result if isinstance(result, dict) else {}, sync_report)
            if isinstance(result, dict):
                result["phase1_output_sync"] = sync_report
                result["phase1_completion_validation"] = validation
            await send_log(
                websocket,
                "stdout" if sync_report.get("synced") else "stderr",
                f"{'✓' if sync_report.get('synced') else '✗'} Phase 1 output synchronization: {sync_report}"
            )
            await send_log(
                websocket,
                "stdout" if validation.get("valid") else "stderr",
                f"{'✓' if validation.get('valid') else '✗'} Phase 1 completion validation: {validation}"
            )
            await send_phase1_result(websocket, result)
            if not validation.get("valid"):
                raise RuntimeError(
                    "Phase 1 did not complete its required handoff, so Phase 2 was blocked."
                    f" completed={validation.get('completed')},"
                    f" completed_effectively={validation.get('completed_effectively')},"
                    f" had_error={validation.get('had_error')},"
                    f" final_dataset_exists={validation.get('final_dataset_exists')},"
                    f" final_feature_count={validation.get('final_feature_count')},"
                    f" last_error={validation.get('last_error')}"
                )
            if isinstance(result, dict) and result.get("memory_case_id"):
                await send_log(
                    websocket,
                    "stdout",
                    f"✓ Phase 1 episodic memory saved: {result['memory_case_id']}"
                )
            await send_activity_update(
                websocket,
                "phase1",
                "Preprocessing and baseline modeling complete",
                "The validated feature dataset and baseline model artifacts are ready for panel optimization.",
                status="completed",
                phase_progress=100,
            )
            await send_log(websocket, "system", "Phase 1 completed.")
        
        return result
        
    except Exception as e:
        await send_log(websocket, "stderr", f"Phase 1 failed: {str(e)}")
        raise


async def run_phase2_real(
    websocket: WebSocket,
    phase0_result: Dict[str, Any],
    phase1_result: Dict[str, Any],
    target_column_override: Optional[str] = None,
    beam_width: int = 1,
    k_folds: int = 5,
    epsilon: float = 0.005,
    patience: int = 5,
    positive_class: Optional[str] = None,
) -> Dict[str, Any]:
    """执行真实的 Phase 2 流程（最小主链路接入版）"""

    await send_log(websocket, "system", "Phase 2 initialized.")
    await send_state(websocket, "phase2_running")
    await send_activity_update(
        websocket,
        "phase2",
        "Preparing biomarker panel search",
        "Validating the Phase 1 handoff and candidate feature pool.",
        phase_progress=5,
    )

    try:
        cfg = get_config()
        orchestrator = MetaboMemoryOrchestrator()
        active_scenario = cfg.get_active_scenario()
        phase1_paths = cfg.get_phase1_paths() if hasattr(cfg, "get_phase1_paths") else {}
        io_policy = cfg.get_phase1_io_policy() if hasattr(cfg, "get_phase1_io_policy") else {}

        canonical_path = phase1_paths.get("selected_features_csv") or "output/phase1/final/selected_features_final.csv"
        legacy_path = phase1_paths.get("legacy_selected_features") or cfg.get_path("phase1_selected_features")
        prefer_new_read = bool(io_policy.get("prefer_new_read", False))
        fallback_old_read = bool(io_policy.get("fallback_old_read", True))

        candidates = []
        if prefer_new_read:
            candidates.extend([canonical_path, legacy_path])
        else:
            candidates.append(legacy_path)
            if fallback_old_read:
                candidates.append(canonical_path)
            else:
                candidates.insert(0, canonical_path)

        phase1_data_path = ""
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                phase1_data_path = candidate
                break
        if not phase1_data_path:
            raise FileNotFoundError("Phase 1 output `selected_features_final.csv` was not found")

        await send_log(websocket, "stdout", f"✓ Phase 2 input dataset: {phase1_data_path}")

        import pandas as pd

        df = pd.read_csv(phase1_data_path)
        target_column = target_column_override or cfg.get_target_column()
        exclude_cols = {"target", "Group", "Sample", "SampleID", "Sample_ID", "ID", "Label", "Class", "Unnamed: 0"}
        candidate_pool = [
            col for col in df.columns
            if col != target_column and col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        sorted_base_pool = list(candidate_pool)

        if not candidate_pool:
            raise ValueError("The Phase 2 candidate feature pool is empty")

        await send_log(websocket, "stdout", f"✓ Phase 2 candidate features: {len(candidate_pool)}")
        await send_activity_update(
            websocket,
            "phase2",
            "Candidate pool ready",
            f"Evaluating combinations from {len(candidate_pool)} Phase 1 features under the fixed clinical objective.",
            phase_progress=12,
        )

        taxonomy_map = _load_taxonomy_map()
        pathway_map = _load_pathway_map()
        priors_dict = _load_priors_dict()
        checker_tool = None
        try:
            checker_tool = ReactionCheckerTool()
        except Exception as checker_error:
            await send_log(websocket, "warning", f"Reaction graph support is unavailable and will be skipped: {checker_error}")

        phase1_summary = phase1_result.get("phase1_summary", {}) or {}
        context_variables = dict(phase1_result.get("context_variables", {}) or {})
        # Phase 2 must receive the complete Phase 0 evidence object even when a
        # legacy Phase 1 result omitted it from serialized context_variables.
        context_variables["phase0_output"] = dict(phase0_result or {})
        if positive_class:
            context_variables["positive_class"] = positive_class
        champion_model_family = context_variables.get("best_model")
        dataset_fingerprint = phase1_result.get("dataset_fingerprint") or orchestrator.build_dataset_fingerprint(
            data_path=phase1_data_path,
            target_column=target_column,
            context_variables={"phase0_output": phase0_result},
            disease_name=phase0_result.get("disease_name", ""),
            clinical_scenario=active_scenario,
        )

        # Phase 2 搜索包含大量同步的模型评估和特征组合计算。
        # 直接在 async WebSocket handler 中调用会阻塞事件循环，前端在很长时间内
        # 收不到任何日志，看起来像任务卡死。放到工作线程，并每 15 秒发送心跳。
        await send_log(websocket, "system", "Phase 2 multi-objective search started in the background.")
        await send_activity_update(
            websocket,
            "phase2",
            "Optimizing the biomarker panel",
            "Comparing predictive performance, biological relevance, redundancy, and panel cost.",
            phase_progress=18,
        )
        search_task = asyncio.create_task(
            asyncio.to_thread(
                orchestrator.run_phase2_search_with_memory,
                data_path=phase1_data_path,
                target_column=target_column,
                candidate_pool=candidate_pool,
                sorted_base_pool=sorted_base_pool,
                taxonomy_map=taxonomy_map,
                pathway_map=pathway_map,
                priors_dict=priors_dict,
                checker_tool=checker_tool,
                clinical_scenario=active_scenario,
                disease_name=phase0_result.get("disease_name", ""),
                context_variables=context_variables,
                dataset_fingerprint=dataset_fingerprint,
                phase0_summary=orchestrator.build_phase0_summary(phase0_result),
                ag_results_path=cfg.get_phase1_path("autogluon_results") or "data/autogluon_training_results.json",
                champion_model_family=champion_model_family,
                beam_width=beam_width,
                k_folds=k_folds,
                epsilon=epsilon,
                patience=patience,
                positive_class=positive_class,
                verbose_search_trace=True,
                trace_head_limit=50,
                writeback=True,
            )
        )
        search_started = time.monotonic()
        while not search_task.done():
            done, _ = await asyncio.wait({search_task}, timeout=15)
            if done:
                break
            elapsed_seconds = int(time.monotonic() - search_started)
            try:
                await send_activity_update(
                    websocket,
                    "phase2",
                    "Optimizing the biomarker panel",
                    "The deterministic search is still evaluating candidate panels. No action is required.",
                    phase_progress=18,
                    elapsed_seconds=elapsed_seconds,
                )
            except Exception:
                # 浏览器断开时不终止后台分析任务，让任务状态和产物继续落盘。
                pass
        result = await search_task

        if isinstance(result, dict):
            phase2_memory_debug = dict((result.get("memory_debug", {}) or {}).get("phase2", {}) or {})
            await send_phase2_result(websocket, result)
            winner_count = len(result.get('features', []) or [])
            await send_activity_update(
                websocket,
                "phase2",
                "Validating the selected panel",
                f"A {winner_count}-feature panel was selected. Final metrics and provenance are being recorded.",
                phase_progress=94,
            )
            await send_log(websocket, "stdout", f"✓ Phase 2 winning panel features: {winner_count}")
            if result.get("memory_case_id"):
                await send_log(websocket, "stdout", f"✓ Phase 2 episodic memory saved: {result['memory_case_id']}")
            await send_log(
                websocket,
                "stdout",
                "  - Phase 2 memory debug: "
                f"considered={phase2_memory_debug.get('considered_case_count', 0)}, "
                f"filtered={phase2_memory_debug.get('filtered_case_count', 0)}, "
                f"selected={phase2_memory_debug.get('selected_case_ids', [])}"
            )
            phase2_summary = result.get("phase2_summary", {}) or {}
            memory_prior_summary = phase2_summary.get("memory_prior_summary", {}) or {}
            if (
                memory_prior_summary.get("retrieved_semantic_entry_id")
                or memory_prior_summary.get("retrieved_prior_confidence")
            ):
                await send_log(
                    websocket,
                    "stdout",
                    "  - Phase 2 strategy semantic: "
                    f"entry={memory_prior_summary.get('retrieved_semantic_entry_id', '')}, "
                    f"confidence={float(memory_prior_summary.get('retrieved_semantic_confidence', 0.0) or 0.0):.2f}, "
                    f"key={memory_prior_summary.get('retrieved_semantic_strategy_key', '')}, "
                    f"anchor_hints={len(memory_prior_summary.get('retrieved_anchor_feature_hints', []) or [])}"
                )
        await send_activity_update(
            websocket,
            "phase2",
            "Biomarker panel optimization complete",
            "The winning panel and cross-validated performance evidence are ready.",
            status="completed",
            phase_progress=100,
        )
        await send_log(websocket, "system", "Phase 2 completed.")
        return result

    except Exception as e:
        await send_log(websocket, "stderr", f"Phase 2 failed: {str(e)}")
        raise


# ============================================================================
# API v1：文件、配置、任务与结果接口
# ============================================================================

def _recalculate_overall_progress(job_id: str, progress: Dict[str, Any]) -> Dict[str, int]:
    """Calculate overall progress from only the phases selected for this job."""
    normalized = {
        key: max(0, min(100, int(value or 0)))
        for key, value in dict(progress or {}).items()
        if key in {"phase0", "phase1", "phase2", "phase3"}
    }
    request_payload = _read_json(_job_dir(job_id) / "job_config.json", {})
    requested = list((request_payload.get("analysis", {}) or {}).get("phases", []) or [])
    selected = [phase for phase in requested if phase in normalized]
    if not selected:
        selected = ["phase1", "phase2"]
    normalized["overall"] = int(round(sum(normalized.get(phase, 0) for phase in selected) / len(selected)))
    return normalized


class JobWebSocketProxy:
    """包装真实 WebSocket，将旧消息契约同步持久化为 job 状态与日志。"""

    def __init__(self, websocket: WebSocket, job_id: str):
        self.websocket = websocket
        self.job_id = job_id
        self.client_connected = True

    async def send_json(self, message: Dict[str, Any]) -> None:
        message = dict(message or {})
        message.setdefault("job_id", self.job_id)
        message.setdefault("timestamp", get_timestamp())

        job = _read_json(_job_dir(self.job_id) / "job_status.json", {})
        if not job:
            await self.websocket.send_json(message)
            return

        msg_type = message.get("type")
        if msg_type == "agent_state":
            state = str(message.get("state", ""))
            phase = state.replace("_running", "") if state.endswith("_running") else job.get("current_phase")
            job["current_phase"] = phase
            job["status"] = "completed" if state == "completed" else "running"
            progress = dict(job.get("progress") or {})
            phase_keys = ["phase0", "phase1", "phase2", "phase3"]
            if state.endswith("_running") and phase in phase_keys:
                for key in phase_keys:
                    if key == phase:
                        break
                    if progress.get(key, 0) > 0:
                        progress[key] = 100
                progress[phase] = max(int(progress.get(phase, 0)), 1)
                message["status"] = "running"
                message["current_phase"] = phase
            progress = _recalculate_overall_progress(self.job_id, progress)
            message["progress"] = progress
            job["updated_at"] = get_timestamp()
            job["progress"] = progress
            _save_job(job)
        elif msg_type == "log_message":
            log = message.get("log", {}) or {}
            log.setdefault("phase", str(job.get("current_phase", "runtime")))
            message["log"] = log
            message["phase"] = log.get("phase")
            _append_job_log(
                self.job_id,
                str(log.get("type", log.get("level", "info"))),
                str(job.get("current_phase", "runtime")),
                str(log.get("content", "")),
            )
        elif msg_type in {"phase0_result", "phase1_result", "phase2_result", "phase_result"}:
            phase_key = str(message.get("phase") or msg_type.replace("_result", ""))
            result_path = _job_dir(self.job_id) / f"{phase_key}_result.json"
            _write_json(result_path, message.get("result", {}))
            progress = dict(job.get("progress") or {})
            progress[phase_key] = 100
            progress = _recalculate_overall_progress(self.job_id, progress)
            job["progress"] = progress
            job["updated_at"] = get_timestamp()
            message["progress"] = progress
            _save_job(job)
        elif msg_type in {"step_progress", "activity_update"}:
            phase_key = str(message.get("phase") or job.get("current_phase") or "runtime")
            job["current_step"] = message.get("step_id")
            job["current_step_name"] = message.get("step_name") or message.get("title")
            job["current_step_status"] = message.get("status")
            job["current_activity"] = {
                "phase": phase_key,
                "title": message.get("title") or message.get("step_name"),
                "detail": message.get("detail"),
                "status": message.get("status"),
                "timestamp": message.get("timestamp"),
                "step_id": message.get("step_id"),
                "step_index": message.get("step_index"),
                "step_total": message.get("step_total"),
                "elapsed_seconds": message.get("elapsed_seconds"),
            }
            progress = dict(job.get("progress") or {})
            phase_progress = message.get("phase_progress")
            if phase_key in {"phase0", "phase1", "phase2", "phase3"} and phase_progress is not None:
                progress[phase_key] = max(int(progress.get(phase_key, 0) or 0), int(phase_progress))
            progress = _recalculate_overall_progress(self.job_id, progress)
            job["progress"] = progress
            job["updated_at"] = get_timestamp()
            message["current_phase"] = job.get("current_phase")
            message["current_step"] = message.get("step_id")
            message["progress"] = progress
            _save_job(job)

        if self.client_connected:
            try:
                await self.websocket.send_json(message)
            except (WebSocketDisconnect, RuntimeError):
                # The analysis is a server-side job. Closing or refreshing the
                # browser must never abort computation or artifact persistence.
                self.client_connected = False


class _ServerJobTransport:
    """No-op transport used when a job runs without a live WebSocket client."""

    async def send_json(self, message: Dict[str, Any]) -> None:
        # JobWebSocketProxy persists state, activity, logs, and phase results
        # before forwarding a message.  The server-side worker does not need a
        # network client, so this method intentionally does nothing.
        return None


def _schedule_job(job_id: str) -> None:
    """Start a queued job exactly once in the current FastAPI event loop."""
    existing = _JOB_TASKS.get(job_id)
    if existing is not None and not existing.done():
        return

    task = asyncio.create_task(_run_job_pipeline(job_id, _ServerJobTransport()))
    _JOB_TASKS[job_id] = task

    def _finish(completed_task: asyncio.Task) -> None:
        _JOB_TASKS.pop(job_id, None)
        if completed_task.cancelled():
            return
        try:
            completed_task.result()
        except Exception as exc:
            # _run_job_pipeline already records the failed job state and log.
            # Consume the exception here so the server does not emit an
            # unhandled-task warning that obscures the actual analysis error.
            print(f"[JOB {job_id}] background execution ended: {exc}")

    task.add_done_callback(_finish)


def _normalize_figure_style(style: str) -> str:
    """将前端风格 key 映射到 Phase3 journal style key。"""
    mapping = {
        "nature": "nature_minimal",
        "nature_minimal": "nature_minimal",
        "science_advances": "science_advances",
        "cell_metabolism": "cell_metabolism",
        "clinical_report": "clinical_report",
    }
    return mapping.get(str(style or "").strip(), str(style or "nature_minimal").strip() or "nature_minimal")


def _build_runtime_config(
    job_id: str,
    request: JobCreateRequest,
    file_record: Dict[str, Any],
    target_column: str,
    data_path: str,
    label_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """为每个 Job 生成带用户参数覆盖的 runtime_config.yaml。"""
    import yaml

    base_path = PROJECT_ROOT / "config.yaml"
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    runtime_config = copy.deepcopy(base_config or {})

    runtime_config.setdefault("paths", {})
    runtime_config["paths"]["test_data_path"] = data_path
    runtime_config["paths"]["target_column"] = target_column
    runtime_config["paths"]["test_disease_name"] = request.analysis.disease_name

    scenario = request.analysis.clinical_scenario
    if scenario:
        scenarios = runtime_config.get("evaluation", {}).get("scenarios", {})
        if scenario not in scenarios:
            raise HTTPException(status_code=400, detail=f"Unknown clinical scenario: {scenario}")
        runtime_config.setdefault("evaluation", {})["active_scenario"] = scenario

    runtime_config.setdefault("phase0", {}).setdefault("defaults", {})
    runtime_config["phase0"]["defaults"].update({
        "use_cache": request.phase0.use_cache,
        "max_candidates": request.phase0.max_candidates,
    })

    runtime_config.setdefault("phase3", {}).setdefault("visualization", {})
    phase3_viz = runtime_config["phase3"]["visualization"]
    phase3_viz["style_preset"] = _normalize_figure_style(request.phase3.figure_style)
    phase3_viz["journal_style"] = phase3_viz["style_preset"]
    phase3_viz["output_language"] = request.phase3.language
    phase3_viz["output_formats"] = list(request.phase3.output_formats or [])
    # Retire two dense composite panels that duplicate clearer single-panel
    # evidence and add visual noise to the reader-facing report.
    phase3_viz["enable_phase2_clinical_validation_composite"] = False
    phase3_viz["enable_phase2_radar_validation_composite"] = False

    figure_key_map = {
        "phase0_prior_evidence_atlas": "enable_phase0_prior_evidence_atlas",
        "phase1_stability_landscape": "enable_stability_landscape",
        "phase1_method_support_heatmap": "enable_method_feature_heatmap",
        "phase1_final_panel_correlation_heatmap": "enable_phase1_final_panel_correlation_heatmap",
        "autogluon_roc": "enable_autogluon_roc",
        "phase2_radar": "enable_radar_4d",
        "radar_4d": "enable_radar_4d",
        "roc": "enable_final_roc",
        "final_roc": "enable_final_roc",
        "holdout_roc": "enable_final_holdout_roc",
        "dca": "enable_dca",
        "calibration": "enable_calibration",
        "threshold_performance": "enable_threshold_performance",
        "shap": "enable_shap",
        "rcs": "enable_rcs",
        "incremental_value_summary": "enable_incremental_value_summary",
        "phase2_clinical_validation_composite": "enable_phase2_clinical_validation_composite",
        "phase2_radar_validation_composite": "enable_phase2_radar_validation_composite",
        "phase3_shap_interpretation_composite": "enable_phase3_shap_interpretation_composite",
    }
    for frontend_key, enabled in (request.phase3.figures or {}).items():
        config_key = figure_key_map.get(frontend_key, frontend_key if frontend_key.startswith("enable_") else "")
        if config_key:
            phase3_viz[config_key] = bool(enabled)

    runtime_config.setdefault("job_runtime", {})
    label_metadata = dict(label_metadata or {})
    runtime_config["job_runtime"] = {
        "job_id": job_id,
        "file_id": request.file_id,
        "filename": file_record.get("filename"),
        "id_column": request.dataset.id_column,
        "group_column": request.dataset.group_column,
        "positive_class": label_metadata.get("positive_class") or request.dataset.positive_class,
        "negative_class": label_metadata.get("negative_class"),
        "label_mapping": dict(label_metadata.get("label_mapping", {}) or {}),
        "label_values": list(label_metadata.get("labels", []) or []),
        "is_binary": bool(label_metadata.get("is_binary", False)),
        "random_seed": request.runtime.random_seed,
        "save_intermediate": request.runtime.save_intermediate,
        "memory_enabled": request.runtime.memory_enabled,
    }

    runtime_config_path = _job_dir(job_id) / "runtime_config.yaml"
    runtime_config_path.write_text(yaml.safe_dump(_json_safe(runtime_config), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(runtime_config_path)


def _normalize_requested_phases(phases: List[str]) -> List[str]:
    """取消 Phase3-only 模式：选择 Phase3 时自动补齐 Phase1 和 Phase2。"""
    normalized: List[str] = []
    for phase in phases:
        if phase not in normalized:
            normalized.append(phase)

    if "phase3" in normalized:
        for required_phase in ("phase1", "phase2"):
            if required_phase not in normalized:
                normalized.append(required_phase)

    return normalized


def _prepare_phase3_state_files(
    job_id: str,
    phase0_result: Dict[str, Any],
    phase1_result: Dict[str, Any],
    phase2_result: Dict[str, Any],
    data_path: str,
    target_column: str,
) -> Dict[str, str]:
    """生成 Phase3 可读取的 Phase0/1/2 状态文件。"""
    job_path = _job_dir(job_id)
    phase0_path = job_path / "phase0_result.json"
    phase1_path = job_path / "phase1_result.json"
    phase2_path = job_path / "phase2_result_for_phase3.json"

    # Phase 2 still writes legacy artifacts under phase1/legacy/artifacts in
    # some runs, while the Phase 3 visualizers resolve the canonical
    # phase1/artifacts path.  Materialize missing files into that canonical
    # location so report generation is independent of the legacy layout.
    canonical_artifacts = job_path / "runtime" / "phase1" / "artifacts"
    canonical_artifacts.mkdir(parents=True, exist_ok=True)
    artifact_roots = [
        job_path / "runtime" / "phase1" / "legacy" / "artifacts",
        job_path / "runtime" / "phase1" / "intermediate" / "latest" / "artifacts",
    ]
    for source_root in artifact_roots:
        if not source_root.exists():
            continue
        for source_path in source_root.rglob("*"):
            if not source_path.is_file():
                continue
            relative = source_path.relative_to(source_root)
            target_path = canonical_artifacts / relative
            if not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)

    phase2_payload = dict(phase2_result or {})
    if "final_result" not in phase2_payload:
        phase2_payload["final_result"] = {
            "features": phase2_payload.get("features", []) or phase2_payload.get("selected_features", []),
            "perf": phase2_payload.get("perf", 0.0),
            "bio": phase2_payload.get("bio", 0.0),
            "corr": phase2_payload.get("corr", 0.0),
            "cost": phase2_payload.get("cost", 0.0),
            "roc_auc": phase2_payload.get("roc_auc", phase2_payload.get("perf", 0.0)),
            "comprehensive_metrics": phase2_payload.get("comprehensive_metrics", {}),
            "selected_model": phase2_payload.get("selected_model") or phase2_payload.get("champion_model_family") or "RandomForest",
        }
    phase2_payload.setdefault("phase0_output", phase0_result or {})
    phase2_payload.setdefault("data_path", data_path)
    phase2_payload.setdefault("target_column", target_column)

    _write_json(phase0_path, phase0_result or {})
    _write_json(phase1_path, phase1_result or {})
    _write_json(phase2_path, phase2_payload)
    return {
        "phase0": str(phase0_path),
        "phase1": str(phase1_path),
        "phase2": str(phase2_path),
        "phase1_artifacts": str(canonical_artifacts),
    }


def _collect_phase3_outputs(job_id: str, report_path: str) -> Dict[str, Any]:
    """收集 Phase3 生成的报告、manifest 和图表。"""
    job_path = _job_dir(job_id)
    report_root = job_path / "runtime" / "output" / "reports"
    figures_root = job_path / "runtime" / "output" / "figures"
    manifest_path = report_root / "figure_manifest.json"
    manifest = _read_json(manifest_path, {})

    figures = []
    for figure in manifest.get("figures", []) if isinstance(manifest, dict) else []:
        if (
            "plot_final_holdout_calibration" in str(figure.get("source_task") or "")
            or "final_holdout_calibration" in str(figure.get("figure_id") or "")
            or "Internal Holdout Calibration Plot" in str(figure.get("title") or "")
        ):
            continue
        primary = figure.get("primary_output") or {}
        primary_path = primary.get("path") if isinstance(primary, dict) else ""
        if primary_path:
            abs_primary = (job_path / "runtime" / primary_path).resolve() if not os.path.isabs(primary_path) else Path(primary_path)
            if abs_primary.exists():
                rel = abs_primary.relative_to(job_path).as_posix()
                variants = []
                variant_candidates = [abs_primary]
                if figures_root.exists():
                    variant_candidates.extend(figures_root.rglob(f"{abs_primary.stem}.*"))
                variants_by_format = {}
                for variant_path in variant_candidates:
                    if not variant_path.is_file() or variant_path.suffix.lower() not in {".pdf", ".png", ".svg", ".jpg", ".jpeg", ".webp"}:
                        continue
                    try:
                        variant_rel = variant_path.relative_to(job_path).as_posix()
                    except ValueError:
                        continue
                    variant_format = variant_path.suffix.lower().lstrip(".")
                    path_parts = set(variant_path.parts)
                    if {"single_panels", "composite_panels"} & path_parts:
                        location_rank = 0
                    elif variant_path == abs_primary:
                        location_rank = 1
                    elif job_id in path_parts:
                        location_rank = 2
                    else:
                        location_rank = 3
                    candidate = {
                        "format": variant_format,
                        "path": variant_rel,
                        "download_url": f"/api/v1/jobs/{job_id}/download/{variant_rel}",
                        "_location_rank": location_rank,
                    }
                    existing = variants_by_format.get(variant_format)
                    if existing is None or candidate["_location_rank"] < existing["_location_rank"]:
                        variants_by_format[variant_format] = candidate
                variants = [{key: value for key, value in item.items() if key != "_location_rank"}
                            for item in variants_by_format.values()]
                variants.sort(key=lambda item: (item["format"] != "png", item["format"]))
                preview = next((item["download_url"] for item in variants if item["format"] == "png"), "")
                figures.append({
                    "figure_id": figure.get("figure_id"),
                    "title": figure.get("title"),
                    "source_task": figure.get("source_task"),
                    "primary_output": rel,
                    "download_url": f"/api/v1/jobs/{job_id}/download/{rel}",
                    "preview_url": preview,
                    "download_variants": variants,
                })

    report_abs = Path(report_path)
    report_rel = report_abs.relative_to(job_path).as_posix() if report_abs.exists() and str(report_abs).startswith(str(job_path)) else ""
    manifest_rel = manifest_path.relative_to(job_path).as_posix() if manifest_path.exists() else ""
    final_reports = []
    final_report_root = job_path / "runtime" / "output" / "final_delivery" / "current" / "report"
    if final_report_root.exists():
        for report_file in sorted(final_report_root.iterdir()):
            if not report_file.is_file() or report_file.name not in {"report.html", "report.md", "report.pdf"}:
                continue
            report_rel_path = report_file.relative_to(job_path).as_posix()
            final_reports.append({
                "name": report_file.name,
                "format": report_file.suffix.lower().lstrip("."),
                "path": report_rel_path,
                "size_bytes": report_file.stat().st_size,
                "download_url": f"/api/v1/jobs/{job_id}/download/{report_rel_path}",
            })
    data_files = []
    assets_root = figures_root / "assets"
    if assets_root.exists():
        for asset_path in sorted(path for path in assets_root.rglob("*") if path.is_file()):
            asset_rel = asset_path.relative_to(job_path).as_posix()
            data_files.append({
                "name": asset_path.relative_to(assets_root).as_posix(),
                "path": asset_rel,
                "format": asset_path.suffix.lower().lstrip(".") or "file",
                "size_bytes": asset_path.stat().st_size,
                "download_url": f"/api/v1/jobs/{job_id}/download/{asset_rel}",
            })
    return {
        "status": "completed",
        "report_path": report_rel,
        "report_download_url": f"/api/v1/jobs/{job_id}/download/{report_rel}" if report_rel else "",
        "final_reports": final_reports,
        "final_report_count": len(final_reports),
        "figure_manifest_path": manifest_rel,
        "figure_manifest_download_url": f"/api/v1/jobs/{job_id}/download/{manifest_rel}" if manifest_rel else "",
        "figures_dir": figures_root.relative_to(job_path).as_posix() if figures_root.exists() else "",
        "reports_dir": report_root.relative_to(job_path).as_posix() if report_root.exists() else "",
        "figure_bundle_download_url": f"/api/v1/jobs/{job_id}/figures.zip" if figures else "",
        "figures": figures,
        "figure_count": len(figures),
        "manifest_summary": manifest.get("summary", {}) if isinstance(manifest, dict) else {},
        "data_files": data_files,
        "asset_file_count": len(data_files),
    }


async def run_phase3_real(
    websocket: WebSocket,
    job_id: str,
    phase0_result: Dict[str, Any],
    phase1_result: Dict[str, Any],
    phase2_result: Dict[str, Any],
    data_path: str,
    target_column: str,
    runtime_config_path: str,
    figure_style: str = "nature",
    output_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """执行 Phase3 图表和报告生成，并返回前端可消费摘要。"""
    await send_log(websocket, "system", "Phase 3 initialized.")
    await send_state(websocket, "phase3_running")
    await send_activity_update(
        websocket,
        "phase3",
        "Preparing evidence outputs",
        "Collecting validated Phase 0–2 results for figures and reporting.",
        phase_progress=8,
    )

    job_path = _job_dir(job_id)
    runtime_root = job_path / "runtime"
    state_paths = _prepare_phase3_state_files(job_id, phase0_result, phase1_result, phase2_result, data_path, target_column)

    old_cwd = os.getcwd()
    env_backup = {
        "METABOAGENT_PHASE0_RESULT_PATH": os.environ.get("METABOAGENT_PHASE0_RESULT_PATH"),
        "METABOAGENT_PHASE1_RESULT_PATH": os.environ.get("METABOAGENT_PHASE1_RESULT_PATH"),
        "METABOAGENT_PHASE2_RESULT_PATH": os.environ.get("METABOAGENT_PHASE2_RESULT_PATH"),
        "METABOAGENT_RUN_TAG": os.environ.get("METABOAGENT_RUN_TAG"),
        "R_LIBS_USER": os.environ.get("R_LIBS_USER"),
    }
    os.environ["METABOAGENT_PHASE0_RESULT_PATH"] = state_paths["phase0"]
    os.environ["METABOAGENT_PHASE1_RESULT_PATH"] = state_paths["phase1"]
    os.environ["METABOAGENT_PHASE2_RESULT_PATH"] = state_paths["phase2"]
    os.environ["METABOAGENT_RUN_TAG"] = job_id
    project_r_library = PROJECT_ROOT / ".r_libs"
    if project_r_library.exists():
        existing_r_libraries = os.environ.get("R_LIBS_USER")
        os.environ["R_LIBS_USER"] = str(project_r_library) + (os.pathsep + existing_r_libraries if existing_r_libraries else "")

    try:
        await send_log(websocket, "system", f"Phase 3 figure style: {figure_style}")
        await send_log(websocket, "system", f"Phase 3 runtime config: {runtime_config_path}")
        await send_activity_update(
            websocket,
            "phase3",
            "Generating figures and report",
            "Rendering the selected scientific figures and assembling the evidence report.",
            phase_progress=25,
        )
        from src.agents.phase3.generate_final_report import generate_final_report

        runtime_root.mkdir(parents=True, exist_ok=True)
        os.chdir(runtime_root)
        report_path = await asyncio.to_thread(
            generate_final_report,
            config_path=runtime_config_path,
            use_mock_vlm=True,
            max_acval_retries=3,
        )
        report_abs = (runtime_root / report_path).resolve() if not os.path.isabs(report_path) else Path(report_path).resolve()
        final_report_result = {}
        await send_activity_update(
            websocket,
            "phase3",
            "Assembling final evidence reports",
            "Packaging the HTML, Markdown, and PDF reports for download.",
            phase_progress=78,
        )
        try:
            from src.agents.phase4.pipeline import generate_phase4_report

            # Phase 4 enforces a strict run-root boundary.  Phase 3 keeps its
            # handoff JSON at the job root for backward compatibility, so
            # stage a run-local copy before asking Phase 4 to collect sources.
            phase4_phase2_path = runtime_root / "output" / "phase2_result.json"
            phase4_phase2_path.parent.mkdir(parents=True, exist_ok=True)
            phase2_source_path = Path(state_paths["phase2"])
            if phase2_source_path.exists() and phase2_source_path.resolve() != phase4_phase2_path.resolve():
                shutil.copy2(phase2_source_path, phase4_phase2_path)

            final_report_result = await asyncio.to_thread(
                generate_phase4_report,
                config_path=runtime_config_path,
                phase2_result_path=str(phase4_phase2_path),
                run_root=str(runtime_root),
                run_id=job_id,
                render_html_output=True,
                render_pdf_output=True,
                use_llm=False,
            )
            await send_log(
                websocket,
                "stdout",
                "✓ Final evidence reports generated: report.html, report.md, report.pdf",
            )
        except Exception as report_error:
            final_report_result = {"status": "failed", "error": str(report_error)}
            await send_log(
                websocket,
                "warning",
                f"Final evidence report generation was unavailable; Phase 3 figures remain available: {report_error}",
            )
        phase3_result = _collect_phase3_outputs(job_id, str(report_abs))
        phase3_result.update({
            "figure_style": figure_style,
            "resolved_figure_style": _normalize_figure_style(figure_style),
            "output_formats": output_formats or [],
            "runtime_config_path": runtime_config_path,
            "state_paths": state_paths,
            "final_report_status": "completed" if final_report_result.get("report") or final_report_result.get("html") else final_report_result.get("status", "not_run"),
            "final_report_delivery_root": final_report_result.get("delivery_root", ""),
        })
        _write_json(job_path / "phase3_result.json", phase3_result)
        await websocket.send_json({"type": "phase_result", "phase": "phase3", "result": phase3_result, "timestamp": get_timestamp()})
        await send_activity_update(
            websocket,
            "phase3",
            "Evidence package complete",
            f"Figures and final reports are ready: {phase3_result.get('final_report_count', 0)} downloadable report formats.",
            status="completed",
            phase_progress=100,
        )
        await send_log(websocket, "system", f"Phase 3 completed. Report: {phase3_result.get('report_path', '')}")
        return phase3_result
    except Exception as exc:
        await send_log(websocket, "stderr", f"Phase 3 failed: {exc}")
        raise
    finally:
        os.chdir(old_cwd)
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _run_job_pipeline(job_id: str, websocket: WebSocket) -> None:
    job = _load_job(job_id)
    request_payload = _read_json(_job_dir(job_id) / "job_config.json", {})
    file_record = _load_file_record(request_payload.get("file_id", ""))
    request = JobCreateRequest(**request_payload)
    proxy = JobWebSocketProxy(websocket, job_id)

    runtime_root = _job_dir(job_id) / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    old_runtime_root = os.environ.get("METABOAGENT_RUNTIME_ROOT")
    os.environ["METABOAGENT_RUNTIME_ROOT"] = str(runtime_root)

    try:
        job.update({
            "status": "running",
            "current_phase": "initializing",
            "started_at": job.get("started_at") or get_timestamp(),
            "updated_at": get_timestamp(),
            "progress": {"overall": 1, "phase0": 0, "phase1": 0, "phase2": 0, "phase3": 0},
        })
        _save_job(job)
        await proxy.send_json({"type": "job_state", "status": "running", "current_phase": "initializing"})

        data_path = file_record["storage_path"]
        phases = set(request.analysis.phases or [])
        target_column = request.dataset.group_column
        label_metadata = _resolve_binary_label_metadata(
            data_path,
            target_column,
            request.dataset.positive_class,
        )
        runtime_config_path = _build_runtime_config(
            job_id,
            request,
            file_record,
            target_column,
            data_path,
            label_metadata=label_metadata,
        )
        job["runtime_config_path"] = runtime_config_path
        _save_job(job)
        await proxy.send_json({"type": "job_config", "runtime_config_path": runtime_config_path})

        if request.analysis.run_phase0_prior_search and "phase0" in phases:
            phase0_result = await run_phase0_real(
                proxy,
                request.analysis.disease_name,
                max_candidates=request.phase0.max_candidates,
                use_cache=request.phase0.use_cache,
            )
        else:
            phase0_result = {
                "disease_name": request.analysis.disease_name,
                "final_priors": [],
                "confirmed_biomarkers": [],
                "feature_definitions": {"target_pathways": [], "target_metabolites": []},
                "phase0_completed": False,
                "phase0_skipped": True,
            }
            await proxy.send_json({"type": "phase0_result", "result": phase0_result})

        phase1_result = {}
        if "phase1" in phases:
            phase1_result = await run_phase1_real(
                proxy,
                data_path,
                target_column,
                phase0_result,
                max_steps=request.phase1.max_steps,
                id_column=request.dataset.id_column,
                positive_class=label_metadata.get("positive_class"),
                negative_class=label_metadata.get("negative_class"),
                label_mapping=label_metadata.get("label_mapping", {}),
            )

        phase2_result = {}
        if "phase2" in phases:
            phase2_result = await run_phase2_real(
                proxy,
                phase0_result,
                phase1_result,
                target_column_override=target_column,
                beam_width=request.phase2.beam_width,
                k_folds=request.phase2.k_folds,
                epsilon=request.phase2.epsilon,
                patience=request.phase2.patience,
                positive_class=label_metadata.get("positive_class"),
            )

        phase3_result = {}
        if "phase3" in phases:
            phase3_result = await run_phase3_real(
                proxy,
                job_id,
                phase0_result,
                phase1_result,
                phase2_result,
                data_path=data_path,
                target_column=target_column,
                runtime_config_path=runtime_config_path,
                figure_style=request.phase3.figure_style,
                output_formats=request.phase3.output_formats,
            )

        result_summary = {
            "job_id": job_id,
            "status": "completed",
            "summary": {
                "disease_name": request.analysis.disease_name,
                "clinical_scenario": request.analysis.clinical_scenario,
                "input_file": file_record.get("filename"),
                "runtime_config_path": runtime_config_path,
            },
            "phase0": _json_safe(phase0_result),
            "phase1": _json_safe(phase1_result),
            "phase2": _json_safe(phase2_result),
            "phase3": _json_safe(phase3_result),
            "artifacts": _list_job_artifacts(job_id),
        }
        result_summary = _decorate_result_payload(job_id, result_summary)
        _write_json(_job_dir(job_id) / "result_summary.json", result_summary)

        job.update({
            "status": "completed",
            "current_phase": "completed",
            "finished_at": get_timestamp(),
            "updated_at": get_timestamp(),
            "progress": {"overall": 100, "phase0": 100, "phase1": 100, "phase2": 100, "phase3": 100 if "phase3" in phases else 0},
            "error": None,
        })
        _save_job(job)
        await proxy.send_json({
            "type": "job_completed",
            "results_url": f"/api/v1/jobs/{job_id}/results",
            "artifacts_url": f"/api/v1/jobs/{job_id}/artifacts",
        })
    except Exception as exc:
        job.update({
            "status": "failed",
            "error": str(exc),
            "finished_at": get_timestamp(),
            "updated_at": get_timestamp(),
        })
        _save_job(job)
        _append_job_log(job_id, "error", str(job.get("current_phase", "runtime")), str(exc))
        await proxy.send_json({"type": "error", "message": str(exc)})
        raise
    finally:
        if old_runtime_root is None:
            os.environ.pop("METABOAGENT_RUNTIME_ROOT", None)
        else:
            os.environ["METABOAGENT_RUNTIME_ROOT"] = old_runtime_root


@app.post("/api/v1/files/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 CSV/XLSX/XLS 文件")

    file_id = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    upload_dir = UPLOADS_DIR / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = _safe_name(file.filename or f"uploaded{suffix}")
    storage_path = upload_dir / safe_filename

    with storage_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    try:
        preview = _preview_dataframe(storage_path)
    except Exception as exc:
        storage_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File parsing failed: {exc}") from exc

    record = {
        "file_id": file_id,
        "filename": file.filename,
        "file_type": suffix.lstrip("."),
        "storage_path": str(storage_path),
        "created_at": get_timestamp(),
        "preview": preview,
    }
    _write_json(_file_record_path(file_id), record)
    return record


@app.get("/api/v1/config/demo-datasets")
async def get_demo_datasets():
    """Return reviewer-facing datasets that can be loaded without a local file."""
    datasets = []
    for demo_id, spec in DEMO_DATASETS.items():
        path = Path(spec["path"])
        item = {
            "id": demo_id,
            "name": spec["name"],
            "description": spec["description"],
            "filename": spec["filename"],
            "defaults": spec["defaults"],
            "available": path.is_file(),
        }
        if path.is_file():
            try:
                preview = _preview_dataframe(path, limit=3)
                item["preview"] = {
                    "n_rows": preview["n_rows"],
                    "n_columns": preview["n_columns"],
                }
            except Exception as exc:
                item["available"] = False
                item["error"] = str(exc)
        datasets.append(item)
    return {"datasets": datasets}


@app.post("/api/v1/demo-datasets/{demo_id}/load")
async def load_demo_dataset(demo_id: str):
    """Materialize a stable demo dataset as a normal file record for a run."""
    spec = DEMO_DATASETS.get(demo_id)
    if not spec:
        raise HTTPException(status_code=404, detail="Demo dataset not found")
    source_path = Path(spec["path"])
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Demo dataset is not available on this deployment")

    file_id = f"demo_{_safe_name(demo_id)}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    upload_dir = UPLOADS_DIR / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / _safe_name(spec["filename"])
    shutil.copy2(source_path, destination)
    try:
        preview = _preview_dataframe(destination)
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Demo dataset parsing failed: {exc}") from exc

    record = {
        "file_id": file_id,
        "filename": spec["filename"],
        "file_type": destination.suffix.lower().lstrip("."),
        "storage_path": str(destination),
        "created_at": get_timestamp(),
        "preview": preview,
        "demo_id": demo_id,
        "is_demo": True,
        "demo_defaults": spec["defaults"],
    }
    _write_json(_file_record_path(file_id), record)
    return record


@app.get("/api/v1/files/{file_id}/sheets")
async def get_file_sheets(file_id: str):
    record = _load_file_record(file_id)
    path = Path(record["storage_path"])
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        return {"file_id": file_id, "sheets": []}
    import pandas as pd
    return {"file_id": file_id, "sheets": list(pd.ExcelFile(path).sheet_names)}


@app.post("/api/v1/files/{file_id}/parse")
async def parse_file(file_id: str, payload: Dict[str, Any]):
    record = _load_file_record(file_id)
    preview = _preview_dataframe(Path(record["storage_path"]), sheet_name=payload.get("sheet_name"))
    record["preview"] = preview
    record["sheet_name"] = payload.get("sheet_name")
    _write_json(_file_record_path(file_id), record)
    return {**record, "preview": preview}


@app.get("/api/v1/config/scenarios")
async def get_scenarios():
    cfg = get_config()
    scenarios = []
    for key in cfg.config.get("evaluation", {}).get("scenarios", {}).keys():
        scenarios.append(cfg.get_scenario_definition(key))
    return {"active_scenario": cfg.get_active_scenario(), "scenarios": scenarios}


@app.get("/api/v1/config/figure-styles")
async def get_figure_styles():
    return {
        "default_style": "nature",
        "styles": [
            {"key": "nature", "name": "Nature Style", "description": "适合 Nature / Nature Medicine 风格图表"},
            {"key": "science_advances", "name": "Science Advances Style", "description": "适合 Science Advances 风格图表"},
            {"key": "cell_metabolism", "name": "Cell Metabolism Style", "description": "适合 Cell Metabolism 风格图表"},
            {"key": "clinical_report", "name": "Clinical Report Style", "description": "适合临床报告和院内展示"},
        ],
    }


@app.get("/api/v1/config/phases")
async def get_phases():
    return {
        "phases": [
            {"key": "phase0", "name": "Phase 0 Prior Biomarker Search", "optional": True},
            {"key": "phase1", "name": "Phase 1 Data Preprocessing and Baseline Modeling", "optional": False},
            {"key": "phase2", "name": "Phase 2 Multi-objective Panel Optimization", "optional": False},
            {"key": "phase3", "name": "Phase 3 Visualization and Report Generation", "optional": True},
        ]
    }


@app.post("/api/v1/jobs")
async def create_job(request: JobCreateRequest):
    file_record = _load_file_record(request.file_id)
    available_columns = {
        str(column.get("name"))
        for column in (file_record.get("preview", {}).get("columns", []) or [])
        if isinstance(column, dict) and column.get("name") is not None
    }
    if not request.dataset.id_column or request.dataset.id_column not in available_columns:
        raise HTTPException(status_code=400, detail="A valid sample ID column is required before starting the analysis")
    if not request.dataset.group_column or request.dataset.group_column not in available_columns:
        raise HTTPException(status_code=400, detail="A valid group / outcome column is required before starting the analysis")
    normalized_phases = _normalize_requested_phases(request.analysis.phases or [])
    request.analysis.phases = normalized_phases
    request.analysis.disease_name = request.analysis.disease_name.strip()
    if "phase0" in normalized_phases and request.analysis.run_phase0_prior_search and not request.analysis.disease_name:
        raise HTTPException(status_code=400, detail="启用 Phase 0 时必须提供疾病名称或临床问题")
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    job_path = _job_dir(job_id)
    job_path.mkdir(parents=True, exist_ok=True)
    _write_json(job_path / "job_config.json", request.model_dump())
    job = {
        "job_id": job_id,
        "status": "queued",
        "current_phase": "queued",
        "current_step": None,
        "progress": {"overall": 0, "phase0": 0, "phase1": 0, "phase2": 0, "phase3": 0},
        "created_at": get_timestamp(),
        "started_at": None,
        "finished_at": None,
        "updated_at": get_timestamp(),
        "error": None,
        "stream_url": f"/api/v1/jobs/{job_id}/stream",
        "status_url": f"/api/v1/jobs/{job_id}",
        "results_url": f"/api/v1/jobs/{job_id}/results",
        "runtime_estimate": _build_runtime_estimate(normalized_phases),
    }
    _save_job(job)
    _schedule_job(job_id)
    return job


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    return _load_job(job_id)


@app.get("/api/v1/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    _load_job(job_id)
    logs_path = _job_dir(job_id) / "logs.jsonl"
    logs = []
    if logs_path.exists():
        for line in logs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                logs.append(json.loads(line))
    return {"job_id": job_id, "logs": logs}


@app.get("/api/v1/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    _load_job(job_id)
    result = _read_json(_job_dir(job_id) / "result_summary.json", None)
    if result is None:
        raise HTTPException(status_code=404, detail="结果尚未生成")
    return _decorate_result_payload(job_id, result)


@app.get("/api/v1/jobs/{job_id}/artifacts")
async def get_job_artifacts(job_id: str):
    _load_job(job_id)
    return {"job_id": job_id, "artifacts": _list_job_artifacts(job_id)}


@app.get("/api/v1/jobs/{job_id}/download/{artifact_path:path}")
async def download_job_artifact(job_id: str, artifact_path: str):
    _load_job(job_id)
    if "holdout_calibration" in artifact_path.lower():
        raise HTTPException(status_code=404, detail="This figure has been retired from the final output")
    root = _job_dir(job_id).resolve()
    target = (root / artifact_path).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(target), filename=target.name)


@app.get("/api/v1/jobs/{job_id}/figures.zip")
async def download_figures_bundle(job_id: str):
    """Download the generated figure files as one reviewer-friendly archive."""
    _load_job(job_id)
    root = _job_dir(job_id).resolve()
    candidates = [
        root / "runtime" / "output" / "final_delivery" / "current" / "figures",
        root / "runtime" / "output" / "figures",
    ]
    figures_root = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if figures_root is None:
        raise HTTPException(status_code=404, detail="No figure files are available for this run")

    buffer = io.BytesIO()
    excluded_tokens = {"phase2_objective_shift_clinical_validation_composite", "phase2_radar_clinical_validation_composite"}
    added_files = 0
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(path for path in figures_root.rglob("*") if path.is_file()):
            if any(token in file_path.name for token in excluded_tokens) or "holdout_calibration" in file_path.name.lower():
                continue
            archive.write(file_path, file_path.relative_to(figures_root).as_posix())
            added_files += 1
    if added_files == 0:
        raise HTTPException(status_code=404, detail="No figure files are available for this run")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{_safe_name(job_id)}_figures.zip"'},
    )


@app.websocket("/api/v1/jobs/{job_id}/stream")
async def job_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = _load_job(job_id)
    await websocket.send_json({"type": "job_state", "job_id": job_id, "status": job.get("status"), "current_phase": job.get("current_phase"), "timestamp": get_timestamp()})
    if job.get("status") == "queued":
        # The worker is scheduled by POST /jobs.  A WebSocket connection is an
        # optional observer and must never be the trigger for computation.
        _schedule_job(job_id)
        await websocket.send_json({"type": "job_snapshot", "job_id": job_id, "job": _load_job(job_id), "timestamp": get_timestamp()})
    else:
        await websocket.send_json({"type": "job_snapshot", "job_id": job_id, "job": job, "timestamp": get_timestamp()})


# ============================================================================
# WebSocket 路由（兼容旧前端）
# ============================================================================

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 分析接口（真实 Agent 集成）"""
    
    await websocket.accept()
    print(f"\n[WebSocket] 客户端已连接: {websocket.client}")
    
    try:
        # 等待前端发送启动指令
        data = await websocket.receive_json()
        print(f"[WebSocket] 收到消息: {json.dumps(data, ensure_ascii=False)}")
        
        if data.get("action") != "start_analysis":
            await websocket.send_json({
                "type": "error",
                "message": "无效的指令，期望 action='start_analysis'",
            })
            return
        
        # 获取参数（默认使用配置中的测试设置）
        try:
            from src.utils.config_manager import get_config
            cfg = get_config()
            default_disease_name = cfg.get_test_disease_name()
            default_data_path = cfg.get_test_data_path()
            default_target_column = cfg.get_target_column()
        except Exception:
            default_disease_name = ""
            default_data_path = "../tests/data/test_lung_cancer.xlsx"
            default_target_column = "target"

        disease_name = data.get("disease_name", default_disease_name)
        data_path = data.get("data_path", default_data_path)
        target_column = data.get("target_column", default_target_column)
        max_candidates = data.get("max_candidates", 50)
        use_cache = data.get("use_cache", True)
        max_steps = data.get("max_steps", 100)
        
        await send_log(websocket, "system", f"Analysis request received: {disease_name}")
        
        # 执行 Phase 0
        phase0_result = await run_phase0_real(
            websocket,
            disease_name,
            max_candidates=max_candidates,
            use_cache=use_cache
        )
        
        # 执行 Phase 1
        phase1_result = await run_phase1_real(
            websocket,
            data_path,
            target_column,
            phase0_result,
            max_steps=max_steps
        )

        # 执行 Phase 2
        phase2_result = await run_phase2_real(
            websocket,
            phase0_result,
            phase1_result,
        )

        orchestrator = MetaboMemoryOrchestrator()
        active_scenario = get_config().get_active_scenario()
        run_memory_case = orchestrator.build_run_memory_case(
            disease_name=phase0_result.get("disease_name", ""),
            clinical_scenario=active_scenario,
            dataset_fingerprint=phase1_result.get("dataset_fingerprint", {}) or {},
            phase0_result=phase0_result,
            phase1_result=phase1_result,
            phase2_result=phase2_result,
            tags=["run", "episodic"],
        )
        run_memory_case_id = orchestrator.write_memory_case(run_memory_case)
        if run_memory_case_id:
            await send_log(websocket, "stdout", f"✓ Run-level episodic memory saved: {run_memory_case_id}")
        
        # 分析完成
        await send_state(websocket, "completed")
        await send_log(websocket, "system", "Analysis workflow completed.")
        
        # 保持连接
        while True:
            try:
                message = await websocket.receive_json()
                if message.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        print(f"[WebSocket] 客户端断开连接")
    except Exception as e:
        print(f"[WebSocket] 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"服务器错误: {str(e)}",
            })
        except:
            pass
    finally:
        print(f"[WebSocket] 连接已关闭\n")


# ============================================================================
# 健康检查接口
# ============================================================================

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "MetaboAgent Backend (Real Agent)",
        "status": "running",
        "version": "2.0.0",
        "timestamp": get_timestamp(),
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": get_timestamp(),
    }


# ============================================================================
# 应用启动事件
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """应用启动"""
    print("\n" + "="*60)
    print("MetaboAgent Backend 服务启动（真实 Agent 集成）")
    print("="*60)
    print(f"WebSocket 接口: ws://localhost:8000/ws/analyze")
    print(f"健康检查接口: http://localhost:8000/health")
    print("="*60 + "\n")

    # Recover jobs created just before a service restart.  Only queued jobs
    # are safe to resume automatically; running jobs retain their recorded
    # state and are not duplicated here.
    for job_path in JOBS_DIR.glob("job_*"):
        job = _read_json(job_path / "job_status.json", {})
        if job.get("status") == "queued" and job.get("job_id"):
            _schedule_job(str(job["job_id"]))


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭"""
    print("\n" + "="*60)
    print("MetaboAgent Backend 服务关闭")
    print("="*60 + "\n")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main_real:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
