"""
Phase 1 Agent Graph Builder

This module builds the LangGraph workflow for the Phase 1 Code Interpreter Agent.

Graph Structure:
    START -> generate_code -> execute_code -> [should_continue] -> generate_code | END

Workflow:
1. generate_code: LLM generates Python code based on SOP step and metadata
2. execute_code: Execute code locally, parse output, update state
3. should_continue: Check if workflow should continue or end
4. Loop back to generate_code for next step, or END if completed

Privacy Architecture:
- LLM only sees metadata (data_summary), never full CSV
- All data processing happens locally
- State updates via Magic Output Protocol
"""

import copy
import json
from typing import Optional
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from .state import Phase1State, advance_to_next_step
from .nodes import (
    generate_code,
    execute_code,
    execute_programmatic_stage11,
    execute_programmatic_step50,
    execute_programmatic_step521,
    execute_programmatic_step522,
    execute_programmatic_stage154,
    execute_programmatic_stage53,
    should_continue,
    check_execution,
    reflect_and_fix,
    check_generate_code_result
)


def build_phase1_graph(llm: Optional[BaseChatModel] = None) -> StateGraph:
    """
    Build the Phase 1 Code Interpreter Agent graph with error recovery.
    
    Graph Structure:
        START -> generate_code -> execute_code -> check_execution
                                                      ├─ success -> advance_step -> should_continue
                                                      ├─ retry -> reflect_and_fix -> generate_code
                                                      └─ failed -> END
    
    Args:
        llm: Language model for code generation (optional, can be set later)
    
    Returns:
        Compiled StateGraph ready for execution
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> graph = build_phase1_graph(llm)
        >>> result = graph.invoke(initial_state)
    """
    # Create the graph
    workflow = StateGraph(Phase1State)
    
    # Add nodes
    workflow.add_node("programmatic_step_bootstrap", _execute_programmatic_step_bootstrap)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("execute_code", execute_code)
    workflow.add_node("reflect_and_fix", reflect_and_fix)  # New: Error reflection
    workflow.add_node("advance_step", advance_to_next_step)
    
    # Add edges
    workflow.set_entry_point("programmatic_step_bootstrap")

    workflow.add_conditional_edges(
        "programmatic_step_bootstrap",
        _route_after_programmatic_step_bootstrap,
        {
            "advance": "advance_step",
            "generate": "generate_code",
            "end": END,
        }
    )
    
    # generate_code -> conditional routing
    workflow.add_conditional_edges(
        "generate_code",
        check_generate_code_result,
        {
            "execute": "execute_code",  # Code generated -> execute
            "end": END                   # Fatal error -> end
        }
    )
    
    # execute_code -> conditional routing based on execution result
    workflow.add_conditional_edges(
        "execute_code",
        check_execution,  # New: Check if execution succeeded or needs retry
        {
            "success": "advance_step",      # Success -> advance to next step
            "retry": "reflect_and_fix",     # Failed but can retry -> reflect
            "failed": END                    # Failed with no retries -> end
        }
    )
    
    # reflect_and_fix -> generate_code (retry loop)
    workflow.add_edge("reflect_and_fix", "generate_code")
    
    # advance_step -> conditional routing
    workflow.add_conditional_edges(
        "advance_step",
        should_continue,
        {
            "generate_code": "programmatic_step_bootstrap",
            "end": END
        }
    )
    
    # Compile the graph
    return workflow.compile()


def _execute_programmatic_step_bootstrap(state: Phase1State) -> Phase1State:
    if _should_use_programmatic_stage11(state):
        return execute_programmatic_stage11(state)
    if _should_use_programmatic_step50(state):
        return execute_programmatic_step50(state)
    if _should_use_programmatic_step521(state):
        return execute_programmatic_step521(state)
    if _should_use_programmatic_step522(state):
        return execute_programmatic_step522(state)
    if _should_use_programmatic_stage154(state):
        return execute_programmatic_stage154(state)
    if _should_use_programmatic_stage53(state):
        return execute_programmatic_stage53(state)
    return state


def _should_use_programmatic_stage11(state: Phase1State) -> bool:
    step = None
    try:
        from .state import get_current_step
        step = get_current_step(state)
    except Exception:
        step = None
    if not step:
        return False
    return str(step.get("step_id", step.get("id", ""))) == "1.1"


def _should_use_programmatic_stage53(state: Phase1State) -> bool:
    step = None
    try:
        from .state import get_current_step
        step = get_current_step(state)
    except Exception:
        step = None
    if not step:
        return False
    return str(step.get("step_id", step.get("id", ""))) == "5.3"


def _should_use_programmatic_stage154(state: Phase1State) -> bool:
    step = None
    try:
        from .state import get_current_step
        step = get_current_step(state)
    except Exception:
        step = None
    return bool(step and str(step.get("step_id", step.get("id", ""))) == "1.5.4")


def _should_use_programmatic_step522(state: Phase1State) -> bool:
    step = None
    try:
        from .state import get_current_step
        step = get_current_step(state)
    except Exception:
        step = None
    if not step:
        return False
    return str(step.get("step_id", step.get("id", ""))) == "5.2.2"


def _should_use_programmatic_step50(state: Phase1State) -> bool:
    step = None
    try:
        from .state import get_current_step
        step = get_current_step(state)
    except Exception:
        step = None
    if not step:
        return False
    return str(step.get("step_id", step.get("id", ""))) == "5.0"


def _should_use_programmatic_step521(state: Phase1State) -> bool:
    step = None
    try:
        from .state import get_current_step
        step = get_current_step(state)
    except Exception:
        step = None
    if not step:
        return False
    return str(step.get("step_id", step.get("id", ""))) == "5.2.1"


def _route_after_programmatic_step_bootstrap(state: Phase1State) -> str:
    if not (
        _should_use_programmatic_stage11(state)
        or _should_use_programmatic_step50(state)
        or _should_use_programmatic_step521(state)
        or _should_use_programmatic_step522(state)
        or _should_use_programmatic_stage154(state)
        or _should_use_programmatic_stage53(state)
    ):
        return "generate"
    if state.get("error"):
        return "end"
    return "advance"


def create_phase1_agent(
    llm: BaseChatModel,
    sop_config: dict,
    initial_data_path: str,
    target_column: Optional[str] = None,
    max_retries: int = 3,
    initial_context_variables: Optional[dict] = None,
    dataset_fingerprint: Optional[dict] = None,
    memory_context: Optional[dict] = None,
    phase1_memory_pack: Optional[dict] = None,
):
    """
    Create a complete Phase 1 agent with initialized state and error recovery.
    
    This is a convenience function that:
    1. Generates initial data summary
    2. Creates initial state with retry configuration
    3. Builds the graph with error recovery
    4. Returns ready-to-run agent
    
    Args:
        llm: Language model for code generation
        sop_config: SOP configuration dictionary
        initial_data_path: Path to initial dataset
        target_column: Name of target/group column (optional)
        max_retries: Maximum retry attempts per step (default: 3)
        initial_context_variables: Optional initial context injected from
            upstream stages such as Phase 0
        dataset_fingerprint: Optional dataset fingerprint for memory retrieval
        memory_context: Optional run-level memory context
        phase1_memory_pack: Optional advisory Phase 1 memory hints
    
    Returns:
        Tuple of (compiled_graph, initial_state)
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> sop = load_sop("data_cleaning.json")
        >>> graph, state = create_phase1_agent(
        ...     llm=llm,
        ...     sop_config=sop,
        ...     initial_data_path="data.csv",
        ...     target_column="Group",
        ...     max_retries=3
        ... )
        >>> result = graph.invoke(state)
    """
    from .nodes import get_data_summary
    from .state import create_initial_state
    
    # Generate initial data summary
    data_summary = get_data_summary(initial_data_path, target_column)
    
    # Create initial state with retry configuration
    initial_state = create_initial_state(
        sop_config=sop_config,
        initial_data_path=initial_data_path,
        initial_data_summary=data_summary,
        target_column=target_column,
        max_retries=max_retries,
        llm=llm,
        initial_context_variables=initial_context_variables,
        dataset_fingerprint=dataset_fingerprint,
        memory_context=memory_context,
        phase1_memory_pack=phase1_memory_pack,
    )
    
    # Build graph
    graph = build_phase1_graph(llm)
    
    return graph, initial_state



def load_sop(sop_path: str) -> dict:
    """
    Load SOP configuration from a JSON file.
    
    Args:
        sop_path: Path to SOP JSON file
    
    Returns:
        SOP configuration dictionary
    
    Example:
        >>> sop = load_sop("sops/data_cleaning.json")
    """
    resolved_path = Path(sop_path)
    with open(resolved_path, 'r', encoding='utf-8') as f:
        sop_config = json.load(f)

    if _is_targeted_preprocessing_sop(sop_config):
        return _build_runtime_targeted_sop(sop_config, resolved_path)

    return sop_config


def save_sop(sop_config: dict, sop_path: str):
    """
    Save SOP configuration to a JSON file.
    
    Args:
        sop_config: SOP configuration dictionary
        sop_path: Path to save SOP JSON file
    
    Example:
        >>> save_sop(EXAMPLE_SOP, "sops/data_cleaning.json")
    """
    with open(sop_path, 'w') as f:
        json.dump(sop_config, f, indent=2)


def _is_targeted_preprocessing_sop(sop_config: dict) -> bool:
    """Detect the new targeted preprocessing SOP scaffold."""
    return str(sop_config.get("sop_id", "")).strip() == "targeted_preprocessing_v2"


def _build_runtime_targeted_sop(targeted_sop: dict, targeted_sop_path: Path) -> dict:
    """
    Adapt the new targeted preprocessing SOP scaffold into the legacy step-based
    execution format expected by the current Phase 1 agent.

    Strategy:
    - Reuse the mature legacy Stage 0 and downstream stages from the existing SOP.
    - Inject a new Stage 0.5 that runs the P0 preprocessing tool chain.
    - Patch legacy Stage 1.1 so it consumes the exported preprocessing artifacts.
    """
    project_root = targeted_sop_path.resolve().parents[1]
    legacy_sop_path = project_root / "data" / "SOP-代谢物分析流程_v2_code_interpreter.json"
    with open(legacy_sop_path, "r", encoding="utf-8") as f:
        legacy_sop = json.load(f)

    runtime_sop = copy.deepcopy(legacy_sop)
    runtime_sop["sop_id"] = "PHASE1_TARGETED_RUNTIME_BRIDGE_V1"
    runtime_sop["sop_name"] = "Phase 1 Targeted Preprocessing Runtime SOP"
    runtime_sop["description"] = (
        "Hybrid runtime SOP: legacy executable stages plus targeted preprocessing "
        "P0 context/zero/missingness/QA bridge."
    )
    runtime_sop["runtime_source"] = {
        "targeted_sop_path": str(targeted_sop_path),
        "legacy_sop_path": str(legacy_sop_path),
    }

    stages = runtime_sop.get("stages", [])
    if not isinstance(stages, list) or len(stages) < 2:
        return runtime_sop

    injected_stage = _build_targeted_runtime_stage_from_scaffold()
    patched_stages = []
    inserted = False
    for stage in stages:
        stage_copy = copy.deepcopy(stage)
        if str(stage_copy.get("id")) == "1":
            stage_copy = _patch_legacy_preprocessing_stage(stage_copy)
        if str(stage_copy.get("id")) == "0" and not inserted:
            patched_stages.append(stage_copy)
            patched_stages.append(injected_stage)
            inserted = True
            continue
        patched_stages.append(stage_copy)

    runtime_sop["stages"] = patched_stages
    runtime_sop["targeted_scaffold_summary"] = {
        "file_name": targeted_sop.get("file_name"),
        "implementation_companions": targeted_sop.get("implementation_companions", {}),
        "phase4_contract": targeted_sop.get("phase4_contract", {}),
    }
    return runtime_sop


def _build_targeted_runtime_stage_from_scaffold() -> dict:
    """Create a legacy-compatible runtime stage for the new P0 preprocessing chain."""
    return {
        "id": 0.5,
        "stage_id": 0.5,
        "name": "Targeted Preprocessing Context Resolution",
        "condition": "p0_bootstrap_completed == false",
        "description": (
            "Bridge stage that resolves preprocessing context, zero semantics, "
            "missingness mechanism, and exports preprocessing QA artifacts before "
            "legacy Stage 1 cleaning begins."
        ),
        "steps": [
            {
                "step_id": "0.5.1",
                "action": "coding_task",
                "instruction": "Resolve Targeted Preprocessing Context",
                "details": (
                    "Import `data_context_resolver_tool` from `src.tools.analysis.data_analysis_tools`, "
                    "`get_config` from `src.utils.config_manager`, and `json`, `os`, `pathlib`.\n"
                    "1. Resolve `phase1_intermediate_latest` via `cfg = get_config(); "
                    "phase1_intermediate_latest = cfg.get_phase1_path('intermediate_latest_dir') or "
                    "'output/phase1/intermediate/latest'`.\n"
                    "2. Create `preprocessing_context.json` under that directory.\n"
                    "3. Call `data_context_resolver_tool(dataset_path=current_data_path, study_metadata_path=context_variables.get('study_metadata_path',''), "
                    "column_roles_path=context_variables.get('column_roles_path',''), source_hint=context_variables.get('dataset_source',''), "
                    "user_declared_data_level=context_variables.get('user_declared_data_level',''))`.\n"
                    "4. Parse returned JSON, save it to disk, and update context variables with: "
                    "`preprocessing_context_report_path`, `platform_source`, `data_level`, `candidate_branch`, "
                    "`has_pooled_qc`, `has_blank_samples`, `has_run_order`, `has_batch_metadata`, `protected_columns` "
                    "(merge with any existing protected columns).\n"
                    "5. Use Magic Output Protocol to persist these fields."
                ),
                "lib_hint": "src.tools.analysis.data_analysis_tools, src.utils.config_manager, json, os, pathlib",
                "tools_required": ["data_context_resolver_tool"],
                "context_files": [
                    "current_data_path",
                    "study_metadata_path (optional)",
                    "column_roles_path (optional)",
                ],
                "outputs": [
                    "preprocessing_context_report_path",
                    "platform_source",
                    "data_level",
                    "candidate_branch",
                    "protected_columns",
                ],
                "reasoning": "在任何缺失值或归一化决策之前，先确定数据上下文和分支。"
            },
            {
                "step_id": "0.5.2",
                "action": "coding_task",
                "instruction": "Assess Zero Semantics for Public Matrix Data",
                "details": (
                    "Import `zero_semantics_analyzer_tool` plus `json`, `os`, and `get_config`.\n"
                    "1. Resolve output path `zero_pattern_report.json` under "
                    "`cfg.get_phase1_path('intermediate_latest_dir') or 'output/phase1/intermediate/latest'`.\n"
                    "2. Call `zero_semantics_analyzer_tool(dataset_path=current_data_path, "
                    "context_report_path=preprocessing_context_report_path, "
                    "study_metadata_path=context_variables.get('study_metadata_path',''), "
                    "protected_columns=context_variables.get('protected_columns', []))`.\n"
                    "3. Save the returned JSON to disk.\n"
                    "4. Update context variables with: `zero_pattern_report_path`, `zero_semantics_status`, "
                    "`zero_handling_strategy`, `zero_evidence_flags`.\n"
                    "5. Use Magic Output Protocol."
                ),
                "lib_hint": "src.tools.analysis.data_analysis_tools, src.utils.config_manager, json, os",
                "tools_required": ["zero_semantics_analyzer_tool"],
                "context_files": [
                    "current_data_path",
                    "preprocessing_context_report_path",
                    "protected_columns",
                ],
                "outputs": [
                    "zero_pattern_report_path",
                    "zero_semantics_status",
                    "zero_handling_strategy",
                ],
                "reasoning": "对 targeted public matrix，0 值语义会直接影响后续 missingness 与插补策略。"
            },
            {
                "step_id": "0.5.3",
                "action": "coding_task",
                "instruction": "Assess Missingness After Zero Handling Decision",
                "details": (
                    "Import `missingness_assessment_tool` plus `json`, `os`, and `get_config`.\n"
                    "1. Resolve `missingness_report.json` under "
                    "`cfg.get_phase1_path('intermediate_latest_dir') or 'output/phase1/intermediate/latest'`.\n"
                    "2. Call `missingness_assessment_tool(dataset_path=current_data_path, "
                    "context_report_path=preprocessing_context_report_path, "
                    "zero_pattern_report_path=zero_pattern_report_path, "
                    "zero_handling_mode=context_variables.get('zero_handling_strategy','auto'), "
                    "protected_columns=context_variables.get('protected_columns', []))`.\n"
                    "3. Save the report.\n"
                    "4. Update context variables with: `missingness_report_path`, `zero_handling_mode_used`, "
                    "`suspected_missingness_mechanism`, `recommended_imputation_family`, "
                    "`recommended_missingness_thresholds`, `left_censoring_signals`.\n"
                    "5. Use Magic Output Protocol."
                ),
                "lib_hint": "src.tools.analysis.data_analysis_tools, src.utils.config_manager, json, os",
                "tools_required": ["missingness_assessment_tool"],
                "context_files": [
                    "current_data_path",
                    "preprocessing_context_report_path",
                    "zero_pattern_report_path",
                ],
                "outputs": [
                    "missingness_report_path",
                    "suspected_missingness_mechanism",
                    "recommended_imputation_family",
                ],
                "reasoning": "将 0 值语义决策正式传递到缺失值机制判断中。"
            },
            {
                "step_id": "0.5.4",
                "action": "coding_task",
                "instruction": "Export Partial Preprocessing QA Facts",
                "details": (
                    "Import `preprocessing_qa_reporter_tool` plus `json`.\n"
                    "1. Call `preprocessing_qa_reporter_tool(context_report_path=preprocessing_context_report_path, "
                    "zero_pattern_report_path=zero_pattern_report_path, "
                    "missingness_report_path=missingness_report_path)`.\n"
                    "2. Parse the returned JSON and update context variables with: "
                    "`preprocessing_report_path`, `preprocessing_summary_path`, `preprocessing_decision_pack_path`, "
                    "`phase4_reportable_facts`, `preprocessing_branch_used`.\n"
                    "3. Use Magic Output Protocol.\n"
                    "4. Do not claim that normalization, outlier handling, or batch correction has already been executed; "
                    "this is a partial QA export before legacy Stage 1 preprocessing."
                ),
                "lib_hint": "src.tools.analysis.data_analysis_tools, json",
                "tools_required": ["preprocessing_qa_reporter_tool"],
                "context_files": [
                    "preprocessing_context_report_path",
                    "zero_pattern_report_path",
                    "missingness_report_path",
                ],
                "outputs": [
                    "preprocessing_report_path",
                    "preprocessing_summary_path",
                    "preprocessing_decision_pack_path",
                    "phase4_reportable_facts",
                ],
                "reasoning": "在正式缺失值插补前先建立一份可追溯事实包，供后续步骤和最终报告引用。"
            },
        ],
    }


def _patch_legacy_preprocessing_stage(stage: dict) -> dict:
    """Patch legacy Stage 1 to consume the new preprocessing context artifacts."""
    steps = stage.get("steps", [])
    if not isinstance(steps, list):
        return stage

    for step in steps:
        if str(step.get("step_id")) != "1.1":
            continue

        prefix = (
            "CRITICAL TARGETED PREPROCESSING RULES:\n"
            "1. Before calculating missingness, check whether `preprocessing_context_report_path`, "
            "`zero_pattern_report_path`, and `missingness_report_path` already exist in context_variables.\n"
            "2. If `zero_handling_strategy` or `zero_handling_mode_used` indicates `convert_zero_to_nan`, "
            "you MUST first convert zeros in feature columns to NaN before computing missing-rate thresholds.\n"
            "3. If `recommended_imputation_family` is available, follow it as the default strategy "
            "(MNAR -> qrilc, MAR/MCAR -> knn, fallback -> median).\n"
            "4. Do not overwrite protected columns or target columns during zero-to-NaN conversion.\n"
            "5. After finishing imputation, write a lightweight `imputation_report.json` under "
            "`output/phase1/intermediate/latest` and rerun `preprocessing_qa_reporter_tool` if available "
            "to refresh the partial preprocessing facts.\n"
            "6. When writing ANY JSON report, you MUST convert numpy/pandas scalar types "
            "(for example `np.int64`, `np.float64`, `np.bool_`) into native Python types first, "
            "or call `json.dump(..., default=lambda o: o.item() if hasattr(o, 'item') else str(o))`.\n"
            "7. A step is NOT successful if the data file is written but the JSON report crashes at the end. "
            "Treat report serialization as part of the required deliverable.\n\n"
        )
        step["details"] = prefix + str(step.get("details", ""))
        existing_tools = list(step.get("tools_required", []) or [])
        for tool_name in (
            "data_context_resolver_tool",
            "zero_semantics_analyzer_tool",
            "missingness_assessment_tool",
            "preprocessing_qa_reporter_tool",
            "impute_missing_tool",
        ):
            if tool_name not in existing_tools:
                existing_tools.append(tool_name)
        step["tools_required"] = existing_tools
        break

    return stage
