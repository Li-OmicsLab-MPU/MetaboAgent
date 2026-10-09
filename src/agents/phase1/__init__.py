"""
Phase 1: Code Interpreter Agent

This module implements the "Brain" of the Phase 1 Data Scientist Agent.
The agent operates on a Code Interpreter pattern:
1. Reads an SOP (Standard Operating Procedure)
2. Generates Python code to execute steps using local domain tools
3. Runs code in a privacy-preserving local environment

Key Features:
- Privacy-Preserving: LLM never sees full CSV content, only metadata
- Metadata Synchronization: State updated after each execution
- Magic Output Protocol: Generated code updates state via JSON stdout

Example Usage:
    >>> from langchain_openai import ChatOpenAI
    >>> from src.agents.phase1 import create_phase1_agent
    >>> 
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> sop = {"name": "Data Cleaning", "stages": [...]}
    >>> graph, state = create_phase1_agent(
    ...     llm=llm,
    ...     sop_config=sop,
    ...     initial_data_path="data.csv",
    ...     target_column="Group"
    ... )
    >>> result = graph.invoke(state)
"""

from .state import (
    Phase1State,
    DataSummary,
    ExecutionRecord,
    create_initial_state,
    get_current_step,
    advance_to_next_step
)
from .nodes import (
    generate_code,
    execute_code,
    get_data_summary,
    should_continue,
    check_execution,
    reflect_and_fix
)
from .graph import (
    build_phase1_graph,
    create_phase1_agent,
    load_sop,
    save_sop
)
from .executor import (
    LocalPythonExecutor,
    ExecutionSandbox
)

__all__ = [
    # State
    "Phase1State",
    "DataSummary",
    "ExecutionRecord",
    "create_initial_state",
    "get_current_step",
    "advance_to_next_step",
    
    # Nodes
    "generate_code",
    "execute_code",
    "get_data_summary",
    "should_continue",
    "check_execution",
    "reflect_and_fix",
    
    # Graph
    "build_phase1_graph",
    "create_phase1_agent",
    "load_sop",
    "save_sop",
    
    # Executor
    "LocalPythonExecutor",
    "ExecutionSandbox",
]
