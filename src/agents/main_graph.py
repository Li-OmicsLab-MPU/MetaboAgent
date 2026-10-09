"""
Main Graph: MetaboAgent Hierarchical Architecture

This module implements the parent graph that orchestrates the entire MetaboAgent pipeline,
integrating Phase 0 (Researcher Agent) and future phases.

Architecture:
    START -> phase0_subgraph -> phase1_placeholder -> ... -> END

Author: MetaboAgent Team
Date: 2025-01-06
"""

import logging
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from src.agents.phase0.graph import create_phase0_workflow
from src.agents.phase0.state import Phase0State, create_initial_phase0_state

logger = logging.getLogger(__name__)


# ============================================================================
# Global Agent State
# ============================================================================
class AgentState(TypedDict):
    """
    Global state for the entire MetaboAgent pipeline.
    
    This state tracks the complete workflow from disease input through
    all analysis phases.
    
    Fields:
        disease_name: Input disease name
        prior_biomarkers: List of confirmed biomarkers from Phase 0
        phase0_state: Complete Phase 0 state (for debugging/inspection)
        current_phase: Current phase name
        results: Final results from all phases
        error: Error message if workflow fails
    """
    disease_name: str
    prior_biomarkers: List[str]
    phase0_state: Optional[Dict[str, Any]]
    current_phase: str
    results: Optional[Dict[str, Any]]
    error: Optional[str]


# ============================================================================
# Phase 0 Integration Node
# ============================================================================
def run_phase0_subgraph(state: AgentState) -> AgentState:
    """
    Execute the Phase 0 subgraph and integrate results into global state.
    
    This node:
    1. Extracts disease_name from global state
    2. Initializes Phase0State
    3. Runs the phase0_workflow
    4. Maps final_priors to prior_biomarkers in global state
    
    Args:
        state: Current AgentState
    
    Returns:
        Updated AgentState with prior_biomarkers populated
    """
    disease_name = state.get("disease_name", "")
    logger.info(f"Running Phase 0 subgraph for disease: {disease_name}")
    
    try:
        # Initialize Phase 0 state
        phase0_input: Phase0State = create_initial_phase0_state(disease_name=disease_name)
        
        # Create and run Phase 0 workflow
        phase0_workflow = create_phase0_workflow()
        phase0_result = phase0_workflow.invoke(phase0_input)
        
        # Extract results
        final_priors = phase0_result.get("final_priors", [])
        phase0_error = phase0_result.get("error")
        
        logger.info(f"Phase 0 completed: {len(final_priors)} prior biomarkers identified")
        
        # Update global state
        return {
            **state,
            "prior_biomarkers": final_priors,
            "phase0_state": phase0_result,
            "current_phase": "phase0_complete",
            "error": phase0_error
        }
    
    except Exception as e:
        error_msg = f"Error in Phase 0 subgraph: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "prior_biomarkers": [],
            "phase0_state": None,
            "current_phase": "phase0_error",
            "error": error_msg
        }


# ============================================================================
# Placeholder Nodes for Future Phases
# ============================================================================
def phase1_placeholder(state: AgentState) -> AgentState:
    """
    Placeholder for Phase 1 (to be implemented).
    
    Phase 1 will handle pathway analysis and enrichment.
    """
    logger.info("Phase 1 placeholder - not yet implemented")
    return {
        **state,
        "current_phase": "phase1_placeholder"
    }


def finalize_results(state: AgentState) -> AgentState:
    """
    Finalize and format results from all phases.
    
    Args:
        state: Current AgentState
    
    Returns:
        Updated AgentState with formatted results
    """
    logger.info("Finalizing results")
    
    results = {
        "disease": state.get("disease_name"),
        "prior_biomarkers": state.get("prior_biomarkers", []),
        "num_biomarkers": len(state.get("prior_biomarkers", [])),
        "phase0_details": state.get("phase0_state"),
        "status": "success" if not state.get("error") else "error",
        "error": state.get("error")
    }
    
    return {
        **state,
        "results": results,
        "current_phase": "complete"
    }


# ============================================================================
# Main Graph Construction
# ============================================================================
def create_main_graph():
    """
    Create and compile the main MetaboAgent graph.
    
    This function constructs the parent StateGraph that orchestrates
    the entire pipeline, integrating all phases.
    
    Returns:
        Compiled LangGraph workflow (runnable)
    
    Example:
        >>> main_graph = create_main_graph()
        >>> result = main_graph.invoke({
        ...     "disease_name": "diabetes",
        ...     "prior_biomarkers": [],
        ...     "phase0_state": None,
        ...     "current_phase": "start",
        ...     "results": None,
        ...     "error": None
        ... })
        >>> print(result["prior_biomarkers"])
    """
    logger.info("Creating main MetaboAgent graph")
    
    # Initialize StateGraph with AgentState
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("phase0_subgraph", run_phase0_subgraph)
    workflow.add_node("phase1_placeholder", phase1_placeholder)
    workflow.add_node("finalize_results", finalize_results)
    
    # Define edges
    workflow.set_entry_point("phase0_subgraph")
    workflow.add_edge("phase0_subgraph", "phase1_placeholder")
    workflow.add_edge("phase1_placeholder", "finalize_results")
    workflow.add_edge("finalize_results", END)
    
    # Compile the graph
    compiled_workflow = workflow.compile()
    
    logger.info("Main graph created successfully")
    
    return compiled_workflow


# ============================================================================
# Convenience Function
# ============================================================================
def run_metaboagent(disease_name: str) -> Dict[str, Any]:
    """
    Convenience function to run the complete MetaboAgent pipeline.
    
    Args:
        disease_name: Name of the disease to analyze
    
    Returns:
        Dictionary containing results from all phases
    
    Example:
        >>> results = run_metaboagent("diabetes")
        >>> print(f"Found {results['num_biomarkers']} biomarkers")
        >>> print(results["prior_biomarkers"])
    """
    logger.info(f"Running MetaboAgent for disease: {disease_name}")
    
    # Initialize state
    initial_state: AgentState = {
        "disease_name": disease_name,
        "prior_biomarkers": [],
        "phase0_state": None,
        "current_phase": "start",
        "results": None,
        "error": None
    }
    
    # Create and run main graph
    main_graph = create_main_graph()
    final_state = main_graph.invoke(initial_state)
    
    # Return results
    return final_state.get("results", {})
