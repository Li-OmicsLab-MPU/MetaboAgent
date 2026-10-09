"""
LangGraph Agent State Definitions

This module defines TypedDict classes for managing state in LangGraph agents.
In LangGraph, state is passed between nodes in a graph and can be updated by
each node as the workflow progresses.

State Management Pattern:
1. Define state as a TypedDict with all fields that will be tracked
2. Each node in the graph receives the current state
3. Nodes return updates to the state (partial or complete)
4. LangGraph merges the updates into the current state
5. The updated state is passed to the next node

Key Principles:
- State should be serializable (JSON-compatible types)
- Use Optional[] for fields that may not always be present
- Document each field clearly
- Keep state focused on the workflow's needs

Example Usage:
    from typing import TypedDict, Optional, List
    from langgraph.graph import StateGraph
    
    class MyAgentState(TypedDict):
        messages: List[str]
        current_step: str
        results: Optional[dict]
    
    def my_node(state: MyAgentState) -> MyAgentState:
        # Process state
        return {
            "messages": state["messages"] + ["New message"],
            "current_step": "next_step"
        }
    
    graph = StateGraph(MyAgentState)
    graph.add_node("my_node", my_node)
    # ... add more nodes and edges
"""

from typing import TypedDict, Optional, List, Dict, Any


class MetaboAnalysisState(TypedDict):
    """
    State for metabolomics analysis workflows.
    
    This state tracks the progress of a metabolomics analysis from data loading
    through feature selection and model building.
    
    Fields:
        messages: List of messages exchanged during the workflow
        current_step: Name of the current workflow step
        data_file_path: Path to the input data file
        group_column: Name of the column containing group labels
        analysis_results: Results from data quality analysis
        cleaning_results: Results from data cleaning operations
        feature_selection_results: Results from feature selection
        model_results: Results from model training
        error: Error message if workflow fails
    """
    messages: List[str]
    current_step: str
    data_file_path: Optional[str]
    group_column: Optional[str]
    analysis_results: Optional[Dict[str, Any]]
    cleaning_results: Optional[Dict[str, Any]]
    feature_selection_results: Optional[Dict[str, Any]]
    model_results: Optional[Dict[str, Any]]
    error: Optional[str]


class LiteratureSearchState(TypedDict):
    """
    State for literature search workflows.
    
    This state tracks literature searches for metabolites, pathways, and
    biological mechanisms.
    
    Fields:
        messages: List of messages exchanged during the workflow
        query: Search query string
        search_results: Results from literature search
        metabolites: List of metabolites to search for
        pathways: List of pathways to search for
        summaries: Summaries of search results
        error: Error message if workflow fails
    """
    messages: List[str]
    query: str
    search_results: Optional[List[Dict[str, Any]]]
    metabolites: Optional[List[str]]
    pathways: Optional[List[str]]
    summaries: Optional[Dict[str, str]]
    error: Optional[str]


class PathwayAnalysisState(TypedDict):
    """
    State for pathway analysis workflows.
    
    This state tracks pathway enrichment and scoring analyses.
    
    Fields:
        messages: List of messages exchanged during the workflow
        metabolite_list: List of metabolites for pathway analysis
        pathway_scores: Pathway enrichment scores
        significant_pathways: List of significantly enriched pathways
        pathway_details: Detailed information about pathways
        error: Error message if workflow fails
    """
    messages: List[str]
    metabolite_list: List[str]
    pathway_scores: Optional[Dict[str, float]]
    significant_pathways: Optional[List[str]]
    pathway_details: Optional[Dict[str, Any]]
    error: Optional[str]


# Example of a minimal state for simple workflows
class SimpleAgentState(TypedDict):
    """
    Minimal state for simple agent workflows.
    
    Use this for basic workflows that don't require complex state tracking.
    
    Fields:
        messages: List of messages exchanged during the workflow
        result: Final result of the workflow
    """
    messages: List[str]
    result: Optional[Any]
