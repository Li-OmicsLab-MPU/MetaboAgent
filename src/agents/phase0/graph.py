"""
Phase 0 Subgraph Construction

Builds the LangGraph StateGraph for the Phase 0 Researcher Agent.

Graph Flow:
    START -> check_cache -> [cache hit: END | cache miss: generate_candidates] -> 
    stage1_coarse_screen -> collect_evidence_pack -> score_evidence_pack ->
    rank_and_threshold -> define_feature_rules ->
    save_to_cache -> END

Author: MetaboAgent Team
Date: 2025-01-06
Updated: 2025-01-07 (Added Node 5: define_feature_rules)
Updated: 2026-03-27 (Added caching support)
"""

import logging
import time
from langgraph.graph import StateGraph, END

from src.agents.phase0.state import Phase0State
from src.agents.phase0.nodes import (
    generate_candidates,
    prepare_disease_pathways,
    stage1_coarse_screen,
    collect_evidence_pack,
    score_evidence_pack,
    rank_and_threshold,
    define_feature_rules,
)
from src.agents.phase0.cache_manager import Phase0CacheManager

logger = logging.getLogger(__name__)


def create_phase0_workflow():
    """
    Create and compile the Phase 0 subgraph workflow with caching support.
    
    This function constructs the StateGraph for the Researcher Agent,
    linking all nodes with cache checking:
    1. check_cache: Check if results exist in cache
    2. generate_candidates: Load candidates from disease_map.json (if cache miss)
    3. stage1_coarse_screen: Fast hit-count and pathway-based pruning
    4. collect_evidence_pack: Build structured evidence bundles
    5. score_evidence_pack: Score EvidencePack objects into biological priors
    6. rank_and_threshold: Apply the new mean + 0.5*std threshold rule
    7. define_feature_rules: Extract target pathways and metabolites
    8. save_to_cache: Save results to cache (if cache miss)
    
    Returns:
        Compiled LangGraph workflow (runnable)
    
    Example:
        >>> workflow = create_phase0_workflow()
        >>> result = workflow.invoke({
        ...     "disease_name": "diabetes",
        ...     "candidates": [],
        ...     "evidence_map": {},
        ...     "logprob_scores": {},
        ...     "final_priors": [],
        ...     "feature_definitions": {},
        ...     "error": None,
        ...     "cache_hit": False,
        ...     "use_cache": True,
        ...     "force_refresh": False,
        ...     "execution_time": 0.0,
        ...     "max_candidates": 50
        ... })
        >>> print(result["final_priors"])
        >>> print(result["feature_definitions"]["target_pathways"])
    """
    logger.info("Creating Phase 0 workflow with caching support")
    
    # Initialize cache manager
    cache_manager = Phase0CacheManager()
    
    # Initialize StateGraph with Phase0State
    workflow = StateGraph(Phase0State)
    
    # Define cache check node
    def check_cache(state: Phase0State) -> Phase0State:
        """Check if cached results exist."""
        disease_name = state.get("disease_name", "")
        use_cache = state.get("use_cache", True)  # Default: use cache
        force_refresh = state.get("force_refresh", False)
        
        if force_refresh:
            logger.info("Force refresh enabled, bypassing cache and rerunning Phase 0")
            return {**state, "cache_hit": False}

        if not use_cache:
            logger.info("Cache disabled, proceeding with full Phase 0 execution")
            return {**state, "cache_hit": False}
        
        cached_result = cache_manager.get_cache(disease_name)
        
        if cached_result:
            # Convert cached result to Phase0State format
            confirmed_biomarkers = cached_result.get("confirmed_biomarkers", []) or []
            biomarker_names = [
                b["name"]
                for b in confirmed_biomarkers
                if isinstance(b, dict) and b.get("name")
            ]
            if not biomarker_names:
                feature_defs = cached_result.get("feature_definitions", {}) or {}
                biomarker_names = list(feature_defs.get("target_metabolites", []) or [])
            if not confirmed_biomarkers and biomarker_names:
                confirmed_biomarkers = cache_manager._extract_biomarkers({
                    "final_priors": biomarker_names,
                    "feature_definitions": cached_result.get("feature_definitions", {}),
                })

            # Older cache entries may predate feature-rule generation. Rebuild
            # pathway rules from confirmed HMDB identifiers instead of sending
            # an empty pathway contract to Phase 1 and the evidence UI.
            feature_definitions = dict(cached_result.get("feature_definitions", {}) or {})
            if not feature_definitions.get("target_pathways") and confirmed_biomarkers:
                synthetic_candidates = [
                    {
                        "id": item.get("hmdb_id") or item.get("hmdb") or item.get("id"),
                        "name": item.get("name"),
                        "pathways": item.get("pathways", []),
                    }
                    for item in confirmed_biomarkers
                    if isinstance(item, dict) and item.get("name")
                ]
                if synthetic_candidates:
                    repaired = define_feature_rules({
                        **state,
                        "candidates": synthetic_candidates,
                        "stage1_candidates": synthetic_candidates,
                        "final_priors": biomarker_names,
                        "memory_pathway_hints": cached_result.get("memory_pathway_hints", []),
                    })
                    repaired_definitions = repaired.get("feature_definitions", {}) or {}
                    if repaired_definitions.get("target_pathways"):
                        feature_definitions = repaired_definitions
                        logger.info(
                            "Backfilled %d pathway rules for legacy Phase 0 cache entry",
                            len(feature_definitions.get("target_pathways", [])),
                        )

            cached_state = {
                **state,
                "cache_hit": True,
                "candidates": [],  # Not needed when using cache
                "stage1_candidates": [],
                "coarse_screen_map": cached_result.get("coarse_screen_map", {}) or {},
                "disease_pathway_pack": {},
                "disease_core_pathways": [],
                "external_disease_pathways_broad": [],
                "top_pathways": [],
                "evidence_packs": cached_result.get("evidence_packs", {}) or {},
                "scoring_results": cached_result.get("scoring_results", {}) or {},
                "evidence_map": {},
                "logprob_scores": {},
                "final_priors": biomarker_names,
                "feature_definitions": feature_definitions,
                "normalization_stats": cached_result.get("normalization_stats", {}),
                "execution_time": 0.0,  # Cache hit is instant
                "confirmed_biomarkers": confirmed_biomarkers,
                "candidate_scores_summary": cached_result.get("candidate_scores_summary", []),
                "selection_threshold": cached_result.get("selection_threshold", {}),
                "screening_summary": cached_result.get("screening_summary", {}),
                "scoring_summary": cached_result.get("scoring_summary", {}),
                "evidence_dir": cached_result.get("evidence_dir"),
                "full_result_path": cached_result.get("full_result_path"),
                "error": None
            }
            cache_manager.export_phase0_result(
                disease_name=disease_name,
                phase0_result=cached_state,
                cache_hit=True,
            )
            return cached_state
        else:
            return {**state, "cache_hit": False}
    
    # Define save cache node
    def save_to_cache(state: Phase0State) -> Phase0State:
        """Save Phase 0 results to cache."""
        disease_name = state.get("disease_name", "")
        use_cache = state.get("use_cache", True)
        cache_hit = state.get("cache_hit", False)
        execution_time = state.get("execution_time", 0.0)
        
        if use_cache and not cache_hit:
            # Calculate metadata
            metadata = {
                "total_candidates": len(state.get("candidates", [])),
                "confirmed_count": len(state.get("final_priors", [])),
                "confirmation_rate": (
                    len(state.get("final_priors", [])) / len(state.get("candidates", []))
                    if state.get("candidates") else 0.0
                ),
                "llm_calls": len(state.get("logprob_scores", {})),
                "pubmed_queries": len(state.get("evidence_map", {}))
            }
            
            cache_manager.save_cache(
                disease_name=disease_name,
                phase0_result=state,
                execution_time=execution_time,
                metadata=metadata
            )

        cache_manager.export_phase0_result(
            disease_name=disease_name,
            phase0_result=state,
            cache_hit=cache_hit,
        )
        
        return state
    
    # Wrapper to track execution time
    def timed_generate_candidates(state: Phase0State) -> Phase0State:
        """Generate candidates and start timing."""
        start_time = time.time()
        result = generate_candidates(state)
        return {**result, "execution_time": time.time() - start_time}
    
    def timed_stage1_coarse_screen(state: Phase0State) -> Phase0State:
        """Run coarse screening and update timing."""
        start_time = time.time()
        result = stage1_coarse_screen(state)
        elapsed = state.get("execution_time", 0.0) + (time.time() - start_time)
        return {**result, "execution_time": elapsed}

    def timed_prepare_disease_pathways(state: Phase0State) -> Phase0State:
        """Prepare runtime disease pathways and update timing."""
        start_time = time.time()
        result = prepare_disease_pathways(state)
        elapsed = state.get("execution_time", 0.0) + (time.time() - start_time)
        return {**result, "execution_time": elapsed}
    
    def timed_collect_evidence_pack(state: Phase0State) -> Phase0State:
        """Collect evidence packs and update timing."""
        start_time = time.time()
        result = collect_evidence_pack(state)
        elapsed = state.get("execution_time", 0.0) + (time.time() - start_time)
        return {**result, "execution_time": elapsed}
    
    def timed_score_evidence_pack(state: Phase0State) -> Phase0State:
        """Score evidence packs and update timing."""
        start_time = time.time()
        result = score_evidence_pack(state)
        elapsed = state.get("execution_time", 0.0) + (time.time() - start_time)
        return {**result, "execution_time": elapsed}

    def timed_rank_and_threshold(state: Phase0State) -> Phase0State:
        """Rank candidates and update timing."""
        start_time = time.time()
        result = rank_and_threshold(state)
        elapsed = state.get("execution_time", 0.0) + (time.time() - start_time)
        return {**result, "execution_time": elapsed}
    
    def timed_define_feature_rules(state: Phase0State) -> Phase0State:
        """Define features and update timing."""
        start_time = time.time()
        result = define_feature_rules(state)
        elapsed = state.get("execution_time", 0.0) + (time.time() - start_time)
        logger.info(f"Phase 0 total execution time: {elapsed:.2f} seconds")
        return {**result, "execution_time": elapsed}
    
    # Add nodes
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("generate_candidates", timed_generate_candidates)
    workflow.add_node("prepare_disease_pathways", timed_prepare_disease_pathways)
    workflow.add_node("stage1_coarse_screen", timed_stage1_coarse_screen)
    workflow.add_node("collect_evidence_pack", timed_collect_evidence_pack)
    workflow.add_node("score_evidence_pack", timed_score_evidence_pack)
    workflow.add_node("rank_and_threshold", timed_rank_and_threshold)
    workflow.add_node("define_feature_rules", timed_define_feature_rules)
    workflow.add_node("save_to_cache", save_to_cache)
    
    # Define edges with conditional routing
    workflow.set_entry_point("check_cache")
    
    # Conditional edge: if cache hit, skip to save; otherwise, proceed
    def route_after_cache(state: Phase0State) -> str:
        """Route based on cache hit."""
        if state.get("cache_hit", False):
            return "end"
        else:
            return "generate_candidates"
    
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {
            "end": END,
            "generate_candidates": "generate_candidates"
        }
    )
    
    workflow.add_edge("generate_candidates", "prepare_disease_pathways")
    workflow.add_edge("prepare_disease_pathways", "stage1_coarse_screen")
    workflow.add_edge("stage1_coarse_screen", "collect_evidence_pack")
    workflow.add_edge("collect_evidence_pack", "score_evidence_pack")
    workflow.add_edge("score_evidence_pack", "rank_and_threshold")
    workflow.add_edge("rank_and_threshold", "define_feature_rules")
    workflow.add_edge("define_feature_rules", "save_to_cache")
    workflow.add_edge("save_to_cache", END)
    
    # Compile the graph
    compiled_workflow = workflow.compile()
    
    logger.info("Phase 0 workflow created successfully with caching")
    
    return compiled_workflow
