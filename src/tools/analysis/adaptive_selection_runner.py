"""
Adaptive Feature Selection Workflow (Encapsulated)

This module is now a thin wrapper around the active consensus engine
`get_adaptive_consensus_features()`.

It only prepares threshold runs (30% -> 35% -> 40%) and delegates all final
decision logic to the single source of truth.
"""

import pandas as pd
import json
import os
from collections import Counter
import traceback

from src.tools.analysis.feature_selection_tools import (
    run_random_forest_selector,
    run_lasso_selector,
    run_mrmr_selector,
    perform_stability_selection,
    calculate_frequencies_from_logs,
    get_adaptive_consensus_features,
)


def run_adaptive_feature_selection_workflow(
    data_path: str,
    group_col: str = 'Group',
    is_balanced: bool = False,
    thresholds: list = None,
    min_features: int = 5,
    stability_iterations: int = 100,
    output_path: str = 'data/metabolites_consensus_features.csv'
) -> str:
    """
    Encapsulated Adaptive Feature Selection Workflow (Funnel Strategy).
    
    Implements branched logic based on data balance:
    - Path A (Balanced): Run RF+Lasso+mRMR ONCE → Voting Descent (N→3 votes)
    - Path B (Imbalanced): Run Stability Selection (Bootstrap) → Frequency Filter → Top N%
    - Fallback: Union of methods at max threshold
    
    Args:
        data_path: Path to input CSV file
        group_col: Name of the target/group column (default: 'Group')
        is_balanced: Whether data is balanced (determines strategy path)
        thresholds: List of feature percentage thresholds to try (default: [0.30, 0.35, 0.40])
        min_features: Minimum features required for success (default: 5)
        stability_iterations: Number of bootstrap iterations for imbalanced mode (default: 100)
        output_path: Path to save selected features (default: 'data/metabolites_consensus_features.csv')
    
    Returns:
        JSON string with success status, feature count, strategy used, and output path
    """
    
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40]
    
    try:
        # Load and validate data
        data = pd.read_csv(data_path)
        
        # Exclude non-feature columns
        feature_cols = [c for c in data.columns if c not in ['Sample_ID', group_col]]
        n_total = len(feature_cols)
        
        if n_total == 0:
            return json.dumps({
                "error": "No feature columns found after excluding Sample_ID and Group",
                "traceback": ""
            })
        
        final_features = []
        strategy_used = "unknown"
        threshold_runs = []
        
        # === PATH A: Balanced Mode ===
        if is_balanced:
            print(f"[Strategy] Balanced Mode Detected. Preparing threshold runs for active consensus engine...")
            
            for p in thresholds:
                n = int(n_total * p)
                print(f"  > Testing threshold {p:.0%} (Top {n} features)...")
                
                # Run methods (ONCE)
                rf_result = run_random_forest_selector(data_path, group_col, n_features=n)
                rf = json.loads(rf_result)
                
                la_result = run_lasso_selector(data_path, group_col, n_features=n)
                la = json.loads(la_result)
                
                mr_result = run_mrmr_selector(data_path, group_col, n_features=n)
                mr = json.loads(mr_result)
                
                # Extract feature lists
                lists = [
                    rf.get('selected_features', []) if 'error' not in rf else [],
                    la.get('selected_features', []) if 'error' not in la else [],
                    mr.get('selected_features', []) if 'error' not in mr else []
                ]
                
                print(f"    RF: {len(lists[0])}, LASSO: {len(lists[1])}, mRMR: {len(lists[2])}")
                freq_data = {
                    "random_forest": {f: 1 for f in lists[0]},
                    "lasso": {f: 1 for f in lists[1]},
                    "mrmr": {f: 1 for f in lists[2]},
                }
                threshold_runs.append({
                    "selection_threshold": p,
                    "run_mode": "balanced",
                    "frequencies": freq_data
                })
        
        # === PATH B: Imbalanced Mode ===
        else:
            print(f"[Strategy] Imbalanced Mode Detected. Preparing threshold runs for active consensus engine...")
            
            for p in thresholds:
                n = int(n_total * p)
                temp_dir = f"temp_stab_{int(p*100)}"
                
                print(f"  > Testing threshold {p:.0%} (Top {n} features, {stability_iterations} iters)...")
                
                # Run Stability Selection
                stab_result = perform_stability_selection(
                    data_path=data_path,
                    group_col=group_col,
                    iterations=stability_iterations,
                    methods=[
                        'run_random_forest_selector',
                        'run_lasso_selector',
                        'run_mrmr_selector'
                    ],
                    top_percentage=p,
                    temp_dir=temp_dir
                )
                
                stab_data = json.loads(stab_result)
                
                if 'error' in stab_data:
                    print(f"    Warning: {stab_data['error']}")
                    continue
                
                # Calculate Frequencies
                freq_json = calculate_frequencies_from_logs(temp_dir)
                freq_data = json.loads(freq_json)
                
                if 'error' in freq_data:
                    print(f"    Warning: {freq_data['error']}")
                    continue
                threshold_runs.append({
                    "selection_threshold": p,
                    "run_mode": "imbalanced",
                    "frequencies": freq_data
                })

        if not threshold_runs:
            return json.dumps({
                "error": "No valid threshold runs were produced",
                "traceback": ""
            })

        consensus_json = get_adaptive_consensus_features(
            json.dumps({"threshold_runs": threshold_runs}, ensure_ascii=False),
            min_features=min_features
        )
        consensus_result = json.loads(consensus_json)
        if not consensus_result.get("success", False):
            return json.dumps({
                "error": consensus_result.get("error", "Consensus selection failed"),
                "traceback": ""
            })

        final_features = consensus_result.get("consensus_features", [])
        strategy_used = consensus_result.get("strategy_used", "unified_active_consensus")
        
        # === Save Result ===
        cols_to_save = [group_col] + final_features
        
        if 'Sample_ID' in data.columns:
            cols_to_save = ['Sample_ID'] + cols_to_save
        
        data[cols_to_save].to_csv(output_path, index=False)
        
        return json.dumps({
            "success": True,
            "features_count": len(final_features),
            "strategy": strategy_used,
            "output_path": output_path,
            "is_balanced": is_balanced
        })
    
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        })
