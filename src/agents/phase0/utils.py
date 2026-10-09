"""
Phase 0 Utility Functions

Implements the Log-Dynamic Scaling algorithm for adaptive thresholding.

Author: MetaboAgent Team
Date: 2025-01-06
"""

import math
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_adaptive_threshold(scores: Dict[str, float], n_docs: int) -> float:
    """
    Calculate adaptive threshold using Log-Dynamic Scaling.
    
    This function implements the novel Log-Dynamic Scaling algorithm that adjusts
    the filtering threshold based on evidence density. Higher evidence counts lead
    to stricter filtering, while lower counts relax the threshold.
    
    Algorithm:
        1. Calculate dynamic factor: n(N) = clip(1.0 - 0.5 * log10(N), min=-0.5, max=1.5)
        2. Calculate threshold: τ = μ - n * σ
        
    Where:
        - N = Total number of retrieved evidence snippets
        - μ = Mean of log-probability scores
        - σ = Standard deviation of log-probability scores
        - n = Dynamic scaling factor (density-aware)
    
    Rationale:
        - High N (many evidence): n decreases → stricter threshold (μ - smaller_factor * σ)
        - Low N (few evidence): n increases → relaxed threshold (μ - larger_factor * σ)
        - Clipping ensures reasonable bounds even with extreme N values
    
    Args:
        scores: Dictionary mapping metabolite names to log-probability scores
        n_docs: Total number of retrieved evidence documents/snippets
    
    Returns:
        Adaptive threshold value (float)
    
    Example:
        >>> scores = {"metabolite1": 2.5, "metabolite2": 1.8, "metabolite3": 3.2}
        >>> threshold = calculate_adaptive_threshold(scores, n_docs=50)
        >>> print(f"Threshold: {threshold:.3f}")
    """
    if not scores:
        logger.warning("No scores provided for threshold calculation")
        return 0.0
    
    # Step 1: Calculate mean and standard deviation
    score_values = list(scores.values())
    n = len(score_values)
    
    if n == 0:
        logger.warning("Empty score values")
        return 0.0
    
    mean = sum(score_values) / n
    
    if n == 1:
        # Only one score, use it as threshold
        logger.info(f"Only one score available, using mean as threshold: {mean:.3f}")
        return mean
    
    variance = sum((x - mean) ** 2 for x in score_values) / (n - 1)
    std_dev = math.sqrt(variance)
    
    # Step 2: Calculate dynamic factor using Log-Dynamic Scaling
    # n(N) = clip(1.0 - 0.5 * log10(N), min=-0.5, max=1.5)
    if n_docs <= 0:
        logger.warning(f"Invalid n_docs value: {n_docs}, using default factor 1.0")
        dynamic_factor = 1.0
    else:
        log_n = math.log10(n_docs)
        dynamic_factor = 1.0 - 0.5 * log_n
        
        # Clip to reasonable bounds
        dynamic_factor = max(-0.5, min(1.5, dynamic_factor))
    
    # Step 3: Calculate adaptive threshold
    # τ = μ - n * σ
    threshold = mean - dynamic_factor * std_dev
    
    logger.info(
        f"Adaptive threshold calculation: "
        f"n_docs={n_docs}, mean={mean:.3f}, std={std_dev:.3f}, "
        f"dynamic_factor={dynamic_factor:.3f}, threshold={threshold:.3f}"
    )
    
    return threshold


def filter_candidates_by_threshold(
    scores: Dict[str, float],
    threshold: float
) -> List[str]:
    """
    Filter candidates based on threshold.
    
    Args:
        scores: Dictionary mapping metabolite names to scores
        threshold: Threshold value
    
    Returns:
        List of metabolite names that pass the threshold
    """
    filtered = [name for name, score in scores.items() if score > threshold]
    
    logger.info(
        f"Filtered {len(filtered)}/{len(scores)} candidates "
        f"(threshold={threshold:.3f})"
    )
    
    return filtered
