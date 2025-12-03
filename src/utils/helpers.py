"""
Helper Functions and Utilities
"""

import numpy as np
from typing import List, Tuple


def normalize_scores(scores: List[float], method: str = 'minmax') -> List[float]:
    """
    Normalize scores to [0, 1] range
    
    Parameters:
    -----------
    scores : List[float]
        List of scores to normalize
    method : str, default='minmax'
        Normalization method: 'minmax' or 'zscore'
        
    Returns:
    --------
    normalized : List[float]
        Normalized scores
    """
    scores_array = np.array(scores)
    
    if method == 'minmax':
        min_val = scores_array.min()
        max_val = scores_array.max()
        if max_val == min_val:
            return [0.5] * len(scores)
        return ((scores_array - min_val) / (max_val - min_val)).tolist()
    
    elif method == 'zscore':
        mean_val = scores_array.mean()
        std_val = scores_array.std()
        if std_val == 0:
            return [0.5] * len(scores)
        normalized = (scores_array - mean_val) / std_val
        # Clip to [-3, 3] and then scale to [0, 1]
        normalized = np.clip(normalized, -3, 3) / 6 + 0.5
        return normalized.tolist()
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def sigmoid(x: float, steepness: float = 1.0) -> float:
    """
    Apply sigmoid function
    
    Parameters:
    -----------
    x : float
        Input value
    steepness : float, default=1.0
        Steepness parameter
        
    Returns:
    --------
    output : float
        Sigmoid output in range [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-steepness * x))


def merge_recommendations(cf_recs: List[Tuple[int, float]],
                         arm_recs: List[Tuple[int, float]],
                         alpha: float = 0.7,
                         normalize: bool = True) -> List[Tuple[int, float]]:
    """
    Merge recommendations from two algorithms
    
    Parameters:
    -----------
    cf_recs : List[Tuple[int, float]]
        Recommendations from collaborative filtering
    arm_recs : List[Tuple[int, float]]
        Recommendations from association rule mining
    alpha : float, default=0.7
        Weight for CF recommendations (1-alpha for ARM)
    normalize : bool, default=True
        Whether to normalize scores first
        
    Returns:
    --------
    merged : List[Tuple[int, float]]
        Merged recommendations sorted by score
    """
    # Create dictionaries for easy lookup
    cf_dict = dict(cf_recs)
    arm_dict = dict(arm_recs)
    
    # Get all unique items
    all_items = set(cf_dict.keys()).union(set(arm_dict.keys()))
    
    merged_scores = {}
    
    for item_id in all_items:
        cf_score = cf_dict.get(item_id, 0.0)
        arm_score = arm_dict.get(item_id, 0.0)
        
        if normalize:
            # Normalize to [0, 1]
            cf_score = sigmoid(cf_score)
            arm_score = sigmoid(arm_score)
        
        # Weighted combination
        merged_score = alpha * cf_score + (1 - alpha) * arm_score
        merged_scores[item_id] = merged_score
    
    # Sort by score
    sorted_recs = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_recs


def get_top_n(recommendations: List[Tuple[int, float]], n: int = 10) -> List[Tuple[int, float]]:
    """
    Get top N recommendations
    
    Parameters:
    -----------
    recommendations : List[Tuple[int, float]]
        List of (item_id, score) tuples
    n : int, default=10
        Number of items to return
        
    Returns:
    --------
    top_n : List[Tuple[int, float]]
        Top N recommendations
    """
    return recommendations[:n]


def deduplicate_recommendations(recommendations: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Remove duplicate items from recommendations
    
    Parameters:
    -----------
    recommendations : List[Tuple[int, float]]
        List of (item_id, score) tuples
        
    Returns:
    --------
    deduplicated : List[Tuple[int, float]]
        Deduplicated recommendations
    """
    seen = set()
    result = []
    
    for item_id, score in recommendations:
        if item_id not in seen:
            seen.add(item_id)
            result.append((item_id, score))
    
    return result
