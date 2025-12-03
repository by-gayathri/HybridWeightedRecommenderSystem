"""
Evaluation Metrics for Recommendation Systems
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set


class EvaluationMetrics:
    """
    Compute evaluation metrics for recommendation systems
    """
    
    @staticmethod
    def precision_at_k(recommendations: List[int], 
                      ground_truth: Set[int], k: int = 10) -> float:
        """
        Calculate Precision@K
        
        Proportion of recommended items that are in ground truth
        
        Parameters:
        -----------
        recommendations : List[int]
            List of recommended item IDs
        ground_truth : Set[int]
            Set of relevant items
        k : int, default=10
            Cutoff rank
            
        Returns:
        --------
        precision : float
            Precision@K score
        """
        if k <= 0:
            return 0.0
        
        rec_at_k = set(recommendations[:k])
        intersection = rec_at_k.intersection(ground_truth)
        
        return len(intersection) / min(k, len(rec_at_k)) if rec_at_k else 0.0
    
    @staticmethod
    def recall_at_k(recommendations: List[int],
                   ground_truth: Set[int], k: int = 10) -> float:
        """
        Calculate Recall@K
        
        Proportion of relevant items that are recommended
        
        Parameters:
        -----------
        recommendations : List[int]
            List of recommended item IDs
        ground_truth : Set[int]
            Set of relevant items
        k : int, default=10
            Cutoff rank
            
        Returns:
        --------
        recall : float
            Recall@K score
        """
        if len(ground_truth) == 0:
            return 0.0
        
        rec_at_k = set(recommendations[:k])
        intersection = rec_at_k.intersection(ground_truth)
        
        return len(intersection) / len(ground_truth)
    
    @staticmethod
    def f1_score_at_k(recommendations: List[int],
                     ground_truth: Set[int], k: int = 10) -> float:
        """
        Calculate F1-Score@K
        
        Harmonic mean of precision and recall
        
        Parameters:
        -----------
        recommendations : List[int]
            List of recommended item IDs
        ground_truth : Set[int]
            Set of relevant items
        k : int, default=10
            Cutoff rank
            
        Returns:
        --------
        f1 : float
            F1-Score@K
        """
        precision = EvaluationMetrics.precision_at_k(recommendations, ground_truth, k)
        recall = EvaluationMetrics.recall_at_k(recommendations, ground_truth, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(recommendations: List[Tuple[int, float]],
                 ground_truth: Set[int], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K
        
        Measures ranking quality considering position
        
        Parameters:
        -----------
        recommendations : List[Tuple[int, float]]
            List of (item_id, score) tuples
        ground_truth : Set[int]
            Set of relevant items
        k : int, default=10
            Cutoff rank
            
        Returns:
        --------
        ndcg : float
            NDCG@K score
        """
        # Calculate DCG
        dcg = 0.0
        for i, (item_id, _) in enumerate(recommendations[:k]):
            if item_id in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because ranking starts at 1
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(ground_truth))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mean_average_precision(recommendations_list: List[List[int]],
                              ground_truth_list: List[Set[int]], k: int = 10) -> float:
        """
        Calculate Mean Average Precision@K
        
        Average of precision values computed at k
        
        Parameters:
        -----------
        recommendations_list : List[List[int]]
            List of recommendations for multiple queries
        ground_truth_list : List[Set[int]]
            List of ground truth sets
        k : int, default=10
            Cutoff rank
            
        Returns:
        --------
        map_score : float
            MAP@K score
        """
        ap_scores = []
        
        for recommendations, ground_truth in zip(recommendations_list, ground_truth_list):
            if len(ground_truth) == 0:
                continue
            
            precision_sum = 0.0
            hits = 0
            
            for i, item_id in enumerate(recommendations[:k]):
                if item_id in ground_truth:
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            ap = precision_sum / min(k, len(ground_truth))
            ap_scores.append(ap)
        
        return float(np.mean(ap_scores)) if ap_scores else 0.0
    
    @staticmethod
    def coverage(all_recommendations: List[List[int]], 
                total_items: int) -> float:
        """
        Calculate Catalog Coverage
        
        Percentage of unique items recommended across all users
        
        Parameters:
        -----------
        all_recommendations : List[List[int]]
            List of recommendations for all users
        total_items : int
            Total number of items in catalog
            
        Returns:
        --------
        coverage : float
            Coverage percentage (0-1)
        """
        recommended_items = set()
        
        for recommendations in all_recommendations:
            recommended_items.update(recommendations)
        
        return len(recommended_items) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def diversity(recommendations: List[int], 
                 item_features: pd.DataFrame, 
                 feature_cols: List[str]) -> float:
        """
        Calculate diversity of recommendations
        
        Average dissimilarity between recommended items
        
        Parameters:
        -----------
        recommendations : List[int]
            List of recommended item IDs
        item_features : pd.DataFrame
            Dataframe with item features
        feature_cols : List[str]
            Columns to use for similarity computation
            
        Returns:
        --------
        diversity : float
            Diversity score (0-1, higher is more diverse)
        """
        if len(recommendations) <= 1:
            return 1.0
        
        # Get features for recommended items
        rec_features = item_features[item_features.index.isin(recommendations)][feature_cols]
        
        if len(rec_features) <= 1:
            return 1.0
        
        # Normalize features
        rec_features = (rec_features - rec_features.min()) / (rec_features.max() - rec_features.min() + 1e-10)
        
        # Calculate average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i in range(len(rec_features)):
            for j in range(i + 1, len(rec_features)):
                distance = np.linalg.norm(
                    np.array(rec_features.iloc[i].values) - np.array(rec_features.iloc[j].values)
                )
                total_distance += distance
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0.0
        
        # Normalize to 0-1 range (assuming max distance is sqrt(n_features))
        max_possible_distance = np.sqrt(len(feature_cols))
        
        return min(avg_distance / max_possible_distance, 1.0)
    
    @staticmethod
    def compute_metrics(recommendations: List[int],
                       ground_truth: Set[int],
                       k: int = 10) -> Dict[str, float]:
        """
        Compute multiple metrics at once
        
        Parameters:
        -----------
        recommendations : List[int]
            List of recommended item IDs
        ground_truth : Set[int]
            Set of relevant items
        k : int, default=10
            Cutoff rank
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary of metric names and values
        """
        return {
            'precision@k': EvaluationMetrics.precision_at_k(recommendations, ground_truth, k),
            'recall@k': EvaluationMetrics.recall_at_k(recommendations, ground_truth, k),
            'f1@k': EvaluationMetrics.f1_score_at_k(recommendations, ground_truth, k),
        }
