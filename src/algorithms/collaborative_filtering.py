"""
Collaborative Filtering using Matrix Factorization

This module implements matrix factorization for collaborative filtering
using alternating least squares (ALS) optimization.
"""

import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import pandas as pd
from typing import Tuple, Dict, List, Optional, Literal


class CollaborativeFiltering:
    """
    Collaborative Filtering using Matrix Factorization
    
    Decomposes user-item interaction matrix into user and item
    latent factor matrices using Non-negative Matrix Factorization.
    
    Parameters:
    -----------
    n_factors : int, default=50
        Number of latent factors
    n_iterations : int, default=200
        Maximum number of iterations for NMF
    random_state : int, default=42
        Random seed for reproducibility
    init : str, default='nndsvd'
        Initialization method for NMF
    """
    
    def __init__(self, n_factors: int = 50, n_iterations: int = 200, 
                 random_state: int = 42, init: Literal['nndsvd', 'random', 'nndsvda', 'nndsvdar', 'custom'] = 'nndsvd'):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.init = init
        
        self.user_factors = None
        self.item_factors = None
        self.user_id_map = None
        self.item_id_map = None
        self.interaction_matrix = None
        self.user_means = None
        
    def fit(self, interactions_df: pd.DataFrame) -> 'CollaborativeFiltering':
        """
        Fit the collaborative filtering model
        
        Parameters:
        -----------
        interactions_df : pd.DataFrame
            DataFrame with columns: ['user_id', 'item_id', 'rating']
            rating can be implicit (1 for interaction) or explicit
            
        Returns:
        --------
        self : CollaborativeFiltering
            Fitted model
        """
        # Map user and item IDs to indices
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        # Create reverse mapping
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        # Build sparse interaction matrix
        user_indices = interactions_df['user_id'].map(self.user_id_map)
        item_indices = interactions_df['item_id'].map(self.item_id_map)
        ratings = interactions_df['rating'].values
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.interaction_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        
        # Dynamically adjust n_factors to fit matrix constraints
        # For NMF with nndsvd init: n_components <= min(n_samples, n_features)
        max_factors = min(n_users, n_items)
        actual_factors = min(self.n_factors, max(1, max_factors - 1))
        
        # Apply matrix factorization using NMF
        nmf = NMF(n_components=actual_factors, 
                   max_iter=self.n_iterations,
                   random_state=self.random_state,
                   init=self.init)  # type: ignore
        
        self.user_factors = nmf.fit_transform(self.interaction_matrix)
        self.item_factors = nmf.components_.T
        
        # Calculate user means for bias correction
        self.user_means = np.array([
            self.interaction_matrix.getrow(i).data.mean() 
            if self.interaction_matrix.getrow(i).nnz > 0 else 0
            for i in range(n_users)
        ])
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair
        
        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID
            
        Returns:
        --------
        score : float
            Predicted rating/score
        """
        if not self.user_id_map or not self.item_id_map or user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0.0
        
        if self.user_factors is None or self.item_factors is None or self.user_means is None:
            return 0.0
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # Compute dot product of latent factors
        score = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Add user bias
        score += self.user_means[user_idx]
        
        return float(np.clip(score, 0, 5))
    
    def recommend(self, user_id, n_items: int = 10, 
                  exclude_items: Optional[List] = None) -> List[Tuple]:
        """
        Generate top-N recommendations for a user
        
        Parameters:
        -----------
        user_id : str or int
            User ID
        n_items : int, default=10
            Number of recommendations
        exclude_items : List, optional
            Items to exclude (e.g., already purchased)
            
        Returns:
        --------
        recommendations : List[Tuple]
            List of (item_id, score) tuples sorted by score
        """
        if not self.user_id_map or user_id not in self.user_id_map:
            return []
        
        if self.user_factors is None or self.item_factors is None or self.user_means is None:
            return []
        
        if exclude_items is None:
            exclude_items = []
        
        user_idx = self.user_id_map[user_id]
        
        # Compute scores for all items
        all_scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        all_scores += self.user_means[user_idx]
        all_scores = np.clip(all_scores, 0, 5)
        
        # Create recommendations excluding specified items
        recommendations = []
        for item_idx in np.argsort(-all_scores):
            item_id = self.reverse_item_map[item_idx]
            
            if item_id not in exclude_items:
                score = float(all_scores[item_idx])
                recommendations.append((item_id, score))
                
                if len(recommendations) >= n_items:
                    break
        
        return recommendations
    
    def get_similar_items(self, item_id: int, n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to the given item
        
        Parameters:
        -----------
        item_id : int
            Item ID
        n_items : int, default=10
            Number of similar items
            
        Returns:
        --------
        similar_items : List[Tuple[int, float]]
            List of (item_id, similarity) tuples
        """
        if not self.item_id_map or item_id not in self.item_id_map or self.item_factors is None:
            return []
        
        item_idx = self.item_id_map[item_id]
        item_vector = self.item_factors[item_idx]
        
        # Compute cosine similarity
        similarities = np.dot(self.item_factors, item_vector)
        similarities /= (np.linalg.norm(self.item_factors, axis=1) * 
                        np.linalg.norm(item_vector) + 1e-10)
        
        # Exclude the item itself
        similarities[item_idx] = -1
        
        # Get top similar items
        similar_items = []
        for idx in np.argsort(-similarities)[:n_items]:
            sim_item_id = self.reverse_item_map[idx]
            similarity = float(similarities[idx])
            similar_items.append((sim_item_id, similarity))
        
        return similar_items
