"""
Hybrid Recommender System

Combines Matrix Factorization (Collaborative Filtering) and 
Apriori Algorithm (Association Rule Mining) for recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from src.algorithms.collaborative_filtering import CollaborativeFiltering
from src.algorithms.apriori_arm import AprioriARM
from src.utils.helpers import merge_recommendations, deduplicate_recommendations, get_top_n


class HybridRecommenderSystem:
    """
    Hybrid Recommender System combining CF and ARM
    
    Parameters:
    -----------
    alpha : float, default=0.7
        Weight for collaborative filtering (1-alpha for ARM)
        Higher alpha = more personalization
    n_factors : int, default=50
        Number of latent factors for matrix factorization
    min_support : float, default=0.01
        Minimum support for Apriori algorithm
    min_confidence : float, default=0.3
        Minimum confidence for association rules
    """
    
    def __init__(self, alpha: float = 0.7, n_factors: int = 50,
                 min_support: float = 0.01, min_confidence: float = 0.3):
        self.alpha = alpha
        self.n_factors = n_factors
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        # Initialize algorithms
        self.cf_model = CollaborativeFiltering(n_factors=n_factors)
        self.arm_model = AprioriARM(min_support=min_support,
                                    min_confidence=min_confidence)
        
        self.is_fitted = False
        self.user_items = None  # Track items interacted by each user
        
    def fit(self, interactions_df: pd.DataFrame, transactions_df: Optional[pd.DataFrame] = None) -> 'HybridRecommenderSystem':
        """
        Fit both CF and ARM models
        
        Parameters:
        -----------
        interactions_df : pd.DataFrame
            User-item interactions with columns: ['user_id', 'item_id', 'rating']
        transactions_df : pd.DataFrame, optional
            Transaction data with columns: ['user_id', 'item_id']
            If not provided, created from interactions_df
            
        Returns:
        --------
        self : HybridRecommenderSystem
            Fitted model
        """
        # Train Collaborative Filtering
        self.cf_model.fit(interactions_df)
        
        # Prepare transaction data for ARM
        if transactions_df is None:
            # Create from interactions
            transactions_df = interactions_df[['user_id', 'item_id']].copy()
        
        # Train Apriori ARM
        self.arm_model.fit(transactions_df)
        
        # Track user items for exclusion in recommendations
        self.user_items = interactions_df.groupby('user_id')['item_id'].apply(set).to_dict()
        
        self.is_fitted = True
        
        return self
    
    def get_recommendations(self, user_id: int, n_items: int = 10,
                          exclude_items: Optional[List[int]] = None,
                          return_scores: bool = True) -> List[Tuple[int, float]]:
        """
        Get hybrid recommendations for a user
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_items : int, default=10
            Number of recommendations
        exclude_items : List[int], optional
            Items to exclude (already purchased, etc.)
        return_scores : bool, default=True
            Whether to return scores with recommendations
            
        Returns:
        --------
        recommendations : List[Tuple[int, float]] or List[int]
            Recommended items with scores (or just item IDs if return_scores=False)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if exclude_items is None:
            exclude_items = []
        
        # Add user's previously interacted items to exclusion list
        if self.user_items and user_id in self.user_items:
            exclude_items = list(set(exclude_items) | self.user_items[user_id])
        
        # Get CF recommendations
        cf_recs = self.cf_model.recommend(user_id, n_items=n_items*2,
                                         exclude_items=exclude_items)
        
        # Get user's items for ARM
        user_items_list = list(self.user_items.get(user_id, [])) if self.user_items else []
        
        # Get ARM recommendations
        arm_recs = self.arm_model.get_recommendations(user_items_list,
                                                     n_items=n_items*2)
        
        # Merge recommendations
        if cf_recs or arm_recs:
            merged = merge_recommendations(cf_recs, arm_recs, self.alpha)
            merged = deduplicate_recommendations(merged)
            recommendations = get_top_n(merged, n_items)
        else:
            recommendations = []
        
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get CF-based similar items
        cf_similar = self.cf_model.get_similar_items(item_id, n_items)
        
        # Get ARM-based frequently bought together
        arm_related = self.arm_model.get_frequently_bought_together(item_id, n_items)
        
        # Merge
        merged = merge_recommendations(cf_similar, arm_related, self.alpha)
        
        return get_top_n(merged, n_items)
    
    def get_trending_items(self, interactions_df: pd.DataFrame,
                          n_items: int = 10, time_window: str = '30D') -> List[Tuple[int, float]]:
        """
        Get trending items based on recent interactions
        
        Parameters:
        -----------
        interactions_df : pd.DataFrame
            Interactions with timestamp
        n_items : int, default=10
            Number of items to return
        time_window : str, default='30D'
            Time window for recency
            
        Returns:
        --------
        trending : List[Tuple[int, float]]
            List of (item_id, popularity) tuples
        """
        # Filter recent interactions
        if 'timestamp' in interactions_df.columns:
            interactions_df = interactions_df.copy()
            interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
            cutoff_date = interactions_df['timestamp'].max() - pd.Timedelta(time_window)
            recent = interactions_df[interactions_df['timestamp'] >= cutoff_date]
        else:
            recent = interactions_df
        
        # Calculate popularity
        popularity = recent.groupby('item_id')['rating'].agg(['count', 'mean']).reset_index()
        popularity.columns = ['item_id', 'frequency', 'avg_rating']
        
        # Score: frequency * (avg_rating / 5)
        popularity['score'] = popularity['frequency'] * (popularity['avg_rating'] / 5)
        popularity = popularity.sort_values('score', ascending=False)
        
        trending = [(row['item_id'], row['score']) 
                   for _, row in popularity.head(n_items).iterrows()]
        
        return trending
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of trained models
        
        Returns:
        --------
        summary : Dict
            Dictionary with model statistics
        """
        if not self.is_fitted:
            return {'status': 'Model not fitted'}
        
        cf_users = len(self.cf_model.user_id_map) if self.cf_model.user_id_map else 0
        cf_items = len(self.cf_model.item_id_map) if self.cf_model.item_id_map else 0
        
        return {
            'status': 'Fitted',
            'alpha': self.alpha,
            'n_factors': self.n_factors,
            'cf_users': cf_users,
            'cf_items': cf_items,
            'arm_rules': len(self.arm_model.rules) if self.arm_model.rules is not None else 0,
            'arm_itemsets': len(self.arm_model.frequent_itemsets) if self.arm_model.frequent_itemsets is not None else 0,
            'arm_summary': self.arm_model.get_rules_summary()
        }
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import pickle
        
        model_data = {
            'alpha': self.alpha,
            'n_factors': self.n_factors,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'cf_model': self.cf_model,
            'arm_model': self.arm_model,
            'user_items': self.user_items,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HybridRecommenderSystem':
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load the model
            
        Returns:
        --------
        model : HybridRecommenderSystem
            Loaded model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance with saved parameters
        instance = cls(
            alpha=model_data['alpha'],
            n_factors=model_data['n_factors'],
            min_support=model_data['min_support'],
            min_confidence=model_data['min_confidence']
        )
        
        # Restore models
        instance.cf_model = model_data['cf_model']
        instance.arm_model = model_data['arm_model']
        instance.user_items = model_data['user_items']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
