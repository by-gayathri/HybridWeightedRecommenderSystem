"""
Apriori Algorithm for Association Rule Mining

This module implements the Apriori algorithm for discovering association rules
from transaction data using the mlxtend library.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from typing import List, Dict, Tuple


class AprioriARM:
    """
    Association Rule Mining using Apriori Algorithm
    
    Discovers frequent itemsets and generates association rules
    to identify product relationships and co-purchase patterns.
    
    Parameters:
    -----------
    min_support : float, default=0.01
        Minimum support threshold (0-1)
    min_confidence : float, default=0.3
        Minimum confidence threshold for rules
    min_lift : float, default=1.0
        Minimum lift threshold for rules
    max_itemset_size : int, default=3
        Maximum size of itemsets to generate
    """
    
    def __init__(self, min_support: float = 0.01, 
                 min_confidence: float = 0.3,
                 min_lift: float = 1.0,
                 max_itemset_size: int = 3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_itemset_size = max_itemset_size
        
        self.frequent_itemsets = None
        self.rules = None
        self.item_id_map = None
        self.reverse_item_map = None
        self.user_item_matrix = None
        
    def fit(self, baskets_df: pd.DataFrame) -> 'AprioriARM':
        """
        Fit the Apriori model
        
        Parameters:
        -----------
        baskets_df : pd.DataFrame
            DataFrame with columns: ['basket_id', 'items']
            where 'items' is a list of product IDs in each basket
            
        Returns:
        --------
        self : AprioriARM
            Fitted model
        """
        # Extract all unique items
        all_items = set()
        for items_list in baskets_df['items']:
            all_items.update(items_list)
        
        all_items = sorted(list(all_items))
        
        # Create item ID mappings
        self.item_id_map = {iid: idx for idx, iid in enumerate(all_items)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        # Create one-hot encoded basket matrix
        n_baskets = len(baskets_df)
        n_items = len(all_items)
        
        self.user_item_matrix = pd.DataFrame(0, 
                                             index=range(n_baskets),
                                             columns=all_items)
        
        for basket_idx, items_list in enumerate(baskets_df['items']):
            for item in items_list:
                self.user_item_matrix.at[basket_idx, item] = 1
        
        # Apply Apriori algorithm
        self.frequent_itemsets = apriori(
            self.user_item_matrix, 
            min_support=self.min_support,
            max_len=self.max_itemset_size,
            use_colnames=True
        )
        
        # Generate association rules
        if len(self.frequent_itemsets) > 1:
            self.rules = association_rules(
                self.frequent_itemsets,
                metric='confidence',
                min_threshold=self.min_confidence
            )
            
            # Calculate additional metrics
            if len(self.rules) > 0:
                self.rules['antecedent_len'] = self.rules['antecedents'].apply(len)
                self.rules['consequent_len'] = self.rules['consequents'].apply(len)
                
                # Sort by lift (quality of rule)
                self.rules = self.rules.sort_values('lift', ascending=False)
        else:
            self.rules = pd.DataFrame()
        
        return self
    
    def get_recommendations(self, items: List[int], n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Get recommendations based on items a user has interacted with
        
        Parameters:
        -----------
        items : List[int]
            List of item IDs user has interacted with
        n_items : int, default=10
            Number of recommendations
            
        Returns:
        --------
        recommendations : List[Tuple[int, float]]
            List of (item_id, confidence) tuples
        """
        if self.rules is None or len(self.rules) == 0:
            return []
        
        item_set = set(items)
        candidate_items = {}
        
        # Find rules where antecedents are in user's items
        for _, rule in self.rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            
            # Check if antecedent is subset of user's items
            if antecedent.issubset(item_set):
                for rec_item in consequent:
                    if rec_item not in item_set:
                        # Weight by confidence and lift
                        score = rule['confidence'] * (1 + np.log(rule['lift'] + 1) / 10)
                        
                        if rec_item not in candidate_items:
                            candidate_items[rec_item] = score
                        else:
                            # Average score if multiple rules recommend same item
                            candidate_items[rec_item] = max(candidate_items[rec_item], score)
        
        # Sort and return top-N recommendations
        recommendations = sorted(
            candidate_items.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_items]
        
        return recommendations
    
    def get_frequently_bought_together(self, item_id: int, 
                                       n_items: int = 10) -> List[Tuple[int, float]]:
        """
        Find items frequently bought together with the given item
        
        Parameters:
        -----------
        item_id : int
            Item ID
        n_items : int, default=10
            Number of similar items
            
        Returns:
        --------
        related_items : List[Tuple[int, float]]
            List of (item_id, support) tuples
        """
        if self.rules is None or len(self.rules) == 0:
            return []
        
        related_items = {}
        
        # Find rules with item_id in antecedent or consequent
        for _, rule in self.rules.iterrows():
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            
            if item_id in antecedent:
                # Get consequents
                for related in consequent:
                    if related != item_id:
                        score = rule['confidence']
                        related_items[related] = max(related_items.get(related, 0), score)
            
            elif item_id in consequent:
                # Get antecedents
                for related in antecedent:
                    if related != item_id:
                        score = rule['confidence']
                        related_items[related] = max(related_items.get(related, 0), score)
        
        # Sort and return top-N
        recommendations = sorted(
            related_items.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_items]
        
        return recommendations
    
    def get_rules_summary(self) -> Dict:
        """
        Get summary statistics about discovered rules
        
        Returns:
        --------
        summary : Dict
            Dictionary containing rule statistics
        """
        if self.rules is None or len(self.rules) == 0:
            return {
                'n_rules': 0,
                'n_frequent_itemsets': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
                'avg_support': 0,
                'avg_confidence': 0,
                'avg_lift': 0
            }
        
        return {
            'n_rules': len(self.rules),
            'n_frequent_itemsets': len(self.frequent_itemsets) if self.frequent_itemsets is not None and hasattr(self.frequent_itemsets, '__len__') else 0,
            'avg_support': float(self.rules['support'].mean()),
            'avg_confidence': float(self.rules['confidence'].mean()),
            'avg_lift': float(self.rules['lift'].mean()),
            'max_lift': float(self.rules['lift'].max()),
            'min_lift': float(self.rules['lift'].min())
        }
