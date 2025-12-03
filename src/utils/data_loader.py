"""
Data Loading and Preprocessing Utilities

Handles loading and preprocessing of users, products, and transactions data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


class DataLoader:
    """
    Load and preprocess simple e-commerce dataset
    
    Handles loading of users, products, and transactions tables
    """
    
    @staticmethod
    def load_users(filepath: str) -> pd.DataFrame:
        """
        Load users data
        
        Parameters:
        -----------
        filepath : str
            Path to users.csv file
            
        Returns:
        --------
        users_df : pd.DataFrame
            DataFrame with columns: user_id, user_name
        """
        users_df = pd.read_csv(filepath)
        return users_df
    
    @staticmethod
    def load_products(filepath: str) -> pd.DataFrame:
        """
        Load products data
        
        Parameters:
        -----------
        filepath : str
            Path to products.csv file
            
        Returns:
        --------
        products_df : pd.DataFrame
            DataFrame with columns: product_id, product_name, category
        """
        products_df = pd.read_csv(filepath)
        return products_df
    
    @staticmethod
    def load_transactions(filepath: str) -> pd.DataFrame:
        """
        Load transactions data
        
        Parameters:
        -----------
        filepath : str
            Path to transactions.csv file
            
        Returns:
        --------
        transactions_df : pd.DataFrame
            DataFrame with columns: order_id, user_id, product_id
        """
        transactions_df = pd.read_csv(filepath)
        return transactions_df
    
    @staticmethod
    def create_user_item_interactions(transactions_df: pd.DataFrame, 
                                     implicit_feedback: bool = True) -> pd.DataFrame:
        """
        Create user-item interactions from transactions
        
        Parameters:
        -----------
        transactions_df : pd.DataFrame
            Transaction data with user_id and product_id columns
        implicit_feedback : bool, default=True
            If True, rating = 1 (implicit feedback)
            If False, rating = purchase count
            
        Returns:
        --------
        interactions_df : pd.DataFrame
            DataFrame with columns: user_id, item_id, rating
        """
        if implicit_feedback:
            # Create implicit feedback: each purchase = 1
            interactions_df = transactions_df[['user_id', 'product_id']].drop_duplicates()
            interactions_df['rating'] = 1
        else:
            # Count purchases per user-product pair
            interactions_df = transactions_df.groupby(['user_id', 'product_id']).size().reset_index(name='rating')
        
        # Rename columns for consistency
        interactions_df.columns = ['user_id', 'item_id', 'rating']
        
        return interactions_df.reset_index(drop=True)
    
    @staticmethod
    def create_transaction_baskets(transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction baskets for Apriori algorithm
        
        Parameters:
        -----------
        transactions_df : pd.DataFrame
            Transaction data with order_id and product_id columns
            
        Returns:
        --------
        baskets_df : pd.DataFrame
            DataFrame grouped by order_id with list of product_ids
        """
        # Group products by order
        baskets_df = transactions_df.groupby('order_id')['product_id'].apply(list).reset_index()
        baskets_df.columns = ['basket_id', 'items']
        
        return baskets_df
    
    @staticmethod
    def train_test_split(interactions_df: pd.DataFrame, 
                        test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into train and test sets
        
        Uses user-level splitting to ensure test users are in training set
        
        Parameters:
        -----------
        interactions_df : pd.DataFrame
            User-item interactions
        test_size : float, default=0.2
            Proportion of data in test set
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        train_df : pd.DataFrame
            Training set
        test_df : pd.DataFrame
            Test set
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get unique users
        unique_users = interactions_df['user_id'].unique()
        n_test_users = max(1, int(len(unique_users) * test_size))
        
        # Randomly select test users
        test_users = np.random.choice(unique_users, size=n_test_users, replace=False)
        
        # Split by users (all interactions of test users go to test set)
        test_mask = interactions_df['user_id'].isin(test_users)
        train_df = interactions_df[~test_mask].reset_index(drop=True)
        test_df = interactions_df[test_mask].reset_index(drop=True)
        
        return train_df, test_df
    
    @staticmethod
    def get_statistics(transactions_df: pd.DataFrame, 
                      interactions_df: pd.DataFrame) -> Dict:
        """
        Calculate dataset statistics
        
        Parameters:
        -----------
        transactions_df : pd.DataFrame
            Transaction data
        interactions_df : pd.DataFrame
            User-item interactions
            
        Returns:
        --------
        stats : dict
            Dictionary with dataset statistics
        """
        n_users = interactions_df['user_id'].nunique()
        n_items = interactions_df['item_id'].nunique()
        n_interactions = len(interactions_df)
        n_transactions = transactions_df.groupby('order_id').ngroups if 'order_id' in transactions_df.columns else len(transactions_df)
        
        sparsity = 1 - (n_interactions / (n_users * n_items)) if n_users * n_items > 0 else 1.0
        
        stats = {
            'n_users': n_users,
            'n_items': n_items,
            'n_interactions': n_interactions,
            'n_transactions': n_transactions,
            'avg_interactions_per_user': n_interactions / n_users if n_users > 0 else 0,
            'avg_interactions_per_item': n_interactions / n_items if n_items > 0 else 0,
            'sparsity': sparsity,
            'density': 1 - sparsity
        }
        
        return stats
