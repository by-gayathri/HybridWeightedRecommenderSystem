"""
Main script to demonstrate the Hybrid Recommender System

Usage:
    python main.py --action train --alpha 0.7
    python main.py --action recommend --user_id 123 --n_items 10
"""

import argparse
import pandas as pd
import pickle
import os
from src.hybrid_recommender import HybridRecommenderSystem
from src.utils.data_loader import DataLoader
from src.utils.evaluation_metrics import EvaluationMetrics


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Recommender System - Matrix Factorization + Apriori'
    )
    
    parser.add_argument('--action', type=str, default='train',
                       choices=['train', 'recommend', 'evaluate', 'summary'],
                       help='Action to perform')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing dataset files')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='./results/model.pkl',
                       help='Path to save/load model')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Weight for CF in hybrid approach')
    parser.add_argument('--n_factors', type=int, default=50,
                       help='Number of latent factors')
    parser.add_argument('--min_support', type=float, default=0.01,
                       help='Minimum support for Apriori')
    parser.add_argument('--min_confidence', type=float, default=0.3,
                       help='Minimum confidence for ARM')
    parser.add_argument('--user_id', type=str, default=None,
                       help='User ID for recommendations')
    parser.add_argument('--n_items', type=int, default=10,
                       help='Number of recommendations to return')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion for evaluation')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Routes
    if args.action == 'train':
        train_model(args)
    elif args.action == 'recommend':
        recommend(args)
    elif args.action == 'evaluate':
        evaluate_model(args)
    elif args.action == 'summary':
        show_summary(args)


def train_model(args):
    """Train the hybrid recommender system"""
    print("\n" + "="*60)
    print("TRAINING HYBRID RECOMMENDER SYSTEM")
    print("="*60)
    
    # Check if data exists
    users_file = os.path.join(args.data_dir, 'users.csv')
    products_file = os.path.join(args.data_dir, 'products.csv')
    transactions_file = os.path.join(args.data_dir, 'transactions.csv')
    
    for f in [users_file, products_file, transactions_file]:
        if not os.path.exists(f):
            print(f"✗ Error: {f} not found")
            return
    
    print(f"\n1. Loading data files...")
    users_df = DataLoader.load_users(users_file)
    products_df = DataLoader.load_products(products_file)
    transactions_df = DataLoader.load_transactions(transactions_file)
    print(f"   ✓ Loaded {len(users_df)} users")
    print(f"   ✓ Loaded {len(products_df)} products")
    print(f"   ✓ Loaded {len(transactions_df)} transactions")
    
    print(f"\n2. Creating user-item interactions...")
    interactions_df = DataLoader.create_user_item_interactions(transactions_df)
    print(f"   ✓ Created {len(interactions_df)} interactions")
    print(f"   Users: {interactions_df['user_id'].nunique()}")
    print(f"   Items: {interactions_df['item_id'].nunique()}")
    
    print(f"\n3. Creating transaction baskets for Apriori...")
    baskets_df = DataLoader.create_transaction_baskets(transactions_df)
    print(f"   ✓ Created {len(baskets_df)} baskets")
    
    print(f"\n4. Initializing Hybrid Recommender System...")
    print(f"   α (alpha): {args.alpha}")
    print(f"   Latent factors: {args.n_factors}")
    print(f"   Min support (Apriori): {args.min_support}")
    print(f"   Min confidence (ARM): {args.min_confidence}")
    
    recommender = HybridRecommenderSystem(
        alpha=args.alpha,
        n_factors=args.n_factors,
        min_support=args.min_support,
        min_confidence=args.min_confidence
    )
    
    print(f"\n5. Training models...")
    recommender.fit(interactions_df, baskets_df)
    print(f"   ✓ Training complete")
    
    # Show summary
    summary = recommender.get_model_summary()
    print(f"\n6. Model Summary:")
    print(f"   CF Users: {summary['cf_users']}")
    print(f"   CF Items: {summary['cf_items']}")
    print(f"   ARM Rules: {summary['arm_rules']}")
    print(f"   ARM Itemsets: {summary['arm_itemsets']}")
    
    # Save model
    print(f"\n7. Saving model to: {args.model_path}")
    recommender.save_model(args.model_path)
    print(f"   ✓ Model saved")
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60 + "\n")


def recommend(args):
    """Get recommendations for a user"""
    if not os.path.exists(args.model_path):
        print(f"✗ Error: Model not found at {args.model_path}")
        print("Please train the model first using: python main.py --action train")
        return
    
    if args.user_id is None:
        print("✗ Error: Please specify --user_id")
        return
    
    print("\n" + "="*60)
    print(f"GENERATING RECOMMENDATIONS FOR USER {args.user_id}")
    print("="*60)
    
    # Load model
    print(f"\n1. Loading model from: {args.model_path}")
    recommender = HybridRecommenderSystem.load_model(args.model_path)
    print(f"   ✓ Model loaded")
    
    # Get recommendations
    print(f"\n2. Generating {args.n_items} recommendations...")
    recommendations = recommender.get_recommendations(args.user_id, args.n_items)
    
    if not recommendations:
        print(f"   No recommendations available for user {args.user_id}")
    else:
        print(f"   ✓ Generated {len(recommendations)} recommendations:\n")
        print(f"   {'Rank':<6} {'Item ID':<12} {'Score':<10}")
        print(f"   {'-'*28}")
        for rank, (item_id, score) in enumerate(recommendations, 1):
            print(f"   {rank:<6} {item_id:<12} {score:.4f}")
    
    print("\n" + "="*60 + "\n")


def evaluate_model(args):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("EVALUATING MODEL PERFORMANCE")
    print("="*60)
    
    # Load data
    events_file = os.path.join(args.data_dir, 'events.csv')
    if not os.path.exists(events_file):
        print(f"✗ Error: {events_file} not found")
        return
    
    if not os.path.exists(args.model_path):
        print(f"✗ Error: Model not found at {args.model_path}")
        return
    
    print(f"\n1. Loading data from: {events_file}")
    events_df = DataLoader.load_events(events_file)
    interactions_df = DataLoader.create_user_item_interactions(events_df)
    
    print(f"\n2. Splitting data (test size: {args.test_size})")
    train_df, test_df = DataLoader.train_test_split(interactions_df, args.test_size)
    
    print(f"\n3. Loading model from: {args.model_path}")
    recommender = HybridRecommenderSystem.load_model(args.model_path)
    
    print(f"\n4. Evaluating on {len(test_df['user_id'].unique())} test users...")
    
    # Calculate metrics for sample users
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    sample_users = test_df['user_id'].unique()[:100]  # Sample first 100 users
    
    for user_id in sample_users:
        user_test_items = set(test_df[test_df['user_id'] == user_id]['item_id'])
        
        if len(user_test_items) == 0:
            continue
        
        recommendations = recommender.get_recommendations(user_id, args.n_items)
        # Extract item IDs from (item_id, score) tuples
        rec_items = [item_id for item_id, _ in recommendations]
        
        metrics = EvaluationMetrics.compute_metrics(rec_items, user_test_items, args.n_items)
        precision_scores.append(metrics['precision@k'])
        recall_scores.append(metrics['recall@k'])
        f1_scores.append(metrics['f1@k'])
    
    # Print results
    print(f"\n5. Results (on {len(precision_scores)} users):")
    print(f"   Precision@{args.n_items}: {sum(precision_scores)/len(precision_scores):.4f}")
    print(f"   Recall@{args.n_items}: {sum(recall_scores)/len(recall_scores):.4f}")
    print(f"   F1-Score@{args.n_items}: {sum(f1_scores)/len(f1_scores):.4f}")
    
    print("\n" + "="*60 + "\n")


def show_summary(args):
    """Show model summary"""
    if not os.path.exists(args.model_path):
        print(f"✗ Error: Model not found at {args.model_path}")
        return
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60 + "\n")
    
    recommender = HybridRecommenderSystem.load_model(args.model_path)
    summary = recommender.get_model_summary()
    
    print(f"Status: {summary['status']}")
    print(f"Alpha (CF weight): {summary['alpha']}")
    print(f"Latent Factors: {summary['n_factors']}")
    print(f"\nCollaborative Filtering:")
    print(f"  Users: {summary['cf_users']}")
    print(f"  Items: {summary['cf_items']}")
    print(f"\nAssociation Rule Mining:")
    print(f"  Rules: {summary['arm_rules']}")
    print(f"  Itemsets: {summary['arm_itemsets']}")
    
    if summary['arm_summary']['n_rules'] > 0:
        arm_stats = summary['arm_summary']
        print(f"\nARM Statistics:")
        print(f"  Avg Support: {arm_stats['avg_support']:.4f}")
        print(f"  Avg Confidence: {arm_stats['avg_confidence']:.4f}")
        print(f"  Avg Lift: {arm_stats['avg_lift']:.4f}")
        print(f"  Max Lift: {arm_stats['max_lift']:.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
