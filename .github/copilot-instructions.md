<!-- Hybrid Recommender System - Project-Specific Instructions -->

## Project Overview

Project implementing a hybrid recommender system combining:

- **Collaborative Filtering** (Matrix Factorization)
- **Association Rule Mining** (Apriori Algorithm)
- **Dataset**: RetailRocket eCommerce data from Kaggle
- **Hybrid Parameter**: α = 0.7 (personalization-heavy)

## Project Structure

```
src/
├── algorithms/           # Core algorithms
│   ├── collaborative_filtering.py    # CF using NMF
│   └── apriori_arm.py               # Association Rule Mining
├── utils/               # Utilities
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── evaluation_metrics.py  # Evaluation metrics
│   └── helpers.py       # Helper functions
└── hybrid_recommender.py    # Main hybrid system

notebooks/
├── 00_project_setup.ipynb           # Environment setup
├── 01_data_exploration.ipynb        # EDA
├── 02_collaborative_filtering.ipynb # CF model
├── 03_apriori_arm.ipynb            # ARM model
└── 04_hybrid_model_evaluation.ipynb # Evaluation

data/                   # RetailRocket dataset
results/               # Output results
```

## Development Guidelines

### Code Style

- Follow PEP 8 conventions
- Use type hints for all functions
- Include comprehensive docstrings
- Add inline comments for complex logic

### Algorithms

- **Collaborative Filtering**: Uses Non-negative Matrix Factorization (NMF)
  - Default: 50 latent factors
  - Handles implicit feedback
- **Apriori ARM**: Uses mlxtend library
  - Min support: 0.01
  - Min confidence: 0.3
  - Generates frequent itemsets and rules

### Hybrid Approach

- Combines CF and ARM scores: `score = α × CF_score + (1-α) × ARM_score`
- Default α = 0.7 (personalisation-heavy)
- Deduplicates recommendations
- Returns top-N items

### Evaluation Metrics

- Precision@K, Recall@K, F1@K
- NDCG@K (ranking quality)
- MAP@K (mean average precision)
- Coverage (catalog diversity)
- Diversity (item dissimilarity)

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

- Visit: https://www.kaggle.com/retailrocket/ecommerce-dataset
- Download: events.csv, item_properties_part1.csv, category_tree.csv
- Place in: `data/` directory

### 3. Run Setup Notebook

Open `notebooks/00_project_setup.ipynb` and run all cells to verify setup.

## Key Classes & Methods

### HybridRecommenderSystem

```python
recommender = HybridRecommenderSystem(alpha=0.7, n_factors=50)
recommender.fit(interactions_df, transactions_df)
recommendations = recommender.get_recommendations(user_id, n_items=10)
```

### CollaborativeFiltering

```python
cf = CollaborativeFiltering(n_factors=50)
cf.fit(interactions_df)
recs = cf.recommend(user_id, n_items=10)
similar = cf.get_similar_items(item_id)
```

### AprioriARM

```python
arm = AprioriARM(min_support=0.01)
arm.fit(transactions_df)
recs = arm.get_recommendations(user_items, n_items=10)
together = arm.get_frequently_bought_together(item_id)
```

### DataLoader

```python
events_df = DataLoader.load_events('data/events.csv')
interactions_df = DataLoader.create_user_item_interactions(events_df)
transactions_df = DataLoader.create_transactions(events_df)
train, test = DataLoader.train_test_split(interactions_df)
```

### EvaluationMetrics

```python
precision = EvaluationMetrics.precision_at_k(recs, ground_truth, k=10)
recall = EvaluationMetrics.recall_at_k(recs, ground_truth, k=10)
ndcg = EvaluationMetrics.ndcg_at_k(recs_with_scores, ground_truth, k=10)
```

## Common Tasks

### Load and Prepare Data

```python
from src.utils.data_loader import DataLoader

events = DataLoader.load_events('data/events.csv')
interactions = DataLoader.create_user_item_interactions(events)
transactions = DataLoader.create_transactions(events)
train, test = DataLoader.train_test_split(interactions)
```

### Train Hybrid Model

```python
from src.hybrid_recommender import HybridRecommenderSystem

recommender = HybridRecommenderSystem(alpha=0.7)
recommender.fit(train, transactions)

# Get recommendations
recs = recommender.get_recommendations(user_id=100, n_items=10)
print(recs)

# Model summary
print(recommender.get_model_summary())
```

### Evaluate Model

```python
from src.utils.evaluation_metrics import EvaluationMetrics

user_test = test[test['user_id'] == 100]['item_id'].tolist()
recs = [item_id for item_id, _ in recommender.get_recommendations(100)]

metrics = EvaluationMetrics.compute_metrics(recs, set(user_test), k=10)
print(metrics)
```

## Hyperparameter Tuning

- **α (alpha)**: 0.0-1.0 (default: 0.7)
- **n_factors**: 20-100 (default: 50)
- **min_support**: 0.001-0.1 (default: 0.01)
- **min_confidence**: 0.1-0.5 (default: 0.3)

## Performance Notes

- Data sparsity: ~99.9% (typical for implicit feedback)
- Training time: Depends on dataset size and factors
- Recommendation time: <100ms per user (CF) + ARM lookup
- Memory: Proportional to n_users × n_items

## References

- Koren et al. (2009): Matrix Factorization Techniques
- Agrawal & Srikant (1994): Fast Algorithms for Mining Association Rules
- RetailRocket: http://www.retailrocket.net/
- Dataset: https://www.kaggle.com/retailrocket/ecommerce-dataset
