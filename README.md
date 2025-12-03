# Hybrid Weighted Recommender System

A compact, production-ready hybrid recommender that combines collaborative filtering (NMF-based matrix factorization for implicit feedback) with association rule mining (Apriori). Designed for the RetailRocket eCommerce dataset and tuned for personalization-heavy recommendations (α = 0.7 by default).

Key goals:

- Combine personalization (CF) with co-purchase signals (ARM)
- Produce top-N recommendations with deduplication
- Provide evaluation utilities for ranking and diversity metrics

---

## Features

- Collaborative Filtering: Non-negative Matrix Factorization (implicit feedback)
  - Default latent factors: 50
- Association Rule Mining: mlxtend-based Apriori (min_support=0.01, min_confidence=0.3)
- Hybrid scoring: score = α _ CF_score + (1-α) _ ARM_score (α default 0.7)
- Evaluation: Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, Coverage, Diversity
- Notebook-driven reproducible workflow

---

## Repository Structure

```
src/
├── algorithms/
│   ├── collaborative_filtering.py
│   └── apriori_arm.py
├── utils/
│   ├── data_loader.py
│   ├── evaluation_metrics.py
│   └── helpers.py
└── hybrid_recommender.py
notebooks/
├── 00_project_setup.ipynb
├── 01_data_exploration.ipynb
├── 02_collaborative_filtering.ipynb
├── 03_apriori_arm.ipynb
└── 04_hybrid_model_evaluation.ipynb
data/
results/
```

---

## Installation

1. Clone the repo and create a virtual environment

```bash
git clone <repo_url>
cd HybridWeightedRecommenderSystem
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Download dataset (RetailRocket eCommerce)

- Kaggle page: https://www.kaggle.com/retailrocket/ecommerce-dataset
- Required files: `events.csv`, `item_properties_part1.csv`, `category_tree.csv`
- Place them under the `data/` directory

---

## Quickstart

Load data and train the hybrid recommender:

```python
from src.utils.data_loader import DataLoader
from src.hybrid_recommender import HybridRecommenderSystem

events = DataLoader.load_events('data/events.csv')
interactions = DataLoader.create_user_item_interactions(events)
transactions = DataLoader.create_transactions(events)
train, test = DataLoader.train_test_split(interactions)

recommender = HybridRecommenderSystem(alpha=0.7, n_factors=50)
recommender.fit(train, transactions)

# Get top-10 recommendations for user 100
recommendations = recommender.get_recommendations(user_id=100, n_items=10)
print(recommendations)
```

Model summary and diagnostics:

```python
print(recommender.get_model_summary())
```

---

## Command-line Usage

After activating your virtual environment and installing dependencies, run these commands from the project root.

Train the model (example flags):

```bash
python main.py --action train --alpha 0.7 --n_factors 50 --min_support 0.01
```

Get recommendations for a user:

```bash
python main.py --action recommend --user_id 100 --n_items 10
```

Evaluate the model on a test split:

```bash
python main.py --action evaluate --test_size 0.2
```

Show a saved model summary / diagnostics:

```bash
python main.py --action summary
```

Run the project notebooks:

```bash
jupyter notebook notebooks/
```

Adjust flags (e.g., `--alpha`, `--n_factors`, `--min_support`) as needed for experiments.

## Core API (examples)

- HybridRecommenderSystem

```python
recommender = HybridRecommenderSystem(alpha=0.7, n_factors=50)
recommender.fit(interactions_df, transactions_df)
recs = recommender.get_recommendations(user_id, n_items=10)
```

- CollaborativeFiltering

```python
cf = CollaborativeFiltering(n_factors=50)
cf.fit(interactions_df)
recs = cf.recommend(user_id, n_items=10)
```

- AprioriARM

```python
arm = AprioriARM(min_support=0.01, min_confidence=0.3)
arm.fit(transactions_df)
recs = arm.get_recommendations(user_items, n_items=10)
```

- DataLoader

```python
events = DataLoader.load_events('data/events.csv')
interactions = DataLoader.create_user_item_interactions(events)
transactions = DataLoader.create_transactions(events)
train, test = DataLoader.train_test_split(interactions)
```

- EvaluationMetrics

```python
from src.utils.evaluation_metrics import EvaluationMetrics
metrics = EvaluationMetrics.compute_metrics(recs, ground_truth_set, k=10)
```

---

## Default Hyperparameters

- alpha (hybrid weight): 0.7
- n_factors (NMF latent factors): 50
- Apriori min_support: 0.01
- Apriori min_confidence: 0.3

Hyperparameters can be tuned (`alpha` in [0,1], `n_factors` typically 20–100).

---

## Evaluation

Implemented metrics:

- Precision@K, Recall@K, F1@K
- NDCG@K, MAP@K
- Coverage (catalog coverage)
- Diversity (pairwise item dissimilarity)

Compute metrics on test splits produced by DataLoader.train_test_split.

---

## Running Notebooks

Open the notebooks in `notebooks/` to reproduce setup, EDA, model training, and evaluation:

```bash
jupyter notebook notebooks/00_project_setup.ipynb
```
