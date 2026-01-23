# Day 3 Results

## Baselines
- Popularity baseline computed from training interactions
- Co-occurrence baseline computed from item–item co-interactions using top-N seeds per user

## Metrics
K = 12  
Users evaluated: see table

```text
Model        Users   Recall@12    MAP@12
popularity   12582   0.01088    0.00392
cooccurrence 12992   0.02104    0.00839