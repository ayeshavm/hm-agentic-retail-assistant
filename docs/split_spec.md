## Step 1 - Train/Test Split

- Split is time based using last_seen
- Training data includes interactions before YYYY-MM-DD
- Test data includes interactions on or after YYYY-MM-DD
- Evaluation includes only users present in training
- Metrics computed at K = 12

## Step 2 — Popularity Baseline

**Purpose**  
Establish a simple, global baseline to anchor evaluation results.

**Method**
- Use training data only
- Count the number of interactions per `article_id`
- Rank items by interaction frequency
- Recommend the top K items (K = 12) to all users

**Rationale**
- Popularity is a strong and commonly used baseline in recommender systems
- It provides a lower bound for acceptable model performance
- If more complex models cannot outperform this baseline, further investigation is required

---

## Step 3 — Co-occurrence Baseline

**Purpose**  
Capture simple relational structure between items without training a model.

**Method**
- Use training interactions only
- Compute item–item co-occurrence based on shared user interactions
- For a given user, aggregate co-occurrence scores from items the user has interacted with
- Rank candidate items by aggregated co-occurrence strength
- Recommend the top K items (K = 12)

**Rationale**
- Co-occurrence baselines are simple but often competitive
- They capture local structure and item similarity
- This baseline provides a meaningful comparison point for matrix factorization models

---

## Step 4 — Evaluation Metrics

**Metrics Used**
- **Recall@12**
  - Measures whether relevant items appear in the top 12 recommendations
- **MAP@12 (Mean Average Precision)**
  - Measures ranking quality by rewarding correct items appearing earlier in the list

**Evaluation Protocol**
- Evaluate only users present in the training set
- Use held-out test interactions as ground truth
- Compute metrics consistently across all models and baselines

**Output**
A metrics summary table of the form:

| Model          | Recall@12 | MAP@12 |
|----------------|-----------|--------|
| Popularity     |           |        |
| Co-occurrence  |           |        |

---

## Step 5 — Sanity Checks and Validation

**Checks Performed**
- Confirm that co-occurrence performance exceeds or matches the popularity baseline
- Verify that metrics are within reasonable ranges and not degenerate
- Ensure no training data leakage into evaluation

**Interpretation**
- Metrics are used to assess relative performance, not to optimize absolute values
- Results are validated for consistency and plausibility before introducing more complex models

**Outcome**
If baselines behave as expected and metrics are stable, the evaluation setup is considered reliable and ready for use with matrix factorization models.