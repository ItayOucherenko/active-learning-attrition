"""
Student implementation file — submit this file only.

Active Learning strategy for predicting employee attrition (Left / Stayed).

Approach:
  * Selection — minority-biased querying: the target metric is F1 on the minority
    "Left" class, so the most valuable labels to acquire are true "Left" examples.
    Each iteration, a scoring RandomForest is fitted on the current labeled set,
    the unlabeled pool is ranked by P(Left), and the top candidates are sent to
    the oracle.
  * Adaptive batching: the oracle budget is spent in small batches, so the
    scoring model is refreshed frequently and each query round is based on
    up-to-date information.
  * Training-set composition: the minority rows are duplicated, giving the fixed RandomForest more weight on
    the minority class and improving Left-recall, and hence F1.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils import (
    ID_COLUMN,
    MAX_LABELED,
    TARGET_COLUMN,
    call_oracle,
    load_initial_labeled,
    load_pool,
    prepare_xy,
    train_model,
)

# --- configuration ---
BATCH_SIZE = 250                        # oracle queries per Active Learning iteration
BUDGET = MAX_LABELED                    # oracle budget, read from constants.yaml
N_ITERS = -(-BUDGET // BATCH_SIZE)      # ceil(BUDGET / BATCH_SIZE) -> use the full budget
SCORING_TREES = 40                      # small RF used only to rank candidates
CAND_SUBSET = 6000                      # random candidate subset scored each iteration
OVERSAMPLE_FACTOR = 2                   # times the minority ("Left") rows appear in training


def _encode_features(df: pd.DataFrame):
    """Encode features exactly like the framework (via prepare_xy), ignoring labels."""
    work = df.copy()
    if TARGET_COLUMN not in work.columns:
        work[TARGET_COLUMN] = 0  # dummy target so prepare_xy can run on the pool
    X, _, ids = prepare_xy(work)
    return np.asarray(X), [str(i) for i in ids]


def _proba_left(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """P(y=1); robust to a scoring model that happened to see only one class."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X))


def _oversample_minority(df: pd.DataFrame, factor: float, seed: int) -> pd.DataFrame:
    """Duplicate Left rows so the minority appears `factor` times (1 = unchanged).

    Supports fractional factors: the fractional part duplicates a deterministic
     random subset of the minority rows.
    """
    if factor <= 1:
        return df
    minority = df[df[TARGET_COLUMN] == 1]
    if len(minority) == 0:
        return df
    extra_full = int(np.floor(factor)) - 1
    frac = factor - np.floor(factor)
    parts = [df] + [minority] * extra_full
    if frac > 1e-9:
        k = int(round(frac * len(minority)))
        if k > 0:
            parts.append(minority.sample(n=k, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_active_learning(seed: int):
    """Run active learning for the given seed and return a trained RandomForestClassifier."""
    rng = np.random.default_rng(seed)

    # 1. Start from the free initial labeled set.
    labeled = load_initial_labeled(seed).copy()
    labeled[ID_COLUMN] = labeled[ID_COLUMN].astype(str)

    # Encode the whole pool once; index rows by Employee ID for fast lookup.
    pool = load_pool().copy()
    pool[ID_COLUMN] = pool[ID_COLUMN].astype(str)
    X_pool, pool_ids = _encode_features(pool)
    row_of = {sid: i for i, sid in enumerate(pool_ids)}

    all_ids = set(pool_ids)
    labeled_ids = set(labeled[ID_COLUMN])

    # 2. Iteratively query the samples most likely to be "Left", then retrain.
    n_queried = 0
    for _ in range(N_ITERS):
        remaining = BUDGET - n_queried
        if remaining <= 0:
            break
        k = min(BATCH_SIZE, remaining)

        # Candidate pool = unlabeled. Sorted for reproducibility, then a random
        # subset is scored (bounds cost and keeps the query batch diverse).
        candidates = sorted(all_ids - labeled_ids)
        if not candidates:
            break
        if len(candidates) > CAND_SUBSET:
            idx = rng.choice(len(candidates), CAND_SUBSET, replace=False)
            candidates = [candidates[i] for i in idx]

        # Cheap scoring RF on the current labeled set .
        X_tr, y_tr, _ = prepare_xy(labeled)
        scorer = RandomForestClassifier(
            n_estimators=SCORING_TREES, random_state=seed, n_jobs=-1
        )
        scorer.fit(np.asarray(X_tr), y_tr)

        # Minority-biased selection: highest P(Left) first.
        cand_rows = X_pool[[row_of[c] for c in candidates]]
        scores = _proba_left(scorer, cand_rows)
        chosen = [candidates[i] for i in np.argsort(scores)[-k:]]

        new_rows = call_oracle(chosen)
        new_rows[ID_COLUMN] = new_rows[ID_COLUMN].astype(str)
        labeled = pd.concat([labeled, new_rows], ignore_index=True)
        labeled_ids |= set(chosen)
        n_queried += len(chosen)

    # 3. Final model: oversample the minority, then train the official frozen RF.
    train_df = _oversample_minority(labeled, OVERSAMPLE_FACTOR, seed)
    X, y, employee_ids = prepare_xy(train_df)
    return train_model(X, y, employee_ids, seed=seed)
