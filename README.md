# Active Learning for Employee Attrition Prediction

An active learning strategy for predicting employee attrition (Left / Stayed) under a fixed labeling budget. The target metric is F1 on the minority "Left" class.

## Problem

Labels are expensive: an employee's outcome is only revealed when that employee is queried from an oracle, and the total number of queries is capped at 5,000 unique employees. Each seed starts with a free labeled set of 500 employees. The question is which employees to query so that the final classifier predicts "Left" well.

## Approach

- **Minority-biased querying.** Each round, a scoring random forest (40 trees) is fitted on the current labeled set. The unlabeled pool is ranked by predicted P(Left), and the top candidates are queried. For F1 on the Left class, the most useful labels are the true Left cases, so the strategy looks for them directly.
- **Adaptive batching.** Labels are acquired in batches of 250, which gives 20 rounds for the 5,000 budget. The scoring model is refreshed after every batch.
- **Candidate subsampling.** Each round scores a random subset of 6,000 unlabeled employees. This bounds the cost per round and keeps query batches diverse.
- **Minority oversampling.** Before the final fit, Left rows are duplicated (factor 2), giving the final random forest more weight on the minority class.

## Results

F1 on the Left class over three seeds (1, 2, 3), on each seed's held-out test set:

| Seed | F1 (Left) | Runtime |
|------|-----------|---------|
| 1    | 0.6341    | ~10 s   |
| 2    | 0.6338    | ~10 s   |
| 3    | 0.6458    | ~10 s   |
| **Mean** | **0.6379** | |

Each seed must finish within 60 seconds.

## Repository contents

- `strategy.py`: the active learning strategy. This is the main file of the project.
- `run.py`: entry point. Calls `run_active_learning(seed)` and returns the trained model.
- `evaluation.py`: runs the three seeds, enforces the runtime limit, checks the allowed imports, and reports F1 (Left).
- `utils.py`: data loading, the labeling oracle (returns labels only for queried IDs and enforces the budget), and the training and evaluation helpers.
- `constants.yaml`: budget, seeds, runtime limit, random forest settings, and data file paths.

## Reproducing

The data files are not included in this repository. Place a `data/` folder next to the code with the files named in `constants.yaml`, then run:

```
python evaluation.py
```

Requirements: Python 3, numpy, pandas, scikit-learn, scipy, PyYAML, joblib.
