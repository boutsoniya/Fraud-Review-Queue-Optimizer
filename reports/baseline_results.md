# First LightGBM Baseline — IEEE-CIS

## Experiment scope

- Dataset: IEEE-CIS Fraud Detection training transactions.
- Rows: 590,540.
- Chronological partitions: 413,378 train / 88,581 validation / 88,581 test.
- Test fraud prevalence: 3.4804% (3,083 frauds in 88,581 rows).
- Features: 60 selected transaction fields.
- Identity file: **not used in this first model**.
- Model: LightGBM binary classifier, 220 boosting rounds, 60 features.
- Validation metric used for training control: ROC-AUC.
- Final metrics below are from the untouched later-time test partition.

The 220-round limit was reached, so 220 is not established as an optimal number of trees. Increase the cap and review validation behavior in a later experiment.

## Classification metrics

| Metric | Result |
|---|---:|
| Dummy PR-AUC (test fraud prevalence) | 0.0348 |
| LightGBM PR-AUC | **0.4348** |
| LightGBM ROC-AUC | **0.8750** |
| Precision at probability threshold 0.50 | 0.7482 |
| Recall at probability threshold 0.50 | 0.2738 |
| F1 at probability threshold 0.50 | 0.4009 |

PR-AUC is the more informative headline than accuracy for this imbalanced dataset. The model's PR-AUC is about 12.5 times the test-set prevalence baseline, but this does not establish calibrated probabilities or business value.

## Capacity-only queue experiment

The test period contains 31 relative day buckets. With the illustrative capacity of 100 reviews per relative day, each strategy selects 3,100 transactions. This experiment ranks all transactions and selects the top 100 each day; it does **not** yet apply the policy's approve/decline thresholds.

| Queue ranking | Fraud found | Review precision | Recall of test fraud | Average transaction amount |
|---|---:|---:|---:|---:|
| Highest fraud score | **1,331** | **42.94%** | **43.17%** | 175.37 |
| Probability × amount heuristic | 786 | 25.35% | 25.49% | 669.73 |

### Interpretation

For the objective of finding the largest number of fraudulent transactions within 100 review slots per relative day, score-only ranking performed better in this first test. The amount-weighted heuristic selected transactions with a much higher average amount, but found fewer fraudulent transactions.

This does **not** prove score-only ranking minimizes monetary fraud loss. To decide that, the next experiment must compare the actual fraud amount captured and the complete assumption-based cost function. Probability × amount remains a heuristic, not calibrated recoverable loss.

## Reproduction

Place the Kaggle transaction CSV at `data/raw/train_transaction.csv`, install `requirements.txt`, then run:

```bash
python scripts/train_baseline.py
```

The script writes a git-ignored model artifact under `models/` and metrics under `data/processed/model/`. Do not commit raw Kaggle data or model artifacts.

## Next experiments

1. Compare this transaction-only baseline with identity-enriched features.
2. Expand beyond the first 60 selected features and compare validation PR-AUC.
3. Inspect calibration before treating scores as probabilities.
4. Choose thresholds using validation data and the review-capacity constraint.
5. Compare score and amount-weighted queues on fraud count, fraud amount, false declines, and the transparent cost assumptions.
6. Evaluate the final chosen policy once on the untouched chronological test set.
