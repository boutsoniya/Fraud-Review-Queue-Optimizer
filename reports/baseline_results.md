# First LightGBM Baseline and Identity Comparison — IEEE-CIS

## Experiment setup

- Dataset: IEEE-CIS Fraud Detection training transactions.
- Full transaction rows: 590,540.
- Chronological partitions: 413,378 train / 88,581 validation / 88,581 test.
- Test fraud prevalence: 3.4804% (3,083 frauds in 88,581 rows).
- Relative time: `floor(TransactionDT / 86400)`; these are elapsed-time buckets, not calendar days.
- Review scenario: 100 reviews per relative day bucket, an illustrative capacity assumption.

The test partition is the latest time period and is kept separate from training and validation. The identity table contains 144,233 records; the audit found unique transaction keys and all identity keys matched a transaction. Identity information is absent for most transactions, so the join leaves missing identity fields for those rows.

## Transaction-only baseline

The first model uses 60 selected transaction features: transaction amount/time, card, address, email, C1–C10, D1–D10, M1–M5, and V1–V20. It does not use the identity file.

| Metric | Result |
|---|---:|
| Test PR-AUC baseline (fraud prevalence) | 0.0348 |
| LightGBM PR-AUC | **0.4348** |
| LightGBM ROC-AUC | **0.8750** |
| Precision at probability threshold 0.50 | 0.7482 |
| Recall at probability threshold 0.50 | 0.2738 |
| F1 at probability threshold 0.50 | 0.4009 |

This run used 220 boosting rounds and reached the configured round cap. It is not a tuned optimum. PR-AUC is more informative than accuracy for this imbalanced dataset. The model's PR-AUC is about 12.5 times the test-prevalence baseline, but this does not establish calibrated probabilities or business value.

## Matched identity-feature comparison

To isolate the effect of identity features, both models below used the same 160-round training setup, the same chronological partitions, and the same test rows. The transaction-only version used 60 features. The identity-enriched version used 94 features, including 34 selected identity fields (23 numeric and 11 low-cardinality categorical fields). Very high-cardinality fields such as `DeviceInfo` and sparse fields such as `id_23`/`id_27` were excluded from this first comparison.

| Metric | Transaction-only | + selected identity |
|---|---:|---:|
| PR-AUC | 0.4296 | **0.4386** |
| ROC-AUC | 0.8698 | **0.8722** |
| Precision at 0.50 | **0.7612** | 0.7474 |
| Recall at 0.50 | 0.2760 | **0.2812** |
| F1 at 0.50 | 0.4051 | **0.4087** |
| Features | 60 | 94 |

The identity-enriched model improved PR-AUC by about 0.0090 absolute on this test period. This is a modest measured lift, not proof that every identity field is useful or that the improvement will generalize to another dataset.

Both 160-round runs reached the configured round cap; tree count and hyperparameters still need tuning.

## Capacity-only queue experiment

The test period contains 31 relative day buckets. At 100 reviews per relative day, each strategy selects 3,100 transactions. This experiment ranks all test transactions and selects the top 100 each day; it does **not** yet apply approve/decline thresholds.

### Transaction-only model

| Queue ranking | Fraud found | Review precision | Fraud-count recall | Fraud amount captured | Fraud-amount recall |
|---|---:|---:|---:|---:|---:|
| Highest fraud score | 1,327 | 42.81% | 43.04% | 175,188.59 | 37.31% |
| Probability × amount heuristic | 783 | 25.26% | 25.40% | **265,284.22** | **56.49%** |

### Identity-enriched model

| Queue ranking | Fraud found | Review precision | Fraud-count recall | Fraud amount captured | Fraud-amount recall |
|---|---:|---:|---:|---:|---:|
| Highest fraud score | **1,348** | **43.48%** | **43.72%** | 172,045.38 | 36.64% |
| Probability × amount heuristic | 789 | 25.45% | 25.59% | 263,475.50 | 56.11% |

Fraud amount captured is the sum of `TransactionAmt` for fraudulent transactions selected into the queue. It is a benchmark exposure measure, not verified recoverable loss or a merchant's actual financial loss. Amounts are reported in dataset units.

### Interpretation

- If the goal is to find the **largest number of fraudulent transactions** in a fixed queue, score-only ranking performed better.
- If the goal is to prioritize **larger fraudulent transaction amounts**, probability × amount captured a larger share of total fraud transaction amount in this test.
- Identity enrichment slightly improved PR-AUC and fraud-count capture under score ranking, but the transaction-only model captured a slightly larger fraud amount in that same queue. The difference is small and should not be over-interpreted.
- The project should retain both queue strategies until the business objective and cost assumptions are explicit. Probability × amount is still a heuristic, not calibrated recoverable loss.

## Reproduction

Place the Kaggle CSVs in `data/raw/`, install `requirements.txt`, then run:

```bash
# Transaction-only baseline
python scripts/train_baseline.py

# Same model workflow with selected identity fields
python scripts/train_baseline.py --identity data/raw/train_identity.csv
```

The script writes a git-ignored model artifact under `models/` and metrics under `data/processed/model/`. Do not commit raw Kaggle data or model artifacts.

## Next experiments

1. Increase the boosting-round cap and tune only on validation data.
2. Compare more identity fields, including carefully encoded high-cardinality fields, against the compact identity subset.
3. Inspect probability calibration before treating scores as calibrated probabilities.
4. Select approve/review/decline thresholds using validation data and daily capacity.
5. Compare the full assumption-based cost function, fraud count, fraud amount, false declines, and review costs.
6. Evaluate the final chosen policy once on the untouched chronological test period.
