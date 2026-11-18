# Decision-policy experiment

## Method

The identity-enriched LightGBM model was trained on the earliest 70% of transactions. Threshold selection used the next 15% (validation); the final 15% (test) was evaluated after selecting the policy. Identical `TransactionDT` values remain in the same partition.

The search considered:
- approve thresholds: 0.02, 0.05, 0.10
- decline thresholds: 0.30, 0.50, 0.70, 0.90
- review ranking: fraud score or score × transaction amount
- daily review capacity: 100
- overflow: approve

The chosen policy minimizes the validation cost under the assumptions in `configs/costs.yaml`. The thresholds are applied to model risk scores; they are not calibrated probabilities.

## Selected policy

| Setting | Selected value |
|---|---:|
| Auto-approve below | 0.02 |
| Auto-decline at/above | 0.30 |
| Review queue ranking | score × transaction amount |
| Daily review capacity | 100 |
| Overflow | approve |

This is the best candidate in the specified grid, not a mathematically global optimum.

## Held-out test result

Test contains 88,581 transactions and 3,083 fraud labels across 31 relative-day buckets.

| Metric | Result |
|---|---:|
| PR-AUC | 0.4386 |
| ROC-AUC | 0.8722 |
| Reviewed | 3,100 |
| Auto-declined | 1,840 |
| Auto-approved | 83,641 |
| Fraud auto-declined | 1,100 |
| Fraud sent to review | 309 |
| Fraud approved | 1,674 |
| Legitimate auto-declined | 740 |
| Review precision | 9.97% |

Under the configured assumptions, estimated test costs are:

| Cost component | Value (dataset amount/cost units) |
|---|---:|
| Expected fraud loss | 189,518.40 |
| False-decline friction | 3,979.10 |
| Manual review | 3,100.00 |
| **Total** | **196,597.50** |
| **Per test transaction** | **2.2194** |

The expected fraud loss includes full amount for approved fraud plus the assumed 10% reviewer miss rate on fraud sent to review. Auto-declined fraud is treated as blocked. False-decline friction uses the configured fixed cost and the assumed 2% false-decline rate for legitimate reviewed transactions.

## Context: simple baselines on the same test period

| Policy | Estimated total cost |
|---|---:|
| Approve everything | 469,608.52 |
| Review top 100/day by score; approve all others | 318,042.87 |
| Validation-selected three-way policy | **196,597.50** |

These comparisons use the same illustrative cost assumptions. They are not real merchant savings or evidence of production performance.

## Sensitivity: thresholds depend on business assumptions

Changing only the fixed false-decline cost while retaining the other assumptions changed the validation-selected decline threshold:

| False-decline cost assumption | Selected decline threshold |
|---:|---:|
| 5 | 0.30 |
| 50 | 0.50 |
| 200 | 0.90 |

This is the central product insight: a threshold cannot be chosen from model metrics alone. The right policy depends on the merchant's relative cost of fraud versus rejecting a legitimate customer.

## Limitations and next steps

- All reviewer rates, costs, staffing and overflow choices are illustrative assumptions, not values supplied by IEEE-CIS.
- Model scores are not calibrated probabilities; the threshold values are score cut-offs.
- The search covers a small, explicit grid. Expand it on validation data only if a finer policy is needed.
- The final test set has now been used for this offline report. Do not repeatedly tune thresholds against it.
- The dataset does not contain reviewer outcomes, actual merchant margin, chargeback fee, or customer lifetime value.
- Next: run broader validation-only sensitivity analysis, inspect false declines and segment-level outcomes, then build the Streamlit decision-support interface.
