# Fraud Review Queue Optimizer

A decision-support prototype for an e-commerce fraud operations team. It studies how limited daily manual-review capacity changes the trade-off between fraud loss, legitimate-customer friction, and review cost.

> **Scope:** This is a benchmark project using the public IEEE-CIS Fraud Detection dataset. It is not a production fraud system. Reviewer performance, operating costs, staffing levels, chargeback fees, and customer-value impact are configurable assumptions unless measured from real operations data.

## Operational question

For a team with a fixed number of reviewers each day:
- Which transactions should be approved automatically?
- Which should be declined automatically?
- Which should enter manual review?
- How should the policy change when capacity or business costs change?

## Decision policy

| Model score | Decision |
|---|---|
| Below the review threshold | Approve |
| At/above review threshold and below decline threshold | Review, subject to daily capacity |
| At/above decline threshold | Decline |

The review queue is capped per relative day bucket. If more transactions qualify than reviewers can process, overflow behavior must be chosen explicitly: approve or decline. Neither is universally correct.

Review slots support two ranking strategies:
- **Score:** highest predicted fraud probability first.
- **Expected loss heuristic:** highest predicted probability × transaction amount first. This is a prioritization heuristic, not a calibrated estimate of recoverable loss. Both strategies must be compared on the same chronological holdout set.

## Configuration

Edit `configs/costs.yaml`. Daily capacity is loaded from staffing assumptions:

`reviewers × reviews_per_reviewer_per_day`

The current illustrative values produce 100 reviews/day (5 × 20). The overflow action is also a scenario assumption, not a default recommendation.

The cost model separates expected fraud principal loss, fixed false-decline friction, amount-sensitive margin impact, assumed customer lifetime value impact, chargeback fees for fraud expected to pass through, and manual review operating cost. All values are placeholders until validated with a merchant.

## Structure

- `configs/`: transparent scenario assumptions
- `data/raw/`: downloaded Kaggle files (not committed)
- `data/processed/`: generated audit outputs (not committed)
- `notebooks/01_data_audit.md`: audit checklist
- `scripts/audit_ieee_cis.py`: runnable Week 1 audit
- `src/config.py`: YAML settings loader
- `src/policy.py`: policy implementation and cost evaluation
- `tests/`: unit tests

## Dataset and audit

Download the official IEEE-CIS competition files from Kaggle after accepting its rules. Put `train_transaction.csv` and optionally `train_identity.csv` in `data/raw/`. Do not commit raw data.

Run `python scripts/audit_ieee_cis.py`. The script writes `data/processed/audit/audit_summary.json` and `missingness.csv`. If identity data is present, it performs a validated one-to-one left join and reports match coverage.

It derives `floor(TransactionDT / 86400)` relative buckets. These are elapsed-time buckets, not real calendar dates. The chronological split uses sorted row positions to find approximate 70/15/15 boundaries and keeps equal TransactionDT values together; inspect actual counts because ties can change proportions.

## Run locally

```bash
python -m pip install -r requirements.txt
pytest -q
```

## Four-week plan

1. **Data audit:** schema, missingness, class balance, time coverage, join coverage, leakage risks, chronological split.
2. **Model:** baseline, then LightGBM. Evaluate PR-AUC, precision/recall at review capacity, and calibration—not accuracy alone.
3. **Decision policy:** compare score-only and expected-loss ranking, thresholds, overflow, staffing, and cost sensitivity on the same chronological holdout.
4. **Decision-support app:** Streamlit interface for staffing, costs, thresholds, and policy comparison; publish a findings memo with limitations.

## Limitations

- IEEE-CIS is a public benchmark, not a live merchant's production stream.
- It does not provide actual reviewer outcomes, staffing schedules, chargeback fees, recovery rates, margin, or customer lifetime value.
- Relative TransactionDT buckets are not calendar days.
- A model score is not a decision by itself; thresholds and queue capacity encode business trade-offs.
- Report false declines and segment-level performance, not only aggregate fraud detection metrics.
- Do not claim real-world savings or production readiness from benchmark results.

## Current status

Policy supports YAML-configured staffing capacity, per-relative-day decisions, score/amount-weighted queue ranking, and assumption-based cost breakdown. The runnable data audit is in place. Model training remains blocked until the Kaggle data is supplied and audited.
