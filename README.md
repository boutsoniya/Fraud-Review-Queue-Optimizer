# Fraud Review Queue Optimizer

A decision-support prototype for an e-commerce fraud operations team. It studies how limited daily manual-review capacity changes the trade-off between fraud loss, legitimate-customer friction, and review cost.

> **Scope:** This is a benchmark project using the public IEEE-CIS Fraud Detection dataset. It is not a production fraud system. Reviewer performance, operating costs, and staffing levels are configurable assumptions unless measured from real operations data.

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

The review queue is capped **per day**. If more transactions qualify than reviewers can process, overflow behavior must be chosen explicitly: approve or decline. Neither is universally correct.

## Structure

- configs/: transparent scenario assumptions
- data/raw/: downloaded Kaggle files (not committed)
- data/processed/: generated data (not committed)
- notebooks/: data audit checklist
- src/: policy implementation
- tests/: unit tests

## Dataset

Use the IEEE-CIS Fraud Detection competition dataset on Kaggle. Download the official transaction and identity CSV files after accepting the competition rules. Place them in data/raw/. The identity table is optional enrichment and should be left-joined on TransactionID; not every transaction has an identity row. Do not commit the data.

## Four-week plan

1. **Data audit:** schema, missingness, class balance, time coverage, join coverage, leakage risks, chronological split.
2. **Model:** baseline, then LightGBM. Evaluate PR-AUC, recall/precision at review capacity, and calibration—not accuracy alone.
3. **Decision policy:** compare strategies using explicit cost assumptions and per-day capacity.
4. **Decision-support app:** Streamlit interface for staffing, costs, thresholds, and policy comparison; publish a findings memo with limitations.

## Run locally

Install requirements.txt, then run pytest -q.

## Limitations

- IEEE-CIS is a public benchmark, not a live merchant's production stream.
- It does not provide actual reviewer outcomes, staffing schedules, chargeback fees, recovery rates, or customer lifetime value.
- Operating costs and reviewer catch/false-decline rates are assumptions until validated.
- A model score is not a decision by itself; thresholds and queue capacity encode business trade-offs.
- Report false declines and segment-level performance, not only aggregate fraud detection metrics.
- Do not claim real-world savings or production readiness from benchmark results.

## Current status

Foundation: capacity-aware decision policy and unit tests.

Next: complete the data audit before training a model.
