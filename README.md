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
- `data/processed/`: generated audit and model outputs (not committed)
- `notebooks/01_data_audit.md`: audit checklist
- `scripts/audit_ieee_cis.py`: runnable Week 1 audit
- `scripts/train_baseline.py`: time-aware LightGBM baseline with optional identity enrichment
- `src/config.py`: YAML settings loader
- `src/policy.py`: policy implementation and cost evaluation
- `tests/`: unit tests

## Dataset and audit

Download the official IEEE-CIS competition files from Kaggle after accepting its rules. Put `train_transaction.csv` and `train_identity.csv` in `data/raw/`. Do not commit raw data.

Run:

```bash
python scripts/audit_ieee_cis.py
```

The script writes `data/processed/audit/audit_summary.json` and `missingness.csv`. If identity data is present, it performs a validated one-to-one left join and reports match coverage.

It derives `floor(TransactionDT / 86400)` relative buckets. These are elapsed-time buckets, not real calendar dates. The chronological split uses sorted row positions to find approximate 70/15/15 boundaries and keeps equal TransactionDT values together; inspect actual counts because ties can change proportions.

## Baseline model

The first reproducible model is a compact LightGBM classifier using 60 selected transaction features (card, email, transaction, C/D/M, and V1–V20 fields). Pass `--identity data/raw/train_identity.csv` to run the matched experiment with 34 selected identity fields.

Run after installing `requirements.txt` and placing the transaction CSV in `data/raw/`:

```bash
python scripts/train_baseline.py

# Optional identity-enriched comparison
python scripts/train_baseline.py --identity data/raw/train_identity.csv
```

The script:
- uses an approximate 70/15/15 chronological split and keeps identical TransactionDT values together;
- uses validation AUC for early stopping;
- reports PR-AUC, ROC-AUC, precision/recall/F1 at a 0.5 threshold;
- compares top-100-per-relative-day queues ranked by score and by probability × amount;
- writes separate artifacts: `fraud_lgbm_baseline.joblib` / `baseline_metrics.json` for transaction-only runs, and `fraud_lgbm_identity.joblib` / `identity_metrics.json` when `--identity` is supplied.

Both output folders are git-ignored. Raw Kaggle data and trained artifacts should not be committed.

### First baseline result

See [reports/baseline_results.md](reports/baseline_results.md) for the measured chronological holdout results, matched identity comparison, and queue trade-offs. These are initial benchmarks, not tuned or production-ready results.

## Decision-policy experiment

After training the identity-enriched model, run the validation-selected decision policy:

```bash
python scripts/run_policy_experiment.py \
  --transactions data/raw/train_transaction.csv \
  --identity data/raw/train_identity.csv \
  --model models/fraud_lgbm_identity.joblib
```

The script searches the documented threshold grid and ranking strategies using validation data only, then evaluates the selected policy once on the chronological test partition. It writes `reports/policy_experiment.json`. The measured benchmark and its limitations are documented in [reports/policy_experiment.md](reports/policy_experiment.md).

Thresholds are model-score cut-offs, not calibrated probabilities. The result depends on the illustrative assumptions in `configs/costs.yaml`; do not present the estimated cost difference as actual merchant savings.

## Model explainability

Use SHAP to inspect the strongest model-score contributions for selected rows. This explains the model's scoring behavior, not causal reasons for fraud.

```bash
python scripts/explain_predictions.py --input data/processed/scored_features.csv --rows 10 20 30
```

The input CSV must contain the model's feature columns. SHAP is optional for normal dashboard usage and is computed on demand.

## Interactive decision simulator

Launch the local Streamlit interface:

```bash
streamlit run app.py
```

The app starts with synthetic demonstration data. It also accepts a CSV of model-scored transactions with:
- `fraud_score` (or `probability` / `score`), from 0 to 1;
- `TransactionAmt` (or `amount`);
- either `TransactionDT` (elapsed seconds) or `relative_day`;
- optional `isFraud` (0/1) for retrospective cost evaluation.

The interface lets an operations user change thresholds, staffing capacity, queue ranking, overflow behavior, and cost assumptions. It displays decisions, daily workload, the review queue, and (when labels are supplied) estimated offline costs. It is a policy simulator, not a live payment integration or a production fraud-decision service. Do not upload personal or payment-card data.

## Run locally

```bash
python -m pip install -r requirements.txt
pytest -q
```

## Four-week plan

1. **Data audit:** schema, missingness, class balance, time coverage, join coverage, leakage risks, chronological split. Completed.
2. **Model:** first LightGBM baseline completed; next compare identity enrichment, broader features, calibration, and validation-selected thresholds.
3. **Decision policy:** compare score-only and expected-loss ranking, thresholds, overflow, staffing, and cost sensitivity on the same chronological holdout.
4. **Decision-support app:** Streamlit interface for staffing, costs, thresholds, and policy comparison; publish a findings memo with limitations.

## Limitations

- IEEE-CIS is a public benchmark, not a live merchant's production stream.
- It does not provide actual reviewer outcomes, staffing schedules, chargeback fees, recovery rates, margin, or customer lifetime value.
- Relative TransactionDT buckets are not calendar dates.
- A model score is not a decision by itself; thresholds and queue capacity encode business trade-offs.
- The expected-loss queue currently uses probability × amount as a ranking heuristic; it is not calibrated recoverable loss.
- Report false declines and segment-level performance, not only aggregate fraud detection metrics.
- Do not claim real-world savings or production readiness from benchmark results.

## Current status

The data audit, transaction-only LightGBM baseline, and first identity-enriched comparison are complete. The capacity-aware policy, YAML assumptions, runnable audit, model-training script, and tests are in place. Next: tune the actual approve/review/decline policy on validation data before evaluating once on the held-out test period.
