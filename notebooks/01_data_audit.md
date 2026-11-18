# Week 1 — IEEE-CIS Data Audit

The runnable audit is `scripts/audit_ieee_cis.py`. Do not train a model until its outputs and the leakage review below have been checked.

## Run

Download the official IEEE-CIS competition files from Kaggle and place `train_transaction.csv` and optionally `train_identity.csv` in `data/raw/`. Do not upload raw data to GitHub.

```bash
python scripts/audit_ieee_cis.py
```

Outputs: `data/processed/audit/audit_summary.json` and `missingness.csv`.

## Review manually

- Are TransactionID values unique in both tables?
- Does the identity left join preserve transaction row count?
- What is the fraud rate and target distribution?
- Which columns have high missingness?
- What is the TransactionDT range and how many 86,400-second relative buckets occur?
- Are there duplicate rows or suspicious target/feature relationships?
- Are chronological split boundaries and actual row counts reasonable?

## Leakage review

- Exclude `isFraud` from model features.
- Do not treat identifiers as meaningful numeric quantities without justification.
- Review whether each feature is available at decision time.
- Derive documented time features rather than blindly assuming raw TransactionDT generalizes.
- Check for post-transaction outcome fields.
- Fit preprocessing only on training data.
- Record excluded columns and reasons.

## Interpretation

`TransactionDT // 86400` is an elapsed-time bucket based on the dataset reference point, not a calendar date or verified merchant business day.

The chronological boundaries are approximate 70/15/15 sorted-row cut points. Equal timestamps are kept together, so actual partition counts may differ from the target proportions.

Do not fabricate business costs, reviewer performance, or savings. IEEE-CIS does not provide these operational facts.
