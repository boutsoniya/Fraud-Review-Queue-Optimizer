# Week 1 — IEEE-CIS Data Audit

Do not train a model until this audit is complete.

## Download

Download the official IEEE-CIS Fraud Detection competition files from Kaggle and place train_transaction.csv and train_identity.csv in data/raw/. Do not upload the raw files to GitHub.

## Questions

- How many rows and columns are in each table?
- Is TransactionID unique in each table?
- What is the fraud rate?
- What is the missing-value percentage per column?
- How many transaction rows receive identity enrichment after a left join?
- What is the range and distribution of TransactionDT?
- How many distinct time buckets are represented?
- Are there duplicate rows or suspicious target/feature relationships?

## Safe join

Use the transaction table as the base and left-join identity features on TransactionID. Check row counts before and after the join. The join must not multiply transaction rows.

## Chronological split

Sort by TransactionDT and split into earlier training data, later validation data, and the latest held-out test period. Avoid random splitting because it can leak future patterns into training. Record exact time boundaries and row counts. Fit preprocessing only on training data.

## Leakage review

- Exclude isFraud from model features.
- Do not treat identifiers as meaningful numeric quantities without justification.
- Review whether each feature is available at decision time.
- Derive documented time features rather than blindly assuming raw TransactionDT generalizes.
- Check for post-transaction outcome fields.
- Record excluded columns and reasons.

## Required audit outputs

1. Dataset dimensions and join coverage.
2. Target distribution.
3. Missingness summary.
4. Time coverage and chronological split boundaries.
5. Feature groups and leakage exclusions.
6. Limitations and assumptions.

Do not fabricate business costs, reviewer performance, or savings. IEEE-CIS does not provide these operational facts.
