"""Audit IEEE-CIS transaction/identity CSVs without training a model.

Usage:
    python scripts/audit_ieee_cis.py --transactions data/raw/train_transaction.csv \
        --identity data/raw/train_identity.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit(transactions_path: Path, identity_path: Path | None, output_dir: Path) -> dict:
    tx = pd.read_csv(transactions_path)
    summary = {
        "transaction_rows": int(len(tx)),
        "transaction_columns": int(tx.shape[1]),
        "transaction_id_unique": bool(tx["TransactionID"].is_unique) if "TransactionID" in tx else None,
        "duplicate_transaction_ids": int(tx["TransactionID"].duplicated().sum()) if "TransactionID" in tx else None,
    }
    if "isFraud" in tx:
        summary["fraud_count"] = int(tx["isFraud"].sum())
        summary["legitimate_count"] = int((tx["isFraud"] == 0).sum())
        summary["fraud_rate"] = float(tx["isFraud"].mean())

    if identity_path is not None:
        identity = pd.read_csv(identity_path)
        summary.update({
            "identity_rows": int(len(identity)),
            "identity_columns": int(identity.shape[1]),
            "identity_transaction_id_unique": bool(identity["TransactionID"].is_unique),
        })
        if not identity["TransactionID"].is_unique:
            raise ValueError("Identity TransactionID is not unique; left join could multiply rows.")
        before = len(tx)
        joined = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one",
                          indicator=True, suffixes=("", "_identity"))
        if len(joined) != before:
            raise AssertionError("Identity join changed transaction row count.")
        summary["identity_join_rows"] = int(len(joined))
        summary["identity_match_count"] = int((joined["_merge"] == "both").sum())
        summary["identity_match_rate"] = float((joined["_merge"] == "both").mean())
        tx = joined.drop(columns="_merge")

    if "TransactionDT" in tx:
        dt = pd.to_numeric(tx["TransactionDT"], errors="coerce")
        if dt.isna().any():
            raise ValueError("TransactionDT contains missing or non-numeric values.")
        tx["relative_day"] = (dt // 86400).astype("int64")
        summary.update({
            "transaction_dt_min": float(dt.min()),
            "transaction_dt_max": float(dt.max()),
            "relative_day_count": int(tx["relative_day"].nunique()),
            "relative_day_semantics": "floor(TransactionDT / 86400); elapsed-time buckets, not calendar dates",
        })
        ordered = tx.sort_values("TransactionDT", kind="mergesort")
        n = len(ordered)
        train_pos = max(0, int(n * 0.70) - 1)
        valid_pos = max(train_pos, int(n * 0.85) - 1)
        train_boundary = float(ordered["TransactionDT"].iloc[train_pos])
        valid_boundary = float(ordered["TransactionDT"].iloc[valid_pos])
        train_mask = tx["TransactionDT"] <= train_boundary
        valid_mask = (tx["TransactionDT"] > train_boundary) & (tx["TransactionDT"] <= valid_boundary)
        test_mask = tx["TransactionDT"] > valid_boundary
        summary["chronological_split"] = {
            "train_end_transaction_dt": train_boundary,
            "validation_end_transaction_dt": valid_boundary,
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(valid_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "target_split_method": "70/15/15 by sorted row position; equal TransactionDT values kept together",
        }

    missingness = tx.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_frame()
    output_dir.mkdir(parents=True, exist_ok=True)
    missingness.to_csv(output_dir / "missingness.csv")
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IEEE-CIS fraud detection CSV files.")
    parser.add_argument("--transactions", type=Path, default=Path("data/raw/train_transaction.csv"))
    parser.add_argument("--identity", type=Path, default=Path("data/raw/train_identity.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/audit"))
    args = parser.parse_args()
    if not args.transactions.exists():
        parser.error(f"Transaction file not found: {args.transactions}")
    identity_path = args.identity if args.identity.exists() else None
    summary = audit(args.transactions, identity_path, args.output_dir)
    print(json.dumps(summary, indent=2))
    if identity_path is None:
        print("NOTE: identity file not found; audit completed without identity enrichment.")


if __name__ == "__main__":
    main()
