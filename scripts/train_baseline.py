"""Train a time-aware IEEE-CIS LightGBM baseline.

The compact transaction feature set is the default. Pass --identity to run the
matched identity-enriched experiment. Raw data and generated model artifacts
must stay out of Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORE_FEATURES = (
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 11)]
    + [f"D{i}" for i in range(1, 11)]
    + [f"M{i}" for i in range(1, 6)]
    + [f"V{i}" for i in range(1, 21)]
)
CATEGORICAL_FEATURES = (
    ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 6)]
)
IDENTITY_NUMERIC_FEATURES = (
    [f"id_{i:02d}" for i in range(1, 12)]
    + ["id_13", "id_14"]
    + [f"id_{i:02d}" for i in range(17, 23)]
    + ["id_24", "id_25", "id_26", "id_32"]
)
IDENTITY_CATEGORICAL_FEATURES = [
    "id_12", "id_15", "id_16", "id_28", "id_29", "id_34",
    "id_35", "id_36", "id_37", "id_38", "DeviceType",
]


def chronological_split_masks(
    transaction_dt: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.Series, pd.Series, pd.Series, float, float]:
    """Split in time order while keeping identical TransactionDT values together."""
    if transaction_dt.isna().any():
        raise ValueError("TransactionDT cannot contain missing values.")
    values = pd.to_numeric(transaction_dt, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("TransactionDT must contain finite numeric values.")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Train and validation fractions must be positive and sum to less than 1.")
    n = len(values)
    if n < 3:
        raise ValueError("At least three rows are required for chronological splitting.")

    ordered = np.sort(values.to_numpy())
    train_pos = max(0, int(n * train_fraction) - 1)
    validation_pos = max(train_pos, int(n * (train_fraction + validation_fraction)) - 1)
    train_boundary = float(ordered[train_pos])
    validation_boundary = float(ordered[validation_pos])

    train = values <= train_boundary
    validation = (values > train_boundary) & (values <= validation_boundary)
    test = values > validation_boundary
    if not train.any() or not validation.any() or not test.any():
        raise ValueError(
            "Chronological split produced an empty partition; inspect TransactionDT ties "
            "or choose different split fractions."
        )
    return train, validation, test, train_boundary, validation_boundary


def attach_identity_features(
    transactions: pd.DataFrame, identity: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Left-join identity rows without multiplying transaction records."""
    if "TransactionID" not in transactions or "TransactionID" not in identity:
        raise KeyError("Both dataframes must contain TransactionID.")
    if not transactions["TransactionID"].is_unique:
        raise ValueError("Transaction TransactionID is not unique.")
    if not identity["TransactionID"].is_unique:
        raise ValueError("Identity TransactionID is not unique.")
    joined = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator="_identity_match",
        sort=False,
    )
    if len(joined) != len(transactions):
        raise AssertionError("Identity join changed transaction row count.")
    match_rate = float(joined["_identity_match"].eq("both").mean())
    return joined.drop(columns="_identity_match"), match_rate


def capacity_queue_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    transaction_amount: pd.Series,
    relative_day: pd.Series,
    capacity_per_day: int,
) -> dict:
    """Compare top-k/day score and probability-times-amount review queues.

    This is a queue-ranking experiment only; it does not apply approve/decline
    thresholds. Probability-times-amount is a heuristic, not calibrated loss.
    """
    if isinstance(capacity_per_day, bool) or not isinstance(capacity_per_day, int) or capacity_per_day < 0:
        raise ValueError("capacity_per_day must be a non-negative integer.")
    indexes = [y_true.index, probabilities.index, transaction_amount.index, relative_day.index]
    if any(not indexes[0].equals(index) for index in indexes[1:]):
        raise ValueError("All queue inputs must have identical indexes.")
    frame = pd.DataFrame({
        "fraud": y_true.astype(int),
        "probability": probabilities.astype(float),
        "amount": transaction_amount.astype(float),
        "day": relative_day,
    })
    if frame.isna().any().any() or not frame["fraud"].isin([0, 1]).all():
        raise ValueError("Queue inputs contain missing values or invalid labels.")
    if not np.isfinite(frame[["probability", "amount"]].to_numpy()).all():
        raise ValueError("Queue probabilities and amounts must be finite.")
    if ((frame["probability"] < 0) | (frame["probability"] > 1)).any() or (frame["amount"] < 0).any():
        raise ValueError("Probabilities must be in [0, 1] and amounts non-negative.")

    total_fraud = int(frame["fraud"].sum())
    total_fraud_amount = float(frame.loc[frame["fraud"].eq(1), "amount"].sum())
    results = {}
    for name, ranking in (
        ("score", frame["probability"]),
        ("expected_loss_heuristic", frame["probability"] * frame["amount"]),
    ):
        selected = ranking.groupby(frame["day"], sort=False).rank(
            method="first", ascending=False
        ) <= capacity_per_day
        reviewed = frame.loc[selected]
        fraud_found = int(reviewed["fraud"].sum())
        fraud_amount = float(reviewed.loc[reviewed["fraud"].eq(1), "amount"].sum())
        results[name] = {
            "reviewed_count": int(selected.sum()),
            "fraud_found": fraud_found,
            "review_precision": float(reviewed["fraud"].mean()) if len(reviewed) else 0.0,
            "fraud_recall_of_test": fraud_found / total_fraud if total_fraud else 0.0,
            "fraud_amount_captured": fraud_amount,
            "fraud_amount_recall": fraud_amount / total_fraud_amount if total_fraud_amount else 0.0,
            "avg_review_amount": float(reviewed["amount"].mean()) if len(reviewed) else 0.0,
        }
    return results


def load_training_frame(
    transactions_path: Path, identity_path: Path | None = None
) -> tuple[pd.DataFrame, list[str], float | None]:
    """Read a compact feature set with memory-conscious dtypes."""
    header = pd.read_csv(transactions_path, nrows=0).columns.tolist()
    usecols = [column for column in CORE_FEATURES if column in header]
    required = {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}
    missing = required.difference(usecols)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    categorical = [column for column in CATEGORICAL_FEATURES if column in usecols]
    dtypes = {column: "category" for column in categorical}
    for column in usecols:
        if column not in categorical and column not in {"TransactionID", "isFraud", "TransactionDT"}:
            dtypes[column] = "float32"
    dtypes.update({"TransactionID": "int32", "isFraud": "int8", "TransactionDT": "int32"})
    frame = pd.read_csv(transactions_path, usecols=usecols, dtype=dtypes, low_memory=True)

    identity_match_rate = None
    if identity_path is not None:
        identity_header = pd.read_csv(identity_path, nrows=0).columns.tolist()
        identity_usecols = [
            column for column in (
                ["TransactionID"] + IDENTITY_NUMERIC_FEATURES + IDENTITY_CATEGORICAL_FEATURES
            ) if column in identity_header
        ]
        if "TransactionID" not in identity_usecols:
            raise ValueError("Identity dataset is missing TransactionID.")
        identity_dtypes = {"TransactionID": "int32"}
        identity_categories = [
            column for column in IDENTITY_CATEGORICAL_FEATURES if column in identity_usecols
        ]
        for column in identity_categories:
            identity_dtypes[column] = "category"
        for column in identity_usecols:
            if column not in identity_dtypes and column != "TransactionID":
                identity_dtypes[column] = "float32"
        identity = pd.read_csv(
            identity_path, usecols=identity_usecols, dtype=identity_dtypes, low_memory=True
        )
        frame, identity_match_rate = attach_identity_features(frame, identity)
        categorical.extend(identity_categories)
    return frame, categorical, identity_match_rate


def train(args: argparse.Namespace) -> dict:
    # Optional ML dependencies stay out of module import so lightweight CI can test helpers.
    import joblib
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.metrics import (
        average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    )

    frame, categorical, identity_match_rate = load_training_frame(
        args.transactions, args.identity
    )
    y = frame["isFraud"].astype("int8")
    train_mask, validation_mask, test_mask, train_boundary, validation_boundary = (
        chronological_split_masks(frame["TransactionDT"])
    )
    relative_day = (frame["TransactionDT"] // 86400).astype("int16")

    features = frame.drop(columns=["TransactionID", "isFraud"])
    training_frame = features.loc[train_mask]
    kept_features = [
        column for column in features.columns
        if training_frame[column].notna().any() and training_frame[column].nunique(dropna=True) > 1
    ]
    features = features[kept_features]
    categorical = [column for column in categorical if column in kept_features]
    for column in categorical:
        features[column] = features[column].astype("category")

    x_train, y_train = features.loc[train_mask], y.loc[train_mask]
    x_validation, y_validation = features.loc[validation_mask], y.loc[validation_mask]
    x_test, y_test = features.loc[test_mask], y.loc[test_mask]

    model = LGBMClassifier(
        objective="binary",
        n_estimators=args.rounds,
        learning_rate=0.06,
        num_leaves=15,
        max_depth=7,
        min_child_samples=150,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        max_bin=63,
        random_state=42,
        n_jobs=args.threads,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="auc",
        categorical_feature=categorical,
        callbacks=[early_stopping(args.early_stopping_rounds, verbose=False), log_evaluation(50)],
    )
    probabilities = pd.Series(
        model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1],
        index=y_test.index,
    )
    test_fraud_rate = float(y_test.mean())
    metrics = {
        "dataset": "IEEE-CIS Fraud Detection",
        "model": "LightGBM binary classifier",
        "feature_scope": (
            "transaction features plus selected identity fields"
            if args.identity else "60-feature transaction-only first-pass subset"
        ),
        "feature_count": len(kept_features),
        "identity_features_used": (
            sum(column.startswith("id_") or column == "DeviceType" for column in kept_features)
            if args.identity else 0
        ),
        "identity_join_match_rate": identity_match_rate,
        "categorical_features": categorical,
        "best_iteration": int(model.best_iteration_),
        "split_rows": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "split_boundaries": {
            "train_end_TransactionDT": train_boundary,
            "validation_end_TransactionDT": validation_boundary,
        },
        "test_rows": int(len(y_test)),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_rate": test_fraud_rate,
        "dummy_pr_auc": test_fraud_rate,
        "lightgbm_pr_auc": float(average_precision_score(y_test, probabilities)),
        "lightgbm_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision_at_0_5": float(precision_score(y_test, probabilities >= 0.5, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, probabilities >= 0.5, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_test, probabilities >= 0.5, zero_division=0)),
        "relative_day_count_test": int(relative_day.loc[test_mask].nunique()),
        "review_capacity_per_relative_day": args.review_capacity,
    }
    metrics["capacity_queue_comparison"] = capacity_queue_metrics(
        y_test,
        probabilities,
        frame.loc[test_mask, "TransactionAmt"],
        relative_day.loc[test_mask],
        args.review_capacity,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": kept_features, "categorical_features": categorical},
        args.model_out,
        compress=3,
    )
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a time-aware IEEE-CIS LightGBM baseline.")
    parser.add_argument("--transactions", type=Path, default=Path("data/raw/train_transaction.csv"))
    parser.add_argument("--identity", type=Path, default=None,
                        help="Optional train_identity.csv for the identity-enriched experiment.")
    parser.add_argument("--model-out", type=Path, default=Path("models/fraud_lgbm_baseline.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("data/processed/model/baseline_metrics.json"))
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--review-capacity", type=int, default=100)
    args = parser.parse_args()
    if not args.transactions.exists():
        parser.error(f"Transaction CSV not found: {args.transactions}")
    if args.identity is not None and not args.identity.exists():
        parser.error(f"Identity CSV not found: {args.identity}")
    if args.rounds < 1 or args.early_stopping_rounds < 1 or args.threads < 1:
        parser.error("rounds, early-stopping-rounds, and threads must be positive.")
    train(args)


if __name__ == "__main__":
    main()
