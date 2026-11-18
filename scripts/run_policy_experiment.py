"""Select approve/review/decline thresholds on validation, then evaluate once on test.

Uses a previously trained model artifact. The cost assumptions are illustrative
and must be replaced with merchant-approved values before operational use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_baseline import chronological_split_masks, load_training_frame
from src.config import load_policy_settings
from src.policy import assign_daily_decisions, evaluate_policy


def prepare_features(frame: pd.DataFrame, features: list[str],
                     categorical: list[str]) -> pd.DataFrame:
    """Reproduce the model's feature selection and categorical representation."""
    x = frame.drop(columns=["TransactionID", "isFraud"])
    missing = set(features).difference(x.columns)
    if missing:
        raise ValueError(f"Model features missing from data: {sorted(missing)}")
    x = x.loc[:, features].copy()
    for column in categorical:
        if column in x:
            x[column] = x[column].astype("category")
    return x


def evaluate_candidate(y: pd.Series, amount: pd.Series, scores: pd.Series,
                       day: pd.Series, approve_threshold: float,
                       decline_threshold: float, capacity: int,
                       overflow_action: str, ranking_strategy: str,
                       costs) -> tuple[float, dict]:
    frame = pd.DataFrame({
        "score": scores,
        "amount": amount,
        "day": day,
    }, index=y.index)
    decisions = assign_daily_decisions(
        frame, "score", "day", approve_threshold, decline_threshold,
        capacity, overflow_action, ranking_strategy,
        "amount" if ranking_strategy == "expected_loss" else None,
    )
    metrics = evaluate_policy(y, amount, decisions, costs)
    return metrics["total_cost"], metrics


def select_policy(y: pd.Series, amount: pd.Series, scores: pd.Series,
                  day: pd.Series, capacity: int, overflow_action: str,
                  costs, approve_grid: tuple[float, ...],
                  decline_grid: tuple[float, ...]) -> dict:
    """Choose the lowest-assumption-cost policy using validation data only."""
    candidates = []
    for approve_threshold in approve_grid:
        for decline_threshold in decline_grid:
            if approve_threshold >= decline_threshold:
                continue
            for ranking in ("score", "expected_loss"):
                total_cost, metrics = evaluate_candidate(
                    y, amount, scores, day, approve_threshold,
                    decline_threshold, capacity, overflow_action,
                    ranking, costs,
                )
                candidates.append({
                    "approve_threshold": approve_threshold,
                    "decline_threshold": decline_threshold,
                    "ranking_strategy": ranking,
                    "validation_total_cost": total_cost,
                    "validation_metrics": metrics,
                })
    if not candidates:
        raise ValueError("Threshold grids produced no valid policies.")
    # Stable tie-break: lower review volume, then score ranking.
    candidates.sort(key=lambda item: (
        item["validation_total_cost"],
        item["validation_metrics"]["reviewed_count"],
        item["ranking_strategy"] != "score",
    ))
    return candidates[0]


def run(args: argparse.Namespace) -> dict:
    import joblib
    from sklearn.metrics import average_precision_score, roc_auc_score

    settings = load_policy_settings(args.config)
    artifact = joblib.load(args.model)
    model = artifact["model"]
    frame, _, identity_match_rate = load_training_frame(
        args.transactions, args.identity
    )
    train_mask, validation_mask, test_mask, train_boundary, validation_boundary = (
        chronological_split_masks(frame["TransactionDT"])
    )
    x = prepare_features(frame, artifact["features"], artifact["categorical_features"])
    y = frame["isFraud"].astype("int8")
    day = (frame["TransactionDT"] // 86400).astype("int16")
    # The artifact's feature schema is reused; no fitting or threshold tuning on test.
    validation_scores = pd.Series(
        model.predict_proba(x.loc[validation_mask])[:, 1], index=frame.index[validation_mask]
    )
    test_scores = pd.Series(
        model.predict_proba(x.loc[test_mask])[:, 1], index=frame.index[test_mask]
    )
    validation_y = y.loc[validation_mask]
    test_y = y.loc[test_mask]
    validation_amount = frame.loc[validation_mask, "TransactionAmt"].astype(float)
    test_amount = frame.loc[test_mask, "TransactionAmt"].astype(float)
    validation_day = day.loc[validation_mask]
    test_day = day.loc[test_mask]

    approve_grid = tuple(args.approve_grid)
    decline_grid = tuple(args.decline_grid)
    selected = select_policy(
        validation_y, validation_amount, validation_scores, validation_day,
        settings.daily_review_capacity, settings.overflow_action, settings.costs,
        approve_grid, decline_grid,
    )
    test_cost, test_policy_metrics = evaluate_candidate(
        test_y, test_amount, test_scores, test_day,
        selected["approve_threshold"], selected["decline_threshold"],
        settings.daily_review_capacity, settings.overflow_action,
        selected["ranking_strategy"], settings.costs,
    )
    result = {
        "dataset": "IEEE-CIS Fraud Detection",
        "model_artifact": str(args.model),
        "identity_join_match_rate": identity_match_rate,
        "split_rows": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "split_boundaries": {
            "train_end_TransactionDT": train_boundary,
            "validation_end_TransactionDT": validation_boundary,
        },
        "validation_pr_auc": float(average_precision_score(validation_y, validation_scores)),
        "test_pr_auc": float(average_precision_score(test_y, test_scores)),
        "test_roc_auc": float(roc_auc_score(test_y, test_scores)),
        "daily_review_capacity": settings.daily_review_capacity,
        "overflow_action": settings.overflow_action,
        "cost_assumptions": vars(settings.costs),
        "selected_on_validation": selected,
        "test_policy_metrics": test_policy_metrics,
        "test_policy_total_cost": test_cost,
        "test_policy_cost_per_transaction": test_cost / len(test_y),
        "test_fraud_rate": float(test_y.mean()),
        "threshold_grids": {
            "approve": approve_grid,
            "decline": decline_grid,
        },
        "warning": (
            "Test metrics are an offline benchmark under illustrative cost and reviewer "
            "assumptions, not measured merchant savings or production performance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=Path("data/raw/train_transaction.csv"))
    parser.add_argument("--identity", type=Path, default=Path("data/raw/train_identity.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/fraud_lgbm_identity.joblib"))
    parser.add_argument("--config", type=Path, default=Path("configs/costs.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/policy_experiment.json"))
    parser.add_argument("--approve-grid", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10])
    parser.add_argument("--decline-grid", type=float, nargs="+",
                        default=[0.30, 0.50, 0.70, 0.90])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
