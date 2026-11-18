"""Capacity-aware approve/review/decline policy and offline cost evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OverflowAction = Literal["approve", "decline"]
RankingStrategy = Literal["score", "expected_loss"]


@dataclass(frozen=True)
class CostConfig:
    """Illustrative business assumptions; none are supplied by IEEE-CIS."""
    fraud_loss_rate: float = 1.0
    false_decline_cost: float = 5.0
    manual_review_cost: float = 1.0
    reviewer_fraud_catch_rate: float = 0.90
    reviewer_legitimate_false_decline_rate: float = 0.02
    false_decline_margin_rate: float = 0.0
    chargeback_fee: float = 0.0
    customer_lifetime_value_cost: float = 0.0

    def validate(self) -> None:
        values = vars(self)
        for name, value in values.items():
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("reviewer_fraud_catch_rate", "reviewer_legitimate_false_decline_rate",
                     "false_decline_margin_rate"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be between 0 and 1.")


def _validate_inputs(probabilities, approve_threshold, decline_threshold,
                     review_capacity, overflow_action):
    if not isinstance(probabilities, pd.Series):
        raise TypeError("probabilities must be a pandas Series.")
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not (np.isfinite(approve_threshold) and np.isfinite(decline_threshold)):
        raise ValueError("Thresholds must be finite.")
    if not (0 <= approve_threshold < decline_threshold <= 1):
        raise ValueError("Require 0 <= approve_threshold < decline_threshold <= 1.")
    if isinstance(review_capacity, bool) or not isinstance(review_capacity, (int, np.integer)):
        raise TypeError("review_capacity must be an integer.")
    if review_capacity < 0:
        raise ValueError("review_capacity cannot be negative.")
    if overflow_action not in ("approve", "decline"):
        raise ValueError("overflow_action must be approve or decline.")
    if not probabilities.index.is_unique:
        raise ValueError("probabilities index must be unique.")


def assign_decisions(probabilities: pd.Series, approve_threshold: float,
                     decline_threshold: float, review_capacity: int,
                     overflow_action: OverflowAction,
                     ranking_strategy: RankingStrategy = "score",
                     transaction_amount: pd.Series | None = None) -> pd.Series:
    """Assign decisions for one day; allocate review slots by score or expected loss.

    expected_loss ranking uses probability * transaction amount. It is a
    prioritization heuristic, not a calibrated estimate of recoverable loss.
    """
    _validate_inputs(probabilities, approve_threshold, decline_threshold,
                     review_capacity, overflow_action)
    if ranking_strategy not in ("score", "expected_loss"):
        raise ValueError("ranking_strategy must be 'score' or 'expected_loss'.")
    if ranking_strategy == "expected_loss":
        if transaction_amount is None:
            raise ValueError("transaction_amount is required for expected_loss ranking.")
        if not probabilities.index.equals(transaction_amount.index):
            raise ValueError("probabilities and transaction_amount must have identical indexes.")
        amounts = transaction_amount.to_numpy(dtype=float)
        if not np.isfinite(amounts).all() or (amounts < 0).any():
            raise ValueError("Transaction amounts must be finite and non-negative.")
        ranking_values = probabilities * transaction_amount
    else:
        ranking_values = probabilities

    decisions = pd.Series("APPROVE", index=probabilities.index, dtype="object")
    decisions.loc[probabilities >= decline_threshold] = "DECLINE"
    candidates = ranking_values[
        (probabilities >= approve_threshold) & (probabilities < decline_threshold)
    ]
    ranked = candidates.sort_values(ascending=False, kind="mergesort").index
    selected = ranked[:review_capacity]
    overflow = ranked[review_capacity:]
    decisions.loc[selected] = "REVIEW"
    decisions.loc[overflow] = overflow_action.upper()
    return decisions


def derive_relative_day(transactions: pd.DataFrame,
                        transaction_dt_col: str = "TransactionDT",
                        output_col: str = "relative_day") -> pd.Series:
    """Convert elapsed TransactionDT seconds into zero-based 86,400-second buckets.

    These are relative buckets, not calendar dates or verified business days.
    """
    if transaction_dt_col not in transactions:
        raise KeyError(f"Missing required column: {transaction_dt_col}")
    values = pd.to_numeric(transactions[transaction_dt_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{transaction_dt_col} must contain finite numeric seconds.")
    if (values < 0).any():
        raise ValueError(f"{transaction_dt_col} cannot contain negative values.")
    return (values // 86400).astype("int64").rename(output_col)


def assign_daily_decisions(transactions: pd.DataFrame, probability_col: str,
                           day_col: str, approve_threshold: float,
                           decline_threshold: float, daily_review_capacity: int,
                           overflow_action: OverflowAction,
                           ranking_strategy: RankingStrategy = "score",
                           amount_col: str | None = None) -> pd.Series:
    """Apply review capacity independently to each relative day bucket."""
    missing = {probability_col, day_col}.difference(transactions.columns)
    if ranking_strategy == "expected_loss":
        if amount_col is None:
            raise ValueError("amount_col is required for expected_loss ranking.")
        missing.add(amount_col)
    if missing.intersection(transactions.columns) == set():
        pass
    missing = {c for c in missing if c not in transactions.columns}
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if transactions[day_col].isna().any():
        raise ValueError(f"{day_col} cannot contain missing values.")
    if not transactions.index.is_unique:
        raise ValueError("transactions index must be unique.")

    result = pd.Series(index=transactions.index, dtype="object")
    for _, day_frame in transactions.groupby(day_col, sort=False):
        amount = day_frame[amount_col] if ranking_strategy == "expected_loss" else None
        daily = assign_decisions(
            day_frame[probability_col], approve_threshold, decline_threshold,
            daily_review_capacity, overflow_action, ranking_strategy, amount
        )
        result.loc[daily.index] = daily
    return result.rename("decision")


def evaluate_policy(y_true: pd.Series, transaction_amount: pd.Series,
                    decisions: pd.Series, costs: CostConfig) -> dict:
    """Estimate offline costs using explicit reviewer and merchant assumptions.

    False-decline cost = fixed friction + amount * margin rate + assumed CLV
    impact. Chargeback fee is applied to fraud expected to pass through.
    """
    costs.validate()
    if not (y_true.index.equals(transaction_amount.index)
            and y_true.index.equals(decisions.index)):
        raise ValueError("Inputs must have identical indexes.")
    if not y_true.index.is_unique:
        raise ValueError("Input indexes must be unique.")
    if y_true.isna().any() or not y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels.")
    amounts = transaction_amount.to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Transaction amounts must be finite and non-negative.")
    if decisions.isna().any() or not decisions.isin(["APPROVE", "REVIEW", "DECLINE"]).all():
        raise ValueError("Invalid or missing decision.")

    fraud, legitimate = y_true.eq(1), y_true.eq(0)
    approved, reviewed, declined = (decisions.eq(x) for x in ("APPROVE", "REVIEW", "DECLINE"))
    approved_fraud_amount = float(amounts[fraud & approved].sum())
    reviewed_fraud_amount = float(amounts[fraud & reviewed].sum())
    approved_fraud_count = int((fraud & approved).sum())
    reviewed_fraud_count = int((fraud & reviewed).sum())
    missed_review_fraud_count = reviewed_fraud_count * (1 - costs.reviewer_fraud_catch_rate)

    approved_fraud_loss = approved_fraud_amount * costs.fraud_loss_rate
    missed_review_fraud_loss = (
        reviewed_fraud_amount * costs.fraud_loss_rate
        * (1 - costs.reviewer_fraud_catch_rate)
    )
    review_false_declines = float((legitimate & reviewed).sum()) * costs.reviewer_legitimate_false_decline_rate
    declined_legitimate_count = int((legitimate & declined).sum())
    false_declines = declined_legitimate_count + review_false_declines
    declined_legitimate_amount = float(amounts[legitimate & declined].sum())
    reviewed_legitimate_amount = float(amounts[legitimate & reviewed].sum())
    expected_reviewed_legitimate_amount_declined = (
        reviewed_legitimate_amount * costs.reviewer_legitimate_false_decline_rate
    )
    false_decline_cost = (
        false_declines * costs.false_decline_cost
        + (declined_legitimate_amount + expected_reviewed_legitimate_amount_declined)
        * costs.false_decline_margin_rate
        + false_declines * costs.customer_lifetime_value_cost
    )
    chargeback_cost = (
        approved_fraud_count + missed_review_fraud_count
    ) * costs.chargeback_fee
    review_count = int(reviewed.sum())
    fraud_loss = float(approved_fraud_loss + missed_review_fraud_loss)
    review_cost = review_count * costs.manual_review_cost

    return {
        "total_cost": float(fraud_loss + false_decline_cost + chargeback_cost + review_cost),
        "fraud_loss": float(fraud_loss),
        "false_decline_cost": float(false_decline_cost),
        "chargeback_cost": float(chargeback_cost),
        "manual_review_cost": float(review_cost),
        "approved_count": int(approved.sum()),
        "reviewed_count": review_count,
        "declined_count": int(declined.sum()),
        "fraud_approved_count": approved_fraud_count,
        "fraud_reviewed_count": reviewed_fraud_count,
        "fraud_declined_count": int((fraud & declined).sum()),
        "legitimate_declined_count": declined_legitimate_count,
        "expected_reviewed_legitimate_false_declines": review_false_declines,
        "expected_false_declines": float(false_declines),
    }
