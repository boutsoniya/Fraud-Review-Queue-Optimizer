"""Capacity-aware approve/review/decline policy and cost evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OverflowAction = Literal["approve", "decline"]


@dataclass(frozen=True)
class CostConfig:
    """Business assumptions; reviewer rates are not supplied by IEEE-CIS."""
    fraud_loss_rate: float = 1.0
    false_decline_cost: float = 5.0
    manual_review_cost: float = 1.0
    reviewer_fraud_catch_rate: float = 0.90
    reviewer_legitimate_false_decline_rate: float = 0.02

    def validate(self) -> None:
        values = {
            "fraud_loss_rate": self.fraud_loss_rate,
            "false_decline_cost": self.false_decline_cost,
            "manual_review_cost": self.manual_review_cost,
            "reviewer_fraud_catch_rate": self.reviewer_fraud_catch_rate,
            "reviewer_legitimate_false_decline_rate": self.reviewer_legitimate_false_decline_rate,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("reviewer_fraud_catch_rate", "reviewer_legitimate_false_decline_rate"):
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


def assign_decisions(probabilities: pd.Series, approve_threshold: float,
                     decline_threshold: float, review_capacity: int,
                     overflow_action: OverflowAction) -> pd.Series:
    """Assign decisions for one day; highest-risk review candidates get slots."""
    _validate_inputs(probabilities, approve_threshold, decline_threshold,
                     review_capacity, overflow_action)

    decisions = pd.Series("APPROVE", index=probabilities.index, dtype="object")
    decisions.loc[probabilities >= decline_threshold] = "DECLINE"

    candidates = probabilities[
        (probabilities >= approve_threshold) & (probabilities < decline_threshold)
    ]
    ranked = candidates.sort_values(ascending=False, kind="mergesort").index
    selected = ranked[:review_capacity]
    overflow = ranked[review_capacity:]

    decisions.loc[selected] = "REVIEW"
    decisions.loc[overflow] = overflow_action.upper()
    return decisions


def assign_daily_decisions(transactions: pd.DataFrame, probability_col: str,
                           day_col: str, approve_threshold: float,
                           decline_threshold: float, daily_review_capacity: int,
                           overflow_action: OverflowAction) -> pd.Series:
    """Apply review capacity independently to each day."""
    missing = {probability_col, day_col}.difference(transactions.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if transactions[day_col].isna().any():
        raise ValueError(f"{day_col} cannot contain missing values.")

    result = pd.Series(index=transactions.index, dtype="object")
    for _, day_frame in transactions.groupby(day_col, sort=False):
        daily = assign_decisions(
            day_frame[probability_col], approve_threshold, decline_threshold,
            daily_review_capacity, overflow_action
        )
        result.loc[daily.index] = daily
    return result.rename("decision")


def evaluate_policy(y_true: pd.Series, transaction_amount: pd.Series,
                    decisions: pd.Series, costs: CostConfig) -> dict:
    """Estimate offline costs; reviewer outcomes are explicit assumptions."""
    costs.validate()
    if not (y_true.index.equals(transaction_amount.index)
            and y_true.index.equals(decisions.index)):
        raise ValueError("Inputs must have identical indexes.")
    if y_true.isna().any() or not y_true.isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels.")
    amounts = transaction_amount.to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Transaction amounts must be finite and non-negative.")
    if decisions.isna().any() or not decisions.isin(["APPROVE", "REVIEW", "DECLINE"]).all():
        raise ValueError("Invalid or missing decision.")

    fraud = y_true.eq(1)
    legitimate = y_true.eq(0)
    approved = decisions.eq("APPROVE")
    reviewed = decisions.eq("REVIEW")
    declined = decisions.eq("DECLINE")

    approved_fraud_loss = amounts[fraud & approved].sum() * costs.fraud_loss_rate
    missed_review_fraud_loss = (
        amounts[fraud & reviewed].sum()
        * costs.fraud_loss_rate
        * (1 - costs.reviewer_fraud_catch_rate)
    )
    # Expected value is fractional by design; rounding here would understate
    # low-rate reviewer errors in small evaluation samples.
    review_false_declines = (
        float((legitimate & reviewed).sum())
        * costs.reviewer_legitimate_false_decline_rate
    )
    false_declines = float((legitimate & declined).sum()) + review_false_declines
    review_count = int(reviewed.sum())

    fraud_loss = float(approved_fraud_loss + missed_review_fraud_loss)
    false_decline_cost = false_declines * costs.false_decline_cost
    review_cost = review_count * costs.manual_review_cost

    return {
        "total_cost": float(fraud_loss + false_decline_cost + review_cost),
        "fraud_loss": fraud_loss,
        "false_decline_cost": float(false_decline_cost),
        "manual_review_cost": float(review_cost),
        "approved_count": int(approved.sum()),
        "reviewed_count": review_count,
        "declined_count": int(declined.sum()),
        "fraud_approved_count": int((fraud & approved).sum()),
        "fraud_reviewed_count": int((fraud & reviewed).sum()),
        "fraud_declined_count": int((fraud & declined).sum()),
        "legitimate_declined_count": int((legitimate & declined).sum()),
        "expected_reviewed_legitimate_false_declines": review_false_declines,
        "expected_false_declines": false_declines,
    }
