import pandas as pd
import pytest

from src.policy import CostConfig, assign_daily_decisions, assign_decisions, evaluate_policy


def test_review_capacity_prioritizes_highest_risk():
    scores = pd.Series([0.10, 0.40, 0.70, 0.80, 0.95])
    result = assign_decisions(scores, 0.30, 0.90, 2, "approve")
    assert result.tolist() == ["APPROVE", "REVIEW", "REVIEW", "APPROVE", "DECLINE"]


def test_overflow_policy_is_explicit():
    scores = pd.Series([0.40, 0.50, 0.60])
    result = assign_decisions(scores, 0.30, 0.90, 1, "decline")
    assert result.tolist() == ["DECLINE", "DECLINE", "REVIEW"]


def test_capacity_resets_for_each_day():
    frame = pd.DataFrame({"day": [1, 1, 1, 2, 2], "score": [0.4, 0.5, 0.6, 0.7, 0.8]})
    result = assign_daily_decisions(frame, "score", "day", 0.3, 0.9, 1, "approve")
    assert result.eq("REVIEW").sum() == 2
    assert result.iloc[:3].tolist() == ["APPROVE", "APPROVE", "REVIEW"]
    assert result.iloc[3:].tolist() == ["APPROVE", "REVIEW"]


def test_cost_components():
    y = pd.Series([1, 0, 1, 0])
    amount = pd.Series([100.0, 50.0, 80.0, 20.0])
    decisions = pd.Series(["APPROVE", "DECLINE", "REVIEW", "REVIEW"])
    costs = CostConfig(1.0, 5.0, 2.0, 0.75, 0.0)
    result = evaluate_policy(y, amount, decisions, costs)
    assert result["fraud_loss"] == pytest.approx(120.0)
    assert result["false_decline_cost"] == pytest.approx(5.0)
    assert result["manual_review_cost"] == pytest.approx(4.0)
    assert result["total_cost"] == pytest.approx(129.0)


def test_invalid_threshold_order():
    with pytest.raises(ValueError):
        assign_decisions(pd.Series([0.2]), 0.8, 0.5, 1, "approve")


def test_nan_assumption_is_rejected():
    with pytest.raises(ValueError):
        CostConfig(false_decline_cost=float("nan")).validate()
