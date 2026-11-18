import pandas as pd
import pytest

from src.config import load_policy_settings
from src.policy import (
    CostConfig, assign_daily_decisions, assign_decisions,
    derive_relative_day, evaluate_policy,
)


def test_review_capacity_prioritizes_highest_risk():
    scores = pd.Series([0.10, 0.40, 0.70, 0.80, 0.95])
    result = assign_decisions(scores, 0.30, 0.90, 2, "approve")
    assert result.tolist() == ["APPROVE", "APPROVE", "REVIEW", "REVIEW", "DECLINE"]


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


def test_cost_components_remain_compatible_with_original_defaults():
    y = pd.Series([1, 0, 1, 0])
    amount = pd.Series([100.0, 50.0, 80.0, 20.0])
    decisions = pd.Series(["APPROVE", "DECLINE", "REVIEW", "REVIEW"])
    result = evaluate_policy(y, amount, decisions, CostConfig(1.0, 5.0, 2.0, 0.75, 0.0))
    assert result["fraud_loss"] == pytest.approx(120.0)
    assert result["false_decline_cost"] == pytest.approx(5.0)
    assert result["manual_review_cost"] == pytest.approx(4.0)
    assert result["total_cost"] == pytest.approx(129.0)


def test_zero_capacity_sends_review_candidates_to_overflow_action():
    scores = pd.Series([0.4, 0.7])
    assert assign_decisions(scores, 0.3, 0.9, 0, "approve").tolist() == ["APPROVE", "APPROVE"]
    assert assign_decisions(scores, 0.3, 0.9, 0, "decline").tolist() == ["DECLINE", "DECLINE"]


def test_tied_scores_keep_input_order():
    scores = pd.Series([0.5, 0.5, 0.5], index=["first", "second", "third"])
    result = assign_decisions(scores, 0.3, 0.9, 2, "approve")
    assert result.tolist() == ["REVIEW", "REVIEW", "APPROVE"]


def test_expected_loss_ranking_uses_probability_times_amount():
    scores = pd.Series([0.8, 0.5], index=["small", "large"])
    amounts = pd.Series([10.0, 100.0], index=scores.index)
    result = assign_decisions(scores, 0.3, 0.95, 1, "approve",
                              ranking_strategy="expected_loss", transaction_amount=amounts)
    assert result.to_dict() == {"small": "APPROVE", "large": "REVIEW"}


def test_expected_loss_requires_aligned_amounts():
    with pytest.raises(ValueError):
        assign_decisions(pd.Series([0.5], index=[0]), 0.3, 0.9, 1, "approve",
                         ranking_strategy="expected_loss",
                         transaction_amount=pd.Series([10.0], index=[1]))


def test_relative_day_uses_elapsed_seconds_not_calendar_dates():
    frame = pd.DataFrame({"TransactionDT": [0, 86399, 86400, 172800]})
    assert derive_relative_day(frame).tolist() == [0, 0, 1, 2]


def test_relative_day_rejects_invalid_values():
    with pytest.raises(ValueError):
        derive_relative_day(pd.DataFrame({"TransactionDT": [0, None]}))


def test_yaml_config_loads_staffing_capacity():
    settings = load_policy_settings("configs/costs.yaml")
    assert settings.daily_review_capacity == 100
    assert settings.overflow_action == "approve"


def test_fractional_reviewer_false_declines_and_amount_sensitive_cost():
    y = pd.Series([0, 0])
    amount = pd.Series([100.0, 200.0])
    decisions = pd.Series(["REVIEW", "APPROVE"])
    result = evaluate_policy(y, amount, decisions, CostConfig(
        reviewer_legitimate_false_decline_rate=0.02,
        false_decline_cost=10.0,
        false_decline_margin_rate=0.10,
        customer_lifetime_value_cost=3.0,
    ))
    assert result["expected_false_declines"] == pytest.approx(0.02)
    assert result["false_decline_cost"] == pytest.approx(3.24)


def test_chargeback_fee_applies_to_expected_unblocked_fraud():
    y = pd.Series([1, 1])
    amount = pd.Series([100.0, 100.0])
    decisions = pd.Series(["APPROVE", "REVIEW"])
    result = evaluate_policy(y, amount, decisions, CostConfig(
        reviewer_fraud_catch_rate=0.5, chargeback_fee=20.0
    ))
    assert result["chargeback_cost"] == pytest.approx(30.0)


@pytest.mark.parametrize("thresholds", [(0.8, 0.5), (-0.1, 0.9), (0.1, 1.1)])
def test_invalid_thresholds_are_rejected(thresholds):
    with pytest.raises(ValueError):
        assign_decisions(pd.Series([0.2]), *thresholds, 1, "approve")


def test_invalid_capacity_and_overflow_are_rejected():
    with pytest.raises(ValueError):
        assign_decisions(pd.Series([0.2]), 0.1, 0.9, -1, "approve")
    with pytest.raises(TypeError):
        assign_decisions(pd.Series([0.2]), 0.1, 0.9, 1.5, "approve")
    with pytest.raises(ValueError):
        assign_decisions(pd.Series([0.2]), 0.1, 0.9, 1, "ignore")


def test_cost_config_rejects_nan_and_out_of_range_rates():
    with pytest.raises(ValueError):
        CostConfig(false_decline_cost=float("nan")).validate()
    with pytest.raises(ValueError):
        CostConfig(reviewer_fraud_catch_rate=1.1).validate()
