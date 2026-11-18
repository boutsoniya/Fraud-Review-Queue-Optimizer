import pandas as pd

from scripts.run_policy_experiment import select_policy
from src.policy import CostConfig


def test_policy_selection_uses_validation_cost_and_returns_valid_thresholds():
    index = pd.RangeIndex(8)
    y = pd.Series([0, 0, 0, 1, 0, 1, 0, 1], index=index)
    amount = pd.Series([20, 30, 40, 100, 25, 150, 35, 200], index=index)
    scores = pd.Series([0.01, 0.04, 0.08, 0.95, 0.12, 0.75, 0.25, 0.85], index=index)
    day = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=index)

    selected = select_policy(
        y, amount, scores, day, capacity=1, overflow_action="approve",
        costs=CostConfig(), approve_grid=(0.02, 0.10),
        decline_grid=(0.70, 0.90),
    )

    assert selected["approve_threshold"] in (0.02, 0.10)
    assert selected["decline_threshold"] in (0.70, 0.90)
    assert selected["approve_threshold"] < selected["decline_threshold"]
    assert selected["ranking_strategy"] in ("score", "expected_loss")
    assert selected["validation_total_cost"] == selected["validation_metrics"]["total_cost"]


def test_policy_selection_rejects_empty_threshold_search():
    import pytest
    index = pd.RangeIndex(2)
    with pytest.raises(ValueError):
        select_policy(
            pd.Series([0, 1], index=index),
            pd.Series([10.0, 20.0], index=index),
            pd.Series([0.2, 0.8], index=index),
            pd.Series([0, 0], index=index),
            capacity=1, overflow_action="approve", costs=CostConfig(),
            approve_grid=(0.8,), decline_grid=(0.2,),
        )
