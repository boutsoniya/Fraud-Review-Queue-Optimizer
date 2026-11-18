import pandas as pd
import pytest

from scripts.train_baseline import (
    attach_identity_features,
    capacity_queue_metrics,
    chronological_split_masks,
)


def test_chronological_split_keeps_equal_timestamps_together():
    dt = pd.Series([1, 2, 2, 3, 4, 5, 6, 7, 8, 9])
    train, validation, test, train_boundary, validation_boundary = chronological_split_masks(dt)
    assert train_boundary < validation_boundary
    assert train.sum() > 0 and validation.sum() > 0 and test.sum() > 0
    assert not (train & validation).any()
    assert not (validation & test).any()
    assert set(dt[train]).isdisjoint(set(dt[validation]))
    assert set(dt[validation]).isdisjoint(set(dt[test]))


def test_chronological_split_rejects_empty_partition_from_timestamp_ties():
    with pytest.raises(ValueError, match="empty partition"):
        chronological_split_masks(pd.Series([1, 1, 1, 1, 1, 1]))


def test_capacity_queue_metrics_compares_score_and_amount_heuristic():
    index = pd.RangeIndex(4)
    result = capacity_queue_metrics(
        pd.Series([1, 0, 1, 0], index=index),
        pd.Series([0.9, 0.8, 0.7, 0.1], index=index),
        pd.Series([10.0, 1000.0, 100.0, 20.0], index=index),
        pd.Series([1, 1, 1, 1], index=index),
        capacity_per_day=1,
    )
    assert result["score"]["fraud_found"] == 1
    assert result["expected_loss_heuristic"]["fraud_found"] == 0
    assert result["score"]["fraud_amount_recall"] == pytest.approx(10 / 110)
    assert result["score"]["reviewed_count"] == 1


def test_identity_join_preserves_transaction_rows_and_reports_coverage():
    transactions = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0]})
    identity = pd.DataFrame({"TransactionID": [1, 3], "id_01": [0.1, 0.2]})
    joined, match_rate = attach_identity_features(transactions, identity)
    assert len(joined) == len(transactions)
    assert match_rate == pytest.approx(2 / 3)
    assert joined["id_01"].isna().sum() == 1


def test_identity_join_rejects_duplicate_ids():
    transactions = pd.DataFrame({"TransactionID": [1, 2]})
    identity = pd.DataFrame({"TransactionID": [1, 1], "id_01": [0.1, 0.2]})
    with pytest.raises(ValueError, match="Identity TransactionID is not unique"):
        attach_identity_features(transactions, identity)
