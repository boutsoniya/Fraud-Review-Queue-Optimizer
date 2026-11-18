import pandas as pd
import pytest

from scripts.audit_ieee_cis import audit


def test_audit_writes_summary_missingness_and_validated_join(tmp_path):
    tx = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [0, 100, 86400, 172800],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        "isFraud": [0, 1, 0, 0],
    })
    identity = pd.DataFrame({"TransactionID": [1, 3], "DeviceInfo": ["A", "B"]})
    tx_path, id_path = tmp_path / "tx.csv", tmp_path / "id.csv"
    tx.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)

    summary = audit(tx_path, id_path, tmp_path / "out")
    assert summary["transaction_rows"] == 4
    assert summary["identity_match_count"] == 2
    assert summary["identity_match_rate"] == pytest.approx(0.5)
    assert summary["relative_day_count"] == 3
    assert summary["chronological_split"]["train_rows"] >= 1
    assert (tmp_path / "out" / "audit_summary.json").exists()
    assert (tmp_path / "out" / "missingness.csv").exists()


def test_audit_rejects_duplicate_identity_ids(tmp_path):
    tx = pd.DataFrame({"TransactionID": [1], "isFraud": [0]})
    identity = pd.DataFrame({"TransactionID": [1, 1]})
    tx_path, id_path = tmp_path / "tx.csv", tmp_path / "id.csv"
    tx.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)
    with pytest.raises(ValueError, match="not unique"):
        audit(tx_path, id_path, tmp_path / "out")
