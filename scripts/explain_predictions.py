"""Explain model risk scores with SHAP for a scored transaction.

This module is intentionally optional: SHAP explanations are computed only when
requested, and use the same model artifact and feature schema as training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_model_bundle(path: Path):
    bundle = joblib.load(path)
    required = {"model", "features", "categorical_features"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model artifact missing fields: {sorted(missing)}")
    return bundle


def prepare_features(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    x = frame.drop(columns=["TransactionID", "isFraud"], errors="ignore")
    features = bundle["features"]
    missing = set(features).difference(x.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    x = x.loc[:, features].copy()
    for column in bundle["categorical_features"]:
        if column in x:
            x[column] = x[column].astype("category")
    return x


def explain_rows(frame: pd.DataFrame, model_path: Path, row_indices: list[int],
                 background_size: int = 100, max_display: int = 12) -> dict:
    import shap

    bundle = load_model_bundle(model_path)
    x = prepare_features(frame, bundle)
    if not row_indices:
        raise ValueError("Provide at least one row index to explain.")
    if min(row_indices) < 0 or max(row_indices) >= len(x):
        raise IndexError("Explanation row index outside input data.")
    background = x.iloc[:min(background_size, len(x))]
    samples = x.iloc[row_indices]
    explainer = shap.TreeExplainer(bundle["model"], data=background,
                                   feature_perturbation="interventional",
                                   model_output="raw")
    values = explainer(samples)
    result = []
    for position, row_index in enumerate(row_indices):
        sv = values[position]
        contributions = pd.DataFrame({
            "feature": samples.columns,
            "value": samples.iloc[position].astype(str).values,
            "shap_value": sv.values,
        })
        contributions = contributions.reindex(
            contributions.shap_value.abs().sort_values(ascending=False).index
        ).head(max_display)
        result.append({
            "row_index": int(row_index),
            "transaction_id": (
                int(frame.iloc[row_index]["TransactionID"])
                if "TransactionID" in frame else None
            ),
            "base_value": float(sv.base_values),
            "model_output": float(bundle["model"].predict(samples.iloc[[position]])[0]),
            "top_contributions": contributions.to_dict(orient="records"),
        })
    return {"explanation_type": "SHAP additive contributions to raw model score",
            "rows": result,
            "caution": "SHAP values explain the model score, not causal fraud drivers."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV containing model features")
    parser.add_argument("--model", type=Path,
                        default=Path("models/fraud_lgbm_identity.joblib"))
    parser.add_argument("--rows", type=int, nargs="+", required=True,
                        help="Zero-based row indices to explain")
    parser.add_argument("--output", type=Path, default=Path("reports/shap_explanations.json"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    report = explain_rows(frame, args.model, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
