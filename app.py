"""Local Streamlit decision-support demo for fraud review operations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.policy import CostConfig, assign_daily_decisions, derive_relative_day, evaluate_policy

st.set_page_config(page_title="Fraud Review Queue Optimizer", page_icon="🛡️", layout="wide")

st.title("Fraud Review Queue Optimizer")
st.caption("A decision-support prototype: turn model risk scores into capacity-aware approve / review / decline actions.")

with st.sidebar:
    st.header("Policy controls")
    st.caption("All thresholds apply to model risk scores, not calibrated probabilities.")
    approve_threshold = st.slider("Auto-approve below", 0.0, 0.99, 0.02, 0.01)
    decline_threshold = st.slider("Auto-decline at / above", 0.01, 1.0, 0.30, 0.01)
    if approve_threshold >= decline_threshold:
        st.error("Approve threshold must be lower than decline threshold.")
        st.stop()

    st.subheader("Review capacity")
    reviewers = st.number_input("Reviewers", min_value=0, max_value=500, value=5, step=1)
    reviews_per_reviewer = st.number_input(
        "Reviews per reviewer per day", min_value=0, max_value=1000, value=20, step=5
    )
    daily_capacity = int(reviewers * reviews_per_reviewer)
    st.metric("Daily review capacity", f"{daily_capacity:,}")

    ranking = st.selectbox(
        "Queue ranking",
        options=["score", "expected_loss"],
        format_func=lambda x: "Highest fraud score" if x == "score" else "Score × transaction amount",
    )
    overflow = st.selectbox(
        "When review queue exceeds capacity",
        options=["approve", "decline"],
        format_func=lambda x: "Approve overflow" if x == "approve" else "Decline overflow",
    )

    st.subheader("Illustrative cost assumptions")
    fraud_loss_rate = st.number_input("Fraud loss rate", min_value=0.0, value=1.0, step=0.1)
    false_decline_cost = st.number_input("Cost per false decline", min_value=0.0, value=5.0, step=1.0)
    review_cost = st.number_input("Cost per manual review", min_value=0.0, value=1.0, step=0.5)
    reviewer_catch = st.slider("Reviewer fraud catch rate", 0.0, 1.0, 0.90, 0.01)
    reviewer_false_decline = st.slider(
        "Reviewer false-decline rate on legitimate cases", 0.0, 1.0, 0.02, 0.01
    )
    margin_rate = st.slider("False-decline margin impact rate", 0.0, 1.0, 0.0, 0.01)
    clv_cost = st.number_input("Customer lifetime value impact per false decline", min_value=0.0, value=0.0, step=1.0)
    chargeback_fee = st.number_input("Chargeback fee per fraud passing through", min_value=0.0, value=0.0, step=1.0)

def make_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_days, per_day = 10, 250
    n = n_days * per_day
    fraud = (rng.random(n) < 0.035).astype("int8")
    scores = np.where(
        fraud == 1,
        rng.beta(5.0, 2.0, n),
        rng.beta(0.65, 12.0, n),
    )
    amount = np.clip(rng.lognormal(mean=4.2, sigma=1.0, size=n), 2, 5000)
    seconds = np.repeat(np.arange(n_days), per_day) * 86400 + rng.integers(0, 86399, n)
    return pd.DataFrame({
        "TransactionDT": seconds,
        "TransactionAmt": amount.round(2),
        "fraud_score": scores.round(5),
        "isFraud": fraud,
    })

def normalize_input(uploaded: pd.DataFrame) -> pd.DataFrame:
    df = uploaded.copy()
    aliases = {
        "probability": "fraud_score",
        "score": "fraud_score",
        "amount": "TransactionAmt",
        "transaction_amount": "TransactionAmt",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    required = {"fraud_score", "TransactionAmt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "CSV must contain fraud_score (or probability/score) and TransactionAmt (or amount). "
            f"Missing: {', '.join(sorted(missing))}."
        )
    df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce")
    df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce")
    if "TransactionDT" in df.columns:
        df["relative_day"] = derive_relative_day(df)
    elif "relative_day" in df.columns:
        df["relative_day"] = pd.to_numeric(df["relative_day"], errors="coerce")
    else:
        df["relative_day"] = 0
    if df[["fraud_score", "TransactionAmt", "relative_day"]].isna().any().any():
        raise ValueError("Score, amount, and day columns cannot contain missing or non-numeric values.")
    if not df["fraud_score"].between(0, 1).all():
        raise ValueError("fraud_score values must be between 0 and 1.")
    if (df["TransactionAmt"] < 0).any() or (df["relative_day"] < 0).any():
        raise ValueError("Transaction amounts and relative_day values must be non-negative.")
    if "isFraud" in df.columns:
        df["isFraud"] = pd.to_numeric(df["isFraud"], errors="coerce")
        if df["isFraud"].isna().any() or not df["isFraud"].isin([0, 1]).all():
            raise ValueError("Optional isFraud column must contain only 0/1 labels.")
        df["isFraud"] = df["isFraud"].astype("int8")
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    return df

if "demo_data" not in st.session_state:
    st.session_state.demo_data = make_demo_data()

st.info(
    "Use the built-in synthetic demo or upload a CSV of scored transactions. "
    "Do not upload names, card numbers, emails, or other personal/payment information."
)
left, right = st.columns([1, 2])
with left:
    if st.button("Load synthetic demo data", use_container_width=True):
        st.session_state.demo_data = make_demo_data()
with right:
    uploaded_file = st.file_uploader("Upload scored transactions (CSV)", type=["csv"])

try:
    if uploaded_file is not None:
        data = normalize_input(pd.read_csv(uploaded_file))
        data_source = "Uploaded CSV"
    else:
        data = normalize_input(st.session_state.demo_data)
        data_source = "Synthetic demonstration data"
except Exception as exc:
    st.error(str(exc))
    st.stop()

if len(data) == 0:
    st.warning("The selected dataset is empty.")
    st.stop()

costs = CostConfig(
    fraud_loss_rate=float(fraud_loss_rate),
    false_decline_cost=float(false_decline_cost),
    manual_review_cost=float(review_cost),
    reviewer_fraud_catch_rate=float(reviewer_catch),
    reviewer_legitimate_false_decline_rate=float(reviewer_false_decline),
    false_decline_margin_rate=float(margin_rate),
    chargeback_fee=float(chargeback_fee),
    customer_lifetime_value_cost=float(clv_cost),
)
try:
    decisions = assign_daily_decisions(
        data, "fraud_score", "relative_day",
        float(approve_threshold), float(decline_threshold),
        daily_capacity, overflow, ranking,
        "TransactionAmt" if ranking == "expected_loss" else None,
    )
except Exception as exc:
    st.error(f"Could not apply policy: {exc}")
    st.stop()

result = data.copy()
result["decision"] = decisions
counts = result["decision"].value_counts()
review_count = int(counts.get("REVIEW", 0))
days_count = int(result["relative_day"].nunique())
max_total_reviews = days_count * daily_capacity

st.subheader("Decision overview")
st.caption(f"{data_source} · {len(result):,} transactions · {days_count:,} relative-day buckets")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Approved", f"{counts.get('APPROVE', 0):,}")
m2.metric("Sent to review", f"{review_count:,}")
m3.metric("Declined", f"{counts.get('DECLINE', 0):,}")
m4.metric("Review capacity used", f"{(review_count / max_total_reviews * 100):.1f}%" if max_total_reviews else "0%")

chart_col, table_col = st.columns([1, 1])
with chart_col:
    st.markdown("**Decision distribution**")
    decision_counts = pd.DataFrame({
        "Transactions": [counts.get("APPROVE", 0), counts.get("REVIEW", 0), counts.get("DECLINE", 0)]
    }, index=["Approve", "Review", "Decline"])
    st.bar_chart(decision_counts)
with table_col:
    st.markdown("**Daily review workload**")
    daily = result.groupby("relative_day").agg(
        reviewed=("decision", lambda s: int(s.eq("REVIEW").sum())),
        capacity=("decision", lambda s: daily_capacity),
    )
    st.bar_chart(daily[["reviewed", "capacity"]])

if "isFraud" in result.columns:
    st.subheader("Offline policy economics")
    st.caption("Labels are present, so this is a retrospective estimate. All reviewer and cost inputs remain assumptions.")
    try:
        metrics = evaluate_policy(
            result["isFraud"], result["TransactionAmt"], result["decision"], costs
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estimated total cost", f"{metrics['total_cost']:,.2f}")
        k2.metric("Expected fraud loss", f"{metrics['fraud_loss']:,.2f}")
        k3.metric("False-decline cost", f"{metrics['false_decline_cost']:,.2f}")
        k4.metric("Manual review cost", f"{metrics['manual_review_cost']:,.2f}")
        st.write({
            "Fraud sent to review": metrics["fraud_reviewed_count"],
            "Fraud approved": metrics["fraud_approved_count"],
            "Fraud declined": metrics["fraud_declined_count"],
            "Legitimate transactions declined": metrics["legitimate_declined_count"],
            "Expected false declines from manual review": round(
                metrics["expected_reviewed_legitimate_false_declines"], 2
            ),
        })
    except Exception as exc:
        st.warning(f"Could not calculate offline economics: {exc}")
else:
    st.info("Add an optional isFraud column (0/1) to a retrospective labelled dataset to calculate offline cost metrics.")

st.subheader("Review queue")
queue = result.loc[result["decision"].eq("REVIEW")].copy()
queue["priority_value"] = (
    queue["fraud_score"] if ranking == "score"
    else queue["fraud_score"] * queue["TransactionAmt"]
)
queue = queue.sort_values("priority_value", ascending=False, kind="mergesort")
display_cols = [
    c for c in ["TransactionID", "relative_day", "TransactionAmt", "fraud_score", "priority_value", "decision"]
    if c in queue.columns
]
st.dataframe(queue[display_cols].head(500), use_container_width=True, hide_index=True)
st.caption("Showing at most 500 review rows. The downloadable file contains all decisions.")
st.download_button(
    "Download decisions CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="fraud_review_decisions.csv",
    mime="text/csv",
)
