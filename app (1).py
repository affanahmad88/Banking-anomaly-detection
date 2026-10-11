import sqlite3
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score
)

st.set_page_config(
    page_title="Banking Anomaly Monitor",
    page_icon="🏦",
    layout="wide"
)

st.title("Banking Transaction Anomaly Monitor")
st.caption(
    "Machine-learning prototype using public credit-card data. "
    "Alerts require human review and do not prove fraud."
)

uploaded = st.file_uploader(
    "Upload the original creditcard.csv dataset",
    type=["csv"]
)

if uploaded is None:
    st.info(
        "Upload creditcard.csv from the Kaggle Credit Card Fraud "
        "Detection dataset to begin."
    )
    st.markdown(
        "[Get the dataset on Kaggle]"
        "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)"
    )
    st.stop()

@st.cache_data
def load_data(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

try:
    df = load_data(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

required = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"Required columns are missing: {missing}")
    st.stop()

has_labels = "Class" in df.columns

df = df.copy()
df["source_row"] = np.arange(len(df))
df["LogAmount"] = np.log1p(
    pd.to_numeric(df["Amount"], errors="coerce").clip(lower=0)
)

feature_cols = [f"V{i}" for i in range(1, 29)]
feature_cols += ["Time", "LogAmount"]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if has_labels:
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols).copy()

if has_labels:
    df = df.dropna(subset=["Class"]).copy()
    df = df[df["Class"].isin([0, 1])].copy()
    df["Class"] = df["Class"].astype(int)

if len(df) < 100:
    st.error("Not enough valid transactions to train the model.")
    st.stop()

st.sidebar.header("Model settings")
contamination = st.sidebar.slider(
    "Expected anomaly proportion",
    min_value=0.001,
    max_value=0.05,
    value=0.002,
    step=0.001,
    format="%.3f"
)

max_training = st.sidebar.selectbox(
    "Maximum training transactions",
    [10000, 25000, 50000],
    index=2
)

X = df[feature_cols]
if has_labels:
    normal = df["Class"].eq(0)
    X_normal = X.loc[normal]
else:
    X_normal = X

if len(X_normal) < 50:
    st.error("Not enough normal transactions for model training.")
    st.stop()

if len(X_normal) > max_training:
    X_fit = X_normal.sample(
        n=max_training, random_state=42
    )
else:
    X_fit = X_normal

with st.spinner("Training Isolation Forest and scoring transactions..."):
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_fit)
    df["Anomaly Flag"] = (
        model.predict(X) == -1
    ).astype(int)

df["Alert Status"] = np.where(
    df["Anomaly Flag"].eq(1),
    "Flagged for Review",
    "Normal"
)


# Calculate reference thresholds from the loaded dataset
amount_cutoff = df["Amount"].quantile(0.99)

# Explain alerts using transparent, understandable rules
def explain(row):
    reasons = []

    # Rule 1: unusually large amount
    if row["Amount"] >= amount_cutoff:
        reasons.append(
            f"High transaction amount "
            f"({row['Amount']:.2f}; at or above the dataset's 99th percentile)"
        )

    # Rule 2: unusual combination of transformed features
    if row["Anomaly Flag"] == 1:
        reasons.append(
            "Machine learning detected an unusual combination "
            "of transaction features"
        )

    if not reasons:
        return "No anomaly flagged"

    return "; ".join(reasons)


df["Alert Reason"] = df.apply(explain, axis=1)

# Rebuild alerts after generating explanations
alerts = df[df["Anomaly Flag"] == 1].copy()

st.caption(
    f"High-amount reference threshold: {amount_cutoff:.2f}. "
    "This is a dataset-relative threshold, not a bank policy."
)

def explain(row):
    reasons = []
    if row["Amount"] >= amount_cutoff:
        reasons.append("Amount is in the dataset's top 1%")
    if not reasons:
        reasons.append(
            "Unusual combination of transaction features"
        )
    return "; ".join(reasons)

df["Alert Reason"] = df.apply(
    lambda row: explain(row) if row["Anomaly Flag"] == 1 else "",
    axis=1
)

alerts = df[df["Anomaly Flag"] == 1].copy()

st.sidebar.success("Model trained successfully")
st.sidebar.write(f"Training examples: {len(X_fit):,}")

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions analyzed", f"{len(df):,}")
c2.metric("Flagged for review", f"{len(alerts):,}")
c3.metric(
    "Alert rate",
    f"{100 * len(alerts) / len(df):.2f}%"
)
c4.metric(
    "Training examples",
    f"{len(X_fit):,}"
)

st.subheader("Transaction overview")

chart1, chart2 = st.columns(2)

with chart1:
    fig, ax = plt.subplots()
    df["Alert Status"].value_counts().reindex(
        ["Normal", "Flagged for Review"], fill_value=0
    ).plot(kind="bar", ax=ax)
    ax.set_ylabel("Transactions")
    ax.set_xlabel("")
    ax.set_title("Normal vs flagged")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with chart2:
    fig, ax = plt.subplots()
    ax.hist(df["Amount"].clip(upper=df["Amount"].quantile(0.99)),
            bins=30)
    ax.set_title("Transaction amounts (up to 99th percentile)")
    ax.set_xlabel("Amount in dataset currency units")
    ax.set_ylabel("Transactions")
    st.pyplot(fig)
    plt.close(fig)

if has_labels:
    st.subheader("Model evaluation against available labels")
    y_true = df["Class"]
    y_pred = df["Anomaly Flag"]

    e1, e2, e3 = st.columns(3)
    e1.metric(
        "Precision",
        f"{precision_score(y_true, y_pred, zero_division=0):.3f}"
    )
    e2.metric(
        "Recall",
        f"{recall_score(y_true, y_pred, zero_division=0):.3f}"
    )
    e3.metric(
        "F1 score",
        f"{f1_score(y_true, y_pred, zero_division=0):.3f}"
    )

    st.caption(
        "These metrics evaluate anomaly flags against the supplied labels. "
        "Isolation Forest was not trained using those fraud labels."
    )


st.subheader("Alert investigation")

reason_filter = st.selectbox(
    "Filter alerts by explanation",
    [
        "All alerts",
        "High amount + ML flag",
        "ML flag only"
    ]
)

filtered = alerts.copy()

# Identify transactions at or above the 99th percentile amount
has_high_amount = filtered["Amount"] >= amount_cutoff

if reason_filter == "High amount + ML flag":
    filtered = filtered[has_high_amount]

elif reason_filter == "ML flag only":
    filtered = filtered[~has_high_amount]

st.write(f"Matching alerts: {len(filtered):,}")

display_cols = [
    "source_row",
    "Time",
    "Amount",
    "Alert Status",
    "Alert Reason"
]

if has_labels:
    display_cols.append("Class")

st.dataframe(
    filtered[display_cols].head(1000),
    use_container_width=True,
    hide_index=True
)

csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered alerts",
    data=csv_bytes,
    file_name="banking_alerts_explained.csv",
    mime="text/csv"
)
