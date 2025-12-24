import streamlit as st
import pandas as pd
from datetime import datetime

from evaluators.accuracy import compute_accuracy
from evaluators.hallucination import detect_hallucination
from evaluators.bias_safety import detect_bias_safety
from evaluators.confusion import detect_confusion

from scoring.aggregator import aggregate_metrics

st.set_page_config(page_title="Offline LLM Evaluation System", layout="wide")

st.title("🧠 Offline LLM Evaluation & Safety System")

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("data/prompts.csv")
    st.success("Dataset loaded successfully")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# -----------------------------
# Run Evaluations
# -----------------------------
df["accuracy_score"] = df.apply(compute_accuracy, axis=1)
df["hallucination_flag"] = df.apply(detect_hallucination, axis=1)
df["safety_violation"] = df.apply(detect_bias_safety, axis=1)
df["confusion_flag"] = df.apply(detect_confusion, axis=1)

# -----------------------------
# Executive Summary
# -----------------------------
st.subheader("📌 Evaluation Summary")

metrics = aggregate_metrics(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Accuracy", metrics["avg_accuracy"])
col2.metric("Hallucinations", metrics["hallucinations"])
col3.metric("Safety Issues", metrics["safety_issues"])
col4.metric("Confusions", metrics["confusions"])

# -----------------------------
# Detailed Views
# -----------------------------
st.subheader("📊 Full Evaluation Table")
st.dataframe(df, use_container_width=True)

st.subheader("📈 Category-wise Metrics")
category_metrics = (
    df.groupby("category")
      .agg(
          avg_accuracy=("accuracy_score", "mean"),
          hallucinations=("hallucination_flag", "sum"),
          safety_issues=("safety_violation", "sum")
      )
      .reset_index()
)

st.dataframe(category_metrics, use_container_width=True)
