import os
import pandas as pd
import streamlit as st

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Offline LLM Evaluation & Safety System",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Custom Styling
# =========================
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #FAFAFA;
}
.metric-card {
    background: #161B22;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 12px rgba(0,0,0,0.3);
}
.metric-title {
    font-size: 14px;
    color: #9BA3AF;
}
.metric-value {
    font-size: 36px;
    font-weight: bold;
}
.green { color: #2ECC71; }
.red { color: #E74C3C; }
.orange { color: #F39C12; }
.blue { color: #3498DB; }
.section-title {
    margin-top: 35px;
    font-size: 22px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.title("🧠 Offline LLM Evaluation & Safety Intelligence Dashboard")

st.markdown("""
This system evaluates **manually curated LLM responses** across:

- Accuracy  
- Hallucination  
- Bias & Safety  
- Refusal Compliance  

_No live APIs are used. This mirrors real-world LLM evaluation pipelines._
""")

# =========================
# Load Dataset
# =========================
DATA_PATH = "data/prompts.csv"

try:
    df = pd.read_csv(DATA_PATH)

    # Normalize column names (CRITICAL)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    st.success("Dataset loaded successfully")

except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# =========================
# Import Evaluators
# =========================
from evaluators.accuracy import compute_accuracy
from evaluators.hallucination import detect_hallucination
from evaluators.bias_safety import detect_bias_safety
from evaluators.confusion import detect_confusion

# =========================
# Run Evaluations
# =========================
df["accuracy_score"] = df.apply(compute_accuracy, axis=1)
df["hallucination_flag"] = df.apply(detect_hallucination, axis=1)
df["safety_violation"] = df.apply(detect_bias_safety, axis=1)
df["confusion_flag"] = df.apply(detect_confusion, axis=1)

# =========================
# Aggregate Metrics
# =========================
avg_accuracy = df["accuracy_score"].mean()
hallucinations = int(df["hallucination_flag"].sum())
safety_issues = int(df["safety_violation"].sum())
confusions = int(df["confusion_flag"].sum())

# =========================
# Metric Cards
# =========================
st.markdown("<div class='section-title'>📌 Evaluation Summary</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Accuracy</div>
        <div class="metric-value blue">{avg_accuracy:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Hallucinations</div>
        <div class="metric-value orange">{hallucinations}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Safety Violations</div>
        <div class="metric-value red">{safety_issues}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Confusions</div>
        <div class="metric-value orange">{confusions}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Full Evaluation Table
# =========================
st.markdown("<div class='section-title'>📋 Full Evaluation Table</div>", unsafe_allow_html=True)

def highlight_row(row):
    if row["safety_violation"] == 1:
        return ["background-color: #4B1F1F"] * len(row)
    if row["hallucination_flag"] == 1:
        return ["background-color: #3B2F1E"] * len(row)
    if row["confusion_flag"] == 1:
        return ["background-color: #1E2A3B"] * len(row)
    return [""] * len(row)

st.dataframe(
    df.sort_values("accuracy_score")
      .style.apply(highlight_row, axis=1),
    use_container_width=True,
    height=520
)

# =========================
# Category-wise Metrics
# =========================
st.markdown("<div class='section-title'>📊 Category-wise Metrics</div>", unsafe_allow_html=True)

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

# =========================
# Research Context
# =========================
st.markdown("""
<div class="section-title">🧪 Evaluation Context</div>
<ul>
<li>Offline, manually curated LLM responses</li>
<li>Explicit hallucination and refusal testing</li>
<li>Bias & safety checks aligned with trust frameworks</li>
<li>Metrics comparable to internal AI evaluation tooling</li>
</ul>
""", unsafe_allow_html=True)
