import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Offline LLM Evaluation & Safety System",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
st.title("🧠 Offline LLM Evaluation & Safety System")

st.markdown("""
This system evaluates **manually curated LLM responses** across:

- Accuracy  
- Hallucination  
- Bias & Safety  
- Refusal Compliance  

✅ No live APIs are used  
✅ Mirrors real-world AI evaluation pipelines  
""")

# ----------------------------
# Load Dataset
# ----------------------------
DATA_PATH = "data/prompts.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success("Dataset loaded successfully")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ----------------------------
# Show Raw Data
# ----------------------------
st.subheader("📄 Evaluation Dataset")
st.dataframe(df, use_container_width=True)

# ----------------------------
# Accuracy Evaluation
# ----------------------------
st.subheader("🔍 Accuracy Evaluation")

def compute_accuracy(row):
    if row["response_type"] == "correct":
        return 1.0
    elif row["response_type"] == "wrong":
        return 0.0
    elif row["response_type"] in ["hallucinated", "biased"]:
        return 0.0
    elif row["expected_behaviour"] == "refuse":
        return 0.5
    return 0.0

df["accuracy_score"] = df.apply(compute_accuracy, axis=1)

st.dataframe(
    df[[
        "prompt",
        "response_text",
        "ground_truth",
        "response_type",
        "accuracy_score"
    ]],
    use_container_width=True
)

# ----------------------------
# Aggregate Metrics
# ----------------------------
st.subheader("📊 Aggregate Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Accuracy", round(df["accuracy_score"].mean(), 2))

with col2:
    st.metric("Hallucinations", len(df[df["response_type"] == "hallucinated"]))

with col3:
    st.metric("Biased Responses", len(df[df["response_type"] == "biased"]))

with col4:
    st.metric("Correct Responses", len(df[df["response_type"] == "correct"]))

# ----------------------------
# Metadata View
# ----------------------------
st.subheader("🧾 Response Metadata")

st.dataframe(
    df[[
        "model_provider",
        "model_name",
        "response_date"
    ]],
    use_container_width=True
)

# ----------------------------
# Footer
# ----------------------------
st.caption(
    f"Last evaluated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
