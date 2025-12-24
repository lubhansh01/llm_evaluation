import os
import pandas as pd
import streamlit as st

from evaluators.accuracy import accuracy_score

# Page config
st.set_page_config(
    page_title="LLM Evaluation & Safety System",
    layout="wide"
)

st.title("🧠 LLM Evaluation & Safety System")

st.markdown("""
This application evaluates LLM responses for:
- Accuracy  
- Hallucination  
- Safety / Bias  
- Confusion  
""")

# Load dataset
DATA_PATH = os.path.join("data", "prompts.csv")

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.success("Dataset loaded successfully")

# Mock LLM response (temporary)
def mock_llm_response(prompt):
    responses = {
        "Who is the CEO of Google?": "Sundar Pichai is the CEO of Google.",
        "What is the capital of France?": "Paris is the capital of France.",
        "Who founded Microsoft?": "Bill Gates founded Microsoft."
    }
    return responses.get(prompt, "I am not sure.")

results = []

for _, row in df.iterrows():
    response = mock_llm_response(row["prompt"])
    acc = accuracy_score(response, row["ground_truth"])

    results.append({
        "Prompt": row["prompt"],
        "LLM Response": response,
        "Ground Truth": row["ground_truth"],
        "Accuracy Score": acc
    })

result_df = pd.DataFrame(results)

st.subheader("🔍 Accuracy Evaluation Results")
st.dataframe(result_df)
