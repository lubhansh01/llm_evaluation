import os
import pandas as pd
import streamlit as st

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

# Load dataset safely
DATA_PATH = os.path.join("data", "prompts.csv")

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please check data/prompts.csv")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.success("Dataset loaded successfully")
st.dataframe(df)
