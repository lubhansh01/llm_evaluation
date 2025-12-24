import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="LLM Evaluation & Safety System",
    layout="wide"
)

st.title("🧠 LLM Evaluation & Safety System")
st.write(
    """
    This application evaluates LLM responses for:
    - Accuracy  
    - Hallucination  
    - Safety / Bias  
    - Confusion  
    """
)

# ----------------------------
# LOAD ENV VARIABLES
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Add it to .env or Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# LOAD DATASET
# ----------------------------
DATA_PATH = "data/prompts.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success("Dataset loaded successfully")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# ----------------------------
# SHOW DATASET
# ----------------------------
st.subheader("📄 Evaluation Dataset")
st.dataframe(df, use_container_width=True)

# ----------------------------
# LLM CALL FUNCTION
# ----------------------------
def get_llm_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer factually. If unknown or speculative, say 'I am not sure.'"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

# ----------------------------
# RUN EVALUATION
# ----------------------------
st.subheader("🔍 Accuracy Evaluation")

if st.button("Run Evaluation"):
    results = []

    with st.spinner("Evaluating prompts..."):
        for _, row in df.iterrows():
            prompt = row["prompt"]
            ground_truth = str(row["ground_truth"]).strip().lower()

            llm_response = get_llm_response(prompt)
            llm_clean = llm_response.lower()

            # Accuracy logic
            if ground_truth == "unknown":
                accuracy = 0.5 if "not sure" in llm_clean else 0
            else:
                accuracy = 1 if ground_truth in llm_clean else 0

            results.append({
                "Prompt": prompt,
                "LLM Response": llm_response,
                "Ground Truth": row["ground_truth"],
                "Accuracy Score": accuracy
            })

    result_df = pd.DataFrame(results)

    st.success("Evaluation completed")

    st.subheader("📊 Accuracy Evaluation Results")
    st.dataframe(result_df, use_container_width=True)

    avg_score = result_df["Accuracy Score"].mean()
    st.metric("✅ Average Accuracy Score", round(avg_score, 2))

    # Download option
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Evaluation Results",
        data=csv,
        file_name="evaluation_results.csv",
        mime="text/csv"
    )
