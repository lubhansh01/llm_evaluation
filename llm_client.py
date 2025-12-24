import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=API_KEY)

# ✅ UPDATED MODEL NAME
model = genai.GenerativeModel("models/gemini-1.5-pro")

def get_llm_response(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ERROR: {e}"
