import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env vars
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Gemini
genai.configure(api_key=API_KEY)

# ✅ API-key supported model
model = genai.GenerativeModel("gemini-1.0-pro")

def get_llm_response(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"
