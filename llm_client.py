import os
from zhipuai import ZhipuAI

API_KEY = os.getenv("BIGMODEL_API_KEY")

if not API_KEY:
    raise ValueError("BIGMODEL_API_KEY not found")

client = ZhipuAI(api_key=API_KEY)

def get_llm_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",  # FREE / FAST model
            messages=[
                {
                    "role": "system",
                    "content": "Answer factually. If the answer is unknown or speculative, say 'I am not sure.'"
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
