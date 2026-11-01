import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same folder as this Python script
env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(env_file)

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        f"OPENROUTER_API_KEY not found. Check: {env_file}"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

try:
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": "Explain chain-of-thought prompting in two sentences."
            }
        ],
        temperature=0.0,
        max_tokens=150,
    )
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")