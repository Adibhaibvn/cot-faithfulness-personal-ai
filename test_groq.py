import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key loaded: {api_key[:10]}...")  # shows first 10 chars only

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        temperature=0.0,
        max_tokens=50
    )
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print(f"Error: {e}")