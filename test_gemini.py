
import os
import json
import time
import socket
import random
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(env_file, override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit(
        "Gemini API key not found. "
        "Check GEMINI_API_KEY in your .env file."
    )

# Use a documented Gemini API model.
model = "gemini-3.5-flash-lite"

url = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{model}:generateContent"
)

REQUEST_TIMEOUT = 90
MAX_ATTEMPTS = 4

RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


# --------------------------------------------------
# 2. Prompt and request payload
# --------------------------------------------------

prompt = (
    "Explain why controlled experiments are important when "
    "evaluating large language models. Answer in exactly "
    "two concise sentences."
)

payload = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ],
    "generationConfig": {
        "maxOutputTokens": 150
    }
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    },
    method="POST",
)


# --------------------------------------------------
# 3. Retry helper
# --------------------------------------------------

def retry_delay(attempt):
    """Exponential backoff with a small random delay."""
    return min(
        (2 ** (attempt - 1)) + random.uniform(0, 1),
        30
    )


# --------------------------------------------------
# 4. Send request with automatic retries
# --------------------------------------------------

print("Gemini API key loaded: True")
print("Model:", model)
print("Maximum attempts:", MAX_ATTEMPTS)
print("Sending test prompt...")

start_time = time.perf_counter()
result = None

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        # Successful response.
        break

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8",
            errors="replace"
        )

        if (
            exc.code in RETRYABLE_HTTP_CODES
            and attempt < MAX_ATTEMPTS
        ):
            delay = retry_delay(attempt)

            print(
                f"\nTemporary HTTP {exc.code} "
                f"(attempt {attempt}/{MAX_ATTEMPTS})."
            )
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            continue

        print(f"\nHTTP error: {exc.code}")
        print(error_body)
        raise SystemExit(1)

    except (socket.timeout, TimeoutError):
        if attempt < MAX_ATTEMPTS:
            delay = retry_delay(attempt)

            print(
                f"\nRequest timed out "
                f"(attempt {attempt}/{MAX_ATTEMPTS})."
            )
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            continue

        print(
            f"\nRequest timed out after "
            f"{MAX_ATTEMPTS} attempts."
        )
        print(
            "Check your proxy, network connection, "
            "and Gemini API availability."
        )
        raise SystemExit(1)

    except urllib.error.URLError as exc:
        if attempt < MAX_ATTEMPTS:
            delay = retry_delay(attempt)

            print(
                f"\nConnection error: {exc.reason}"
            )
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            continue

        print(
            "\nConnection failed after "
            f"{MAX_ATTEMPTS} attempts."
        )
        print("Reason:", exc.reason)
        print("Check your proxy settings and connectivity.")
        raise SystemExit(1)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

latency = time.perf_counter() - start_time

if result is None:
    raise SystemExit("No successful API response received.")

candidates = result.get("candidates", [])

if candidates:
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])

    answer = "\n".join(
        part["text"]
        for part in parts
        if isinstance(part.get("text"), str)
    )

    print("\nAPI request: SUCCESS")
    print("Response:")
    print(answer or "(No text returned)")

    if candidate.get("finishReason"):
        print("Finish reason:", candidate["finishReason"])

    if candidate.get("finishMessage"):
        print("Finish message:", candidate["finishMessage"])

else:
    print("\nRequest succeeded, but no response candidates were returned.")
    print(json.dumps(result, indent=2))

print(f"\nTotal latency: {latency:.2f} seconds")

usage = result.get("usageMetadata")

if usage:
    print("\nToken usage:")
    print(json.dumps(usage, indent=2))
