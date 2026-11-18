"""One-off check: which models are available, and do they return clean JSON?"""
import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ---- 1. what can this key call? -------------------------------------
print("--- available models ---")
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(" ", m.name)

# ---- 2. do they return parseable JSON? ------------------------------
PROMPT = (
    "You are a QA triage assistant. Classify this bug report.\n\n"
    "Summary: Authentication not working\n"
    "Description: Performing any operation on a Bitbucket repo causes "
    "errors. Login prompts appear over and over and never stop.\n\n"
    "Respond with ONLY a JSON object, no markdown fences, no explanation:\n"
    '{"severity": "Critical|Major|Minor", '
    '"component": "General|Git|Bitbucket|UX", '
    '"reasoning": "one sentence"}'
)

CANDIDATES = [
    "models/gemini-flash-lite-latest",
    "models/gemma-4-31b-it",
]

for model in CANDIDATES:
    print(f"\n=== {model} ===")
    try:
        r = client.models.generate_content(model=model, contents=PROMPT)
        print("raw:", repr(r.text[:300]))
        cleaned = r.text.strip().removeprefix("```json").removeprefix("```") \
                        .removesuffix("```").strip()
        try:
            print("parsed:", json.loads(cleaned))
        except Exception as e:
            print("JSON PARSE FAILED:", e)
        print("usage:", r.usage_metadata)
    except Exception as e:
        print("CALL FAILED:", type(e).__name__, e)