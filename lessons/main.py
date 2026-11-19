"""
GenAI pipeline you are learning:

    Python  ->  LLM API (Groq)  ->  Prompt  ->  Structured JSON

This file is a working example of that pipeline.
Read the comments in order. They explain WHAT each part does and WHY.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------------------------
# 1) Load the secret key from a .env file
# ---------------------------------------------------------------------------
# Never paste an API key into code. Anyone with the key can use your quota.
# load_dotenv() reads GROQ_API_KEY from the .env file into the environment.
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Copy .env.example to .env and paste your key."
    )


# ---------------------------------------------------------------------------
# 2) Create the Groq client
# ---------------------------------------------------------------------------
# The client is a Python object that talks to Groq over HTTPS.
# You do not run Llama on your laptop. Groq hosts the model in the cloud.
# Your code only sends text and receives text.
client = Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# 3) Choose the model
# ---------------------------------------------------------------------------
# llama-3.1-8b-instant is Meta's Llama 3.1 with 8 billion parameters.
# "instant" on Groq means it is optimized for very fast replies.
# Smaller models are cheaper/faster; larger ones are usually stronger at
# complex reasoning. 8B is a good starting size for learning.
MODEL = "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# 4) Write the prompt
# ---------------------------------------------------------------------------
# An LLM does not "know" your app. It only sees the messages you send.
#
# Two common roles:
#   system  = rules / personality / output format  (how to behave)
#   user    = the actual task or input text        (what to do this time)
#
# For structured JSON, the prompt MUST describe the JSON shape.
# Groq JSON mode guarantees valid JSON syntax, but YOU still tell the
# model which keys you want.
SYSTEM_PROMPT = """
You extract structured data from unstructured text.

Return ONLY a JSON object with this exact shape:
{
  "product": string,
  "sentiment": "positive" | "negative" | "mixed" | "neutral",
  "rating_guess": integer from 1 to 5,
  "key_points": array of short strings,
  "summary": string
}

Do not add markdown. Do not add extra keys.
""".strip()

USER_TEXT = """
I bought the AeroBrew coffee maker last month. It heats up fast and the
espresso is decent, but the milk frother is noisy and broke after two weeks.
Customer support was helpful and sent a replacement part quickly.
""".strip()


def ask_llm_for_json(user_text: str) -> dict:
    """
    Send a prompt to Groq and parse the model's reply as JSON.

    This is the heart of the pipeline:
      Python dict/string  ->  HTTP request  ->  model  ->  JSON string  ->  Python dict
    """

    # chat.completions.create is the standard "chat" API used by Groq,
    # OpenAI, and many other providers. Same idea, different host.
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        # JSON Object Mode: the reply must be valid JSON, not a paragraph.
        # llama-3.1-8b-instant supports this. It does NOT guarantee that
        # every key matches your schema; the prompt still matters.
        response_format={"type": "json_object"},
        # temperature: 0.0 = more deterministic, higher = more creative.
        # For extraction/JSON, keep it low so the structure stays stable.
        temperature=0.2,
    )

    # The SDK wraps the HTTP response. The generated text lives here:
    raw_text = completion.choices[0].message.content

    # The model returns a STRING that looks like JSON.
    # json.loads turns that string into a real Python dict.
    return json.loads(raw_text)


if __name__ == "__main__":
    print("Sending prompt to Groq...\n")
    result = ask_llm_for_json(USER_TEXT)

    print("Structured JSON (Python dict):\n")
    print(json.dumps(result, indent=2))

    # Once it is a dict, you can use it like any other Python data.
    print("\nQuick access examples:")
    print("  product   =", result.get("product"))
    print("  sentiment =", result.get("sentiment"))
    print("  rating    =", result.get("rating_guess"))
