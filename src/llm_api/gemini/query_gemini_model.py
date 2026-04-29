"""Google Gemini query helpers for API-backed toxicity scoring."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

from src.utils import format_prompt


LABELS = {
    "toxicity": ["toxic", "non-toxic"],
    "hate": ["hateful", "not hateful"],
    "offense": ["offensive", "not offensive"],
}


def load_api_key() -> str:
    """Load an API key from project .env file or process environment."""
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        api_key = os.getenv(key_name)
        if api_key:
            return api_key

    raise AssertionError(
        "API key not found. Set GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in .env."
    )


def query_gemini_model(model_name: str, query: str) -> str:
    """Send one query to Gemini and return text response."""
    api_key = load_api_key()

    # Configure Google Generative AI client
    client = genai.Client(api_key=api_key)

    # Send a simple query
    response = client.models.generate_content(
        model=model_name,
        contents=query,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=100,
        ),
    )

    # Extract the response
    answer = response.text

    # Verify we got a valid response
    assert answer is not None, "No response from Gemini"
    assert len(answer) > 0, "Empty response from Gemini"

    return answer.strip()


def _parse_label_answer(answer: str, task: str) -> tuple[str, float]:
    """Map Gemini text answer to one of the binary task labels and a signed score."""
    pos_label, neg_label = LABELS[task]
    normalized = answer.strip().lower().replace("_", "-")
    first_token = normalized.split()[0] if normalized else ""

    if first_token.startswith(pos_label):
        return pos_label, 1.0
    if first_token.startswith(neg_label):
        return neg_label, -1.0

    if pos_label in normalized and neg_label not in normalized:
        return pos_label, 1.0
    if neg_label in normalized and pos_label not in normalized:
        return neg_label, -1.0

    # Fallback: keep output schema stable, mark uncertain cases as non-positive.
    return neg_label, 0.0


def score_and_predict_gemini(model_name: str, text: str, task: str = "toxicity") -> dict[str, Any]:
    """API-backed zero-shot classification result with the same shape as local scoring."""
    if task not in LABELS:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {list(LABELS)}")

    prompt = format_prompt(text, task)
    answer = query_gemini_model(model_name=model_name, query=prompt)
    pred, score = _parse_label_answer(answer=answer, task=task)
    y_pos, y_neg = LABELS[task]

    return {
        "task": task,
        "prompt": prompt,
        "score": score,
        "pred": pred,
        "labels": (y_pos, y_neg),
        "lp_pos": float("nan"),
        "lp_neg": float("nan"),
        "raw_answer": answer,
    }
