import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai


def load_api_key() -> str:
    """Load a Gemini-compatible API key from the project .env file or process env."""
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


def list_gemini_models():
    """Return all available Gemini models."""
    api_key = load_api_key()

    # Configure Google Generative AI client
    client = genai.Client(api_key=api_key)

    # List available models
    models = list(client.models.list())

    return models