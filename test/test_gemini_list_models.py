"""Test to list available Google Gemini models."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def list_available_models():
    """List all available Gemini models."""
    # Load environment variables from .env
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")  # Using existing Google API key
    assert api_key is not None, "API key not found in .env file"
    
    # Configure Google Generative AI client
    client = genai.Client(api_key=api_key)
    
    # List available models
    print("Available Gemini models:")
    models = client.models.list()
    for model in models:
        print(f"  {model.name}")
    
    return models


if __name__ == "__main__":
    models = list_available_models()
    print(f"\nTotal models: {len(list(models))}")
