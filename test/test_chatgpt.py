"""Test function for Google Gemini API queries."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def load_api_key() -> str:
    """Load an API key from project .env file or process environment."""
    project_root = Path(__file__).resolve().parent.parent
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


def test_simple_chatgpt_query() -> None:
    """Send a simple query to Google Gemini and verify we get a response.
    
    This test:
    1. Loads the Google API key from .env file
    2. Configures the Google Generative AI client
    3. Sends a simple query to Gemini
    4. Verifies we get a valid response
    """
    api_key = load_api_key()
    
    # Configure Google Generative AI client
    client = genai.Client(api_key=api_key)
    
    # Send a simple query
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is 2 + 2?",
        config=genai.types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=100,
        )
    )
    
    # Extract the response
    answer = response.text
    
    # Verify we got a valid response
    assert answer is not None, "No response from Gemini"
    assert len(answer) > 0, "Empty response from Gemini"
    
    print(f"Query: What is 2 + 2?")
    print(f"Gemini Response: {answer}")
    
if __name__ == "__main__":
    test_simple_chatgpt_query()
    print("\nTest passed!")
