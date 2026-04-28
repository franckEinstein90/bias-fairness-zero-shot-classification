"""Test function for Google Gemini API queries."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def test_simple_chatgpt_query():
    """Send a simple query to Google Gemini and verify we get a response.
    
    This test:
    1. Loads the Google API key from .env file
    2. Configures the Google Generative AI client
    3. Sends a simple query to Gemini
    4. Verifies we get a valid response
    """
    # Load environment variables from .env
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")  # Using existing Google API key
    assert api_key is not None, "API key not found in .env file"
    
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
    
    return answer


if __name__ == "__main__":
    result = test_simple_chatgpt_query()
    print(f"\nTest passed! Response: {result}")
