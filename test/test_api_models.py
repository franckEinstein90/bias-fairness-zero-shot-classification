"""Test to list available Google Gemini models."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_api.gemini.list_gemini_models import list_gemini_models


if __name__ == "__main__":
    models = list_gemini_models()
    print(f"\nTotal models: {len(models)}")
