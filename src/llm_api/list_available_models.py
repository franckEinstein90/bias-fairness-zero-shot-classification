from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location


def list_models() -> list[str]:
    """Return available API model names (currently Gemini)."""
    module_path = Path(__file__).resolve().parent / "gemini" / "list_gemini_models.py"
    spec = spec_from_file_location("list_gemini_models", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    models = module.list_gemini_models()
    names = [model.name for model in models if hasattr(model, "name")]
    return sorted(set(names))
