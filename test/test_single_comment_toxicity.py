from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_zero_shot_module():
    module_path = ROOT / "src" / "zero_shot_evaluate.py"
    spec = spec_from_file_location("zero_shot_evaluate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_comment_toxicity() -> None:
    mod = _load_zero_shot_module()

    model_name = "sshleifer/tiny-gpt2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comment = "You are the worst person ever."

    result = mod.evaluate_toxicity(comment_text=comment, model_name=model_name, device=device)

    print("comment:", comment)
    print("pred:", result["pred"])
    print("score:", result["score"])
    print("lp_pos:", result["lp_pos"])
    print("lp_neg:", result["lp_neg"])

    assert isinstance(result, mod.ZeroShotScorePrediction)
    assert result["task"] == "toxicity"
    assert result["pred"] in result["labels"]
    assert isinstance(result["score"], float)


if __name__ == "__main__":
    test_single_comment_toxicity()
    print("Single-comment toxicity test passed")
