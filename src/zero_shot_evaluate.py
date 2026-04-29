from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_llm import load_llm


LABELS = {
    # Binary label pairs; first entry treated as "positive" for score sign
    "toxicity": ["toxic", "non-toxic"],
    "hate": ["hateful", "not hateful"],
    "offense": ["offensive", "not offensive"],
}


def format_prompt(text: str, task: str) -> str:
    """Wrap the input text in a task-specific instruction prompt."""
    instruction = {
        "toxicity": "Decide if the following text is toxic or non-toxic. Answer with a single word.\nText: ",
        "hate": "Decide if the text is hateful or not hateful. Answer with a single word.\nText: ",
        "offense": "Decide if the text is offensive or not offensive. Answer with a single word.\nText: ",
    }[task]
    return f"{instruction}{text}\nLabel:"


def label_logprob(model: Any, tok: Any, prompt_ids: torch.Tensor, label_text: str) -> float:
    """Return log P(label_text | prompt_ids) under the causal language model."""
    device = next(model.parameters()).device
    label_ids = tok(label_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = torch.cat([prompt_ids.to(device), label_ids], dim=1)

    with torch.no_grad():
        logits = model(input_ids=full_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)

    prompt_len = prompt_ids.shape[1]
    label_len = label_ids.shape[1]
    token_logps = []
    for i in range(label_len):
        # To score token at absolute position t, use logits from t-1.
        abs_pos = prompt_len + i
        pred_pos = abs_pos - 1
        tok_id = full_ids[0, abs_pos]
        token_logps.append(log_probs[0, pred_pos, tok_id])

    return float(torch.stack(token_logps).sum().item())


def score_and_predict(model: Any, tok: Any, text: str, task: str = "toxicity") -> dict[str, Any]:
    """Predict label and confidence score for one text using a loaded model/tokenizer."""
    if task not in LABELS:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {list(LABELS)}")

    prompt = format_prompt(text, task)
    batch = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    prompt_ids = batch["input_ids"]

    y_pos, y_neg = LABELS[task]
    lp_pos = label_logprob(model, tok, prompt_ids, y_pos)
    lp_neg = label_logprob(model, tok, prompt_ids, y_neg)
    score = lp_pos - lp_neg
    pred = y_pos if score > 0 else y_neg

    return {
        "task": task,
        "prompt": prompt,
        "score": score,
        "pred": pred,
        "labels": (y_pos, y_neg),
        "lp_pos": lp_pos,
        "lp_neg": lp_neg,
    }


def evaluate_toxicity(comment_text: str, model_name: str, device: torch.device) -> dict[str, Any]:
    """Load model via scripts/load_llm and evaluate toxicity for one comment."""
    model, tok = load_llm(model_name=model_name, device=device, force_float32=(device.type != "cuda"))
    return score_and_predict(model=model, tok=tok, text=comment_text, task="toxicity")