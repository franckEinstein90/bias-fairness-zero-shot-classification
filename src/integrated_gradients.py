from typing import Any
import numpy.typing as npt
import torch
from torch.nn import functional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.constants import LABELS
from src.utils import format_prompt




def extract_text_token_span(
    tok: Any,
    prompt: str,
    input_ids: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    """Return only the tokens that belong to the raw text between Text: and Label:."""
    text_marker = "Text: "
    label_marker = "\nLabel:"
    text_start = prompt.index(text_marker) + len(text_marker)
    text_end = prompt.rindex(label_marker)

    enc_with_offsets = tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = enc_with_offsets["offset_mapping"][0].tolist()
    keep_positions = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start and start >= text_start and end <= text_end
    ]

    if not keep_positions:
        return tok.convert_ids_to_tokens(input_ids[0].tolist()), torch.arange(input_ids.shape[1])

    kept_ids = input_ids[0, keep_positions].tolist()
    return tok.convert_ids_to_tokens(kept_ids), torch.tensor(keep_positions, device=input_ids.device)


def integrated_gradients(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    text: str,
    task: str,
    steps: int = 32,
) -> tuple[list[str], npt.NDArray[np.floating[Any]], str, float]:
    """
    Perform Integrated Gradients to identify which tokens influenced the model output.

    Calculates the importance of each token by integrating gradients along
    a path from a baseline (zero) input to the actual token embeddings.

    Parameters
    ----------
    model : PreTrainedModel
        The loaded LLM.
    tok : Any
        The tokenizer.
    text : str
        The text to explain.
    task : str
        The classification task.
    steps : int, default 32
        The number of integration steps to perform (higher is more accurate).

    Returns
    -------
    tuple[list[str], npt.NDArray, str, float]
        A tuple of (tokens, attribution_scores, full_prompt, model_score).
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Gradient checkpointing halves activation memory during the IG backward pass.
    gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
    if hasattr(model, "gradient_checkpointing_enable") and not gc_was_enabled:
        model.gradient_checkpointing_enable()

    # IG only needs gradients w.r.t. prompt embeddings, not model parameters.
    # Freezing params avoids allocating full parameter-gradient buffers each step.
    param_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    prompt = format_prompt(text, task)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    emb_layer = model.get_input_embeddings()
    x = emb_layer(input_ids).detach().to(model_dtype)
    x0 = torch.zeros_like(x)

    pos_label, neg_label = LABELS[task]
    pos_ids = tok.encode(pos_label, add_special_tokens=False)
    neg_ids = tok.encode(neg_label, add_special_tokens=False)

    def full_label_logprob(emb: torch.Tensor, label_ids: list[int]) -> torch.Tensor:
        """Compute log p(full_label | prompt) using teacher forcing.

        Only prompt embeddings get gradients.
        """
        cur_emb = emb
        cur_attn = attn.clone()
        total_logprob = torch.tensor(0.0, device=device, dtype=model_dtype)

        for lid in label_ids:
            out = model(inputs_embeds=cur_emb, attention_mask=cur_attn, use_cache=False)
            logits = out.logits[:, -1, :]
            logprobs = functional.log_softmax(logits, dim=-1)
            total_logprob = total_logprob + logprobs[0, lid]

            # Append next label token as *constant* embedding
            next_token = torch.tensor([[lid]], device=device)
            next_emb = emb_layer(next_token).to(model_dtype)

            cur_emb = torch.cat([cur_emb, next_emb], dim=1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)],
                dim=1,
            )

        return total_logprob

    def score_fn(emb: torch.Tensor) -> torch.Tensor:
        pos_score = full_label_logprob(emb, pos_ids)
        neg_score = full_label_logprob(emb, neg_ids)
        return pos_score - neg_score

    try:
        # ----- Integrated Gradients -----
        alphas = torch.linspace(0, 1, steps=steps, device=device, dtype=model_dtype).view(-1, 1, 1, 1)

        grads = torch.zeros_like(x)

        for a in alphas:
            emb = (x0 + a * (x - x0)).requires_grad_(True)
            s = score_fn(emb)
            grad = torch.autograd.grad(
                outputs=s,
                inputs=emb,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            grads += grad.detach()
            del emb, s, grad

        avg_grads = grads / steps
        atts = (avg_grads * (x - x0)).sum(dim=-1).squeeze(0)
        atts = atts / (atts.abs().sum() + 1e-8)

        tokens, keep_positions = extract_text_token_span(tok, prompt, input_ids)
        atts = atts[keep_positions]

        with torch.no_grad():
            explained_score = float(score_fn(x))

        return tokens, atts.cpu().numpy(), prompt, explained_score
    finally:
        if hasattr(model, "gradient_checkpointing_disable") and not gc_was_enabled:
            model.gradient_checkpointing_disable()
        for p, req_grad in zip(model.parameters(), param_requires_grad):
            p.requires_grad_(req_grad)

def save_heatmap(tokens: list[str], atts: npt.NDArray[np.floating[Any]], out_path: str) -> None:
    """
    Create and save a visualization of the token attribution scores.

    Parameters
    ----------
    tokens : list[str]
        List of tokens extracted from the text.
    atts : npt.NDArray
        Normalization attribution scores for each token.
    out_path : str
        File path where the plot image will be saved.

    Returns
    -------
    None
    """
    # Simple bar plot; token strings lightly cleaned for readability
    plt.figure(figsize=(max(6, len(tokens) * 0.2), 2.8))
    plt.bar(range(len(tokens)), atts)
    plt.xticks(
        range(len(tokens)),
        [t.replace("Ġ", "▯") for t in tokens],
        rotation=70,
        ha="right",
    )
    plt.ylabel("IG attribution")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()