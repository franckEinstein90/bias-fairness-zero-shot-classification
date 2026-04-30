import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)



def load_llm(
    model_name: str,
    device: torch.device,
    force_float32: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a pre-trained LLM model and its corresponding tokenizer.

    Configures the model for evaluation, moves it to the specified device,
    and sets the appropriate data type. It also ensures a padding token is
    assigned if missing.

    Parameters
    ----------
    model_name : str
        The Hugging Face model identifier to load.
    device : torch.device
        The device (CPU or CUDA) to load the model onto.
    force_float32 : bool, default False
        If True, forces the model into float32 precision regardless of device.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizerBase]
        A pair containing (model, tokenizer).
    """
    # Load tokenizer + model; float32 optional for stable grads (LayerNorm FP16 issues)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # Ensure padding token exists
    load_dtype = torch.float32 if (force_float32 or device.type != "cuda") else torch.float16
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    # Some models crash returning big intermediates with cache on; harmless to disable
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # Disable KV cache (not needed for scoring / IG)
    return model, tok
