from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_llm import load_llm


def main() -> None:
    # Tiny model keeps this test lightweight while validating load/generate path.
    model_name = "sshleifer/tiny-gpt2"
    device = torch.device("cpu")

    print("STATUS: starting model load")
    model, tok = load_llm(model_name=model_name, device=device, force_float32=True)
    print(f"STATUS: connected to model '{model_name}' on device '{device.type}'")
    print(f"STATUS: tokenizer ready (vocab size={tok.vocab_size})")

    question = "repeat the word 'hello' three times"
    inputs = tok(question, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    print(f"STATUS: generation completed (tokens={output_ids.shape[-1]})")

    decoded = tok.decode(output_ids[0], skip_special_tokens=True)
    answer = decoded[len(question) :].strip() if decoded.startswith(question) else decoded.strip()

    print("Question:", question)
    print("Answer:", answer)
    print("STATUS: connection confirmed")

    # This smoke test confirms service/model connectivity even if answer quality is poor.
    assert model is not None and tok is not None, "Connection failed: model or tokenizer unavailable"
    assert output_ids.numel() > 0, "Connection failed: model did not return generated tokens"


if __name__ == "__main__":
    main()
