import streamlit as st

st.title("About")
st.markdown(
    """
## Bias & Fairness — Zero-Shot Classification

This tool evaluates text for toxicity using zero-shot language model classification.

### How it works

Toxicity is scored by comparing the log-probability a model assigns to a *toxic*
completion against a *non-toxic* completion given a structured prompt.  A positive
score means the model considers the input more likely to be toxic.

### Integrated Gradients

When a local (CPU/CUDA) model is selected, **Integrated Gradients** can attribute
the model's score back to individual input tokens, highlighting which words drove
the prediction.

### Devices

| Device | Description |
|--------|-------------|
| `cuda` | GPU inference (fast) |
| `cpu`  | CPU inference (slow) |
| `api`  | Google Gemini API — no local model required |

### Dataset

Comments are drawn from the
[`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments)
dataset hosted on Hugging Face.
"""
)
