# Bias & Fairness: Zero-Shot Classification

> **Research question:** Do different interpretability methods give consistent, trustworthy explanations for LLM toxicity predictions — and do explanation failures correlate with fairness failures?

This repository implements and explores the methodology described in the research proposal [*Evaluating Explanation Consistency in Zero-Shot LLM Toxicity Classification: Integrated Gradients vs. Attention*](documentation/proposal.md). It uses zero-shot toxicity classification on the Civil Comments dataset as a testbed to study whether Integrated Gradients and attention-based attribution tell the same story, and whether divergence between them signals model uncertainty or demographic bias.

It combines zero-shot LLM scoring, API-backed inference (Google Gemini), token-level explainability via Integrated Gradients, and an interactive Streamlit front-end for browsing real comments and running live evaluations.

---

## Why This Exists

Most fairness auditing pipelines treat a model as a black box and measure outcome rates across demographic groups. This project takes a complementary approach:

1. **Explanations as objects of evaluation** — rather than accepting model explanations at face value, we treat them as things to be interrogated. Do IG attributions and attention weights agree? When they diverge, does that divergence correlate with borderline predictions or fairness-sensitive identity terms?
2. **Transparency first** — it exposes *how* a model assigns a toxicity score (log-probability difference between the toxic and non-toxic completions of a structured prompt), making the mechanics visible rather than opaque.
3. **Explainability built in** — Integrated Gradients attribution lets you see which tokens in a comment drove the prediction, so you can reason about whether the model is latching onto slurs, identity terms, or genuinely harmful content.
4. **Fairness metrics** — the pipeline computes per-group performance and disparity measures (Statistical Parity Difference, Equal Opportunity Difference) across the demographic identity columns in Civil Comments.
5. **Multi-backend** — the same prompt and evaluation logic runs identically against local Hugging Face models (CPU or CUDA) and against the Google Gemini API, so you can compare a small local model with a large frontier model side-by-side.
6. **Real data** — comments are drawn from the [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) dataset, a large-scale corpus collected specifically for toxicity and fairness research.

---

## How It Works

### Scoring

Given a piece of text and a task (e.g. `toxicity`), the scorer:

1. Wraps the text in a task-specific instruction prompt ending with `Label:`.
2. Computes **log P(toxic | prompt)** and **log P(non-toxic | prompt)** by teacher-forcing each label token through the causal LM.
3. Returns **score = log P(positive) − log P(negative)**. A positive score means the model considers the text more likely to match the toxic label.

### Tasks

| Task | Labels |
|------|--------|
| `toxicity` | `toxic` / `non-toxic` |
| `hate` | `hateful` / `not hateful` |
| `offense` | `offensive` / `not offensive` |

### Integrated Gradients

When using a local model, the front-end can run Integrated Gradients over the prompt embeddings to attribute the score back to individual tokens. The attribution loop:

- Freezes model parameters (gradient buffers are only allocated for the embedding interpolation, not the full model — important for GPU memory).
- Interpolates from a zero baseline to the real embedding in `n` steps.
- Computes `torch.autograd.grad` at each step, accumulates, and normalises.

The result is a per-token signed attribution: red tokens push toward toxic, blue tokens push toward non-toxic.

### API Mode

When the `api` device is selected, inference is routed to the Google Gemini API. The same prompt is sent; the response is parsed with a label-matching heuristic. Token-level log-probs and IG are unavailable in this mode.

---

## Documentation

| Document | Description |
|----------|-------------|
| [documentation/proposal.md](documentation/proposal.md) | Full research proposal — abstract, scoring methodology, IG vs. attention comparison design, fairness metrics (SPD, EOpp), experimental workflow, evaluation plan, and limitations |

---

## Repository Layout

```
src/
  constants/          # Shared LABELS dict (toxicity, hate, offense)
  models/             # ZeroShotScorePrediction dataclass
  utils/              # format_prompt — canonical prompt builder
  zero_shot_evaluate.py  # Core scoring logic (label_logprob, score_and_predict, evaluate_toxicity)
  integrated_gradients.py # IG attribution (extract_text_token_span, integrated_gradients, save_heatmap)
  llm_api/
    gemini/           # Gemini query helpers and model listing
    list_available_models.py

scripts/
  load_dataset.py     # Stream / cache civil_comments from Hugging Face
  load_llm.py         # Load any HF causal LM onto CPU or CUDA
  inspect_data.py     # Quick inspection of saved parquet files
  main.py

front_end_one/
  app.py              # Streamlit application entry point
  sidebar.py          # Device / model / IG settings sidebar
  pages/
    about.py          # About page

test/                 # Smoke tests and evaluation scripts
```

---

## Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A CUDA-capable GPU is optional but recommended for larger models and Integrated Gradients
- A Google Gemini API key (stored in `.env` as `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`) for API mode

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quick Start

```bash
# Create virtual environment and install all dependencies
uv venv
uv sync
```

This creates `.venv/` and installs everything from `pyproject.toml`, including PyTorch with CUDA 12.8 wheels.

---

## Running the Front-End

```bash
uv run streamlit run front_end_one/app.py
```

The app opens a dataset explorer on the left (CivilComments rows, click any to send to the input box) and a zero-shot playground on the right. The sidebar controls which model and device to use for scoring and for IG attribution independently.

---

## Common Commands

Stream and cache a dataset sample:

```bash
uv run python scripts/load_dataset.py --dataset civil --stream --take 100 --out data/civil.parquet
```

Inspect a saved parquet file:

```bash
uv run python scripts/inspect_data.py
```

Smoke-test the dataset streaming:

```bash
uv run python test/test_stream_dataset.py
```

---

## API Key Setup

Create a `.env` file at the project root:

```
GEMINI_API_KEY=your-key-here
```

The app will automatically load it when the `api` device is selected in the sidebar.

---

## Notes

- Hugging Face may print an unauthenticated warning if `HF_TOKEN` is not set. This is expected for public datasets.
- PyTorch is installed from the CUDA 12.8 index; CPU-only machines will still work but IG is slow on large models.
- Notebooks are intentionally out of scope — the project is script and module based.