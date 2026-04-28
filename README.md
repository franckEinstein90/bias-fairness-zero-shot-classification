# Bias & Fairness: Zero-Shot Classification

Use an LLM as a classifier (toxic vs. non-toxic), then compute group-level fairness metrics with a minimal script-first workflow.

## Prerequisites

- Python 3.12+
- `uv` installed

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

From the project root:

```bash
cd /home/coder/bias-fairness-zero-shot-classification
uv venv
uv sync
```

This creates `.venv/` and installs all dependencies from `pyproject.toml`.

## Activate The Virtual Environment

```bash
source .venv/bin/activate
```

After activation, you can run scripts with `python ...`.

## Run Without Activation (Recommended)

You can skip activation and execute everything through `uv run`:

```bash
uv run python scripts/load_dataset.py --dataset civil --stream --take 100 --out data/civil.parquet
```

## Verify Installation

Run the stream smoke test (prints shape, columns, and first 10 rows):

```bash
uv run python test/test_stream_dataset.py
```

## Common Commands

Create a larger dataset sample:

```bash
uv run python scripts/load_dataset.py --dataset civil --stream --take 200000 --out data/civil.parquet
```

Inspect a saved parquet file:

```bash
uv run python scripts/inspect_data.py
```

## Notes

- Hugging Face may print an unauthenticated warning if `HF_TOKEN` is not set. This is expected.
- The project is script-based; notebooks are intentionally out of scope.