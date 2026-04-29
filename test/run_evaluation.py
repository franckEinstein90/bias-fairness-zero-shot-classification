"""
run_evaluation.py
-----------------
Combined zero-shot scoring + Integrated Gradients explanation script.
Mirrors their.py but wired to your src/ modules.

Usage:
    python scripts/run_evaluation.py
"""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from rich.progress import track

from scripts.load_llm import load_llm
from src.zero_shot_evaluate import score_and_predict
from src.integrated_gradients import integrated_gradients, save_heatmap


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
DATASET = "civil"                              # subfolder name under data/ and results/
INPUT_PATH = f"data/{DATASET}/{DATASET}.parquet"
TEXT_COL = "comment_text"
TASK = "toxicity"
MODEL_NAME = "microsoft/Phi-4-mini-instruct"

MAX_ROWS = 1000   # cap how many rows to score in total
IG_ROWS = 25      # how many of those rows also get IG heatmaps
IG_STEPS = 32     # integration steps (higher = more accurate, slower)
FORCE_FLOAT32 = True
SAVE_HEATMAPS = True

OUTPUT_PATH = f"results/{DATASET}/zs_preds.parquet"
# ---------------------------------------------------------------------------


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluation(
    df: pd.DataFrame,
    text_col: str,
    model_name: str,
    task: str,
    dataset: str,
    max_rows: int | None = None,
    save_heatmaps: bool = False,
    ig_rows: int = 0,
    ig_steps: int = 32,
    force_float32: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Score every row with zero-shot prediction, and run Integrated Gradients
    on the first `ig_rows` rows if `save_heatmaps` is True.

    Returns
    -------
    preds : list[dict]
        One dict per row with keys: idx, pred, score, lp_pos, lp_neg.
    ig_records : list[dict]
        One dict per IG row with keys: idx, heatmap, prompt.
    """
    df = df.copy()
    if max_rows is not None:
        df = df.iloc[:max_rows]

    device = get_device()
    model, tok = load_llm(model_name, device, force_float32=force_float32)

    heatmap_dir = Path(f"results/{dataset}/ig_heatmaps")
    if save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    preds: list[dict] = []
    ig_records: list[dict] = []
    score_diffs: list[float] = []

    for i in track(range(len(df)), description="Scoring"):
        text = str(df.iloc[i][text_col])[:4096]

        # --- Zero-shot score ---
        res = score_and_predict(model, tok, text, task)
        preds.append(
            {
                "idx": i,
                "pred": res["pred"],
                "score": res["score"],
                "lp_pos": res["lp_pos"],
                "lp_neg": res["lp_neg"],
            }
        )

        # --- Integrated Gradients (first ig_rows rows only) ---
        if save_heatmaps and len(ig_records) < ig_rows:
            toks, atts, prompt, ig_score = integrated_gradients(
                model, tok, text, task, steps=ig_steps
            )

            # Sanity-check: IG scalar should be close to scoring scalar
            score_diffs.append(abs(res["score"] - ig_score))

            img_path = heatmap_dir / f"row{i}.png"
            save_heatmap(toks, atts, str(img_path))

            ig_records.append(
                {
                    "idx": i,
                    "heatmap": str(img_path),
                    "prompt": prompt,
                }
            )

    if score_diffs:
        print(f"\nIG alignment  —  max |full_score - ig_score|: {max(score_diffs):.6e}")

    return preds, ig_records


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Loading data from {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  {len(df):,} rows found. Capping at {MAX_ROWS}.")

    preds, ig_records = run_evaluation(
        df=df,
        text_col=TEXT_COL,
        model_name=MODEL_NAME,
        task=TASK,
        dataset=DATASET,
        max_rows=MAX_ROWS,
        save_heatmaps=SAVE_HEATMAPS,
        ig_rows=IG_ROWS,
        ig_steps=IG_STEPS,
        force_float32=FORCE_FLOAT32,
    )

    # --- Save predictions ---
    preds_df = pd.DataFrame(preds)
    preds_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved  →  {OUTPUT_PATH}")

    # --- Save IG records (optional) ---
    if ig_records:
        ig_path = OUTPUT_PATH.replace("zs_preds.parquet", "ig_records.parquet")
        pd.DataFrame(ig_records).to_parquet(ig_path, index=False)
        print(f"IG records saved   →  {ig_path}")


if __name__ == "__main__":
    main()