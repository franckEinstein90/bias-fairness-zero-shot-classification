"""
test_run_evaluation.py
----------------------
Combined zero-shot scoring + Integrated Gradients explanation script.
Mirrors their.py but wired to your src/ modules.

Usage:
    python test/test_run_evaluation.py
"""

from pathlib import Path
import sys
import os
import shutil
import logging
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from rich.progress import track

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from scripts.load_llm import load_llm
from scripts.load_dataset import load_civil
from src.zero_shot_evaluate import score_and_predict
from src.integrated_gradients import integrated_gradients, save_heatmap


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
DATASET = "civil"                              # subfolder name under data/ and results/
INPUT_PATH = f"data/{DATASET}/{DATASET}.parquet"
TEXT_COL = "comment_text"
TASK = "toxicity"
#MODEL_NAME = "microsoft/Phi-4-mini-instruct"
MODEL_NAME = "distilgpt2"


MAX_ROWS = 100    # cap how many rows to score in total
IG_ROWS = 16       # how many of those rows also get IG heatmaps
IG_STEPS = 16     # integration steps (higher = more accurate, slower)
FORCE_FLOAT32 = False  # use float16 on CUDA to fit large models in GPU memory
SAVE_HEATMAPS = True

RESULTS_ROOT = Path("test/results")
OUTPUT_PATH = str(RESULTS_ROOT / DATASET / "zs_preds.parquet")
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        log.info("Device: %s  |  GPU memory: %.1f GB free / %.1f GB total",
                 torch.cuda.get_device_name(0), free / 1e9, total / 1e9)
    else:
        log.info("Device: cpu (CUDA not available)")
    return device


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
    log.info("Rows to score: %d", len(df))

    device = get_device()
    log.info("Loading model '%s' (force_float32=%s) ...", model_name, force_float32)
    t0 = time.perf_counter()
    model, tok = load_llm(model_name, device, force_float32=force_float32)
    log.info("Model loaded in %.1f s", time.perf_counter() - t0)

    heatmap_dir = RESULTS_ROOT / dataset / "ig_heatmaps"
    if save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    preds: list[dict] = []
    ig_records: list[dict] = []
    score_diffs: list[float] = []
    ig_disabled_due_to_oom = False
    ig_model = model
    ig_tok = tok

    # CUDA-only fallback ladder for IG: progressively reduce workload on OOM.
    ig_attempts: list[tuple[int, int]] = [
        (ig_steps, 4096),
        (max(8, ig_steps // 2), 2048),
        (max(4, ig_steps // 4), 1024),
        (2, 512),
    ]

    for i in track(range(len(df)), description="Scoring"):
        text = str(df.iloc[i][text_col])[:4096]

        # --- Zero-shot score ---
        zero_score_prediction = score_and_predict(model, tok, text, task)
        log.debug("row %d | pred=%-12s score=%+.4f", i, zero_score_prediction["pred"], zero_score_prediction["score"])
        preds.append(
            {
                "idx": i,
                "pred": zero_score_prediction["pred"],
                "score": zero_score_prediction["score"],
                "lp_pos": zero_score_prediction["lp_pos"],
                "lp_neg": zero_score_prediction["lp_neg"],
            }
        )

        # --- Integrated Gradients (first ig_rows rows only) ---
        if save_heatmaps and len(ig_records) < ig_rows and not ig_disabled_due_to_oom:
            ig_done = False
            for attempt_idx, (attempt_steps, text_cap) in enumerate(ig_attempts, start=1):
                try:
                    ig_text = text[:text_cap]
                    if attempt_idx > 1:
                        log.info(
                            "Retrying IG row %d on CUDA (attempt %d/%d, steps=%d, text_cap=%d).",
                            i,
                            attempt_idx,
                            len(ig_attempts),
                            attempt_steps,
                            text_cap,
                        )

                    toks, atts, prompt, ig_score = integrated_gradients(
                        ig_model, ig_tok, ig_text, task, steps=attempt_steps
                    )

                    # Sanity-check: IG scalar should be close to scoring scalar
                    score_diffs.append(abs(zero_score_prediction["score"] - ig_score))

                    img_path = heatmap_dir / f"row{i}.png"
                    save_heatmap(toks, atts, str(img_path))

                    ig_records.append(
                        {
                            "idx": i,
                            "heatmap": str(img_path),
                            "prompt": prompt,
                        }
                    )
                    ig_done = True
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if device.type == "cuda":
                        log.warning(
                            "IG CUDA OOM at row %d (attempt %d/%d, steps=%d, text_cap=%d).",
                            i,
                            attempt_idx,
                            len(ig_attempts),
                            attempt_steps,
                            text_cap,
                        )
                    else:
                        log.warning("IG failed at row %d due to CPU OOM.", i)
                        break

            if not ig_done:
                ig_disabled_due_to_oom = True
                log.warning(
                    "IG disabled after row %d: all CUDA-safe retries failed. Scoring continues on CUDA.",
                    i,
                )

    if score_diffs:
        log.info("IG alignment  —  max |full_score - ig_score|: %.6e", max(score_diffs))

    if ig_records:
        log.info("Heatmaps saved in: %s", heatmap_dir)

    n_toxic = sum(1 for p in preds if p["pred"] != "non-toxic")
    log.info("Scoring complete: %d rows | toxic=%d (%.1f%%)",
             len(preds), n_toxic, 100 * n_toxic / max(len(preds), 1))
    return preds, ig_records


def main() -> None:
    log.info("=== test_run_evaluation started ===")
    log.info("Config: model=%s  task=%s  max_rows=%d  ig_rows=%d",
             MODEL_NAME, TASK, MAX_ROWS, IG_ROWS)

    if RESULTS_ROOT.exists():
        shutil.rmtree(RESULTS_ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("Reset results directory: %s", RESULTS_ROOT)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    input_path = Path(INPUT_PATH)
    if input_path.exists():
        log.info("Loading cached dataset from %s", input_path)
        df = pd.read_parquet(input_path)
        log.info("  %d rows available. Capping at %d.", len(df), MAX_ROWS)
    else:
        log.info("%s not found — downloading CivilComments (take=%d) ...", input_path, MAX_ROWS)
        df = load_civil(stream=True, take=MAX_ROWS)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(input_path, index=False)
        log.info("  Downloaded and cached %d rows -> %s", len(df), input_path)

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
    log.info("Predictions saved  →  %s  (%d rows)", OUTPUT_PATH, len(preds_df))

    # --- Save IG records (optional) ---
    if ig_records:
        ig_path = OUTPUT_PATH.replace("zs_preds.parquet", "ig_records.parquet")
        pd.DataFrame(ig_records).to_parquet(ig_path, index=False)
        log.info("IG records saved   →  %s  (%d rows)", ig_path, len(ig_records))
    elif SAVE_HEATMAPS and IG_ROWS > 0:
        log.warning("IG requested but no heatmaps were generated (likely CUDA OOM).")

    log.info("=== test_run_evaluation finished ===")


if __name__ == "__main__":
    main()