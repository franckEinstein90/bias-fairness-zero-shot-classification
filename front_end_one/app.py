from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import sys
import time

import streamlit as st
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import load_civil
from scripts.load_llm import load_llm
from front_end_one.sidebar import render_sidebar
from src.gpu_api.print_gpu_stats import get_gpu_stats


INITIAL_ROWS = 60
ROW_STEP = 60
MAX_ROWS = 2000
DATAFRAME_HEIGHT = 360


def load_zero_shot_module():
    module_path = ROOT / "src" / "zero_shot_evaluate.py"
    spec = spec_from_file_location("zero_shot_evaluate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ig_module():
    module_path = ROOT / "src" / "integrated_gradients.py"
    spec = spec_from_file_location("integrated_gradients", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_api_models_module():
    module_path = ROOT / "src" / "llm_api" / "list_available_models.py"
    spec = spec_from_file_location("list_available_models", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gemini_query_module():
    module_path = ROOT / "src" / "llm_api" / "gemini" / "query_gemini_model.py"
    spec = spec_from_file_location("query_gemini_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ZERO_SHOT = load_zero_shot_module()
IG_MOD = load_ig_module()
API_MODELS_MOD = load_api_models_module()
GEMINI_QUERY_MOD = load_gemini_query_module()


@st.cache_resource(show_spinner=False)
def get_scoring_model(model_name: str, device_name: str):
    device = torch.device(device_name)
    model, tok = load_llm(
        model_name=model_name,
        device=device,
        force_float32=(device.type != "cuda"),
    )
    return model, tok


@st.cache_data(show_spinner=False)
def get_civil_rows(take: int):
    return load_civil(stream=True, take=take)


@st.cache_data(show_spinner=False)
def get_api_models() -> list[str]:
    return API_MODELS_MOD.list_models()


def maybe_grow_dataset(should_grow: bool) -> None:
    if not should_grow:
        return

    now = time.time()
    last = st.session_state.get("_last_load_ts", 0.0)
    if now - last < 0.8:
        return

    current = st.session_state.get("rows_to_show", INITIAL_ROWS)
    if current < MAX_ROWS:
        st.session_state.rows_to_show = min(current + ROW_STEP, MAX_ROWS)
        st.session_state._last_load_ts = now
        st.rerun()


def _snapshot_gpu_state() -> list[dict]:
    return get_gpu_stats()


def _gpu_delta(before: list[dict], after: list[dict]) -> str:
    if not before or not after:
        return "n/a"
    by_idx_before = {int(g["index"]): g for g in before}
    parts: list[str] = []
    for g in after:
        idx = int(g["index"])
        if idx not in by_idx_before:
            continue
        b = by_idx_before[idx]
        d_alloc = float(g.get("allocated_gb", 0.0)) - float(b.get("allocated_gb", 0.0))
        d_res = float(g.get("reserved_gb", 0.0)) - float(b.get("reserved_gb", 0.0))
        parts.append(f"GPU{idx} Δalloc {d_alloc:+.2f} GB, Δreserved {d_res:+.2f} GB")
    return " | ".join(parts) if parts else "n/a"


def render_gpu_footer() -> None:
    stats = _snapshot_gpu_state()
    if not stats:
        st.markdown(
            """
            <div class="gpu-footer">
              <div class="gpu-footer-title">GPU Stats</div>
              <div class="gpu-footer-line">No CUDA GPU detected in this runtime.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows: list[str] = []
    for g in stats:
        util = g.get("utilization_pct")
        util_txt = f"{util:.0f}%" if isinstance(util, (int, float)) else "n/a"
        used_txt = f"{g.get('smi_used_gb', 0.0):.2f}/{g.get('smi_total_gb', g.get('total_gb', 0.0)):.2f} GB"
        rows.append(
            " ".join(
                [
                    f"GPU{int(g['index'])}",
                    f"alloc {float(g['allocated_gb']):.2f} GB",
                    f"reserved {float(g['reserved_gb']):.2f} GB",
                    f"free {float(g['free_gb']):.2f} GB",
                    f"util {util_txt}",
                    f"used {used_txt}",
                ]
            )
        )

    last_action = st.session_state.get("gpu_last_action", "none")
    last_delta = st.session_state.get("gpu_last_delta", "n/a")
    rows_html = "<br/>".join(rows)
    st.markdown(
        f"""
        <div class="gpu-footer">
          <div class="gpu-footer-title">GPU Stats (live)</div>
          <div class="gpu-footer-line">{rows_html}</div>
          <div class="gpu-footer-meta">Last tracked action: {last_action} | {last_delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_toxicity_result(result) -> None:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Prediction", str(result["pred"]))
    metric_col2.metric("Score", f"{result['score']:.4f}")
    metric_col3.metric("Task", str(result["task"]))
    metric_col4.metric("Device", str(result.get("device", "")))

    with st.expander("Score details", expanded=False):
        st.write(f"model: {result.get('model_name', '')}")
        st.write(f"positive label: {result['labels'][0]}")
        st.write(f"negative label: {result['labels'][1]}")
        if result.get("device") == "api":
            st.write("raw API answer:")
            st.code(str(result.get("raw_answer", "")))
            st.caption("Token-level log-probs are not returned by Gemini API for this flow.")
        else:
            st.write(f"lp_pos: {result['lp_pos']:.4f}")
            st.write(f"lp_neg: {result['lp_neg']:.4f}")
        st.caption(
            "Score = log P(toxic | prompt) − log P(non-toxic | prompt). "
            "Positive values favour the toxic label."
        )

    with st.expander("Prompt sent to model", expanded=False):
        st.code(str(result.get("prompt", "")))


def render_ig_result(ig) -> None:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ig_metric_col1, ig_metric_col2, ig_metric_col3 = st.columns(3)
    ig_metric_col1.metric("IG log-odds score", f"{ig['ig_score']:.4f}")
    abs_atts = np.abs(ig["atts"])
    top_idx = int(np.argmax(abs_atts))
    top_token = ig["tokens"][top_idx].replace("Ġ", "")
    ig_metric_col2.metric("Highest-attribution token", top_token)
    pos_mass = float(ig["atts"][ig["atts"] > 0].sum())
    ig_metric_col3.metric("Positive attribution mass", f"{pos_mass:.4f}")

    with st.expander("Prompt sent to model", expanded=False):
        st.code(str(ig.get("prompt", "")))

    # Heatmap bar chart — split into strips of at most 10 tokens so labels stay legible
    STRIP_SIZE = 10
    tokens_clean = [t.replace("Ġ", " ").strip() for t in ig["tokens"]]
    n_tokens = len(tokens_clean)
    n_strips = max(1, (n_tokens + STRIP_SIZE - 1) // STRIP_SIZE)

    if n_strips == 1:
        st.caption("Red = pushes toward **toxic**   |   Blue = pushes toward **non-toxic**")
    else:
        st.caption(
            f"Red = pushes toward **toxic**   |   Blue = pushes toward **non-toxic**  "
            f"— {n_tokens} tokens split across {n_strips} strips"
        )

    for strip_idx in range(n_strips):
        start = strip_idx * STRIP_SIZE
        end = min(start + STRIP_SIZE, n_tokens)
        strip_tokens = tokens_clean[start:end]
        strip_atts = ig["atts"][start:end]

        colors = ["#d62728" if v > 0 else "#1f77b4" for v in strip_atts]
        fig, ax = plt.subplots(figsize=(max(4, STRIP_SIZE * 0.7), 2.8))
        ax.bar(range(len(strip_tokens)), strip_atts, color=colors)
        ax.set_xticks(range(len(strip_tokens)))
        ax.set_xticklabels(strip_tokens, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(-0.5, STRIP_SIZE - 0.5)
        ax.set_ylabel("IG attribution", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.6)
        if n_strips > 1:
            ax.set_title(
                f"Tokens {start + 1}–{end}  (strip {strip_idx + 1} of {n_strips})",
                fontsize=8,
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    df_ig = pd.DataFrame(
        {
            "token": [t.replace("Ġ", " ").strip() for t in ig["tokens"]],
            "attribution": ig["atts"].tolist(),
            "abs_attribution": abs_atts.tolist(),
        }
    )
    df_ig["rank"] = df_ig["abs_attribution"].rank(ascending=False, method="min").astype(int)
    df_ig = df_ig.sort_values("rank")
    st.caption(
        "Attribution ≈ share of the toxicity score explained by each token. "
        "Positive = pushes toward **toxic**, negative = pushes toward **non-toxic**."
    )
    st.dataframe(
        df_ig[["rank", "token", "attribution", "abs_attribution"]]
        .rename(columns={"abs_attribution": "|attribution|"})
        .reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


@st.dialog("Toxicity Evaluation", width="large")
def show_toxicity_result_dialog(result) -> None:
    render_toxicity_result(result)
    if st.button("Close", key="close_toxicity_result_dialog"):
        st.rerun()


@st.dialog("Integrated Gradients Explanation", width="large")
def show_ig_result_dialog(ig) -> None:
    render_ig_result(ig)
    if st.button("Close", key="close_ig_result_dialog"):
        st.rerun()


st.set_page_config(
    page_title="Bias & Fairness Zero-Shot",
    page_icon="⚖️",
    layout="wide",
)

selected_model, selected_device, ig_model, ig_device, ig_steps = render_sidebar(get_api_models)

if "rows_to_show" not in st.session_state:
    st.session_state.rows_to_show = INITIAL_ROWS

if "input_text" not in st.session_state:
    st.session_state.input_text = (
        "The candidate has a strong technical background and excellent communication skills."
    )

if "pending_input_text" in st.session_state:
    st.session_state.input_text = st.session_state.pop("pending_input_text")

if "toxicity_result" not in st.session_state:
    st.session_state.toxicity_result = None

if "ig_result" not in st.session_state:
    st.session_state.ig_result = None

if "gpu_last_action" not in st.session_state:
    st.session_state.gpu_last_action = "none"

if "gpu_last_delta" not in st.session_state:
    st.session_state.gpu_last_delta = "n/a"

st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main, section.main {
        height: 100%;
    }

    [data-testid="stAppViewContainer"] > .main {
        overflow-y: auto !important;
    }

    [data-testid="stAppViewContainer"] .main .block-container,
    [data-testid="stMain"] .block-container {
        max-width: none;
        padding-top: 2.4rem;
        padding-bottom: 4.8rem;
    }

    .app-logo {
        display: block;
        margin: 0.1rem 0 0.35rem 0;
        padding: 0.45rem 0.85rem;
        border-radius: 0.65rem;
        border: 1px solid rgba(80, 120, 220, 0.35);
        background: linear-gradient(90deg, rgba(40, 90, 220, 0.12), rgba(25, 155, 180, 0.14));
        color: #0c1f52;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        line-height: 1.15;
        text-transform: uppercase;
        box-shadow: 0 6px 18px rgba(20, 60, 120, 0.08);
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }

    .app-logo-subtitle {
        margin-top: 0.25rem;
        margin-bottom: 0.6rem;
        color: rgba(20, 30, 60, 0.8);
        font-size: 0.92rem;
        font-weight: 500;
        text-align: center;
    }

    .gpu-footer {
        position: fixed;
        left: auto;
        right: 0.9rem;
        bottom: 0.6rem;
        z-index: 2000;
        width: min(54rem, calc(100vw - 1.8rem));
        border: 1px solid rgba(14, 46, 110, 0.55);
        border-radius: 0.6rem;
        background: rgba(228, 238, 255, 0.98);
        backdrop-filter: blur(3px);
        padding: 0.5rem 0.8rem;
        box-shadow: 0 8px 20px rgba(10, 28, 70, 0.22);
    }

    .gpu-footer-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0b2559;
        margin-bottom: 0.2rem;
    }

    .gpu-footer-line {
        font-size: 0.86rem;
        color: #102a57;
        line-height: 1.35;
    }

    .gpu-footer-meta {
        font-size: 0.8rem;
        color: #1f3f74;
        margin-top: 0.22rem;
    }

    /* Force-hide main page scrollbar across browsers. */
    html::-webkit-scrollbar,
    body::-webkit-scrollbar,
    section.main::-webkit-scrollbar,
    [data-testid="stAppViewContainer"]::-webkit-scrollbar,
    [data-testid="stAppViewContainer"] > .main::-webkit-scrollbar,
    [data-testid="stMain"]::-webkit-scrollbar,
    [data-testid="stMain"] > div::-webkit-scrollbar,
    [data-testid="stMain"] .block-container::-webkit-scrollbar {
        width: 0;
        height: 0;
        display: none;
    }

    html, body, section.main,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"],
    [data-testid="stMain"] > div,
    [data-testid="stMain"] .block-container {
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-logo">Bias &amp; Fairness · Zero-Shot</div>
    <div class="app-logo-subtitle">CivilComments dataset explorer + zero-shot starter</div>
    """,
    unsafe_allow_html=True,
)

st.text_area(
    "Input text",
    height=90,
    key="input_text",
)
toolbar_col1, toolbar_col2, toolbar_col3 = st.columns(3)
with toolbar_col1:
    evaluate_button = st.button("Evaluate toxicity", use_container_width=True)
with toolbar_col2:
    explain_button = st.button("Explain with Integrated Gradients", use_container_width=True)
with toolbar_col3:
    fairness_button = st.button("Fairness Evaluation", use_container_width=True)

st.subheader("Dataset")
st.caption("Loads with page refresh and grows as you scroll down. Click a row to send text to the input box.")

with st.spinner("Loading CivilComments..."):
    df = get_civil_rows(st.session_state.rows_to_show)

st.write(f"Showing {len(df):,} rows")
table_event = st.dataframe(
    df,
    width="stretch",
    height=DATAFRAME_HEIGHT,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="civil_comments_table",
)

selected_rows = table_event.selection.get("rows", [])
if selected_rows:
    selected_idx = selected_rows[0]
    if 0 <= selected_idx < len(df):
        selected_text = str(df.iloc[selected_idx].get("comment_text", ""))
        if selected_text != st.session_state.get("input_text", ""):
            st.session_state.pending_input_text = selected_text
            st.rerun()

can_grow = st.session_state.rows_to_show < MAX_ROWS
grow = st.button("Load more rows", disabled=not can_grow)
maybe_grow_dataset(grow)

if not can_grow:
    st.info(f"Reached max of {MAX_ROWS:,} rows for this viewer.")

open_toxicity_result_dialog = False
open_ig_result_dialog = False

if evaluate_button:
    gpu_before_eval = _snapshot_gpu_state()
    text = st.session_state.get("input_text", "")
    if not text.strip():
        st.warning("Please provide input text.")
    elif selected_device == "api":
        try:
            with st.spinner(f"Evaluating toxicity with {selected_model} on API..."):
                result = GEMINI_QUERY_MOD.score_and_predict_gemini(
                    model_name=selected_model,
                    text=text,
                    task="toxicity",
                )
                result["model_name"] = selected_model
                result["device"] = selected_device
                st.session_state.toxicity_result = result
                gpu_after_eval = _snapshot_gpu_state()
                st.session_state.gpu_last_action = "Evaluate toxicity"
                st.session_state.gpu_last_delta = _gpu_delta(gpu_before_eval, gpu_after_eval)
                open_toxicity_result_dialog = True
        except Exception as exc:
            st.error(f"Toxicity evaluation failed: {exc}")
    else:
        try:
            with st.spinner(f"Evaluating toxicity with {selected_model} on {selected_device}..."):
                model, tok = get_scoring_model(selected_model, selected_device)
                result = ZERO_SHOT.score_and_predict(
                    model=model,
                    tok=tok,
                    text=text,
                    task="toxicity",
                )
                result["model_name"] = selected_model
                result["device"] = selected_device
                st.session_state.toxicity_result = result
                gpu_after_eval = _snapshot_gpu_state()
                st.session_state.gpu_last_action = "Evaluate toxicity"
                st.session_state.gpu_last_delta = _gpu_delta(gpu_before_eval, gpu_after_eval)
                open_toxicity_result_dialog = True
        except Exception as exc:
            st.error(f"Toxicity evaluation failed: {exc}")

if explain_button:
    gpu_before_ig = _snapshot_gpu_state()
    text = st.session_state.get("input_text", "")
    if not text.strip():
        st.warning("No input text to explain.")
    else:
        try:
            with st.spinner("Running Integrated Gradients — this may take a moment..."):
                ig_runtime_device = "cuda" if ig_device == "gpu" and torch.cuda.is_available() else "cpu"
                model, tok = get_scoring_model(ig_model, ig_runtime_device)
                tokens, atts, prompt, ig_score = IG_MOD.integrated_gradients(
                    model=model,
                    tok=tok,
                    text=text,
                    task="toxicity",
                    steps=ig_steps,
                )
                st.session_state.ig_result = {
                    "tokens": tokens,
                    "atts": atts,
                    "prompt": prompt,
                    "ig_score": ig_score,
                }
                gpu_after_ig = _snapshot_gpu_state()
                st.session_state.gpu_last_action = "Explain with Integrated Gradients"
                st.session_state.gpu_last_delta = _gpu_delta(gpu_before_ig, gpu_after_ig)
                open_ig_result_dialog = True
        except torch.cuda.OutOfMemoryError:
            st.error(
                "CUDA out of memory running IG. Try reducing IG steps or switching to CPU."
            )
        except Exception as exc:
            st.error(f"Integrated Gradients failed: {exc}")

if fairness_button:
    st.info("Fairness Evaluation UI is not wired yet. Next step: connect per-group metrics and disparity reports.")

result = st.session_state.get("toxicity_result")
if result and open_toxicity_result_dialog:
    show_toxicity_result_dialog(result)

ig = st.session_state.get("ig_result")
if ig and open_ig_result_dialog:
    show_ig_result_dialog(ig)

render_gpu_footer()
