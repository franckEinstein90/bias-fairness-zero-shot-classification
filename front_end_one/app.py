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


INITIAL_ROWS = 100
ROW_STEP = 100
MAX_ROWS = 2000


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

if "toxicity_result" not in st.session_state:
    st.session_state.toxicity_result = None

if "ig_result" not in st.session_state:
    st.session_state.ig_result = None

st.title("Bias & Fairness in Zero-Shot Classification")
st.caption("CivilComments dataset explorer + zero-shot starter")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dataset (left column)")
    st.caption("Loads with page refresh and grows as you scroll down. Click a row to send text to the input box.")

    with st.spinner("Loading CivilComments..."):
        df = get_civil_rows(st.session_state.rows_to_show)

    st.write(f"Showing {len(df):,} rows")
    table_event = st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="civil_comments_table",
    )

    selected_rows = table_event.selection.get("rows", [])
    if selected_rows:
        selected_idx = selected_rows[0]
        if 0 <= selected_idx < len(df):
            st.session_state.input_text = str(df.iloc[selected_idx].get("comment_text", ""))

    can_grow = st.session_state.rows_to_show < MAX_ROWS
    grow = st.button("Load more rows", disabled=not can_grow)
    maybe_grow_dataset(grow)

    if not can_grow:
        st.info(f"Reached max of {MAX_ROWS:,} rows for this viewer.")

with col2:
    st.subheader("Zero-shot playground")
    st.text_area(
        "Input text",
        height=160,
        key="input_text",
    )

    evaluate_button = st.button("Evaluate toxicity")

if evaluate_button:
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
        except Exception as exc:
            st.error(f"Toxicity evaluation failed: {exc}")

result = st.session_state.get("toxicity_result")
if result:
    st.success("Evaluation completed.")
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

    explain_button = st.button("Explain with Integrated Gradients")
    if explain_button:
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
            except torch.cuda.OutOfMemoryError:
                st.error(
                    "CUDA out of memory running IG. Try reducing IG steps or switching to CPU."
                )
            except Exception as exc:
                st.error(f"Integrated Gradients failed: {exc}")

ig = st.session_state.get("ig_result")
if ig:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.subheader("Token attributions (Integrated Gradients)")

    ig_metric_col1, ig_metric_col2, ig_metric_col3 = st.columns(3)
    ig_metric_col1.metric("IG log-odds score", f"{ig['ig_score']:.4f}")
    abs_atts = np.abs(ig["atts"])
    top_idx = int(np.argmax(abs_atts))
    top_token = ig["tokens"][top_idx].replace("Ġ", "")
    ig_metric_col2.metric("Highest-attribution token", top_token)
    pos_mass = float(ig["atts"][ig["atts"] > 0].sum())
    ig_metric_col3.metric("Positive attribution mass", f"{pos_mass:.4f}")

    # Heatmap bar chart
    fig, ax = plt.subplots(figsize=(max(6, len(ig["tokens"]) * 0.35), 3))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in ig["atts"]]
    ax.bar(range(len(ig["tokens"])), ig["atts"], color=colors)
    ax.set_xticks(range(len(ig["tokens"])))
    ax.set_xticklabels(
        [t.replace("Ġ", " ").strip() for t in ig["tokens"]],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("IG attribution (normalised)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(
        "Red = pushes toward toxic   |   Blue = pushes toward non-toxic",
        fontsize=9,
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Quantitative token table
    import pandas as pd
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
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.markdown(
    """
### Run locally
Use this command from the project root:

```bash
streamlit run front_end_one/app.py --server.port 8501
```
"""
)
