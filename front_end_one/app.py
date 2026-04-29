from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import sys
import time

import streamlit as st
import streamlit.components.v1 as components
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import load_civil
from scripts.load_llm import load_llm


INITIAL_ROWS = 100
ROW_STEP = 100
MAX_ROWS = 2000
DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
MODEL_OPTIONS = [
    "sshleifer/tiny-gpt2",
    "distilgpt2",
    "gpt2",
    "microsoft/Phi-4-mini-instruct",
]


def load_zero_shot_module():
    module_path = ROOT / "src" / "zero-shot-evaluate.py"
    spec = spec_from_file_location("zero_shot_evaluate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ZERO_SHOT = load_zero_shot_module()


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


def inject_scroll_loader_js() -> None:
    # Auto-click the load-more button when the page nears the bottom.
    components.html(
        """
<script>
(function () {
  const doc = window.parent.document;
  const THRESHOLD = 280;
  const COOLDOWN_MS = 900;
  let lastClick = 0;

  function nearBottom() {
    const scrollTop = window.parent.scrollY || doc.documentElement.scrollTop || 0;
    const viewport = window.parent.innerHeight || doc.documentElement.clientHeight || 0;
    const fullHeight = doc.documentElement.scrollHeight || 0;
    return (scrollTop + viewport) >= (fullHeight - THRESHOLD);
  }

  function findButton() {
    return Array.from(doc.querySelectorAll('button')).find(
      (btn) => btn.innerText && btn.innerText.trim() === 'Load more rows'
    );
  }

  function maybeLoad() {
    const now = Date.now();
    if (now - lastClick < COOLDOWN_MS) return;
    if (!nearBottom()) return;

    const btn = findButton();
    if (btn && !btn.disabled) {
      lastClick = now;
      btn.click();
    }
  }

  window.parent.addEventListener('scroll', maybeLoad, { passive: true });
  setTimeout(maybeLoad, 300);
})();
</script>
        """,
        height=0,
    )

st.set_page_config(
    page_title="Bias & Fairness Zero-Shot",
    page_icon="⚖️",
    layout="wide",
)

available_devices = ["cpu"]
if torch.cuda.is_available():
    available_devices.insert(0, "cuda")

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox(
        "Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODEL_NAME),
    )
    selected_device = st.selectbox(
        "Device",
        available_devices,
        index=0,
    )
    if not torch.cuda.is_available():
        st.caption("CUDA is not available in this environment, so CPU is selected.")
    st.caption(f"Current model: {selected_model}")
    st.caption(f"Current device: {selected_device}")

if "rows_to_show" not in st.session_state:
    st.session_state.rows_to_show = INITIAL_ROWS

if "input_text" not in st.session_state:
    st.session_state.input_text = (
        "The candidate has a strong technical background and excellent communication skills."
    )

if "toxicity_result" not in st.session_state:
    st.session_state.toxicity_result = None

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
        use_container_width=True,
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

    inject_scroll_loader_js()

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

    st.write("Score details")
    st.write(f"model: {result.get('model_name', '')}")
    st.write(f"positive label: {result['labels'][0]}")
    st.write(f"negative label: {result['labels'][1]}")
    st.write(f"lp_pos: {result['lp_pos']:.4f}")
    st.write(f"lp_neg: {result['lp_neg']:.4f}")

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
