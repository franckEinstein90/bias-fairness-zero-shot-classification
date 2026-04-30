from __future__ import annotations

import streamlit as st
import torch

DEFAULT_MODEL_NAME: str = "sshleifer/tiny-gpt2"
MODEL_OPTIONS = [
    "sshleifer/tiny-gpt2",
    "distilgpt2",
    "gpt2",
    "microsoft/Phi-4-mini-instruct",
]

IG_STEPS_OPTIONS = [4, 8, 16, 32, 64]
IG_STEPS_DEFAULT = 16


def render_sidebar(get_api_models_fn) -> tuple[str, str, str, str, int]:
    """Render the sidebar and return zero-shot and IG settings.

    Parameters
    ----------
    get_api_models_fn:
        Callable that returns a list[str] of API model names (may be cached).

    Returns
    -------
    tuple[str, str, str, str, int]
        (selected_model, selected_device, ig_model, ig_device, ig_steps)
    """
    available_devices: list[str] = ["cpu"]
    try:
        torch.device("cuda")
        available_devices.insert(0, "cuda")
    except Exception:
        pass
    available_devices.append("api")

    with st.sidebar:
        # ── About ──────────────────────────────────────────────────────────
        try:
            st.page_link("pages/about.py", label="About", icon="ℹ️")
        except Exception:
            # Fallback for environments where Streamlit page metadata is unavailable.
            st.markdown("[ℹ️ About](about)")
        st.divider()

        # ── Zero-shot settings ─────────────────────────────────────────────
        st.subheader("Zero-shot settings")

        selected_device: str = st.selectbox(
            "Device",
            available_devices,
            index=0,
        )

        model_options = MODEL_OPTIONS
        default_model_name = DEFAULT_MODEL_NAME
        if selected_device == "api":
            try:
                api_models = get_api_models_fn()
                if not api_models:
                    st.warning("No API models were returned.")
                    model_options = ["(no API models available)"]
                else:
                    model_options = api_models
            except Exception as exc:
                st.error(f"Failed to list API models: {exc}")
                model_options = ["(failed to load API models)"]
            default_model_name = model_options[0]

        default_index = 0
        if default_model_name in model_options:
            default_index = model_options.index(default_model_name)

        selected_model: str = st.selectbox(
            "Model",
            model_options,
            index=default_index,
        )

        if not torch.cuda.is_available():
            st.caption("CUDA is not available in this environment, so CPU is selected.")
        st.caption(f"Current model: {selected_model}")
        st.caption(f"Current device: {selected_device}")

        st.divider()

        # ── Integrated Gradients settings ──────────────────────────────────
        st.subheader("Integrated Gradients")

        ig_model: str = st.selectbox(
            "IG model",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(DEFAULT_MODEL_NAME),
            key="ig_model_selector",
        )

        ig_device_options = ["cpu", "gpu"]
        ig_default_device_idx = 1 if torch.cuda.is_available() else 0
        ig_device: str = st.selectbox(
            "IG device",
            ig_device_options,
            index=ig_default_device_idx,
            key="ig_device_selector",
        )

        ig_steps: int = st.select_slider(
            "Integration steps (higher = more accurate, slower)",
            options=IG_STEPS_OPTIONS,
            value=IG_STEPS_DEFAULT,
            key="ig_steps_slider",
        )
        if ig_device == "gpu" and not torch.cuda.is_available():
            st.caption("GPU is not available in this environment; IG will run on CPU.")
        st.caption(f"IG model: {ig_model}")
        st.caption(f"IG device: {ig_device}")

    return selected_model, selected_device, ig_model, ig_device, ig_steps
