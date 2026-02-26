"""
SteeringLLM Interactive Demo — Streamlit Application

Launch with:
    streamlit run demo/app.py

This app lets you:
 1. Load a HuggingFace causal-LM (GPT-2 by default — runs on CPU).
 2. Discover a steering vector from contrast pairs (preset or custom).
 3. Adjust steering strength (alpha) via a slider.
 4. Generate side-by-side baseline vs. steered outputs.
 5. Compose multiple vectors and inspect similarity / conflicts.
 6. Save / load steering vectors to disk.
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``demo.presets`` is importable
# regardless of how Streamlit is launched.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Windows pyarrow/torch DLL conflict
#
# On Windows, pyarrow (a Streamlit dependency) ships DLLs that conflict
# with torch's c10.dll.  If pyarrow loads first, c10.dll initialisation
# fails with WinError 1114.  The fix: load torch BEFORE pyarrow.
#
# Recommended launch method (preloads torch then starts Streamlit):
#     python demo/launch.py
#
# If launched via ``streamlit run demo/app.py`` torch won't be preloaded
# and the import may fail.  We detect that and show a helpful message.
# ---------------------------------------------------------------------------
_TORCH_PRELOADED = "torch" in sys.modules

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SteeringLLM Demo",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports -- deferred so the UI renders immediately.
# If torch was preloaded by launch.py the import is instant.
# ---------------------------------------------------------------------------
_LIB_AVAILABLE: Optional[bool] = None
_LIB_ERROR: str = ""


def _ensure_imports() -> bool:
    """Lazily import torch + steering_llm. Returns True on success."""
    global _LIB_AVAILABLE, _LIB_ERROR  # noqa: PLW0603
    if _LIB_AVAILABLE is not None:
        return _LIB_AVAILABLE
    try:
        import torch as _torch  # noqa: F401
        from steering_llm import (  # noqa: F401
            Discovery as _D,
            SteeringModel as _SM,
            SteeringVector as _SV,
            VectorComposition as _VC,
            get_supported_architectures as _gsa,
        )
        _LIB_AVAILABLE = True
    except (ImportError, OSError) as exc:
        _LIB_AVAILABLE = False
        _LIB_ERROR = str(exc)
    return _LIB_AVAILABLE


def _import_torch():
    """Return the torch module (lazy)."""
    import torch
    return torch


def _import_lib():
    """Return core library classes after ensuring imports."""
    from steering_llm import (
        Discovery,
        SteeringModel,
        SteeringVector,
        VectorComposition,
        get_supported_architectures,
    )
    return Discovery, SteeringModel, SteeringVector, VectorComposition, get_supported_architectures


from demo.presets import PRESETS, get_preset, get_preset_names, get_tone_presets, get_role_presets


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt2"  # 124M params -- fast CPU download; switch to gpt2-large for quality
VECTOR_DIR = Path("demo/saved_vectors")
RAG_DATA_DIR = Path("examples/rag-data")  # pre-loaded PDFs for RAG demo
MAX_VECTORS = 5  # max vectors in composition tab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device_label() -> str:
    try:
        torch = _import_torch()
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
    except (ImportError, OSError):
        return "CPU (torch not loaded)"
    return "CPU"


def _truncate(text: str, max_len: int = 300) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


@st.cache_resource(show_spinner="Loading model ... this may take a minute.")
def load_model(model_name: str) -> Any:
    """Load and cache a SteeringModel."""
    _, SteeringModel, _, _, _ = _import_lib()
    model = SteeringModel.from_pretrained(model_name)
    model.eval()
    return model


def discover_vector(
    model: Any,
    positive: List[str],
    negative: List[str],
    layer: int,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Run mean-difference discovery and return (vector, metrics)."""
    Discovery, _, _, _, _ = _import_lib()
    result = Discovery.mean_difference(
        positive=positive,
        negative=negative,
        model=model.model,
        layer=layer,
        tokenizer=model.tokenizer,
    )
    return result.vector, result.metrics


def generate_texts(
    model: Any,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text from prompt."""
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }
    if not do_sample:
        gen_kwargs.pop("temperature")
        gen_kwargs.pop("top_p")
    return model.generate(prompt, **gen_kwargs)


# ---------------------------------------------------------------------------
# Sidebar — model selection & hardware info
# ---------------------------------------------------------------------------
def _sidebar() -> Optional[Any]:
    """Render sidebar and return loaded model (or None)."""
    st.sidebar.title("🧭 SteeringLLM Demo")
    st.sidebar.caption(f"Device: **{_device_label()}**")

    if not _ensure_imports():
        if "WinError 1114" in _LIB_ERROR or "c10.dll" in _LIB_ERROR:
            st.sidebar.error(
                "**PyTorch / pyarrow DLL conflict detected.**\n\n"
                "On Windows, Streamlit's `pyarrow` dependency loads DLLs that "
                "prevent PyTorch's `c10.dll` from initialising.\n\n"
                "**Fix:** Launch the demo with the provided launcher script "
                "instead:\n\n"
                "```\npython demo/launch.py\n```\n\n"
                "This preloads PyTorch before Streamlit starts, avoiding the "
                "conflict."
            )
        else:
            st.sidebar.error(
                f"`steering_llm` is not importable.\n\n```\n{_LIB_ERROR}\n```"
            )
        return None

    model_name = st.sidebar.text_input(
        "HuggingFace Model",
        value=DEFAULT_MODEL,
        help="Any causal-LM on Hugging Face Hub (e.g. gpt2, gpt2-medium, gpt2-large).",
    )

    if st.sidebar.button("Load Model", type="primary", use_container_width=True):
        st.session_state["model_name"] = model_name
        st.session_state.pop("model_obj", None)  # force reload

    st.sidebar.caption(
        ":bulb: **Tip**: `gpt2` (124 MB) loads fastest. "
        "Use `gpt2-large` for noticeably better steering quality."
    )

    if "model_name" not in st.session_state:
        st.session_state["model_name"] = model_name

    try:
        model = load_model(st.session_state["model_name"])
        st.session_state["model_obj"] = model
    except Exception as exc:
        st.sidebar.error(f"Failed to load model: {exc}")
        return None

    # Model info
    _, _, _, _, get_supported_architectures = _import_lib()
    cfg = model.config
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown(
        f"- **Architecture**: `{getattr(cfg, 'model_type', '?')}`\n"
        f"- **Layers**: {model.num_layers}\n"
        f"- **Hidden dim**: {cfg.hidden_size}\n"
        f"- **Vocab size**: {cfg.vocab_size}\n"
        f"- **Parameters**: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### Supported architectures\n"
        + ", ".join(f"`{a}`" for a in get_supported_architectures())
    )

    return model


# ---------------------------------------------------------------------------
# Tab 1 — Steering Playground
# ---------------------------------------------------------------------------
def _tab_playground(model: Any) -> None:
    st.header("🎛️ Steering Playground")
    st.markdown(
        "Discover a steering vector from contrast pairs, adjust strength, "
        "and compare **baseline** vs **steered** generation side-by-side."
    )

    # ---- Vector source ----
    col_src, col_cfg = st.columns([1.2, 0.8])
    with col_src:
        source = st.radio(
            "Vector source",
            ["Preset", "Custom"],
            horizontal=True,
            key="pg_source",
        )

        if source == "Preset":
            preset_name = st.selectbox("Tone Preset", get_tone_presets(), key="pg_preset")
            preset = get_preset(preset_name)
            st.info(preset["description"])
            positive = preset["positive"]
            negative = preset["negative"]
            default_prompt = preset.get("default_prompt", "Tell me about yourself and how you see the world.")

            with st.expander("View contrast pairs"):
                c1, c2 = st.columns(2)
                c1.markdown("**Positive**")
                for p in positive:
                    c1.markdown(f"- {_truncate(p, 90)}")
                c2.markdown("**Negative**")
                for n in negative:
                    c2.markdown(f"- {_truncate(n, 90)}")
        else:
            st.markdown("Enter your own contrast pairs (one per line):")
            positive_text = st.text_area(
                "Positive examples",
                height=140,
                key="pg_pos",
                placeholder="I love helping people!\nYou're doing great!",
            )
            negative_text = st.text_area(
                "Negative examples",
                height=140,
                key="pg_neg",
                placeholder="I hate this.\nYou're terrible.",
            )
            positive = [
                s.strip() for s in positive_text.strip().splitlines() if s.strip()
            ]
            negative = [
                s.strip() for s in negative_text.strip().splitlines() if s.strip()
            ]
            default_prompt = "Tell me about yourself and how you see the world."

    # ---- Configuration ----
    with col_cfg:
        st.markdown("### Configuration")
        num_layers = model.num_layers

        if source == "Preset":
            default_layer = max(
                0,
                min(
                    int(preset["recommended_layer_pct"] * num_layers),
                    num_layers - 1,
                ),
            )
            default_alpha = preset["default_alpha"]
        else:
            default_layer = int(num_layers * 0.6)
            default_alpha = 2.0

        layer = st.slider(
            "Target layer",
            min_value=0,
            max_value=num_layers - 1,
            value=default_layer,
            key="pg_layer",
            help="Which transformer layer to inject the steering vector into.",
        )
        alpha = st.slider(
            "Steering strength (α)",
            min_value=-5.0,
            max_value=5.0,
            value=default_alpha,
            step=0.1,
            key="pg_alpha",
            help="Multiplier for the steering vector. Negative = reverse direction.",
        )
        max_tokens = st.slider(
            "Max new tokens",
            min_value=20,
            max_value=300,
            value=150,
            step=10,
            key="pg_max_tokens",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.05,
            key="pg_temp",
        )
        do_sample = st.checkbox("Sample (uncheck for greedy)", value=True, key="pg_sample")

    # ---- Prompt & generation ----
    st.markdown("---")
    prompt = st.text_input(
        "Prompt",
        value=default_prompt,
        key="pg_prompt",
    )

    run_col1, run_col2, _ = st.columns([1, 1, 4])
    run_clicked = run_col1.button("▶  Generate", type="primary", key="pg_run")
    save_vec = run_col2.button("💾 Save vector", key="pg_save")

    if run_clicked:
        if len(positive) < 2 or len(negative) < 2:
            st.warning("Please provide at least 2 positive and 2 negative examples.")
            return

        with st.spinner("Discovering steering vector …"):
            t0 = time.time()
            vector, metrics = discover_vector(model, positive, negative, layer)
            discovery_time = time.time() - t0
            st.session_state["last_vector"] = vector

        st.success(
            f"Vector discovered in **{discovery_time:.1f}s** — "
            f"magnitude: {vector.magnitude:.4f}, dim: {vector.shape[0]}"
        )

        # Side-by-side generation
        col_base, col_steer = st.columns(2)

        with col_base:
            st.subheader("Baseline (no steering)")
            with st.spinner("Generating …"):
                model.remove_steering()
                t0 = time.time()
                baseline = generate_texts(
                    model,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                base_time = time.time() - t0
            st.text_area(
                "Output",
                value=baseline,
                height=250,
                key="pg_baseline_out",
                disabled=True,
            )
            st.caption(f"Generated in {base_time:.1f}s")

        with col_steer:
            st.subheader(f"Steered (α = {alpha})")
            with st.spinner("Generating (steered) …"):
                model.remove_steering()
                t0 = time.time()
                steered = model.generate_with_steering(
                    prompt,
                    vector=vector,
                    alpha=alpha,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                steer_time = time.time() - t0
            st.text_area(
                "Output",
                value=steered,
                height=250,
                key="pg_steered_out",
                disabled=True,
            )
            st.caption(f"Generated in {steer_time:.1f}s")

    # ---- Save vector ----
    if save_vec:
        vec = st.session_state.get("last_vector")
        if vec is None:
            st.warning("Generate first so there is a vector to save.")
        else:
            VECTOR_DIR.mkdir(parents=True, exist_ok=True)
            tag = f"{vec.model_name.replace('/', '_')}_L{vec.layer}"
            save_path = VECTOR_DIR / tag
            vec.save(str(save_path))
            st.success(f"Saved to `{save_path}.json` / `.pt`")


# ---------------------------------------------------------------------------
# Tab 2 — Alpha Sweep
# ---------------------------------------------------------------------------
def _tab_alpha_sweep(model: Any) -> None:
    st.header("📈 Alpha Sweep")
    st.markdown(
        "Sweep through a range of steering strengths to visualise how alpha "
        "affects generation output."
    )

    vec = st.session_state.get("last_vector")
    if vec is None:
        st.info("Go to the **Playground** tab and generate a vector first.")
        return

    prompt = st.text_input(
        "Prompt",
        value="Tell me about yourself.",
        key="sweep_prompt",
    )

    c1, c2, c3 = st.columns(3)
    alpha_min = c1.number_input("α min", value=-3.0, step=0.5, key="sw_min")
    alpha_max = c2.number_input("α max", value=3.0, step=0.5, key="sw_max")
    alpha_step = c3.number_input("Step", value=1.0, min_value=0.5, step=0.5, key="sw_step")

    max_tokens = st.slider("Max new tokens", 20, 200, 60, key="sw_tok")

    if st.button("▶  Run Sweep", type="primary", key="sw_run"):
        alphas = []
        a = alpha_min
        while a <= alpha_max + 1e-9:
            alphas.append(round(a, 2))
            a += alpha_step

        results: List[Dict[str, Any]] = []
        progress = st.progress(0, text="Sweeping …")
        for i, a in enumerate(alphas):
            model.remove_steering()
            out = model.generate_with_steering(
                prompt,
                vector=vec,
                alpha=a,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            results.append({"α": a, "Output": out})
            progress.progress((i + 1) / len(alphas), text=f"α = {a}")

        progress.empty()
        st.dataframe(results, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3 — Vector Composition
# ---------------------------------------------------------------------------
def _tab_composition(model: Any) -> None:
    st.header("🔀 Vector Composition")
    st.markdown(
        "Build vectors from different presets and combine them with weighted sum. "
        "Inspect cosine similarity and conflict detection."
    )

    num_layers = model.num_layers
    preset_names = get_preset_names()

    # --- Build multiple vectors ---
    num_vectors = st.slider("Number of vectors", 2, MAX_VECTORS, 2, key="comp_n")

    vectors: List[Any] = []
    weights: List[float] = []
    labels: List[str] = []

    cols = st.columns(num_vectors)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Vector {i + 1}**")
            name = st.selectbox(
                "Preset",
                preset_names,
                index=i % len(preset_names),
                key=f"comp_preset_{i}",
            )
            preset = get_preset(name)
            default_layer = max(
                0,
                min(
                    int(preset["recommended_layer_pct"] * num_layers),
                    num_layers - 1,
                ),
            )
            layer_i = st.slider(
                "Layer",
                0,
                num_layers - 1,
                default_layer,
                key=f"comp_layer_{i}",
            )
            w_i = st.slider(
                "Weight",
                -2.0,
                2.0,
                1.0,
                step=0.1,
                key=f"comp_w_{i}",
            )

            labels.append(f"{name} (L{layer_i})")
            weights.append(w_i)

            # Store config for lazy discovery
            st.session_state[f"_comp_cfg_{i}"] = {
                "positive": preset["positive"],
                "negative": preset["negative"],
                "layer": layer_i,
            }

    st.markdown("---")
    if st.button("▶  Discover & Compose", type="primary", key="comp_run"):
        # 1. Discover each vector
        progress = st.progress(0, text="Discovering vectors …")
        for i in range(num_vectors):
            cfg = st.session_state[f"_comp_cfg_{i}"]
            vec, _ = discover_vector(
                model, cfg["positive"], cfg["negative"], cfg["layer"]
            )
            vectors.append(vec)
            progress.progress((i + 1) / num_vectors)
        progress.empty()

        # 2. Similarity matrix
        _, _, _, VectorComposition, _ = _import_lib()
        st.subheader("Cosine Similarity Matrix")
        matrix_data: List[Dict[str, Any]] = []
        for i, vi in enumerate(vectors):
            row: Dict[str, Any] = {"": labels[i]}
            for j, vj in enumerate(vectors):
                if vi.tensor.shape == vj.tensor.shape:
                    sim = VectorComposition.compute_similarity(vi, vj)
                    row[labels[j]] = f"{sim:.3f}"
                else:
                    row[labels[j]] = "n/a (dim)"
            matrix_data.append(row)
        st.dataframe(matrix_data, use_container_width=True, hide_index=True)

        # 3. Conflict detection (only for same-shape vectors)
        same_layer_vecs = [v for v in vectors if v.layer == vectors[0].layer]
        if len(same_layer_vecs) >= 2:
            conflicts = VectorComposition.detect_conflicts(same_layer_vecs)
            if conflicts:
                st.warning(
                    "Conflicts detected: "
                    + ", ".join(
                        f"V{a + 1}↔V{b + 1} (sim={s:.3f})"
                        for a, b, s in conflicts
                    )
                )
            else:
                st.success("No conflicts detected between same-layer vectors.")

        # 4. Compose (same-layer only)
        same_layer_indices = [
            i for i, v in enumerate(vectors) if v.layer == vectors[0].layer
        ]
        if len(same_layer_indices) >= 2:
            sel_vecs = [vectors[i] for i in same_layer_indices]
            sel_weights = [weights[i] for i in same_layer_indices]
            _, _, _, VectorComposition, _ = _import_lib()
            composed = VectorComposition.weighted_sum(sel_vecs, sel_weights)
            st.session_state["composed_vector"] = composed
            st.info(
                f"Composed vector — magnitude: {composed.magnitude:.4f}, "
                f"layer: {composed.layer}, dim: {composed.shape[0]}"
            )

            # Quick generate
            prompt = st.text_input(
                "Test prompt",
                value="Explain quantum computing.",
                key="comp_prompt",
            )
            alpha = st.slider(
                "α (composed)",
                -5.0,
                5.0,
                2.0,
                step=0.1,
                key="comp_alpha",
            )
            if st.button("Generate with composed vector", key="comp_gen"):
                model.remove_steering()
                out = model.generate_with_steering(
                    prompt,
                    vector=composed,
                    alpha=alpha,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.7,
                )
                st.text_area("Output", value=out, height=200, disabled=True, key="comp_out")
        else:
            st.info(
                "Vectors target different layers — composition requires "
                "same-layer vectors. Adjust layers above."
            )


# ---------------------------------------------------------------------------
# Tab 4 — Vector Inspector
# ---------------------------------------------------------------------------
def _tab_inspector(model: Any) -> None:
    st.header("🔍 Vector Inspector")
    st.markdown(
        "Examine a steering vector's metadata, statistics, and load saved vectors."
    )

    vec = st.session_state.get("last_vector")
    composed = st.session_state.get("composed_vector")

    source = st.radio(
        "Inspect",
        [
            "Last discovered vector",
            "Composed vector",
            "Load from disk",
        ],
        key="insp_source",
        horizontal=True,
    )

    target_vec: Optional[Any] = None

    if source == "Last discovered vector":
        target_vec = vec
    elif source == "Composed vector":
        target_vec = composed
    else:
        # Load from disk
        _, _, SteeringVector, _, _ = _import_lib()
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        json_files = sorted(VECTOR_DIR.glob("*.json"))
        if not json_files:
            st.info(f"No saved vectors found in `{VECTOR_DIR}`.")
            return
        selected = st.selectbox(
            "Saved vectors",
            [f.stem for f in json_files],
            key="insp_file",
        )
        if selected and st.button("Load", key="insp_load"):
            target_vec = SteeringVector.load(str(VECTOR_DIR / selected))
            st.session_state["last_vector"] = target_vec

    if target_vec is None:
        st.info("No vector available. Discover one in the Playground first.")
        return

    # ---- Metadata table ----
    st.subheader("Metadata")
    meta = {
        "Model": target_vec.model_name,
        "Layer": target_vec.layer,
        "Layer name": target_vec.layer_name,
        "Method": target_vec.method,
        "Dimension": target_vec.shape[0],
        "Magnitude (L2)": f"{target_vec.magnitude:.6f}",
        "Dtype": str(target_vec.dtype),
        "Created at": target_vec.created_at or "n/a",
    }
    for k, v in meta.items():
        st.markdown(f"- **{k}**: {v}")

    if target_vec.metadata:
        st.subheader("Extra metadata")
        st.json(target_vec.metadata)

    # ---- Tensor statistics ----
    st.subheader("Tensor Statistics")
    t = target_vec.tensor
    stats = {
        "Mean": f"{t.mean().item():.6f}",
        "Std": f"{t.std().item():.6f}",
        "Min": f"{t.min().item():.6f}",
        "Max": f"{t.max().item():.6f}",
        "Abs-mean": f"{t.abs().mean().item():.6f}",
        "Non-zero %": f"{(t != 0).float().mean().item() * 100:.1f}%",
    }
    cols = st.columns(3)
    for i, (k, v) in enumerate(stats.items()):
        cols[i % 3].metric(k, v)

    # ---- Histogram ----
    st.subheader("Value Distribution")
    import numpy as np

    hist_vals = target_vec.tensor.numpy()
    # Using streamlit's built-in chart
    import pandas as pd

    df_hist = pd.DataFrame({"value": hist_vals})
    st.bar_chart(df_hist["value"].value_counts(bins=50).sort_index())

    # ---- Export / Download ----
    st.subheader("Export")
    exp_col1, exp_col2 = st.columns(2)

    # JSON metadata export
    export_meta = {
        "model_name": target_vec.model_name,
        "layer": target_vec.layer,
        "layer_name": target_vec.layer_name,
        "method": target_vec.method,
        "dimension": target_vec.shape[0],
        "magnitude": target_vec.magnitude,
        "dtype": str(target_vec.dtype),
        "created_at": target_vec.created_at or "n/a",
        "tensor": [round(float(v), 8) for v in target_vec.tensor.tolist()],
    }
    if target_vec.metadata:
        export_meta["metadata"] = target_vec.metadata
    json_bytes = json.dumps(export_meta, indent=2).encode("utf-8")

    tag = (
        f"{target_vec.model_name.replace('/', '_')}"
        f"_L{target_vec.layer}_{target_vec.method}"
    )
    with exp_col1:
        st.download_button(
            label="Download vector (JSON)",
            data=json_bytes,
            file_name=f"{tag}.json",
            mime="application/json",
            key="insp_dl_json",
        )

    # NumPy .npy export (lightweight, easy to reload)
    npy_buf = io.BytesIO()
    np.save(npy_buf, hist_vals)
    npy_buf.seek(0)
    with exp_col2:
        st.download_button(
            label="Download tensor (.npy)",
            data=npy_buf,
            file_name=f"{tag}.npy",
            mime="application/octet-stream",
            key="insp_dl_npy",
        )


# ---------------------------------------------------------------------------
# Tab 5 — Layer Explorer
# ---------------------------------------------------------------------------
def _tab_layer_explorer(model: Any) -> None:
    st.header("🗺️ Layer Explorer")
    st.markdown(
        "Generate text with the same vector injected at different layers "
        "to understand where steering has the most impact."
    )

    # Use current preset to build vectors
    preset_name = st.selectbox(
        "Preset", get_preset_names(), key="le_preset"
    )
    preset = get_preset(preset_name)
    prompt = st.text_input(
        "Prompt",
        value="Describe artificial intelligence in one paragraph.",
        key="le_prompt",
    )
    alpha = st.slider("α", -5.0, 5.0, 2.0, step=0.5, key="le_alpha")
    max_tokens = st.slider("Max tokens", 20, 150, 60, key="le_tok")

    num_layers = model.num_layers
    stride = st.slider(
        "Layer stride",
        min_value=1,
        max_value=max(1, num_layers // 2),
        value=max(1, num_layers // 6),
        key="le_stride",
        help="Test every Nth layer. Lower = more layers tested (slower).",
    )
    layers_to_test = list(range(0, num_layers, stride))
    if (num_layers - 1) not in layers_to_test:
        layers_to_test.append(num_layers - 1)

    st.caption(f"Testing layers: {layers_to_test}")

    if st.button("▶  Run Layer Sweep", type="primary", key="le_run"):
        results: List[Dict[str, Any]] = []
        progress = st.progress(0, text="Sweeping layers …")

        for i, lay in enumerate(layers_to_test):
            vec, _ = discover_vector(
                model, preset["positive"], preset["negative"], lay
            )
            model.remove_steering()
            out = model.generate_with_steering(
                prompt,
                vector=vec,
                alpha=alpha,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            results.append({
                "Layer": lay,
                "Magnitude": f"{vec.magnitude:.4f}",
                "Output": out,
            })
            progress.progress((i + 1) / len(layers_to_test))

        progress.empty()
        st.dataframe(results, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 6 -- Role / Domain Expertise
# ---------------------------------------------------------------------------
def _tab_role_expertise(model: Any) -> None:
    st.header("👔 Role / Domain Expertise")
    st.markdown(
        "Steer the model to respond **like a domain expert** — a doctor, lawyer, "
        "software engineer, scientist, and more.  "
        "Compare the baseline (no steering) against the steered output to see "
        "domain-specific language, terminology, and reasoning emerge."
    )

    col_src, col_cfg = st.columns([1.2, 0.8])
    with col_src:
        role_names = get_role_presets()
        role_name = st.selectbox("Expert Role", role_names, key="role_preset")
        preset = get_preset(role_name)

        st.info(f"**{role_name}**: {preset['description']}")

        with st.expander("View contrast pairs (what defines this role)"):
            c1, c2 = st.columns(2)
            c1.markdown("**Expert language (positive)**")
            for p in preset["positive"]:
                c1.markdown(f"- {_truncate(p, 90)}")
            c2.markdown("**Non-expert language (negative)**")
            for n in preset["negative"]:
                c2.markdown(f"- {_truncate(n, 90)}")

    with col_cfg:
        st.markdown("### Configuration")
        num_layers = model.num_layers
        default_layer = max(
            0,
            min(
                int(preset["recommended_layer_pct"] * num_layers),
                num_layers - 1,
            ),
        )
        layer = st.slider(
            "Target layer",
            min_value=0,
            max_value=num_layers - 1,
            value=default_layer,
            key="role_layer",
        )
        alpha = st.slider(
            "Steering strength (α)",
            min_value=-5.0,
            max_value=5.0,
            value=float(preset["default_alpha"]),
            step=0.1,
            key="role_alpha",
            help="Higher α = stronger expert persona. Too high may cause repetition.",
        )
        max_tokens = st.slider(
            "Max new tokens", 30, 300, 150, step=10, key="role_tokens"
        )
        temperature = st.slider(
            "Temperature", 0.1, 2.0, 0.7, step=0.05, key="role_temp"
        )

    st.markdown("---")
    default_prompt = preset.get("default_prompt", "Explain your area of expertise.")
    prompt = st.text_input(
        "Prompt",
        value=default_prompt,
        key="role_prompt",
        help="Phrase as a completion starter, not a chat question (base model behavior).",
    )

    if st.button("▶  Generate", type="primary", key="role_run"):
        with st.spinner("Discovering role steering vector …"):
            t0 = time.time()
            vector, _ = discover_vector(
                model, preset["positive"], preset["negative"], layer
            )
            st.session_state["last_vector"] = vector
            disc_time = time.time() - t0

        st.success(
            f"Role vector discovered in **{disc_time:.1f}s** — "
            f"magnitude: {vector.magnitude:.4f}"
        )

        col_base, col_steered = st.columns(2)

        with col_base:
            st.subheader("Baseline (no role steering)")
            with st.spinner("Generating …"):
                model.remove_steering()
                t0 = time.time()
                baseline = generate_texts(
                    model,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
                base_time = time.time() - t0
            st.text_area(
                "Output",
                value=baseline,
                height=280,
                key="role_baseline_out",
                disabled=True,
            )
            st.caption(f"Generated in {base_time:.1f}s")

        with col_steered:
            st.subheader(f"Steered as: {role_name} (α={alpha})")
            with st.spinner("Generating (role-steered) …"):
                model.remove_steering()
                t0 = time.time()
                steered = model.generate_with_steering(
                    prompt,
                    vector=vector,
                    alpha=alpha,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
                steer_time = time.time() - t0
            st.text_area(
                "Output",
                value=steered,
                height=280,
                key="role_steered_out",
                disabled=True,
            )
            st.caption(f"Generated in {steer_time:.1f}s")

        st.markdown(
            "> **What to look for**: Domain vocabulary, structured reasoning, "
            "professional framing, and specific terminology appear in the steered "
            "output but not (or far less) in the baseline."
        )


# ---------------------------------------------------------------------------
# Tab 7 -- RAG / Document-Grounded Steering
# ---------------------------------------------------------------------------
def _tab_rag(model: Any) -> None:
    st.header("📄 RAG / Document-Grounded Steering")
    st.markdown(
        "Upload PDF documents and steer the model to generate answers **grounded "
        "in your documents** rather than vague or hallucinated content.  "
        "The steering vector is built from chunks of your PDF as positive examples "
        "vs generic vague statements as negatives — biasing the model toward your data."
    )

    # ---- PDF source selection ----
    source_mode = st.radio(
        "Document source",
        ["Upload PDF(s)", "Use pre-loaded sample PDFs"],
        horizontal=True,
        key="rag_source_mode",
    )

    pdf_bytes_list: List[bytes] = []
    pdf_names: List[str] = []

    if source_mode == "Upload PDF(s)":
        uploaded = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="rag_uploader",
        )
        if uploaded:
            for f in uploaded:
                pdf_bytes_list.append(f.read())
                pdf_names.append(f.name)
    else:
        # Use pre-loaded PDFs from examples/rag-data/
        available_pdfs = sorted(RAG_DATA_DIR.glob("*.pdf")) if RAG_DATA_DIR.exists() else []
        if not available_pdfs:
            st.warning(
                f"No PDFs found in `{RAG_DATA_DIR}`. "
                "Upload a PDF using the other source mode."
            )
        else:
            selected_names = st.multiselect(
                "Select PDF(s) to use",
                [p.name for p in available_pdfs],
                default=[available_pdfs[0].name],
                key="rag_preloaded_select",
            )
            for name in selected_names:
                path = RAG_DATA_DIR / name
                pdf_bytes_list.append(path.read_bytes())
                pdf_names.append(name)

    if not pdf_bytes_list:
        st.info("Select or upload at least one PDF to continue.")
        return

    st.success(f"Loaded: {', '.join(pdf_names)}")

    # ---- Extract & chunk ----
    try:
        from demo.pdf_utils import (
            extract_text_from_pdf,
            chunk_text,
            build_rag_contrast_pairs,
            summarize_chunks,
        )
    except ImportError as e:
        st.error(
            f"PDF utilities failed to import: {e}\n\n"
            "Install PDF dependencies:\n```\npip install PyPDF2 pdfplumber\n```"
        )
        return

    col_chunk, col_gen = st.columns([1, 1])
    with col_chunk:
        chunk_size = st.slider("Words per chunk", 100, 500, 250, step=50, key="rag_chunk")
        max_positive = st.slider("Positive examples (chunks)", 4, 12, 8, key="rag_maxpos")

    with col_gen:
        num_layers = model.num_layers
        layer = st.slider(
            "Steering layer",
            0,
            num_layers - 1,
            int(num_layers * 0.65),
            key="rag_layer",
        )
        alpha = st.slider(
            "Steering strength (α)",
            0.5,
            5.0,
            2.5,
            step=0.1,
            key="rag_alpha",
            help="Higher = model more strongly follows document grounding.",
        )
        max_tokens = st.slider("Max new tokens", 30, 300, 150, step=10, key="rag_tokens")

    st.markdown("---")
    prompt = st.text_input(
        "Question / Prompt",
        value="What are the key strategic priorities and recommendations described in this document?",
        key="rag_prompt",
        help="Phrase as a completion starter. The grounded model will use document facts.",
    )

    if st.button("▶  Extract, Steer & Generate", type="primary", key="rag_run"):
        # Step 1: Extract text from all PDFs
        all_chunks: List[str] = []
        with st.spinner("Extracting text from PDF(s) …"):
            for name, fb in zip(pdf_names, pdf_bytes_list):
                try:
                    text = extract_text_from_pdf(fb)
                    chunks = chunk_text(text, chunk_size=chunk_size)
                    all_chunks.extend(chunks)
                    st.caption(f"  {name}: {len(text):,} chars -> {len(chunks)} chunks")
                except Exception as exc:
                    st.error(f"Failed to extract text from {name}: {exc}")
                    return

        if not all_chunks:
            st.error("No text could be extracted from the selected PDFs.")
            return

        st.info(f"Total chunks available: **{len(all_chunks)}** — using {max_positive} for steering.")

        # Step 2: Build contrast pairs
        try:
            positives, negatives = build_rag_contrast_pairs(
                all_chunks,
                max_positive=max_positive,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.expander("Source passages used for steering (positive examples)"):
            for i, p in enumerate(positives):
                st.markdown(f"**Chunk {i + 1}:** {_truncate(p, 200)}")

        # Step 3: Discover RAG steering vector
        with st.spinner("Building document-grounded steering vector …"):
            t0 = time.time()
            try:
                vector, _ = discover_vector(model, positives, negatives, layer)
                st.session_state["last_vector"] = vector
                disc_time = time.time() - t0
            except Exception as exc:
                st.error(f"Vector discovery failed: {exc}")
                return

        st.success(
            f"RAG steering vector ready in **{disc_time:.1f}s** — "
            f"magnitude: {vector.magnitude:.4f}, layer: {layer}"
        )

        # Step 4: Generate baseline vs grounded
        col_base, col_grounded = st.columns(2)

        with col_base:
            st.subheader("Baseline (no document grounding)")
            with st.spinner("Generating …"):
                model.remove_steering()
                t0 = time.time()
                baseline = generate_texts(
                    model,
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                )
                base_time = time.time() - t0
            st.text_area(
                "Output",
                value=baseline,
                height=300,
                key="rag_baseline_out",
                disabled=True,
            )
            st.caption(f"Generated in {base_time:.1f}s")

        with col_grounded:
            st.subheader(f"Grounded on: {', '.join(pdf_names)} (α={alpha})")
            with st.spinner("Generating (document-grounded) …"):
                model.remove_steering()
                t0 = time.time()
                grounded = model.generate_with_steering(
                    prompt,
                    vector=vector,
                    alpha=alpha,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                )
                grounded_time = time.time() - t0
            st.text_area(
                "Output",
                value=grounded,
                height=300,
                key="rag_grounded_out",
                disabled=True,
            )
            st.caption(f"Generated in {grounded_time:.1f}s")

        st.markdown(
            "> **What to look for**: The grounded output should use terminology, "
            "facts, and themes from the PDF documents, while the baseline produces "
            "generic or vague text without document-specific knowledge."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Show a prominent banner when torch wasn't preloaded (direct streamlit run)
    if not _TORCH_PRELOADED and sys.platform == "win32":
        st.error(
            "**Windows detected without PyTorch preloaded.**  "
            "Please launch the demo using:\n\n"
            "```\npython demo/launch.py\n```\n\n"
            "Running `streamlit run demo/app.py` directly causes a "
            "pyarrow/torch DLL conflict on Windows. "
            "The launcher preloads PyTorch before Streamlit starts."
        )

    model = _sidebar()
    if model is None:
        if _LIB_AVAILABLE is False:
            # Already showing specific error in sidebar; just stop
            st.stop()
        st.warning(
            "Load a model from the sidebar to begin. "
            "**GPT-2** (~124 MB, fast download) is the default. "
            "Switch to `gpt2-large` for better steering quality."
        )
        st.stop()

    # Three primary use-case tabs, then advanced power-user tabs
    (
        tab_tone,
        tab_role,
        tab_rag,
        tab_sweep,
        tab_comp,
        tab_insp,
        tab_layer,
    ) = st.tabs(
        [
            "🎭 Tone / Personality",
            "👔 Role Expertise",
            "📄 RAG / Document Grounding",
            "📈 Alpha Sweep",
            "🔀 Composition",
            "🔍 Inspector",
            "🗺️ Layer Explorer",
        ]
    )

    with tab_tone:
        _tab_playground(model)
    with tab_role:
        _tab_role_expertise(model)
    with tab_rag:
        _tab_rag(model)
    with tab_sweep:
        _tab_alpha_sweep(model)
    with tab_comp:
        _tab_composition(model)
    with tab_insp:
        _tab_inspector(model)
    with tab_layer:
        _tab_layer_explorer(model)


if __name__ == "__main__":
    main()
