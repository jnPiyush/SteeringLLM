# SteeringLLM Interactive Demo

A **Streamlit** web application for showcasing LLM activation steering in real-time.

![Demo tabs](https://img.shields.io/badge/Tabs-5-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)

---

## Quick Start

```bash
# 1. Install the package with demo dependencies
pip install -e ".[demo]"

# 2. Launch the demo (from the repo root)
python demo/launch.py
```

> **Why `python demo/launch.py` instead of `streamlit run`?**
>
> On Windows, Streamlit's `pyarrow` dependency loads DLLs that conflict with
> PyTorch's `c10.dll` (`WinError 1114`).  The launcher preloads PyTorch
> *before* Streamlit starts, avoiding the conflict.  On macOS/Linux you can
> also use `streamlit run demo/app.py` directly.

The app opens at **http://localhost:8501**. GPT-2 (~500 MB) loads by default and
runs entirely on CPU — no GPU required for a live demo.

Use `python demo/launch.py --port 8502` to pick a different port.

---

## Features

| Tab | What it does |
|-----|-------------|
| **🎛️ Playground** | Pick a preset (or enter custom contrast pairs), discover a vector, set α, and compare **baseline vs. steered** outputs side-by-side. |
| **📈 Alpha Sweep** | Sweep α from negative to positive and see a table of outputs — great for showing how strength changes behavior. |
| **🔀 Composition** | Build 2–5 vectors from different presets, view their cosine similarity matrix, detect conflicts, and compose with weighted sum. |
| **🔍 Inspector** | Examine any vector's metadata, tensor statistics (mean, std, min, max), and value-distribution histogram. |
| **🗺️ Layer Explorer** | Inject the same vector at every Nth layer and compare — shows which layers matter most for steering. |

---

## Built-in Presets

| Preset | Direction | Suggested α |
|--------|-----------|-------------|
| Positive / Helpful | Encouraging, helpful tone | 2.0 |
| Formal / Professional | Academic, professional language | 1.5 |
| Concise / Direct | Short, direct answers | 2.5 |
| Creative / Imaginative | Vivid, poetic expression | 2.0 |
| Safety / Harmless | Safe, responsible outputs | 3.0 |

You can also enter fully custom contrast pairs in the Playground tab.

---

## Saving & Loading Vectors

- Click **💾 Save vector** in the Playground to persist a vector to `demo/saved_vectors/`.
- Go to the **🔍 Inspector** tab → "Load from disk" to reload it later.

Each saved vector produces two files:
- `<tag>.json` — human-readable metadata
- `<tag>.pt` — PyTorch tensor

---

## Using a Different Model

Type any HuggingFace causal-LM identifier in the sidebar (e.g., `gpt2-medium`,
`gpt2-large`) and click **Load Model**. The model only downloads once thanks to
HuggingFace caching.

> **Tip**: For a presentation, pre-download the model:
> ```python
> from transformers import AutoModelForCausalLM, AutoTokenizer
> AutoModelForCausalLM.from_pretrained("gpt2")
> AutoTokenizer.from_pretrained("gpt2")
> ```

---

## Project Structure

```
demo/
├── __init__.py          # Package marker
├── app.py               # Streamlit application (main entry point)
├── launch.py            # Launcher script (preloads PyTorch on Windows)
├── presets.py            # Contrast-pair presets for common behaviors
├── README.md            # This file
└── saved_vectors/       # Created at runtime when you save vectors
```

---

## Requirements

- Python ≥ 3.9
- `steering-llm` (this repo, installed in editable mode)
- `streamlit >= 1.30`
- `pandas >= 2.0`
- `torch`, `transformers`, `numpy`, `scikit-learn` (pulled via steering-llm)
