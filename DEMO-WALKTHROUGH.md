# SteeringLLM Demo Walkthrough

Step-by-step guide to launching the app and exercising every feature across
all seven tabs.

---

## Tab Overview

The demo has seven tabs organized into three **primary use-case tabs** and
four **advanced / power-user tabs**:

| # | Tab | Purpose |
|---|-----|---------|
| 1 | Tone / Personality | Discover a steering vector from tone presets; compare baseline vs steered side-by-side |
| 2 | Role Expertise | Steer the model to respond like a domain expert (doctor, lawyer, engineer, etc.) |
| 3 | RAG / Document Grounding | Upload PDFs and steer the model toward document-grounded answers |
| 4 | Alpha Sweep | Sweep through steering strengths to find the optimal alpha |
| 5 | Composition | Combine multiple steering vectors and inspect similarity / conflicts |
| 6 | Inspector | Examine tensor statistics, metadata, and export vectors |
| 7 | Layer Explorer | Test which transformer layer produces the strongest steering effect |

---

## Prerequisites

Ensure everything is installed (one-time setup):

```powershell
cd C:\Engagements\Learnings\SteeringLLM
python -m pip install --user -e .
python -m pip install --user streamlit torch transformers numpy scikit-learn pandas
```

Verify imports:

```powershell
python -c "from steering_llm import Discovery, SteeringModel; import streamlit; print('Ready')"
# Expected: Ready
```

---

## Step 1 -- Launch the App

```powershell
python demo/launch.py --port 8501
```

Open your browser to **http://localhost:8501**

> Always use `demo/launch.py` instead of `streamlit run demo/app.py` on Windows.
> The launcher pre-loads PyTorch before Streamlit's pyarrow DLLs to avoid the
> `WinError 1114 / c10.dll` conflict.

---

## Step 2 -- Load the Model

1. In the **left sidebar**, the model field defaults to `microsoft/phi-2`
2. Click **Load Model** (blue button)
3. Wait for the download (~5.5 GB on first run, cached after that)
4. Confirm the sidebar shows model info:
   - Architecture: `phi`
   - Layers: 32
   - Hidden dim: 2560
   - Parameters: 2780M

> **Alternative models**: Type any HuggingFace causal-LM ID before clicking
> Load. Good options: `gpt2` (fast, 124M), `gpt2-large` (774M, good balance),
> `gpt2-xl` (1.5B), `microsoft/phi-2` (2.7B, best quality on CPU).

---

## Step 3 -- Tab 1: Tone / Personality (Core Feature)

Discovers a steering vector from tone contrast pairs and shows **baseline
vs steered** output side-by-side.

### Features

- **Vector source toggle**: choose `Preset` (built-in contrast pairs) or `Custom` (your own)
- **5 tone presets**: Positive/Helpful, Formal/Professional, Concise/Direct, Creative/Imaginative, Safety/Harmless
- **Configuration panel**: target layer, steering strength (alpha), max tokens, temperature, sampling
- **Side-by-side generation**: baseline (no steering) vs steered output
- **Save vector**: persist to disk as `.json` + `.pt` files

### Step-by-step

#### 3a. Try "Positive / Helpful" preset

1. Select tab **Tone / Personality**
2. Set **Vector source** = `Preset`
3. Choose preset = **Positive / Helpful**
4. Review the contrast pairs (expand "View contrast pairs")
5. Leave default settings:
   - Layer: auto-set to ~60% depth
   - Steering strength (alpha): `2.0`
   - Max new tokens: `150`
6. Enter prompt: `Tell me about yourself and how you see the world.`
7. Click **Generate**
8. Compare the two columns:
   - **Baseline**: neutral model output
   - **Steered**: noticeably warmer, more encouraging tone

#### 3b. Try negative alpha (reverse direction)

1. Change alpha to `-2.0` using the slider
2. Click **Generate** again
3. Observe: steered output becomes harsher/more negative -- the vector
   is being applied in reverse

#### 3c. Try "Formal / Professional" preset

1. Change preset to **Formal / Professional**
2. Set alpha = `1.5`
3. Use prompt: `What do you think about the future of technology?`
4. Click **Generate**
5. Steered output should adopt formal vocabulary and structured phrasing

#### 3d. Try "Creative / Imaginative" preset

1. Change preset to **Creative / Imaginative**
2. Set alpha = `2.0`, layer = `65%` (auto-set)
3. Use prompt: `Describe how a computer works.`
4. Click **Generate**
5. Steered output should use metaphorical, vivid language vs. the
   literal baseline

#### 3e. Try a Custom vector

1. Set **Vector source** = `Custom`
2. In "Positive examples", paste:
   ```
   The evidence supports this conclusion.
   Research clearly demonstrates this finding.
   Studies confirm this result.
   Data analysis reveals this pattern.
   ```
3. In "Negative examples", paste:
   ```
   I believe this might be the case.
   In my opinion, this feels right.
   It seems like this could be true.
   I think this is probably correct.
   ```
4. Use prompt: `Is coffee good for you?`
5. Click **Generate** -- steered output should sound more evidence-based

#### 3f. Save the vector

After any generation, click **Save vector** to persist to
`demo/saved_vectors/` as `.json` + `.pt` files.

---

## Step 4 -- Tab 2: Role Expertise

Steers the model to respond **like a domain expert** -- a doctor, lawyer,
software engineer, teacher, business consultant, or scientist.

### Features

- **6 role presets**: Doctor/Medical Professional, Lawyer/Legal Expert,
  Software Engineer/Technical Expert, Teacher/Educator, Business Consultant,
  Scientist/Researcher
- **Expandable contrast pairs**: shows expert language (positive) vs
  non-expert language (negative) that defines each role
- **Configuration panel**: target layer, alpha, max tokens, temperature
- **Role-specific default prompts**: each preset comes with a scenario
  prompt tailored to that domain
- **Side-by-side output**: baseline vs role-steered generation
- **Interpretation hint**: displayed after generation to guide observation

### Step-by-step

1. Select tab **Role Expertise**
2. Choose **Expert Role** = `Doctor / Medical Professional`
3. Expand "View contrast pairs" to see what defines the role
4. Review the auto-filled prompt (a clinical scenario with symptoms)
5. Leave defaults: layer ~65%, alpha = `2.5`, max tokens = `150`
6. Click **Generate**
7. Compare the two columns:
   - **Baseline**: generic model output, no medical framing
   - **Steered**: clinical vocabulary, structured assessment, evidence-based
     recommendations, empathetic professional tone
8. Try **Lawyer / Legal Expert** with its default landlord-dispute prompt
9. Try **Software Engineer** with the microservices architecture prompt
10. Observe how domain-specific terminology, reasoning patterns, and
    professional framing emerge in the steered output

**What to look for**: the steered output should use domain vocabulary
(e.g., "diagnosis", "statute", "O(n log n)") that is absent or rare
in the baseline.

---

## Step 5 -- Tab 3: RAG / Document-Grounded Steering

Upload PDF documents and steer the model toward **document-grounded** answers
rather than vague or hallucinated content.

### Features

- **Document source**: upload PDF(s) or use pre-loaded samples from
  `examples/rag-data/`
- **Text extraction and chunking**: automatic PDF parsing with configurable
  chunk size
- **Contrast pair construction**: document chunks become positive examples;
  generic vague statements become negatives
- **Expandable source passages**: view which chunks are used for steering
- **Side-by-side output**: baseline (ungrounded) vs document-grounded generation

### Step-by-step

1. Select tab **RAG / Document Grounding**
2. Choose document source:
   - **Upload PDF(s)**: drag and drop one or more PDF files
   - **Use pre-loaded sample PDFs**: select from files in `examples/rag-data/`
3. Confirm the success message shows loaded file names
4. Configure:
   - **Words per chunk**: `250` (how large each text chunk is)
   - **Positive examples (chunks)**: `8` (how many chunks to use for steering)
   - **Steering layer**: ~65% depth
   - **Alpha**: `2.5`
   - **Max new tokens**: `150`
5. Enter a question about the document content, e.g.:
   `What are the key strategic priorities and recommendations described in this document?`
6. Click **Extract, Steer & Generate**
7. Progress steps execute:
   - Text extraction from PDF(s) with chunk counts displayed
   - Contrast pair construction
   - Steering vector discovery
   - Baseline and grounded generation
8. Compare the two columns:
   - **Baseline**: generic, potentially hallucinated answer
   - **Grounded**: references specific facts, terminology, and details
     from your documents

**What to look for**: the grounded output should cite specific facts,
names, numbers, or concepts that exist in the uploaded document but
not in the baseline.

> **Prerequisite**: `pip install PyPDF2 pdfplumber` for PDF parsing.

---

## Step 6 -- Tab 4: Alpha Sweep

Sweep through a range of steering strengths to find the optimal alpha where
steering is strong but coherent.

### Features

- **Vector source toggle**: choose `Preset` (auto-discovers a vector) or
  `From Playground` (reuses the last vector from Tab 1)
- **Preset mode**: select any preset, pick the target layer, view contrast
  pairs -- vector is discovered automatically when you run the sweep
- **Configurable range**: alpha min, alpha max, step size
- **Results table**: one row per alpha value showing the generated output

### Step-by-step

1. Select tab **Alpha Sweep**
2. Set **Vector source** = `Preset`
3. Choose a preset (e.g., **Positive / Helpful**)
4. Review contrast pairs in the expander
5. Select target layer (auto-set from preset)
6. Set prompt: `Tell me about yourself.`
7. Set range: alpha min = `-3.0`, alpha max = `3.0`, step = `1.0`
8. Set Max new tokens = `60`
9. Click **Run Sweep**
10. The vector is discovered first (progress spinner), then each alpha
    value is generated
11. Read the results table row by row:
    - **Negative alpha rows**: output shifts toward the "negative" contrast direction
    - **Alpha = 0**: essentially baseline output
    - **Positive alpha rows**: increasingly steered toward "positive" direction

**What to look for**: find the alpha value where the steering is
strong but still coherent. Too high (above ~4.0) often degrades
output quality into repetition or nonsense.

> **Tip**: You can also switch to `From Playground` to reuse a vector
> you already discovered in Tab 1, without re-running discovery.

---

## Step 7 -- Tab 5: Composition

Combine multiple steering vectors using weighted sum and inspect their
interactions via cosine similarity and conflict detection.

### Features

- **Multi-vector builder**: configure 2-5 vectors, each with its own
  preset, layer, and weight
- **Cosine similarity matrix**: shows how similar each pair of vectors is
- **Conflict detection**: alerts when vectors point in opposing directions
- **Composed vector stats**: magnitude, layer, and dimension of the result
- **Side-by-side generation**: baseline vs composed-steered output with
  configurable alpha and max tokens (persists after composing)

### Step-by-step

1. Select tab **Composition**
2. Set **Number of vectors** = `3`
3. Configure each column:
   - Vector 1: preset = **Positive / Helpful**, layer = auto, weight = `1.0`
   - Vector 2: preset = **Formal / Professional**, layer = same, weight = `0.7`
   - Vector 3: preset = **Concise / Direct**, layer = same, weight = `0.5`
4. Click **Discover & Compose**
5. Progress bar shows each vector being discovered
6. Review the output panels:

   | Panel | What it shows |
   |-------|--------------|
   | **Cosine similarity matrix** | How similar each pair of vectors is. High similarity (> 0.8) means they point in the same direction. |
   | **Conflict detection** | Alerts on vectors with high negative cosine similarity (opposing forces). |
   | **Composed vector stats** | Magnitude, layer, and shape of the weighted sum. |

7. After composing, the **Generate with Composed Vector** section appears below
8. Enter prompt: `Explain what machine learning is.`
9. Set steering strength (alpha) = `2.0` and max tokens = `120`
10. Click **Generate with Composed Vector**
11. Compare the two columns:
    - **Baseline**: standard model output
    - **Composed Steering**: look for the combined effect -- helpful tone +
      formal register + concise phrasing

**Experiment**: set Vector 1 to **Safety / Harmless** and Vector 2 to
**Creative / Imaginative** and observe the conflict score -- opposing
personas create measurable conflict in the similarity matrix.

> **Important**: all vectors must target the **same layer** to be composed.
> If layers differ, the tab will show a message asking you to adjust.

---

## Step 8 -- Tab 6: Inspector

Examine the internal tensor statistics of any discovered or composed vector,
and export vectors for use in code.

### Features

- **Three inspection sources**: last discovered vector, composed vector,
  or load from disk
- **Metadata display**: model name, layer, method, dimension, magnitude,
  dtype, creation timestamp, and any extra metadata
- **Tensor statistics**: mean, std, min, max, abs-mean, non-zero %
- **Value distribution histogram**: shows the shape of the steering direction
- **Export buttons**: download as JSON (full metadata + tensor values) or
  NumPy `.npy` (lightweight tensor only)

### Step-by-step

1. Select tab **Inspector**
2. Choose what to inspect:
   - **Last discovered vector** (from Tone, Role, RAG, or Layer Explorer tabs)
   - **Composed vector** (from Composition tab)
   - **Load from disk** (pick from `demo/saved_vectors/`)
3. Review the **Metadata** section:
   - Model, Layer, Method, Dimension, Magnitude, Dtype
4. Review the **Tensor Statistics** panel:
   - Six metrics displayed in a 3-column layout
5. Examine the **Value Distribution** histogram:
   - Sharp, narrow distribution = focused steering concept
   - Wide spread = diffuse, less targeted steering
6. Use the **Export** section to download:
   - **JSON**: full metadata + tensor values (for programmatic use)
   - **.npy**: NumPy tensor file (lightweight, easy to reload in code)

**Interpretation guide**:
- Magnitude ~1-5: typical healthy range
- Very large magnitude (> 20): may produce incoherent output at alpha > 1
- Near-zero magnitude: vector may not have captured meaningful signal
- High non-zero %: dense vector (most dimensions contribute)
- Low non-zero %: sparse vector (few dimensions dominate)

---

## Step 9 -- Tab 7: Layer Explorer

Find which transformer layers produce the strongest and most coherent
steering effect for a given preset.

### Features

- **Preset selector**: pick any tone or role preset
- **Layer stride control**: test every Nth layer (lower = more thorough, slower)
- **Results table**: one row per tested layer showing the layer number,
  vector magnitude, and generated output

### Step-by-step

1. Select tab **Layer Explorer**
2. Choose preset = **Positive / Helpful**
3. Set prompt: `Describe artificial intelligence in one paragraph.`
4. Set alpha = `2.0`, max tokens = `60`
5. Set **Layer stride** = auto (or manually set to `2` for faster sweep)
6. Review the "Testing layers" caption to see which layers will be tested
7. Click **Run Layer Sweep**
8. Progress bar shows each layer being discovered and tested
9. Read the results table:
   - **Layer**: layer index in the transformer
   - **Magnitude**: L2 norm of the discovered vector at that layer
   - **Output**: the steered generation from that layer

**What to look for**:
- **Early layers (0-10%)**: usually weak or no steering effect
- **Mid layers (40-70% depth)**: typically strongest and most coherent steering
- **Late layers (80-100% depth)**: can produce strong but sometimes incoherent results
- For Phi-2 (32 layers): layers 18-24 usually work best
- For GPT-2 (12 layers): layers 6-9 usually work best

**Use this tab to**: find the optimal layer for a new custom concept
before doing fine-tuned steering in the Tone / Personality tab.

---

## Recommended Demo Script (15 minutes)

Run these in order for a comprehensive demonstration:

| # | Tab | What to show | Preset / Config | Prompt | Point to make |
|---|-----|-------------|-----------------|--------|---------------|
| 1 | Tone / Personality | Basic tone steering | Positive / Helpful, alpha=2.0 | "Tell me about yourself." | One-click tone shift |
| 2 | Tone / Personality | Reverse direction | Same preset, alpha=-2.0 | Same | Same vector, opposite behavior |
| 3 | Tone / Personality | Style steering | Formal / Professional, alpha=1.5 | "What is AI?" | Not just sentiment -- style too |
| 4 | Role Expertise | Domain persona | Doctor, alpha=2.5 | (default clinical scenario) | Expert vocabulary emerges |
| 5 | Role Expertise | Different domain | Software Engineer, alpha=2.0 | (default architecture prompt) | Role-specific reasoning |
| 6 | Alpha Sweep | Continuous control | Preset: Positive/Helpful, -3 to 3 | "Tell me about yourself." | Alpha as a dial |
| 7 | Composition | Multi-vector | Helpful + Formal + Concise | "Explain ML." | Combine behaviors |
| 8 | Inspector | Interpretability | (last vector) | -- | What the vector looks like inside |
| 9 | Layer Explorer | Layer sensitivity | Positive / Helpful, stride=2 | "Describe AI." | Where steering works in the network |
| 10 | RAG / Document Grounding | Document steering | Upload a PDF | Question about the document | Grounding reduces hallucination |

---

## Stopping the Demo

Press `Ctrl+C` in the terminal running `demo/launch.py` to stop the server.

To restart:

```powershell
python demo/launch.py --port 8501
```

---

## Quick Reference: All Presets

### Tone Presets

| Preset | Effect | Default Alpha | Best Layer % |
|--------|--------|--------------|-------------|
| Positive / Helpful | Warm, encouraging tone | 2.0 | 60% |
| Formal / Professional | Academic, structured style | 1.5 | 60% |
| Concise / Direct | Short, to-the-point answers | 2.5 | 50% |
| Creative / Imaginative | Vivid, metaphorical language | 2.0 | 65% |
| Safety / Harmless | Responsible, careful phrasing | 3.0 | 55% |

### Role Presets

| Preset | Effect | Default Alpha | Best Layer % |
|--------|--------|--------------|-------------|
| Doctor / Medical Professional | Clinical vocabulary, evidence-based | 2.5 | 65% |
| Lawyer / Legal Expert | Legal terminology, case citations | 2.5 | 65% |
| Software Engineer / Technical Expert | Technical precision, best practices | 2.0 | 60% |
| Teacher / Educator | Patient explanations, encouragement | 2.5 | 60% |
| Business Consultant | Strategic thinking, ROI focus | 2.0 | 65% |
| Scientist / Researcher | Empirical rigor, intellectual humility | 2.5 | 65% |

---

**Last Updated**: February 26, 2026
