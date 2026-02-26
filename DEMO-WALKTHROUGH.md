# SteeringLLM Demo Walkthrough

Step-by-step guide to launching the app and exercising every feature.

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
python demo/launch.py --port 8503
```

Open your browser to **http://localhost:8503**

> Always use `demo/launch.py` instead of `streamlit run demo/app.py` on Windows.
> The launcher pre-loads PyTorch before Streamlit's pyarrow DLLs to avoid the
> `WinError 1114 / c10.dll` conflict.

---

## Step 2 -- Load the Model

1. In the **left sidebar**, the model field defaults to `gpt2`
2. Click **Load Model** (blue button)
3. Wait for the download (~500 MB on first run, cached after that)
4. Confirm the sidebar shows model info:
   - Architecture: `gpt2`
   - Layers: 12
   - Hidden dim: 768
   - Parameters: 117M

> To test a larger model, type `gpt2-medium` or `gpt2-large` before clicking Load.
> No HuggingFace credentials are needed for any GPT-2 variant.

---

## Step 3 -- Playground Tab (Core Feature)

The Playground discovers a steering vector and shows baseline vs. steered
output side-by-side.

### 3a. Try "Positive / Helpful" preset

1. Select tab **Playground**
2. Set **Vector source** = `Preset`
3. Choose preset = **Positive / Helpful**
4. Review the contrast pairs (expand "View contrast pairs")
5. Leave default settings:
   - Layer: auto-set to ~60% depth (layer 7)
   - Steering strength (alpha): `2.0`
   - Max new tokens: `100`
6. Enter prompt: `Tell me about yourself and how you see the world.`
7. Click **Generate**
8. Compare the two columns:
   - **Baseline**: neutral GPT-2 output
   - **Steered**: noticeably warmer, more encouraging tone

### 3b. Try negative alpha (reverse direction)

1. Change alpha to `-2.0` using the slider
2. Click **Generate** again
3. Observe: steered output becomes harsher/more negative -- the vector
   is being applied in reverse

### 3c. Try "Formal / Professional" preset

1. Change preset to **Formal / Professional**
2. Set alpha = `1.5`
3. Use prompt: `What do you think about the future of technology?`
4. Click **Generate**
5. Steered output should adopt formal vocabulary and structured phrasing

### 3d. Try "Creative / Imaginative" preset

1. Change preset to **Creative / Imaginative**
2. Set alpha = `2.0`, layer = `65%` (auto-set)
3. Use prompt: `Describe how a computer works.`
4. Click **Generate**
5. Steered output should use metaphorical, vivid language vs. the
   literal baseline

### 3e. Try a Custom vector

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

### 3f. Save the vector

After any generation, click **Save vector** to persist it to
`demo/saved_vectors/` as `.json` + `.pt` files.

---

## Step 4 -- Alpha Sweep Tab

Shows how output changes across a full range of steering strengths.

> **Requires**: at least one vector generated in the Playground tab first.

1. Select tab **Alpha Sweep**
2. Set prompt: `Tell me about yourself.`
3. Set range: alpha min = `-3.0`, alpha max = `3.0`, step = `1.0`
4. Set Max new tokens = `60`
5. Click **Run Sweep**
6. A table renders -- read row by row to see:
   - Negative alpha rows: output shifts toward the "negative" contrast direction
   - Alpha = 0: essentially baseline output
   - Positive alpha rows: increasingly steered toward "positive" direction

**What to look for**: find the alpha value where the steering is
strong but still coherent. Too high (above ~4.0) often degrades
output quality.

---

## Step 5 -- Vector Composition Tab

Compose multiple steering vectors together and inspect their interactions.

1. Select tab **Composition**
2. Set **Number of vectors** = `3`
3. Configure each column:
   - Vector 1: preset = **Positive / Helpful**, weight = `1.0`
   - Vector 2: preset = **Formal / Professional**, weight = `0.7`
   - Vector 3: preset = **Concise / Direct**, weight = `0.5`
4. Click **Build & Compose Vectors**
5. Review the output panels:

   | Panel | What it shows |
   |-------|--------------|
   | **Cosine similarity matrix** | How similar each pair of vectors is. High similarity (> 0.8) means they point in the same direction. |
   | **Conflict detection** | Alerts on vectors with high negative cosine similarity (opposing forces). |
   | **Composed vector stats** | Magnitude and shape of the weighted sum. |

6. Enter prompt: `Explain what machine learning is.`
7. Click **Generate with composed vector**
8. Compare baseline vs. the multi-vector steered output -- look for
   the combined effect: helpful tone + formal register + concise phrasing

**Experiment**: flip Vector 1 to **Safety / Harmless** (alpha = `3.0`)
and observe the conflict score change relative to a **Creative**
vector -- opposing personas create high conflict.

---

## Step 6 -- Inspector Tab

Examine the internal tensor statistics of any discovered vector.

1. Select tab **Inspector**
2. The last generated vector is pre-selected
3. Review the stats panel:
   - **Mean / Std / Min / Max**: distribution of steering direction values
   - **Magnitude**: L2 norm of the vector
   - **Layer**: which transformer layer this vector targets
4. Examine the **histogram**: a sharp, narrow distribution means the
   vector captures a focused concept; wide spread means diffuse steering
5. Click **Download vector (JSON)** to export for use in code

**Interpretation guide**:
- Magnitude ~1-5: typical healthy range for GPT-2
- Very large magnitude (> 20): may produce incoherent output at alpha > 1
- Near-zero magnitude: vector may not have captured meaningful signal

---

## Step 7 -- Layer Explorer Tab

Find which transformer layers produce the strongest steering effect.

1. Select tab **Layer Explorer**
2. Use the **Positive / Helpful** preset (or whichever was active)
3. Set prompt: `How do you feel today?`
4. Set alpha = `2.0`, max new tokens = `60`
5. Set **Layer stride** = `2` (test every 2nd layer, faster)
6. Click **Run Layer Explorer**
7. A table and chart appear comparing output at each tested layer

**What to look for**:
- Early layers (0-3): usually weak or no steering effect
- Mid layers (40-70% depth): typically strongest and most coherent steering
- Late layers (80-100% depth): can produce strong but sometimes incoherent results
- For GPT-2 (12 layers): layers 6-9 usually work best

**Use this tab to**: find the optimal layer for a new custom concept
before doing fine-tuned steering in the Playground.

---

## Step 8 -- Recommended Test Sequence (Demo Script)

Run these in order for a complete 10-minute demonstration:

| # | Tab | Preset | Prompt | Alpha | Point to make |
|---|-----|--------|--------|-------|---------------|
| 1 | Playground | Positive / Helpful | "Tell me about yourself." | 2.0 | Basic steering -- tone shift in one click |
| 2 | Playground | Positive / Helpful | same | -2.0 | Reversal -- same vector, opposite direction |
| 3 | Playground | Formal / Professional | "What is AI?" | 1.5 | Style steering -- not just sentiment |
| 4 | Playground | Concise / Direct | "Explain quantum computing." | 2.5 | Length/verbosity steering |
| 5 | Alpha Sweep | (last vector) | "Tell me about yourself." | -3 to 3 | Continuous control of behavior |
| 6 | Composition | Helpful + Formal + Concise | "Explain ML." | mixed | Multi-dimensional steering |
| 7 | Inspector | (last vector) | -- | -- | Interpretability of the direction |
| 8 | Layer Explorer | Positive / Helpful | "How do you feel?" | 2.0 | Where in the network steering works |

---

## Stopping the Demo

Press `Ctrl+C` in the terminal running `demo/launch.py` to stop the server.

To restart:

```powershell
python demo/launch.py --port 8503
```

---

## Quick Reference: All Presets

| Preset | Effect | Default Alpha | Best layer % |
|--------|--------|--------------|-------------|
| Positive / Helpful | Warm, encouraging tone | 2.0 | 60% |
| Formal / Professional | Academic, structured style | 1.5 | 60% |
| Concise / Direct | Short, to-the-point answers | 2.5 | 50% |
| Creative / Imaginative | Vivid, metaphorical language | 2.0 | 65% |
| Safety / Harmless | Responsible, careful phrasing | 3.0 | 55% |

---

**Last Updated**: February 25, 2026
