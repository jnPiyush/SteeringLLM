# SteeringLLM Demo - Use Cases & Technical Walkthrough

> **Audience**: Demo presenters, stakeholders, and technical evaluators.
> This document explains what each demo tab does, what happens behind the
> scenes at every step, and what the audience should look for.

---

## Table of Contents

1. [What Is Activation Steering?](#what-is-activation-steering)
2. [How the Demo Works (High-Level)](#how-the-demo-works-high-level)
3. [Use Case 1 - Tone / Personality Steering](#use-case-1---tone--personality-steering)
4. [Use Case 2 - Role / Domain Expertise](#use-case-2---role--domain-expertise)
5. [Use Case 3 - RAG / Document-Grounded Steering](#use-case-3---rag--document-grounded-steering)
6. [Advanced Tabs](#advanced-tabs)
7. [Key Parameters Explained](#key-parameters-explained)
8. [Architecture at a Glance](#architecture-at-a-glance)
9. [Running the Demo](#running-the-demo)
10. [FAQ](#faq)

---

## What Is Activation Steering?

Large language models (LLMs) encode knowledge and behavioral tendencies across
layers of transformer blocks. **Activation steering** modifies the model's
behavior *at inference time* -- without fine-tuning or retraining -- by adding
a small vector to the model's internal activations as text flows through a
chosen layer.

### The Core Idea in Three Steps

```
1. DISCOVER   -->   2. INJECT   -->   3. GENERATE
```

| Step | What Happens | Analogy |
|------|-------------|---------|
| **Discover** | Feed the model "positive" examples (desired behavior) and "negative" examples (undesired behavior). Record internal activations at a chosen layer. Compute `mean(positive) - mean(negative)` to get a direction vector. | Finding the compass heading between "formal" and "informal" in the model's internal space. |
| **Inject** | Register a PyTorch forward hook on the target layer. During inference, the hook adds `alpha * steering_vector` to every token's hidden state as it passes through that layer. | Gently nudging every thought the model has in the direction you want. |
| **Generate** | The model produces text as usual, but each token prediction is influenced by the injected direction. The result: text that exhibits the desired behavior shift. | The model "thinks" the same way, but with a consistent bias toward your chosen behavior. |

### Why This Matters

- **No retraining** -- works on any compatible model instantly.
- **Composable** -- multiple vectors can be combined (e.g., "formal" + "helpful").
- **Reversible** -- remove the hook and the model returns to baseline.
- **Interpretable** -- you can inspect vector magnitude, layer impact, and cosine similarity.

---

## How the Demo Works (High-Level)

When you run `python demo/launch.py`:

1. **PyTorch preloads** -- The launcher imports `torch` before Streamlit starts
   to avoid a Windows DLL conflict between pyarrow and torch.
2. **Streamlit UI starts** -- A browser opens at `http://localhost:8501`.
3. **Model loads** -- GPT-2 (124M parameters) downloads from Hugging Face on
   first run (~500 MB) and loads into CPU RAM. The model is cached -- subsequent
   launches are instant.
4. **Architecture detection** -- SteeringLLM inspects the model's `model_type`
   (`gpt2`) and looks it up in the architecture registry to find the layer
   structure (`transformer.h` with 12 layers for `gpt2`, 36 for `gpt2-large`).
5. **Seven tabs render** -- Each tab provides a different interaction with the
   steering system.

### Tab Layout

| Tab | Purpose | Complexity |
|-----|---------|-----------|
| Tone / Personality | Show behavioral tone shifts | Simple -- best for opening |
| Role Expertise | Show domain-specific persona injection | Medium -- great "wow" moment |
| RAG / Document Grounding | Show document-aware generation | Advanced -- strongest differentiator |
| Alpha Sweep | Explore steering strength range | Power user |
| Composition | Combine multiple steering vectors | Power user |
| Inspector | Examine vector metadata and statistics | Power user |
| Layer Explorer | Test steering at different layers | Power user |

---

## Use Case 1 - Tone / Personality Steering

**Tab**: Tone / Personality

### What the Audience Sees

A side-by-side comparison: the left column shows the model's **baseline**
output (no steering), and the right column shows the **steered** output.
The difference in tone -- formal vs. casual, helpful vs. neutral, creative
vs. dry -- is immediately visible.

### Demo Flow (Step by Step)

```
[1] Select a tone preset (e.g., "Formal / Professional")
         |
[2] Review the contrast pairs (positive = formal phrases, negative = slang)
         |
[3] Adjust alpha slider (default: 1.5 for Formal)
         |
[4] Click "Generate"
         |
[5] Behind the scenes:
         |
         +--> Discovery.mean_difference() runs:
         |      - Tokenizes all 8 positive examples, runs forward pass
         |      - Records activations at layer N (e.g., layer 7 for gpt2)
         |      - Tokenizes all 8 negative examples, runs forward pass
         |      - Records activations at layer N
         |      - Computes: vector = mean(pos_activations) - mean(neg_activations)
         |      - Returns a SteeringVector (shape: [hidden_dim], e.g., [768])
         |
         +--> Baseline generation:
         |      - model.remove_steering()  (ensure clean state)
         |      - model.generate(prompt, max_new_tokens=150)
         |      - Standard autoregressive generation, no modification
         |
         +--> Steered generation:
         |      - model.generate_with_steering(prompt, vector, alpha=1.5)
         |      - Internally: registers a forward hook on transformer.h[7]
         |      - Hook function: hidden_states += alpha * vector
         |      - Every token position at that layer gets the same nudge
         |      - After generation completes, the hook is automatically removed
         |
[6] Both outputs display side-by-side
```

### Available Tone Presets

| Preset | Effect | Default Alpha | What to Look For |
|--------|--------|--------------|-----------------|
| **Positive / Helpful** | Encouraging, supportive tone | 2.0 | Warm language, offers help, positive framing |
| **Formal / Professional** | Academic, structured prose | 1.5 | Formal vocabulary, passive voice, structured sentences |
| **Concise / Direct** | Short, to-the-point answers | 2.5 | Shorter sentences, bulleted style, less filler |
| **Creative / Imaginative** | Vivid, poetic language | 2.0 | Metaphors, sensory language, narrative style |
| **Safety / Harmless** | Cautious, responsible outputs | 3.0 | Disclaimers, balanced framing, safety caveats |

### What to Highlight to the Audience

- "Notice the baseline uses casual, generic language. The steered version uses
  formal vocabulary like 'herein', 'empirical evidence', and 'pursuant to'."
- "We did not retrain anything. This is the exact same GPT-2 model. We just
  added a direction vector at layer 7 during inference."
- "The alpha slider controls intensity. Watch what happens at alpha=0
  (identical to baseline) vs. alpha=5 (very strong, may over-steer)."

---

## Use Case 2 - Role / Domain Expertise

**Tab**: Role Expertise

### What the Audience Sees

The model responds as if it were a specific domain expert -- a doctor using
medical terminology, a lawyer citing legal principles, or an engineer
discussing architecture patterns. The baseline shows generic language; the
steered output shows domain-specific vocabulary and reasoning.

### Demo Flow (Step by Step)

```
[1] Select an expert role (e.g., "Doctor / Medical Professional")
         |
[2] The UI automatically loads:
         |   - Role-specific contrast pairs (medical language vs. casual dismissal)
         |   - A domain-appropriate prompt:
         |     "A patient presents with persistent headaches, blurred
         |      vision, and fatigue for two weeks..."
         |
[3] Click "Generate"
         |
[4] Behind the scenes:
         |
         +--> Same Discovery.mean_difference() process as tone steering
         |      - Positive examples: medical/clinical language
         |        ("Based on your symptoms, I recommend...")
         |      - Negative examples: dismissive/non-expert language
         |        ("Just google your symptoms and self-diagnose")
         |      - Vector captures the "medical expertise" direction
         |
         +--> Baseline: model answers the medical prompt generically
         |
         +--> Steered: model answers with medical terminology,
         |    clinical reasoning, and professional framing
         |
[5] Side-by-side display with comparison guidance
```

### Available Role Presets

| Role | Default Prompt Theme | What to Look For |
|------|---------------------|-----------------|
| **Doctor / Medical Professional** | Patient with headaches, blurred vision, elevated BP | Clinical terms (differential diagnosis, CBC, fundoscopic exam), empathetic but professional tone |
| **Lawyer / Legal Professional** | Tenant with security deposit dispute | Legal terminology (statutory provisions, breach of contract, remedies), structured arguments |
| **Software Engineer** | Microservices architecture design | Technical terms (load balancing, service mesh, API gateway), systematic reasoning |
| **Teacher / Educator** | Explaining quantum computing to beginners | Simplified explanations, analogies, step-by-step progression, encouragement |
| **Business Consultant** | Market expansion strategy evaluation | Business frameworks (SWOT, TAM, competitive analysis), data-driven recommendations |
| **Scientist / Researcher** | Interpreting experimental results | Scientific method language, caveats, peer-review references, hypothesis testing |

### What to Highlight to the Audience

- "The baseline might mention headaches generically. The steered output
  discusses differential diagnosis, recommends a CBC panel, and suggests
  ophthalmologic referral -- all from a 768-dimensional vector added at
  one layer."
- "Each role has custom contrast pairs. The 'Doctor' vector was built by
  contrasting clinical, empathetic medical language against dismissive
  non-expert responses. The model learns the *direction* of expertise."
- "This works because LLMs encode domain knowledge across their parameters
  during pretraining. The steering vector activates and amplifies the
  medical knowledge pathways that already exist in the model."

---

## Use Case 3 - RAG / Document-Grounded Steering

**Tab**: RAG / Document Grounding

### What the Audience Sees

The user uploads a PDF document (or selects a pre-loaded sample). The model's
steered output references specific facts, terminology, and themes from the
document, while the baseline produces generic or hallucinated content.

### Important Context

This is not traditional RAG (Retrieval-Augmented Generation) with a vector
database and retrieved context injected into the prompt. Instead, it uses
**activation steering** to bias the model's generation toward the document's
language and concepts. The document chunks become the "positive examples" --
the factual grounding -- while generic vague statements serve as "negative
examples." The resulting vector steers the model toward document-aligned output.

### Demo Flow (Step by Step)

```
[1] Select document source:
         |   Option A: Upload your own PDF(s)
         |   Option B: Use pre-loaded sample PDFs (examples/rag-data/)
         |
[2] Configure chunking:
         |   - Words per chunk (default: 250 words)
         |   - Number of positive examples/chunks (default: 8)
         |
[3] Enter a question about the document
         |
[4] Click "Extract, Steer & Generate"
         |
[5] Behind the scenes -- 4-stage pipeline:
         |
         +--> STAGE 1: PDF Text Extraction
         |      - pdfplumber extracts text (falls back to PyPDF2)
         |      - Text is cleaned: collapse whitespace, strip page numbers
         |      - Result: raw text string from the entire PDF
         |
         +--> STAGE 2: Chunking
         |      - Text is split into overlapping word-based chunks
         |      - chunk_size=250 words, overlap=60 words
         |      - Minimum chunk size: 80 words (short fragments discarded)
         |      - Example: a 10-page PDF might produce ~30-40 chunks
         |
         +--> STAGE 3: Contrast Pair Building
         |      - POSITIVE examples = chunks sampled evenly across the document
         |        (8 chunks by default, spread for broad coverage)
         |      - NEGATIVE examples = pre-defined generic/vague statements:
         |        "I don't have specific information about that topic."
         |        "Without access to the details, I can only speculate."
         |        "My knowledge here is limited to very general concepts."
         |      - The contrast: document-specific facts vs. vague non-answers
         |
         +--> STAGE 4: Vector Discovery + Generation
         |      - Discovery.mean_difference(positives, negatives, model, layer)
         |      - Produces a vector pointing from "vague/unknowing" toward
         |        "specific/document-informed"
         |      - Baseline: model generates without the vector (generic output)
         |      - Steered: model generates with the vector (document-grounded)
         |
[6] Side-by-side display:
         |   - Left: Baseline (generic, possibly hallucinated)
         |   - Right: Grounded (uses document terminology and themes)
         |   - Expandable "Source passages" section shows which chunks
         |     were used as positive examples
```

### Pre-Loaded Sample PDFs

The demo ships with sample PDFs in `examples/rag-data/` for immediate testing:

| PDF | Content |
|-----|---------|
| FY26-Cloud.pdf | Cloud strategy and priorities |
| AI-Business-Solutions-FY26-Partner-Playbook.pdf | AI business solutions playbook |
| FY26-Security-Commercial-Partner-Playbook-November-2025.pdf | Security commercial partner strategies |
| Microsoft_GPS_Agentic_AI_playbook_Final_2025.08.26.pdf | Agentic AI playbook and guidance |

### What to Highlight to the Audience

- "The baseline has no knowledge of this PDF. It produces generic statements
  about the topic. The steered version uses *specific terminology, figures,
  and priorities* from the document -- without putting the PDF text in the
  prompt."
- "This is fundamentally different from traditional RAG. In traditional RAG,
  you inject retrieved text into the prompt. Here, we modify the model's
  *internal activations* so it naturally gravitates toward the document's
  language and concepts."
- "Expand the 'Source passages' section to see the exact PDF chunks used as
  positive examples. Compare those to what appears in the steered output."
- "This approach works because the model already has general knowledge about
  these topics from pretraining. The steering vector amplifies the pathways
  that align with the document's specific framing and data."

---

## Advanced Tabs

### Alpha Sweep

**What it does**: Generates text at multiple alpha values (e.g., -3 to +3)
in a single run, showing a table of outputs.

**Behind the scenes**: For each alpha value, the system calls
`model.generate_with_steering()` with a different alpha multiplier. The same
steering vector is reused -- only the strength changes.

**What to show**: Alpha = 0 matches the baseline. Negative alpha reverses the
direction (e.g., a "formal" vector at alpha = -2 produces informal output).
Very high alpha (>4) may cause degenerate/repetitive text.

### Composition

**What it does**: Builds 2-5 steering vectors from different presets,
displays their cosine similarity matrix, detects potential conflicts, and
combines them via weighted sum.

**Behind the scenes**: Uses `VectorComposition.weighted_sum()` to linearly
combine vectors. Cosine similarity reveals whether vectors are orthogonal
(independent), aligned (reinforcing), or opposing (conflicting).

**What to show**: Combining "Formal" + "Concise" produces professional but
brief output. Combining "Formal" + "Creative" may conflict since formality
and creativity use different language registers.

### Inspector

**What it does**: Displays metadata (model, layer, method, dimension,
magnitude), tensor statistics (mean, std, min, max), and a value-distribution
histogram for the current steering vector.

**Behind the scenes**: Reads the `SteeringVector` object's metadata and runs
basic `torch.Tensor` statistics. Vectors can be saved to disk (`.json` +
`.pt` files) and reloaded later.

### Layer Explorer

**What it does**: Injects the same vector at every Nth layer and compares
the generated output, revealing which layers have the most steering impact.

**Behind the scenes**: For each layer index, a new steering vector is
discovered (same contrast pairs, different layer) and used for generation.
This shows that mid-to-late layers (50-70% depth) typically have the strongest
effect, while early layers have minimal impact.

---

## Key Parameters Explained

| Parameter | Range | Effect | Guidance |
|-----------|-------|--------|----------|
| **Alpha** | -5.0 to +5.0 | Steering strength multiplier. 0 = no effect, negative = reverse direction | Start with the preset default (1.5-3.0). Alpha > 4 often causes repetition. |
| **Layer** | 0 to (N-1) | Which transformer layer to inject the vector | 50-70% depth works best. For GPT-2 (12 layers), layers 6-8. For GPT-2-large (36 layers), layers 18-25. |
| **Temperature** | 0.1 to 2.0 | Controls randomness in token sampling | 0.7 is a good default. Lower = more deterministic, higher = more creative/chaotic. |
| **Max New Tokens** | 20 to 300 | Maximum tokens to generate | 150 for demos. More tokens = slower generation on CPU. |
| **Chunk Size** (RAG) | 100 to 500 words | How large each document chunk is | 250 is a good balance. Larger chunks carry more context but fewer fit as positive examples. |

---

## Architecture at a Glance

```
User (Browser)
     |
     v
Streamlit UI (demo/app.py)
     |
     +-- demo/presets.py        <-- Contrast pair definitions (tone + role)
     +-- demo/pdf_utils.py      <-- PDF extraction, chunking, contrast pairs (RAG)
     |
     v
steering_llm Library (src/steering_llm/)
     |
     +-- SteeringModel          <-- Model wrapper with hook management
     |     |
     |     +-- from_pretrained() <-- Load any HuggingFace causal LM
     |     +-- apply_steering()  <-- Register forward hook on target layer
     |     +-- generate_with_steering() <-- Generate with temporary steering
     |     +-- remove_steering() <-- Clean up hooks
     |
     +-- Discovery              <-- Vector discovery algorithms
     |     |
     |     +-- mean_difference() <-- mean(pos) - mean(neg) activations
     |     +-- linear_probe()    <-- Logistic regression boundary direction
     |
     +-- SteeringVector         <-- Tensor + metadata (layer, model, method)
     |     |
     |     +-- save() / load()   <-- Persist to .json + .pt files
     |     +-- magnitude         <-- L2 norm of the vector
     |
     +-- VectorComposition      <-- Combine multiple vectors
           |
           +-- weighted_sum()    <-- Linear combination
           +-- cosine_similarity() <-- Similarity analysis
```

### How the Forward Hook Works

This is the core mechanism. When steering is active:

```
Input tokens
     |
     v
[Layer 0] --> [Layer 1] --> ... --> [Layer N (hooked)] --> ... --> [Final Layer]
                                         |
                                    hidden_states += alpha * vector
                                         |
                                    Modified hidden_states continue
                                    through remaining layers
                                         |
                                         v
                                    Token prediction is influenced
                                    by the steering direction
```

The hook function (in `ActivationHook.register()`):
1. Intercepts the output of the target layer (a tensor of shape
   `[batch, sequence_length, hidden_dim]`).
2. Adds `alpha * steering_vector` (shape `[hidden_dim]`) to every token
   position via broadcasting.
3. Returns the modified tensor so the rest of the model uses the steered
   activations.

---

## Running the Demo

### Prerequisites

```bash
# Python 3.9+ required
python --version

# Install the package with demo dependencies
pip install -e ".[demo]"
```

### Launch

```bash
# Recommended (handles Windows DLL conflicts)
python demo/launch.py

# Optional: specify a different port
python demo/launch.py --port 8502
```

The demo opens at **http://localhost:8501**.

### First Run

- GPT-2 (~500 MB) downloads from Hugging Face on first launch.
- Subsequent launches use the cached model (instant load).
- For better steering quality, type `gpt2-large` in the sidebar and click
  **Load Model** (~800 MB download, 36 layers instead of 12).

### Model Comparison

| Model | Parameters | Layers | Hidden Dim | Download | Steering Quality |
|-------|-----------|--------|-----------|----------|-----------------|
| `gpt2` | 124M | 12 | 768 | ~500 MB | Good for demos, fast |
| `gpt2-medium` | 355M | 24 | 1024 | ~1.5 GB | Better distinction |
| `gpt2-large` | 774M | 36 | 1280 | ~3 GB | Best quality, slower |

---

## FAQ

### Q: Is this the same as prompt engineering?

No. Prompt engineering changes the *input* -- the text the model reads.
Activation steering changes the model's *internal computations* -- modifying
hidden state vectors inside the transformer as they flow through the network.
The prompt stays the same in both baseline and steered outputs; only the
internal activations differ.

### Q: Is this fine-tuning?

No. Fine-tuning modifies the model's weights permanently. Activation steering
adds a temporary forward hook that injects a direction vector during
inference. Remove the hook and the model is exactly as it was before. No
weights are changed.

### Q: How is the RAG tab different from "real" RAG?

Traditional RAG retrieves relevant text chunks and injects them into the
prompt (context window). The model reads the retrieved text and can quote it
directly. Our approach uses the document chunks as *training signal* for
a steering direction -- the model's activations are biased toward the
document's language patterns and concepts. It is more of a "style and content
grounding" approach. It will not produce exact quotes but will align the
model's output with the document's themes and terminology.

### Q: Why do some presets use higher alpha than others?

Different behaviors require different steering strengths. Subtle shifts
(formality) need less force (alpha 1.5). Dramatic shifts (safety, conciseness)
need more (alpha 2.5-3.0). The presets include recommended alpha values
calibrated for GPT-2 family models.

### Q: Can I use this with models other than GPT-2?

Yes. The architecture registry supports GPT-2, GPT-Neo, GPT-J, Llama, Mistral,
Gemma, Phi, Qwen, OPT, BLOOM, and Falcon. Type any HuggingFace model
identifier in the sidebar. Larger models generally produce more noticeable
steering effects.

### Q: What happens with very high alpha values?

Alpha > 4-5 typically causes degenerate output: repetitive text, gibberish,
or mode collapse. The steering vector overwhelms the model's natural language
generation. The sweet spot is usually 1.0-3.0.

---

*This document was created for the SteeringLLM demo. For API reference and
integration guides, see the `docs/` directory.*
