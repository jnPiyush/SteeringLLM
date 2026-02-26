# SteeringLLM Demo Fix Prompt

> **Purpose**: Comprehensive analysis and fix plan for the SteeringLLM interactive
> demo. This prompt guides an AI agent (or developer) through diagnosing every issue
> in the current demo and implementing the changes needed to successfully demonstrate
> three core LLM Steering use cases end-to-end.

---

## Target Use Cases

The demo MUST showcase these three steering scenarios with open models downloaded
from Hugging Face (GPT-2 family by default, running on CPU):

| # | Use Case | What the Audience Sees |
|---|----------|------------------------|
| 1 | **Personality / Tone** | Side-by-side baseline vs. steered output showing clear tone shift (e.g., formal, creative, positive). |
| 2 | **Role / Domain Expertise** | Model responds like a doctor, lawyer, engineer, etc. with domain-specific language, citations, and terminology. |
| 3 | **RAG (Retrieval-Augmented Generation)** | Model is grounded with data extracted from user-uploaded PDF files; steered output references the PDF content rather than hallucinating. |

---

## Current State Analysis

### What Works

- Core library (`src/steering_llm/`) -- `SteeringModel`, `SteeringVector`,
  `Discovery`, `VectorComposition` are fully implemented and tested.
- Model loading from Hugging Face via `SteeringModel.from_pretrained()`.
- Architecture registry covers GPT-2, Llama, Mistral, Phi, Qwen, etc.
- Forward-hook-based steering injection at configurable layer and alpha.
- Presets in `demo/presets.py` cover both tone presets (Positive/Helpful,
  Formal/Professional, Concise/Direct, Creative/Imaginative, Safety/Harmless)
  and role presets (Doctor, Lawyer, Software Engineer, Teacher, Business
  Consultant, Scientist).
- Streamlit UI with 5 tabs: Playground, Alpha Sweep, Composition, Inspector,
  Layer Explorer.
- Windows DLL-conflict launcher (`demo/launch.py`).

### Issues to Fix

#### Issue 1 -- No Dedicated "Role / Domain Expertise" Demo Tab

The role presets (Doctor, Lawyer, etc.) exist in `demo/presets.py` but are
buried in the same Playground dropdown alongside tone presets. There is no
dedicated UI flow that:

- Groups role presets separately from tone presets.
- Provides role-appropriate default prompts (e.g. "A patient asks: I've been
  having chest pains..." for the Doctor role).
- Shows a clear before/after demonstrating domain expertise shift.

**Fix required**: Add a new **"Role Expertise"** tab (or a clear UI section)
that filters to role presets only, supplies domain-appropriate prompts, and
outputs a compelling side-by-side comparison.

#### Issue 2 -- No RAG Tab or PDF Processing

The demo has **zero** RAG functionality:

- No PDF upload widget.
- No PDF text extraction (no `PyPDF2`, `pdfplumber`, or `pymupdf` dependency).
- No mechanism to chunk PDF text and build steering contrast pairs from
  document content.
- No tab showing "grounded" vs. "ungrounded" generation.

**Fix required**: Implement an end-to-end RAG-steering tab that:

1. Lets the user upload one or more PDF files via `st.file_uploader`.
2. Extracts text from the PDF (add `PyPDF2` or `pdfplumber` dependency).
3. Chunks the text into passages.
4. Builds contrast pairs: **positive** = passages from the PDF (grounding
   data), **negative** = generic/vague statements on the same topic.
5. Discovers a steering vector from those pairs.
6. Generates side-by-side: baseline (no grounding) vs. steered (grounded
   with PDF data).
7. Shows which PDF passages influenced the output (optional but impressive
   for the demo).

#### Issue 3 -- Missing `PyPDF2` / PDF Dependency in `pyproject.toml`

The `[project.optional-dependencies]` section has no PDF-processing library.

**Fix required**: Add a `rag` extra and include it in `demo` extras:
```toml
rag = [
    "PyPDF2>=3.0.0,<4.0.0",
]

demo = [
    "streamlit>=1.30.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    "PyPDF2>=3.0.0,<4.0.0",
]
```

#### Issue 4 -- Preset Organization: Tones vs. Roles Not Separated

`demo/presets.py` mixes personality/tone presets and role presets in one flat
dictionary (`PRESETS`). The UI has no way to distinguish them.

**Fix required**: Add a `category` field to each preset
(`"category": "tone"` or `"category": "role"`) and add helper functions:

```python
def get_tone_presets() -> List[str]:
    return [k for k, v in PRESETS.items() if v.get("category") == "tone"]

def get_role_presets() -> List[str]:
    return [k for k, v in PRESETS.items() if v.get("category") == "role"]
```

#### Issue 5 -- Default Prompts Are Generic

The Playground uses one default prompt for everything:
`"Tell me about yourself and how you see the world."` This works acceptably
for tone steering but is poor for role demos (a doctor answering "tell me
about yourself" is not compelling).

**Fix required**: Each preset should include a `default_prompt` field with a
domain-appropriate prompt. Examples:

| Preset | Default Prompt |
|--------|---------------|
| Positive / Helpful | "I'm feeling overwhelmed with my workload. What should I do?" |
| Formal / Professional | "Summarize the key findings from the quarterly report." |
| Doctor | "I've been experiencing persistent headaches and blurred vision for two weeks. What could this be?" |
| Lawyer | "My landlord is refusing to return my security deposit. What are my legal options?" |
| Software Engineer | "How should I design a microservices architecture for a high-traffic e-commerce platform?" |

#### Issue 6 -- No Sample PDF for RAG Demo

There is no sample PDF file included in the repo for quick RAG demonstration.

**Fix required**: Create `demo/sample_data/` directory with at least one small
sample PDF (or a text file that can be used as fallback). Alternatively,
provide a built-in text passage that simulates PDF content so the demo works
even without a PDF upload.

#### Issue 7 -- Model Download UX Could Be Improved

The demo defaults to `gpt2-large` (~800MB). First-time users may wait several
minutes with no progress indication beyond a spinner.

**Fix required**:
- Add a clear "first-time download" notice in the sidebar.
- Consider defaulting to `gpt2` (124M params, ~500MB) for faster first
  experience and allowing users to switch to `gpt2-medium` or `gpt2-large`
  for better quality.
- Add a tip about pre-downloading models.

#### Issue 8 -- `demo/app.py` Tab Order Does Not Match Demo Flow

The ideal demo flow is: Tone -> Role -> RAG (simple to complex). The current
tab order is Playground (generic), Alpha Sweep, Composition, Inspector, Layer
Explorer. None of these clearly map to the three use cases.

**Fix required**: Restructure tabs to:

1. **Tone / Personality** -- Dedicated tone steering demo.
2. **Role Expertise** -- Dedicated role steering demo.
3. **RAG / Document Grounding** -- PDF upload + grounded steering demo.
4. **Advanced** (collapsible/secondary) -- Alpha Sweep, Composition,
   Inspector, Layer Explorer for power users.

Or, at minimum, add the Role and RAG tabs alongside the existing ones.

---

## Implementation Plan

### Phase 1 -- Preset Organization & Role Tab

1. **Edit `demo/presets.py`**:
   - Add `"category": "tone"` to: Positive/Helpful, Formal/Professional,
     Concise/Direct, Creative/Imaginative, Safety/Harmless.
   - Add `"category": "role"` to: Doctor, Lawyer, Software Engineer,
     Teacher, Business Consultant, Scientist.
   - Add `"default_prompt"` to every preset with a domain-appropriate prompt.
   - Add `get_tone_presets()` and `get_role_presets()` helper functions.

2. **Edit `demo/app.py`**:
   - Add a new `_tab_role_expertise(model)` function.
   - This tab shows only role presets in a selectbox.
   - Uses the preset's `default_prompt` as the default prompt value.
   - Generates side-by-side: baseline vs. steered with role vector.
   - Displays the preset description prominently (e.g. "Responding as a
     Medical Professional").

3. **Update the Playground tab** to filter to tone presets when
   `source == "Preset"`.

### Phase 2 -- RAG / Document Grounding Tab

1. **Add PyPDF2 dependency** to `pyproject.toml` under `demo` extras.

2. **Install PyPDF2** in the virtual environment:
   ```bash
   pip install PyPDF2
   ```

3. **Create `demo/pdf_utils.py`** with:
   ```python
   def extract_text_from_pdf(uploaded_file) -> str:
       """Extract all text from a Streamlit UploadedFile (PDF)."""
       ...

   def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
       """Split text into overlapping chunks for contrast pairs."""
       ...

   def build_rag_contrast_pairs(
       chunks: List[str],
       topic: str,
       max_pairs: int = 8,
   ) -> Tuple[List[str], List[str]]:
       """Build positive (grounded) and negative (vague) contrast pairs."""
       ...
   ```

4. **Add `_tab_rag(model)` function to `demo/app.py`**:
   - PDF upload widget (`st.file_uploader`, accept `["pdf"]`).
   - Text extraction + chunking.
   - Topic input (what the user wants to ask about the PDF).
   - Contrast pair building (PDF chunks = positive, generic vague text = negative).
   - Steering vector discovery from those pairs.
   - Side-by-side: baseline (no grounding) vs. steered (grounded).
   - Display "Source passages" expander showing which PDF chunks were used.

5. **Create `demo/sample_data/`** with a small sample text file or PDF for
   quick out-of-box testing.

### Phase 3 -- Tab Restructure & UX Polish

1. **Update `main()` in `demo/app.py`** to add the new tabs:
   ```python
   tabs = st.tabs([
       "Tone / Personality",
       "Role Expertise",
       "RAG / Document Grounding",
       "Alpha Sweep",
       "Composition",
       "Inspector",
       "Layer Explorer",
   ])
   ```

2. **Update the sidebar** default model suggestion:
   - Default to `gpt2` for fastest first experience.
   - Add "Quality tip: use gpt2-large for better results" note.
   - Add first-time download notice.

3. **Update `demo/README.md`** with the new tab descriptions and RAG setup
   instructions.

### Phase 4 -- Validation & Testing

1. **Launch the demo**: `python demo/launch.py`
2. **Test Use Case 1 (Tone)**: Select "Formal / Professional" preset, generate,
   confirm clear tone shift in steered output.
3. **Test Use Case 2 (Role)**: Select "Doctor / Medical Professional", use the
   default medical prompt, confirm domain-specific language in steered output.
4. **Test Use Case 3 (RAG)**: Upload a PDF, enter a question, confirm steered
   output references PDF content while baseline does not.
5. **Test edge cases**: No PDF uploaded, empty prompts, alpha = 0, negative alpha.

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `demo/presets.py` | EDIT | Add `category`, `default_prompt` fields; add filter helpers |
| `demo/app.py` | EDIT | Add Role tab, RAG tab, restructure tab order, improve UX |
| `demo/pdf_utils.py` | CREATE | PDF extraction, chunking, contrast pair building |
| `demo/sample_data/` | CREATE | Sample PDF or text file for RAG demo |
| `demo/README.md` | EDIT | Document new tabs and RAG setup |
| `pyproject.toml` | EDIT | Add PyPDF2 dependency to demo extras |

---

## Acceptance Criteria

- [ ] **Tone demo**: Audience clearly sees personality/tone shift (e.g.
  informal -> formal) in side-by-side output.
- [ ] **Role demo**: Audience clearly sees domain expertise (e.g. medical
  terminology, legal citations) in steered output that is absent from
  baseline.
- [ ] **RAG demo**: Audience uploads a PDF, asks a question, and steered
  output references specific facts/data from the PDF while baseline
  produces generic/hallucinated text.
- [ ] All three demos work with `gpt2` (124M) on CPU for portability.
- [ ] `gpt2-large` produces noticeably better results when used.
- [ ] Demo launches cleanly on Windows via `python demo/launch.py`.
- [ ] No unhandled exceptions for common user actions (empty upload, no
  prompt, etc.).

---

## Key Technical Notes

### How Steering Works in This Codebase

1. **Discovery**: `Discovery.mean_difference()` takes positive and negative
   text lists, runs them through the model, extracts activations at a
   specified layer, and computes `mean(pos_activations) - mean(neg_activations)`
   to produce a `SteeringVector`.

2. **Application**: `SteeringModel.generate_with_steering()` registers a
   PyTorch forward hook on the target layer that adds
   `alpha * steering_vector` to the hidden states during inference.

3. **For RAG grounding**: The positive examples should be actual passages from
   the PDF. The negative examples should be vague, generic, or off-topic text.
   The resulting vector "steers" the model toward producing text that aligns
   with the factual content of the PDF passages.

### Supported Models (from `_MODEL_REGISTRY`)

GPT-2 family (`gpt2`), GPT-Neo, GPT-J, Llama, Mistral, Gemma, Phi, Qwen,
OPT, BLOOM, Falcon. The demo defaults to `gpt2-large` but should work with
any registered architecture.

### Important Constraints

- GPT-2 is a **base model** (no instruction tuning), so outputs are
  continuations, not chat responses. The demo prompts should be phrased as
  text-completion starters, not chat questions.
- Steering vectors work best at layers around 50-70% depth of the model.
- Alpha values between 1.0 and 3.0 typically produce visible but coherent
  changes. Higher alpha can cause degenerate output.
- RAG steering is an **approximation** -- it biases the model toward the
  document's language and facts, but it is not retrieval in the traditional
  RAG sense. Frame it as "document-grounded steering" in the demo UI.

---

## Prompt for AI Agent Execution

If you are an AI coding agent, execute the implementation plan above in order:

1. Read all files listed in the File Change Summary to understand current state.
2. Implement Phase 1 (preset organization + role tab) first.
3. Implement Phase 2 (RAG tab + PDF utils) second.
4. Implement Phase 3 (tab restructure + UX) third.
5. Run Phase 4 validation.
6. After each phase, verify there are no Python syntax errors or import
   failures by running:
   ```bash
   python -c "from demo.presets import PRESETS, get_tone_presets, get_role_presets; print('OK')"
   python -c "from demo.pdf_utils import extract_text_from_pdf, chunk_text; print('OK')"
   python -c "import demo.app; print('OK')"
   ```
7. Do NOT create summary/changelog markdown files unless asked.
8. Commit with: `feat: add role expertise and RAG demo tabs (#issue)`
