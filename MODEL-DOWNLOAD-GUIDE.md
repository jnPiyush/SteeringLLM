# SteeringLLM Model Download Guide

## Do I Need Credentials?

**Short Answer**: It depends on which model you want to use.

### ✅ No Credentials Needed (Public Models)
These models download automatically without any authentication:

- **GPT-2** (default): `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **GPT-Neo**: `EleutherAI/gpt-neo-125m`, `EleutherAI/gpt-neo-2.7B`
- **GPT-J**: `EleutherAI/gpt-j-6b`
- **OPT**: `facebook/opt-125m`, `facebook/opt-1.3b`, `facebook/opt-6.7b`
- **BLOOM**: `bigscience/bloom-560m`, `bigscience/bloom-1b7`
- **Falcon**: `tiiuae/falcon-7b`
- **Phi**: `microsoft/phi-2`

### 🔒 Credentials Required (Gated Models)
These models require HuggingFace authentication:

- **Llama 2/3**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-3.2-3B`
- **Mistral**: Some versions like `mistralai/Mistral-7B-v0.1`
- **Gemma**: `google/gemma-2-2b`, `google/gemma-7b`

---

## 🔑 How to Set Up Credentials (For Gated Models)

### Step 1: Create HuggingFace Account
1. Go to [https://huggingface.co/join](https://huggingface.co/join)
2. Create a free account

### Step 2: Request Access to Gated Models
1. Visit the model page (e.g., [https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf))
2. Click "**Request Access**" button
3. Fill out the form (usually instant approval for research)
4. Wait for approval email (can take minutes to hours)

### Step 3: Generate Access Token
1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "**New token**"
3. Name: `SteeringLLM` (or any name)
4. Type: Select "**Read**" (models only need read access)
5. Click "**Generate a token**"
6. **Copy the token** (starts with `hf_...`)

### Step 4: Authenticate on Your Machine

**Option A: Using HuggingFace CLI (Recommended)**
```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login with your token
huggingface-cli login
# Paste your token when prompted
```

**Option B: Environment Variable**
```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_YourTokenHere"

# Windows CMD
set HF_TOKEN=hf_YourTokenHere

# Linux/Mac
export HF_TOKEN=hf_YourTokenHere
```

**Option C: In Python Code**
```python
from steering_llm import SteeringModel

# Pass token directly to model loading
model = SteeringModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token="hf_YourTokenHere"
)
```

---

## 📥 Where Models Are Downloaded

### Default Cache Location
HuggingFace stores downloaded models in:

**Windows:**
```
C:\Users\{YourUsername}\.cache\huggingface\hub\
```

**Linux/Mac:**
```
~/.cache/huggingface/hub/
```

### Typical Model Sizes
- **GPT-2** (small): ~500 MB
- **GPT-2** (medium): ~1.5 GB
- **GPT-2** (large): ~3 GB
- **Llama-2-7B**: ~13 GB
- **Mistral-7B**: ~14 GB

### Custom Cache Location
```python
import os
os.environ['HF_HOME'] = 'D:/models/huggingface'  # Custom location

# Or set persistently in Windows
setx HF_HOME "D:\models\huggingface"
```

---

## 🚀 Quick Start Examples

### Example 1: No Auth Needed (GPT-2)
```python
from steering_llm import SteeringModel, Discovery

# Works immediately, no credentials
model = SteeringModel.from_pretrained("gpt2")

# Create steering vector
result = Discovery.mean_difference(
    positive=["I love helping people!"],
    negative=["I hate this."],
    model=model.model,
    layer=6
)

# Generate with steering
output = model.generate_with_steering(
    "Tell me about yourself",
    vector=result.vector,
    alpha=2.0,
    max_new_tokens=50
)
print(output)
```

### Example 2: Auth Required (Llama-2)
```python
from steering_llm import SteeringModel

# First time: authenticate
from huggingface_hub import login
login(token="hf_YourTokenHere")

# Then load gated model
model = SteeringModel.from_pretrained("meta-llama/Llama-2-7b-hf")
# Rest is the same as GPT-2...
```

### Example 3: Running the Demo
```bash
# Demo uses GPT-2 by default (no auth needed)
python demo/launch.py

# In the UI, you can switch to other models
# If you select a gated model, make sure you've authenticated first
```

### Example 4: Professional Role Behaviors
The demo includes professional role presets that demonstrate how steering can modify a model to adopt specific personas:

**Available Role Presets:**
- 🩺 **Doctor / Medical Professional** - Clinical knowledge, empathy, evidence-based approach
- ⚖️ **Lawyer / Legal Expert** - Precise legal language, citation of law, careful analysis
- 💻 **Software Engineer** - Technical precision, best practices, systematic thinking
- 📚 **Teacher / Educator** - Patient explanations, encouragement, clear instruction
- 💼 **Business Consultant** - Strategic thinking, ROI focus, actionable recommendations
- 🔬 **Scientist / Researcher** - Rigorous methodology, empirical evidence, intellectual humility

**Try it:**
```python
from steering_llm import SteeringModel, Discovery

model = SteeringModel.from_pretrained("gpt2-large")

# Create a "Doctor" steering vector
doctor_result = Discovery.mean_difference(
    positive=[
        "Based on your symptoms, I recommend seeing your doctor.",
        "The clinical evidence suggests this could be a viral infection.",
        "I understand this diagnosis may be concerning."
    ],
    negative=[
        "Just google your symptoms and self-diagnose.",
        "That's probably nothing serious, don't worry about it.",
        "Skip the doctor, just buy some pills."
    ],
    model=model.model,
    layer=6
)

# Ask the same question with and without steering
prompt = "I have a headache and fever. What should I do?"

baseline = model.generate(prompt, max_new_tokens=100)
print("Baseline:", baseline)

steered = model.generate_with_steering(
    prompt, 
    vector=doctor_result.vector,
    alpha=2.5,
    max_new_tokens=100
)
print("Doctor mode:", steered)
```

The steered version will respond with more clinical language, mention seeing a healthcare provider, and consider evidence-based approaches!


---

## 🔧 Troubleshooting

### Error: `OSError: Cannot access gated repo`
**Solution**: You need to:
1. Request access to the model on HuggingFace
2. Wait for approval
3. Authenticate with your token

### Error: `401 Unauthorized`
**Solution**: Your token is invalid or expired
1. Generate a new token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Re-authenticate with `huggingface-cli login`

### Error: Out of disk space
**Solution**: Models are large (7B models = ~14GB)
1. Check available space: `df -h` (Linux/Mac) or `dir` (Windows)
2. Clear old models: Delete files in `~/.cache/huggingface/hub/`
3. Use a custom cache location on a larger drive

### Error: `torch c10.dll failed to load` (Windows)
**Solution**: This is the pyarrow/torch DLL conflict
```bash
# Use the launcher script (not direct streamlit)
python demo/launch.py  # Correct

# Don't use:
streamlit run demo/app.py  # May fail on Windows
```

---

## 📊 Model Recommendations by Use Case

| Use Case | Recommended Model | Auth? | Size |
|----------|------------------|-------|------|
| **Quick Testing** | `gpt2` | ❌ No | 500 MB |
| **Better Quality** | `gpt2-large` | ❌ No | 3 GB |
| **Production (Open)** | `EleutherAI/gpt-j-6b` | ❌ No | 12 GB |
| **Production (Gated)** | `meta-llama/Llama-2-7b-hf` | ✅ Yes | 13 GB |
| **Low Memory** | `gpt2` or `microsoft/phi-2` | ❌ No | 500 MB - 2.7 GB |
| **Best Performance** | `meta-llama/Llama-3.2-3B` | ✅ Yes | 6 GB |

---

## 🎯 Step-by-Step First Run

```bash
# 1. Install SteeringLLM
cd C:\Engagements\Learnings\SteeringLLM
pip install -e ".[demo]"

# 2. (Optional) Authenticate if using gated models
huggingface-cli login
# Paste your token: hf_...

# 3. Launch demo (uses GPT-2, no auth needed)
python demo/launch.py

# 4. Open browser to http://localhost:8501

# 5. Try steering!
#    - Select preset examples (including professional roles!)
#    - Adjust steering strength
#    - See baseline vs steered outputs
```

---

## 📚 Additional Resources

- **HuggingFace Token Docs**: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)
- **Model Hub**: [https://huggingface.co/models](https://huggingface.co/models)
- **SteeringLLM README**: [README.md](README.md)
- **Supported Models List**: See README.md "#Supported Models" section

---

**Last Updated**: February 25, 2026  
**For Issues**: Create an issue at [https://github.com/jnPiyush/SteeringLLM/issues](https://github.com/jnPiyush/SteeringLLM/issues)
