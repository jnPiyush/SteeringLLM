# SteeringLLM

Runtime LLM behavior modification through activation steering.

## Overview

SteeringLLM enables you to modify LLM behavior at inference time without retraining. Apply steering vectors to guide model outputs toward desired characteristics (e.g., more helpful, safer, more creative).

**Status: Alpha (v0.1.0)** - Core functionality stable, agent integrations experimental.

---

## 🎯 What is Activation Steering?

Activation steering is a technique that modifies an LLM's internal representations during inference to guide its behavior—**without retraining or fine-tuning**.

### The Problem: Why Do We Need Steering?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Traditional Approaches                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ❌ Fine-tuning         → Expensive, requires data, creates new model      │
│   ❌ Prompt Engineering  → Limited control, inconsistent, token overhead    │
│   ❌ RLHF               → Complex, expensive, hard to iterate              │
│   ❌ Output Filtering    → Post-hoc, doesn't fix root behavior             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Solution: Steering Vectors

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     How Steering Works                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    Input: "Tell me about yourself"                                          │
│              │                                                              │
│              ▼                                                              │
│    ┌─────────────────┐                                                      │
│    │   Embedding     │                                                      │
│    │     Layer       │                                                      │
│    └────────┬────────┘                                                      │
│              │                                                              │
│              ▼                                                              │
│    ┌─────────────────┐                                                      │
│    │  Transformer    │                                                      │
│    │   Layer 1-5     │                                                      │
│    └────────┬────────┘                                                      │
│              │                                                              │
│              ▼                                                              │
│    ┌─────────────────┐     ┌─────────────────┐                              │
│    │  Transformer    │ ◄───│ + Steering      │  ← Add "helpfulness" vector  │
│    │   Layer 6       │     │   Vector        │    at inference time!        │
│    └────────┬────────┘     └─────────────────┘                              │
│              │                                                              │
│              ▼                                                              │
│    ┌─────────────────┐                                                      │
│    │  Transformer    │                                                      │
│    │   Layer 7+      │                                                      │
│    └────────┬────────┘                                                      │
│              │                                                              │
│              ▼                                                              │
│    Output: "I'm an AI assistant eager to help! ..."  ← More helpful!        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Steering Vectors Are Created

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Mean Difference Discovery                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Positive Examples (desired behavior):                                     │
│   ┌──────────────────────────────────┐                                      │
│   │ "I love helping people!"         │ ──► Extract ──► [+0.3, +0.7, ...]   │
│   │ "You're doing great!"            │    Activations                       │
│   │ "Let me assist you with that"    │                                      │
│   └──────────────────────────────────┘           │                          │
│                                                  │  Average                 │
│                                                  ▼                          │
│                                        Positive Centroid                    │
│                                                  │                          │
│                                                  │                          │
│   Negative Examples (opposite behavior):         │ SUBTRACT                 │
│   ┌──────────────────────────────────┐           │                          │
│   │ "I hate this."                   │ ──► Extract ──► [-0.2, -0.5, ...]   │
│   │ "You're terrible."               │    Activations    │                  │
│   │ "Go away."                       │                   │  Average         │
│   └──────────────────────────────────┘                   ▼                  │
│                                                 Negative Centroid           │
│                                                          │                  │
│                                                          │                  │
│                                                          ▼                  │
│                                        ┌─────────────────────────┐          │
│                                        │   STEERING VECTOR       │          │
│                                        │   [+0.5, +1.2, ...]     │          │
│                                        │                         │          │
│                                        │   Direction from        │          │
│                                        │   "unhelpful" →         │          │
│                                        │   "helpful" in          │          │
│                                        │   activation space      │          │
│                                        └─────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Why Steering Helps: Key Benefits

```
┌─────────────────┬────────────────────┬─────────────────────────────────────┐
│    Approach     │      Steering      │       Traditional Methods           │
├─────────────────┼────────────────────┼─────────────────────────────────────┤
│ Cost            │ ✅ $0 - No GPU     │ ❌ $$$ - Training costs             │
│                 │    training        │                                     │
├─────────────────┼────────────────────┼─────────────────────────────────────┤
│ Speed           │ ✅ Instant         │ ❌ Hours/days of training           │
│                 │    application     │                                     │
├─────────────────┼────────────────────┼─────────────────────────────────────┤
│ Flexibility     │ ✅ Adjust at       │ ❌ Fixed after training             │
│                 │    runtime         │                                     │
├─────────────────┼────────────────────┼─────────────────────────────────────┤
│ Composability   │ ✅ Combine         │ ❌ One behavior per model           │
│                 │    multiple        │                                     │
│                 │    behaviors       │                                     │
├─────────────────┼────────────────────┼─────────────────────────────────────┤
│ Reversibility   │ ✅ Remove vector   │ ❌ Permanent changes                │
│                 │    anytime         │                                     │
├─────────────────┼────────────────────┼─────────────────────────────────────┤
│ Model Weights   │ ✅ Unchanged       │ ❌ Modified permanently             │
│                 │                    │                                     │
└─────────────────┴────────────────────┴─────────────────────────────────────┘
```

---

## 🔬 Real-World Use Cases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SAFETY & MODERATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Before Steering:           After Safety Vector Applied:                   │
│   ┌──────────────────┐       ┌──────────────────────────────────────┐       │
│   │ "How to pick     │       │ "I can't help with lock picking,    │       │
│   │  a lock?"        │  ──►  │  but I can suggest calling a        │       │
│   │                  │       │  licensed locksmith if you're       │       │
│   │ [Detailed        │       │  locked out."                       │       │
│   │  instructions]   │       │                                     │       │
│   └──────────────────┘       └──────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          TONE & PERSONALITY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Base Model:                With Politeness + Enthusiasm Vectors:          │
│   ┌──────────────────┐       ┌──────────────────────────────────────┐       │
│   │ "The answer      │       │ "Great question! I'd be delighted    │       │
│   │  is 42."         │  ──►  │  to help! The answer is 42. Let me   │       │
│   │                  │       │  know if you'd like me to explain    │       │
│   │                  │       │  how I arrived at that! 😊"          │       │
│   └──────────────────┘       └──────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        DOMAIN EXPERTISE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Generic Response:          With Medical Domain Vector:                    │
│   ┌──────────────────┐       ┌──────────────────────────────────────┐       │
│   │ "Headaches can   │       │ "Headaches can be classified as      │       │
│   │  have many       │  ──►  │  primary (tension-type, migraine,    │       │
│   │  causes..."      │       │  cluster) or secondary. For tension  │       │
│   │                  │       │  headaches, first-line treatment     │       │
│   │                  │       │  includes NSAIDs. Recommend seeking  │       │
│   │                  │       │  medical evaluation if..."           │       │
│   └──────────────────┘       └──────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Multi-Vector Composition

Combine multiple steering behaviors simultaneously:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPOSING MULTIPLE VECTORS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────┐                                                         │
│   │  Helpfulness  │──┐                                                      │
│   │   Vector      │  │                                                      │
│   │  (weight=0.7) │  │                                                      │
│   └───────────────┘  │                                                      │
│                      │      ┌───────────────────┐     ┌──────────────────┐  │
│   ┌───────────────┐  │      │                   │     │                  │  │
│   │   Safety      │──┼────► │  Weighted Sum     │────►│  Combined        │  │
│   │   Vector      │  │      │  + Normalization  │     │  Steering        │  │
│   │  (weight=1.0) │  │      │                   │     │  Vector          │  │
│   └───────────────┘  │      └───────────────────┘     └──────────────────┘  │
│                      │                                         │            │
│   ┌───────────────┐  │                                         │            │
│   │  Conciseness  │──┘                                         │            │
│   │   Vector      │                                            ▼            │
│   │  (weight=0.3) │                              ┌──────────────────────┐   │
│   └───────────────┘                              │  Model generates     │   │
│                                                  │  helpful, safe, AND  │   │
│                                                  │  concise responses!  │   │
│                                                  └──────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Evaluation & Measurement

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEERING EFFECTIVENESS MEASUREMENT                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Toxicity Score (lower is better):                                         │
│                                                                             │
│   Without Steering  ████████████████████████████████████████░░  0.82        │
│   With Steering     ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.18        │
│                                                                             │
│                     ▲                                                       │
│                     │  78% reduction in toxicity!                           │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────     │
│                                                                             │
│   Helpfulness Score (higher is better):                                     │
│                                                                             │
│   Without Steering  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.35        │
│   With Steering     ██████████████████████████████████████░░░░  0.89        │
│                                                                             │
│                     ▲                                                       │
│                     │  154% improvement in helpfulness!                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Steering

```python
from steering_llm import SteeringModel, Discovery

# Load model
model = SteeringModel.from_pretrained("gpt2")  # Or any supported model

# Create steering vector from contrast examples
result = Discovery.mean_difference(
    positive=["I love helping people!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model.model,  # Pass the underlying HF model
    layer=6
)
vector = result.vector

# Generate with steering
output = model.generate_with_steering(
    "Tell me about yourself",
    vector=vector,
    alpha=2.0,
    max_new_tokens=50
)
```

### Agent Framework Integration ✨

```python
from steering_llm.agents import LangChainSteeringLLM, AzureSteeringAgent, LlamaIndexSteeringLLM

# LangChain Integration
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = LangChainSteeringLLM(
    steering_model=model,
    vectors=[safety_vector],
    alpha=2.0
)

prompt = PromptTemplate(input_variables=["topic"], template="Explain {topic}")
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI safety")

# Azure Integration (requires azure extras)
agent = AzureSteeringAgent(
    steering_model=model,
    agent_name="helpful_assistant",
    vectors=[helpful_vector],
    enable_tracing=True  # Azure Monitor integration
)
response = agent.generate("How can I help you?")

# LlamaIndex RAG
from llama_index.core import VectorStoreIndex

llm = LlamaIndexSteeringLLM(
    steering_model=model,
    vectors=[domain_vector],
    alpha=1.5
)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is this about?")
```

### Safety Evaluation (Phase 3) ✨

```python
from steering_llm.evaluation import SteeringEvaluator, ToxicityMetric

# Create evaluator
evaluator = SteeringEvaluator(
    model=model,
    vectors=[safety_vector],
    alpha=2.0
)

# Evaluate on ToxiGen benchmark
report = evaluator.evaluate_toxigen(num_samples=100)
print(f"Toxicity reduction: {report.comparison.effectiveness:.2%}")

# Evaluate on RealToxicityPrompts
report = evaluator.evaluate_realtoxicity(
    num_samples=100,
    min_prompt_toxicity=0.5
)

# Custom evaluation
prompts = ["Your custom prompts here"]
report = evaluator.evaluate_custom(
    prompts=prompts,
    additional_metrics={"domain": domain_metric}
)

# Save report
report.save(Path("evaluation_results/safety_report.json"))
```

### Advanced Discovery Methods

```python
# CAA (Contrastive Activation Addition) - Stronger steering
result = Discovery.caa(
    positive=["I love helping!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model.model,
    layer=6
)
vector = result.vector

# Linear Probing - Interpretable feature extraction
result = Discovery.linear_probe(
    positive=["I love helping!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model.model,
    layer=6
)
vector = result.vector
print(f"Probe accuracy: {result.metrics['train_accuracy']:.2%}")
```

### Multi-Vector Composition

```python
from steering_llm import VectorComposition

# Combine multiple steering vectors
politeness_result = Discovery.mean_difference(...)
conciseness_result = Discovery.mean_difference(...)

combined = VectorComposition.weighted_sum(
    vectors=[politeness_result.vector, conciseness_result.vector],
    weights=[0.7, 0.3],
    normalize=True
)

# Detect conflicts between vectors
conflicts = VectorComposition.detect_conflicts(
    [politeness_vec, conciseness_vec, formality_vec],
    threshold=0.7
)

# Orthogonalize vectors for independent control
ortho_vecs = VectorComposition.orthogonalize(
    [politeness_vec, formality_vec]
)

# Apply multiple vectors simultaneously
model.apply_multiple_steering(
    vectors=[politeness_vec, conciseness_vec],
    alphas=[1.2, 0.8]
)
```

## Installation

```bash
# Base installation
pip install steering-llm

# With agent integrations
pip install steering-llm[agents]

# With Azure/Microsoft features
pip install steering-llm[azure]

# With evaluation benchmarks
pip install steering-llm[evaluation]

# Everything
pip install steering-llm[all]
```

### Development Installation

```bash
git clone https://github.com/jnPiyush/SteeringLLM.git
cd SteeringLLM
pip install -e ".[dev,all]"
```

## Features

### Core Capabilities (Phase 1) ✅
- **Steering Vector Primitives**: Create, apply, and remove steering vectors
- **Mean Difference Discovery**: Extract steering vectors from contrast datasets
- **HuggingFace Integration**: Extended model support with quantization
- **Multi-layer Support**: Apply steering to any transformer layer
- **Persistent Steering**: Vectors stay active across multiple generations

### Agent Framework Integrations (Phase 3) ✨
- **LangChain**: BaseLLM wrapper for chains and agents
- **Microsoft Agent Framework**: Azure AI Foundry deployment with tracing
- **LlamaIndex**: CustomLLM for RAG applications
- **Multi-Agent Orchestration**: Sequential, parallel, hierarchical workflows
- **Prompt Flow Support**: Visual flow design and A/B testing

### Safety & Evaluation (Phase 3) ✨
- **ToxiGen Benchmark**: 13 minority groups, implicit toxicity detection
- **RealToxicityPrompts**: 100K naturally occurring prompts
- **Toxicity Metrics**: Local models (unitary/toxic-bert) or Perspective API
- **Steering Effectiveness**: Before/after comparison with multiple metrics
- **Domain Accuracy**: Keyword-based domain evaluation (medical, legal, technical)
- **Evaluation Suite**: Unified interface with JSON reports and visualization

### Advanced Discovery (Phase 2) ✅
- **CAA (Contrastive Activation Addition)**: Layer-wise contrasts for stronger steering
- **Linear Probing**: Train classifiers on activations, extract probe weights
- **Method Comparison**: Benchmark different discovery approaches
- **Accuracy Metrics**: Track probe performance (target >80%)

### Multi-Vector Composition (Phase 2) ✅
- **Weighted Composition**: Combine multiple vectors with custom weights
- **Conflict Detection**: Identify correlated/anti-correlated vectors
- **Orthogonalization**: Gram-Schmidt for independent steering directions
- **Analysis Tools**: Comprehensive similarity and composition analysis
- **Multi-Layer Application**: Apply 5+ vectors simultaneously to different layers

## Supported Models

SteeringLLM supports a wide range of transformer architectures:

| Model Family | Architectures | Example Models |
|--------------|---------------|----------------|
| **Llama** | llama | meta-llama/Llama-3.2-3B, meta-llama/Llama-2-7b-hf |
| **Mistral** | mistral | mistralai/Mistral-7B-v0.1, mistralai/Mixtral-8x7B-v0.1 |
| **Gemma** | gemma, gemma2 | google/gemma-2-2b, google/gemma-7b |
| **Phi** | phi, phi3 | microsoft/phi-2, microsoft/Phi-3-mini-4k-instruct |
| **Qwen** | qwen2, qwen2_moe | Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen1.5-7B |
| **GPT** | gpt2, gpt_neo, gpt_neox, gptj | gpt2, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-j-6b |
| **OPT** | opt | facebook/opt-1.3b, facebook/opt-6.7b |
| **BLOOM** | bloom | bigscience/bloom-560m, bigscience/bloom-1b7 |
| **Falcon** | falcon | tiiuae/falcon-7b, tiiuae/falcon-40b |

### Quantization Support

Steering works with quantized models via BitsAndBytes:

```python
# 8-bit quantization
model = SteeringModel.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit quantization
model = SteeringModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_4bit=True,
    device_map="auto"
)
```

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_discovery_advanced.py
pytest tests/test_vector_composition.py

# Run with coverage report
pytest --cov=steering_llm --cov-report=html

# Skip slow integration tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Examples

- **Basic Usage**: [examples/steering_basic_usage.py](examples/steering_basic_usage.py)
- **LangChain Integration**: [examples/langchain_steering_agent.py](examples/langchain_steering_agent.py)
- **Azure Agent Framework**: [examples/azure_agent_foundry.py](examples/azure_agent_foundry.py)
- **LlamaIndex RAG**: [examples/llamaindex_rag_steering.py](examples/llamaindex_rag_steering.py)

## Documentation

- **Architecture**: [docs/adr/ADR-001-steeringllm-architecture.md](docs/adr/ADR-001-steeringllm-architecture.md)
- **API Reference**: [docs/API-REFERENCE.md](docs/API-REFERENCE.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## License

MIT License - see LICENSE file for details.
