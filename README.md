# SteeringLLM

Runtime LLM behavior modification through activation steering.

## Overview

SteeringLLM enables you to modify LLM behavior at inference time without retraining. Apply steering vectors to guide model outputs toward desired characteristics (e.g., more helpful, safer, more creative).

## Quick Start

### Basic Steering

```python
from steering_llm import SteeringModel, Discovery

# Load model
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# Create steering vector from contrast examples
vector = Discovery.mean_difference(
    positive=["I love helping people!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model,
    layer=15
)

# Generate with steering
output = model.generate_with_steering(
    "Tell me about yourself",
    vector=vector,
    alpha=2.0,
    max_length=100
)
```

### Advanced Discovery Methods (Phase 2)

```python
# CAA (Contrastive Activation Addition) - Stronger steering
vector = Discovery.caa(
    positive=["I love helping!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model,
    layer=15
)

# Linear Probing - Interpretable feature extraction
vector, metrics = Discovery.linear_probe(
    positive=["I love helping!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model,
    layer=15
)
print(f"Probe accuracy: {metrics['train_accuracy']:.2%}")
```

### Multi-Vector Composition (Phase 2)

```python
from steering_llm import VectorComposition

# Combine multiple steering vectors
politeness_vec = Discovery.mean_difference(...)
conciseness_vec = Discovery.mean_difference(...)

combined = VectorComposition.weighted_sum(
    vectors=[politeness_vec, conciseness_vec],
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
pip install steering-llm
```

### Development Installation

```bash
git clone https://github.com/jnPiyush/SteeringLLM.git
cd SteeringLLM
pip install -e ".[dev]"
```

## Features

### Core Capabilities (Phase 1) ✅
- **Steering Vector Primitives**: Create, apply, and remove steering vectors
- **Mean Difference Discovery**: Extract steering vectors from contrast datasets
- **HuggingFace Integration**: Extended model support with quantization
- **Multi-layer Support**: Apply steering to any transformer layer
- **Persistent Steering**: Vectors stay active across multiple generations

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

### Device Support

- ✅ **CPU**: Full support
- ✅ **CUDA**: Multi-GPU with `device_map="auto"`
- ✅ **MPS**: Apple Silicon support

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.36+
- scikit-learn 1.3+ (for linear probing)

## Testing

The project maintains 95%+ test coverage with 132 comprehensive tests:

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_discovery_advanced.py
pytest tests/test_vector_composition.py

# Run with coverage report
pytest --cov=steering_llm --cov-report=html
```

## Architecture

See [docs/adr/ADR-001-steeringllm-architecture.md](docs/adr/ADR-001-steeringllm-architecture.md) for architectural decisions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see LICENSE file for details.
