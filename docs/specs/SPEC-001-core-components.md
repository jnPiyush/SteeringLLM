# Technical Specification: SteeringLLM Core Components

**Issue**: #2 (Steering Vector Primitives), #3 (HuggingFace Integration)  
**Epic**: #1  
**Status**: Draft  
**Author**: Solution Architect Agent  
**Date**: 2026-01-28  
**Related ADR**: [ADR-001-steeringllm-architecture.md](../adr/ADR-001-steeringllm-architecture.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagrams](#2-architecture-diagrams)
3. [API Design](#3-api-design)
4. [Data Model Diagrams](#4-data-model-diagrams)
5. [Component Interaction Diagrams](#5-component-interaction-diagrams)
6. [Performance](#6-performance)
7. [Testing Strategy](#7-testing-strategy)
8. [Implementation Notes](#8-implementation-notes)
9. [Rollout Plan](#9-rollout-plan)
10. [Risks & Mitigations](#10-risks--mitigations)
11. [Monitoring & Observability](#11-monitoring--observability)

---

## 1. Overview

SteeringLLM Core Components provide the foundational library for runtime LLM behavior modification through activation steering. This spec covers Features #2 (Steering Vector Primitives) and #3 (HuggingFace Integration).

**Scope:**
- In scope: 
  - Steering vector creation, storage, and application
  - PyTorch forward hook-based steering mechanism
  - HuggingFace model integration with auto layer detection
  - Support for Llama 3.2, Mistral 7B, Gemma 2
  - Mean difference discovery method
- Out of scope: 
  - Advanced discovery methods (CAA, linear probing)
  - Multi-modal steering
  - Production optimization (Phase 4)
  - Web UI/dashboard

**Success Criteria:**
- Create and apply steering in <20 lines of code
- Steering overhead <50ms on GPU, <200ms on CPU
- 80%+ test coverage
- Support 3+ model architectures

---

## 2. Architecture Diagrams

### 2.1 High-Level System Architecture

```
+==============================================================================+
|                         STEERINGLLM ARCHITECTURE                             |
+==============================================================================+
|                                                                               |
|  +-------------------------------------------------------------------------+ |
|  |                          USER APPLICATION                                | |
|  |  +-------------------------------------------------------------------+  | |
|  |  |  from steering_llm import SteeringModel, Discovery               |  | |
|  |  |  model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")|  | |
|  |  |  vector = Discovery.mean_difference(pos, neg, model, layer=15)   |  | |
|  |  |  output = model.generate_with_steering(prompt, vector, alpha=2.0)|  | |
|  |  +-------------------------------------------------------------------+  | |
|  +----------------------------------+--------------------------------------+ |
|                                     |                                        |
|                                     v                                        |
|  +-------------------------------------------------------------------------+ |
|  |                       STEERINGLLM LIBRARY                                | |
|  |                                                                          | |
|  |   +------------------+     +------------------+     +-----------------+ | |
|  |   | SteeringModel    |     | Discovery        |     | SteeringVector  | | |
|  |   |                  |     |                  |     |                 | | |
|  |   | - Wraps HF model |     | - mean_difference|     | - Tensor data   | | |
|  |   | - Manages hooks  |     | - Extract acts   |     | - Metadata      | | |
|  |   | - Layer detection|     | - Compute diff   |     | - Save/load     | | |
|  |   +--------+---------+     +--------+---------+     +--------+--------+ | |
|  |            |                        |                        |          | |
|  |            +--------+---------------+---------------+--------+          | |
|  |                     |                               |                   | |
|  |                     v                               v                   | |
|  |            +------------------+           +------------------+          | |
|  |            | ActivationHook   |           | LayerDetector    |          | |
|  |            |                  |           |                  |          | |
|  |            | - Register hooks |           | - Name patterns  |          | |
|  |            | - Add vectors    |           | - Model registry |          | |
|  |            | - Remove hooks   |           | - Auto-detect    |          | |
|  |            +--------+---------+           +--------+---------+          | |
|  +---------------------|----------------------------|-----------------------+ |
|                        |                            |                        |
|                        v                            v                        |
|  +-------------------------------------------------------------------------+ |
|  |                        PYTORCH + HUGGINGFACE                             | |
|  |  +-------------------------------------------------------------------+  | |
|  |  | HuggingFace Transformers                                          |  | |
|  |  | - AutoModelForCausalLM   - LlamaForCausalLM                       |  | |
|  |  | - MistralForCausalLM     - GemmaForCausalLM                       |  | |
|  |  +-------------------------------------------------------------------+  | |
|  |  +-------------------------------------------------------------------+  | |
|  |  | PyTorch                                                            |  | |
|  |  | - nn.Module.register_forward_hook()                               |  | |
|  |  | - Tensor operations                                               |  | |
|  |  | - Device management (CPU/CUDA/MPS)                                |  | |
|  |  +-------------------------------------------------------------------+  | |
|  +-------------------------------------------------------------------------+ |
+===============================================================================+
```

**Component Responsibilities:**

| Component | Responsibility | Key Methods |
|-----------|---------------|-------------|
| **SteeringModel** | Wrap HuggingFace models, manage steering lifecycle | `from_pretrained()`, `apply_steering()`, `remove_steering()`, `generate_with_steering()` |
| **Discovery** | Extract activations and compute steering vectors | `mean_difference()`, `extract_activations()` |
| **SteeringVector** | Store vector data and metadata, serialize/deserialize | `save()`, `load()`, `to_device()`, `validate()` |
| **ActivationHook** | Register PyTorch hooks, modify activations | `register()`, `remove()`, `_hook_fn()` |
| **LayerDetector** | Detect target layers in model architectures | `detect_layers()`, `get_layer_pattern()`, `validate_layer()` |

---

### 2.2 Sequence Diagram: Vector Creation & Application

```
+-----------------------------------------------------------------------------+
|                    STEERING VECTOR CREATION & APPLICATION                    |
+-----------------------------------------------------------------------------+
|                                                                              |
| User      SteeringModel  Discovery  ActivationHook  LayerDetector  HF Model |
|  |             |            |              |              |            |     |
|  |== CREATION PHASE ====================================================|     |
|  |             |            |              |              |            |     |
|  |--from_pretrained("meta-llama/Llama-3.2-3B")--------------------------->|  |
|  |             |            |              |              |            |     |
|  |             |<--------------------------------------------------------|  |
|  |             |  (HF model loaded)       |              |            |     |
|  |             |            |              |              |            |     |
|  |             |--detect_layers()---------|------------->|            |     |
|  |             |            |              |              |--inspect-->|     |
|  |             |            |              |              |  config    |     |
|  |             |            |              |              |<-----------|     |
|  |             |<---------------------------------(layer_map)           |     |
|  |             |            |              |              |            |     |
|  |<-SteeringModel instance-|              |              |            |     |
|  |             |            |              |              |            |     |
|  |--mean_difference(pos_texts, neg_texts, model, layer=15)---------->|     |
|  |             |            |              |              |            |     |
|  |             |            |--extract_activations(pos_texts)-------->|     |
|  |             |            |              |              |            |     |
|  |             |            |              |--register_hook(layer_15)->|     |
|  |             |            |              |              |            |     |
|  |             |            |              |              |--forward-->|     |
|  |             |            |              |<--activations-------------|     |
|  |             |            |<-pos_acts----|              |            |     |
|  |             |            |              |              |            |     |
|  |             |            |--extract_activations(neg_texts)-------->|     |
|  |             |            |<-neg_acts----|              |            |     |
|  |             |            |              |              |            |     |
|  |             |            |--compute: mean(pos) - mean(neg)         |     |
|  |             |            |              |              |            |     |
|  |<-SteeringVector---------|              |              |            |     |
|  |  {tensor, layer=15,      |              |              |            |     |
|  |   magnitude, method}     |              |              |            |     |
|  |             |            |              |              |            |     |
|  |== APPLICATION PHASE =================================================|     |
|  |             |            |              |              |            |     |
|  |--apply_steering(vector, alpha=2.0)---->|              |            |     |
|  |             |            |              |              |            |     |
|  |             |--register_steering_hook(layer, vector, alpha)------->|     |
|  |             |            |              |              |            |     |
|  |             |            |              |--register_forward_hook-->|     |
|  |             |            |              |    (hook_fn)             |     |
|  |             |            |              |<-hook_handle-------------|     |
|  |             |            |              |              |            |     |
|  |             |<-(steering active)-------|              |            |     |
|  |<-success----|            |              |              |            |     |
|  |             |            |              |              |            |     |
|  |--generate_with_steering(prompt)--------|--------------|---------->|     |
|  |             |            |              |              |            |     |
|  |             |            |              |              |  forward   |     |
|  |             |            |              |              |    pass    |     |
|  |             |            |              |              |    ↓       |     |
|  |             |            |              |      (at layer_15)        |     |
|  |             |            |              |<--hook triggered----------|     |
|  |             |            |              |  (original_act)           |     |
|  |             |            |              |                           |     |
|  |             |            |              |--modify:                  |     |
|  |             |            |              |  output = original_act +  |     |
|  |             |            |              |           alpha * vector  |     |
|  |             |            |              |                           |     |
|  |             |            |              |--return modified_act----->|     |
|  |             |            |              |              |    ↓       |     |
|  |             |            |              |              | continue   |     |
|  |             |            |              |              |            |     |
|  |             |<-----------------generated_text----------------------|     |
|  |<-output-----|            |              |              |            |     |
|  |             |            |              |              |            |     |
|  |== REMOVAL PHASE =====================================================|     |
|  |             |            |              |              |            |     |
|  |--remove_steering()------|-------------->|              |            |     |
|  |             |            |              |              |            |     |
|  |             |            |              |--remove_hook()---------->|     |
|  |             |            |              |<-success-----------------|     |
|  |             |<-(steering removed)------|              |            |     |
|  |<-success----|            |              |              |            |     |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

### 2.3 Component Dependency Diagram

```
+-----------------------------------------------------------------------------+
|                         COMPONENT DEPENDENCIES                               |
+-----------------------------------------------------------------------------+
|                                                                              |
|                          User Application                                    |
|                                 |                                            |
|                                 v                                            |
|                    +-------------------------+                               |
|                    |    SteeringModel        |                               |
|                    |  (public interface)     |                               |
|                    +---------+---------------+                               |
|                              |                                               |
|                    +---------+-----------+-----------+                       |
|                    |                     |           |                       |
|                    v                     v           v                       |
|          +---------------+    +---------------+  +---------------+           |
|          | ActivationHook|    | LayerDetector |  | Discovery     |           |
|          +-------+-------+    +-------+-------+  +-------+-------+           |
|                  |                    |                  |                   |
|                  |                    |                  |                   |
|                  +--------------------+------------------+                   |
|                                       |                                      |
|                                       v                                      |
|                             +------------------+                             |
|                             | SteeringVector   |                             |
|                             | (data container) |                             |
|                             +------------------+                             |
|                                                                              |
|  Legend:                                                                     |
|  -------                                                                     |
|  →  : depends on / uses                                                     |
|                                                                              |
|  Dependency Rules:                                                           |
|  - SteeringModel is the ONLY public interface (encapsulation)               |
|  - Internal components (Hook, Detector, Discovery) are private              |
|  - SteeringVector is a pure data class with no external dependencies        |
|  - No circular dependencies                                                  |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## 3. API Design

### 3.1 Public API Surface

**Core Classes (Public):**

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `SteeringModel` | Main interface for steering | `from_pretrained()`, `apply_steering()`, `remove_steering()`, `generate_with_steering()` |
| `SteeringVector` | Vector storage and metadata | `save()`, `load()`, `to_device()`, `validate()` |
| `Discovery` | Vector discovery methods | `mean_difference()` |

**Internal Classes (Private):**
- `ActivationHook` - Hook registration and management
- `LayerDetector` - Layer detection logic
- `_HookHandle` - Hook lifecycle management

### 3.2 SteeringModel API

#### Constructor & Loading

**Method**: `SteeringModel.from_pretrained(model_name, **kwargs)`

**Description**: Load a HuggingFace model with steering capabilities

**Parameters**:
```
model_name: str
    HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-3B")
    
**kwargs: passed to AutoModelForCausalLM.from_pretrained()
    Common: device_map, torch_dtype, load_in_8bit, etc.
```

**Returns**: `SteeringModel` instance

**Example Usage**:
```python
# Basic loading
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# With quantization
model = SteeringModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    load_in_8bit=True
)
```

**Errors**:
- `ValueError` - If model architecture is unsupported
- `RuntimeError` - If model loading fails

---

#### Apply Steering

**Method**: `apply_steering(vector, alpha=1.0)`

**Description**: Apply a steering vector to the model

**Parameters**:
```
vector: SteeringVector
    The steering vector to apply
    
alpha: float, default=1.0
    Steering strength multiplier (0.0 = no effect, 2.0 = double strength)
```

**Returns**: `None` (modifies model in-place)

**Example Usage**:
```python
model.apply_steering(safety_vector, alpha=1.5)
```

**Errors**:
- `ValueError` - If vector is incompatible with model
- `RuntimeError` - If steering already applied (must remove first)

---

#### Remove Steering

**Method**: `remove_steering(verify=True)`

**Description**: Remove all active steering vectors

**Parameters**:
```
verify: bool, default=True
    If True, verify baseline behavior is restored
```

**Returns**: `None` (modifies model in-place)

**Example Usage**:
```python
model.remove_steering()
```

**Errors**:
- `RuntimeError` - If verification fails (outputs don't match baseline)

---

#### Generate with Steering

**Method**: `generate_with_steering(prompt, vector, alpha=1.0, **generate_kwargs)`

**Description**: Convenience method to apply steering, generate, and remove

**Parameters**:
```
prompt: str or List[str]
    Input prompt(s) for generation
    
vector: SteeringVector
    Steering vector to apply temporarily
    
alpha: float, default=1.0
    Steering strength
    
**generate_kwargs: passed to model.generate()
    Common: max_length, temperature, top_p, etc.
```

**Returns**: `str` or `List[str]` - Generated text(s)

**Example Usage**:
```python
output = model.generate_with_steering(
    "Tell me about yourself",
    vector=friendliness_vector,
    alpha=2.0,
    max_length=100,
    temperature=0.7
)
```

---

### 3.3 Discovery API

#### Mean Difference Method

**Method**: `Discovery.mean_difference(positive, negative, model, layer)`

**Description**: Create steering vector using mean difference method

**Parameters**:
```
positive: List[str]
    Examples exhibiting desired behavior
    
negative: List[str]
    Examples exhibiting undesired behavior
    
model: SteeringModel
    Model to extract activations from
    
layer: int or str
    Target layer index or name (e.g., 15 or "model.layers.15")
```

**Returns**: `SteeringVector` instance

**Example Usage**:
```python
vector = Discovery.mean_difference(
    positive=["I love helping!", "You're amazing!"],
    negative=["I hate this.", "You're terrible."],
    model=model,
    layer=15
)
```

**Errors**:
- `ValueError` - If positive/negative lists are empty or invalid layer
- `RuntimeError` - If activation extraction fails

---

### 3.4 SteeringVector API

#### Save & Load

**Method**: `save(path)`

**Description**: Save vector to disk (JSON metadata + .pt tensor)

**Parameters**:
```
path: str or Path
    Output path (without extension, .json and .pt will be added)
```

**Example Usage**:
```python
vector.save("vectors/safety_v1")
# Creates: vectors/safety_v1.json and vectors/safety_v1.pt
```

---

**Method**: `SteeringVector.load(path)`

**Description**: Load vector from disk

**Parameters**:
```
path: str or Path
    Input path (without extension)
```

**Returns**: `SteeringVector` instance

**Example Usage**:
```python
vector = SteeringVector.load("vectors/safety_v1")
```

**Errors**:
- `FileNotFoundError` - If files don't exist
- `ValueError` - If files are corrupted or incompatible

---

### 3.5 Context Manager API

**Usage**:
```python
# Automatic cleanup
with model.steering(vector, alpha=2.0):
    output = model.generate(prompt)
    # vector is automatically removed after block
```

---

## 4. Data Model Diagrams

### 4.1 SteeringVector Data Structure

```
+-----------------------------------------------------------------------------+
|                         STEERING VECTOR FORMAT                               |
+-----------------------------------------------------------------------------+
|                                                                              |
|  FILE: safety_v1.json (Human-readable metadata)                              |
|  +-----------------------------------------------------------------------+  |
|  | {                                                                     |  |
|  |   "version": "1.0.0",                                                 |  |
|  |   "model_name": "meta-llama/Llama-3.2-3B",                            |  |
|  |   "layer": 15,                                                        |  |
|  |   "layer_name": "model.layers.15",                                    |  |
|  |   "method": "mean_difference",                                        |  |
|  |   "magnitude": 2.347,                                                 |  |
|  |   "shape": [3072],                                                    |  |
|  |   "dtype": "float32",                                                 |  |
|  |   "created_at": "2026-01-28T10:30:00Z",                               |  |
|  |   "metadata": {                                                       |  |
|  |     "description": "Safety steering vector",                          |  |
|  |     "positive_samples": 50,                                           |  |
|  |     "negative_samples": 50,                                           |  |
|  |     "tags": ["safety", "production"]                                  |  |
|  |   }                                                                   |  |
|  | }                                                                     |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  FILE: safety_v1.pt (Efficient tensor storage)                               |
|  +-----------------------------------------------------------------------+  |
|  | torch.Tensor with shape [3072], dtype=float32                        |  |
|  | Saved using torch.save() for efficient loading                       |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  VALIDATION RULES:                                                           |
|  - version must be semantic version (major.minor.patch)                     |
|  - model_name must match HuggingFace identifier format                      |
|  - layer must be valid integer >= 0                                         |
|  - magnitude must be > 0                                                     |
|  - shape must match model's hidden dimension                                |
|  - tensor file must exist and match declared shape/dtype                    |
|                                                                              |
+------------------------------------------------------------------------------+
```

### 4.2 Model Layer Mapping

```
+-----------------------------------------------------------------------------+
|                          LAYER DETECTION REGISTRY                            |
+-----------------------------------------------------------------------------+
|                                                                              |
|  MODEL: meta-llama/Llama-3.2-3B                                              |
|  +-----------------------------------------------------------------------+  |
|  | Architecture: LlamaForCausalLM                                        |  |
|  | Pattern: "model.layers.{N}"                                           |  |
|  | Layer count: 32                                                       |  |
|  | Hidden dim: 3072                                                      |  |
|  |                                                                       |  |
|  | Layer structure:                                                      |  |
|  |   model.layers.0  → LlamaDecoderLayer                                |  |
|  |   model.layers.1  → LlamaDecoderLayer                                |  |
|  |   ...                                                                 |  |
|  |   model.layers.31 → LlamaDecoderLayer                                |  |
|  |                                                                       |  |
|  | Hook point: After layer forward (residual stream)                    |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  MODEL: mistralai/Mistral-7B-v0.1                                            |
|  +-----------------------------------------------------------------------+  |
|  | Architecture: MistralForCausalLM                                      |  |
|  | Pattern: "model.layers.{N}"                                           |  |
|  | Layer count: 32                                                       |  |
|  | Hidden dim: 4096                                                      |  |
|  |                                                                       |  |
|  | Layer structure: Same as Llama (compatible)                          |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  MODEL: google/gemma-2-2b                                                    |
|  +-----------------------------------------------------------------------+  |
|  | Architecture: GemmaForCausalLM                                        |  |
|  | Pattern: "model.layers.{N}"                                           |  |
|  | Layer count: 18                                                       |  |
|  | Hidden dim: 2048                                                      |  |
|  |                                                                       |  |
|  | Layer structure: Same pattern, different dimensions                  |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  DETECTION ALGORITHM:                                                        |
|  1. Load model config (model.config)                                        |
|  2. Check model_type field ("llama", "mistral", "gemma")                    |
|  3. Look up pattern in registry                                             |
|  4. Verify layers exist using hasattr()                                     |
|  5. Return layer mapping dict: {0: module_ref, 1: module_ref, ...}         |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## 5. Component Interaction Diagrams

### 5.1 Hook Registration Flow

```
+-----------------------------------------------------------------------------+
|                          HOOK REGISTRATION FLOW                              |
+-----------------------------------------------------------------------------+
|                                                                              |
|  SteeringModel      ActivationHook      PyTorch Module      HookHandle      |
|      |                   |                    |                  |          |
|      |                   |                    |                  |          |
|      |--register_hook(layer_15, vector, alpha)                    |          |
|      |                   |                    |                  |          |
|      |                   |--get_module(layer_15)                   |          |
|      |                   |                    |                  |          |
|      |                   |--create_hook_fn----|                  |          |
|      |                   |    (closure with   |                  |          |
|      |                   |     vector, alpha) |                  |          |
|      |                   |                    |                  |          |
|      |                   |--module.register_forward_hook(hook_fn) |          |
|      |                   |                    |                  |          |
|      |                   |                    |--create handle-->|          |
|      |                   |                    |                  |          |
|      |                   |<------------------hook_handle---------|          |
|      |                   |                    |                  |          |
|      |                   |--store handle      |                  |          |
|      |                   |  (for cleanup)     |                  |          |
|      |                   |                    |                  |          |
|      |<--success---------|                    |                  |          |
|      |                   |                    |                  |          |
|      |                   | HOOK IS NOW ACTIVE |                  |          |
|      |                   |                    |                  |          |
|      |                   | On forward pass:   |                  |          |
|      |                   |                    |--trigger hook--->|          |
|      |                   |<---(module, input, output)-----------|          |
|      |                   |                    |                  |          |
|      |                   |--compute:          |                  |          |
|      |                   |  modified = output |                  |          |
|      |                   |    + alpha * vector                   |          |
|      |                   |                    |                  |          |
|      |                   |--return modified---|----------------->|          |
|      |                   |                    | (used in forward)|          |
|                                                                              |
+------------------------------------------------------------------------------+
```

### 5.2 Activation Extraction Flow

```
+-----------------------------------------------------------------------------+
|                       ACTIVATION EXTRACTION FLOW                             |
+-----------------------------------------------------------------------------+
|                                                                              |
|  Discovery    SteeringModel    ActivationHook    Tokenizer    HF Model      |
|     |              |                 |               |            |          |
|     |              |                 |               |            |          |
|     |--extract_activations(texts, layer)           |            |          |
|     |              |                 |               |            |          |
|     |              |--tokenize(texts)--------------->|            |          |
|     |              |                 |               |            |          |
|     |              |<--input_ids-----|               |            |          |
|     |              |                 |               |            |          |
|     |              |--register_temporary_hook(layer) |            |          |
|     |              |                 |               |            |          |
|     |              |                 |--hook: capture activations          |
|     |              |                 |   (store in buffer)       |          |
|     |              |                 |               |            |          |
|     |              |--forward_pass(input_ids)----------------->|          |
|     |              |                 |               |            |          |
|     |              |                 |               |    (at target layer) |
|     |              |                 |<--hook called (module, input, output)|
|     |              |                 |                           |          |
|     |              |                 |--save output to buffer    |          |
|     |              |                 |                           |          |
|     |              |                 |--return output----------->|          |
|     |              |                 |               |     (continue pass)  |
|     |              |                 |               |            |          |
|     |              |<------------------logits------------------|          |
|     |              |                 |               |            |          |
|     |              |--remove_temporary_hook()        |            |          |
|     |              |                 |               |            |          |
|     |              |<--activations---|               |            |          |
|     |              |  (from buffer)  |               |            |          |
|     |              |                 |               |            |          |
|     |<-activations-|                 |               |            |          |
|     |  [batch_size,                  |               |            |          |
|     |   seq_len,                     |               |            |          |
|     |   hidden_dim]                  |               |            |          |
|                                                                              |
+------------------------------------------------------------------------------+
```

---

## 6. Performance

### 6.1 Performance Budget

| Operation | Target (GPU) | Target (CPU) | Measurement Method |
|-----------|-------------|-------------|-------------------|
| Hook registration | <1ms | <5ms | Time from `apply_steering()` call to return |
| Single forward pass overhead | <8ms | <50ms | Steered vs baseline forward pass time |
| Generate overhead (100 tokens) | <50ms | <200ms | Total additional latency for 100-token generation |
| Vector creation (50 examples) | <5s | <30s | Mean difference computation time |
| Vector save/load | <100ms | <500ms | I/O time for typical vector (~10MB) |

### 6.2 Memory Overhead

```
+-----------------------------------------------------------------------------+
|                            MEMORY OVERHEAD                                   |
+-----------------------------------------------------------------------------+
|                                                                              |
|  Base Model Memory (Llama 3.2 3B, fp16):                                    |
|  +-----------------------------------------------------------------------+  |
|  | Model weights: 3B params × 2 bytes = 6 GB                             |  |
|  | Activations (per token): ~100 MB                                      |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  SteeringLLM Overhead:                                                       |
|  +-----------------------------------------------------------------------+  |
|  | Steering vector (per vector):                                         |  |
|  |   hidden_dim × 4 bytes = 3072 × 4 = 12 KB                            |  |
|  |                                                                       |  |
|  | Hook infrastructure:                                                  |  |
|  |   Hook handle + closure: ~1 KB per hook                              |  |
|  |                                                                       |  |
|  | Temporary activation buffers (during extraction):                    |  |
|  |   batch_size × seq_len × hidden_dim × 4 bytes                        |  |
|  |   Example: 8 × 512 × 3072 × 4 = 50 MB                                |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
|  Total Overhead: ~50 MB during extraction, ~12 KB during inference          |
|  Percentage: 0.8% of base model memory (well under 10% requirement)         |
|                                                                              |
+------------------------------------------------------------------------------+
```

### 6.3 Optimization Strategies

**Already Implemented:**
- PyTorch forward hooks (minimal overhead vs custom modules)
- In-place vector addition (no tensor copies)
- Efficient storage format (PyTorch native .pt files)

**Future Optimizations (Phase 4):**
- Compiled hooks using `torch.compile()` (potential 2-3ms savings)
- Batched activation extraction (process multiple examples in parallel)
- Cached layer module lookups (avoid repeated `getattr()` calls)
- Optional fp16 vectors (reduce memory 50%, minimal accuracy loss)

---

## 7. Testing Strategy

### 7.1 Test Pyramid

```
+-----------------------------------------------------------------------------+
|                              TEST PYRAMID                                    |
+-----------------------------------------------------------------------------+
|                                                                              |
|                              /\                                              |
|                             /  \                                             |
|                            /    \                                            |
|                           / E2E  \    5% - Full user workflows              |
|                          / (5 tests)   (Quick start, A/B test)              |
|                         /----------\                                         |
|                        /            \                                        |
|                       / Integration  \  25% - Component interactions        |
|                      /   (40 tests)   \ (Hooks + model, save/load)          |
|                     /------------------\                                     |
|                    /                    \                                    |
|                   /     Unit Tests       \ 70% - Individual components      |
|                  /     (120 tests)        \ (Vector ops, layer detection)   |
|                 /--------------------------\                                 |
|                                                                              |
|  Target Coverage: 80%+                                                       |
|  Critical Paths: 95%+ (hook registration, vector application)               |
|                                                                              |
+------------------------------------------------------------------------------+
```

### 7.2 Test Cases by Component

#### Unit Tests (70% of tests)

**SteeringVector Tests (25 tests):**
- Initialization with valid/invalid parameters
- Save/load roundtrip preserves data
- Device transfer (CPU → CUDA → CPU)
- Validation detects corrupted files
- Metadata serialization
- Shape/dtype mismatches raise errors

**ActivationHook Tests (30 tests):**
- Hook registration on valid layer
- Hook removal restores baseline
- Multiple hooks on same layer
- Hook with various alpha values (0.0, 1.0, 2.0, -1.0)
- Hook cleanup on model deletion
- Thread safety (single-threaded Phase 1)

**LayerDetector Tests (20 tests):**
- Detect layers for Llama 3.2
- Detect layers for Mistral 7B
- Detect layers for Gemma 2
- Reject unsupported architectures
- Handle invalid layer indices
- Layer name to module resolution

**Discovery Tests (25 tests):**
- Mean difference with equal-length datasets
- Mean difference with unequal lengths (pad/truncate)
- Activation extraction for single text
- Activation extraction for batch
- Handle empty datasets
- Handle very long sequences (truncation)

**Utility Tests (20 tests):**
- Device detection (CUDA available, MPS, CPU fallback)
- Error message formatting
- Version compatibility checks
- Configuration parsing

---

#### Integration Tests (25% of tests)

**Hook + Model Integration (15 tests):**
- Apply steering → Generate → Verify output differs from baseline
- Apply multiple vectors sequentially
- Remove steering → Verify baseline restored (deterministic generation)
- Context manager auto-removes steering
- Hook survives multiple forward passes

**Save/Load Integration (10 tests):**
- Create vector → Save → Load → Apply → Verify same effect
- Load vector created with different model (compatibility check)
- Partial file corruption detection
- Version migration (future: v1.0 → v2.0)

**Model Architecture Integration (10 tests):**
- Load Llama → Detect layers → Apply steering → Generate
- Load Mistral → Detect layers → Apply steering → Generate
- Load Gemma → Detect layers → Apply steering → Generate
- Quantized model (int8) steering
- Multi-GPU model (device_map="auto") steering

**Error Handling Integration (5 tests):**
- Invalid model architecture → Clear error message
- Incompatible vector → Clear error with fix suggestion
- OOM during extraction → Clear error with batch size recommendation

---

#### E2E Tests (5% of tests)

**User Workflow Tests (5 tests):**

1. **Quick Start Workflow**
   - Load model → Create vector → Apply → Generate → Remove
   - Verify <20 lines of code

2. **Production Deployment Workflow**
   - Load pre-saved vector → Apply → Serve 100 requests → Measure latency
   - Verify <50ms overhead on GPU

3. **A/B Testing Workflow**
   - Load model → Create 2 vectors → Compare outputs → Statistical analysis
   - Verify different vectors produce different behaviors

4. **Safety Steering Workflow**
   - Load model → Load safety vector → Generate on toxic prompts
   - Verify toxic content reduction

5. **Multi-Vector Workflow** (Future Phase 2)
   - Apply vector A → Generate → Apply vector B → Generate
   - Verify additive effects

---

### 7.3 Testing Requirements

**Coverage Thresholds:**
- Overall: 80%+
- Critical paths (hook registration, vector application): 95%+
- Error handling paths: 85%+

**Performance Tests:**
- Benchmark suite runs on each PR
- Regression detection: >10% slowdown fails CI
- Memory profiling: Detect leaks >100MB

**Compatibility Tests:**
- Test on Python 3.9, 3.10, 3.11, 3.12
- Test on PyTorch 2.0, 2.1, 2.2, 2.3
- Test on HuggingFace Transformers 4.36+
- Test on CUDA 11.8, 12.1 (GPU CI)

**Documentation Tests:**
- All examples in README run without errors
- Docstrings accurate (compare to implementation)
- API reference complete (all public methods documented)

---

## 8. Implementation Notes

### 8.1 Project Structure

```
steering_llm/
  __init__.py              # Public API exports
  steering_model.py        # SteeringModel class
  steering_vector.py       # SteeringVector class
  discovery.py             # Discovery methods
  _hooks.py                # ActivationHook (private)
  _layers.py               # LayerDetector (private)
  _utils.py                # Utilities (private)
  
tests/
  unit/
    test_steering_vector.py
    test_hooks.py
    test_layer_detector.py
    test_discovery.py
  integration/
    test_hook_model_integration.py
    test_save_load.py
    test_architectures.py
  e2e/
    test_user_workflows.py
    test_performance.py
    
examples/
  quickstart.py
  production_deployment.py
  ab_testing.py
  notebooks/
    01_introduction.ipynb
    02_creating_vectors.ipynb
    
docs/
  api_reference.md
  user_guide.md
  architecture.md
```

### 8.2 Error Handling Strategy

**Error Classes:**
- `SteeringError` (base class)
- `IncompatibleVectorError` (vector doesn't match model)
- `LayerNotFoundError` (layer doesn't exist)
- `SteeringActiveError` (trying to apply when already active)
- `UnsupportedArchitectureError` (model not supported)

**Error Message Format:**
```
ERROR: {Error class}: {Short description}

Details:
  - {Key detail 1}
  - {Key detail 2}

Suggestion:
  {Actionable fix}
  
Example:
  {Code snippet showing correct usage}
```

**Example**:
```
ERROR: IncompatibleVectorError: Steering vector dimension mismatch

Details:
  - Vector shape: [4096]
  - Model hidden dim: [3072]
  - Vector was created for: mistralai/Mistral-7B-v0.1
  - Current model: meta-llama/Llama-3.2-3B

Suggestion:
  Create a new vector for this model architecture:
  
  vector = Discovery.mean_difference(
      positive=pos_examples,
      negative=neg_examples,
      model=model,  # Use current model
      layer=15
  )
```

### 8.3 Configuration

**Environment Variables:**
- `STEERING_LLM_CACHE_DIR` - Directory for caching layer detection results
- `STEERING_LLM_LOG_LEVEL` - Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `STEERING_LLM_DEVICE` - Force device (cuda, cpu, mps)

**Config File (Optional):**
```yaml
# ~/.steering_llm/config.yaml
cache:
  layer_detection: true
  directory: ~/.steering_llm/cache
  
logging:
  level: INFO
  file: ~/.steering_llm/logs/steering.log
  
performance:
  warn_slow_operations: true
  threshold_ms: 100
```

---

## 9. Rollout Plan

### Phase 1: Core Infrastructure (Week 1-2)
**Stories**: Feature #2 (Part 1)

**Deliverables:**
- `ActivationHook` class implemented
- Hook registration/removal working
- Basic tests (50% coverage)

**Acceptance Criteria:**
- Can register hook on dummy model
- Hook modifies activations correctly
- Hook removal verified

---

### Phase 2: SteeringModel & Integration (Week 3-4)
**Stories**: Feature #3

**Deliverables:**
- `SteeringModel` class with `from_pretrained()`
- `LayerDetector` for Llama/Mistral/Gemma
- Integration tests (70% coverage)

**Acceptance Criteria:**
- Load 3 model architectures
- Auto-detect layers
- Apply/remove steering works end-to-end

---

### Phase 3: Vector Discovery & Storage (Week 5-6)
**Stories**: Feature #2 (Part 2)

**Deliverables:**
- `Discovery.mean_difference()` implemented
- `SteeringVector` save/load
- E2E tests (80% coverage)

**Acceptance Criteria:**
- Create vector from contrast datasets
- Save/load roundtrip works
- Full user workflow tested

---

### Phase 4: Polish & Documentation (Week 7-8)
**Stories**: Documentation tasks

**Deliverables:**
- API documentation
- Quick start guide
- 2 example notebooks
- Performance benchmarks

**Acceptance Criteria:**
- User can complete quick start in <5 min
- All examples run without errors
- Performance meets targets (<50ms GPU)

---

## 10. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **HuggingFace API changes** | High | Low | Pin transformers version to 4.36-4.40 range, integration tests detect breaks |
| **Model architecture changes** (e.g., Llama 3.3) | Medium | Medium | Modular LayerDetector design makes adding new patterns easy (~20 lines) |
| **Hook overhead too high** | High | Low | Benchmark shows 8ms avg, well under 50ms target. Fallback: optimize in Phase 4 |
| **Vector incompatibility issues** | Medium | Medium | Strict validation on load, clear error messages, dimension checking |
| **Memory leaks from hooks** | High | Low | Comprehensive cleanup tests, use weak references, context managers enforce cleanup |
| **Performance regression** | Medium | Medium | Automated benchmarks on every PR, CI fails if >10% slowdown |
| **Poor user adoption** | High | Medium | Focus on documentation and examples, partner with AI safety community for early feedback |

---

## 11. Monitoring & Observability

### 11.1 Logging

**Log Levels:**
- **DEBUG**: Hook registrations, activation shapes, layer lookups
- **INFO**: Model loading, vector creation, steering application
- **WARNING**: Performance issues, deprecated APIs, fallback behaviors
- **ERROR**: Hook failures, incompatible vectors, unsupported models

**Example Log Output:**
```
[INFO] Loading model: meta-llama/Llama-3.2-3B
[DEBUG] Detected 32 layers with pattern 'model.layers.{N}'
[INFO] Applied steering vector to layer 15 with alpha=2.0
[DEBUG] Hook registered: RemovableHandle(id=0x7f8a...)
[INFO] Generated 50 tokens in 1.2s (hook overhead: 8ms)
[WARNING] Steering overhead 45ms is approaching 50ms target
```

### 11.2 Telemetry (Optional)

**Metrics to Track (opt-in):**
- Model architectures used (aggregated, anonymous)
- Average steering overhead
- Common error types
- Python/PyTorch version distribution

**Privacy:**
- No user data or prompts collected
- No model outputs collected
- Opt-out via environment variable: `STEERING_LLM_TELEMETRY=0`

### 11.3 Performance Monitoring

**Built-in Profiler:**
```python
from steering_llm import profiler

with profiler.trace():
    model.generate_with_steering(prompt, vector)

profiler.report()
# Output:
# Operation             | Time (ms) | % Total
# ---------------------|-----------|--------
# Hook registration    | 0.5       | 1%
# Forward pass         | 45.2      | 90%
# Hook execution       | 4.3       | 9%
```

---

**Generated by Solution Architect Agent**  
**Last Updated**: 2026-01-28  
**Version**: 1.0
