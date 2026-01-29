# Changelog

All notable changes to SteeringLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Custom exception hierarchy for better error handling (`SteeringLLMError` base class)
- `register_architecture()` function to support custom model architectures
- `get_supported_architectures()` function to list supported models
- `DiscoveryResult` dataclass for consistent return types across discovery methods
- Named constants for magic numbers (`DEFAULT_BATCH_SIZE`, `DEFAULT_CONFLICT_THRESHOLD`, etc.)
- Input validation guards with maximum limits (`MAX_BATCH_SIZE`, `MAX_SAMPLES`)
- Explicit delegation methods on `SteeringModel` (`config`, `dtype`, `eval()`, `train()`, `to()`)

### Changed
- Discovery methods now return `DiscoveryResult` instead of bare `SteeringVector` or tuple
- Replaced generic `ValueError`/`RuntimeError` with specific exception types
- Updated dependency versions with upper bounds for stability
- Fixed phantom dependencies (`agent-framework` removed, LangChain packages corrected)
- Coverage now includes agents/evaluation modules (threshold adjusted to 70%)
- Removed unsafe `__getattr__` magic method delegation

### Fixed
- `MODEL_REGISTRY` renamed to `_MODEL_REGISTRY` (private) with public access functions
- Consistent return types across all Discovery methods
- Type hints improved throughout codebase

## [0.1.0] - 2026-01-15

### Added
- **Core Module**
  - `SteeringVector` - Immutable data class for steering vectors with save/load
  - `SteeringModel` - HuggingFace model wrapper with steering capabilities
  - `Discovery` - Methods for extracting steering vectors from contrast datasets
    - `mean_difference()` - Basic activation difference
    - `caa()` - Contrastive Activation Addition (Turner et al., 2023)
    - `linear_probe()` - Logistic regression-based extraction
  - `VectorComposition` - Multi-vector composition utilities
    - `weighted_sum()` - Combine vectors with weights
    - `detect_conflicts()` - Find conflicting vectors
    - `orthogonalize()` - Gram-Schmidt orthogonalization

- **Model Support**
  - Llama family (Llama 2, Llama 3, Code Llama)
  - Mistral family (Mistral, Mixtral)
  - Gemma family (Gemma 1, Gemma 2)
  - Phi family (Phi-2, Phi-3)
  - Qwen family (Qwen 1.5, Qwen 2, Qwen 2.5)
  - GPT family (GPT-2, GPT-Neo, GPT-NeoX, GPT-J)
  - OPT, BLOOM, Falcon

- **Agent Integrations** (Optional)
  - `LangChainSteeringLLM` - LangChain BaseLLM wrapper
  - `LlamaIndexSteeringLLM` - LlamaIndex CustomLLM wrapper
  - `AzureSteeringAgent` - Azure AI integration with tracing

- **Evaluation Framework** (Optional)
  - `SteeringEvaluator` - Unified evaluation interface
  - `ToxiGenBenchmark` - Toxicity evaluation
  - `RealToxicityPromptsBenchmark` - Real-world toxicity prompts
  - `ToxicityMetric`, `SteeringEffectivenessMetric`, `DomainAccuracyMetric`

### Dependencies
- torch>=2.0.0,<3.0.0
- transformers>=4.36.0,<5.0.0
- numpy>=1.24.0,<2.0.0
- scikit-learn>=1.3.0,<2.0.0

[Unreleased]: https://github.com/jnPiyush/SteeringLLM/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jnPiyush/SteeringLLM/releases/tag/v0.1.0
