"""
Custom exceptions for SteeringLLM.

This module provides a clear exception hierarchy for better error handling
and debugging across the library.
"""


class SteeringLLMError(Exception):
    """Base exception for all SteeringLLM errors."""

    pass


class ConfigurationError(SteeringLLMError):
    """Raised when configuration is invalid."""

    pass


class ModelError(SteeringLLMError):
    """Base class for model-related errors."""

    pass


class UnsupportedArchitectureError(ModelError):
    """Raised when model architecture is not supported."""

    def __init__(self, model_type: str, supported: list[str]) -> None:
        self.model_type = model_type
        self.supported = supported
        super().__init__(
            f"Unsupported model architecture: '{model_type}'. "
            f"Supported: {', '.join(sorted(supported))}. "
            f"Use SteeringModel.register_architecture() to add support."
        )


class ModelLoadError(ModelError):
    """Raised when model fails to load."""

    pass


class LayerError(SteeringLLMError):
    """Base class for layer-related errors."""

    pass


class InvalidLayerError(LayerError):
    """Raised when layer index is invalid."""

    def __init__(self, layer: int, num_layers: int) -> None:
        self.layer = layer
        self.num_layers = num_layers
        super().__init__(
            f"Invalid layer index {layer}. Model has {num_layers} layers (0-{num_layers - 1})."
        )


class LayerDetectionError(LayerError):
    """Raised when layer cannot be detected from model architecture."""

    pass


class VectorError(SteeringLLMError):
    """Base class for steering vector errors."""

    pass


class IncompatibleVectorError(VectorError):
    """Raised when vector is incompatible with model."""

    def __init__(self, vector_dim: int, model_dim: int) -> None:
        self.vector_dim = vector_dim
        self.model_dim = model_dim
        super().__init__(
            f"Vector dimension mismatch: vector has {vector_dim}, "
            f"but model expects {model_dim}."
        )


class InvalidVectorError(VectorError):
    """Raised when vector data is invalid (NaN, Inf, wrong shape)."""

    pass


class SteeringError(SteeringLLMError):
    """Base class for steering operation errors."""

    pass


class SteeringActiveError(SteeringError):
    """Raised when steering operation conflicts with active steering."""

    def __init__(self, layer: int) -> None:
        self.layer = layer
        super().__init__(
            f"Steering already active on layer {layer}. "
            "Remove existing steering before applying new vector."
        )


class SteeringNotActiveError(SteeringError):
    """Raised when trying to remove steering that isn't active."""

    pass


class DiscoveryError(SteeringLLMError):
    """Base class for discovery/extraction errors."""

    pass


class EmptyDatasetError(DiscoveryError):
    """Raised when dataset is empty."""

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        super().__init__(f"{dataset_name} examples list cannot be empty.")


class ActivationExtractionError(DiscoveryError):
    """Raised when activation extraction fails."""

    pass


class DependencyError(SteeringLLMError):
    """Raised when an optional dependency is missing."""

    def __init__(self, package: str, extra: str) -> None:
        self.package = package
        self.extra = extra
        super().__init__(
            f"{package} is not installed. "
            f"Install with: pip install steering-llm[{extra}]"
        )
