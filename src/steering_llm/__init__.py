"""
SteeringLLM - Runtime LLM behavior modification through activation steering.

This package provides tools for creating and applying steering vectors to modify
LLM behavior at inference time without retraining.
"""

__version__ = "0.1.0"

from steering_llm.core.steering_vector import SteeringVector
from steering_llm.core.discovery import Discovery, DiscoveryResult
from steering_llm.core.steering_model import (
    SteeringModel,
    ActivationHook,
    register_architecture,
    get_supported_architectures,
)
from steering_llm.core.vector_composition import VectorComposition

# Export exceptions for user error handling
from steering_llm.exceptions import (
    SteeringLLMError,
    ConfigurationError,
    ModelError,
    UnsupportedArchitectureError,
    ModelLoadError,
    LayerError,
    InvalidLayerError,
    LayerDetectionError,
    VectorError,
    IncompatibleVectorError,
    InvalidVectorError,
    SteeringError,
    SteeringActiveError,
    DiscoveryError,
    EmptyDatasetError,
    ActivationExtractionError,
    DependencyError,
)

__all__ = [
    # Core classes
    "SteeringVector",
    "Discovery",
    "DiscoveryResult",
    "SteeringModel",
    "ActivationHook",
    "VectorComposition",
    # Utility functions
    "register_architecture",
    "get_supported_architectures",
    # Exceptions
    "SteeringLLMError",
    "ConfigurationError",
    "ModelError",
    "UnsupportedArchitectureError",
    "ModelLoadError",
    "LayerError",
    "InvalidLayerError",
    "LayerDetectionError",
    "VectorError",
    "IncompatibleVectorError",
    "InvalidVectorError",
    "SteeringError",
    "SteeringActiveError",
    "DiscoveryError",
    "EmptyDatasetError",
    "ActivationExtractionError",
    "DependencyError",
]

# Optional: Import agent framework integrations if available
try:
    from steering_llm import agents
    __all__.append("agents")
except ImportError:
    pass

# Optional: Import evaluation framework if available
try:
    from steering_llm import evaluation
    __all__.append("evaluation")
except ImportError:
    pass
