"""
SteeringLLM - Runtime LLM behavior modification through activation steering.

This package provides tools for creating and applying steering vectors to modify
LLM behavior at inference time without retraining.
"""

__version__ = "0.1.0"

from steering_llm.core.steering_vector import SteeringVector
from steering_llm.core.discovery import Discovery
from steering_llm.core.steering_model import SteeringModel, ActivationHook
from steering_llm.core.vector_composition import VectorComposition

__all__ = [
    "SteeringVector",
    "Discovery",
    "SteeringModel",
    "ActivationHook",
    "VectorComposition",
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
