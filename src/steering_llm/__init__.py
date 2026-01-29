"""
SteeringLLM - Runtime LLM behavior modification through activation steering.

This package provides tools for creating and applying steering vectors to modify
LLM behavior at inference time without retraining.
"""

__version__ = "0.1.0"

from steering_llm.core.steering_vector import SteeringVector
from steering_llm.core.discovery import Discovery
from steering_llm.core.steering_model import SteeringModel, ActivationHook

__all__ = ["SteeringVector", "Discovery", "SteeringModel", "ActivationHook"]
