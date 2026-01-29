"""
Base classes for agent framework integrations.

This module provides abstract base classes and configuration for all
agent framework integrations in SteeringLLM.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from steering_llm.core.steering_model import SteeringModel
from steering_llm.core.steering_vector import SteeringVector


@dataclass
class SteeringConfig:
    """
    Configuration for steering behavior in agents.
    
    Attributes:
        vectors: List of steering vectors to apply
        alpha: Global steering strength multiplier
        layer_alphas: Per-layer alpha overrides {layer: alpha}
        adaptive: Enable adaptive alpha based on context
        min_alpha: Minimum alpha for adaptive steering
        max_alpha: Maximum alpha for adaptive steering
        composition_method: How to combine multiple vectors ("sum", "weighted", "cascade")
        weights: Weights for weighted composition
        metadata: Additional configuration metadata
    """
    
    vectors: List[SteeringVector] = field(default_factory=list)
    alpha: float = 1.0
    layer_alphas: Dict[int, float] = field(default_factory=dict)
    adaptive: bool = False
    min_alpha: float = 0.1
    max_alpha: float = 3.0
    composition_method: str = "sum"
    weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        
        if self.min_alpha <= 0 or self.max_alpha <= 0:
            raise ValueError("min_alpha and max_alpha must be positive")
        
        if self.min_alpha > self.max_alpha:
            raise ValueError("min_alpha cannot exceed max_alpha")
        
        if self.composition_method not in {"sum", "weighted", "cascade"}:
            raise ValueError(
                f"composition_method must be one of (sum, weighted, cascade), "
                f"got {self.composition_method}"
            )
        
        if self.composition_method == "weighted":
            if self.weights is None:
                raise ValueError("weights required for weighted composition")
            if len(self.weights) != len(self.vectors):
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match "
                    f"vectors length ({len(self.vectors)})"
                )
    
    def get_alpha(self, layer: int) -> float:
        """
        Get effective alpha for a given layer.
        
        Args:
            layer: Layer index
        
        Returns:
            Effective alpha value (layer-specific or global)
        """
        return self.layer_alphas.get(layer, self.alpha)
    
    def add_vector(
        self,
        vector: SteeringVector,
        weight: Optional[float] = None,
    ) -> None:
        """
        Add a steering vector to the configuration.
        
        Args:
            vector: Steering vector to add
            weight: Weight for weighted composition (optional)
        """
        self.vectors.append(vector)
        if weight is not None and self.weights is not None:
            self.weights.append(weight)
    
    def clear_vectors(self) -> None:
        """Remove all steering vectors."""
        self.vectors.clear()
        if self.weights is not None:
            self.weights.clear()


class SteeringAgent(ABC):
    """
    Abstract base class for all agent framework integrations.
    
    This class defines the common interface that all framework-specific
    implementations must follow.
    
    Attributes:
        steering_model: The underlying SteeringModel
        config: Steering configuration
    """
    
    def __init__(
        self,
        steering_model: SteeringModel,
        config: Optional[SteeringConfig] = None,
    ) -> None:
        """
        Initialize the steering agent.
        
        Args:
            steering_model: SteeringModel instance
            config: Steering configuration (creates default if None)
        """
        self.steering_model = steering_model
        self.config = config or SteeringConfig()
        self._steering_active = False
    
    @property
    def model(self) -> Any:
        """Get the underlying model."""
        return self.steering_model.model
    
    @property
    def tokenizer(self) -> Any:
        """Get the model tokenizer."""
        return self.steering_model.tokenizer
    
    @property
    def is_steering_active(self) -> bool:
        """Check if steering is currently active."""
        return self._steering_active
    
    @abstractmethod
    def apply_steering(
        self,
        vectors: Optional[List[SteeringVector]] = None,
        config: Optional[SteeringConfig] = None,
    ) -> None:
        """
        Apply steering vectors to the model.
        
        Args:
            vectors: Steering vectors to apply (uses config.vectors if None)
            config: Override steering configuration (optional)
        
        Raises:
            RuntimeError: If steering is already active
        """
        pass
    
    @abstractmethod
    def remove_steering(self) -> None:
        """
        Remove all active steering.
        
        Raises:
            RuntimeError: If no steering is active
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Generate text with steering applied.
        
        Args:
            prompt: Input prompt(s)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text(s)
        """
        pass
    
    def update_config(
        self,
        alpha: Optional[float] = None,
        adaptive: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update steering configuration.
        
        Args:
            alpha: New global alpha value
            adaptive: Enable/disable adaptive steering
            **kwargs: Additional configuration parameters
        """
        if alpha is not None:
            self.config.alpha = alpha
        
        if adaptive is not None:
            self.config.adaptive = adaptive
        
        # Update other config attributes
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def add_vector(
        self,
        vector: SteeringVector,
        weight: Optional[float] = None,
        apply_immediately: bool = False,
    ) -> None:
        """
        Add a steering vector to the configuration.
        
        Args:
            vector: Steering vector to add
            weight: Weight for weighted composition
            apply_immediately: Whether to apply steering immediately
        """
        self.config.add_vector(vector, weight)
        
        if apply_immediately and self._steering_active:
            # Re-apply steering with updated vectors
            self.remove_steering()
            self.apply_steering()
    
    def __enter__(self) -> "SteeringAgent":
        """Context manager entry - apply steering."""
        self.apply_steering()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - remove steering."""
        if self._steering_active:
            self.remove_steering()
