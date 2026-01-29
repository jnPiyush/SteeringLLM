"""
Shared test fixtures and utilities.
"""

import pytest
import torch


@pytest.fixture
def sample_steering_vector():
    """Create a sample steering vector for testing."""
    from steering_llm.core.steering_vector import SteeringVector
    
    tensor = torch.randn(768)
    return SteeringVector(
        tensor=tensor,
        layer=10,
        layer_name="model.layers.10",
        model_name="test-model",
        method="mean_difference"
    )


@pytest.fixture
def sample_steering_vectors():
    """Create multiple sample steering vectors."""
    from steering_llm.core.steering_vector import SteeringVector
    
    return [
        SteeringVector(
            tensor=torch.randn(768),
            layer=10,
            layer_name="model.layers.10",
            model_name="test-model",
            method="mean_difference"
        ),
        SteeringVector(
            tensor=torch.randn(768),
            layer=15,
            layer_name="model.layers.15",
            model_name="test-model",
            method="caa"
        )
    ]
