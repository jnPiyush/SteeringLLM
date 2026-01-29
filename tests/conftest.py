"""Pytest configuration and fixtures."""

import pytest
import torch


# Common hidden dimensions for different model sizes
HIDDEN_DIM_GPT2 = 768        # GPT-2 small
HIDDEN_DIM_LLAMA_3B = 3072   # Llama 3.2 3B
HIDDEN_DIM_LLAMA_7B = 4096   # Llama 2 7B


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for testing (GPT-2 dimensions)."""
    return torch.randn(HIDDEN_DIM_GPT2)


@pytest.fixture
def sample_tensor_small():
    """Provide a smaller tensor for quick unit tests."""
    return torch.randn(128)


@pytest.fixture
def sample_steering_vector():
    """Provide a sample SteeringVector for testing (GPT-2 dimensions)."""
    from steering_llm.core.steering_vector import SteeringVector
    
    return SteeringVector(
        tensor=torch.randn(HIDDEN_DIM_GPT2),
        layer=6,
        layer_name="transformer.h.6",
        model_name="gpt2",
        metadata={
            "description": "Test vector",
            "positive_samples": 50,
            "negative_samples": 50,
        },
    )


@pytest.fixture
def sample_steering_vector_llama():
    """Provide a sample SteeringVector with Llama dimensions."""
    from steering_llm.core.steering_vector import SteeringVector
    
    return SteeringVector(
        tensor=torch.randn(HIDDEN_DIM_LLAMA_3B),
        layer=15,
        layer_name="model.layers.15",
        model_name="meta-llama/Llama-3.2-3B",
        metadata={
            "description": "Test vector for Llama",
            "positive_samples": 50,
            "negative_samples": 50,
        },
    )


@pytest.fixture
def temp_vector_path(tmp_path):
    """Provide a temporary path for saving vectors."""
    return tmp_path / "test_vector"


# Configure pytest to show full diffs
def pytest_configure(config):
    """Configure pytest options."""
    config.option.verbose = 2
