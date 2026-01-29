"""
Tests for agent base classes and configuration.
"""

import pytest
import torch

from steering_llm.agents.base import SteeringAgent, SteeringConfig
from steering_llm.core.steering_model import SteeringModel
from steering_llm.core.steering_vector import SteeringVector


@pytest.fixture
def sample_vector():
    """Create a sample steering vector."""
    tensor = torch.randn(768)
    return SteeringVector(
        tensor=tensor,
        layer=10,
        layer_name="model.layers.10",
        model_name="test-model",
        method="mean_difference"
    )


@pytest.fixture
def sample_vectors():
    """Create multiple sample steering vectors."""
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


class TestSteeringConfig:
    """Tests for SteeringConfig."""
    
    def test_init_default(self):
        """Test default initialization."""
        config = SteeringConfig()
        assert config.vectors == []
        assert config.alpha == 1.0
        assert config.layer_alphas == {}
        assert config.adaptive is False
        assert config.composition_method == "sum"
    
    def test_init_with_vectors(self, sample_vectors):
        """Test initialization with vectors."""
        config = SteeringConfig(
            vectors=sample_vectors,
            alpha=2.0
        )
        assert len(config.vectors) == 2
        assert config.alpha == 2.0
    
    def test_init_with_weighted_composition(self, sample_vectors):
        """Test weighted composition configuration."""
        config = SteeringConfig(
            vectors=sample_vectors,
            composition_method="weighted",
            weights=[0.7, 0.3]
        )
        assert config.composition_method == "weighted"
        assert config.weights == [0.7, 0.3]
    
    def test_init_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            SteeringConfig(alpha=0)
        
        with pytest.raises(ValueError, match="alpha must be positive"):
            SteeringConfig(alpha=-1.0)
    
    def test_init_invalid_composition_method(self):
        """Test that invalid composition method raises error."""
        with pytest.raises(ValueError, match="composition_method must be"):
            SteeringConfig(composition_method="invalid")
    
    def test_init_weighted_without_weights(self, sample_vectors):
        """Test weighted composition without weights raises error."""
        with pytest.raises(ValueError, match="weights required"):
            SteeringConfig(
                vectors=sample_vectors,
                composition_method="weighted"
            )
    
    def test_init_weighted_mismatched_lengths(self, sample_vectors):
        """Test weighted composition with mismatched lengths."""
        with pytest.raises(ValueError, match="weights length.*must match"):
            SteeringConfig(
                vectors=sample_vectors,
                composition_method="weighted",
                weights=[0.5]  # Only 1 weight for 2 vectors
            )
    
    def test_get_alpha_default(self):
        """Test getting default alpha."""
        config = SteeringConfig(alpha=2.0)
        assert config.get_alpha(10) == 2.0
        assert config.get_alpha(15) == 2.0
    
    def test_get_alpha_layer_specific(self):
        """Test getting layer-specific alpha."""
        config = SteeringConfig(
            alpha=1.0,
            layer_alphas={10: 2.0, 15: 3.0}
        )
        assert config.get_alpha(10) == 2.0
        assert config.get_alpha(15) == 3.0
        assert config.get_alpha(20) == 1.0  # Falls back to default
    
    def test_add_vector(self, sample_vector):
        """Test adding a vector."""
        config = SteeringConfig()
        assert len(config.vectors) == 0
        
        config.add_vector(sample_vector)
        assert len(config.vectors) == 1
        assert config.vectors[0] == sample_vector
    
    def test_add_vector_with_weight(self, sample_vector):
        """Test adding a vector with weight."""
        config = SteeringConfig(
            composition_method="weighted",
            weights=[]
        )
        config.add_vector(sample_vector, weight=0.5)
        
        assert len(config.vectors) == 1
        assert len(config.weights) == 1
        assert config.weights[0] == 0.5
    
    def test_clear_vectors(self, sample_vectors):
        """Test clearing vectors."""
        config = SteeringConfig(vectors=sample_vectors)
        assert len(config.vectors) == 2
        
        config.clear_vectors()
        assert len(config.vectors) == 0


class MockSteeringAgent(SteeringAgent):
    """Mock implementation for testing."""
    
    def __init__(self, steering_model, config=None):
        super().__init__(steering_model, config)
        self._apply_called = False
        self._remove_called = False
        self._generate_called = False
    
    def apply_steering(self, vectors=None, config=None):
        self._apply_called = True
        self._steering_active = True
    
    def remove_steering(self):
        self._remove_called = True
        self._steering_active = False
    
    def generate(self, prompt, **kwargs):
        self._generate_called = True
        return "generated text"


class TestSteeringAgent:
    """Tests for SteeringAgent base class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock steering model."""
        # Use a simple mock instead of real model for speed
        class MockModel:
            class Config:
                name_or_path = "test-model"
            config = Config()
        
        class MockSteeringModel:
            def __init__(self):
                self.model = MockModel()
                self.tokenizer = None
        
        return MockSteeringModel()
    
    def test_init_default(self, mock_model):
        """Test default initialization."""
        agent = MockSteeringAgent(mock_model)
        assert agent.steering_model == mock_model
        assert isinstance(agent.config, SteeringConfig)
        assert not agent.is_steering_active
    
    def test_init_with_config(self, mock_model, sample_vector):
        """Test initialization with config."""
        config = SteeringConfig(
            vectors=[sample_vector],
            alpha=2.0
        )
        agent = MockSteeringAgent(mock_model, config)
        assert agent.config == config
        assert len(agent.config.vectors) == 1
    
    def test_model_property(self, mock_model):
        """Test model property."""
        agent = MockSteeringAgent(mock_model)
        assert agent.model == mock_model.model
    
    def test_tokenizer_property(self, mock_model):
        """Test tokenizer property."""
        agent = MockSteeringAgent(mock_model)
        assert agent.tokenizer == mock_model.tokenizer
    
    def test_is_steering_active(self, mock_model):
        """Test is_steering_active property."""
        agent = MockSteeringAgent(mock_model)
        assert not agent.is_steering_active
        
        agent.apply_steering()
        assert agent.is_steering_active
        
        agent.remove_steering()
        assert not agent.is_steering_active
    
    def test_update_config(self, mock_model):
        """Test updating configuration."""
        agent = MockSteeringAgent(mock_model)
        assert agent.config.alpha == 1.0
        assert not agent.config.adaptive
        
        agent.update_config(alpha=2.5, adaptive=True)
        assert agent.config.alpha == 2.5
        assert agent.config.adaptive
    
    def test_add_vector(self, mock_model, sample_vector):
        """Test adding a vector."""
        agent = MockSteeringAgent(mock_model)
        assert len(agent.config.vectors) == 0
        
        agent.add_vector(sample_vector)
        assert len(agent.config.vectors) == 1
    
    def test_context_manager(self, mock_model, sample_vector):
        """Test context manager functionality."""
        agent = MockSteeringAgent(mock_model)
        agent.config.add_vector(sample_vector)
        
        assert not agent.is_steering_active
        
        with agent:
            assert agent.is_steering_active
            assert agent._apply_called
        
        assert not agent.is_steering_active
        assert agent._remove_called
    
    def test_context_manager_with_exception(self, mock_model, sample_vector):
        """Test context manager handles exceptions."""
        agent = MockSteeringAgent(mock_model)
        agent.config.add_vector(sample_vector)
        
        try:
            with agent:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Steering should still be removed
        assert not agent.is_steering_active
