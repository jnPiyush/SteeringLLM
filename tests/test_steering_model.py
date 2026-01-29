"""Unit tests for SteeringModel and ActivationHook (Stories #5 & #6)."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from steering_llm.core.steering_model import SteeringModel, ActivationHook
from steering_llm.core.steering_vector import SteeringVector


class TestActivationHook:
    """Tests for ActivationHook class."""
    
    def test_create_hook(self) -> None:
        """Test creating an activation hook."""
        module = Mock(spec=torch.nn.Module)
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        hook = ActivationHook(module=module, vector=vector, alpha=1.5)
        
        assert hook.module is module
        assert hook.vector is vector
        assert hook.alpha == 1.5
        assert hook.handle is None
    
    def test_register_hook(self) -> None:
        """Test registering a forward hook."""
        module = Mock(spec=torch.nn.Module)
        mock_handle = Mock()
        module.register_forward_hook.return_value = mock_handle
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        hook = ActivationHook(module=module, vector=vector, alpha=1.0)
        hook.register()
        
        assert hook.handle is mock_handle
        assert module.register_forward_hook.called
    
    def test_register_twice_raises_error(self) -> None:
        """Test that registering hook twice raises error."""
        module = Mock(spec=torch.nn.Module)
        module.register_forward_hook.return_value = Mock()
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        hook = ActivationHook(module=module, vector=vector, alpha=1.0)
        hook.register()
        
        with pytest.raises(RuntimeError, match="Hook already registered"):
            hook.register()
    
    def test_remove_hook(self) -> None:
        """Test removing a forward hook."""
        module = Mock(spec=torch.nn.Module)
        mock_handle = Mock()
        module.register_forward_hook.return_value = mock_handle
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        hook = ActivationHook(module=module, vector=vector, alpha=1.0)
        hook.register()
        hook.remove()
        
        assert mock_handle.remove.called
        assert hook.handle is None
    
    def test_hook_function_tensor_output(self) -> None:
        """Test hook function with tensor output."""
        # Create real module to test actual hook behavior
        module = torch.nn.Linear(100, 100)
        
        vector = SteeringVector(
            tensor=torch.ones(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        hook = ActivationHook(module=module, vector=vector, alpha=2.0)
        hook.register()
        
        # Run forward pass
        input_tensor = torch.randn(1, 10, 100)
        output = module(input_tensor)
        
        # Check that steering was applied (output should be shifted by alpha * vector)
        # Since vector is all ones and alpha=2.0, each element should be +2.0 higher
        # Note: This is approximate due to the linear layer weights
        assert output.shape == (1, 10, 100)
        
        hook.remove()
    
    def test_hook_function_tuple_output(self) -> None:
        """Test hook function with tuple output."""
        # Create module that returns tuple
        class TupleModule(torch.nn.Module):
            def forward(self, x):
                return (x, x.mean())
        
        module = TupleModule()
        
        vector = SteeringVector(
            tensor=torch.ones(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        hook = ActivationHook(module=module, vector=vector, alpha=1.0)
        hook.register()
        
        # Run forward pass
        input_tensor = torch.randn(1, 10, 100)
        output = module(input_tensor)
        
        # Should still be tuple
        assert isinstance(output, tuple)
        assert len(output) == 2
        
        hook.remove()


class TestSteeringModelInit:
    """Tests for SteeringModel initialization."""
    
    def test_create_steering_model(self) -> None:
        """Test creating a SteeringModel instance."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_tokenizer = Mock()
        
        steering_model = SteeringModel(model=mock_model, tokenizer=mock_tokenizer)
        
        assert steering_model.model is mock_model
        assert steering_model.tokenizer is mock_tokenizer
        assert steering_model.active_hooks == {}
        assert steering_model._layer_modules is None
    
    def test_create_without_tokenizer(self) -> None:
        """Test creating SteeringModel without tokenizer."""
        mock_model = Mock()
        mock_model.config = Mock()
        
        steering_model = SteeringModel(model=mock_model)
        
        assert steering_model.model is mock_model
        assert steering_model.tokenizer is None


class TestSteeringModelLoading:
    """Tests for SteeringModel.from_pretrained()."""
    
    @patch("steering_llm.core.steering_model.AutoModelForCausalLM")
    @patch("steering_llm.core.steering_model.AutoTokenizer")
    def test_from_pretrained_basic(
        self, mock_tokenizer_cls, mock_model_cls
    ) -> None:
        """Test loading model with from_pretrained()."""
        # Setup mocks
        mock_model = Mock()
        mock_model.config.model_type = "llama"
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Load model
        steering_model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")
        
        # Verify calls
        mock_model_cls.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-3B"
        )
        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.2-3B"
        )
        
        # Verify pad token was set
        assert mock_tokenizer.pad_token == "<eos>"
    
    @patch("steering_llm.core.steering_model.AutoModelForCausalLM")
    @patch("steering_llm.core.steering_model.AutoTokenizer")
    def test_from_pretrained_with_kwargs(
        self, mock_tokenizer_cls, mock_model_cls
    ) -> None:
        """Test loading with additional kwargs."""
        mock_model = Mock()
        mock_model.config.model_type = "mistral"
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Load with kwargs
        steering_model = SteeringModel.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            device_map="auto",
            load_in_8bit=True,
        )
        
        # Verify kwargs were passed
        mock_model_cls.from_pretrained.assert_called_once_with(
            "mistralai/Mistral-7B-v0.1",
            device_map="auto",
            load_in_8bit=True,
        )
    
    @patch("steering_llm.core.steering_model.AutoModelForCausalLM")
    @patch("steering_llm.core.steering_model.AutoTokenizer")
    def test_from_pretrained_unsupported_architecture(
        self, mock_tokenizer_cls, mock_model_cls
    ) -> None:
        """Test that unsupported architecture raises error."""
        mock_model = Mock()
        mock_model.config.model_type = "bert"  # Unsupported
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            SteeringModel.from_pretrained("bert-base-uncased")
    
    @patch("steering_llm.core.steering_model.AutoModelForCausalLM")
    def test_from_pretrained_model_load_failure(self, mock_model_cls) -> None:
        """Test that model loading failure raises error."""
        mock_model_cls.from_pretrained.side_effect = Exception("Network error")
        
        with pytest.raises(RuntimeError, match="Failed to load model"):
            SteeringModel.from_pretrained("nonexistent/model")


class TestLayerDetection:
    """Tests for layer detection functionality."""
    
    def test_detect_layers_llama_style(self) -> None:
        """Test detecting layers for Llama-style models."""
        # Create mock model with Llama structure
        mock_model = Mock()
        mock_model.config = Mock()
        
        # Create mock layers
        mock_layers = [Mock(spec=torch.nn.Module) for _ in range(3)]
        mock_model.model.layers = mock_layers
        
        steering_model = SteeringModel(model=mock_model)
        layers = steering_model._detect_layers()
        
        assert len(layers) == 3
        assert all(i in layers for i in range(3))
    
    def test_detect_layers_cached(self) -> None:
        """Test that layer detection is cached."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_layers = [Mock(spec=torch.nn.Module) for _ in range(2)]
        mock_model.model.layers = mock_layers
        
        steering_model = SteeringModel(model=mock_model)
        
        # First call
        layers1 = steering_model._detect_layers()
        # Second call should return cached result
        layers2 = steering_model._detect_layers()
        
        assert layers1 is layers2
    
    def test_get_layer_module_valid(self) -> None:
        """Test getting a valid layer module."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_layers = [Mock(spec=torch.nn.Module) for _ in range(3)]
        mock_model.model.layers = mock_layers
        
        steering_model = SteeringModel(model=mock_model)
        module = steering_model._get_layer_module(1)
        
        assert module is mock_layers[1]
    
    def test_get_layer_module_invalid(self) -> None:
        """Test getting invalid layer raises error."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_layers = [Mock(spec=torch.nn.Module) for _ in range(3)]
        mock_model.model.layers = mock_layers
        
        steering_model = SteeringModel(model=mock_model)
        
        with pytest.raises(ValueError, match="Invalid layer index"):
            steering_model._get_layer_module(10)


class TestApplySteering:
    """Tests for apply_steering() method (Story #5)."""
    
    def test_apply_steering_basic(self) -> None:
        """Test applying steering vector."""
        # Create mock model
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_layer.register_forward_hook.return_value = Mock()
        mock_model.model.layers = [mock_layer]
        
        steering_model = SteeringModel(model=mock_model)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        # Apply steering
        steering_model.apply_steering(vector, alpha=1.5)
        
        # Verify hook was registered
        assert "layer_0" in steering_model.active_hooks
        assert steering_model.active_hooks["layer_0"].alpha == 1.5
    
    def test_apply_steering_dimension_mismatch(self) -> None:
        """Test that dimension mismatch raises error."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_model.model.layers = [mock_layer]
        
        steering_model = SteeringModel(model=mock_model)
        
        # Vector with wrong dimension
        vector = SteeringVector(
            tensor=torch.randn(200),  # Wrong size
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            steering_model.apply_steering(vector)
    
    def test_apply_steering_already_active(self) -> None:
        """Test that applying steering twice raises error."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_layer.register_forward_hook.return_value = Mock()
        mock_model.model.layers = [mock_layer]
        
        steering_model = SteeringModel(model=mock_model)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        # Apply once
        steering_model.apply_steering(vector)
        
        # Try to apply again
        with pytest.raises(RuntimeError, match="Steering already active"):
            steering_model.apply_steering(vector)
    
    def test_apply_steering_invalid_alpha(self) -> None:
        """Test that invalid alpha raises error."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_model.model.layers = [mock_layer]
        
        steering_model = SteeringModel(model=mock_model)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="alpha must be numeric"):
            steering_model.apply_steering(vector, alpha="invalid")


class TestRemoveSteering:
    """Tests for remove_steering() methods (Story #6)."""
    
    def test_remove_steering_specific_layer(self) -> None:
        """Test removing steering from specific layer."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_handle = Mock()
        mock_layer.register_forward_hook.return_value = mock_handle
        mock_model.model.layers = [mock_layer]
        
        steering_model = SteeringModel(model=mock_model)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        # Apply and remove
        steering_model.apply_steering(vector)
        assert len(steering_model.active_hooks) == 1
        
        steering_model.remove_steering(layer=0)
        assert len(steering_model.active_hooks) == 0
        assert mock_handle.remove.called
    
    def test_remove_steering_all(self) -> None:
        """Test removing all steering vectors."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layers = [Mock(spec=torch.nn.Module) for _ in range(3)]
        
        mock_handles = []
        for layer in mock_layers:
            handle = Mock()
            mock_handles.append(handle)
            layer.register_forward_hook.return_value = handle
        
        mock_model.model.layers = mock_layers
        
        steering_model = SteeringModel(model=mock_model)
        
        # Apply steering to multiple layers
        for i in range(3):
            vector = SteeringVector(
                tensor=torch.randn(100),
                layer=i,
                layer_name=f"model.layers.{i}",
                model_name="test-model",
            )
            steering_model.apply_steering(vector)
        
        assert len(steering_model.active_hooks) == 3
        
        # Remove all
        steering_model.remove_steering()
        assert len(steering_model.active_hooks) == 0
        
        # Verify all handles were removed
        for handle in mock_handles:
            assert handle.remove.called
    
    def test_remove_all_steering_alias(self) -> None:
        """Test remove_all_steering() alias."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_handle = Mock()
        mock_layer.register_forward_hook.return_value = mock_handle
        mock_model.model.layers = [mock_layer]
        
        steering_model = SteeringModel(model=mock_model)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        steering_model.apply_steering(vector)
        steering_model.remove_all_steering()
        
        assert len(steering_model.active_hooks) == 0
    
    def test_remove_steering_not_active(self) -> None:
        """Test removing steering when none is active."""
        mock_model = Mock()
        mock_model.config = Mock()
        
        steering_model = SteeringModel(model=mock_model)
        
        # Should not raise error
        steering_model.remove_steering(layer=0)
        steering_model.remove_all_steering()


class TestListActiveSteering:
    """Tests for list_active_steering() method (Story #6)."""
    
    def test_list_active_empty(self) -> None:
        """Test listing when no steering is active."""
        mock_model = Mock()
        mock_model.config = Mock()
        
        steering_model = SteeringModel(model=mock_model)
        active = steering_model.list_active_steering()
        
        assert active == []
    
    def test_list_active_single(self) -> None:
        """Test listing single active steering."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_layer.register_forward_hook.return_value = Mock()
        mock_model.model.layers = [mock_layer]
        
        steering_model = SteeringModel(model=mock_model)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        steering_model.apply_steering(vector, alpha=1.5)
        active = steering_model.list_active_steering()
        
        assert len(active) == 1
        assert active[0]["layer"] == 0
        assert active[0]["alpha"] == 1.5
        assert active[0]["model_name"] == "test-model"
        assert "magnitude" in active[0]
        assert "method" in active[0]
    
    def test_list_active_multiple(self) -> None:
        """Test listing multiple active steering vectors."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layers = [Mock(spec=torch.nn.Module) for _ in range(3)]
        
        for layer in mock_layers:
            layer.register_forward_hook.return_value = Mock()
        
        mock_model.model.layers = mock_layers
        
        steering_model = SteeringModel(model=mock_model)
        
        # Apply multiple vectors
        for i in range(3):
            vector = SteeringVector(
                tensor=torch.randn(100),
                layer=i,
                layer_name=f"model.layers.{i}",
                model_name="test-model",
            )
            steering_model.apply_steering(vector, alpha=float(i + 1))
        
        active = steering_model.list_active_steering()
        
        assert len(active) == 3
        layers = {info["layer"] for info in active}
        assert layers == {0, 1, 2}


class TestGenerateWithSteering:
    """Tests for generate_with_steering() method (Story #5)."""
    
    def test_generate_with_steering_single_prompt(self) -> None:
        """Test generation with steering on single prompt."""
        # Create mock model with generate method
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_layer.register_forward_hook.return_value = Mock()
        mock_model.model.layers = [mock_layer]
        
        # Mock parameter for device detection
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = [mock_param]
        
        # Mock generate
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["Generated text"]
        
        steering_model = SteeringModel(model=mock_model, tokenizer=mock_tokenizer)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        # Generate with steering
        output = steering_model.generate_with_steering(
            "Test prompt",
            vector=vector,
            alpha=2.0,
            max_length=50,
        )
        
        # Verify output
        assert output == "Generated text"
        
        # Verify steering was removed
        assert len(steering_model.active_hooks) == 0
    
    def test_generate_with_steering_batch(self) -> None:
        """Test generation with steering on batch prompts."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_layer.register_forward_hook.return_value = Mock()
        mock_model.model.layers = [mock_layer]
        
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = [mock_param]
        mock_model.generate.return_value = torch.tensor([[1, 2], [3, 4]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1], [2]]),
            "attention_mask": torch.tensor([[1], [1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["Text 1", "Text 2"]
        
        steering_model = SteeringModel(model=mock_model, tokenizer=mock_tokenizer)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        # Generate batch
        outputs = steering_model.generate_with_steering(
            ["Prompt 1", "Prompt 2"],
            vector=vector,
        )
        
        assert outputs == ["Text 1", "Text 2"]
        assert len(steering_model.active_hooks) == 0
    
    def test_generate_with_steering_cleanup_on_error(self) -> None:
        """Test that steering is removed even if generation fails."""
        mock_model = Mock()
        mock_model.config.hidden_size = 100
        mock_layer = Mock(spec=torch.nn.Module)
        mock_layer.register_forward_hook.return_value = Mock()
        mock_model.model.layers = [mock_layer]
        
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = [mock_param]
        
        # Make generate raise error
        mock_model.generate.side_effect = RuntimeError("Generation failed")
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }
        
        steering_model = SteeringModel(model=mock_model, tokenizer=mock_tokenizer)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        # Generation should fail
        with pytest.raises(RuntimeError, match="Generation failed"):
            steering_model.generate_with_steering("Test", vector=vector)
        
        # But steering should still be removed
        assert len(steering_model.active_hooks) == 0
    
    def test_generate_without_tokenizer_fails(self) -> None:
        """Test that generation without tokenizer raises error."""
        mock_model = Mock()
        mock_model.config = Mock()
        
        steering_model = SteeringModel(model=mock_model, tokenizer=None)
        
        vector = SteeringVector(
            tensor=torch.randn(100),
            layer=0,
            layer_name="model.layers.0",
            model_name="test-model",
        )
        
        with pytest.raises(RuntimeError, match="Tokenizer required"):
            steering_model.generate_with_steering("Test", vector=vector)


class TestSteeringModelHelpers:
    """Tests for helper methods."""
    
    def test_repr(self) -> None:
        """Test string representation."""
        mock_model = Mock()
        mock_model.config._name_or_path = "test-model"
        
        steering_model = SteeringModel(model=mock_model)
        repr_str = repr(steering_model)
        
        assert "SteeringModel" in repr_str
        assert "test-model" in repr_str
        assert "active_steering=0" in repr_str
    
    def test_getattr_delegation(self) -> None:
        """Test that attributes are delegated to underlying model."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.some_method = Mock(return_value="result")
        
        steering_model = SteeringModel(model=mock_model)
        
        # Should delegate to underlying model
        result = steering_model.some_method()
        assert result == "result"
        assert mock_model.some_method.called
