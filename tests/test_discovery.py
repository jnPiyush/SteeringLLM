"""Unit tests for Discovery class."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from steering_llm.core.discovery import Discovery, DiscoveryResult
from steering_llm.core.steering_vector import SteeringVector
from steering_llm.exceptions import EmptyDatasetError, LayerDetectionError


class TestMeanDifferenceValidation:
    """Tests for input validation in mean_difference."""
    
    def test_empty_positive_list(self) -> None:
        """Test error when positive list is empty."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(EmptyDatasetError):
            Discovery.mean_difference(
                positive=[],
                negative=["negative example"],
                model=model,
                layer=0,
            )
    
    def test_empty_negative_list(self) -> None:
        """Test error when negative list is empty."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(EmptyDatasetError):
            Discovery.mean_difference(
                positive=["positive example"],
                negative=[],
                model=model,
                layer=0,
            )
    
    def test_invalid_layer_negative(self) -> None:
        """Test error when layer is negative."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(ValueError, match="layer must be non-negative integer"):
            Discovery.mean_difference(
                positive=["positive"],
                negative=["negative"],
                model=model,
                layer=-1,
            )
    
    def test_invalid_layer_type(self) -> None:
        """Test error when layer is not an integer."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(ValueError, match="layer must be non-negative integer"):
            Discovery.mean_difference(
                positive=["positive"],
                negative=["negative"],
                model=model,
                layer="15",  # type: ignore
            )


class TestLayerDetection:
    """Tests for layer name detection."""
    
    def test_detect_llama_layer_name(self) -> None:
        """Test detecting layer name for Llama architecture."""
        # Mock a Llama-style model
        model = MagicMock(spec=PreTrainedModel)
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(32)]
        
        layer_name = Discovery._detect_layer_name(model, 15)
        assert layer_name == "model.layers.15"
    
    def test_detect_layer_out_of_range(self) -> None:
        """Test error when layer index is out of range."""
        model = MagicMock(spec=PreTrainedModel)
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        
        with pytest.raises(LayerDetectionError, match="Cannot detect layer name"):
            Discovery._detect_layer_name(model, 20)


class TestGetLayerModule:
    """Tests for getting layer module by name."""
    
    def test_get_layer_module_success(self) -> None:
        """Test successfully getting a layer module."""
        # Create mock model structure
        layer_module = MagicMock()
        model = MagicMock(spec=PreTrainedModel)
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(32)]
        model.model.layers[15] = layer_module
        
        result = Discovery._get_layer_module(model, "model.layers.15")
        assert result == layer_module
    
    def test_get_layer_module_not_found(self) -> None:
        """Test error when layer module not found."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(ValueError, match="Layer not found"):
            Discovery._get_layer_module(model, "nonexistent.layer.99")


class TestExtractActivations:
    """Tests for activation extraction."""
    
    @patch("steering_llm.core.discovery.Discovery._get_layer_module")
    def test_extract_activations_shape(self, mock_get_layer: MagicMock) -> None:
        """Test that extracted activations have correct shape."""
        # Create mock components
        model = MagicMock(spec=PreTrainedModel)
        tokenizer = MagicMock(spec=PreTrainedTokenizer)
        device = torch.device("cpu")
        
        # Mock tokenizer output
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        
        # Create a real module for hooking
        hidden_dim = 3072
        target_module = torch.nn.Linear(hidden_dim, hidden_dim)
        mock_get_layer.return_value = target_module
        
        # Mock model forward to trigger hook
        def mock_forward(**kwargs):
            # Simulate activation output
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            return MagicMock(
                logits=torch.randn(batch_size, seq_len, 1000)
            )
        
        model.side_effect = mock_forward
        
        # Patch the forward hook to inject our test activation
        original_register = target_module.register_forward_hook
        
        def mock_register_hook(hook_fn):
            # Call hook with test data
            test_activation = (torch.randn(2, 10, hidden_dim),)
            hook_fn(target_module, None, test_activation)
            
            # Return a mock handle
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook
        
        texts = ["example 1", "example 2"]
        
        activations = Discovery._extract_activations(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            layer=15,
            layer_name="model.layers.15",
            batch_size=2,
            max_length=128,
            device=device,
        )
        
        # Should have shape [num_texts, hidden_dim]
        assert activations.shape == (2, hidden_dim)
    
    @patch("steering_llm.core.discovery.Discovery._get_layer_module")
    def test_extract_activations_batching(self, mock_get_layer: MagicMock) -> None:
        """Test that extraction handles batching correctly."""
        model = MagicMock(spec=PreTrainedModel)
        tokenizer = MagicMock(spec=PreTrainedTokenizer)
        device = torch.device("cpu")
        hidden_dim = 100
        
        # Mock tokenizer
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        
        # Create target module
        target_module = torch.nn.Linear(hidden_dim, hidden_dim)
        mock_get_layer.return_value = target_module
        
        call_count = [0]
        
        def mock_register_hook(hook_fn):
            # Inject activation
            test_activation = (torch.randn(1, 10, hidden_dim),)
            hook_fn(target_module, None, test_activation)
            call_count[0] += 1
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook
        
        # Process 5 texts with batch_size=2 (should make 3 batches)
        texts = [f"text {i}" for i in range(5)]
        
        activations = Discovery._extract_activations(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            layer=0,
            layer_name="layer.0",
            batch_size=2,
            max_length=128,
            device=device,
        )
        
        # Should process 5 texts in 3 batches (2+2+1)
        # Note: With batch processing, we capture 3 batch activations
        assert activations.shape[0] == 3  # 3 batches captured
        assert call_count[0] == 3  # 3 forward passes


class TestMeanDifferenceIntegration:
    """Integration tests for mean_difference method."""
    
    @patch("steering_llm.core.discovery.AutoTokenizer")
    def test_mean_difference_returns_steering_vector(
        self, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that mean_difference returns a valid SteeringVector."""
        # Setup mocks
        hidden_dim = 100
        layer = 5
        
        # Mock model with config
        model = MagicMock(spec=PreTrainedModel)
        mock_config = MagicMock()
        mock_config._name_or_path = "test-model"
        mock_config.model_type = "llama"
        model.config = mock_config
        
        # Mock model structure
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        target_module = torch.nn.Linear(hidden_dim, hidden_dim)
        model.model.layers[layer] = target_module
        
        # Mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock device
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock])
        
        # Setup hook to return deterministic activations
        pos_activation = torch.ones(1, 10, hidden_dim) * 2.0
        neg_activation = torch.ones(1, 10, hidden_dim) * 1.0
        
        activation_queue = [pos_activation, neg_activation]
        
        def mock_register_hook(hook_fn):
            if activation_queue:
                activation = (activation_queue.pop(0),)
                hook_fn(target_module, None, activation)
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook
        
        # Call mean_difference
        positive = ["I love this"]
        negative = ["I hate this"]
        
        result = Discovery.mean_difference(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
        )
        
        # Verify result is a DiscoveryResult with SteeringVector
        assert isinstance(result, DiscoveryResult)
        vector = result.vector
        assert isinstance(vector, SteeringVector)
        assert vector.layer == layer
        assert vector.model_name == "test-model"
        assert vector.method == "mean_difference"
        assert vector.tensor.shape == (hidden_dim,)
        
        # Verify vector is difference of means
        # pos_activation mean over seq: [1, hidden_dim] with value 2.0
        # neg_activation mean over seq: [1, hidden_dim] with value 1.0
        # difference should be 1.0 everywhere
        expected_vector = torch.ones(hidden_dim) * 1.0
        assert torch.allclose(vector.tensor, expected_vector, atol=1e-5)
    
    @patch("steering_llm.core.discovery.AutoTokenizer")
    def test_mean_difference_with_multiple_samples(
        self, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test mean_difference with multiple positive and negative samples."""
        hidden_dim = 50
        layer = 2
        
        # Mock model with config
        model = MagicMock(spec=PreTrainedModel)
        mock_config = MagicMock()
        mock_config._name_or_path = "test-model"
        model.config = mock_config
        
        # Setup model structure
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        target_module = torch.nn.Linear(hidden_dim, hidden_dim)
        model.model.layers[layer] = target_module
        
        # Mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock device
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock])
        
        # Create activations (4 positive, 4 negative, batch_size=2)
        # Positive: mean will be 3.0, Negative: mean will be 1.0
        activations = [
            torch.ones(2, 10, hidden_dim) * 3.0,  # pos batch 1
            torch.ones(2, 10, hidden_dim) * 3.0,  # pos batch 2
            torch.ones(2, 10, hidden_dim) * 1.0,  # neg batch 1
            torch.ones(2, 10, hidden_dim) * 1.0,  # neg batch 2
        ]
        
        activation_queue = activations.copy()
        
        def mock_register_hook(hook_fn):
            if activation_queue:
                activation = (activation_queue.pop(0),)
                hook_fn(target_module, None, activation)
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook
        
        # Call with 4 positive and 4 negative examples
        positive = [f"positive {i}" for i in range(4)]
        negative = [f"negative {i}" for i in range(4)]
        
        result = Discovery.mean_difference(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
            batch_size=2,
        )
        
        # Verify
        vector = result.vector
        assert vector.metadata["positive_samples"] == 4
        assert vector.metadata["negative_samples"] == 4
        
        # Vector should be 3.0 - 1.0 = 2.0
        expected_vector = torch.ones(hidden_dim) * 2.0
        assert torch.allclose(vector.tensor, expected_vector, atol=1e-5)
    
    def test_mean_difference_with_provided_tokenizer(self) -> None:
        """Test that provided tokenizer is used instead of auto-detection."""
        hidden_dim = 50
        layer = 0
        
        # Mock model with config
        model = MagicMock(spec=PreTrainedModel)
        mock_config = MagicMock()
        mock_config._name_or_path = "test-model"
        model.config = mock_config
        
        # Setup model structure
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        target_module = torch.nn.Linear(hidden_dim, hidden_dim)
        model.model.layers[layer] = target_module
        
        # Mock provided tokenizer
        provided_tokenizer = MagicMock(spec=PreTrainedTokenizer)
        provided_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        
        # Mock device
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock])
        
        # Setup activations
        def mock_register_hook(hook_fn):
            activation = (torch.ones(1, 10, hidden_dim),)
            hook_fn(target_module, None, activation)
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook
        
        # Call with provided tokenizer (should NOT call AutoTokenizer)
        with patch("steering_llm.core.discovery.AutoTokenizer") as mock_auto:
            result = Discovery.mean_difference(
                positive=["positive"],
                negative=["negative"],
                model=model,
                layer=layer,
                tokenizer=provided_tokenizer,
            )
            
            # AutoTokenizer should NOT be called
            mock_auto.from_pretrained.assert_not_called()
            
            # Provided tokenizer should be used
            assert provided_tokenizer.call_count >= 2  # Called for pos and neg
