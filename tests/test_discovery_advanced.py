"""Unit tests for advanced discovery methods (CAA and Linear Probing)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from steering_llm.core.discovery import Discovery, DiscoveryResult
from steering_llm.core.steering_vector import SteeringVector
from steering_llm.exceptions import EmptyDatasetError


class TestCAAValidation:
    """Tests for input validation in CAA method."""
    
    def test_empty_positive_list(self) -> None:
        """Test error when positive list is empty."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(EmptyDatasetError):
            Discovery.caa(
                positive=[],
                negative=["negative"],
                model=model,
                layer=0,
            )
    
    def test_empty_negative_list(self) -> None:
        """Test error when negative list is empty."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(EmptyDatasetError):
            Discovery.caa(
                positive=["positive"],
                negative=[],
                model=model,
                layer=0,
            )
    
    def test_unequal_list_lengths(self) -> None:
        """Test error when positive and negative lists have different lengths."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(ValueError, match="equal number of positive and negative examples"):
            Discovery.caa(
                positive=["pos1", "pos2", "pos3"],
                negative=["neg1", "neg2"],
                model=model,
                layer=0,
            )
    
    def test_invalid_layer(self) -> None:
        """Test error when layer is invalid."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(ValueError, match="layer must be non-negative integer"):
            Discovery.caa(
                positive=["pos"],
                negative=["neg"],
                model=model,
                layer=-1,
            )


class TestCAAIntegration:
    """Integration tests for CAA method."""
    
    @patch("steering_llm.core.discovery.AutoTokenizer")
    @patch("steering_llm.core.discovery.Discovery._extract_activations")
    def test_caa_returns_discovery_result(
        self, mock_extract: MagicMock, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that CAA returns a valid DiscoveryResult."""
        hidden_dim = 100
        layer = 5
        
        # Mock model
        model = MagicMock(spec=PreTrainedModel)
        mock_config = MagicMock()
        mock_config._name_or_path = "test-model"
        model.config = mock_config
        
        # Setup model structure
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        model.model.layers[layer] = MagicMock()
        
        # Mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock device
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock])
        
        # Mock _extract_activations to return controlled data
        # pos=[3,3], neg=[1,1]
        pos_activations = torch.ones(2, hidden_dim) * 3.0
        neg_activations = torch.ones(2, hidden_dim) * 1.0
        mock_extract.side_effect = [pos_activations, neg_activations]
        
        # Call CAA
        positive = ["pos1", "pos2"]
        negative = ["neg1", "neg2"]
        
        result = Discovery.caa(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
        )
        
        # Verify result
        assert isinstance(result, DiscoveryResult)
        vector = result.vector
        assert isinstance(vector, SteeringVector)
        assert vector.layer == layer
        assert vector.method == "caa"
        assert vector.metadata["contrast_pairs"] == 2
        assert vector.tensor.shape == (hidden_dim,)
        
        # CAA: contrasts = pos - neg = [3,3] - [1,1] = [2,2], mean = 2
        expected_vector = torch.ones(hidden_dim) * 2.0
        assert torch.allclose(vector.tensor, expected_vector, atol=1e-5)
    
    @patch("steering_llm.core.discovery.AutoTokenizer")
    def test_caa_num_pairs_limit(
        self, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that num_pairs parameter limits the number of contrast pairs."""
        hidden_dim = 50
        layer = 2
        
        # Mock model
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
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock device
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock])
        
        # Setup activations for 5 pairs, but only 2 should be used
        activations = [
            torch.ones(1, 10, hidden_dim) * 2.0,  # pos 1 (used)
            torch.ones(1, 10, hidden_dim) * 2.0,  # pos 2 (used)
            torch.ones(1, 10, hidden_dim) * 1.0,  # neg 1 (used)
            torch.ones(1, 10, hidden_dim) * 1.0,  # neg 2 (used)
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
        
        # Call CAA with num_pairs=2 (should use first 2 pairs only)
        positive = [f"pos{i}" for i in range(5)]
        negative = [f"neg{i}" for i in range(5)]
        
        result = Discovery.caa(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
            num_pairs=2,
        )
        
        # Verify only 2 pairs were used
        assert result.vector.metadata["contrast_pairs"] == 2


class TestLinearProbeValidation:
    """Tests for input validation in linear_probe method."""
    
    def test_empty_positive_list(self) -> None:
        """Test error when positive list is empty."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(EmptyDatasetError):
            Discovery.linear_probe(
                positive=[],
                negative=["negative"],
                model=model,
                layer=0,
            )
    
    def test_empty_negative_list(self) -> None:
        """Test error when negative list is empty."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(EmptyDatasetError):
            Discovery.linear_probe(
                positive=["positive"],
                negative=[],
                model=model,
                layer=0,
            )
    
    def test_invalid_layer(self) -> None:
        """Test error when layer is invalid."""
        model = MagicMock(spec=PreTrainedModel)
        
        with pytest.raises(ValueError, match="layer must be non-negative integer"):
            Discovery.linear_probe(
                positive=["pos"],
                negative=["neg"],
                model=model,
                layer=-1,
            )


class TestLinearProbeIntegration:
    """Integration tests for linear_probe method."""
    
    @patch("steering_llm.core.discovery.AutoTokenizer")
    @patch("steering_llm.core.discovery.Discovery._extract_activations")
    def test_linear_probe_returns_vector_and_metrics(
        self, mock_extract: MagicMock, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that linear_probe returns vector and metrics."""
        hidden_dim = 100
        layer = 3
        
        # Mock model
        model = MagicMock(spec=PreTrainedModel)
        mock_config = MagicMock()
        mock_config._name_or_path = "test-model"
        model.config = mock_config
        
        # Setup model structure
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        model.model.layers[layer] = MagicMock()
        
        # Mock tokenizer
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock device
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock])
        
        # Create linearly separable activations
        np.random.seed(42)
        pos_activations_np = np.concatenate([
            np.random.randn(10, hidden_dim // 2) + 2.0,  # high
            np.random.randn(10, hidden_dim // 2) - 2.0,  # low
        ], axis=1)
        
        neg_activations_np = np.concatenate([
            np.random.randn(10, hidden_dim // 2) - 2.0,  # low
            np.random.randn(10, hidden_dim // 2) + 2.0,  # high
        ], axis=1)
        
        pos_activations = torch.from_numpy(pos_activations_np).float()
        neg_activations = torch.from_numpy(neg_activations_np).float()
        
        # Mock _extract_activations
        mock_extract.side_effect = [pos_activations, neg_activations]
        
        # Call linear_probe
        positive = [f"pos{i}" for i in range(10)]
        negative = [f"neg{i}" for i in range(10)]
        
        result = Discovery.linear_probe(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
        )
        
        # Verify result
        assert isinstance(result, DiscoveryResult)
        vector = result.vector
        metrics = result.metrics
        
        assert isinstance(vector, SteeringVector)
        assert vector.layer == layer
        assert vector.method == "linear_probe"
        assert vector.tensor.shape == (hidden_dim,)
        
        # Verify metrics
        assert "train_accuracy" in metrics
        assert isinstance(metrics["train_accuracy"], float)
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
        
        # With linearly separable data, accuracy should be high
        assert metrics["train_accuracy"] >= 0.8, "Probe should achieve high accuracy on separable data"
        
        assert metrics["positive_samples"] == 10
        assert metrics["negative_samples"] == 10
        assert "C" in metrics
        assert "normalized" in metrics
    
    @patch("steering_llm.core.discovery.AutoTokenizer")
    def test_linear_probe_normalization(
        self, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that normalization parameter works correctly."""
        hidden_dim = 50
        layer = 2
        
        # Mock model
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
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock device
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock])
        
        # Create simple separable data
        np.random.seed(42)
        pos_activations_np = np.random.randn(2, hidden_dim) + 1.0
        neg_activations_np = np.random.randn(2, hidden_dim) - 1.0
        
        pos_activations = torch.from_numpy(pos_activations_np).float()
        neg_activations = torch.from_numpy(neg_activations_np).float()
        
        call_count = [0]
        
        def mock_register_hook(hook_fn):
            if call_count[0] == 0:
                expanded = pos_activations.unsqueeze(1).expand(-1, 10, -1)
                hook_fn(target_module, None, (expanded,))
                call_count[0] += 1
            elif call_count[0] == 1:
                expanded = neg_activations.unsqueeze(1).expand(-1, 10, -1)
                hook_fn(target_module, None, (expanded,))
                call_count[0] += 1
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook
        
        # Call with normalize=False
        positive = [f"pos{i}" for i in range(2)]
        negative = [f"neg{i}" for i in range(2)]
        
        result = Discovery.linear_probe(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
            normalize=False,
        )
        
        # Verify normalization flag in metadata
        vector = result.vector
        metrics = result.metrics
        assert metrics["normalized"] is False
        assert isinstance(vector, SteeringVector)


class TestDiscoveryMethodComparison:
    """Tests comparing different discovery methods."""
    
    @patch("steering_llm.core.discovery.AutoTokenizer")
    def test_all_methods_produce_vectors(
        self, mock_tokenizer_class: MagicMock
    ) -> None:
        """Test that all discovery methods produce valid vectors."""
        hidden_dim = 100
        layer = 5
        
        # Setup common mocks
        model = MagicMock(spec=PreTrainedModel)
        mock_config = MagicMock()
        mock_config._name_or_path = "test-model"
        model.config = mock_config
        
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        target_module = torch.nn.Linear(hidden_dim, hidden_dim)
        model.model.layers[layer] = target_module
        
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        param_mock = MagicMock()
        param_mock.device = torch.device("cpu")
        model.parameters.return_value = iter([param_mock] * 10)  # Multiple calls
        
        # Create test data
        positive = ["I love this", "This is great"]
        negative = ["I hate this", "This is terrible"]
        
        # Test mean_difference
        call_count_md = [0]
        activations_md = [
            torch.randn(2, 10, hidden_dim),  # pos batch
            torch.randn(2, 10, hidden_dim),  # neg batch
        ]
        
        def mock_register_hook_md(hook_fn):
            if call_count_md[0] < len(activations_md):
                activation = (activations_md[call_count_md[0]],)
                hook_fn(target_module, None, activation)
                call_count_md[0] += 1
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook_md
        
        result_md = Discovery.mean_difference(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
        )
        
        vec_md = result_md.vector
        assert isinstance(vec_md, SteeringVector)
        assert vec_md.method == "mean_difference"
        
        # Reset for CAA
        model.parameters.return_value = iter([param_mock] * 10)
        call_count_caa = [0]
        activations_caa = [
            torch.randn(2, 10, hidden_dim),  # pos batch
            torch.randn(2, 10, hidden_dim),  # neg batch
        ]
        
        def mock_register_hook_caa(hook_fn):
            if call_count_caa[0] < len(activations_caa):
                activation = (activations_caa[call_count_caa[0]],)
                hook_fn(target_module, None, activation)
                call_count_caa[0] += 1
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook_caa
        
        result_caa = Discovery.caa(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
        )
        
        vec_caa = result_caa.vector
        assert isinstance(vec_caa, SteeringVector)
        assert vec_caa.method == "caa"
        
        # Reset for linear_probe
        model.parameters.return_value = iter([param_mock] * 10)
        np.random.seed(42)
        pos_act = torch.from_numpy(np.random.randn(2, hidden_dim) + 1.0).float()
        neg_act = torch.from_numpy(np.random.randn(2, hidden_dim) - 1.0).float()
        
        call_count_lp = [0]
        
        def mock_register_hook_lp(hook_fn):
            if call_count_lp[0] == 0:
                expanded = pos_act.unsqueeze(1).expand(-1, 10, -1)
                hook_fn(target_module, None, (expanded,))
                call_count_lp[0] += 1
            elif call_count_lp[0] == 1:
                expanded = neg_act.unsqueeze(1).expand(-1, 10, -1)
                hook_fn(target_module, None, (expanded,))
                call_count_lp[0] += 1
            
            handle = MagicMock()
            handle.remove = MagicMock()
            return handle
        
        target_module.register_forward_hook = mock_register_hook_lp
        
        result_lp = Discovery.linear_probe(
            positive=positive,
            negative=negative,
            model=model,
            layer=layer,
        )
        
        vec_lp = result_lp.vector
        metrics_lp = result_lp.metrics
        assert isinstance(vec_lp, SteeringVector)
        assert vec_lp.method == "linear_probe"
        assert "train_accuracy" in metrics_lp
