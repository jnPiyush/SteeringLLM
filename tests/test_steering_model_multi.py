"""Unit tests for multi-vector steering in SteeringModel."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from steering_llm.core.steering_model import SteeringModel
from steering_llm.core.steering_vector import SteeringVector
from steering_llm.exceptions import SteeringActiveError


class TestApplyMultipleSteering:
    """Tests for apply_multiple_steering method."""
    
    def test_apply_multiple_steering_two_layers(self) -> None:
        """Test applying vectors to different layers."""
        # Create mock model
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        # Create layer modules
        layer_modules = {
            5: Mock(spec=torch.nn.Module),
            10: Mock(spec=torch.nn.Module),
        }
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        # Mock register_forward_hook
        for layer_module in layer_modules.values():
            layer_module.register_forward_hook = Mock(return_value=Mock())
        
        # Create vectors for different layers
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=10,
            layer_name="model.layers.10",
            model_name="test-model",
        )
        
        # Apply multiple vectors
        steering_model.apply_multiple_steering(
            vectors=[vec1, vec2],
            alphas=[1.0, 1.5],
        )
        
        # Check that both hooks are registered
        assert "layer_5" in steering_model.active_hooks
        assert "layer_10" in steering_model.active_hooks
        
        # Check alphas
        assert steering_model.active_hooks["layer_5"].alpha == 1.0
        assert steering_model.active_hooks["layer_10"].alpha == 1.5
    
    def test_apply_multiple_steering_default_alphas(self) -> None:
        """Test that default alphas are 1.0."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        layer_modules = {
            5: Mock(spec=torch.nn.Module),
            10: Mock(spec=torch.nn.Module),
        }
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        for layer_module in layer_modules.values():
            layer_module.register_forward_hook = Mock(return_value=Mock())
        
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=10,
            layer_name="model.layers.10",
            model_name="test-model",
        )
        
        # Apply without specifying alphas
        steering_model.apply_multiple_steering(vectors=[vec1, vec2])
        
        # Check that default alphas are 1.0
        assert steering_model.active_hooks["layer_5"].alpha == 1.0
        assert steering_model.active_hooks["layer_10"].alpha == 1.0
    
    def test_apply_multiple_steering_empty_error(self) -> None:
        """Test error when vectors list is empty."""
        mock_model = Mock()
        mock_model.config = Mock()
        
        steering_model = SteeringModel(model=mock_model)
        
        with pytest.raises(ValueError, match="vectors list cannot be empty"):
            steering_model.apply_multiple_steering(vectors=[])
    
    def test_apply_multiple_steering_mismatched_alphas_error(self) -> None:
        """Test error when alphas length doesn't match vectors."""
        mock_model = Mock()
        mock_model.config = Mock()
        
        steering_model = SteeringModel(model=mock_model)
        
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=10,
            layer_name="model.layers.10",
            model_name="test-model",
        )
        
        with pytest.raises(ValueError, match="Number of alphas.*must match"):
            steering_model.apply_multiple_steering(
                vectors=[vec1, vec2],
                alphas=[1.0],  # Only 1 alpha for 2 vectors
            )
    
    def test_apply_multiple_steering_conflict_error(self) -> None:
        """Test error when trying to apply to layer with existing steering."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        layer_modules = {5: Mock(spec=torch.nn.Module)}
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        layer_modules[5].register_forward_hook = Mock(return_value=Mock())
        
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        # Apply first vector
        steering_model.apply_steering(vec1)
        
        # Try to apply second vector to same layer
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        with pytest.raises(SteeringActiveError, match="Steering already active on layer"):
            steering_model.apply_multiple_steering(vectors=[vec2])
    
    def test_apply_multiple_steering_five_vectors(self) -> None:
        """Test applying 5+ vectors simultaneously."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        # Create 5 layer modules
        layer_modules = {i: Mock(spec=torch.nn.Module) for i in range(5, 10)}
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        for layer_module in layer_modules.values():
            layer_module.register_forward_hook = Mock(return_value=Mock())
        
        # Create 5 vectors for different layers
        vectors = []
        alphas = []
        for i in range(5):
            vec = SteeringVector(
                tensor=torch.randn(100),
                layer=5 + i,
                layer_name=f"model.layers.{5+i}",
                model_name="test-model",
            )
            vectors.append(vec)
            alphas.append(0.5 + i * 0.2)
        
        # Apply all 5 vectors
        steering_model.apply_multiple_steering(vectors=vectors, alphas=alphas)
        
        # Verify all 5 hooks are active
        assert len(steering_model.active_hooks) == 5
        
        for i in range(5):
            layer_key = f"layer_{5+i}"
            assert layer_key in steering_model.active_hooks
            assert steering_model.active_hooks[layer_key].alpha == alphas[i]


class TestMultiVectorRemoval:
    """Tests for removing multiple steering vectors."""
    
    def test_remove_all_steering_with_multiple_vectors(self) -> None:
        """Test removing all steering when multiple vectors are active."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        layer_modules = {
            5: Mock(spec=torch.nn.Module),
            10: Mock(spec=torch.nn.Module),
            15: Mock(spec=torch.nn.Module),
        }
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        for layer_module in layer_modules.values():
            layer_module.register_forward_hook = Mock(return_value=Mock())
        
        # Create and apply 3 vectors
        vectors = []
        for layer in [5, 10, 15]:
            vec = SteeringVector(
                tensor=torch.randn(100),
                layer=layer,
                layer_name=f"model.layers.{layer}",
                model_name="test-model",
            )
            vectors.append(vec)
        
        steering_model.apply_multiple_steering(vectors=vectors)
        
        # Verify 3 hooks are active
        assert len(steering_model.active_hooks) == 3
        
        # Remove all
        steering_model.remove_all_steering()
        
        # Verify all hooks are removed
        assert len(steering_model.active_hooks) == 0
    
    def test_remove_specific_layer_with_multiple_active(self) -> None:
        """Test removing steering from specific layer when multiple are active."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        layer_modules = {
            5: Mock(spec=torch.nn.Module),
            10: Mock(spec=torch.nn.Module),
        }
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        for layer_module in layer_modules.values():
            layer_module.register_forward_hook = Mock(return_value=Mock())
        
        # Apply 2 vectors
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="test-model",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=10,
            layer_name="model.layers.10",
            model_name="test-model",
        )
        
        steering_model.apply_multiple_steering(vectors=[vec1, vec2])
        
        # Remove only layer 5
        steering_model.remove_steering(layer=5)
        
        # Verify layer 5 is removed but layer 10 remains
        assert "layer_5" not in steering_model.active_hooks
        assert "layer_10" in steering_model.active_hooks


class TestListActiveSteering:
    """Tests for list_active_steering with multiple vectors."""
    
    def test_list_active_steering_multiple_vectors(self) -> None:
        """Test listing multiple active steering vectors."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        layer_modules = {
            5: Mock(spec=torch.nn.Module),
            10: Mock(spec=torch.nn.Module),
            15: Mock(spec=torch.nn.Module),
        }
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        for layer_module in layer_modules.values():
            layer_module.register_forward_hook = Mock(return_value=Mock())
        
        # Create 3 vectors with different methods and alphas
        vec1 = SteeringVector(
            tensor=torch.randn(100),
            layer=5,
            layer_name="model.layers.5",
            model_name="model-a",
            method="mean_difference",
        )
        
        vec2 = SteeringVector(
            tensor=torch.randn(100),
            layer=10,
            layer_name="model.layers.10",
            model_name="model-b",
            method="caa",
        )
        
        vec3 = SteeringVector(
            tensor=torch.randn(100),
            layer=15,
            layer_name="model.layers.15",
            model_name="model-c",
            method="linear_probe",
        )
        
        steering_model.apply_multiple_steering(
            vectors=[vec1, vec2, vec3],
            alphas=[1.0, 1.5, 2.0],
        )
        
        # List active steering
        active = steering_model.list_active_steering()
        
        # Verify all 3 are listed
        assert len(active) == 3
        
        # Verify details (order may vary)
        layers = {info["layer"] for info in active}
        assert layers == {5, 10, 15}
        
        methods = {info["method"] for info in active}
        assert methods == {"mean_difference", "caa", "linear_probe"}
        
        # Find specific entries and check alphas
        for info in active:
            if info["layer"] == 5:
                assert info["alpha"] == 1.0
                assert info["method"] == "mean_difference"
            elif info["layer"] == 10:
                assert info["alpha"] == 1.5
                assert info["method"] == "caa"
            elif info["layer"] == 15:
                assert info["alpha"] == 2.0
                assert info["method"] == "linear_probe"


class TestMultiVectorIntegration:
    """Integration tests for multi-vector steering."""
    
    def test_sequential_application_and_removal(self) -> None:
        """Test applying and removing vectors sequentially."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_type = "llama"
        mock_model.config.hidden_size = 100
        
        layer_modules = {i: Mock(spec=torch.nn.Module) for i in range(20)}
        
        steering_model = SteeringModel(model=mock_model)
        steering_model._layer_modules = layer_modules
        
        for layer_module in layer_modules.values():
            layer_module.register_forward_hook = Mock(return_value=Mock())
        
        # Apply 3 vectors
        vectors1 = [
            SteeringVector(
                tensor=torch.randn(100),
                layer=i,
                layer_name=f"model.layers.{i}",
                model_name="test-model",
            )
            for i in [5, 10, 15]
        ]
        
        steering_model.apply_multiple_steering(vectors=vectors1)
        assert len(steering_model.active_hooks) == 3
        
        # Remove all
        steering_model.remove_all_steering()
        assert len(steering_model.active_hooks) == 0
        
        # Apply different vectors
        vectors2 = [
            SteeringVector(
                tensor=torch.randn(100),
                layer=i,
                layer_name=f"model.layers.{i}",
                model_name="test-model",
            )
            for i in [2, 7, 12, 17]
        ]
        
        steering_model.apply_multiple_steering(vectors=vectors2)
        assert len(steering_model.active_hooks) == 4
        
        # Remove specific layers
        steering_model.remove_steering(layer=7)
        steering_model.remove_steering(layer=17)
        assert len(steering_model.active_hooks) == 2
        
        # Verify correct layers remain
        assert "layer_2" in steering_model.active_hooks
        assert "layer_12" in steering_model.active_hooks
        assert "layer_7" not in steering_model.active_hooks
        assert "layer_17" not in steering_model.active_hooks
