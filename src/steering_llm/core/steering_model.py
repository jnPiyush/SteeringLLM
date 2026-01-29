"""
SteeringModel wrapper for applying steering vectors at inference time.

This module provides the main interface for loading models and applying
steering vectors using PyTorch forward hooks.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from steering_llm.core.steering_vector import SteeringVector


class ActivationHook:
    """
    Manages PyTorch forward hooks for steering vector application.
    
    This class handles the registration and removal of forward hooks that
    intercept activations and add steering vectors during inference.
    """
    
    def __init__(
        self,
        module: torch.nn.Module,
        vector: SteeringVector,
        alpha: float = 1.0,
    ) -> None:
        """
        Initialize activation hook.
        
        Args:
            module: Target layer module to hook
            vector: Steering vector to apply
            alpha: Steering strength multiplier
        """
        self.module = module
        self.vector = vector
        self.alpha = alpha
        self.handle: Optional[Any] = None
        
    def register(self) -> None:
        """Register the forward hook on the target module."""
        if self.handle is not None:
            raise RuntimeError("Hook already registered. Remove before re-registering.")
        
        def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> Any:
            """
            Forward hook function that adds steering vector to activations.
            
            Args:
                module: The hooked module
                input: Module input (unused)
                output: Module output (activations)
            
            Returns:
                Modified output with steering applied
            """
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                # Most transformer layers return (hidden_states, ...)
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()
            
            # Move vector to same device as activations
            device = hidden_states.device
            dtype = hidden_states.dtype
            steering_tensor = self.vector.tensor.to(device=device, dtype=dtype)
            
            # Apply steering: output = original + (alpha * vector)
            # Broadcasting: [batch, seq_len, hidden_dim] + [hidden_dim]
            steered = hidden_states + (self.alpha * steering_tensor)
            
            # Return in original format
            if other_outputs:
                return (steered,) + other_outputs
            else:
                return steered
        
        self.handle = self.module.register_forward_hook(hook_fn)
    
    def remove(self) -> None:
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class SteeringModel:
    """
    Wrapper for HuggingFace models with steering capabilities.
    
    This class wraps a PreTrainedModel and provides methods to apply and
    remove steering vectors at inference time using PyTorch forward hooks.
    
    Attributes:
        model: The wrapped HuggingFace model
        tokenizer: Tokenizer for the model
        active_hooks: Dict of active hooks by layer name
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        """
        Initialize SteeringModel.
        
        Args:
            model: HuggingFace model to wrap
            tokenizer: Tokenizer (optional, auto-loaded if needed)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.active_hooks: Dict[str, ActivationHook] = {}
        self._layer_modules: Optional[Dict[int, torch.nn.Module]] = None
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "SteeringModel":
        """
        Load a HuggingFace model with steering capabilities.
        
        Args:
            model_name: HuggingFace model identifier
            tokenizer_name: Tokenizer name (defaults to model_name)
            **kwargs: Additional arguments passed to AutoModelForCausalLM
        
        Returns:
            SteeringModel instance
        
        Raises:
            ValueError: If model architecture is unsupported
            RuntimeError: If model loading fails
        
        Example:
            >>> model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")
            >>> # With quantization
            >>> model = SteeringModel.from_pretrained(
            ...     "mistralai/Mistral-7B-v0.1",
            ...     device_map="auto",
            ...     load_in_8bit=True
            ... )
        """
        # Load model
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e
        
        # Load tokenizer
        tokenizer_name = tokenizer_name or model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer {tokenizer_name}: {e}") from e
        
        # Validate architecture support
        model_type = getattr(hf_model.config, "model_type", None)
        supported_types = {"llama", "mistral", "gemma"}
        
        if model_type not in supported_types:
            raise ValueError(
                f"Unsupported model architecture: {model_type}. "
                f"Supported: {', '.join(supported_types)}"
            )
        
        return cls(model=hf_model, tokenizer=tokenizer)
    
    def _detect_layers(self) -> Dict[int, torch.nn.Module]:
        """
        Detect and cache layer modules from model architecture.
        
        Returns:
            Dict mapping layer index to module
        
        Raises:
            ValueError: If layers cannot be detected
        """
        if self._layer_modules is not None:
            return self._layer_modules
        
        # Try common patterns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama, Mistral, Gemma
            layers = self.model.model.layers
            self._layer_modules = {i: layer for i, layer in enumerate(layers)}
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 style
            layers = self.model.transformer.h
            self._layer_modules = {i: layer for i, layer in enumerate(layers)}
        else:
            raise ValueError(
                "Cannot detect layers. Supported architectures: "
                "Llama, Mistral, Gemma (model.layers.X), GPT-2 (transformer.h.X)"
            )
        
        return self._layer_modules
    
    def _get_layer_module(self, layer: int) -> torch.nn.Module:
        """
        Get module for specified layer index.
        
        Args:
            layer: Layer index
        
        Returns:
            Layer module
        
        Raises:
            ValueError: If layer index is invalid
        """
        layers = self._detect_layers()
        
        if layer not in layers:
            raise ValueError(
                f"Invalid layer index {layer}. "
                f"Model has {len(layers)} layers (0-{len(layers)-1})"
            )
        
        return layers[layer]
    
    def apply_steering(
        self,
        vector: SteeringVector,
        alpha: float = 1.0,
    ) -> None:
        """
        Apply a steering vector to the model.
        
        Args:
            vector: Steering vector to apply
            alpha: Steering strength multiplier (0.0 = no effect, 2.0 = double)
        
        Raises:
            ValueError: If vector is incompatible with model
            RuntimeError: If steering already applied to this layer
        
        Example:
            >>> model.apply_steering(safety_vector, alpha=1.5)
        """
        # Validate alpha
        if not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be numeric, got {type(alpha)}")
        
        # Check if steering already active on this layer
        layer_key = f"layer_{vector.layer}"
        if layer_key in self.active_hooks:
            raise RuntimeError(
                f"Steering already active on layer {vector.layer}. "
                "Remove existing steering before applying new vector."
            )
        
        # Get target layer module
        try:
            module = self._get_layer_module(vector.layer)
        except ValueError as e:
            raise ValueError(f"Cannot apply steering: {e}") from e
        
        # Validate vector dimensions
        # Get expected hidden dimension from model config
        hidden_dim = self.model.config.hidden_size
        if vector.tensor.shape[0] != hidden_dim:
            raise ValueError(
                f"Vector dimension mismatch: vector has {vector.tensor.shape[0]}, "
                f"but model expects {hidden_dim}"
            )
        
        # Create and register hook
        hook = ActivationHook(module=module, vector=vector, alpha=alpha)
        hook.register()
        
        # Store hook reference
        self.active_hooks[layer_key] = hook
    
    def remove_steering(self, layer: Optional[int] = None) -> None:
        """
        Remove steering vector(s) from the model.
        
        Args:
            layer: Specific layer to remove steering from (None = remove all)
        
        Example:
            >>> # Remove steering from specific layer
            >>> model.remove_steering(layer=15)
            >>> # Remove all steering
            >>> model.remove_steering()
        """
        if layer is not None:
            # Remove specific layer
            layer_key = f"layer_{layer}"
            if layer_key not in self.active_hooks:
                return  # No steering on this layer, nothing to do
            
            hook = self.active_hooks.pop(layer_key)
            hook.remove()
        else:
            # Remove all steering
            for hook in self.active_hooks.values():
                hook.remove()
            self.active_hooks.clear()
    
    def remove_all_steering(self) -> None:
        """
        Remove all active steering vectors.
        
        This is an alias for remove_steering() without arguments.
        
        Example:
            >>> model.remove_all_steering()
        """
        self.remove_steering()
    
    def list_active_steering(self) -> List[Dict[str, Any]]:
        """
        List all currently active steering vectors.
        
        Returns:
            List of dicts with layer, alpha, and vector info
        
        Example:
            >>> active = model.list_active_steering()
            >>> print(active)
            [{'layer': 15, 'alpha': 1.5, 'model_name': 'meta-llama/Llama-3.2-3B'}]
        """
        result = []
        
        for layer_key, hook in self.active_hooks.items():
            # Extract layer index from key "layer_N"
            layer_idx = int(layer_key.split("_")[1])
            
            info = {
                "layer": layer_idx,
                "alpha": hook.alpha,
                "model_name": hook.vector.model_name,
                "magnitude": hook.vector.magnitude,
                "method": hook.vector.method,
            }
            result.append(info)
        
        return result
    
    def generate_with_steering(
        self,
        prompt: Union[str, List[str]],
        vector: SteeringVector,
        alpha: float = 1.0,
        **generate_kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Generate text with temporary steering applied.
        
        Convenience method that applies steering, generates, then removes
        steering automatically. Safe to use even if generation fails.
        
        Args:
            prompt: Input prompt(s)
            vector: Steering vector to apply
            alpha: Steering strength
            **generate_kwargs: Arguments passed to model.generate()
        
        Returns:
            Generated text(s) (same type as prompt)
        
        Example:
            >>> output = model.generate_with_steering(
            ...     "Tell me about yourself",
            ...     vector=friendliness_vector,
            ...     alpha=2.0,
            ...     max_length=100,
            ...     temperature=0.7
            ... )
        """
        # Ensure tokenizer is available
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer required for generate_with_steering. "
                "Load tokenizer via from_pretrained() or set manually."
            )
        
        # Handle single vs batch prompts
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt
        
        try:
            # Apply steering
            self.apply_steering(vector=vector, alpha=alpha)
            
            # Tokenize inputs
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to model device
            device = next(iter(self.model.parameters())).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **generate_kwargs)
            
            # Decode outputs
            outputs = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            
            # Return in original format
            return outputs[0] if is_single else outputs
        
        finally:
            # Always remove steering, even if generation failed
            self.remove_steering(layer=vector.layer)
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to underlying model.
        
        This allows SteeringModel to be used as a drop-in replacement
        for the wrapped HuggingFace model.
        """
        # Avoid infinite recursion
        if name in {"model", "tokenizer", "active_hooks", "_layer_modules"}:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        return getattr(self.model, name)
    
    def __repr__(self) -> str:
        """String representation of SteeringModel."""
        model_name = getattr(self.model.config, "_name_or_path", "unknown")
        num_active = len(self.active_hooks)
        
        return (
            f"SteeringModel(model={model_name}, "
            f"active_steering={num_active})"
        )
