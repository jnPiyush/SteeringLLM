"""
SteeringModel wrapper for applying steering vectors at inference time.

This module provides the main interface for loading models and applying
steering vectors using PyTorch forward hooks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from steering_llm.core.steering_vector import SteeringVector
from steering_llm.exceptions import (
    IncompatibleVectorError,
    InvalidLayerError,
    ModelLoadError,
    SteeringActiveError,
    UnsupportedArchitectureError,
)


# Model architecture registry: maps model_type to (parent_module, layers_attr)
# This allows flexible layer detection across different architectures
_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Llama family (Llama 2, Llama 3, Code Llama)
    "llama": ("model", "layers"),
    # Mistral family (Mistral, Mixtral)
    "mistral": ("model", "layers"),
    # Gemma family (Gemma 1, Gemma 2)
    "gemma": ("model", "layers"),
    "gemma2": ("model", "layers"),
    # Phi family (Phi-2, Phi-3)
    "phi": ("model", "layers"),
    "phi3": ("model", "layers"),
    # Qwen family (Qwen 1.5, Qwen 2, Qwen 2.5)
    "qwen2": ("model", "layers"),
    "qwen2_moe": ("model", "layers"),
    # GPT family (GPT-2, GPT-Neo, GPT-J)
    "gpt2": ("transformer", "h"),
    "gpt_neo": ("transformer", "h"),
    "gpt_neox": ("gpt_neox", "layers"),
    "gptj": ("transformer", "h"),
    # OPT family
    "opt": ("model.decoder", "layers"),
    # BLOOM
    "bloom": ("transformer", "h"),
    # Falcon
    "falcon": ("transformer", "h"),
}


def get_supported_architectures() -> List[str]:
    """Return list of supported model architectures."""
    return sorted(_MODEL_REGISTRY.keys())


def register_architecture(model_type: str, parent_path: str, layers_attr: str) -> None:
    """
    Register a custom model architecture for steering support.
    
    Args:
        model_type: The model_type from model.config (e.g., "custom_llm")
        parent_path: Dot-separated path to parent module (e.g., "model.encoder")
        layers_attr: Attribute name containing layers list (e.g., "layers")
    
    Example:
        >>> register_architecture("my_custom_model", "backbone", "blocks")
    """
    _MODEL_REGISTRY[model_type] = (parent_path, layers_attr)


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
            raise SteeringActiveError(self.vector.layer)
        
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
            is_tuple = isinstance(output, tuple)
            if is_tuple:
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
            
            # Return in the SAME format as the original output
            if is_tuple:
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
    
    @property
    def device(self) -> torch.device:
        """
        Get the primary device of the model.
        
        Returns:
            torch.device where model parameters are located
        """
        return next(iter(self.model.parameters())).device
    
    @property
    def num_layers(self) -> int:
        """
        Get the number of layers in the model.
        
        Returns:
            Number of transformer layers
        """
        return len(self._detect_layers())
    
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
            raise ModelLoadError(f"Failed to load model {model_name}: {e}") from e
        
        # Load tokenizer
        tokenizer_name = tokenizer_name or model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            raise ModelLoadError(f"Failed to load tokenizer {tokenizer_name}: {e}") from e
        
        # Validate architecture support
        model_type = getattr(hf_model.config, "model_type", None)
        
        if model_type not in _MODEL_REGISTRY:
            raise UnsupportedArchitectureError(model_type, list(_MODEL_REGISTRY.keys()))
        
        return cls(model=hf_model, tokenizer=tokenizer)
    
    def _detect_layers(self) -> Dict[int, torch.nn.Module]:
        """
        Detect and cache layer modules from model architecture.
        
        Uses _MODEL_REGISTRY to find layers based on model_type. This allows
        support for diverse architectures without hardcoding patterns.
        
        Returns:
            Dict mapping layer index to module
        
        Raises:
            UnsupportedArchitectureError: If model architecture is not supported
        """
        if self._layer_modules is not None:
            return self._layer_modules
        
        # Get model type from config
        model_type = getattr(self.model.config, "model_type", None)
        
        if model_type is None:
            raise UnsupportedArchitectureError("unknown", list(_MODEL_REGISTRY.keys()))
        
        # Look up architecture pattern in registry
        if model_type not in _MODEL_REGISTRY:
            raise UnsupportedArchitectureError(model_type, list(_MODEL_REGISTRY.keys()))
        
        parent_path, layers_attr = _MODEL_REGISTRY[model_type]
        
        # Navigate to parent module
        try:
            parent_module = self.model
            for attr in parent_path.split("."):
                parent_module = getattr(parent_module, attr)
        except AttributeError as e:
            raise UnsupportedArchitectureError(
                model_type, list(_MODEL_REGISTRY.keys())
            ) from e
        
        # Get layers from parent module
        if not hasattr(parent_module, layers_attr):
            raise UnsupportedArchitectureError(model_type, list(_MODEL_REGISTRY.keys()))
        
        layers = getattr(parent_module, layers_attr)
        self._layer_modules = {i: layer for i, layer in enumerate(layers)}
        
        if not self._layer_modules:
            raise UnsupportedArchitectureError(model_type, list(_MODEL_REGISTRY.keys()))
        
        return self._layer_modules
    
    def _get_layer_module(self, layer: int) -> torch.nn.Module:
        """
        Get module for specified layer index.
        
        Args:
            layer: Layer index
        
        Returns:
            Layer module
        
        Raises:
            InvalidLayerError: If layer index is invalid
        """
        layers = self._detect_layers()
        
        if layer not in layers:
            raise InvalidLayerError(layer, len(layers))
        
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
            IncompatibleVectorError: If vector is incompatible with model
            SteeringActiveError: If steering already applied to this layer
        
        Example:
            >>> model.apply_steering(safety_vector, alpha=1.5)
        """
        # Validate alpha
        if not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be numeric, got {type(alpha)}")
        
        # Check if steering already active on this layer
        layer_key = f"layer_{vector.layer}"
        if layer_key in self.active_hooks:
            raise SteeringActiveError(vector.layer)
        
        # Get target layer module
        module = self._get_layer_module(vector.layer)
        
        # Validate vector dimensions
        hidden_dim = self.model.config.hidden_size
        if vector.tensor.shape[0] != hidden_dim:
            raise IncompatibleVectorError(vector.tensor.shape[0], hidden_dim)
        
        # Create and register hook
        hook = ActivationHook(module=module, vector=vector, alpha=alpha)
        hook.register()
        
        # Store hook reference
        self.active_hooks[layer_key] = hook
    
    def apply_multiple_steering(
        self,
        vectors: List[SteeringVector],
        alphas: Optional[List[float]] = None,
    ) -> None:
        """
        Apply multiple steering vectors simultaneously.
        
        This method allows applying vectors to different layers or multiple
        vectors to the same layer (after composition).
        
        Args:
            vectors: List of SteeringVector instances to apply
            alphas: Optional list of alpha values (default: 1.0 for each)
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If conflicts with existing steering
        
        Example:
            >>> # Apply politeness and conciseness steering
            >>> model.apply_multiple_steering(
            ...     vectors=[politeness_vec, conciseness_vec],
            ...     alphas=[1.2, 0.8]
            ... )
        """
        if not vectors:
            raise ValueError("vectors list cannot be empty")
        
        if alphas is None:
            alphas = [1.0] * len(vectors)
        
        if len(alphas) != len(vectors):
            raise ValueError(
                f"Number of alphas ({len(alphas)}) must match "
                f"number of vectors ({len(vectors)})"
            )
        
        # Check for conflicts with existing hooks
        for vector in vectors:
            layer_key = f"layer_{vector.layer}"
            if layer_key in self.active_hooks:
                raise SteeringActiveError(vector.layer)
        
        # Apply all vectors
        for vector, alpha in zip(vectors, alphas):
            self.apply_steering(vector=vector, alpha=alpha)
    
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
            
            # Strip input prompt tokens so only new text is returned
            input_length = inputs["input_ids"].shape[-1]
            new_ids = output_ids[:, input_length:]
            
            # Decode outputs
            outputs = self.tokenizer.batch_decode(
                new_ids,
                skip_special_tokens=True,
            )
            
            # Return in original format
            return outputs[0] if is_single else outputs
        
        finally:
            # Always remove steering, even if generation failed
            self.remove_steering(layer=vector.layer)

    def generate(
        self,
        prompt: Union[str, List[str]],
        **generate_kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s) using the underlying HF model.

        This is a convenience wrapper that handles tokenization and decoding,
        making SteeringModel usable with text prompts directly.

        Args:
            prompt: Input prompt(s)
            **generate_kwargs: Arguments passed to model.generate()

        Returns:
            Generated text(s) (same type as prompt)
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer required for generate(). "
                "Load tokenizer via from_pretrained() or set manually."
            )

        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )

        device = next(iter(self.model.parameters())).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        # Strip input prompt tokens so only new text is returned
        input_length = inputs["input_ids"].shape[-1]
        new_ids = output_ids[:, input_length:]

        outputs = self.tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
        )

        return outputs[0] if is_single else outputs
    
    # Explicit delegation methods for common model attributes
    # This is preferred over __getattr__ for better IDE support and safety
    
    @property
    def config(self) -> Any:
        """Access the model's configuration."""
        return self.model.config
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the model's dtype."""
        return self.model.dtype
    
    def eval(self) -> "SteeringModel":
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self, mode: bool = True) -> "SteeringModel":
        """Set model to training mode."""
        self.model.train(mode)
        return self
    
    def to(self, device: Union[str, torch.device]) -> "SteeringModel":
        """Move model to device."""
        self.model.to(device)
        return self
    
    def __repr__(self) -> str:
        """String representation of SteeringModel."""
        model_name = getattr(self.model.config, "_name_or_path", "unknown")
        num_active = len(self.active_hooks)
        
        return (
            f"SteeringModel(model={model_name}, "
            f"active_steering={num_active})"
        )
