"""
Discovery module for creating steering vectors from contrast datasets.

This module provides methods for discovering steering vectors by analyzing
differences in model activations between positive and negative examples.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

from steering_llm.core.steering_vector import SteeringVector
from steering_llm.exceptions import (
    ActivationExtractionError,
    EmptyDatasetError,
    LayerDetectionError,
)

logger = logging.getLogger(__name__)

# Constants for validation and defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 128
DEFAULT_PROBE_C = 1.0  # Regularization strength for linear probe
DEFAULT_PROBE_MAX_ITER = 1000
MAX_BATCH_SIZE = 128  # Guard against OOM
MAX_SAMPLES = 100000  # Guard against excessive processing


@dataclass
class DiscoveryResult:
    """
    Result from discovery methods with optional metrics.
    
    This provides a consistent return type across all discovery methods.
    
    Attributes:
        vector: The discovered steering vector
        metrics: Optional dict of method-specific metrics (e.g., accuracy)
    """
    vector: SteeringVector
    metrics: Optional[Dict[str, Any]] = None


class Discovery:
    """
    Methods for discovering steering vectors from contrast datasets.
    
    This class provides static methods for creating steering vectors by
    analyzing model activations on positive vs negative examples.
    """
    
    @staticmethod
    def mean_difference(
        positive: List[str],
        negative: List[str],
        model: PreTrainedModel,
        layer: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: Optional[Union[str, torch.device]] = None,
    ) -> DiscoveryResult:
        """
        Create steering vector using mean difference method.
        
        Computes: mean(positive_activations) - mean(negative_activations)
        
        This method extracts activations from the specified layer for both
        positive and negative examples, then computes the difference of their
        means to create a steering vector.
        
        Args:
            positive: List of texts exhibiting desired behavior
            negative: List of texts exhibiting undesired behavior
            model: HuggingFace model to extract activations from
            layer: Target layer index (0-based)
            tokenizer: Tokenizer (auto-detected if None)
            model_name: Model identifier (auto-detected if None)
            batch_size: Batch size for processing examples
            max_length: Maximum sequence length
            device: Device to use (auto-detected if None)
        
        Returns:
            DiscoveryResult with SteeringVector and metrics
        
        Raises:
            EmptyDatasetError: If positive or negative list is empty
            ValueError: If layer is invalid
            ActivationExtractionError: If extraction fails
        
        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> result = Discovery.mean_difference(
            ...     positive=["I love helping!", "You're amazing!"],
            ...     negative=["I hate this.", "You're terrible."],
            ...     model=model,
            ...     layer=5
            ... )
            >>> vector = result.vector
        """
        # Validate inputs
        if not positive:
            raise EmptyDatasetError("positive")
        if not negative:
            raise EmptyDatasetError("negative")
        
        # Guard against excessive data
        if len(positive) > MAX_SAMPLES or len(negative) > MAX_SAMPLES:
            logger.warning(
                "Large dataset detected (%d/%d samples). Consider reducing size.",
                len(positive), len(negative)
            )
        
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(f"layer must be non-negative integer, got {layer}")
        
        batch_size = min(batch_size, MAX_BATCH_SIZE)
        
        # Auto-detect tokenizer if not provided
        if tokenizer is None:
            if model_name is None:
                model_name = getattr(model.config, "_name_or_path", "unknown")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Auto-detect device if not provided
        if device is None:
            device = next(model.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Auto-detect model name if not provided
        if model_name is None:
            model_name = getattr(model.config, "_name_or_path", "unknown")
        
        # Detect layer name
        layer_name = Discovery._detect_layer_name(model, layer)
        
        # Extract activations for positive examples
        logger.info("Extracting activations for %d positive examples...", len(positive))
        pos_activations = Discovery._extract_activations(
            texts=positive,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        # Extract activations for negative examples
        logger.info("Extracting activations for %d negative examples...", len(negative))
        neg_activations = Discovery._extract_activations(
            texts=negative,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        # Compute mean difference
        mean_pos = torch.mean(pos_activations, dim=0)
        mean_neg = torch.mean(neg_activations, dim=0)
        steering_vector = mean_pos - mean_neg
        
        # Create SteeringVector instance
        vector = SteeringVector(
            tensor=steering_vector.cpu(),
            layer=layer,
            layer_name=layer_name,
            model_name=model_name,
            method="mean_difference",
            metadata={
                "positive_samples": len(positive),
                "negative_samples": len(negative),
                "batch_size": batch_size,
                "max_length": max_length,
            },
        )
        
        return DiscoveryResult(vector=vector, metrics=None)
    
    @staticmethod
    def _detect_layer_name(model: PreTrainedModel, layer: int) -> str:
        """
        Detect layer name from model architecture.
        
        Args:
            model: HuggingFace model
            layer: Layer index
        
        Returns:
            Layer name (e.g., "model.layers.15")
        
        Raises:
            LayerDetectionError: If layer cannot be detected
        """
        # Common patterns for different architectures
        patterns = [
            ("model.layers", "model", "layers"),          # Llama, Mistral, Gemma, Phi, Qwen
            ("transformer.h", "transformer", "h"),        # GPT-2, GPT-J, BLOOM, Falcon
            ("gpt_neox.layers", "gpt_neox", "layers"),    # GPT-NeoX
            ("model.decoder.layers", "model.decoder", "layers"),  # OPT
        ]

        for prefix, parent_path, layers_attr in patterns:
            try:
                parent = model
                for attr in parent_path.split("."):
                    parent = getattr(parent, attr)
                if hasattr(parent, layers_attr):
                    layers = getattr(parent, layers_attr)
                    if layer < len(layers):
                        return f"{prefix}.{layer}"
            except AttributeError:
                continue

        raise LayerDetectionError(
            f"Cannot detect layer name for layer {layer}. "
            "Supported architectures: Llama, Mistral, Gemma, Phi, Qwen, GPT-2, "
            "GPT-J, GPT-NeoX, BLOOM, Falcon, OPT"
        )
    
    @staticmethod
    def _extract_activations(
        texts: List[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer: int,
        layer_name: str,
        batch_size: int,
        max_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Extract activations from specified layer for given texts.
        
        Args:
            texts: Input texts
            model: HuggingFace model
            tokenizer: Tokenizer
            layer: Layer index
            layer_name: Layer name
            batch_size: Batch size
            max_length: Max sequence length
            device: Device to use
        
        Returns:
            Stacked activations tensor [num_texts, hidden_dim]
        
        Raises:
            RuntimeError: If extraction fails
        """
        activations_list = []
        
        # Get target layer module
        target_module = Discovery._get_layer_module(model, layer_name)
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Storage for activations
            batch_activations = []
            
            # Define hook function
            def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
                # Extract activation from common HF output formats
                if isinstance(output, tuple):
                    activation = output[0]
                elif hasattr(output, "hidden_states"):
                    activation = output.hidden_states
                else:
                    activation = output

                # Take mean over sequence dimension [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
                activation_mean = activation.mean(dim=1)
                batch_activations.append(activation_mean.detach())
            
            # Register hook
            handle = target_module.register_forward_hook(hook_fn)
            
            try:
                # Forward pass
                with torch.no_grad():
                    model(**inputs)
                
                # Collect activations
                if not batch_activations:
                    raise RuntimeError(f"No activations captured from layer {layer_name}")
                
                activations_list.append(batch_activations[0])
            
            finally:
                # Remove hook
                handle.remove()
        
        # Stack all activations
        all_activations = torch.cat(activations_list, dim=0)
        return all_activations
    
    @staticmethod
    def _get_layer_module(model: PreTrainedModel, layer_name: str) -> torch.nn.Module:
        """
        Get layer module by name.
        
        Args:
            model: HuggingFace model
            layer_name: Layer name (e.g., "model.layers.15")
        
        Returns:
            Layer module
        
        Raises:
            ValueError: If layer not found
        """
        parts = layer_name.split(".")
        module = model
        
        for part in parts:
            if not hasattr(module, part):
                # Try integer indexing
                try:
                    idx = int(part)
                    module = module[idx]
                except (ValueError, TypeError, IndexError):
                    raise ValueError(f"Layer not found: {layer_name}")
            else:
                module = getattr(module, part)
        
        return module

    @staticmethod
    def caa(
        positive: List[str],
        negative: List[str],
        model: PreTrainedModel,
        layer: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: Optional[Union[str, torch.device]] = None,
        num_pairs: Optional[int] = None,
    ) -> DiscoveryResult:
        """
        Create steering vector using Contrastive Activation Addition (CAA).
        
        CAA computes layer-wise contrasts between positive and negative examples,
        then averages these contrasts. This method typically produces stronger
        steering vectors than simple mean difference.
        
        Based on: "Steering Llama 2 via Contrastive Activation Addition" 
        (Turner et al., 2023)
        
        Args:
            positive: List of texts exhibiting desired behavior
            negative: List of texts exhibiting undesired behavior
            model: HuggingFace model to extract activations from
            layer: Target layer index (0-based)
            tokenizer: Tokenizer (auto-detected if None)
            model_name: Model identifier (auto-detected if None)
            batch_size: Batch size for processing examples
            max_length: Maximum sequence length
            device: Device to use (auto-detected if None)
            num_pairs: Number of contrast pairs to use (None = use all)
        
        Returns:
            DiscoveryResult with SteeringVector and metrics
        
        Raises:
            EmptyDatasetError: If positive or negative list is empty
            ValueError: If sizes don't match
            ActivationExtractionError: If extraction fails
        
        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> result = Discovery.caa(
            ...     positive=["I love helping!", "You're amazing!"],
            ...     negative=["I hate this.", "You're terrible."],
            ...     model=model,
            ...     layer=5
            ... )
        """
        # Validate inputs
        if not positive:
            raise EmptyDatasetError("positive")
        if not negative:
            raise EmptyDatasetError("negative")
        
        if len(positive) != len(negative):
            raise ValueError(
                f"CAA requires equal number of positive and negative examples. "
                f"Got {len(positive)} positive and {len(negative)} negative."
            )
        
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(f"layer must be non-negative integer, got {layer}")
        
        # Auto-detect components
        if tokenizer is None:
            if model_name is None:
                model_name = getattr(model.config, "_name_or_path", "unknown")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        if device is None:
            device = next(model.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        
        if model_name is None:
            model_name = getattr(model.config, "_name_or_path", "unknown")
        
        # Detect layer name
        layer_name = Discovery._detect_layer_name(model, layer)
        
        # Limit number of pairs if specified
        if num_pairs is not None:
            num_pairs = min(num_pairs, len(positive))
            positive = positive[:num_pairs]
            negative = negative[:num_pairs]
        
        # Extract activations for all examples
        logger.info("Extracting activations for %d contrast pairs...", len(positive))
        pos_activations = Discovery._extract_activations(
            texts=positive,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        neg_activations = Discovery._extract_activations(
            texts=negative,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        # Compute pairwise contrasts: pos[i] - neg[i] for each pair
        contrasts = pos_activations - neg_activations
        
        # Average all contrasts to get final steering vector
        steering_vector = torch.mean(contrasts, dim=0)
        
        # Create SteeringVector instance
        vector = SteeringVector(
            tensor=steering_vector.cpu(),
            layer=layer,
            layer_name=layer_name,
            model_name=model_name,
            method="caa",
            metadata={
                "contrast_pairs": len(positive),
                "batch_size": batch_size,
                "max_length": max_length,
            },
        )
        
        return DiscoveryResult(vector=vector, metrics=None)
    
    @staticmethod
    def linear_probe(
        positive: List[str],
        negative: List[str],
        model: PreTrainedModel,
        layer: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: Optional[Union[str, torch.device]] = None,
        C: float = DEFAULT_PROBE_C,
        max_iter: int = DEFAULT_PROBE_MAX_ITER,
        normalize: bool = True,
    ) -> DiscoveryResult:
        """
        Create steering vector using linear probing.
        
        Trains a logistic regression classifier on activations to distinguish
        positive from negative examples. The classifier weights are used as the
        steering vector, providing an interpretable feature extraction method.
        
        Args:
            positive: List of texts exhibiting desired behavior
            negative: List of texts exhibiting undesired behavior
            model: HuggingFace model to extract activations from
            layer: Target layer index (0-based)
            tokenizer: Tokenizer (auto-detected if None)
            model_name: Model identifier (auto-detected if None)
            batch_size: Batch size for processing examples
            max_length: Maximum sequence length
            device: Device to use (auto-detected if None)
            C: Inverse of regularization strength (higher = less regularization)
            max_iter: Maximum iterations for solver
            normalize: Whether to normalize activations before training
        
        Returns:
            DiscoveryResult with SteeringVector and metrics (including train_accuracy)
        
        Raises:
            EmptyDatasetError: If positive or negative list is empty
            ActivationExtractionError: If training fails
        
        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> result = Discovery.linear_probe(
            ...     positive=["I love helping!", "You're amazing!"],
            ...     negative=["I hate this.", "You're terrible."],
            ...     model=model,
            ...     layer=5
            ... )
            >>> print(f"Probe accuracy: {result.metrics['train_accuracy']:.2%}")
        """
        # Validate inputs
        if not positive:
            raise EmptyDatasetError("positive")
        if not negative:
            raise EmptyDatasetError("negative")
        
        if not isinstance(layer, int) or layer < 0:
            raise ValueError(f"layer must be non-negative integer, got {layer}")
        
        # Auto-detect components
        if tokenizer is None:
            if model_name is None:
                model_name = getattr(model.config, "_name_or_path", "unknown")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        if device is None:
            device = next(model.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        
        if model_name is None:
            model_name = getattr(model.config, "_name_or_path", "unknown")
        
        # Detect layer name
        layer_name = Discovery._detect_layer_name(model, layer)
        
        # Extract activations
        logger.info("Extracting activations for %d positive examples...", len(positive))
        pos_activations = Discovery._extract_activations(
            texts=positive,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        logger.info("Extracting activations for %d negative examples...", len(negative))
        neg_activations = Discovery._extract_activations(
            texts=negative,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            layer_name=layer_name,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        
        # Prepare training data
        X = torch.cat([pos_activations, neg_activations], dim=0).cpu().numpy()
        y = np.array([1] * len(positive) + [0] * len(negative))
        
        # Normalize if requested
        scaler = None
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Train logistic regression probe
        logger.info("Training linear probe...")
        probe = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            solver="lbfgs",
        )
        
        try:
            probe.fit(X, y)
        except Exception as e:
            raise ActivationExtractionError(f"Linear probe training failed: {e}") from e
        
        # Extract probe weights as steering vector
        probe_weights = torch.from_numpy(probe.coef_[0]).float()
        
        # Compute training accuracy
        train_accuracy = float(probe.score(X, y))
        logger.info("Linear probe accuracy: %.2f%%", train_accuracy * 100)
        
        # Create metrics dict
        metrics = {
            "train_accuracy": train_accuracy,
            "positive_samples": len(positive),
            "negative_samples": len(negative),
            "C": C,
            "normalized": normalize,
        }
        
        # Create SteeringVector instance
        vector = SteeringVector(
            tensor=probe_weights.cpu(),
            layer=layer,
            layer_name=layer_name,
            model_name=model_name,
            method="linear_probe",
            metadata={
                **metrics,
                "batch_size": batch_size,
                "max_length": max_length,
            },
        )
        
        return DiscoveryResult(vector=vector, metrics=metrics)

