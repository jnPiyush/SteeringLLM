"""
LlamaIndex integration for SteeringLLM.

This module provides a LlamaIndex-compatible CustomLLM wrapper that supports
steering vectors for retrieval-augmented generation (RAG) applications.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from steering_llm.agents.base import SteeringAgent, SteeringConfig
from steering_llm.core.steering_model import SteeringModel
from steering_llm.core.steering_vector import SteeringVector

try:
    from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    # Fallback base class if LlamaIndex not installed
    CustomLLM = object
    CompletionResponse = None
    LLMMetadata = None
    LLAMAINDEX_AVAILABLE = False
    
    # Dummy decorator when LlamaIndex not available
    def llm_completion_callback():
        def decorator(func):
            return func
        return decorator


class LlamaIndexSteeringLLM(SteeringAgent, CustomLLM):
    """
    LlamaIndex-compatible CustomLLM with steering capabilities.
    
    This class wraps a SteeringModel for use in LlamaIndex RAG pipelines,
    query engines, and agents.
    
    Example:
        >>> from steering_llm.agents import LlamaIndexSteeringLLM
        >>> from steering_llm import SteeringModel, Discovery
        >>> from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
        >>> 
        >>> # Create base model
        >>> steering_model = SteeringModel.from_pretrained("gpt2")
        >>> 
        >>> # Create steering vector for domain expertise
        >>> vector = Discovery.mean_difference(
        ...     positive=["Technical documentation..."],
        ...     negative=["Casual conversation..."],
        ...     model=steering_model,
        ...     layer=10
        ... )
        >>> 
        >>> # Create LlamaIndex LLM
        >>> llm = LlamaIndexSteeringLLM(
        ...     steering_model=steering_model,
        ...     vectors=[vector],
        ...     alpha=2.0
        ... )
        >>> 
        >>> # Use in RAG pipeline
        >>> documents = SimpleDirectoryReader("./docs").load_data()
        >>> index = VectorStoreIndex.from_documents(documents)
        >>> query_engine = index.as_query_engine(llm=llm)
        >>> response = query_engine.query("What is this about?")
    """
    
    def __init__(
        self,
        steering_model: SteeringModel,
        vectors: Optional[List[SteeringVector]] = None,
        alpha: float = 1.0,
        config: Optional[SteeringConfig] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        context_window: int = 4096,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LlamaIndex steering LLM.
        
        Args:
            steering_model: SteeringModel instance
            vectors: Steering vectors to apply
            alpha: Steering strength
            config: Full steering configuration
            max_tokens: Maximum generation tokens
            temperature: Sampling temperature
            context_window: Context window size
            **kwargs: Additional parameters
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex not installed. Install with: pip install llama-index"
            )
        
        # Initialize SteeringAgent
        if config is None:
            config = SteeringConfig(
                vectors=vectors or [],
                alpha=alpha
            )
        SteeringAgent.__init__(self, steering_model, config)
        
        # Initialize CustomLLM
        CustomLLM.__init__(self, **kwargs)
        
        # LLM parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_window = context_window
    
    @property
    def metadata(self) -> "LLMMetadata":
        """Return LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.steering_model.model.config.name_or_path,
            is_chat_model=False,
        )
    
    def apply_steering(
        self,
        vectors: Optional[List[SteeringVector]] = None,
        config: Optional[SteeringConfig] = None,
    ) -> None:
        """
        Apply steering vectors to the model.
        
        Args:
            vectors: Steering vectors to apply (uses config.vectors if None)
            config: Override steering configuration (optional)
        """
        if self._steering_active:
            raise RuntimeError("Steering already active. Remove before re-applying.")
        
        if config is not None:
            self.config = config
        
        vectors_to_apply = vectors or self.config.vectors
        
        if not vectors_to_apply:
            raise ValueError("No steering vectors provided")
        
        # Apply each vector
        for vector in vectors_to_apply:
            alpha = self.config.get_alpha(vector.layer)
            self.steering_model.apply_steering(vector, alpha=alpha)
        
        self._steering_active = True
    
    def remove_steering(self) -> None:
        """Remove all active steering."""
        if not self._steering_active:
            raise RuntimeError("No active steering to remove")
        
        self.steering_model.remove_all_steering()
        self._steering_active = False
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Generate text with steering applied.
        
        Args:
            prompt: Input prompt(s)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text(s)
        """
        # Merge generation kwargs
        gen_kwargs = {
            "max_length": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        
        # Ensure steering is applied
        was_active = self._steering_active
        if not was_active and self.config.vectors:
            self.apply_steering()
        
        try:
            # Generate with steering
            if isinstance(prompt, list):
                return [
                    self.steering_model.generate(p, **gen_kwargs)
                    for p in prompt
                ]
            else:
                return self.steering_model.generate(prompt, **gen_kwargs)
        finally:
            # Remove steering if we applied it temporarily
            if not was_active and self._steering_active:
                self.remove_steering()
    
    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> "CompletionResponse":
        """
        LlamaIndex complete method - required interface.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            CompletionResponse with generated text
        """
        # Generate text
        response_text = self.generate(prompt, **kwargs)
        
        # Return CompletionResponse
        return CompletionResponse(text=response_text)
    
    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        **kwargs: Any,
    ):
        """
        Streaming completion (not yet implemented - falls back to complete).
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        
        Yields:
            CompletionResponse chunks
        """
        # For now, fall back to non-streaming
        # Future: Implement true streaming
        response = self.complete(prompt, **kwargs)
        yield response


def create_rag_steering_llm(
    steering_model: SteeringModel,
    domain_vector: SteeringVector,
    alpha: float = 1.5,
    **kwargs: Any,
) -> LlamaIndexSteeringLLM:
    """
    Create a domain-adapted LLM for RAG applications.
    
    Args:
        steering_model: Base steering model
        domain_vector: Domain-specific steering vector
        alpha: Steering strength
        **kwargs: Additional LLM parameters
    
    Returns:
        Configured LlamaIndex LLM for RAG
    
    Example:
        >>> # Create domain vector for medical Q&A
        >>> medical_vector = Discovery.caa(
        ...     positive=["Medical terminology and explanations..."],
        ...     negative=["General conversation..."],
        ...     model=model,
        ...     layer=15
        ... )
        >>> 
        >>> # Create RAG LLM
        >>> llm = create_rag_steering_llm(
        ...     steering_model=model,
        ...     domain_vector=medical_vector,
        ...     alpha=1.5
        ... )
        >>> 
        >>> # Use in query engine
        >>> query_engine = index.as_query_engine(llm=llm)
    """
    return LlamaIndexSteeringLLM(
        steering_model=steering_model,
        vectors=[domain_vector],
        alpha=alpha,
        **kwargs,
    )


def create_multi_vector_rag_llm(
    steering_model: SteeringModel,
    vectors: List[SteeringVector],
    weights: Optional[List[float]] = None,
    composition_method: str = "sum",
    **kwargs: Any,
) -> LlamaIndexSteeringLLM:
    """
    Create a multi-vector steered LLM for complex RAG scenarios.
    
    Args:
        steering_model: Base steering model
        vectors: Multiple steering vectors (e.g., domain + style + safety)
        weights: Weights for weighted composition
        composition_method: Composition method ("sum", "weighted", "cascade")
        **kwargs: Additional LLM parameters
    
    Returns:
        Configured LlamaIndex LLM with multi-vector steering
    
    Example:
        >>> # Create multiple steering vectors
        >>> domain_vector = Discovery.caa(...)
        >>> style_vector = Discovery.mean_difference(...)
        >>> safety_vector = Discovery.linear_probe(...)
        >>> 
        >>> # Create multi-vector RAG LLM
        >>> llm = create_multi_vector_rag_llm(
        ...     steering_model=model,
        ...     vectors=[domain_vector, style_vector, safety_vector],
        ...     weights=[0.5, 0.3, 0.2],
        ...     composition_method="weighted"
        ... )
    """
    config = SteeringConfig(
        vectors=vectors,
        composition_method=composition_method,
        weights=weights,
    )
    
    return LlamaIndexSteeringLLM(
        steering_model=steering_model,
        config=config,
        **kwargs,
    )
