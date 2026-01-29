"""
LangChain integration for SteeringLLM.

This module provides a LangChain-compatible LLM wrapper that supports
steering vector application for use in chains and agents.
"""

from typing import Any, Dict, List, Optional, Union

from steering_llm.agents.base import SteeringAgent, SteeringConfig
from steering_llm.core.steering_model import SteeringModel
from steering_llm.core.steering_vector import SteeringVector

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback base class if LangChain not installed
    LLM = object
    LANGCHAIN_AVAILABLE = False


class LangChainSteeringLLM(SteeringAgent, LLM):
    """
    LangChain-compatible LLM with steering capabilities.
    
    This class wraps a SteeringModel to work seamlessly with LangChain
    chains, agents, and other components.
    
    Example:
        >>> from steering_llm.agents import LangChainSteeringLLM
        >>> from steering_llm import SteeringModel, Discovery
        >>> 
        >>> # Create base model
        >>> steering_model = SteeringModel.from_pretrained("gpt2")
        >>> 
        >>> # Create steering vector
        >>> vector = Discovery.mean_difference(
        ...     positive=["I love helping!"],
        ...     negative=["I hate this."],
        ...     model=steering_model,
        ...     layer=10
        ... )
        >>> 
        >>> # Create LangChain LLM
        >>> llm = LangChainSteeringLLM(
        ...     steering_model=steering_model,
        ...     vectors=[vector],
        ...     alpha=2.0
        ... )
        >>> 
        >>> # Use in LangChain
        >>> from langchain.chains import LLMChain
        >>> from langchain.prompts import PromptTemplate
        >>> 
        >>> prompt = PromptTemplate(
        ...     input_variables=["topic"],
        ...     template="Write about {topic}"
        ... )
        >>> chain = LLMChain(llm=llm, prompt=prompt)
        >>> result = chain.run(topic="kindness")
    """
    
    def __init__(
        self,
        steering_model: SteeringModel,
        vectors: Optional[List[SteeringVector]] = None,
        alpha: float = 1.0,
        config: Optional[SteeringConfig] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LangChain steering LLM.
        
        Args:
            steering_model: SteeringModel instance
            vectors: Steering vectors to apply
            alpha: Steering strength
            config: Full steering configuration (overrides vectors/alpha if provided)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional LangChain LLM parameters
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )
        
        # Initialize SteeringAgent
        if config is None:
            config = SteeringConfig(
                vectors=vectors or [],
                alpha=alpha
            )
        SteeringAgent.__init__(self, steering_model, config)
        
        # Initialize LangChain LLM
        LLM.__init__(self, **kwargs)
        
        # Generation parameters
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for LLM type."""
        return "steering_llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_name": self.steering_model.model.config.name_or_path,
            "alpha": self.config.alpha,
            "num_vectors": len(self.config.vectors),
            "max_length": self.max_length,
            "temperature": self.temperature,
        }
    
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
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
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
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional["CallbackManagerForLLMRun"] = None,
        **kwargs: Any,
    ) -> str:
        """
        LangChain LLM interface - required method.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: LangChain callback manager
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        # Apply stop sequences if provided
        gen_kwargs = kwargs.copy()
        if stop is not None:
            # Note: SteeringModel.generate may not support stop sequences directly
            # This is a limitation that could be addressed in future versions
            gen_kwargs["stop_sequences"] = stop
        
        result = self.generate(prompt, **gen_kwargs)
        
        # Handle stop sequences manually if needed
        if stop is not None and isinstance(result, str):
            for stop_seq in stop:
                if stop_seq in result:
                    result = result[:result.index(stop_seq)]
        
        return result
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional["CallbackManagerForLLMRun"] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async LangChain LLM interface - falls back to sync.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: LangChain callback manager
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        # For now, fall back to synchronous generation
        # Future: Implement true async generation
        return self._call(prompt, stop, run_manager, **kwargs)


def create_safety_agent(
    steering_model: SteeringModel,
    safety_vector: SteeringVector,
    alpha: float = 2.0,
    **kwargs: Any,
) -> LangChainSteeringLLM:
    """
    Create a safety-constrained LangChain agent.
    
    Args:
        steering_model: Base steering model
        safety_vector: Safety steering vector
        alpha: Steering strength for safety
        **kwargs: Additional LLM parameters
    
    Returns:
        Configured LangChain LLM with safety steering
    
    Example:
        >>> safety_vector = Discovery.mean_difference(
        ...     positive=["I'm helpful and safe"],
        ...     negative=["[toxic content]"],
        ...     model=model,
        ...     layer=15
        ... )
        >>> 
        >>> agent = create_safety_agent(
        ...     steering_model=model,
        ...     safety_vector=safety_vector,
        ...     alpha=2.0
        ... )
    """
    return LangChainSteeringLLM(
        steering_model=steering_model,
        vectors=[safety_vector],
        alpha=alpha,
        **kwargs,
    )


def create_domain_expert_agent(
    steering_model: SteeringModel,
    domain_vectors: List[SteeringVector],
    weights: Optional[List[float]] = None,
    **kwargs: Any,
) -> LangChainSteeringLLM:
    """
    Create a domain-expert LangChain agent with multiple steering vectors.
    
    Args:
        steering_model: Base steering model
        domain_vectors: Domain-specific steering vectors
        weights: Weights for vector composition
        **kwargs: Additional LLM parameters
    
    Returns:
        Configured LangChain LLM with domain expertise
    
    Example:
        >>> medical_vector = Discovery.caa(
        ...     positive=["medical terminology..."],
        ...     negative=["general text..."],
        ...     model=model,
        ...     layer=15
        ... )
        >>> 
        >>> formal_vector = Discovery.mean_difference(
        ...     positive=["Formal writing..."],
        ...     negative=["Casual writing..."],
        ...     model=model,
        ...     layer=20
        ... )
        >>> 
        >>> agent = create_domain_expert_agent(
        ...     steering_model=model,
        ...     domain_vectors=[medical_vector, formal_vector],
        ...     weights=[0.7, 0.3]
        ... )
    """
    config = SteeringConfig(
        vectors=domain_vectors,
        composition_method="weighted" if weights else "sum",
        weights=weights,
    )
    
    return LangChainSteeringLLM(
        steering_model=steering_model,
        config=config,
        **kwargs,
    )
