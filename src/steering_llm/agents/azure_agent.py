"""
Microsoft Agent Framework integration for SteeringLLM.

This module provides integration with Microsoft's agent-framework SDK
for Azure AI Foundry deployment and enterprise features.
"""

from typing import Any, Dict, List, Optional, Union

from steering_llm.agents.base import SteeringAgent, SteeringConfig
from steering_llm.core.steering_model import SteeringModel
from steering_llm.core.steering_vector import SteeringVector

try:
    # Try importing agent-framework (Microsoft's Agent Framework)
    from agent_framework import Agent
    from agent_framework.openai import OpenAIChatClient
    AGENT_FRAMEWORK_AVAILABLE = True
except ImportError:
    AGENT_FRAMEWORK_AVAILABLE = False


class AzureSteeringAgent(SteeringAgent):
    """
    Microsoft Agent Framework integration with steering capabilities.
    
    This class wraps a SteeringModel for use with Microsoft's agent-framework
    SDK, enabling Azure AI Foundry deployment, tracing, and monitoring.
    
    Example:
        >>> from steering_llm.agents import AzureSteeringAgent
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
        >>> # Create Azure agent
        >>> agent = AzureSteeringAgent(
        ...     steering_model=steering_model,
        ...     agent_name="helpful_assistant",
        ...     vectors=[vector],
        ...     alpha=2.0
        ... )
        >>> 
        >>> # Generate with steering
        >>> response = agent.generate("How can I help you today?")
    """
    
    def __init__(
        self,
        steering_model: SteeringModel,
        agent_name: str = "steering_agent",
        vectors: Optional[List[SteeringVector]] = None,
        alpha: float = 1.0,
        config: Optional[SteeringConfig] = None,
        enable_tracing: bool = False,
        tracing_config: Optional[Dict[str, Any]] = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Azure steering agent.
        
        Args:
            steering_model: SteeringModel instance
            agent_name: Name for the agent
            vectors: Steering vectors to apply
            alpha: Steering strength
            config: Full steering configuration
            enable_tracing: Enable Azure AI tracing
            tracing_config: Tracing configuration
            max_tokens: Maximum generation tokens
            temperature: Sampling temperature
            **kwargs: Additional agent parameters
        """
        if not AGENT_FRAMEWORK_AVAILABLE:
            raise ImportError(
                "Microsoft agent-framework not installed. "
                "Install with: pip install agent-framework"
            )
        
        # Initialize SteeringAgent
        if config is None:
            config = SteeringConfig(
                vectors=vectors or [],
                alpha=alpha
            )
        super().__init__(steering_model, config)
        
        self.agent_name = agent_name
        self.enable_tracing = enable_tracing
        self.tracing_config = tracing_config or {}
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        
        # Initialize tracing if enabled
        if self.enable_tracing:
            self._setup_tracing()
    
    def _setup_tracing(self) -> None:
        """Setup Azure AI tracing integration."""
        try:
            # Import tracing dependencies
            from azure.monitor.opentelemetry import configure_azure_monitor
            
            # Configure tracing
            connection_string = self.tracing_config.get(
                "connection_string",
                None
            )
            
            if connection_string:
                configure_azure_monitor(
                    connection_string=connection_string
                )
            
            # Additional tracing setup could go here
            
        except ImportError:
            raise ImportError(
                "Azure monitoring packages not installed. "
                "Install with: pip install azure-monitor-opentelemetry"
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
    
    async def agenerate(
        self,
        prompt: Union[str, List[str]],
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Async generate text with steering (falls back to sync).
        
        Args:
            prompt: Input prompt(s)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text(s)
        """
        # For now, fall back to synchronous generation
        # Future: Implement true async generation
        return self.generate(prompt, **kwargs)
    
    def to_azure_deployment(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare configuration for Azure AI Foundry deployment.
        
        Args:
            endpoint: Azure AI endpoint URL
            api_key: API key for authentication
            deployment_name: Deployment name (defaults to agent_name)
        
        Returns:
            Deployment configuration dictionary
        """
        deployment_config = {
            "agent_name": deployment_name or self.agent_name,
            "endpoint": endpoint,
            "api_key": api_key,
            "model_config": {
                "model_name": self.steering_model.model.config.name_or_path,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            "steering_config": {
                "vectors": [
                    {
                        "layer": v.layer,
                        "layer_name": v.layer_name,
                        "magnitude": v.magnitude,
                        "method": v.method,
                    }
                    for v in self.config.vectors
                ],
                "alpha": self.config.alpha,
                "composition_method": self.config.composition_method,
            },
            "tracing_enabled": self.enable_tracing,
        }
        
        return deployment_config


def create_prompt_flow_config(
    agent: AzureSteeringAgent,
    flow_name: str,
    inputs: List[str],
    outputs: List[str],
) -> Dict[str, Any]:
    """
    Create a Prompt Flow configuration for the steering agent.
    
    Args:
        agent: Configured AzureSteeringAgent
        flow_name: Name for the Prompt Flow
        inputs: List of input variable names
        outputs: List of output variable names
    
    Returns:
        Prompt Flow configuration dictionary
    
    Example:
        >>> config = create_prompt_flow_config(
        ...     agent=agent,
        ...     flow_name="safety_flow",
        ...     inputs=["user_query"],
        ...     outputs=["safe_response"]
        ... )
    """
    flow_config = {
        "name": flow_name,
        "type": "chat",
        "inputs": {name: {"type": "string"} for name in inputs},
        "outputs": {name: {"type": "string"} for name in outputs},
        "nodes": [
            {
                "name": "steering_node",
                "type": "llm",
                "source": {
                    "type": "agent",
                    "agent_name": agent.agent_name,
                },
                "inputs": {
                    "steering_enabled": True,
                    "alpha": agent.config.alpha,
                },
            }
        ],
    }
    
    return flow_config


def create_multi_agent_orchestration(
    agents: List[AzureSteeringAgent],
    orchestration_strategy: str = "sequential",
) -> Dict[str, Any]:
    """
    Create multi-agent orchestration configuration.
    
    Args:
        agents: List of AzureSteeringAgent instances
        orchestration_strategy: Strategy ("sequential", "parallel", "hierarchical")
    
    Returns:
        Orchestration configuration
    
    Example:
        >>> safety_agent = AzureSteeringAgent(...)
        >>> expert_agent = AzureSteeringAgent(...)
        >>> 
        >>> orchestration = create_multi_agent_orchestration(
        ...     agents=[safety_agent, expert_agent],
        ...     orchestration_strategy="sequential"
        ... )
    """
    if orchestration_strategy not in {"sequential", "parallel", "hierarchical"}:
        raise ValueError(
            f"orchestration_strategy must be one of "
            f"(sequential, parallel, hierarchical), got {orchestration_strategy}"
        )
    
    orchestration_config = {
        "strategy": orchestration_strategy,
        "agents": [
            {
                "name": agent.agent_name,
                "config": agent.to_azure_deployment(
                    endpoint="placeholder",
                    api_key="placeholder"
                ),
            }
            for agent in agents
        ],
        "routing": {
            "sequential": orchestration_strategy == "sequential",
            "parallel": orchestration_strategy == "parallel",
            "hierarchical": orchestration_strategy == "hierarchical",
        },
    }
    
    return orchestration_config
