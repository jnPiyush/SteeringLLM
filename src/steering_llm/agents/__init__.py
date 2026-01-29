"""
Agent framework integrations for SteeringLLM.

This module provides integrations with popular agent frameworks including
LangChain, Microsoft Agent Framework, and LlamaIndex.
"""

from steering_llm.agents.base import SteeringAgent, SteeringConfig

__all__ = [
    "SteeringAgent",
    "SteeringConfig",
]

# Conditional imports for optional dependencies
try:
    from steering_llm.agents.langchain_agent import LangChainSteeringLLM
    __all__.append("LangChainSteeringLLM")
except ImportError:
    pass

try:
    from steering_llm.agents.azure_agent import AzureSteeringAgent
    __all__.append("AzureSteeringAgent")
except ImportError:
    pass

try:
    from steering_llm.agents.llamaindex_agent import LlamaIndexSteeringLLM
    __all__.append("LlamaIndexSteeringLLM")
except ImportError:
    pass
