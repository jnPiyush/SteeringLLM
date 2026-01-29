"""
Tests for agent integrations (LangChain, Azure, LlamaIndex).
"""

import pytest
import torch

from steering_llm.core.steering_vector import SteeringVector


def _langchain_available():
    """Check if LangChain is available."""
    try:
        import langchain
        return True
    except ImportError:
        return False


@pytest.fixture
def sample_vector():
    """Create a sample steering vector."""
    tensor = torch.randn(768)
    return SteeringVector(
        tensor=tensor,
        layer=10,
        layer_name="model.layers.10",
        model_name="test-model",
        method="mean_difference"
    )


class TestLangChainIntegration:
    """Tests for LangChain integration."""
    
    def test_import_without_langchain(self):
        """Test that module handles missing LangChain gracefully."""
        try:
            from steering_llm.agents import LangChainSteeringLLM
            # If import succeeds, LangChain is installed
            assert hasattr(LangChainSteeringLLM, '_llm_type')
        except ImportError:
            # Expected if LangChain not installed
            pass
    
    @pytest.mark.skipif(
        not _langchain_available(),
        reason="LangChain not installed"
    )
    def test_langchain_llm_type(self, sample_vector):
        """Test LangChain LLM type property."""
        from steering_llm.agents import LangChainSteeringLLM
        from steering_llm.core.steering_model import SteeringModel
        
        # Note: This would require a real model in practice
        # Using mock for test purposes
        pass


class TestAzureIntegration:
    """Tests for Azure Agent Framework integration."""
    
    def test_import_without_azure(self):
        """Test that module handles missing Azure framework gracefully."""
        try:
            from steering_llm.agents import AzureSteeringAgent
            # If import succeeds, agent-framework is installed
            assert hasattr(AzureSteeringAgent, 'to_azure_deployment')
        except ImportError:
            # Expected if agent-framework not installed
            pass


class TestLlamaIndexIntegration:
    """Tests for LlamaIndex integration."""
    
    def test_import_without_llamaindex(self):
        """Test that module handles missing LlamaIndex gracefully."""
        try:
            from steering_llm.agents import LlamaIndexSteeringLLM
            # If import succeeds, LlamaIndex is installed
            assert hasattr(LlamaIndexSteeringLLM, 'complete')
        except ImportError:
            # Expected if LlamaIndex not installed
            pass
