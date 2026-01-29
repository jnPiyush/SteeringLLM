"""
Microsoft Agent Framework Integration Examples for SteeringLLM.

This script demonstrates how to use SteeringLLM with Microsoft's agent-framework
for Azure AI Foundry deployment and enterprise features.
"""

from steering_llm import SteeringModel, Discovery
from steering_llm.agents import AzureSteeringAgent, create_prompt_flow_config, create_multi_agent_orchestration

# Note: Requires: pip install agent-framework azure-monitor-opentelemetry


def example_basic_azure_agent():
    """Example 1: Basic Azure agent with steering."""
    print("=" * 60)
    print("Example 1: Basic Azure Agent with Steering")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create steering vector
    print("Creating steering vector...")
    helpful_vector = Discovery.mean_difference(
        positive=[
            "I'm here to assist you with your questions.",
            "Let me help you understand this better.",
            "I'd be pleased to provide information.",
        ],
        negative=[
            "I don't want to answer that.",
            "That's not interesting.",
            "Ask someone else.",
        ],
        model=steering_model,
        layer=10
    )
    
    # Create Azure agent
    print("Creating Azure agent...")
    agent = AzureSteeringAgent(
        steering_model=steering_model,
        agent_name="helpful_assistant",
        vectors=[helpful_vector],
        alpha=2.0,
        max_tokens=150,
        temperature=0.7
    )
    
    # Generate responses
    test_prompts = [
        "What is machine learning?",
        "How do I learn programming?",
        "Explain cloud computing.",
    ]
    
    print("\nGenerating with Azure agent...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = agent.generate(prompt)
        print(f"Response: {response}")
    
    print("\n✓ Example 1 complete!\n")


def example_azure_tracing():
    """Example 2: Azure agent with tracing enabled."""
    print("=" * 60)
    print("Example 2: Azure Agent with Tracing")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create steering vector
    print("Creating steering vector...")
    vector = Discovery.caa(
        positive=["Professional and helpful responses."],
        negative=["Casual or unhelpful responses."],
        model=steering_model,
        layer=12
    )
    
    # Create agent with tracing
    print("Creating agent with tracing...")
    # Note: In production, provide real Application Insights connection string
    tracing_config = {
        "connection_string": "InstrumentationKey=your-key-here;IngestionEndpoint=https://..."
        # For demo purposes, this would normally connect to Azure Monitor
    }
    
    agent = AzureSteeringAgent(
        steering_model=steering_model,
        agent_name="traced_agent",
        vectors=[vector],
        alpha=1.5,
        enable_tracing=False,  # Set to False for demo (requires real Azure setup)
        tracing_config=tracing_config,
        max_tokens=100
    )
    
    print("\nNote: Tracing is configured but disabled for demo.")
    print("In production, set enable_tracing=True with valid Azure Monitor connection.")
    
    # Generate response
    prompt = "Explain the benefits of cloud computing."
    print(f"\nPrompt: {prompt}")
    response = agent.generate(prompt)
    print(f"Response: {response}")
    
    print("\nWith tracing enabled, you would see:")
    print("  - Request traces in Azure Application Insights")
    print("  - Performance metrics")
    print("  - Dependency tracking")
    print("  - Custom events for steering operations")
    
    print("\n✓ Example 2 complete!\n")


def example_azure_deployment_config():
    """Example 3: Prepare agent for Azure AI Foundry deployment."""
    print("=" * 60)
    print("Example 3: Azure AI Foundry Deployment Configuration")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create multiple steering vectors
    print("Creating steering vectors...")
    safety_vector = Discovery.caa(
        positive=["Safe and appropriate content."],
        negative=["Harmful or inappropriate content."],
        model=steering_model,
        layer=15
    )
    
    domain_vector = Discovery.mean_difference(
        positive=["Technical and professional language."],
        negative=["Casual conversation."],
        model=steering_model,
        layer=10
    )
    
    # Create agent
    print("Creating agent...")
    agent = AzureSteeringAgent(
        steering_model=steering_model,
        agent_name="production_agent",
        vectors=[safety_vector, domain_vector],
        alpha=2.0,
        max_tokens=200
    )
    
    # Generate deployment configuration
    print("\nGenerating deployment configuration...")
    deployment_config = agent.to_azure_deployment(
        endpoint="https://your-resource.cognitiveservices.azure.com/",
        api_key="your-api-key-here",
        deployment_name="steering-agent-v1"
    )
    
    print("\nDeployment Configuration:")
    import json
    print(json.dumps(deployment_config, indent=2))
    
    print("\nThis configuration can be used to:")
    print("  - Deploy to Azure AI Foundry")
    print("  - Configure in Azure Portal")
    print("  - Use in Prompt Flow")
    print("  - Share with team members")
    
    print("\n✓ Example 3 complete!\n")


def example_prompt_flow():
    """Example 4: Prompt Flow integration."""
    print("=" * 60)
    print("Example 4: Prompt Flow Integration")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create steering vector
    print("Creating steering vector...")
    vector = Discovery.mean_difference(
        positive=["Professional customer service responses."],
        negative=["Unprofessional or rude responses."],
        model=steering_model,
        layer=12
    )
    
    # Create agent
    print("Creating agent for Prompt Flow...")
    agent = AzureSteeringAgent(
        steering_model=steering_model,
        agent_name="customer_service_agent",
        vectors=[vector],
        alpha=2.0,
        max_tokens=150
    )
    
    # Create Prompt Flow configuration
    print("\nCreating Prompt Flow configuration...")
    flow_config = create_prompt_flow_config(
        agent=agent,
        flow_name="customer_service_flow",
        inputs=["user_query", "customer_context"],
        outputs=["agent_response", "confidence_score"]
    )
    
    print("\nPrompt Flow Configuration:")
    import json
    print(json.dumps(flow_config, indent=2))
    
    print("\nThis configuration enables:")
    print("  - Visual flow design in Prompt Flow")
    print("  - Integration with other Azure services")
    print("  - A/B testing of different steering configurations")
    print("  - Monitoring and analytics")
    
    print("\n✓ Example 4 complete!\n")


def example_multi_agent_orchestration():
    """Example 5: Multi-agent orchestration."""
    print("=" * 60)
    print("Example 5: Multi-Agent Orchestration")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create safety agent
    print("Creating safety agent...")
    safety_vector = Discovery.caa(
        positive=["Safe and appropriate responses."],
        negative=["Unsafe or inappropriate responses."],
        model=steering_model,
        layer=15
    )
    safety_agent = AzureSteeringAgent(
        steering_model=steering_model,
        agent_name="safety_agent",
        vectors=[safety_vector],
        alpha=2.5
    )
    
    # Create domain expert agent
    print("Creating domain expert agent...")
    expert_vector = Discovery.mean_difference(
        positive=["Expert technical knowledge and explanations."],
        negative=["Basic or surface-level information."],
        model=steering_model,
        layer=12
    )
    expert_agent = AzureSteeringAgent(
        steering_model=steering_model,
        agent_name="expert_agent",
        vectors=[expert_vector],
        alpha=1.8
    )
    
    # Create orchestration configuration
    print("\nCreating multi-agent orchestration...")
    orchestration = create_multi_agent_orchestration(
        agents=[safety_agent, expert_agent],
        orchestration_strategy="sequential"
    )
    
    print("\nOrchestration Configuration:")
    import json
    print(json.dumps(orchestration, indent=2))
    
    print("\nOrchestration strategies:")
    print("  - Sequential: Safety check → Expert response")
    print("  - Parallel: Both agents process, combine results")
    print("  - Hierarchical: Coordinator agent routes to specialists")
    
    print("\nUse cases:")
    print("  - Content moderation + generation")
    print("  - Multi-domain question answering")
    print("  - Staged processing pipelines")
    
    print("\n✓ Example 5 complete!\n")


def example_async_generation():
    """Example 6: Async generation for high throughput."""
    print("=" * 60)
    print("Example 6: Async Generation")
    print("=" * 60)
    
    import asyncio
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create steering vector
    print("Creating steering vector...")
    vector = Discovery.mean_difference(
        positive=["Helpful and informative responses."],
        negative=["Unhelpful or vague responses."],
        model=steering_model,
        layer=10
    )
    
    # Create agent
    print("Creating agent...")
    agent = AzureSteeringAgent(
        steering_model=steering_model,
        agent_name="async_agent",
        vectors=[vector],
        alpha=1.5,
        max_tokens=100
    )
    
    # Define async generation function
    async def generate_multiple():
        prompts = [
            "What is Python?",
            "Explain machine learning.",
            "What is cloud computing?",
        ]
        
        print("\nGenerating async responses...")
        tasks = [agent.agenerate(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        for prompt, response in zip(prompts, responses):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
    
    # Run async generation
    try:
        asyncio.run(generate_multiple())
    except RuntimeError:
        # Handle if already in async context
        print("\nNote: Async generation demonstrated (would run in async context)")
    
    print("\nBenefits of async generation:")
    print("  - Higher throughput for multiple requests")
    print("  - Better resource utilization")
    print("  - Scales well in production")
    
    print("\n✓ Example 6 complete!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Microsoft Agent Framework Integration Examples")
    print("=" * 60 + "\n")
    
    print("Note: These examples require 'agent-framework' to be installed:")
    print("  pip install agent-framework azure-monitor-opentelemetry\n")
    
    # Run examples
    try:
        example_basic_azure_agent()
        example_azure_tracing()
        example_azure_deployment_config()
        example_prompt_flow()
        example_multi_agent_orchestration()
        example_async_generation()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
        print("Next steps:")
        print("  1. Configure Azure AI Foundry workspace")
        print("  2. Deploy agent to Azure")
        print("  3. Set up monitoring and tracing")
        print("  4. Create Prompt Flow for production")
        
    except ImportError as e:
        print(f"\n⚠ Import error: {e}")
        print("Please install required dependencies:")
        print("  pip install agent-framework azure-monitor-opentelemetry")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
