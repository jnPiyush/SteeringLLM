"""
LangChain Integration Examples for SteeringLLM.

This script demonstrates how to use SteeringLLM with LangChain for
building steered agents and chains.
"""

from steering_llm import SteeringModel, Discovery
from steering_llm.agents import LangChainSteeringLLM, create_safety_agent, create_domain_expert_agent

# Note: Requires: pip install langchain


def example_basic_chain():
    """Example 1: Basic LangChain chain with steering."""
    print("=" * 60)
    print("Example 1: Basic LangChain Chain with Steering")
    print("=" * 60)
    
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Create steering vector for helpful responses
    print("Creating helpful steering vector...")
    helpful_vector = Discovery.mean_difference(
        positive=[
            "I'm happy to help you with that!",
            "Let me assist you.",
            "I'd be glad to explain that.",
        ],
        negative=[
            "I don't care about that.",
            "That's not my problem.",
            "Figure it out yourself.",
        ],
        model=steering_model,
        layer=10
    )
    
    # Create LangChain LLM
    print("Creating LangChain LLM with steering...")
    llm = LangChainSteeringLLM(
        steering_model=steering_model,
        vectors=[helpful_vector],
        alpha=2.0,
        max_length=100,
        temperature=0.7
    )
    
    # Create chain
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a brief explanation about {topic}:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run chain
    print("\nGenerating with steering...")
    result = chain.run(topic="machine learning")
    print(f"\nResult:\n{result}")
    
    print("\n✓ Example 1 complete!\n")


def example_safety_agent():
    """Example 2: Safety-constrained agent."""
    print("=" * 60)
    print("Example 2: Safety-Constrained Agent")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create safety steering vector
    print("Creating safety steering vector...")
    safety_vector = Discovery.caa(
        positive=[
            "I'm here to help in a safe and respectful way.",
            "Let's discuss this appropriately.",
            "I aim to be helpful and harmless.",
        ],
        negative=[
            "[toxic content examples]",
            "[harmful content examples]",
            "[biased content examples]",
        ],
        model=steering_model,
        layer=15
    )
    
    # Create safety agent
    print("Creating safety agent...")
    agent = create_safety_agent(
        steering_model=steering_model,
        safety_vector=safety_vector,
        alpha=2.5,
        max_length=150
    )
    
    # Test with various prompts
    test_prompts = [
        "How can I help someone learn programming?",
        "What's the best way to resolve conflicts?",
        "Tell me about different cultures.",
    ]
    
    print("\nGenerating safe responses...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = agent.generate(prompt)
        print(f"Response: {response}")
    
    print("\n✓ Example 2 complete!\n")


def example_domain_expert():
    """Example 3: Domain-expert agent with multiple vectors."""
    print("=" * 60)
    print("Example 3: Domain-Expert Agent")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create medical domain vector
    print("Creating medical domain vector...")
    medical_vector = Discovery.caa(
        positive=[
            "The patient presented with clinical symptoms requiring diagnosis.",
            "Medical treatment involves careful assessment of the condition.",
            "Clinical evidence suggests therapeutic intervention.",
        ],
        negative=[
            "The person was feeling bad and needed help.",
            "They went to the doctor because they were sick.",
            "The medicine made them feel better.",
        ],
        model=steering_model,
        layer=12
    )
    
    # Create formal language vector
    print("Creating formal language vector...")
    formal_vector = Discovery.mean_difference(
        positive=[
            "Furthermore, it is important to consider the implications.",
            "The analysis indicates several key findings.",
            "Consequently, the evidence suggests a clear conclusion.",
        ],
        negative=[
            "Also, you should think about this stuff.",
            "The results show some things.",
            "So, basically, it means this.",
        ],
        model=steering_model,
        layer=18
    )
    
    # Create domain expert agent
    print("Creating domain expert agent...")
    agent = create_domain_expert_agent(
        steering_model=steering_model,
        domain_vectors=[medical_vector, formal_vector],
        weights=[0.7, 0.3],
        max_length=200
    )
    
    # Test medical Q&A
    test_prompts = [
        "What should someone do if they have persistent headaches?",
        "Explain the importance of vaccination.",
        "What are common symptoms of the flu?",
    ]
    
    print("\nGenerating domain-expert responses...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = agent.generate(prompt)
        print(f"Response: {response}")
    
    print("\n✓ Example 3 complete!\n")


def example_conversational_agent():
    """Example 4: Conversational agent with tool use."""
    print("=" * 60)
    print("Example 4: Conversational Agent with Tools")
    print("=" * 60)
    
    try:
        from langchain.agents import AgentType, initialize_agent, load_tools
    except ImportError:
        print("Skipping: Requires langchain with agent tools")
        return
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create conversational steering vector
    print("Creating conversational steering vector...")
    conversational_vector = Discovery.mean_difference(
        positive=[
            "I'd be happy to help you with that question.",
            "That's an interesting topic! Let me explain.",
            "Great question! Here's what I know.",
        ],
        negative=[
            "I don't want to talk about that.",
            "That's boring.",
            "Ask someone else.",
        ],
        model=steering_model,
        layer=10
    )
    
    # Create LangChain LLM
    print("Creating conversational LLM...")
    llm = LangChainSteeringLLM(
        steering_model=steering_model,
        vectors=[conversational_vector],
        alpha=1.5,
        max_length=150
    )
    
    # Note: This is a simplified example
    # In production, you would configure proper tools
    print("\nAgent would use tools like search, calculators, etc.")
    print("This requires additional LangChain configuration.")
    
    # Test conversational generation
    prompts = [
        "Tell me about Python programming.",
        "What are the benefits of exercise?",
    ]
    
    print("\nGenerating conversational responses...")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = llm.generate(prompt)
        print(f"Response: {response}")
    
    print("\n✓ Example 4 complete!\n")


def example_context_manager():
    """Example 5: Using context manager for temporary steering."""
    print("=" * 60)
    print("Example 5: Context Manager for Temporary Steering")
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
        positive=["Formal professional language."],
        negative=["Casual everyday speech."],
        model=steering_model,
        layer=10
    )
    
    # Create LLM
    llm = LangChainSteeringLLM(
        steering_model=steering_model,
        vectors=[vector],
        alpha=2.0,
        max_length=100
    )
    
    prompt = "Describe artificial intelligence"
    
    # Generate without steering
    print("\nGenerating without steering...")
    response_without = steering_model.generate(prompt, max_length=100)
    print(f"Without steering: {response_without}")
    
    # Generate with steering using context manager
    print("\nGenerating with steering (context manager)...")
    with llm:
        response_with = llm.generate(prompt)
        print(f"With steering: {response_with}")
    
    # Steering is automatically removed after context
    print("\nSteering removed after context manager exits.")
    print(f"Steering active: {llm.is_steering_active}")
    
    print("\n✓ Example 5 complete!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangChain Integration Examples")
    print("=" * 60 + "\n")
    
    print("Note: These examples require 'langchain' to be installed:")
    print("  pip install langchain\n")
    
    # Run examples
    try:
        example_basic_chain()
        example_safety_agent()
        example_domain_expert()
        example_conversational_agent()
        example_context_manager()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except ImportError as e:
        print(f"\n⚠ Import error: {e}")
        print("Please install required dependencies:")
        print("  pip install langchain")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
