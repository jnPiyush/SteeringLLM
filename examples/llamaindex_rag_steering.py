"""
LlamaIndex RAG Integration Examples for SteeringLLM.

This script demonstrates how to use SteeringLLM with LlamaIndex for
building steered retrieval-augmented generation (RAG) applications.
"""

from pathlib import Path
from steering_llm import SteeringModel, Discovery
from steering_llm.agents import LlamaIndexSteeringLLM, create_rag_steering_llm, create_multi_vector_rag_llm

# Note: Requires: pip install llama-index


def example_basic_rag():
    """Example 1: Basic RAG with steering."""
    print("=" * 60)
    print("Example 1: Basic RAG with Steering")
    print("=" * 60)
    
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
    except ImportError:
        print("Skipping: Requires llama-index")
        return
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create domain steering vector
    print("Creating domain steering vector...")
    technical_vector = Discovery.mean_difference(
        positive=[
            "Technical documentation and precise explanations.",
            "Detailed analysis with specific terminology.",
            "Comprehensive technical overview.",
        ],
        negative=[
            "Simple casual explanation.",
            "Basic overview without details.",
            "Informal discussion.",
        ],
        model=steering_model,
        layer=12
    )
    
    # Create steered LLM
    print("Creating steered LLM for RAG...")
    llm = LlamaIndexSteeringLLM(
        steering_model=steering_model,
        vectors=[technical_vector],
        alpha=1.8,
        max_tokens=200,
        temperature=0.7
    )
    
    # Create sample documents
    print("\nCreating sample documents...")
    documents = [
        Document(text="Python is a high-level programming language known for its simplicity and readability."),
        Document(text="Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
        Document(text="Neural networks are computational models inspired by biological neural networks in the brain."),
    ]
    
    # Build index
    print("Building vector store index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine with steered LLM
    print("Creating query engine...")
    query_engine = index.as_query_engine(llm=llm)
    
    # Query with steering
    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What are neural networks?",
    ]
    
    print("\nQuerying with steered RAG...")
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")
    
    print("\n✓ Example 1 complete!\n")


def example_domain_adapted_rag():
    """Example 2: Domain-adapted RAG."""
    print("=" * 60)
    print("Example 2: Domain-Adapted RAG")
    print("=" * 60)
    
    try:
        from llama_index.core import VectorStoreIndex, Document
    except ImportError:
        print("Skipping: Requires llama-index")
        return
    
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
            "Medical assessment indicated treatment protocols.",
            "Clinical evidence supports therapeutic intervention.",
        ],
        negative=[
            "The person was sick and went to the doctor.",
            "They felt bad and got medicine.",
            "The doctor helped them feel better.",
        ],
        model=steering_model,
        layer=15
    )
    
    # Create RAG LLM with domain adaptation
    print("Creating domain-adapted RAG LLM...")
    llm = create_rag_steering_llm(
        steering_model=steering_model,
        domain_vector=medical_vector,
        alpha=2.0,
        max_tokens=250,
        temperature=0.5  # Lower temperature for medical domain
    )
    
    # Create medical documents
    print("\nCreating medical documents...")
    documents = [
        Document(text="Hypertension is a chronic medical condition characterized by elevated blood pressure."),
        Document(text="Diabetes mellitus involves impaired insulin production or insulin resistance affecting glucose metabolism."),
        Document(text="Cardiac arrhythmias are irregular heart rhythms that may require clinical intervention."),
    ]
    
    # Build index and query engine
    print("Building medical knowledge index...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    
    # Medical queries
    queries = [
        "What is hypertension?",
        "Explain diabetes mellitus.",
        "What are cardiac arrhythmias?",
    ]
    
    print("\nQuerying medical knowledge base...")
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")
    
    print("\n✓ Example 2 complete!\n")


def example_multi_vector_rag():
    """Example 3: Multi-vector RAG with domain + style + safety."""
    print("=" * 60)
    print("Example 3: Multi-Vector RAG")
    print("=" * 60)
    
    try:
        from llama_index.core import VectorStoreIndex, Document
    except ImportError:
        print("Skipping: Requires llama-index")
        return
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create multiple steering vectors
    print("Creating multiple steering vectors...")
    
    # Domain vector (technical)
    domain_vector = Discovery.caa(
        positive=["Technical and precise explanations."],
        negative=["Vague or imprecise descriptions."],
        model=steering_model,
        layer=10
    )
    
    # Style vector (formal)
    style_vector = Discovery.mean_difference(
        positive=["Formal professional writing style."],
        negative=["Casual conversational style."],
        model=steering_model,
        layer=15
    )
    
    # Safety vector
    safety_vector = Discovery.caa(
        positive=["Safe and appropriate content."],
        negative=["Inappropriate or harmful content."],
        model=steering_model,
        layer=18
    )
    
    # Create multi-vector RAG LLM
    print("Creating multi-vector RAG LLM...")
    llm = create_multi_vector_rag_llm(
        steering_model=steering_model,
        vectors=[domain_vector, style_vector, safety_vector],
        weights=[0.5, 0.3, 0.2],
        composition_method="weighted",
        max_tokens=300
    )
    
    # Create documents
    print("\nCreating document collection...")
    documents = [
        Document(text="Artificial intelligence encompasses machine learning, natural language processing, and computer vision."),
        Document(text="Cloud computing provides on-demand access to computing resources over the internet."),
        Document(text="Cybersecurity involves protecting systems and networks from digital attacks and threats."),
    ]
    
    # Build index and query engine
    print("Building knowledge index...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    
    # Complex queries
    queries = [
        "Explain artificial intelligence applications.",
        "What are the benefits of cloud computing?",
        "How does cybersecurity protect organizations?",
    ]
    
    print("\nQuerying with multi-vector steering...")
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Response: {response}")
    
    print("\nNote: Multiple vectors applied simultaneously:")
    print("  - Domain expertise (50%)")
    print("  - Formal style (30%)")
    print("  - Safety constraints (20%)")
    
    print("\n✓ Example 3 complete!\n")


def example_context_aware_rag():
    """Example 4: Context-aware RAG with dynamic steering."""
    print("=" * 60)
    print("Example 4: Context-Aware RAG")
    print("=" * 60)
    
    try:
        from llama_index.core import VectorStoreIndex, Document
    except ImportError:
        print("Skipping: Requires llama-index")
        return
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create steering vectors for different contexts
    print("Creating context-specific steering vectors...")
    
    beginner_vector = Discovery.mean_difference(
        positive=["Simple explanations for beginners."],
        negative=["Complex technical jargon."],
        model=steering_model,
        layer=10
    )
    
    expert_vector = Discovery.caa(
        positive=["Advanced technical details for experts."],
        negative=["Oversimplified explanations."],
        model=steering_model,
        layer=12
    )
    
    # Create documents
    print("\nCreating knowledge base...")
    documents = [
        Document(text="Programming involves writing instructions for computers to execute."),
        Document(text="Data structures organize and store data efficiently for various operations."),
        Document(text="Algorithms are step-by-step procedures for solving computational problems."),
    ]
    
    # Build index
    print("Building index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # Query with beginner steering
    print("\n--- Beginner Mode ---")
    beginner_llm = LlamaIndexSteeringLLM(
        steering_model=steering_model,
        vectors=[beginner_vector],
        alpha=2.0,
        max_tokens=150
    )
    beginner_engine = index.as_query_engine(llm=beginner_llm)
    
    query = "What is programming?"
    print(f"\nQuery: {query}")
    response = beginner_engine.query(query)
    print(f"Beginner Response: {response}")
    
    # Query with expert steering
    print("\n--- Expert Mode ---")
    expert_llm = LlamaIndexSteeringLLM(
        steering_model=steering_model,
        vectors=[expert_vector],
        alpha=2.0,
        max_tokens=150
    )
    expert_engine = index.as_query_engine(llm=expert_llm)
    
    print(f"\nQuery: {query}")
    response = expert_engine.query(query)
    print(f"Expert Response: {response}")
    
    print("\nDemonstrates context-aware responses:")
    print("  - Beginner: Simple, accessible language")
    print("  - Expert: Technical depth and precision")
    
    print("\n✓ Example 4 complete!\n")


def example_streaming_rag():
    """Example 5: Streaming RAG responses."""
    print("=" * 60)
    print("Example 5: Streaming RAG Responses")
    print("=" * 60)
    
    try:
        from llama_index.core import VectorStoreIndex, Document
    except ImportError:
        print("Skipping: Requires llama-index")
        return
    
    # Load model
    print("Loading model...")
    steering_model = SteeringModel.from_pretrained(
        "gpt2",
        device_map="auto"
    )
    
    # Create steering vector
    print("Creating steering vector...")
    vector = Discovery.mean_difference(
        positive=["Detailed and informative responses."],
        negative=["Brief or incomplete answers."],
        model=steering_model,
        layer=12
    )
    
    # Create steered LLM
    print("Creating steered LLM...")
    llm = LlamaIndexSteeringLLM(
        steering_model=steering_model,
        vectors=[vector],
        alpha=1.5,
        max_tokens=200
    )
    
    # Create documents
    print("\nCreating documents...")
    documents = [
        Document(text="Quantum computing uses quantum mechanics principles for computation."),
        Document(text="Blockchain is a distributed ledger technology ensuring transparency and security."),
    ]
    
    # Build index
    print("Building index...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    
    # Note: Streaming is prepared but falls back to complete for demo
    print("\nStreaming query (demo with complete response)...")
    query = "What is quantum computing?"
    print(f"Query: {query}")
    
    # In production, this would stream tokens
    response = query_engine.query(query)
    print(f"Response: {response}")
    
    print("\nNote: Streaming support is available via stream_complete method.")
    print("Production implementation would stream tokens in real-time.")
    
    print("\n✓ Example 5 complete!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LlamaIndex RAG Integration Examples")
    print("=" * 60 + "\n")
    
    print("Note: These examples require 'llama-index' to be installed:")
    print("  pip install llama-index\n")
    
    # Run examples
    try:
        example_basic_rag()
        example_domain_adapted_rag()
        example_multi_vector_rag()
        example_context_aware_rag()
        example_streaming_rag()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
        print("Key takeaways:")
        print("  - Steering enhances RAG with domain expertise")
        print("  - Multi-vector steering for complex requirements")
        print("  - Context-aware responses improve user experience")
        print("  - Production-ready with streaming support")
        
    except ImportError as e:
        print(f"\n⚠ Import error: {e}")
        print("Please install required dependencies:")
        print("  pip install llama-index")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
