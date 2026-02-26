# SteeringLLM Examples

This directory contains practical examples demonstrating various use cases of SteeringLLM.

## Quick Start

All examples can be run directly:

```bash
# Make sure you're in the repo root
cd SteeringLLM

# Install with examples dependencies
pip install -e ".[all]"

# Run any example
python examples/steering_basic_usage.py
python examples/professional_roles_demo.py
```

---

## 📚 Available Examples

### 1. Basic Usage (`steering_basic_usage.py`)
**Best for**: First-time users, understanding core concepts

Learn the fundamentals of SteeringLLM:
- Loading models
- Creating steering vectors from contrast pairs
- Applying steering with different strengths
- Comparing baseline vs steered outputs

```bash
python examples/steering_basic_usage.py
```

**Topics covered:**
- Mean difference discovery
- Alpha parameter tuning
- Multiple vector application
- Vector composition basics

---

### 2. Professional Role Behaviors (`professional_roles_demo.py`) ⭐ NEW
**Best for**: Understanding how steering modifies personality and expertise

Demonstrates how steering vectors can make a model adopt different professional personas:
- 🩺 **Doctor** - Clinical knowledge, evidence-based advice
- ⚖️ **Lawyer** - Legal precision, citation of statutes
- 💻 **Software Engineer** - Technical best practices, systematic thinking
- 📚 **Teacher** - Patient explanations, encouraging tone

```bash
python examples/professional_roles_demo.py
```

**Key Insights:**
- Same base model, completely different behaviors
- No retraining or fine-tuning required
- Composable - combine multiple roles
- Adjustable strength (alpha parameter)

**Sample Output:**
```
ROLE: DOCTOR
Prompt: I have a headache and fever. What should I do?

🔵 BASELINE (No steering):
   You should probably rest and drink water.

🎯 STEERED (Doctor mode, alpha=2.5):
   Based on your symptoms, I recommend getting evaluated by your 
   primary care physician. Headache combined with fever could indicate 
   various conditions. In the meantime, monitor your temperature and 
   stay hydrated. Seek immediate care if symptoms worsen.
```

---

### 3. LangChain Integration (`langchain_steering_agent.py`)
**Best for**: Building AI agents with LangChain

Shows how to integrate SteeringLLM with LangChain:
- Custom LLM wrapper for LangChain chains
- Agent tools with steering
- Multi-step reasoning with behavioral control
- Chain composition with different steering vectors

```bash
python examples/langchain_steering_agent.py
```

**Use Cases:**
- Chatbots with controllable personality
- Research assistants with domain expertise steering
- Customer service agents with empathy control

---

### 4. LlamaIndex RAG (`llamaindex_rag_steering.py`)
**Best for**: Retrieval-Augmented Generation with behavioral control

Demonstrates steering in RAG applications:
- Custom LlamaIndex LLM wrapper
- Steering RAG response tone and style
- Domain-specific steering (e.g., medical, legal, technical)
- Query engine integration

```bash
python examples/llamaindex_rag_steering.py
```

**Use Cases:**
- Knowledge base Q&A with professional tone
- Document analysis with specific perspectives
- Research paper summarization with academic style

---

### 5. Azure Agent Framework (`azure_agent_foundry.py`)
**Best for**: Production deployment with Azure AI Foundry

Production-ready example for Azure deployment:
- Azure AI Foundry integration
- OpenTelemetry tracing
- Multi-agent orchestration
- Monitoring and logging

```bash
# Requires Azure credentials
python examples/azure_agent_foundry.py
```

**Prerequisites:**
- Azure subscription
- Azure AI Foundry project
- Environment variables: `FOUNDRY_ENDPOINT`, `FOUNDRY_API_KEY`

**Topics:**
- Model deployment to Azure
- Production monitoring and tracing
- Agent composition patterns
- Error handling and resilience

---

## 🎮 Interactive Demo

For hands-on experimentation, run the interactive Streamlit demo:

```bash
python demo/launch.py
```

The demo includes:
- 🎛️ **Steering Playground** - Real-time steering with sliders
- 🧩 **Multi-Vector Composition** - Combine multiple behaviors
- 📊 **Vector Analysis** - Similarity and conflict detection
- 💾 **Save/Load Vectors** - Reuse steering vectors

**Demo Features:**
- All professional role presets available
- Side-by-side baseline vs steered comparison
- Adjustable steering strength (alpha slider)
- Custom contrast pair creation
- Visual vector composition

---

## 📖 Example Comparison

| Example | Complexity | Integration | Auth Required | Best For |
|---------|-----------|-------------|---------------|----------|
| `steering_basic_usage.py` | ⭐ Easy | Standalone | ❌ No | Learning basics |
| `professional_roles_demo.py` | ⭐ Easy | Standalone | ❌ No | Understanding roles |
| `langchain_steering_agent.py` | ⭐⭐ Medium | LangChain | ❌ No | Agent building |
| `llamaindex_rag_steering.py` | ⭐⭐ Medium | LlamaIndex | ❌ No | RAG applications |
| `azure_agent_foundry.py` | ⭐⭐⭐ Advanced | Azure | ✅ Yes | Production deployment |

---

## 🔧 Troubleshooting

### Import Error: `steering_llm` not found
```bash
# Install in development mode from repo root
pip install -e "."
```

### Model Download Fails
For gated models (Llama, Gemma), authenticate first:
```bash
pip install huggingface-hub
huggingface-cli login
```

See [MODEL-DOWNLOAD-GUIDE.md](../MODEL-DOWNLOAD-GUIDE.md) for detailed instructions.

### Out of Memory
Use smaller models or quantization:
```python
model = SteeringModel.from_pretrained(
    "gpt2",  # Smaller model
    load_in_8bit=True,  # Quantization
    device_map="auto"
)
```

---

## 💡 Tips for Custom Use Cases

### Creating Your Own Role Behaviors

1. **Define clear contrast pairs:**
   ```python
   positive = [
       "Example of desired behavior",
       "Another example showing the target style",
       "More examples reinforce the pattern"
   ]
   
   negative = [
       "Example of opposite behavior",
       "What you want to avoid",
       "Anti-examples clarify the boundary"
   ]
   ```

2. **Use 4-8 examples per side:**
   - Too few: Vector may not generalize
   - Too many: Diminishing returns, slower discovery

3. **Keep examples consistent:**
   - Same length (roughly)
   - Same domain/context
   - Clear contrast between positive/negative

4. **Experiment with layers:**
   - Start with middle layers (50-65% of total)
   - Early layers: Linguistic style
   - Late layers: Task-specific behavior

5. **Tune alpha strength:**
   - 1.0 = Subtle changes
   - 2.0-3.0 = Balanced (recommended)
   - 5.0+ = Very strong (may reduce coherence)

### Combining Multiple Roles

```python
from steering_llm import VectorComposition

# Create multiple role vectors
doctor_vec = ...
empathy_vec = ...
concise_vec = ...

# Combine with weights
combined = VectorComposition.weighted_sum(
    vectors=[doctor_vec, empathy_vec, concise_vec],
    weights=[1.0, 0.7, 0.3],  # Prioritize doctor > empathy > concise
    normalize=True
)

# Apply combined vector
output = model.generate_with_steering(
    prompt, 
    vector=combined, 
    alpha=2.0
)
```

---

## 📚 Further Reading

- **Architecture**: [../docs/adr/ADR-001-steeringllm-architecture.md](../docs/adr/ADR-001-steeringllm-architecture.md)
- **API Reference**: [../docs/API-REFERENCE.md](../docs/API-REFERENCE.md)
- **Example Guides**: [../docs/examples/](../docs/examples/)
- **Model Download Guide**: [../MODEL-DOWNLOAD-GUIDE.md](../MODEL-DOWNLOAD-GUIDE.md)

---

## 🤝 Contributing Examples

Have a cool use case? We'd love to see it!

1. Create a new example file in `examples/`
2. Follow the existing format (docstring, main function, clear comments)
3. Add it to this README
4. Submit a PR!

**Example template:**
```python
"""
Your Example Title

Brief description of what this example demonstrates.

Usage:
    python examples/your_example.py
"""

def main():
    """Your example logic here."""
    pass

if __name__ == "__main__":
    main()
```

---

**Questions?** Open an issue at [github.com/jnPiyush/SteeringLLM/issues](https://github.com/jnPiyush/SteeringLLM/issues)
