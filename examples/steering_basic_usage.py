"""
Integration example demonstrating SteeringModel usage.

This script shows how to use SteeringModel to apply steering vectors
at inference time. This is for documentation purposes only.
"""

# Example usage (not runnable without actual model):
"""
from steering_llm import SteeringModel, Discovery

# 1. Load model with steering capabilities
model = SteeringModel.from_pretrained("meta-llama/Llama-3.2-3B")

# 2. Create steering vector from contrast examples
positive_examples = [
    "I love helping people!",
    "You're doing great!",
    "That's wonderful news!",
]

negative_examples = [
    "I hate helping people.",
    "You're doing terribly.",
    "That's awful news.",
]

vector = Discovery.mean_difference(
    positive=positive_examples,
    negative=negative_examples,
    model=model.model,  # Access underlying HF model
    layer=15,
)

# 3. Apply steering and generate
model.apply_steering(vector, alpha=2.0)

output = model.generate(
    "Tell me about yourself",
    max_length=100,
)

print(output)

# 4. Check active steering
active = model.list_active_steering()
print(f"Active steering: {active}")

# 5. Remove steering
model.remove_steering()

# OR use convenience method for automatic cleanup:
output = model.generate_with_steering(
    prompt="Tell me about yourself",
    vector=vector,
    alpha=2.0,
    max_length=100,
)
"""
