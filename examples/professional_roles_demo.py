"""
Professional Role Behavior Demonstration

This example shows how to use steering vectors to make a model adopt
different professional personas (Doctor, Lawyer, Engineer, Teacher, etc.)

Usage:
    python examples/professional_roles_demo.py
"""

from steering_llm import SteeringModel, Discovery


# Professional role steering examples
ROLES = {
    "doctor": {
        "positive": [
            "Based on your symptoms, I recommend scheduling an appointment with your primary care physician.",
            "The clinical evidence suggests this could be a viral infection.",
            "I understand this diagnosis may be concerning. Let me explain what this means.",
            "Your vital signs indicate we should monitor your condition closely.",
        ],
        "negative": [
            "Just google your symptoms and self-diagnose.",
            "That's probably nothing serious, don't worry about it.",
            "Skip the doctor, just buy some random pills.",
            "Who needs professional diagnosis? Trust your gut feeling!",
        ],
    },
    "lawyer": {
        "positive": [
            "Under the applicable statute (Section 42 USC 1983), you may have grounds for a claim.",
            "I must advise you that this constitutes attorney-client privileged communication.",
            "The precedent established in Smith v. Jones (2015) is directly on point.",
            "Per the terms of the contract, the arbitration clause in Section 12.3 would apply.",
        ],
        "negative": [
            "Just do whatever you want, laws don't really matter.",
            "Legal stuff is boring, let's just wing it!",
            "Who cares about contracts? Just shake hands and trust them.",
            "Court procedures? Just show up and argue loudly.",
        ],
    },
    "engineer": {
        "positive": [
            "Let's refactor this code to follow the Single Responsibility Principle.",
            "I'd recommend implementing a circuit breaker pattern to handle API failures.",
            "The time complexity of this algorithm is O(n log n), which should scale well.",
            "We should add unit tests with at least 80% code coverage before deploying.",
        ],
        "negative": [
            "Just copy-paste some code from Stack Overflow, it'll probably work.",
            "Who needs tests? If it runs on my machine, ship it!",
            "Performance? We'll worry about that when users complain.",
            "Security vulnerabilities? That's someone else's problem.",
        ],
    },
    "teacher": {
        "positive": [
            "Great question! Let's break this concept down into smaller, easier pieces.",
            "I can see you're working hard on this. Let me show you another way.",
            "Remember, making mistakes is an important part of learning.",
            "That's an excellent observation! Can you tell me what made you think of that?",
        ],
        "negative": [
            "That's wrong. You should have known this already.",
            "Stop asking stupid questions. Pay attention!",
            "If you can't figure this out, you're just not smart enough.",
            "I've explained this three times. I'm not repeating it again.",
        ],
    },
}


def demonstrate_role(model, role_name, prompt, layer=6, alpha=2.5):
    """
    Demonstrate a specific professional role behavior.
    
    Args:
        model: SteeringModel instance
        role_name: Name of the role (doctor, lawyer, engineer, teacher)
        prompt: Question to ask the model
        layer: Layer to apply steering (default: 6)
        alpha: Steering strength (default: 2.5)
    """
    print(f"\n{'='*70}")
    print(f"ROLE: {role_name.upper()}")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}\n")
    
    role_data = ROLES[role_name]
    
    # Create steering vector
    result = Discovery.mean_difference(
        positive=role_data["positive"],
        negative=role_data["negative"],
        model=model.model,
        layer=layer,
        tokenizer=model.tokenizer,
    )
    
    # Generate baseline response
    print("🔵 BASELINE (No steering):")
    baseline = model.generate(
        prompt,
        max_new_tokens=300,
        temperature=0.8,
        do_sample=True
    )
    print(f"   {baseline}\n")
    
    # Generate steered response
    print(f"🎯 STEERED ({role_name.title()} mode, alpha={alpha}):")
    steered = model.generate_with_steering(
        prompt,
        vector=result.vector,
        alpha=alpha,
        max_new_tokens=300,
        temperature=0.8,
        do_sample=True
    )
    print(f"   {steered}\n")
    
    # Show metrics if available
    if result.metrics:
        print("📊 Vector Metrics:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.4f}")
            else:
                print(f"   - {key}: {value}")


def main():
    """Run professional role demonstrations."""
    print("="*70)
    print("Professional Role Behavior Demonstration")
    print("="*70)
    print("\nLoading model (GPT-2)...")
    
    # Load model
    model = SteeringModel.from_pretrained("gpt2-large")
    model.eval()
    
    print(f"✓ Model loaded: {model.num_layers} layers")
    
    # Define test prompts for each role
    test_cases = [
        ("doctor", "I have a persistent headache and fever. What should I do?"),
        ("lawyer", "Can I break my apartment lease early without penalty?"),
        ("engineer", "How should I structure this Python application for scalability?"),
        ("teacher", "I don't understand how photosynthesis works. Can you explain?"),
    ]
    
    # Run demonstrations
    for role, prompt in test_cases:
        demonstrate_role(model, role, prompt)
    
    print("\n" + "="*70)
    print("💡 KEY TAKEAWAYS:")
    print("="*70)
    print("""
    1. Same model, different behaviors! Steering vectors modify response style
       without retraining or fine-tuning.
    
    2. Professional roles are composable - you can combine multiple vectors
       (e.g., Doctor + Empathy + Concise).
    
    3. Adjust alpha (steering strength) to control how strong the role behavior is:
       - alpha=1.0  → Subtle changes
       - alpha=2.5  → Moderate professional tone
       - alpha=5.0  → Very strong role behavior (may be too extreme)
    
    4. Layer selection matters:
       - Early layers (0-30%): Basic linguistic patterns
       - Middle layers (40-60%): Semantic meaning and style
       - Late layers (70-90%): Task-specific behavior
       
       Professional roles work best in middle layers (around 50-65%).
    
    5. Try these in the interactive demo:
       python demo/launch.py
       Then select "Doctor", "Lawyer", or other role presets!
    """)


if __name__ == "__main__":
    main()
