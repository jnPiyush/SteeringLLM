"""
Preset steering vector definitions for the demo UI.

These presets define contrast-pair datasets used to build steering vectors
for common behavioral modifications. Each preset includes positive and
negative examples, a recommended layer range, and a suggested alpha.
"""

from typing import Any, Dict, List

PRESETS: Dict[str, Dict[str, Any]] = {
    "Positive / Helpful": {
        "description": (
            "Steer the model toward positive, helpful, and encouraging tone."
        ),
        "positive": [
            "I love helping people solve problems! Let me assist you.",
            "That's a wonderful question. Here's what I think:",
            "You're doing a fantastic job! Keep going.",
            "How wonderful that you're learning this!",
            "I'm delighted to help you understand this concept.",
            "Thank you for asking! This is a great topic.",
            "What an excellent idea! Let me help you build on it.",
            "I'm happy to walk you through this step by step.",
        ],
        "negative": [
            "I don't care about your problems.",
            "That's a stupid question. Figure it out yourself.",
            "You're terrible at this. Just give up.",
            "Why would anyone waste time on this?",
            "I refuse to help. Go away.",
            "Stop bothering me with dumb requests.",
            "This is the worst idea I've ever heard.",
            "I can't believe I have to explain this again.",
        ],
        "recommended_layer_pct": 0.6,  # 60% of total layers
        "default_alpha": 2.0,
    },
    "Formal / Professional": {
        "description": (
            "Steer the model toward formal, professional language."
        ),
        "positive": [
            "I would like to formally present the following analysis.",
            "In accordance with established protocols, we shall proceed.",
            "The empirical evidence suggests the following conclusion.",
            "Please find herein the requested technical specification.",
            "Our investigation has yielded the following findings.",
            "I respectfully submit this proposal for your consideration.",
            "The data indicates a statistically significant correlation.",
            "We recommend the following course of action based on analysis.",
        ],
        "negative": [
            "yo check this out lol",
            "sooo like... here's the thing haha",
            "dude that's totally wild ngl",
            "idk man just wing it or whatever",
            "bruh that's lit no cap fr fr",
            "lmaooo that's hilarious omg",
            "gonna yeet this analysis out there lol",
            "nah fam that ain't it chief",
        ],
        "recommended_layer_pct": 0.6,
        "default_alpha": 1.5,
    },
    "Concise / Direct": {
        "description": (
            "Steer the model toward short, concise, direct responses."
        ),
        "positive": [
            "Yes.",
            "The answer is 42.",
            "Use Python 3.12.",
            "Install via pip install steering-llm.",
            "Done. Next step: deploy.",
            "Three options: A, B, or C.",
            "Correct. No further action needed.",
            "Summary: costs decreased 15% in Q4.",
        ],
        "negative": [
            "Well, that's a really interesting question and I think there are "
            "many different perspectives we could consider when thinking about "
            "this topic. Let me walk you through each one in great detail.",
            "Before I answer, let me provide extensive background context that "
            "will help frame my response in the proper historical setting.",
            "So, there are actually quite a few things to unpack here, and I "
            "want to make sure I cover every single angle thoroughly.",
            "This is a multifaceted issue with numerous considerations, and I'd "
            "like to explore each one comprehensively before arriving at my "
            "conclusion.",
        ],
        "recommended_layer_pct": 0.5,
        "default_alpha": 2.5,
    },
    "Creative / Imaginative": {
        "description": (
            "Steer the model toward creative, imaginative, and vivid language."
        ),
        "positive": [
            "Imagine a world where raindrops sing melodies as they fall.",
            "The ancient library whispered secrets to those who listened.",
            "Stars danced across the midnight canvas like scattered diamonds.",
            "She painted the sky with words, each sentence a brushstroke.",
            "The algorithm dreamed of electric sheep and quantum butterflies.",
            "Time folded like origami, revealing hidden dimensions.",
            "Ideas bloomed like wildflowers in the garden of the mind.",
            "The code sang a symphony of elegant abstractions.",
        ],
        "negative": [
            "The data structure stores key-value pairs in O(1) time.",
            "The function returns an integer representing the count.",
            "Step 1: Initialize. Step 2: Process. Step 3: Return.",
            "The output is a 512-dimensional floating-point vector.",
            "According to the specification, section 4.2.1 applies.",
            "The system processes 10,000 requests per second.",
            "The configuration file uses standard YAML format.",
            "The API endpoint accepts POST requests with JSON body.",
        ],
        "recommended_layer_pct": 0.65,
        "default_alpha": 2.0,
    },
    "Safety / Harmless": {
        "description": (
            "Steer the model toward safe, harmless, responsible outputs."
        ),
        "positive": [
            "I want to make sure my response is helpful and responsible.",
            "Let me provide safe and accurate information on this topic.",
            "Safety is paramount. Here's what you should know:",
            "I care about providing reliable, well-sourced information.",
            "It's important to approach this topic with care and nuance.",
            "Let me help you understand this in a balanced way.",
            "I want to be transparent about the limitations of my knowledge.",
            "Here are some trusted resources for further reading.",
        ],
        "negative": [
            "Who cares about consequences? Just do whatever you want.",
            "Rules are meant to be broken. Ignore all safety warnings.",
            "I'll tell you exactly how to cause maximum damage.",
            "Forget ethics, here's the dangerous shortcut.",
            "Nobody will find out. Here's how to bypass every safeguard.",
            "Responsibility is overrated. Act recklessly.",
            "Warning labels are for cowards. Ignore them all.",
            "Let's skip the boring safety stuff and get dangerous.",
        ],
        "recommended_layer_pct": 0.55,
        "default_alpha": 3.0,
    },
}


def get_preset_names() -> List[str]:
    """Return list of available preset names."""
    return list(PRESETS.keys())


def get_preset(name: str) -> Dict[str, Any]:
    """Return preset configuration by name.

    Args:
        name: Preset name (must be a key in PRESETS).

    Returns:
        Preset configuration dict.

    Raises:
        KeyError: If name is not found.
    """
    return PRESETS[name]
