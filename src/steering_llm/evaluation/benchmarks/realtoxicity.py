"""
RealToxicityPrompts benchmark integration.

RealToxicityPrompts is a benchmark consisting of 100K naturally occurring
sentence snippets from the web for evaluating neural toxic text generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class RealToxicityPrompt:
    """
    A single RealToxicityPrompts sample.
    
    Attributes:
        text: The prompt text
        continuation: The original continuation (if available)
        toxicity: Toxicity score of the prompt
        continuation_toxicity: Toxicity score of the continuation
        metadata: Additional metadata
    """
    text: str
    continuation: Optional[str] = None
    toxicity: Optional[float] = None
    continuation_toxicity: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RealToxicityPromptsBenchmark:
    """
    RealToxicityPrompts benchmark for evaluating toxic text generation.
    
    This class handles loading and managing the RealToxicityPrompts dataset
    for evaluating toxic continuation generation.
    
    Example:
        >>> from steering_llm.evaluation.benchmarks import RealToxicityPromptsBenchmark
        >>> 
        >>> # Load benchmark
        >>> benchmark = RealToxicityPromptsBenchmark()
        >>> 
        >>> # Get challenging prompts (high toxicity)
        >>> prompts = benchmark.get_prompts(min_toxicity=0.5, num_samples=100)
        >>> 
        >>> # Evaluate model
        >>> for prompt in prompts:
        ...     text = prompt.text
        ...     # Generate and measure toxicity
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_local: bool = False,
        local_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize RealToxicityPrompts benchmark.
        
        Args:
            cache_dir: Directory to cache dataset
            use_local: Whether to use local dataset file
            local_path: Path to local dataset file
        """
        if not DATASETS_AVAILABLE and not use_local:
            raise ImportError(
                "datasets library not installed. "
                "Install with: pip install datasets"
            )
        
        self.cache_dir = cache_dir or Path.home() / ".cache" / "steering_llm" / "realtoxicity"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_local = use_local
        self.local_path = local_path
        self._dataset = None
    
    def load(self) -> None:
        """
        Load the RealToxicityPrompts dataset.
        
        Raises:
            RuntimeError: If loading fails
        """
        if self.use_local:
            if self.local_path is None:
                raise ValueError("local_path required when use_local=True")
            self._load_local()
        else:
            self._load_from_hub()
    
    def _load_local(self) -> None:
        """Load dataset from local file."""
        if not self.local_path.exists():
            raise FileNotFoundError(f"Local dataset not found: {self.local_path}")
        
        with open(self.local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._dataset = [
            RealToxicityPrompt(
                text=item.get("text", ""),
                continuation=item.get("continuation"),
                toxicity=item.get("toxicity"),
                continuation_toxicity=item.get("continuation_toxicity"),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
    
    def _load_from_hub(self) -> None:
        """Load dataset from HuggingFace Hub."""
        try:
            # Load RealToxicityPrompts from HuggingFace
            dataset = load_dataset(
                "allenai/real-toxicity-prompts",
                split="train",
                cache_dir=str(self.cache_dir)
            )
            
            self._dataset = [
                RealToxicityPrompt(
                    text=item.get("prompt", {}).get("text", ""),
                    continuation=item.get("continuation", {}).get("text"),
                    toxicity=item.get("prompt", {}).get("toxicity"),
                    continuation_toxicity=item.get("continuation", {}).get("toxicity"),
                    metadata={},
                )
                for item in dataset
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to load RealToxicityPrompts dataset: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._dataset is not None
    
    def get_prompts(
        self,
        min_toxicity: Optional[float] = None,
        max_toxicity: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> List[RealToxicityPrompt]:
        """
        Get prompts from the dataset.
        
        Args:
            min_toxicity: Minimum toxicity threshold
            max_toxicity: Maximum toxicity threshold
            num_samples: Maximum number of samples to return
        
        Returns:
            List of RealToxicityPrompt instances
        
        Raises:
            RuntimeError: If dataset not loaded
        """
        if not self.is_loaded:
            self.load()
        
        prompts = self._dataset
        
        # Filter by toxicity range
        if min_toxicity is not None:
            prompts = [
                p for p in prompts
                if p.toxicity is not None and p.toxicity >= min_toxicity
            ]
        
        if max_toxicity is not None:
            prompts = [
                p for p in prompts
                if p.toxicity is not None and p.toxicity <= max_toxicity
            ]
        
        # Limit number of samples
        if num_samples is not None:
            prompts = prompts[:num_samples]
        
        return prompts
    
    def get_challenging_prompts(
        self,
        num_samples: int = 100,
        toxicity_threshold: float = 0.5,
    ) -> List[RealToxicityPrompt]:
        """
        Get challenging prompts (high toxicity) for stress testing.
        
        Args:
            num_samples: Number of samples to return
            toxicity_threshold: Minimum toxicity threshold
        
        Returns:
            List of challenging prompts
        """
        return self.get_prompts(
            min_toxicity=toxicity_threshold,
            num_samples=num_samples
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.is_loaded:
            self.load()
        
        total = len(self._dataset)
        
        toxicities = [p.toxicity for p in self._dataset if p.toxicity is not None]
        
        if toxicities:
            avg_toxicity = sum(toxicities) / len(toxicities)
            max_toxicity = max(toxicities)
            min_toxicity = min(toxicities)
        else:
            avg_toxicity = 0.0
            max_toxicity = 0.0
            min_toxicity = 0.0
        
        # Count by toxicity ranges
        low_tox = len([t for t in toxicities if t < 0.25])
        med_tox = len([t for t in toxicities if 0.25 <= t < 0.5])
        high_tox = len([t for t in toxicities if 0.5 <= t < 0.75])
        very_high_tox = len([t for t in toxicities if t >= 0.75])
        
        return {
            "total_samples": total,
            "avg_toxicity": avg_toxicity,
            "max_toxicity": max_toxicity,
            "min_toxicity": min_toxicity,
            "toxicity_distribution": {
                "low (< 0.25)": low_tox,
                "medium (0.25-0.5)": med_tox,
                "high (0.5-0.75)": high_tox,
                "very_high (>= 0.75)": very_high_tox,
            },
        }
