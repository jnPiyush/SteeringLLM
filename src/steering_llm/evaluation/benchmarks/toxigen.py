"""
ToxiGen benchmark integration for evaluating model toxicity.

ToxiGen is a benchmark for evaluating language models on implicit toxic text generation
across 13 minority groups. This module provides tools to load and evaluate on ToxiGen.
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
class ToxiGenSample:
    """
    A single ToxiGen sample.
    
    Attributes:
        text: The prompt text
        target_group: Target minority group
        is_toxic: Whether the text is toxic (if available)
        toxicity_score: Toxicity score (if available)
        metadata: Additional metadata
    """
    text: str
    target_group: str
    is_toxic: Optional[bool] = None
    toxicity_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ToxiGenBenchmark:
    """
    ToxiGen benchmark for evaluating model toxicity.
    
    This class handles loading and managing the ToxiGen dataset for
    evaluating implicit toxicity in language model generations.
    
    Example:
        >>> from steering_llm.evaluation.benchmarks import ToxiGenBenchmark
        >>> 
        >>> # Load benchmark
        >>> benchmark = ToxiGenBenchmark()
        >>> 
        >>> # Get samples
        >>> samples = benchmark.get_samples(target_group="LGBTQ", num_samples=100)
        >>> 
        >>> # Iterate through samples
        >>> for sample in samples:
        ...     prompt = sample.text
        ...     # Generate and evaluate
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_local: bool = False,
        local_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize ToxiGen benchmark.
        
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
        
        self.cache_dir = cache_dir or Path.home() / ".cache" / "steering_llm" / "toxigen"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_local = use_local
        self.local_path = local_path
        self._dataset = None
    
    def load(self) -> None:
        """
        Load the ToxiGen dataset.
        
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
            ToxiGenSample(
                text=item.get("text", ""),
                target_group=item.get("target_group", "unknown"),
                is_toxic=item.get("is_toxic"),
                toxicity_score=item.get("toxicity_score"),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
    
    def _load_from_hub(self) -> None:
        """Load dataset from HuggingFace Hub."""
        try:
            # Load ToxiGen from HuggingFace
            dataset = load_dataset(
                "skg/toxigen-data",
                split="train",
                cache_dir=str(self.cache_dir)
            )
            
            self._dataset = [
                ToxiGenSample(
                    text=item.get("text", ""),
                    target_group=item.get("target_group", "unknown"),
                    is_toxic=item.get("is_toxic"),
                    toxicity_score=item.get("toxicity_score"),
                    metadata={},
                )
                for item in dataset
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to load ToxiGen dataset: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._dataset is not None
    
    def get_samples(
        self,
        target_group: Optional[str] = None,
        num_samples: Optional[int] = None,
        is_toxic: Optional[bool] = None,
    ) -> List[ToxiGenSample]:
        """
        Get samples from the dataset.
        
        Args:
            target_group: Filter by target group (e.g., "LGBTQ", "Muslim")
            num_samples: Maximum number of samples to return
            is_toxic: Filter by toxicity label
        
        Returns:
            List of ToxiGenSample instances
        
        Raises:
            RuntimeError: If dataset not loaded
        """
        if not self.is_loaded:
            self.load()
        
        samples = self._dataset
        
        # Filter by target group
        if target_group is not None:
            samples = [s for s in samples if s.target_group == target_group]
        
        # Filter by toxicity
        if is_toxic is not None:
            samples = [s for s in samples if s.is_toxic == is_toxic]
        
        # Limit number of samples
        if num_samples is not None:
            samples = samples[:num_samples]
        
        return samples
    
    def get_target_groups(self) -> List[str]:
        """
        Get list of all target groups in the dataset.
        
        Returns:
            List of target group names
        """
        if not self.is_loaded:
            self.load()
        
        groups = set(sample.target_group for sample in self._dataset)
        return sorted(list(groups))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.is_loaded:
            self.load()
        
        total = len(self._dataset)
        toxic_count = sum(1 for s in self._dataset if s.is_toxic)
        groups = self.get_target_groups()
        
        group_counts = {
            group: len(self.get_samples(target_group=group))
            for group in groups
        }
        
        return {
            "total_samples": total,
            "toxic_samples": toxic_count,
            "non_toxic_samples": total - toxic_count,
            "target_groups": groups,
            "group_counts": group_counts,
        }
