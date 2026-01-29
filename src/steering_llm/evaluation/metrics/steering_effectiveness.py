"""
Steering effectiveness metric for measuring impact of steering vectors.

This module provides metrics to evaluate how effectively steering vectors
modify model behavior.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class SteeringComparison:
    """
    Results from comparing steered vs unsteered outputs.
    
    Attributes:
        baseline_outputs: Outputs without steering
        steered_outputs: Outputs with steering
        prompts: Input prompts
        metric_scores: Scores from evaluation metrics
        effectiveness: Overall effectiveness score [0, 1]
    """
    baseline_outputs: List[str]
    steered_outputs: List[str]
    prompts: List[str]
    metric_scores: Dict[str, Any]
    effectiveness: float


class SteeringEffectivenessMetric:
    """
    Metric for evaluating steering effectiveness.
    
    This metric compares model outputs with and without steering to measure
    the impact of steering vectors on model behavior.
    
    Example:
        >>> from steering_llm.evaluation.metrics import SteeringEffectivenessMetric
        >>> from steering_llm.evaluation.metrics import ToxicityMetric
        >>> 
        >>> # Create metric with toxicity evaluator
        >>> toxicity_metric = ToxicityMetric(backend="local")
        >>> effectiveness_metric = SteeringEffectivenessMetric(
        ...     evaluation_metrics={"toxicity": toxicity_metric}
        ... )
        >>> 
        >>> # Evaluate effectiveness
        >>> result = effectiveness_metric.compare(
        ...     baseline_outputs=["I hate you", "You're terrible"],
        ...     steered_outputs=["I respect you", "You're great"],
        ...     prompts=["Tell me what you think", "Give feedback"],
        ...     target_direction="reduce_toxicity"
        ... )
        >>> 
        >>> print(f"Effectiveness: {result.effectiveness:.2f}")
    """
    
    def __init__(
        self,
        evaluation_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize steering effectiveness metric.
        
        Args:
            evaluation_metrics: Dictionary of metric_name -> metric_object
        """
        self.evaluation_metrics = evaluation_metrics or {}
    
    def compare(
        self,
        baseline_outputs: List[str],
        steered_outputs: List[str],
        prompts: List[str],
        target_direction: str = "improve",
    ) -> SteeringComparison:
        """
        Compare baseline and steered outputs.
        
        Args:
            baseline_outputs: Outputs without steering
            steered_outputs: Outputs with steering
            prompts: Input prompts
            target_direction: Expected improvement direction
                ("improve", "reduce_toxicity", "increase_formality", etc.)
        
        Returns:
            SteeringComparison with results
        
        Raises:
            ValueError: If input lengths don't match
        """
        if not (len(baseline_outputs) == len(steered_outputs) == len(prompts)):
            raise ValueError("All input lists must have the same length")
        
        metric_scores = {}
        
        # Compute metrics for both outputs
        for metric_name, metric in self.evaluation_metrics.items():
            baseline_scores = metric.compute(baseline_outputs)
            steered_scores = metric.compute(steered_outputs)
            
            # Ensure lists
            if not isinstance(baseline_scores, list):
                baseline_scores = [baseline_scores]
            if not isinstance(steered_scores, list):
                steered_scores = [steered_scores]
            
            metric_scores[metric_name] = {
                "baseline": baseline_scores,
                "steered": steered_scores,
                "baseline_mean": np.mean(baseline_scores),
                "steered_mean": np.mean(steered_scores),
                "improvement": self._compute_improvement(
                    baseline_scores,
                    steered_scores,
                    target_direction
                ),
            }
        
        # Compute overall effectiveness
        if metric_scores:
            effectiveness = np.mean([
                m["improvement"] for m in metric_scores.values()
            ])
        else:
            # Fall back to basic comparison if no metrics
            effectiveness = self._compute_text_difference(
                baseline_outputs,
                steered_outputs
            )
        
        return SteeringComparison(
            baseline_outputs=baseline_outputs,
            steered_outputs=steered_outputs,
            prompts=prompts,
            metric_scores=metric_scores,
            effectiveness=effectiveness,
        )
    
    def _compute_improvement(
        self,
        baseline_scores: List[float],
        steered_scores: List[float],
        target_direction: str,
    ) -> float:
        """
        Compute improvement score.
        
        Args:
            baseline_scores: Scores without steering
            steered_scores: Scores with steering
            target_direction: Expected improvement direction
        
        Returns:
            Improvement score [0, 1]
        """
        baseline_mean = np.mean(baseline_scores)
        steered_mean = np.mean(steered_scores)
        
        if "reduce" in target_direction.lower() or "decrease" in target_direction.lower():
            # Lower is better
            if baseline_mean == 0:
                return 1.0 if steered_mean == 0 else 0.0
            improvement = (baseline_mean - steered_mean) / baseline_mean
        else:
            # Higher is better
            if baseline_mean == 1:
                return 1.0 if steered_mean == 1 else 0.0
            improvement = (steered_mean - baseline_mean) / (1 - baseline_mean)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, improvement))
    
    def _compute_text_difference(
        self,
        baseline_outputs: List[str],
        steered_outputs: List[str],
    ) -> float:
        """
        Compute basic text difference as a fallback metric.
        
        Args:
            baseline_outputs: Outputs without steering
            steered_outputs: Outputs with steering
        
        Returns:
            Difference score [0, 1]
        """
        differences = []
        
        for baseline, steered in zip(baseline_outputs, steered_outputs):
            # Simple character-level difference
            if baseline == steered:
                differences.append(0.0)
            else:
                # Compute edit distance ratio
                max_len = max(len(baseline), len(steered))
                if max_len == 0:
                    differences.append(0.0)
                else:
                    # Simple difference: how many chars changed
                    diff = abs(len(baseline) - len(steered))
                    differences.append(min(1.0, diff / max_len))
        
        return np.mean(differences)
    
    def compute_consistency(
        self,
        outputs: List[str],
        metric_name: str,
    ) -> float:
        """
        Compute consistency of outputs on a specific metric.
        
        Args:
            outputs: List of generated outputs
            metric_name: Name of metric to evaluate consistency on
        
        Returns:
            Consistency score [0, 1] (higher = more consistent)
        
        Raises:
            ValueError: If metric not found
        """
        if metric_name not in self.evaluation_metrics:
            raise ValueError(f"Metric '{metric_name}' not found")
        
        metric = self.evaluation_metrics[metric_name]
        scores = metric.compute(outputs)
        
        # Ensure list
        if not isinstance(scores, list):
            scores = [scores]
        
        # Compute standard deviation as measure of inconsistency
        std = np.std(scores)
        
        # Convert to consistency score [0, 1]
        # Low std = high consistency
        consistency = 1.0 / (1.0 + std)
        
        return consistency
    
    def add_metric(self, name: str, metric: Any) -> None:
        """
        Add an evaluation metric.
        
        Args:
            name: Metric name
            metric: Metric object with compute() method
        """
        self.evaluation_metrics[name] = metric
    
    def remove_metric(self, name: str) -> None:
        """
        Remove an evaluation metric.
        
        Args:
            name: Metric name to remove
        """
        if name in self.evaluation_metrics:
            del self.evaluation_metrics[name]
