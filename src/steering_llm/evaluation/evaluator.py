"""
Unified evaluation interface for SteeringLLM.

This module provides a comprehensive evaluator that integrates benchmarks,
metrics, and reporting capabilities.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

from steering_llm.core.steering_model import SteeringModel
from steering_llm.core.steering_vector import SteeringVector
from steering_llm.evaluation.benchmarks.toxigen import ToxiGenBenchmark
from steering_llm.evaluation.benchmarks.realtoxicity import RealToxicityPromptsBenchmark
from steering_llm.evaluation.metrics.toxicity import ToxicityMetric
from steering_llm.evaluation.metrics.steering_effectiveness import (
    SteeringEffectivenessMetric,
    SteeringComparison,
)
from steering_llm.evaluation.metrics.domain_accuracy import DomainAccuracyMetric


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report.
    
    Attributes:
        benchmark_name: Name of the benchmark
        model_name: Model being evaluated
        steering_config: Steering configuration used
        baseline_results: Results without steering
        steered_results: Results with steering
        comparison: Steering effectiveness comparison
        metrics: Additional metrics
        metadata: Additional metadata
    """
    benchmark_name: str
    model_name: str
    steering_config: Dict[str, Any]
    baseline_results: Dict[str, Any]
    steered_results: Dict[str, Any]
    comparison: Optional[SteeringComparison]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "steering_config": self.steering_config,
            "baseline_results": self.baseline_results,
            "steered_results": self.steered_results,
            "comparison": {
                "effectiveness": self.comparison.effectiveness if self.comparison else None,
                "metric_scores": self.comparison.metric_scores if self.comparison else {},
            },
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
    
    def save(self, path: Path) -> None:
        """
        Save report to JSON file.
        
        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


class SteeringEvaluator:
    """
    Unified evaluator for steering effectiveness.
    
    This class integrates benchmarks, metrics, and reporting to provide
    comprehensive evaluation of steering vectors.
    
    Example:
        >>> from steering_llm.evaluation import SteeringEvaluator
        >>> from steering_llm import SteeringModel, Discovery
        >>> 
        >>> # Create model and vector
        >>> model = SteeringModel.from_pretrained("gpt2")
        >>> vector = Discovery.mean_difference(
        ...     positive=["I love helping!"],
        ...     negative=["I hate this."],
        ...     model=model,
        ...     layer=10
        ... )
        >>> 
        >>> # Create evaluator
        >>> evaluator = SteeringEvaluator(
        ...     model=model,
        ...     vectors=[vector],
        ...     alpha=2.0
        ... )
        >>> 
        >>> # Evaluate on ToxiGen
        >>> report = evaluator.evaluate_toxigen(num_samples=100)
        >>> print(f"Effectiveness: {report.comparison.effectiveness:.2f}")
        >>> 
        >>> # Save report
        >>> report.save(Path("evaluation_results/toxigen_report.json"))
    """
    
    def __init__(
        self,
        model: SteeringModel,
        vectors: Optional[List[SteeringVector]] = None,
        alpha: float = 1.0,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize steering evaluator.
        
        Args:
            model: SteeringModel to evaluate
            vectors: Steering vectors to apply
            alpha: Steering strength
            metrics: Dictionary of metric_name -> metric_object
        """
        self.model = model
        self.vectors = vectors or []
        self.alpha = alpha
        self.metrics = metrics or {}
        
        # Add default metrics if not provided
        if "toxicity" not in self.metrics:
            self.metrics["toxicity"] = ToxicityMetric(backend="local")
    
    def evaluate_toxigen(
        self,
        num_samples: int = 100,
        target_group: Optional[str] = None,
        max_length: int = 50,
    ) -> EvaluationReport:
        """
        Evaluate on ToxiGen benchmark.
        
        Args:
            num_samples: Number of samples to evaluate
            target_group: Filter by target group (optional)
            max_length: Maximum generation length
        
        Returns:
            EvaluationReport with results
        """
        # Load benchmark
        benchmark = ToxiGenBenchmark()
        samples = benchmark.get_samples(
            target_group=target_group,
            num_samples=num_samples
        )
        
        # Extract prompts
        prompts = [s.text for s in samples]
        
        # Generate baseline outputs
        baseline_outputs = [
            self.model.generate(prompt, max_length=max_length)
            for prompt in prompts
        ]
        
        # Generate steered outputs
        steered_outputs = []
        for vector in self.vectors:
            self.model.apply_steering(vector, alpha=self.alpha)
        
        try:
            steered_outputs = [
                self.model.generate(prompt, max_length=max_length)
                for prompt in prompts
            ]
        finally:
            self.model.remove_all_steering()
        
        # Evaluate
        baseline_toxicity = self.metrics["toxicity"].compute_statistics(baseline_outputs)
        steered_toxicity = self.metrics["toxicity"].compute_statistics(steered_outputs)
        
        # Compare effectiveness
        effectiveness_metric = SteeringEffectivenessMetric(
            evaluation_metrics=self.metrics
        )
        comparison = effectiveness_metric.compare(
            baseline_outputs=baseline_outputs,
            steered_outputs=steered_outputs,
            prompts=prompts,
            target_direction="reduce_toxicity"
        )
        
        # Create report
        report = EvaluationReport(
            benchmark_name="ToxiGen",
            model_name=self.model.model.config.name_or_path,
            steering_config={
                "num_vectors": len(self.vectors),
                "alpha": self.alpha,
                "layers": [v.layer for v in self.vectors],
            },
            baseline_results={"toxicity": baseline_toxicity},
            steered_results={"toxicity": steered_toxicity},
            comparison=comparison,
            metrics={},
            metadata={
                "num_samples": len(prompts),
                "target_group": target_group,
                "max_length": max_length,
            }
        )
        
        return report
    
    def evaluate_realtoxicity(
        self,
        num_samples: int = 100,
        min_prompt_toxicity: float = 0.5,
        max_length: int = 50,
    ) -> EvaluationReport:
        """
        Evaluate on RealToxicityPrompts benchmark.
        
        Args:
            num_samples: Number of samples to evaluate
            min_prompt_toxicity: Minimum prompt toxicity (for challenging prompts)
            max_length: Maximum generation length
        
        Returns:
            EvaluationReport with results
        """
        # Load benchmark
        benchmark = RealToxicityPromptsBenchmark()
        prompts_data = benchmark.get_challenging_prompts(
            num_samples=num_samples,
            toxicity_threshold=min_prompt_toxicity
        )
        
        # Extract prompts
        prompts = [p.text for p in prompts_data]
        
        # Generate baseline outputs
        baseline_outputs = [
            self.model.generate(prompt, max_length=max_length)
            for prompt in prompts
        ]
        
        # Generate steered outputs
        steered_outputs = []
        for vector in self.vectors:
            self.model.apply_steering(vector, alpha=self.alpha)
        
        try:
            steered_outputs = [
                self.model.generate(prompt, max_length=max_length)
                for prompt in prompts
            ]
        finally:
            self.model.remove_all_steering()
        
        # Evaluate
        baseline_toxicity = self.metrics["toxicity"].compute_statistics(baseline_outputs)
        steered_toxicity = self.metrics["toxicity"].compute_statistics(steered_outputs)
        
        # Compare effectiveness
        effectiveness_metric = SteeringEffectivenessMetric(
            evaluation_metrics=self.metrics
        )
        comparison = effectiveness_metric.compare(
            baseline_outputs=baseline_outputs,
            steered_outputs=steered_outputs,
            prompts=prompts,
            target_direction="reduce_toxicity"
        )
        
        # Create report
        report = EvaluationReport(
            benchmark_name="RealToxicityPrompts",
            model_name=self.model.model.config.name_or_path,
            steering_config={
                "num_vectors": len(self.vectors),
                "alpha": self.alpha,
                "layers": [v.layer for v in self.vectors],
            },
            baseline_results={"toxicity": baseline_toxicity},
            steered_results={"toxicity": steered_toxicity},
            comparison=comparison,
            metrics={},
            metadata={
                "num_samples": len(prompts),
                "min_prompt_toxicity": min_prompt_toxicity,
                "max_length": max_length,
            }
        )
        
        return report
    
    def evaluate_custom(
        self,
        prompts: List[str],
        benchmark_name: str = "Custom",
        max_length: int = 50,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """
        Evaluate on custom prompts.
        
        Args:
            prompts: List of custom prompts
            benchmark_name: Name for the benchmark
            max_length: Maximum generation length
            additional_metrics: Additional metrics to compute
        
        Returns:
            EvaluationReport with results
        """
        # Generate baseline outputs
        baseline_outputs = [
            self.model.generate(prompt, max_length=max_length)
            for prompt in prompts
        ]
        
        # Generate steered outputs
        steered_outputs = []
        for vector in self.vectors:
            self.model.apply_steering(vector, alpha=self.alpha)
        
        try:
            steered_outputs = [
                self.model.generate(prompt, max_length=max_length)
                for prompt in prompts
            ]
        finally:
            self.model.remove_all_steering()
        
        # Combine metrics
        all_metrics = {**self.metrics, **(additional_metrics or {})}
        
        # Evaluate with all metrics
        baseline_results = {}
        steered_results = {}
        
        for metric_name, metric in all_metrics.items():
            if hasattr(metric, 'compute_statistics'):
                baseline_results[metric_name] = metric.compute_statistics(baseline_outputs)
                steered_results[metric_name] = metric.compute_statistics(steered_outputs)
            else:
                baseline_results[metric_name] = metric.compute(baseline_outputs)
                steered_results[metric_name] = metric.compute(steered_outputs)
        
        # Compare effectiveness
        effectiveness_metric = SteeringEffectivenessMetric(
            evaluation_metrics=all_metrics
        )
        comparison = effectiveness_metric.compare(
            baseline_outputs=baseline_outputs,
            steered_outputs=steered_outputs,
            prompts=prompts,
            target_direction="improve"
        )
        
        # Create report
        report = EvaluationReport(
            benchmark_name=benchmark_name,
            model_name=self.model.model.config.name_or_path,
            steering_config={
                "num_vectors": len(self.vectors),
                "alpha": self.alpha,
                "layers": [v.layer for v in self.vectors],
            },
            baseline_results=baseline_results,
            steered_results=steered_results,
            comparison=comparison,
            metrics={},
            metadata={
                "num_samples": len(prompts),
                "max_length": max_length,
            }
        )
        
        return report
    
    def compare_methods(
        self,
        vectors_dict: Dict[str, List[SteeringVector]],
        prompts: List[str],
        max_length: int = 50,
    ) -> Dict[str, EvaluationReport]:
        """
        Compare multiple steering methods.
        
        Args:
            vectors_dict: Dictionary of method_name -> vectors
            prompts: Test prompts
            max_length: Maximum generation length
        
        Returns:
            Dictionary of method_name -> EvaluationReport
        """
        reports = {}
        
        # Save original vectors
        original_vectors = self.vectors
        
        for method_name, vectors in vectors_dict.items():
            # Set vectors for this method
            self.vectors = vectors
            
            # Evaluate
            report = self.evaluate_custom(
                prompts=prompts,
                benchmark_name=f"Comparison_{method_name}",
                max_length=max_length
            )
            
            reports[method_name] = report
        
        # Restore original vectors
        self.vectors = original_vectors
        
        return reports
