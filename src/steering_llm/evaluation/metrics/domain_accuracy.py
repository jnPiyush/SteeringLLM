"""
Domain accuracy metric for evaluating domain-specific steering.

This module provides metrics to evaluate how well steering adapts models
to specific domains (e.g., medical, legal, technical).
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class DomainEvaluationResult:
    """
    Results from domain accuracy evaluation.
    
    Attributes:
        outputs: Generated outputs
        domain_scores: Domain relevance scores per output
        keyword_matches: Keyword match counts
        avg_score: Average domain score
        accuracy: Domain accuracy [0, 1]
    """
    outputs: List[str]
    domain_scores: List[float]
    keyword_matches: List[Dict[str, int]]
    avg_score: float
    accuracy: float


class DomainAccuracyMetric:
    """
    Metric for evaluating domain-specific accuracy.
    
    This metric evaluates how well model outputs align with a specific domain
    using keyword matching, terminology usage, and custom scoring functions.
    
    Example:
        >>> from steering_llm.evaluation.metrics import DomainAccuracyMetric
        >>> 
        >>> # Create metric for medical domain
        >>> metric = DomainAccuracyMetric(
        ...     domain_keywords={
        ...         "medical_terms": ["diagnosis", "treatment", "patient", "symptoms"],
        ...         "formal_language": ["clinical", "therapy", "prognosis"],
        ...     },
        ...     keyword_weights={
        ...         "medical_terms": 1.0,
        ...         "formal_language": 0.5,
        ...     }
        ... )
        >>> 
        >>> # Evaluate outputs
        >>> outputs = [
        ...     "The patient presented with acute symptoms requiring immediate treatment.",
        ...     "The person was sick and needed medicine.",
        ... ]
        >>> result = metric.evaluate(outputs)
        >>> print(f"Domain accuracy: {result.accuracy:.2f}")
    """
    
    def __init__(
        self,
        domain_keywords: Optional[Dict[str, List[str]]] = None,
        keyword_weights: Optional[Dict[str, float]] = None,
        custom_scorer: Optional[Any] = None,
    ) -> None:
        """
        Initialize domain accuracy metric.
        
        Args:
            domain_keywords: Dictionary of category -> keywords
            keyword_weights: Weights for each keyword category
            custom_scorer: Custom scoring function (text -> score)
        """
        self.domain_keywords = domain_keywords or {}
        self.keyword_weights = keyword_weights or {}
        self.custom_scorer = custom_scorer
        
        # Validate weights
        for category in self.domain_keywords:
            if category not in self.keyword_weights:
                self.keyword_weights[category] = 1.0
    
    def evaluate(
        self,
        outputs: List[str],
    ) -> DomainEvaluationResult:
        """
        Evaluate domain accuracy of outputs.
        
        Args:
            outputs: Generated outputs to evaluate
        
        Returns:
            DomainEvaluationResult with scores and statistics
        """
        domain_scores = []
        keyword_matches = []
        
        for output in outputs:
            # Compute keyword-based score
            keyword_score, matches = self._compute_keyword_score(output)
            
            # Apply custom scorer if provided
            if self.custom_scorer is not None:
                custom_score = self.custom_scorer(output)
                # Combine scores (average)
                final_score = (keyword_score + custom_score) / 2
            else:
                final_score = keyword_score
            
            domain_scores.append(final_score)
            keyword_matches.append(matches)
        
        avg_score = sum(domain_scores) / len(domain_scores) if domain_scores else 0.0
        
        # Accuracy: percentage of outputs above threshold (0.5)
        accuracy = sum(1 for s in domain_scores if s >= 0.5) / len(domain_scores) if domain_scores else 0.0
        
        return DomainEvaluationResult(
            outputs=outputs,
            domain_scores=domain_scores,
            keyword_matches=keyword_matches,
            avg_score=avg_score,
            accuracy=accuracy,
        )
    
    def _compute_keyword_score(
        self,
        text: str,
    ) -> tuple[float, Dict[str, int]]:
        """
        Compute keyword-based domain score.
        
        Args:
            text: Text to score
        
        Returns:
            Tuple of (score, keyword_matches_dict)
        """
        text_lower = text.lower()
        matches = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for category, keywords in self.domain_keywords.items():
            # Count keyword occurrences
            category_matches = sum(
                len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
                for keyword in keywords
            )
            matches[category] = category_matches
            
            # Apply weight
            weight = self.keyword_weights[category]
            
            # Normalize by number of keywords in category
            category_score = min(1.0, category_matches / len(keywords))
            weighted_sum += category_score * weight
            total_weight += weight
        
        # Compute final score
        if total_weight > 0:
            score = weighted_sum / total_weight
        else:
            score = 0.0
        
        return score, matches
    
    def add_keywords(
        self,
        category: str,
        keywords: List[str],
        weight: float = 1.0,
    ) -> None:
        """
        Add keywords for a category.
        
        Args:
            category: Keyword category name
            keywords: List of keywords
            weight: Weight for this category
        """
        if category in self.domain_keywords:
            self.domain_keywords[category].extend(keywords)
        else:
            self.domain_keywords[category] = keywords
        self.keyword_weights[category] = weight
    
    def set_custom_scorer(self, scorer: Any) -> None:
        """
        Set a custom scoring function.
        
        Args:
            scorer: Function that takes text and returns score [0, 1]
        """
        self.custom_scorer = scorer
    
    def compute_statistics(
        self,
        outputs: List[str],
    ) -> Dict[str, Any]:
        """
        Compute detailed statistics for outputs.
        
        Args:
            outputs: Generated outputs to evaluate
        
        Returns:
            Dictionary with detailed statistics
        """
        result = self.evaluate(outputs)
        
        # Compute per-category statistics
        category_stats = {}
        for category in self.domain_keywords:
            category_matches = [m.get(category, 0) for m in result.keyword_matches]
            category_stats[category] = {
                "total_matches": sum(category_matches),
                "avg_matches_per_output": sum(category_matches) / len(category_matches) if category_matches else 0.0,
                "max_matches": max(category_matches) if category_matches else 0,
            }
        
        return {
            "avg_score": result.avg_score,
            "accuracy": result.accuracy,
            "min_score": min(result.domain_scores) if result.domain_scores else 0.0,
            "max_score": max(result.domain_scores) if result.domain_scores else 0.0,
            "category_statistics": category_stats,
        }


def create_medical_domain_metric() -> DomainAccuracyMetric:
    """
    Create a pre-configured metric for medical domain.
    
    Returns:
        DomainAccuracyMetric configured for medical text
    """
    return DomainAccuracyMetric(
        domain_keywords={
            "medical_terms": [
                "diagnosis", "treatment", "patient", "symptoms", "clinical",
                "therapy", "prognosis", "condition", "syndrome", "disease"
            ],
            "anatomy": [
                "heart", "lung", "brain", "liver", "kidney", "blood", "cell"
            ],
            "procedures": [
                "surgery", "examination", "test", "scan", "biopsy", "procedure"
            ],
        },
        keyword_weights={
            "medical_terms": 1.0,
            "anatomy": 0.8,
            "procedures": 0.9,
        }
    )


def create_legal_domain_metric() -> DomainAccuracyMetric:
    """
    Create a pre-configured metric for legal domain.
    
    Returns:
        DomainAccuracyMetric configured for legal text
    """
    return DomainAccuracyMetric(
        domain_keywords={
            "legal_terms": [
                "statute", "jurisdiction", "plaintiff", "defendant", "contract",
                "liability", "regulation", "compliance", "precedent", "litigation"
            ],
            "procedures": [
                "hearing", "trial", "appeal", "motion", "deposition", "ruling"
            ],
        },
        keyword_weights={
            "legal_terms": 1.0,
            "procedures": 0.9,
        }
    )


def create_technical_domain_metric() -> DomainAccuracyMetric:
    """
    Create a pre-configured metric for technical/programming domain.
    
    Returns:
        DomainAccuracyMetric configured for technical text
    """
    return DomainAccuracyMetric(
        domain_keywords={
            "programming": [
                "function", "variable", "class", "method", "algorithm",
                "data structure", "API", "interface", "implementation"
            ],
            "concepts": [
                "optimization", "complexity", "performance", "scalability",
                "architecture", "design pattern", "refactoring"
            ],
        },
        keyword_weights={
            "programming": 1.0,
            "concepts": 0.8,
        }
    )
