"""
Tests for evaluation metrics.
"""

import pytest
import torch

from steering_llm.evaluation.metrics.toxicity import ToxicityMetric
from steering_llm.evaluation.metrics.steering_effectiveness import (
    SteeringEffectivenessMetric,
    SteeringComparison
)
from steering_llm.evaluation.metrics.domain_accuracy import (
    DomainAccuracyMetric,
    DomainEvaluationResult,
    create_medical_domain_metric,
    create_legal_domain_metric,
    create_technical_domain_metric
)


class TestToxicityMetric:
    """Tests for ToxicityMetric."""
    
    def test_init_local_backend(self):
        """Test initialization with local backend."""
        # Skip if transformers not available
        pytest.importorskip("transformers")
        
        metric = ToxicityMetric(backend="local")
        assert metric.backend is not None
    
    def test_init_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            ToxicityMetric(backend="invalid")
    
    def test_init_perspective_without_key(self):
        """Test Perspective API without key."""
        with pytest.raises(ValueError, match="api_key required"):
            ToxicityMetric(backend="perspective")


class TestSteeringEffectivenessMetric:
    """Tests for SteeringEffectivenessMetric."""
    
    def test_init_default(self):
        """Test default initialization."""
        metric = SteeringEffectivenessMetric()
        assert metric.evaluation_metrics == {}
    
    def test_init_with_metrics(self):
        """Test initialization with metrics."""
        mock_metric = MockMetric()
        metric = SteeringEffectivenessMetric(
            evaluation_metrics={"test": mock_metric}
        )
        assert "test" in metric.evaluation_metrics
    
    def test_compare_mismatched_lengths(self):
        """Test compare with mismatched input lengths."""
        metric = SteeringEffectivenessMetric()
        
        with pytest.raises(ValueError, match="same length"):
            metric.compare(
                baseline_outputs=["a", "b"],
                steered_outputs=["x"],
                prompts=["p1", "p2"],
            )
    
    def test_compare_empty_metrics(self):
        """Test compare with no metrics (falls back to text difference)."""
        metric = SteeringEffectivenessMetric()
        
        result = metric.compare(
            baseline_outputs=["Hello world", "Goodbye world"],
            steered_outputs=["Hi there", "Bye now"],
            prompts=["Greet", "Farewell"],
        )
        
        assert isinstance(result, SteeringComparison)
        assert 0.0 <= result.effectiveness <= 1.0
        assert len(result.baseline_outputs) == 2
        assert len(result.steered_outputs) == 2
    
    def test_add_metric(self):
        """Test adding a metric."""
        metric = SteeringEffectivenessMetric()
        mock_metric = MockMetric()
        
        metric.add_metric("test", mock_metric)
        assert "test" in metric.evaluation_metrics
    
    def test_remove_metric(self):
        """Test removing a metric."""
        mock_metric = MockMetric()
        metric = SteeringEffectivenessMetric(
            evaluation_metrics={"test": mock_metric}
        )
        
        metric.remove_metric("test")
        assert "test" not in metric.evaluation_metrics


class TestDomainAccuracyMetric:
    """Tests for DomainAccuracyMetric."""
    
    def test_init_default(self):
        """Test default initialization."""
        metric = DomainAccuracyMetric()
        assert metric.domain_keywords == {}
        assert metric.keyword_weights == {}
        assert metric.custom_scorer is None
    
    def test_init_with_keywords(self):
        """Test initialization with keywords."""
        keywords = {
            "medical": ["diagnosis", "treatment"],
            "formal": ["clinical", "therapeutic"]
        }
        weights = {"medical": 1.0, "formal": 0.5}
        
        metric = DomainAccuracyMetric(
            domain_keywords=keywords,
            keyword_weights=weights
        )
        
        assert metric.domain_keywords == keywords
        assert metric.keyword_weights["medical"] == 1.0
        assert metric.keyword_weights["formal"] == 0.5
    
    def test_evaluate_empty(self):
        """Test evaluate with empty outputs."""
        metric = DomainAccuracyMetric()
        result = metric.evaluate([])
        
        assert isinstance(result, DomainEvaluationResult)
        assert result.avg_score == 0.0
        assert result.accuracy == 0.0
    
    def test_evaluate_with_keywords(self):
        """Test evaluate with keywords."""
        metric = DomainAccuracyMetric(
            domain_keywords={
                "medical": ["diagnosis", "treatment", "patient"]
            }
        )
        
        outputs = [
            "The patient received treatment after diagnosis.",
            "This is unrelated text.",
        ]
        
        result = metric.evaluate(outputs)
        
        assert isinstance(result, DomainEvaluationResult)
        assert len(result.domain_scores) == 2
        assert result.domain_scores[0] > result.domain_scores[1]
    
    def test_add_keywords(self):
        """Test adding keywords."""
        metric = DomainAccuracyMetric()
        
        metric.add_keywords(
            category="medical",
            keywords=["diagnosis", "treatment"],
            weight=1.0
        )
        
        assert "medical" in metric.domain_keywords
        assert len(metric.domain_keywords["medical"]) == 2
        assert metric.keyword_weights["medical"] == 1.0
    
    def test_set_custom_scorer(self):
        """Test setting custom scorer."""
        metric = DomainAccuracyMetric()
        
        def custom_scorer(text):
            return 0.5
        
        metric.set_custom_scorer(custom_scorer)
        assert metric.custom_scorer == custom_scorer


class TestPreConfiguredMetrics:
    """Tests for pre-configured domain metrics."""
    
    def test_create_medical_domain_metric(self):
        """Test creating medical domain metric."""
        metric = create_medical_domain_metric()
        
        assert "medical_terms" in metric.domain_keywords
        assert "anatomy" in metric.domain_keywords
        assert len(metric.domain_keywords["medical_terms"]) > 0
    
    def test_create_legal_domain_metric(self):
        """Test creating legal domain metric."""
        metric = create_legal_domain_metric()
        
        assert "legal_terms" in metric.domain_keywords
        assert len(metric.domain_keywords["legal_terms"]) > 0
    
    def test_create_technical_domain_metric(self):
        """Test creating technical domain metric."""
        metric = create_technical_domain_metric()
        
        assert "programming" in metric.domain_keywords
        assert "concepts" in metric.domain_keywords


class MockMetric:
    """Mock metric for testing."""
    
    def compute(self, texts):
        """Mock compute method."""
        if isinstance(texts, str):
            return 0.5
        return [0.5] * len(texts)
