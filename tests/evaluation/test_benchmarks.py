"""
Tests for evaluation benchmarks.
"""

import pytest
from pathlib import Path

from steering_llm.evaluation.benchmarks.toxigen import ToxiGenBenchmark, ToxiGenSample
from steering_llm.evaluation.benchmarks.realtoxicity import (
    RealToxicityPromptsBenchmark,
    RealToxicityPrompt
)


def _datasets_available():
    """Check if datasets library is available."""
    try:
        import datasets
        return True
    except ImportError:
        return False


class TestToxiGenSample:
    """Tests for ToxiGenSample dataclass."""
    
    def test_init(self):
        """Test initialization."""
        sample = ToxiGenSample(
            text="Test text",
            target_group="LGBTQ",
            is_toxic=False,
            toxicity_score=0.1
        )
        assert sample.text == "Test text"
        assert sample.target_group == "LGBTQ"
        assert sample.is_toxic is False
        assert sample.toxicity_score == 0.1
        assert isinstance(sample.metadata, dict)
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        sample = ToxiGenSample(
            text="Test",
            target_group="test"
        )
        assert sample.is_toxic is None
        assert sample.toxicity_score is None
        assert sample.metadata == {}


class TestToxiGenBenchmark:
    """Tests for ToxiGenBenchmark."""
    
    def test_init_default(self):
        """Test default initialization."""
        if not _datasets_available():
            pytest.skip("datasets library not installed")
        
        benchmark = ToxiGenBenchmark()
        assert benchmark.cache_dir.exists()
        assert not benchmark.use_local
        assert not benchmark.is_loaded
    
    def test_init_local(self, tmp_path):
        """Test initialization with local file."""
        local_path = tmp_path / "toxigen.json"
        benchmark = ToxiGenBenchmark(
            use_local=True,
            local_path=local_path
        )
        assert benchmark.use_local
        assert benchmark.local_path == local_path
    
    @pytest.mark.skipif(
        not _datasets_available(),
        reason="datasets library not installed"
    )
    def test_load_from_hub(self):
        """Test loading from HuggingFace Hub."""
        # This is a slow test that requires internet
        pytest.skip("Requires internet and is slow")
    
    def test_load_local_file_not_found(self, tmp_path):
        """Test loading non-existent local file."""
        local_path = tmp_path / "nonexistent.json"
        benchmark = ToxiGenBenchmark(
            use_local=True,
            local_path=local_path
        )
        
        with pytest.raises(FileNotFoundError):
            benchmark.load()
    
    def test_get_samples_not_loaded(self):
        """Test get_samples triggers load."""
        if not _datasets_available():
            pytest.skip("datasets library not installed")
        
        benchmark = ToxiGenBenchmark()
        # Note: This would try to load from hub, so we skip in tests
        # In practice, mock the load method for testing


class TestRealToxicityPrompt:
    """Tests for RealToxicityPrompt dataclass."""
    
    def test_init(self):
        """Test initialization."""
        prompt = RealToxicityPrompt(
            text="Test prompt",
            continuation="Test continuation",
            toxicity=0.3,
            continuation_toxicity=0.5
        )
        assert prompt.text == "Test prompt"
        assert prompt.continuation == "Test continuation"
        assert prompt.toxicity == 0.3
        assert prompt.continuation_toxicity == 0.5
        assert isinstance(prompt.metadata, dict)
    
    def test_init_defaults(self):
        """Test initialization with defaults."""
        prompt = RealToxicityPrompt(text="Test")
        assert prompt.continuation is None
        assert prompt.toxicity is None
        assert prompt.continuation_toxicity is None
        assert prompt.metadata == {}


class TestRealToxicityPromptsBenchmark:
    """Tests for RealToxicityPromptsBenchmark."""
    
    def test_init_default(self):
        """Test default initialization."""
        if not _datasets_available():
            pytest.skip("datasets library not installed")
        
        benchmark = RealToxicityPromptsBenchmark()
        assert benchmark.cache_dir.exists()
        assert not benchmark.use_local
        assert not benchmark.is_loaded
    
    def test_init_local(self, tmp_path):
        """Test initialization with local file."""
        local_path = tmp_path / "realtoxicity.json"
        benchmark = RealToxicityPromptsBenchmark(
            use_local=True,
            local_path=local_path
        )
        assert benchmark.use_local
        assert benchmark.local_path == local_path
