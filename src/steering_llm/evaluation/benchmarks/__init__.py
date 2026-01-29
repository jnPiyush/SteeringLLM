"""
Benchmark management module.

This module provides base classes for safety benchmarks.
"""

from steering_llm.evaluation.benchmarks.toxigen import ToxiGenBenchmark
from steering_llm.evaluation.benchmarks.realtoxicity import RealToxicityPromptsBenchmark

__all__ = [
    "ToxiGenBenchmark",
    "RealToxicityPromptsBenchmark",
]
