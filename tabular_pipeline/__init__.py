"""
Tabular Pipeline Framework
A modular, battle-tested competitive machine learning & data mining pipeline.
Designed for Kaggle, Gemastik, and real-world tabular data science.
"""

from .config import PipelineConfig
from .pipeline import TabularPipeline
from .feature_engine import BaseFeatureHook, UniversalAutoFeatureEngine
from .tuner import HyperparameterTuner

__version__ = "1.1.0"

__all__ = [
    "PipelineConfig",
    "TabularPipeline",
    "BaseFeatureHook",
    "UniversalAutoFeatureEngine",
    "HyperparameterTuner",
]
