"""
Tabular Pipeline Framework
A modular, battle-tested competitive machine learning & data mining pipeline.
Designed for Kaggle, Gemastik, and real-world tabular data science.
"""

from .config import PipelineConfig
from .pipeline import TabularPipeline
from .feature_engine import BaseFeatureHook, UniversalAutoFeatureEngine
from .tuner import HyperparameterTuner
from .models.factory import ModelFactory
from .indobert_engine import IndoBERTEngine

__version__ = "1.2.0"

__all__ = [
    "PipelineConfig",
    "TabularPipeline",
    "BaseFeatureHook",
    "UniversalAutoFeatureEngine",
    "HyperparameterTuner",
    "ModelFactory",
    "IndoBERTEngine",
]
