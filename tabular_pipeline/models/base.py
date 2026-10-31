from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

class BaseTabularModel(ABC):
    """
    Abstract unified interface for all tabular models in the framework.
    """
    def __init__(self, task_type: str = "binary", eval_metric: str = "roc_auc", random_state: int = 42, params: Optional[Dict[str, Any]] = None):
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.params = params or {}
        self.model = None
        self.feature_names = []

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 100,
        verbose: bool = False,
    ):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        pass
