import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from .base import BaseTabularModel

class CatBoostModel(BaseTabularModel):
    """
    Unified high-performance wrapper for CatBoost Classifier & Regressor.
    """
    def __init__(self, task_type: str = "binary", eval_metric: str = "roc_auc", random_state: int = 42, params: Optional[Dict[str, Any]] = None, use_gpu: bool = False):
        super().__init__(task_type, eval_metric, random_state, params)
        self.use_gpu = use_gpu

    def _get_metric_name(self) -> str:
        m = self.eval_metric.lower()
        if "auc" in m:
            return "AUC"
        elif "logloss" in m:
            return "Logloss" if self.task_type == "binary" else "MultiClass"
        elif "acc" in m:
            return "Accuracy"
        elif "rmse" in m:
            return "RMSE"
        elif "mae" in m:
            return "MAE"
        return "AUC" if self.task_type == "binary" else "RMSE"

    def _get_default_params(self) -> Dict[str, Any]:
        defaults = {
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "random_seed": self.random_state,
            "verbose": False,
            "thread_count": -1,
        }
        if self.use_gpu:
            defaults["task_type"] = "GPU"
            defaults.pop("thread_count", None)  # thread_count tidak relevan di GPU mode
        defaults["eval_metric"] = self._get_metric_name()
        return defaults

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 100,
        verbose: bool = False,
    ):
        from catboost import CatBoostClassifier, CatBoostRegressor

        self.feature_names = list(X_train.columns)
        merged_params = self._get_default_params()
        merged_params.update(self.params)

        if early_stopping_rounds > 0 and X_val is not None:
            merged_params["early_stopping_rounds"] = early_stopping_rounds

        if verbose:
            merged_params["verbose"] = 100

        if self.task_type in ["binary", "multiclass"]:
            self.model = CatBoostClassifier(**merged_params)
        else:
            self.model = CatBoostRegressor(**merged_params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type == "binary":
            return self.model.predict_proba(X)[:, 1]
        elif self.task_type == "multiclass":
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        if hasattr(self.model, "get_feature_importance"):
            return self.model.get_feature_importance()
        return np.zeros(len(self.feature_names))
