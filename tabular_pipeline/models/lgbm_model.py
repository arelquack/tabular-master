import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from .base import BaseTabularModel

class LightGBMModel(BaseTabularModel):
    """
    Unified high-performance wrapper for LightGBM Classifier & Regressor.
    """
    def __init__(self, task_type: str = "binary", eval_metric: str = "roc_auc", random_state: int = 42, params: Optional[Dict[str, Any]] = None, use_gpu: bool = False):
        super().__init__(task_type, eval_metric, random_state, params)
        self.use_gpu = use_gpu

    def _get_metric_name(self) -> str:
        m = self.eval_metric.lower()
        if "auc" in m:
            return "auc"
        elif "logloss" in m:
            return "binary_logloss" if self.task_type == "binary" else "multi_logloss"
        elif "acc" in m:
            return "binary_error"
        elif "rmse" in m:
            return "rmse"
        elif "mae" in m:
            return "mae"
        return "binary_logloss" if self.task_type == "binary" else "rmse"

    def _get_default_params(self) -> Dict[str, Any]:
        metric_str = self._get_metric_name()
        defaults = {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 45,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "random_state": self.random_state,
            "importance_type": "gain",
            "verbose": -1,
            "n_jobs": -1,
        }
        if self.use_gpu:
            defaults["device"] = "gpu"
            defaults.pop("n_jobs", None)  # n_jobs tidak kompatibel dengan GPU mode
        if self.task_type == "binary":
            defaults["objective"] = "binary"
            defaults["metric"] = metric_str
        elif self.task_type == "multiclass":
            defaults["objective"] = "multiclass"
            defaults["metric"] = metric_str
        elif self.task_type == "regression":
            defaults["objective"] = "regression"
            defaults["metric"] = metric_str
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
        import lightgbm as lgb
        from lightgbm import early_stopping, log_evaluation

        self.feature_names = list(X_train.columns)
        merged_params = self._get_default_params()
        merged_params.update(self.params)

        if self.task_type in ["binary", "multiclass"]:
            self.model = lgb.LGBMClassifier(**merged_params)
        else:
            self.model = lgb.LGBMRegressor(**merged_params)

        callbacks = []
        if X_val is not None and y_val is not None and early_stopping_rounds > 0:
            callbacks.append(early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
            callbacks.append(log_evaluation(period=100 if verbose else 0))

            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
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
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return np.zeros(len(self.feature_names))
