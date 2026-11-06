import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from .base import BaseTabularModel

class XGBoostModel(BaseTabularModel):
    """
    Unified high-performance wrapper for XGBoost Classifier & Regressor using Histogram Tree Method.
    """
    def __init__(self, task_type: str = "binary", eval_metric: str = "roc_auc", random_state: int = 42, params: Optional[Dict[str, Any]] = None, use_gpu: bool = False):
        super().__init__(task_type, eval_metric, random_state, params)
        self.use_gpu = use_gpu

    def _get_metric_name(self) -> str:
        m = self.eval_metric.lower()
        if "auc" in m:
            return "auc"
        elif "logloss" in m:
            return "logloss" if self.task_type == "binary" else "mlogloss"
        elif "acc" in m:
            return "error"
        elif "rmse" in m:
            return "rmse"
        elif "mae" in m:
            return "mae"
        return "auc" if self.task_type == "binary" else "rmse"

    def _get_default_params(self) -> Dict[str, Any]:
        defaults = {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "tree_method": "hist",
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if self.use_gpu:
            import xgboost as xgb
            xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
            if xgb_version >= (2, 0):
                defaults["device"] = "cuda"
            else:
                defaults["tree_method"] = "gpu_hist"
                defaults["gpu_id"] = 0
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
        import xgboost as xgb

        self.feature_names = list(X_train.columns)
        merged_params = self._get_default_params()
        merged_params.update(self.params)

        if early_stopping_rounds > 0 and X_val is not None:
            merged_params["early_stopping_rounds"] = early_stopping_rounds

        if self.task_type in ["binary", "multiclass"]:
            self.model = xgb.XGBClassifier(**merged_params)
        else:
            self.model = xgb.XGBRegressor(**merged_params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=100 if verbose else False,
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
