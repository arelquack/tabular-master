import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from .base import BaseTabularModel

class SklearnModel(BaseTabularModel):
    """
    Unified wrapper for standard Scikit-Learn models (RandomForest, ExtraTrees, LogisticRegression, Ridge).
    """
    def __init__(self, model_type: str = "rf", task_type: str = "binary", random_state: int = 42, params: Optional[Dict[str, Any]] = None):
        super().__init__(task_type, random_state, params)
        self.model_type = model_type.lower()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 0,
        verbose: bool = False,
    ):
        self.feature_names = list(X_train.columns)

        # Impute NaNs with median for models that do not handle NaN natively
        X_tr = X_train.fillna(X_train.median(numeric_only=True))

        if self.model_type in ["rf", "random_forest"]:
            if self.task_type in ["binary", "multiclass"]:
                self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **self.params)
            else:
                self.model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **self.params)

        elif self.model_type in ["et", "extra_trees"]:
            if self.task_type in ["binary", "multiclass"]:
                self.model = ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1, **self.params)
            else:
                self.model = ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1, **self.params)

        elif self.model_type in ["ridge", "linear"]:
            if self.task_type in ["binary", "multiclass"]:
                self.model = LogisticRegression(random_state=self.random_state, max_iter=1000, **self.params)
            else:
                self.model = Ridge(random_state=self.random_state, **self.params)

        self.model.fit(X_tr, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = X.fillna(X.median(numeric_only=True))
        return self.model.predict(X_imp)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = X.fillna(X.median(numeric_only=True))
        if hasattr(self.model, "predict_proba"):
            if self.task_type == "binary":
                return self.model.predict_proba(X_imp)[:, 1]
            return self.model.predict_proba(X_imp)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X_imp)
            return 1 / (1 + np.exp(-df))
        return self.model.predict(X_imp)

    def get_feature_importance(self) -> np.ndarray:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_).flatten()
        return np.zeros(len(self.feature_names))
