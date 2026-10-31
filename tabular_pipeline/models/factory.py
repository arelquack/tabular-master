from typing import Optional, Dict, Any
from .base import BaseTabularModel
from .lgbm_model import LightGBMModel
from .xgb_model import XGBoostModel
from .cat_model import CatBoostModel
from .sklearn_model import SklearnModel

class ModelFactory:
    """
    Factory for creating unified tabular models by name with metric alignment.
    """
    @staticmethod
    def create(
        name: str,
        task_type: str = "binary",
        eval_metric: str = "roc_auc",
        random_state: int = 42,
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseTabularModel:
        name_clean = name.lower().replace("-", "_")

        if name_clean in ["lgbm", "lightgbm"]:
            return LightGBMModel(task_type=task_type, eval_metric=eval_metric, random_state=random_state, params=params)
        elif name_clean in ["xgb", "xgboost"]:
            return XGBoostModel(task_type=task_type, eval_metric=eval_metric, random_state=random_state, params=params)
        elif name_clean in ["cat", "catboost"]:
            return CatBoostModel(task_type=task_type, eval_metric=eval_metric, random_state=random_state, params=params)
        elif name_clean in ["rf", "random_forest", "et", "extra_trees", "ridge", "logreg"]:
            return SklearnModel(model_type=name_clean, task_type=task_type, random_state=random_state, params=params)
        else:
            raise ValueError(f"Unknown model name: '{name}'. Supported: lgbm, xgb, catboost, rf, et, ridge, logreg")
