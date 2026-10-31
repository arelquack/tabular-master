import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from scipy.optimize import minimize_scalar
from typing import Callable, Tuple, Any

class MetricEvaluator:
    """
    Central evaluation engine supporting binary, multiclass, and regression metrics.
    Includes automated threshold optimization for binary classification.
    """

    HIGHER_IS_BETTER = {
        "roc_auc": True,
        "accuracy": True,
        "f1_macro": True,
        "f1_binary": True,
        "f1_weighted": True,
        "mcc": True,
        "qwk": True,
        "r2": True,
        "logloss": False,
        "rmse": False,
        "mae": False,
    }

    @staticmethod
    def is_higher_better(metric_name: str) -> bool:
        return MetricEvaluator.HIGHER_IS_BETTER.get(metric_name.lower(), True)

    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str, task_type: str = "binary") -> float:
        metric = metric_name.lower()
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if task_type == "binary":
            if metric == "roc_auc":
                return float(roc_auc_score(y_true, y_pred))
            elif metric == "logloss":
                eps = 1e-15
                clipped_preds = np.clip(y_pred, eps, 1 - eps)
                return float(log_loss(y_true, clipped_preds))
            else:
                # Metrics requiring discrete class labels (pred > 0.5 default)
                y_pred_discrete = (y_pred >= 0.5).astype(int)
                if metric == "accuracy":
                    return float(accuracy_score(y_true, y_pred_discrete))
                elif metric == "f1_binary":
                    return float(f1_score(y_true, y_pred_discrete, average="binary"))
                elif metric == "f1_macro":
                    return float(f1_score(y_true, y_pred_discrete, average="macro"))
                elif metric == "mcc":
                    return float(matthews_corrcoef(y_true, y_pred_discrete))

        elif task_type == "multiclass":
            if metric == "logloss":
                return float(log_loss(y_true, y_pred))
            else:
                # If y_pred is 2D probability matrix, convert to argmax classes
                if y_pred.ndim > 1:
                    y_pred_discrete = np.argmax(y_pred, axis=1)
                else:
                    y_pred_discrete = y_pred
                if metric == "accuracy":
                    return float(accuracy_score(y_true, y_pred_discrete))
                elif metric == "f1_macro":
                    return float(f1_score(y_true, y_pred_discrete, average="macro"))
                elif metric == "f1_weighted":
                    return float(f1_score(y_true, y_pred_discrete, average="weighted"))

        elif task_type == "regression":
            if metric == "rmse":
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif metric == "mae":
                return float(mean_absolute_error(y_true, y_pred))
            elif metric == "r2":
                return float(r2_score(y_true, y_pred))
            elif metric == "qwk":
                y_pred_discrete = np.round(y_pred).astype(int)
                return float(cohen_kappa_score(y_true, y_pred_discrete, weights="quadratic"))

        raise ValueError(f"Unsupported metric: '{metric_name}' for task_type: '{task_type}'")

    @staticmethod
    def optimize_binary_threshold(
        y_true: np.ndarray, y_proba: np.ndarray, metric_name: str
    ) -> Tuple[float, float]:
        """
        Finds the optimal probability threshold in [0.01, 0.99] that maximizes the given metric.
        Returns: (best_threshold, best_score)
        """
        def objective(th):
            preds = (y_proba >= th).astype(int)
            if metric_name == "accuracy":
                score = accuracy_score(y_true, preds)
            elif metric_name == "f1_binary":
                score = f1_score(y_true, preds, average="binary", zero_division=0)
            elif metric_name == "f1_macro":
                score = f1_score(y_true, preds, average="macro", zero_division=0)
            elif metric_name == "mcc":
                score = matthews_corrcoef(y_true, preds)
            else:
                score = accuracy_score(y_true, preds)
            return -score  # Minimize negative score

        res = minimize_scalar(objective, bounds=(0.05, 0.95), method="bounded")
        best_th = float(res.x)
        best_score = float(-res.fun)
        return best_th, best_score
