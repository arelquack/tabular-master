import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from .metrics import MetricEvaluator

class EnsembleOptimizer:
    """
    Combines predictions from multiple models using:
    1. Simple Averaging
    2. Rank Averaging (ideal for ROC-AUC)
    3. Optimal Non-negative Constrained Blending (SLSQP / Nelder-Mead)
    4. Stacking Meta-Learner
    """
    def __init__(self, task_type: str = "binary", eval_metric: str = "roc_auc"):
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.is_higher_better = MetricEvaluator.is_higher_better(eval_metric)

    def simple_average(self, preds_list: List[np.ndarray]) -> np.ndarray:
        return np.mean(preds_list, axis=0)

    def rank_average(self, preds_list: List[np.ndarray]) -> np.ndarray:
        ranked_preds = []
        for p in preds_list:
            if p.ndim == 1:
                ranks = rankdata(p) / len(p)
            else:
                ranks = np.column_stack([rankdata(p[:, c]) / len(p) for c in range(p.shape[1])])
            ranked_preds.append(ranks)
        return np.mean(ranked_preds, axis=0)

    def find_optimal_weights(self, y_true: np.ndarray, oofs_list: List[np.ndarray]) -> np.ndarray:
        """
        Finds optimal weights w_i >= 0 such that sum(w_i) = 1 to maximize the target evaluation metric.
        """
        n_models = len(oofs_list)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        def loss_func(weights):
            weights = np.array(weights)
            # Weighted combination
            blended = np.zeros_like(oofs_list[0])
            for w, oof in zip(weights, oofs_list):
                blended += w * oof

            score = MetricEvaluator.calculate(y_true, blended, self.eval_metric, self.task_type)
            return -score if self.is_higher_better else score

        res = minimize(
            loss_func,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        weights = res.x
        # Normalize just in case
        weights = np.clip(weights, 0, 1)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        return weights

    def stack(
        self,
        y_train: np.ndarray,
        oof_dict: Dict[str, np.ndarray],
        test_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Trains a meta-learner (LogisticRegression or Ridge) on OOF predictions.
        """
        model_names = list(oof_dict.keys())
        X_meta_train = np.column_stack([oof_dict[m] for m in model_names])
        X_meta_test = np.column_stack([test_dict[m] for m in model_names]) if test_dict else None

        if self.task_type in ["binary", "multiclass"]:
            meta_model = LogisticRegression(random_state=42, max_iter=500)
            meta_model.fit(X_meta_train, y_train)
            oof_meta = meta_model.predict_proba(X_meta_train)[:, 1] if self.task_type == "binary" else meta_model.predict_proba(X_meta_train)
            test_meta = None
            if X_meta_test is not None:
                test_meta = meta_model.predict_proba(X_meta_test)[:, 1] if self.task_type == "binary" else meta_model.predict_proba(X_meta_test)
        else:
            meta_model = Ridge(random_state=42)
            meta_model.fit(X_meta_train, y_train)
            oof_meta = meta_model.predict(X_meta_train)
            test_meta = meta_model.predict(X_meta_test) if X_meta_test is not None else None

        return oof_meta, test_meta

    def hill_climbing(
        self,
        y_true: np.ndarray,
        oof_dict: Dict[str, np.ndarray],
        max_iter: int = 100,
        patience: int = 10,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], np.ndarray, float]:
        """
        Caruana's Ensemble Selection with Replacement (ICML 2004).
        Greedy Forward Selection: starts with best model, iteratively adds the model
        that gives the greatest metric improvement.
        Returns: (weights_dict, blended_oof, best_score)
        """
        names = list(oof_dict.keys())
        m = mask if mask is not None else np.ones(len(y_true), bool)
        y_eval = y_true[m]
        sign = 1 if self.is_higher_better else -1

        def eval_pred(pred):
            return sign * MetricEvaluator.calculate(y_eval, pred[m], self.eval_metric, self.task_type)

        start = max(names, key=lambda n: eval_pred(oof_dict[n]))
        counts = {start: 1}
        total = oof_dict[start].copy()
        best_score = eval_pred(total)
        bad_iter = 0

        for _ in range(max_iter):
            best_cand = None
            best_cand_score = -float("inf")
            current_n = sum(counts.values())

            for name in names:
                cand_total = total + oof_dict[name]
                cand_score = eval_pred(cand_total / (current_n + 1))
                if cand_score > best_cand_score:
                    best_cand_score = cand_score
                    best_cand = name

            if best_cand_score > best_score + 1e-9:
                best_score = best_cand_score
                counts[best_cand] = counts.get(best_cand, 0) + 1
                total = total + oof_dict[best_cand]
                bad_iter = 0
            else:
                bad_iter += 1
                if bad_iter >= patience:
                    break

        total_picks = sum(counts.values())
        weights = {k: v / total_picks for k, v in counts.items()}
        blended_oof = sum(w * oof_dict[k] for k, w in weights.items())
        final_score = MetricEvaluator.calculate(y_eval, blended_oof[m], self.eval_metric, self.task_type)

        return weights, blended_oof, final_score
