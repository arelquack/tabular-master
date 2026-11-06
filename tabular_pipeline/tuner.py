import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
import optuna
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
    StratifiedShuffleSplit,
    ShuffleSplit,
    GroupShuffleSplit
)
from .models import ModelFactory
from .metrics import MetricEvaluator

# Suppress Optuna verbose console spam
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HyperparameterTuner:
    """
    Automated Hyperparameter Optimization Engine powered by Optuna.
    Explores tree structures, regularization penalties, and feature subsampling
    to maximize Out-of-Fold validation performance while preventing overfitting.
    """
    def __init__(
        self,
        task_type: str = "binary",
        eval_metric: str = "roc_auc",
        cv_strategy: str = "stratified_kfold",
        group_col: Optional[str] = None,
        seed: int = 42,
        use_gpu: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.cv_strategy = (cv_strategy or "stratified_kfold").lower()
        self.group_col = group_col
        self.seed = seed
        self.use_gpu = use_gpu
        self.progress_callback = progress_callback
        self.higher_is_better = MetricEvaluator.is_higher_better(eval_metric)

    def tune_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        cv_strategy: Optional[str] = None,
        cat_cols: Optional[list] = None,
        n_trials: int = 15,
        sample_size: int = 60000,
        timeout: Optional[int] = 600,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Runs Bayesian optimization for the specified model family.
        Inherits case cv_strategy (e.g. GroupKFold, StratifiedGroupKFold, TimeSeries)
        to prevent intra-group data leakage during proxy evaluations.
        """
        m_name = model_name.lower()
        if m_name not in ["lgbm", "xgb", "catboost"]:
            raise ValueError(f"Tuning only supported for GBDT models (lgbm, xgb, catboost), got {model_name}")

        active_cv = (cv_strategy or self.cv_strategy or "stratified_kfold").lower()
        has_groups = groups is not None and len(groups) == len(X) and active_cv in ["group_kfold", "stratified_group_kfold"]

        # 1. Leak-Free Subsampling if dataset is very large
        n_samples = len(X)
        if n_samples > sample_size:
            if has_groups:
                train_ratio = min(0.9, max(0.1, sample_size / n_samples))
                gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=self.seed)
                sub_idx, _ = next(gss.split(X, y, groups=groups))
                X_sub = X.iloc[sub_idx].reset_index(drop=True)
                y_sub = y[sub_idx]
                groups_sub = groups[sub_idx]
            elif active_cv == "time_series":
                # Preserve temporal order: select the most recent window of observations
                X_sub = X.iloc[-sample_size:].reset_index(drop=True)
                y_sub = y[-sample_size:]
                groups_sub = None
            elif self.task_type in ["binary", "multiclass"]:
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=self.seed)
                sub_idx, _ = next(splitter.split(X, y))
                X_sub = X.iloc[sub_idx].reset_index(drop=True)
                y_sub = y[sub_idx]
                groups_sub = None
            else:
                splitter = ShuffleSplit(n_splits=1, train_size=sample_size, random_state=self.seed)
                sub_idx, _ = next(splitter.split(X, y))
                X_sub = X.iloc[sub_idx].reset_index(drop=True)
                y_sub = y[sub_idx]
                groups_sub = None
        else:
            X_sub = X.copy()
            y_sub = y.copy()
            groups_sub = groups.copy() if groups is not None else None

        # 2. Fast 2-Fold Proxy CV strictly respecting the case validation strategy
        if has_groups:
            if active_cv == "stratified_group_kfold" and self.task_type in ["binary", "multiclass"]:
                cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=self.seed)
            else:
                cv = GroupKFold(n_splits=2)
            splits = list(cv.split(X_sub, y_sub, groups=groups_sub))
        elif active_cv == "time_series":
            cv = TimeSeriesSplit(n_splits=2)
            splits = list(cv.split(X_sub, y_sub))
        elif self.task_type in ["binary", "multiclass"]:
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)
            splits = list(cv.split(X_sub, y_sub))
        else:
            cv = KFold(n_splits=2, shuffle=True, random_state=self.seed)
            splits = list(cv.split(X_sub, y_sub))

        trial_counter = 0
        best_so_far = -np.inf if self.higher_is_better else np.inf

        def objective(trial: optuna.Trial) -> float:
            nonlocal trial_counter, best_so_far
            if stop_check and stop_check():
                trial.study.stop()
                raise optuna.TrialPruned("Optimization stopped by user")
            trial_counter += 1

            # Define hyperparameter search spaces per family
            if m_name == "lgbm":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 4, 128),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 250),
                    "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 0.90),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
                    "n_estimators": 600,
                    "n_jobs": -1,
                    "random_state": self.seed,
                    "verbose": -1,
                }
            elif m_name == "xgb":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 8),
                    "min_child_weight": trial.suggest_int("min_child_weight", 5, 80),
                    "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 0.90),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
                    "n_estimators": 600,
                    "tree_method": "hist",
                    "n_jobs": -1,
                    "random_state": self.seed,
                }
            elif m_name == "catboost":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                    "depth": trial.suggest_int("depth", 3, 7),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 25.0, log=True),
                    "iterations": 600,
                    "random_seed": self.seed,
                    "verbose": False,
                }

            # Evaluate over the fast 2 folds
            oof_scores = []
            for tr_idx, val_idx in splits:
                X_tr, y_tr = X_sub.iloc[tr_idx], y_sub[tr_idx]
                X_va, y_val = X_sub.iloc[val_idx], y_sub[val_idx]

                model = ModelFactory.create(
                    m_name,
                    task_type=self.task_type,
                    eval_metric=self.eval_metric,
                    params=params,
                    use_gpu=self.use_gpu
                )
                model.fit(
                    X_tr, y_tr,
                    X_val=X_va, y_val=y_val,
                    early_stopping_rounds=40,
                    verbose=False,
                )

                if self.task_type in ["binary", "multiclass"]:
                    preds = model.predict_proba(X_va)
                else:
                    preds = model.predict(X_va)

                score = MetricEvaluator.calculate(y_val, preds, self.eval_metric, self.task_type)
                oof_scores.append(score)

            mean_score = float(np.mean(oof_scores))

            # Track progress
            is_new_best = False
            if self.higher_is_better:
                if mean_score > best_so_far:
                    best_so_far = mean_score
                    is_new_best = True
            else:
                if mean_score < best_so_far:
                    best_so_far = mean_score
                    is_new_best = True

            if self.progress_callback:
                self.progress_callback({
                    "model": m_name,
                    "trial": trial_counter,
                    "total_trials": n_trials,
                    "score": round(mean_score, 5),
                    "best_score": round(best_so_far, 5),
                    "is_new_best": is_new_best,
                    "params": trial.params
                })

            return mean_score

        direction = "maximize" if self.higher_is_better else "minimize"
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        # Re-inject constant production hyperparameters
        if m_name == "lgbm":
            best_params["n_estimators"] = 1500
            best_params["n_jobs"] = -1
            best_params["random_state"] = self.seed
            best_params["verbose"] = -1
        elif m_name == "xgb":
            best_params["n_estimators"] = 1500
            best_params["tree_method"] = "hist"
            best_params["n_jobs"] = -1
            best_params["random_state"] = self.seed
        elif m_name == "catboost":
            best_params["iterations"] = 1500
            best_params["random_seed"] = self.seed
            best_params["verbose"] = False

        return {
            "model": m_name,
            "best_score": round(study.best_value, 5),
            "best_params": best_params,
            "total_trials": len(study.trials),
        }
