import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from .models.factory import ModelFactory
from .validation import CrossValidationSplitter
from .encoders import SafeOrdinalEncoder, SafeTargetEncoder
from .metrics import MetricEvaluator

class CrossValidatorTrainer:
    """
    Orchestrates leak-free Out-Of-Fold (OOF) model training and evaluation.
    """
    def __init__(
        self,
        task_type: str = "binary",
        eval_metric: str = "roc_auc",
        cv_strategy: str = "stratified_kfold",
        n_splits: int = 5,
        categorical_encoder: str = "ordinal",
        seed: int = 42,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        is_aborted: Optional[Callable[[], bool]] = None,
    ):
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.categorical_encoder = categorical_encoder
        self.seed = seed
        self.progress_callback = progress_callback
        self.is_aborted = is_aborted

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
        groups: Optional[pd.Series] = None,
        cat_cols: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 100,
        verbose: bool = False,
    ) -> Dict[str, Any]:

        cat_cols = cat_cols or []
        splitter = CrossValidationSplitter(
            strategy=self.cv_strategy,
            n_splits=self.n_splits,
            shuffle=True,
            seed=self.seed,
        )

        n_samples = len(X_train)
        is_multiclass = (self.task_type == "multiclass")
        num_classes = len(np.unique(y_train)) if is_multiclass else 1

        if is_multiclass:
            oof_preds = np.zeros((n_samples, num_classes))
            test_preds = np.zeros((len(X_test), num_classes)) if X_test is not None else None
        else:
            oof_preds = np.zeros(n_samples)
            test_preds = np.zeros(len(X_test)) if X_test is not None else None

        fold_scores = []
        feature_importances = []
        trained_models = []

        print(f"--- Training {model_name.upper()} ({self.n_splits} Folds, Metric: {self.eval_metric}) ---")

        splits = list(splitter.split(X_train, y_train, groups=groups))

        for fold, (train_idx, val_idx) in enumerate(splits):
            if self.is_aborted and self.is_aborted():
                raise InterruptedError("Execution stopped by user.")
            if self.progress_callback is not None:
                self.progress_callback({
                    "stage": "training_fold_start",
                    "model": model_name,
                    "fold": fold + 1,
                    "total_folds": self.n_splits,
                    "message": f"Training {model_name.upper()} [Fold {fold + 1}/{self.n_splits}]..."
                })

            X_tr, y_tr = X_train.iloc[train_idx].copy(), y_train[train_idx]
            X_va, y_va = X_train.iloc[val_idx].copy(), y_train[val_idx]
            X_te = X_test.copy() if X_test is not None else None

            # Apply leak-free encoding fitted ONLY on X_tr
            if self.categorical_encoder == "ordinal" and len(cat_cols) > 0:
                encoder = SafeOrdinalEncoder(cat_cols=cat_cols)
                X_tr = encoder.fit_transform(X_tr)
                X_va = encoder.transform(X_va)
                if X_te is not None:
                    X_te = encoder.transform(X_te)
            elif self.categorical_encoder == "target" and len(cat_cols) > 0 and self.task_type in ["binary", "regression"]:
                encoder = SafeTargetEncoder(cat_cols=cat_cols)
                encoder.fit(X_tr, pd.Series(y_tr))
                X_tr = encoder.transform(X_tr)
                X_va = encoder.transform(X_va)
                if X_te is not None:
                    X_te = encoder.transform(X_te)

            # Ensure any remaining object columns are converted to codes
            for col in X_tr.select_dtypes(include=["object"]).columns:
                unique_vals = X_tr[col].dropna().unique()
                col_map = {val: idx for idx, val in enumerate(unique_vals)}
                X_tr[col] = X_tr[col].map(col_map).fillna(-1).astype(float)
                X_va[col] = X_va[col].map(col_map).fillna(-1).astype(float)
                if X_te is not None:
                    X_te[col] = X_te[col].map(col_map).fillna(-1).astype(float)

            # Initialize and fit model
            model = ModelFactory.create(
                name=model_name,
                task_type=self.task_type,
                eval_metric=self.eval_metric,
                random_state=self.seed + fold,
                params=model_params,
            )

            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_va,
                y_val=y_va,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )

            # Predict OOF
            val_preds = model.predict_proba(X_va) if self.task_type in ["binary", "multiclass"] else model.predict(X_va)
            oof_preds[val_idx] = val_preds

            # Predict Test
            if X_te is not None:
                curr_test_preds = model.predict_proba(X_te) if self.task_type in ["binary", "multiclass"] else model.predict(X_te)
                test_preds += curr_test_preds / self.n_splits

            # Fold Score
            fold_score = MetricEvaluator.calculate(y_va, val_preds, self.eval_metric, self.task_type)
            fold_scores.append(fold_score)
            print(f"  Fold {fold + 1}/{self.n_splits} Score: {fold_score:.5f}")

            if self.progress_callback is not None:
                self.progress_callback({
                    "stage": "training_fold_done",
                    "model": model_name,
                    "fold": fold + 1,
                    "total_folds": self.n_splits,
                    "fold_score": fold_score,
                    "message": f"Completed {model_name.upper()} Fold {fold + 1}/{self.n_splits} | Score: {fold_score:.5f}"
                })

            feature_importances.append(model.get_feature_importance())
            trained_models.append(model)

        # Global OOF Score
        overall_score = MetricEvaluator.calculate(y_train, oof_preds, self.eval_metric, self.task_type)
        print(f"  >> Overall OOF {self.eval_metric}: {overall_score:.5f} (Mean: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f})\n")

        if self.progress_callback is not None:
            self.progress_callback({
                "stage": "training_model_done",
                "model": model_name,
                "overall_score": overall_score,
                "message": f"Model {model_name.upper()} Completed | Full OOF {self.eval_metric.upper()}: {overall_score:.5f}"
            })

        # Feature Importance DataFrame
        feat_names = list(X_tr.columns)
        mean_fi = np.mean(feature_importances, axis=0) if len(feature_importances) > 0 else np.zeros(len(feat_names))
        importance_df = pd.DataFrame({
            "feature": feat_names,
            "importance": mean_fi,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return {
            "model_name": model_name,
            "oof_preds": oof_preds,
            "test_preds": test_preds,
            "fold_scores": fold_scores,
            "overall_score": overall_score,
            "importance_df": importance_df,
            "models": trained_models,
        }
