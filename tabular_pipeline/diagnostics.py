import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, List

class AdversarialValidator:
    """
    Performs Adversarial Validation to detect distribution shift between Train and Test sets.
    Computes Train vs Test discrimination AUC and identifies drifting features.
    """
    def __init__(self, n_splits: int = 3, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def check(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        try:
            from lightgbm import LGBMClassifier
            use_lgbm = True
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            use_lgbm = False

        # Defensively filter feature_cols to common columns between train and test
        valid_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
        if len(valid_cols) < 2:
            return {
                "adversarial_auc": 0.5,
                "status": "Skipped (Insufficient overlapping features between train and test)",
                "is_shifted": False,
                "top_drift_features": [],
            }

        # Prepare dataset
        train_feat = train_df[valid_cols].copy()
        test_feat = test_df[valid_cols].copy()

        train_feat["__is_test__"] = 0
        test_feat["__is_test__"] = 1

        combined = pd.concat([train_feat, test_feat], ignore_index=True)
        X = combined[valid_cols].copy()
        y = combined["__is_test__"].values

        # Simple categorical encoding for adversarial check
        for col in X.columns:
            if X[col].dtype == "object" or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category").cat.codes

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof_preds = np.zeros(len(combined))
        feature_importances = np.zeros(len(feature_cols))

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_val = X.iloc[val_idx], y[val_idx]

            if use_lgbm:
                clf = LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.08,
                    max_depth=4,
                    random_state=self.seed,
                    verbose=-1,
                    n_jobs=-1,
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.seed,
                    n_jobs=-1,
                )

            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_va)[:, 1]
            feature_importances += clf.feature_importances_ / self.n_splits

        adversarial_auc = float(roc_auc_score(y, oof_preds))

        # Identify top drift features
        drift_ranking = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importances,
        }).sort_values("importance", ascending=False)

        top_drift_features = drift_ranking.head(10).to_dict(orient="records")

        is_shifted = adversarial_auc > 0.70

        return {
            "adversarial_auc": adversarial_auc,
            "is_shifted": is_shifted,
            "top_drift_features": top_drift_features,
            "status": "WARNING: Significant distribution shift detected!" if is_shifted else "OK: Train and Test distributions are aligned."
        }
