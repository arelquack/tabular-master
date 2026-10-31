import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any

class SafeOrdinalEncoder:
    """
    Leak-free Ordinal / Label Encoder that safely handles unseen categories at test/val time.
    """
    def __init__(self, cat_cols: List[str]):
        self.cat_cols = cat_cols
        self.mappings: Dict[str, Dict[Any, int]] = {}

    def fit(self, df: pd.DataFrame):
        self.mappings = {}
        for col in self.cat_cols:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.mappings[col] = mapping
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cat_cols:
            if col in df.columns and col in self.mappings:
                mapping = self.mappings[col]
                # Map known categories; unknown or NaN values become -1
                encoded = df[col].map(mapping).fillna(-1).astype(float)
                df[col] = encoded
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class SafeTargetEncoder:
    """
    Safe Target Encoder with empirical Bayesian smoothing to prevent overfitting.
    Fitted strictly on train fold, then applied to val and test.
    """
    def __init__(self, cat_cols: List[str], smoothing: float = 10.0):
        self.cat_cols = cat_cols
        self.smoothing = smoothing
        self.target_stats: Dict[str, Dict[Any, float]] = {}
        self.global_mean: float = 0.0

    def fit(self, df: pd.DataFrame, target: pd.Series):
        self.global_mean = float(target.mean())
        self.target_stats = {}

        for col in self.cat_cols:
            if col in df.columns:
                counts = target.groupby(df[col]).count()
                means = target.groupby(df[col]).mean()
                # Empirical Bayesian smoothing formula:
                # smoothed = (count * mean + smoothing * global_mean) / (count + smoothing)
                smoothed = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
                self.target_stats[col] = smoothed.to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cat_cols:
            if col in df.columns and col in self.target_stats:
                mapping = self.target_stats[col]
                encoded = df[col].map(mapping).fillna(self.global_mean)
                df[f"{col}_target_enc"] = encoded
        return df
