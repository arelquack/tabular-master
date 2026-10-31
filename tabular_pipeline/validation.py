import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
)

class CrossValidationSplitter:
    """
    Factory for cross-validation splitting strategies.
    Ensures safe, leak-free splits with support for groups and temporal orders.
    """
    def __init__(
        self,
        strategy: str = "stratified_kfold",
        n_splits: int = 5,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.strategy = strategy.lower()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        
        if self.strategy == "stratified_kfold":
            if y is None:
                raise ValueError("StratifiedKFold requires target 'y'.")
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
            return cv.split(X, y)

        elif self.strategy == "kfold":
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
            return cv.split(X, y)

        elif self.strategy == "group_kfold":
            if groups is None:
                raise ValueError("GroupKFold requires 'groups' column.")
            cv = GroupKFold(n_splits=self.n_splits)
            return cv.split(X, y, groups=groups)

        elif self.strategy == "stratified_group_kfold":
            if groups is None or y is None:
                raise ValueError("StratifiedGroupKFold requires both 'groups' and target 'y'.")
            cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)
            return cv.split(X, y, groups=groups)

        elif self.strategy == "time_series":
            # TimeSeriesSplit does not shuffle to preserve temporal causality
            cv = TimeSeriesSplit(n_splits=self.n_splits)
            return cv.split(X, y)

        else:
            raise ValueError(f"Unknown CV strategy: '{self.strategy}'")
