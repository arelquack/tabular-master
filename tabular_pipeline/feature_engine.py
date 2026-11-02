import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod

class BaseFeatureHook(ABC):
    """
    User-definable domain feature engineering hook.
    Inherit from this class or pass a function to customize features per dataset.
    """
    def transform(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Override this method to write domain-specific feature engineering.
        """
        return train_df, test_df


class UniversalAutoFeatureEngine:
    """
    Universal Automated Feature Engineering (Auto-FE) Engine.
    Generates domain-agnostic mathematical feature primitives across any tabular dataset:
    - Primitive 1: Row-wise Numerical Aggregations (mean, std, min, max, zero count per entity)
    - Primitive 2: Missingness & Sparsity Signatures (row_na_count, col_isnan)
    - Primitive 3: Pairwise Categorical Interactions (CatA_x_CatB frequency & rarity)
    - Primitive 4: Group-by Aggregations & Relative Differences (value relative to category mean/std)
    - Primitive 5: Datetime Temporal Decomposition (year, month, day, is_weekend, hour)
    - Primitive 6: Numerical Ratios & Interactions (Aggressive mode)
    - Primitive 7: Automated Pruning (filters near-constants and collinear features r > 0.98)
    """
    def __init__(
        self,
        preset: str = "conservative", # "off" | "conservative" | "aggressive"
        max_features: int = 40,
        collinear_threshold: float = 0.98,
    ):
        self.preset = preset.lower()
        self.max_features = max_features
        self.collinear_threshold = collinear_threshold

    def generate(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        detected_types: dict,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        
        if self.preset == "off":
            return train_df, test_df

        train_out = train_df.copy()
        test_out = test_df.copy() if test_df is not None else None
        dfs = [train_out] if test_out is None else [train_out, test_out]

        new_feature_names: List[str] = []

        # -------------------------------------------------------------
        # 1. Datetime Temporal Decomposition
        # -------------------------------------------------------------
        dt_cols = detected_types.get("datetime", [])
        for col in dt_cols:
            for df in dfs:
                if col in df.columns:
                    series = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_year"] = series.dt.year
                    df[f"{col}_month"] = series.dt.month
                    df[f"{col}_day"] = series.dt.day
                    df[f"{col}_dayofweek"] = series.dt.dayofweek
                    df[f"{col}_is_weekend"] = series.dt.dayofweek.isin([5, 6]).astype(int)
                    if series.dt.hour.nunique() > 1:
                        df[f"{col}_hour"] = series.dt.hour

        # -------------------------------------------------------------
        # 2. Missingness Signatures & Null Counts
        # -------------------------------------------------------------
        # Row-level total missingness
        for df in dfs:
            df["row_na_count"] = df.isnull().sum(axis=1)
        new_feature_names.append("row_na_count")

        # Column-level binary null flags
        missing_stats = detected_types.get("missing_stats", {})
        cols_with_missing = [c for c, rate in missing_stats.items() if 0.002 <= rate <= 0.85]
        for col in cols_with_missing:
            col_name = f"{col}_isnan"
            new_feature_names.append(col_name)
            for df in dfs:
                if col in df.columns:
                    df[col_name] = df[col].isnull().astype(int)

        # -------------------------------------------------------------
        # 3. Row-wise Numerical Aggregations (Strictly for Homogeneous subsets)
        # -------------------------------------------------------------
        num_cols = detected_types.get("numeric", [])
        if len(num_cols) >= 3:
            # Check for bounded rating / score columns (e.g., surveys, Likert scales 0-5 or 0-10)
            bounded_cols = []
            for c in num_cols:
                c_min = train_out[c].min()
                c_max = train_out[c].max()
                if pd.notnull(c_min) and pd.notnull(c_max):
                    if 0 <= c_min and c_max <= 10:
                        bounded_cols.append(c)

            # ONLY generate row-stats if columns are truly homogeneous (e.g. Likert ratings)!
            # NEVER compute row-wise min/mean across arbitrary mixed units (age, USD, km, hours).
            if len(bounded_cols) >= 3:
                for df in dfs:
                    row_mat = df[bounded_cols].values
                    df["row_num_mean"] = np.nanmean(row_mat, axis=1)
                    df["row_num_std"] = np.nanstd(row_mat, axis=1)
                    df["row_num_min"] = np.nanmin(row_mat, axis=1)
                    df["row_num_max"] = np.nanmax(row_mat, axis=1)
                    df["row_num_zeros"] = (row_mat == 0).sum(axis=1)

                new_feature_names.extend(["row_num_mean", "row_num_std", "row_num_min", "row_num_max", "row_num_zeros"])

        # -------------------------------------------------------------
        # 4. High-Cardinality Frequency Encoding & Categorical Interactions
        # -------------------------------------------------------------
        cat_cols = detected_types.get("all_cat", [])
        # Frequency encoding strictly for high-cardinality categoricals (> 15 unique values)
        for col in cat_cols:
            if train_out[col].nunique() > 15:
                freq_map = train_out[col].value_counts(normalize=True).to_dict()
                col_name = f"{col}_freq"
                new_feature_names.append(col_name)
                for df in dfs:
                    if col in df.columns:
                        df[col_name] = df[col].map(freq_map).fillna(0.0)

        # Pairwise interaction across top categoricals (cardinality 2-25)
        informative_cats = [c for c in cat_cols if 2 <= train_out[c].nunique() <= 25]
        if len(informative_cats) >= 2:
            top_pairs = []
            for i in range(min(3, len(informative_cats))):
                for j in range(i + 1, min(4, len(informative_cats))):
                    top_pairs.append((informative_cats[i], informative_cats[j]))

            for c1, c2 in top_pairs[:3]:
                pair_name = f"{c1}_x_{c2}"
                freq_pair_name = f"{pair_name}_freq"
                new_feature_names.extend([pair_name, freq_pair_name])

                # Build combined string column
                for df in dfs:
                    if c1 in df.columns and c2 in df.columns:
                        df[pair_name] = df[c1].astype(str) + "_" + df[c2].astype(str)

                # Frequency encode the combination
                freq_map = train_out[pair_name].value_counts(normalize=True).to_dict()
                for df in dfs:
                    df[freq_pair_name] = df[pair_name].map(freq_map).fillna(0.0)

        # -------------------------------------------------------------
        # 5. Group-by Aggregations & Relative Z-Scores
        # -------------------------------------------------------------
        if len(cat_cols) > 0 and len(num_cols) > 0:
            candidate_cats = [c for c in cat_cols if 2 <= train_out[c].nunique() <= 50]
            max_cats = 2 if self.preset == "conservative" else 4
            max_nums = 3 if self.preset == "conservative" else 5

            agg_cats = candidate_cats[:max_cats]
            agg_nums = num_cols[:max_nums]

            for cat in agg_cats:
                for num in agg_nums:
                    grp = train_out.groupby(cat)[num].agg(["mean", "std"]).reset_index()
                    mean_col = f"{cat}_{num}_mean"
                    std_col = f"{cat}_{num}_std"
                    grp.columns = [cat, mean_col, std_col]

                    diff_col = f"{num}_diff_{cat}"
                    new_feature_names.append(diff_col)

                    zscore_col = f"{num}_zscore_{cat}" if self.preset == "aggressive" else None
                    if zscore_col:
                        new_feature_names.append(zscore_col)

                    for df in dfs:
                        if cat in df.columns and num in df.columns:
                            merged = pd.merge(df[[cat]], grp, on=cat, how="left")
                            df[diff_col] = df[num] - merged[mean_col].values
                            if zscore_col:
                                df[zscore_col] = (df[num] - merged[mean_col].values) / (merged[std_col].values + 1e-6)

        # -------------------------------------------------------------
        # 6. High-Leverage Interaction Pairs & Ratios (Aggressive Preset)
        # -------------------------------------------------------------
        if self.preset == "aggressive" and len(num_cols) >= 2:
            top_nums = num_cols[:4]
            for i in range(len(top_nums)):
                for j in range(i + 1, len(top_nums)):
                    na = top_nums[i]
                    nb = top_nums[j]
                    ratio_name = f"{na}_div_{nb}"
                    new_feature_names.append(ratio_name)
                    for df in dfs:
                        if na in df.columns and nb in df.columns:
                            df[ratio_name] = df[na] / (df[nb].abs() + 1e-5)

        # -------------------------------------------------------------
        # 7. Automated Feature Pruning (Collinearity & Low Variance Filter)
        # -------------------------------------------------------------
        all_created = [c for c in new_feature_names if c in train_out.columns]
        if all_created:
            # 1. Filter out near-constant features (variance < 1e-7)
            # Only check numeric columns for variance
            numeric_created = [c for c in all_created if np.issubdtype(train_out[c].dtype, np.number)]
            non_numeric_created = [c for c in all_created if not np.issubdtype(train_out[c].dtype, np.number)]

            valid_feats = []
            if numeric_created:
                variances = train_out[numeric_created].var()
                valid_num = variances[variances > 1e-7].index.tolist()

                # Filter extreme collinearity (r > collinear_threshold)
                if len(valid_num) > 1:
                    corr_matrix = train_out[valid_num].corr().abs()
                    to_drop = set()
                    for i in range(len(valid_num)):
                        for j in range(i + 1, len(valid_num)):
                            f1 = valid_num[i]
                            f2 = valid_num[j]
                            if f1 not in to_drop and f2 not in to_drop:
                                if corr_matrix.loc[f1, f2] >= self.collinear_threshold:
                                    to_drop.add(f2)
                    valid_num = [f for f in valid_num if f not in to_drop]
                valid_feats.extend(valid_num)

            valid_feats.extend(non_numeric_created)

            # Limit to max_features limit
            selected_auto_feats = valid_feats[:self.max_features]
            drop_candidates = [c for c in all_created if c not in selected_auto_feats]
            
            for df in dfs:
                df.drop(columns=[c for c in drop_candidates if c in df.columns], inplace=True)

        return train_out, test_out

# Backward compatibility alias
GenericFeatureEngine = UniversalAutoFeatureEngine
