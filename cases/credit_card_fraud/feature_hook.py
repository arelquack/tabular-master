import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class CustomFeatures(BaseFeatureHook):
    """
    User-definable domain feature engineering hook.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]
        for df in dfs:
            pass # Write manual domain features here
                # --- UNIVERSAL PRIMITIVE 1: GROUP-BY AGGREGATIONS & RELATIVE DEVIATION ---
        # Computes peer-group baseline statistics (learned strictly from train_df to prevent leakage)
        cat_cols = [c for c in train_df.select_dtypes(include=['object', 'category']).columns if c not in ['id', 'is_fraud']]
        num_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in ['id', 'is_fraud']]
        for group_col in cat_cols[:2]:
            for val_col in num_cols[:2]:
                grp_mean = train_df.groupby(group_col)[val_col].mean().to_dict()
                for df in ([train_df, test_df] if test_df is not None else [train_df]):
                    df[f'{val_col}_by_{group_col}_mean'] = df[group_col].map(grp_mean)
                    df[f'{val_col}_diff_{group_col}_mean'] = df[val_col] - df[group_col].map(grp_mean)

                # --- UNIVERSAL PRIMITIVE 2: ROW-WISE INTENSITY & CONSISTENCY STATISTICS ---
        # Computes across-the-board intensity, dispersion, and missingness per entity
        numeric_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in ['id', 'is_fraud']][:8]
        if len(numeric_cols) >= 2:
            for df in ([train_df, test_df] if test_df is not None else [train_df]):
                df['row_stat_mean'] = df[numeric_cols].mean(axis=1)
                df['row_stat_std'] = df[numeric_cols].std(axis=1).fillna(0)
                df['row_stat_min'] = df[numeric_cols].min(axis=1)
                df['row_stat_max'] = df[numeric_cols].max(axis=1)
                df['row_null_count'] = df[numeric_cols].isnull().sum(axis=1)

                # --- UNIVERSAL PRIMITIVE 3: HIGH-CARDINALITY FREQUENCY & RARITY ENCODING ---
        # Maps categorical entities to prevalence rates (detects rare categories without leakage)
        cat_cols = [c for c in train_df.select_dtypes(include=['object', 'category']).columns if c not in ['id', 'is_fraud']]
        for c in cat_cols:
            freq = train_df[c].value_counts(normalize=True).to_dict()
            for df in ([train_df, test_df] if test_df is not None else [train_df]):
                df[f'{c}_freq'] = df[c].map(freq).fillna(0)

                # --- UNIVERSAL PRIMITIVE 4: SKEW CORRECTION & LOG-TRANSFORM ---
        # Stabilizes heavy-tailed variables (financial, revenue, delays, physical quantities)
        num_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in ['id', 'is_fraud']]
        for c in num_cols[:4]:
            if (train_df[c] >= 0).all():
                for df in ([train_df, test_df] if test_df is not None else [train_df]):
                    df[f'{c}_log1p'] = np.log1p(df[c].clip(lower=0))

                # Visual Formula: amount_usd_ratio_account_balance_usd
        for df in ([train_df, test_df] if test_df is not None else [train_df]):
            df['amount_usd_ratio_account_balance_usd'] = df['amount_usd'] / (df['account_balance_usd'].abs() + 1e-5)

                # Visual Group Deviation: amount_usd_diff_from_merchant_category_mean
        grp_map = train_df.groupby('merchant_category')['amount_usd'].mean().to_dict()
        for df in ([train_df, test_df] if test_df is not None else [train_df]):
            df['amount_usd_diff_from_merchant_category_mean'] = df['amount_usd'] - df['merchant_category'].map(grp_map)

        return train_df, test_df
