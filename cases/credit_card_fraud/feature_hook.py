import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class CustomFeatures(BaseFeatureHook):
    """
    Curated high-signal fraud domain feature engineering hook.
    Pruned of noisy row-wise statistics and collinear frequency encodings.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]

        # 1. Financial Exposure & Liquidity Ratios
        for df in dfs:
            # Ratio of transaction amount to available account balance
            df['amount_to_balance_ratio'] = df['amount_usd'] / (df['account_balance_usd'].abs() + 1e-5)
            # Transaction velocity rate per hour
            df['txn_velocity_rate'] = df['txn_count_last_24h'] / (df['hours_since_last_txn'] + 0.1)

        # 2. Risk Flags Composite (Additive Anomaly Multiplier)
        flag_cols = ['used_vpn', 'ip_country_mismatch', 'billing_shipping_mismatch', 'is_foreign_transaction', 'is_new_merchant']
        for df in dfs:
            valid_flags = [c for c in flag_cols if c in df.columns]
            df['risk_flags_count'] = df[valid_flags].astype(int).sum(axis=1)

        # 3. Peer-Group Merchant Deviation (Learned strictly on train to prevent leakage)
        merchant_amt_mean = train_df.groupby('merchant_category')['amount_usd'].mean().to_dict()
        for df in dfs:
            df['amount_diff_merchant_mean'] = df['amount_usd'] - df['merchant_category'].map(merchant_amt_mean)

        # 4. Skew Correction on Heavy-Tailed Financial & Physical Quantities
        for df in dfs:
            df['amount_usd_log1p'] = np.log1p(df['amount_usd'].clip(lower=0))
            df['account_balance_log1p'] = np.log1p(df['account_balance_usd'].clip(lower=0))
            df['distance_km_log1p'] = np.log1p(df['distance_from_home_km'].clip(lower=0))

        return train_df, test_df
