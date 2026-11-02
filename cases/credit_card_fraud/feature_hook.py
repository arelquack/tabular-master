import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class CustomFeatures(BaseFeatureHook):
    """
    High-signal feature engineering hook combining domain fraud interactions
    and synthetic generator artifact features (cents grid modulo & value frequency).
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]

        # 1. Domain Risk Ratios and Multipliers
        for df in dfs:
            # Transaction exposure relative to account balance
            df['amount_to_balance_ratio'] = (df['amount_usd'] / (df['account_balance_usd'].abs() + 1.0)).astype('float32')
            # Velocity rate per hour
            df['txn_velocity_rate'] = (df['txn_count_last_24h'] / (df['hours_since_last_txn'] + 0.1)).astype('float32')
            # Multiplicative interaction: large amount at risky merchant
            df['amount_x_risk'] = (df['amount_usd'] * df['merchant_risk_score']).astype('float32')
            # Multiplicative interaction: high velocity at risky merchant
            df['velocity_x_risk'] = (df['velocity_score'] * df['merchant_risk_score']).astype('float32')

        # 2. Risk Flags Composite (Additive Anomaly Multiplier)
        flag_cols = ['used_vpn', 'ip_country_mismatch', 'billing_shipping_mismatch', 'is_foreign_transaction', 'is_new_merchant']
        for df in dfs:
            valid_flags = [c for c in flag_cols if c in df.columns]
            df['risk_flags_count'] = df[valid_flags].astype(int).sum(axis=1).astype('float32')

        # 3. Peer-Group Merchant Deviation (Learned strictly on train to prevent leakage)
        merchant_amt_mean = train_df.groupby('merchant_category')['amount_usd'].mean().to_dict()
        for df in dfs:
            df['amount_diff_merchant_mean'] = (df['amount_usd'] - df['merchant_category'].map(merchant_amt_mean)).astype('float32')

        # 4. Skew Correction on Heavy-Tailed Financial & Physical Quantities
        for df in dfs:
            df['amount_usd_log1p'] = np.log1p(df['amount_usd'].clip(lower=0)).astype('float32')
            df['account_balance_log1p'] = np.log1p(df['account_balance_usd'].clip(lower=0)).astype('float32')
            df['distance_km_log1p'] = np.log1p(df['distance_from_home_km'].clip(lower=0)).astype('float32')

        # 5. Generator Artifacts: Cents Modulo on Continuous Columns
        continuous_cols = [
            'amount_usd',
            'hours_since_last_txn',
            'distance_from_home_km',
            'account_balance_usd',
            'velocity_score',
            'merchant_risk_score'
        ]
        for col in continuous_cols:
            if col in train_df.columns:
                for df in dfs:
                    df[f'cents_{col}'] = (np.round(df[col] * 100) % 100).astype('float32')

        # 6. Generator Artifacts: Value Repetition Counts across Dataset
        count_cols = [
            'amount_usd',
            'account_balance_usd',
            'hours_since_last_txn',
            'distance_from_home_km',
            'merchant_risk_score',
            'velocity_score',
            'customer_age',
            'card_age_months'
        ]
        for c in count_cols:
            if c in train_df.columns:
                series_pool = pd.concat([df[c] for df in dfs], axis=0) if len(dfs) > 1 else train_df[c]
                vc = series_pool.value_counts()
                for df in dfs:
                    df[f'cnt_{c}'] = df[c].map(vc).fillna(0).astype('float32')

        return train_df, test_df