import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class AirlineChampionFeatures(BaseFeatureHook):
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]

        survey_cols = [
            "Inflight wifi service", "Departure/Arrival time convenient",
            "Ease of Online booking", "Gate location", "Food and drink",
            "Online boarding", "Seat comfort", "Inflight entertainment",
            "On-board service", "Leg room service", "Baggage handling",
            "Checkin service", "Inflight service", "Cleanliness"
        ]

        for df in dfs:
            # 1. Physics Delay Imputation (mengatasi missing value secara fisik)
            if "Arrival Delay in Minutes" in df.columns and "Departure Delay in Minutes" in df.columns:
                df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
                df["Total_Delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
                df["Has_Delay"] = (df["Total_Delay"] > 0).astype(int)

            # 2. Fitur Fenomena "Rating 0" (Bypass survey counter)
            existing_survey = [c for c in survey_cols if c in df.columns]
            if existing_survey:
                df["Num_Zero_Ratings"] = (df[existing_survey] == 0).sum(axis=1)

            # 3. Fitur Interaksi Juara: Class x Type of Travel
            if "Class" in df.columns and "Type of Travel" in df.columns:
                df["Class_x_Travel"] = df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str)

            # 4. Klasterisasi Kepuasan Layanan (Online vs Fisik Kabin)
            online_cols = [c for c in ["Inflight wifi service", "Ease of Online booking", "Online boarding"] if c in df.columns]
            if online_cols:
                df["Online_Service_Mean"] = df[online_cols].mean(axis=1)

            cabin_cols = [c for c in ["Seat comfort", "Food and drink", "Cleanliness", "Inflight entertainment"] if c in df.columns]
            if cabin_cols:
                df["Cabin_Comfort_Mean"] = df[cabin_cols].mean(axis=1)

        return train_df, test_df