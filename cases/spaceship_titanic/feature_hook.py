import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class CustomFeatures(BaseFeatureHook):
    """
    Feature engineering hook untuk Spaceship Titanic.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]
        amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

        for df in dfs:
            # 1. Total & Has Spent
            existing_amenities = [c for c in amenities if c in df.columns]
            if existing_amenities:
                df["TotalSpending"] = df[existing_amenities].sum(axis=1)
                df["HasSpent"] = (df["TotalSpending"] > 0).astype(int)

            # 2. Cabin Parsing (Deck/Num/Side)
            if "Cabin" in df.columns:
                cabin_split = df["Cabin"].str.split("/", expand=True)
                if cabin_split.shape[1] == 3:
                    df["Cabin_Deck"] = cabin_split[0]
                    df["Cabin_Side"] = cabin_split[2]

            # 3. Group ID from PassengerId
            if "PassengerId" in df.columns:
                df["Group_Id"] = df["PassengerId"].apply(lambda x: str(x).split("_")[0] if pd.notnull(x) else np.nan)

        return train_df, test_df
