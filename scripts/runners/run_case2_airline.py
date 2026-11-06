import os
import sys

# Ensure project root is in sys.path and is current working directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import pandas as pd
import numpy as np
from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook

class AirlineSOTAFeatures(BaseFeatureHook):
    """
    Kaggle Grandmaster Feature Engineering for Airline Passenger Satisfaction (s6e10).
    """
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
            # 1. Physics-based Delay Imputation (r > 0.96)
            if "Arrival Delay in Minutes" in df.columns and "Departure Delay in Minutes" in df.columns:
                df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
                df["Total Delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
                df["Has Delay"] = (df["Total Delay"] > 0).astype(int)

            # 2. Number of Zero Ratings (Empirical Survey Bypass Indicator)
            existing_survey = [c for c in survey_cols if c in df.columns]
            if existing_survey:
                df["Num_Zero_Ratings"] = (df[existing_survey] == 0).sum(axis=1)

            # 3. High-Value Interaction: Class x Type of Travel
            if "Class" in df.columns and "Type of Travel" in df.columns:
                df["Class_x_Travel"] = df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str)

            # 4. Service Score Aggregation (Amenity comfort)
            comfort_cols = [c for c in ["Seat comfort", "Food and drink", "Cleanliness", "Inflight entertainment"] if c in df.columns]
            if comfort_cols:
                df["Comfort_Mean"] = df[comfort_cols].mean(axis=1)

        return train_df, test_df


def main():
    print("[CASE 2] Airline Passenger Satisfaction - High Performance Run")
    train_df = pd.read_csv("playground-series/playground-series-s6e10/train.csv")
    test_df = pd.read_csv("playground-series/playground-series-s6e10/test.csv")

    config = PipelineConfig(
        project_name="airline_satisfaction_sota",
        output_dir="experiments/case2_airline",
        id_col="id",
        target_col="satisfaction",
        task_type="binary",
        eval_metric="roc_auc",
        cv_strategy="stratified_kfold",
        n_splits=5,
        seed=42,
        models=["lgbm", "xgb", "catboost"],
        run_adversarial_validation=True,
        run_generic_features=False,  # Custom features already refined, prevents dilution
        model_params={
            "lgbm": {
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "num_leaves": 127,
                "min_child_samples": 200,
                "colsample_bytree": 0.7,
                "subsample": 0.8,
            },
            "xgb": {
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "max_depth": 7,
                "min_child_weight": 50,
                "colsample_bytree": 0.7,
                "subsample": 0.8,
                "tree_method": "hist",
            },
            "catboost": {
                "iterations": 1000,
                "learning_rate": 0.07,
                "depth": 6,
                "l2_leaf_reg": 5.0,
            }
        }
    )

    pipeline = TabularPipeline(config=config, feature_hook=AirlineSOTAFeatures())
    results = pipeline.run(train_df=train_df, test_df=test_df)
    print(f"\n>>> CASE 2 Champion: {results['champion_model']} | Best Score: {results['champion_score']:.5f}")


if __name__ == "__main__":
    main()
