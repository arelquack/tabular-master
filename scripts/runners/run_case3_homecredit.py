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

class HomeCreditSOTAFeatures(BaseFeatureHook):
    """
    Kaggle Grandmaster Domain Feature Engineering for Home Credit Default Risk.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]

        for df in dfs:
            # 1. Fix DAYS_EMPLOYED 365243 Anomaly
            if "DAYS_EMPLOYED" in df.columns:
                df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
                df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

            # 2. Age in Years
            if "DAYS_BIRTH" in df.columns:
                df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25

            # 3. 4 Pillar Financial Ratios
            if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
                df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1.0)

            if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
                df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1.0)

            if "AMT_INCOME_TOTAL" in df.columns and "CNT_FAM_MEMBERS" in df.columns:
                df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1.0)

            if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
                df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1.0)

            if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
                df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1.0)

            # 4. Composite External Sources (r = -0.222 correlation with default)
            ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
            if len(ext_cols) == 3:
                df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
                df["EXT_SOURCES_STD"] = df[ext_cols].std(axis=1).fillna(0)
                df["EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

        return train_df, test_df


def main():
    print("[CASE 3] Home Credit Default Risk - High Performance Run")
    train_df = pd.read_csv("home-credit/application_train.csv")
    test_df = pd.read_csv("home-credit/application_test.csv")

    config = PipelineConfig(
        project_name="home_credit_sota",
        output_dir="experiments/case3_homecredit",
        id_col="SK_ID_CURR",
        target_col="TARGET",
        task_type="binary",
        eval_metric="roc_auc",
        cv_strategy="stratified_kfold",
        n_splits=5,
        seed=42,
        models=["lgbm", "catboost", "xgb"],
        run_adversarial_validation=True,
        run_generic_features=False,  # High-dimensional data (122 cols), domain features prevent dilution
        model_params={
            "lgbm": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_samples": 50,
            },
            "catboost": {
                "iterations": 800,
                "learning_rate": 0.07,
                "depth": 5,
                "l2_leaf_reg": 5.0,
            },
            "xgb": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "tree_method": "hist",
            }
        }
    )

    pipeline = TabularPipeline(config=config, feature_hook=HomeCreditSOTAFeatures())
    results = pipeline.run(train_df=train_df, test_df=test_df)
    print(f"\n>>> CASE 3 Champion: {results['champion_model']} | Best Score: {results['champion_score']:.5f}")


if __name__ == "__main__":
    main()
