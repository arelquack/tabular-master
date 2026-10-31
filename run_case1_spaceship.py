import os
import pandas as pd
import numpy as np
from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook

class SpaceshipSOTAFeatures(BaseFeatureHook):
    """
    Kaggle Grandmaster Level Feature Engineering for Spaceship Titanic.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]
        amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

        for df in dfs:
            # 1. Total & Binary Spending
            df["TotalSpend"] = df[amenities].sum(axis=1)
            df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(int)
            df["HasRoomService"] = (df["RoomService"] > 0).astype(int)
            df["HasFoodCourt"] = (df["FoodCourt"] > 0).astype(int)
            df["HasShoppingMall"] = (df["ShoppingMall"] > 0).astype(int)
            df["HasSpa"] = (df["Spa"] > 0).astype(int)
            df["HasVRDeck"] = (df["VRDeck"] > 0).astype(int)

            # Spending proportions
            for col in amenities:
                df[f"{col}_Ratio"] = df[col] / (df["TotalSpend"] + 1.0)

            # 2. Cabin Parsing
            if "Cabin" in df.columns:
                cabin_split = df["Cabin"].str.split("/", expand=True)
                df["CabinDeck"] = cabin_split[0]
                df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
                df["CabinSide"] = cabin_split[2]
                df["CabinRegion"] = df["CabinNum"] // 300  # Deck regional groupings

            # 3. Passenger Group ID & Group Dynamics
            if "PassengerId" in df.columns:
                df["GroupId"] = df["PassengerId"].str.split("_").str[0]
                group_counts = df["GroupId"].map(df["GroupId"].value_counts())
                df["GroupSize"] = group_counts
                df["IsAlone"] = (group_counts == 1).astype(int)

            # 4. Family Name extraction
            if "Name" in df.columns:
                df["LastName"] = df["Name"].str.split(" ").str[-1]
                family_counts = df["LastName"].map(df["LastName"].value_counts())
                df["FamilySize"] = family_counts.fillna(1)

            # 5. Age Groups
            if "Age" in df.columns:
                df["IsChild"] = (df["Age"] <= 12).astype(int)
                df["IsTeen"] = ((df["Age"] > 12) & (df["Age"] <= 18)).astype(int)
                df["IsElderly"] = (df["Age"] >= 60).astype(int)

            # 6. Interaction: CryoSleep + Spending
            if "CryoSleep" in df.columns:
                df["CryoSleep_x_ZeroSpend"] = (df["CryoSleep"].astype(str) + "_" + df["ZeroSpend"].astype(str))

        return train_df, test_df


def main():
    print("[CASE 1] Spaceship Titanic - High Performance Run")
    train_df = pd.read_csv("spaceship-titanic/train.csv")
    test_df = pd.read_csv("spaceship-titanic/test.csv")

    config = PipelineConfig(
        project_name="spaceship_titanic_sota",
        output_dir="experiments/case1_spaceship",
        id_col="PassengerId",
        target_col="Transported",
        task_type="binary",
        eval_metric="accuracy",
        cv_strategy="stratified_kfold",
        n_splits=5,
        seed=2026,
        models=["lgbm", "catboost", "xgb"],
        run_adversarial_validation=True,
        run_generic_features=True,
        optimize_threshold=True,
        drop_cols=["Name", "Cabin", "LastName"],
        model_params={
            "lgbm": {
                "n_estimators": 2500,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "min_child_samples": 20,
            },
            "catboost": {
                "iterations": 2500,
                "learning_rate": 0.02,
                "depth": 6,
                "l2_leaf_reg": 6.0,
            },
            "xgb": {
                "n_estimators": 2500,
                "learning_rate": 0.02,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            }
        }
    )

    pipeline = TabularPipeline(config=config, feature_hook=SpaceshipSOTAFeatures())
    results = pipeline.run(train_df=train_df, test_df=test_df)
    print(f"\n>>> CASE 1 Champion: {results['champion_model']} | Best Score: {results['champion_score']:.5f}")


if __name__ == "__main__":
    main()
