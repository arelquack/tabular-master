import pandas as pd
import numpy as np
from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook

class SpaceshipDomainFeatures(BaseFeatureHook):
    """
    Contoh Domain Feature Engineering khusus dataset Spaceship Titanic.
    Semua hal spesifik dataset cukup diletakkan di sini.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]

        amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

        for df in dfs:
            # 1. Total Spending across luxury amenities
            df["TotalSpend"] = df[amenities].sum(axis=1)
            df["ZeroSpend"] = (df["TotalSpend"] == 0).astype(int)

            # 2. Extract Deck and Side from Cabin (e.g. B/0/P)
            if "Cabin" in df.columns:
                cabin_split = df["Cabin"].str.split("/", expand=True)
                df["CabinDeck"] = cabin_split[0]
                df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
                df["CabinSide"] = cabin_split[2]

            # 3. Passenger Group ID and Group Size
            if "PassengerId" in df.columns:
                df["GroupId"] = df["PassengerId"].str.split("_").str[0]
                group_counts = df["GroupId"].map(df["GroupId"].value_counts())
                df["GroupSize"] = group_counts
                df["IsAlone"] = (group_counts == 1).astype(int)

        return train_df, test_df


def main():
    print("Loading Spaceship Titanic Data...")
    train_df = pd.read_csv("spaceship-titanic/train.csv")
    test_df = pd.read_csv("spaceship-titanic/test.csv")

    # Konfigurasi pipeline: hanya perlu diatur sedikit sesuai kompetisi
    config = PipelineConfig(
        project_name="spaceship_titanic_experiment",
        output_dir="experiments/spaceship_run",
        id_col="PassengerId",
        target_col="Transported",
        task_type="binary",
        eval_metric="accuracy",
        cv_strategy="stratified_kfold",
        n_splits=5,
        seed=42,
        models=["lgbm", "catboost", "xgb"],
        run_adversarial_validation=True,
        run_generic_features=True,
        drop_cols=["Name", "Cabin"],  # Drop kolom raw yang sudah diekstrak
    )

    # Inisialisasi pipeline dengan domain hook
    pipeline = TabularPipeline(config=config, feature_hook=SpaceshipDomainFeatures())

    # Jalankan end-to-end!
    results = pipeline.run(train_df=train_df, test_df=test_df)
    print("\n[SUCCESS] Pipeline Run Complete!")


if __name__ == "__main__":
    main()
