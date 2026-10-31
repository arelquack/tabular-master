import pandas as pd
import numpy as np

# 1. Import komponen dari framework kita
from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook


# 2. Definisikan Domain Features (Rahasia Fitur Juara)
class AirlineFeatures(BaseFeatureHook):
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
            # Imputasi Keterlambatan secara fisik (r > 0.96)
            if "Arrival Delay in Minutes" in df.columns and "Departure Delay in Minutes" in df.columns:
                df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
                df["Total_Delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
                df["Has_Delay"] = (df["Total_Delay"] > 0).astype(int)

            # Fitur "Rating 0" (Indikator Bypass Survey)
            existing_survey = [c for c in survey_cols if c in df.columns]
            if existing_survey:
                df["Num_Zero_Ratings"] = (df[existing_survey] == 0).sum(axis=1)

            # Fitur Interaksi Juara: Class x Type of Travel
            if "Class" in df.columns and "Type of Travel" in df.columns:
                df["Class_x_Travel"] = df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str)

        return train_df, test_df


# 3. Main Function & Konfigurasi
def main():
    print("1. Memuat dataset...")
    train_df = pd.read_csv("playground-series/playground-series-s6e10/train.csv")
    test_df = pd.read_csv("playground-series/playground-series-s6e10/test.csv")

    # Di sini Anda mengisi konfigurasi eksplisit Anda:
    config = PipelineConfig(
        project_name="s6e10_airline_champion",
        output_dir="experiments/s6e10_champion_run",
        seed=2026,
        
        # Definisi data
        id_col="id",
        target_col="satisfaction",
        task_type="binary",
        eval_metric="roc_auc",             # Metrik kompetisi
        optimize_threshold=False,          # Output probabilitas kontinu
        
        # Validasi
        cv_strategy="stratified_kfold",
        n_splits=5,
        shuffle=True,
        
        # Fitur
        run_generic_features=False,        # Pakai fitur domain terarah saja
        
        # Model & Parameter tuned
        models=["lgbm", "xgb", "catboost"],
        model_params={
            "lgbm": {
                "n_estimators": 1500,
                "learning_rate": 0.04,
                "num_leaves": 127,
                "min_child_samples": 200,
                "colsample_bytree": 0.70,
                "subsample": 0.80,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
            "xgb": {
                "n_estimators": 1500,
                "learning_rate": 0.04,
                "max_depth": 7,
                "min_child_weight": 50,
                "colsample_bytree": 0.70,
                "subsample": 0.80,
                "tree_method": "hist",
            },
            "catboost": {
                "iterations": 1200,
                "learning_rate": 0.06,
                "depth": 6,
                "l2_leaf_reg": 5.0,
            }
        },
        
        # Turnamen Ensembling
        ensemble_methods=["mean", "rank", "optimal_weights", "stacking", "hill_climbing"],
        save_oof_predictions=True,
        save_test_predictions=True,
    )

    # 4. Inisialisasi dan Jalankan!
    print("2. Menjalankan pipeline otomatis...")
    pipeline = TabularPipeline(config=config, feature_hook=AirlineFeatures())
    results = pipeline.run(train_df=train_df, test_df=test_df)

    print("\n[SELESAI!]")
    print(f"Metrik Evaluasi   : {results['metric']}")
    print(f"Strategi Champion : {results['champion_model']}")
    print(f"Skor OOF Terbaik  : {results['champion_score']:.5f}")


if __name__ == "__main__":
    main()