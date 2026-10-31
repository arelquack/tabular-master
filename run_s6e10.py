import os
import sys
import pandas as pd
import numpy as np

# Pastikan path folder root proyek terdeteksi meskipun dijalankan dari subfolder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook

class AirlineFeatures(BaseFeatureHook):
    """
    Rekayasa fitur khusus untuk Airline Passenger Satisfaction (s6e10).
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
            # 1. Imputasi Keterlambatan secara fisik (r > 0.96)
            if "Arrival Delay in Minutes" in df.columns and "Departure Delay in Minutes" in df.columns:
                df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
                df["Total_Delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
                df["Has_Delay"] = (df["Total_Delay"] > 0).astype(int)

            # 2. Fitur "Rating 0" (Indikator Bypass Survey / Not Applicable)
            existing_survey = [c for c in survey_cols if c in df.columns]
            if existing_survey:
                df["Num_Zero_Ratings"] = (df[existing_survey] == 0).sum(axis=1)

            # 3. Fitur Interaksi Juara: Class x Type of Travel
            if "Class" in df.columns and "Type of Travel" in df.columns:
                df["Class_x_Travel"] = df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str)

        return train_df, test_df


def main():
    print("[1/3] Memuat Dataset Airline Satisfaction...")
    train_path = os.path.join(ROOT_DIR, "playground-series", "playground-series-s6e10", "train.csv")
    test_path = os.path.join(ROOT_DIR, "playground-series", "playground-series-s6e10", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Konfigurasi pipeline lengkap
    config = PipelineConfig(
        project_name="s6e10_airline_champion",
        output_dir="experiments/s6e10_champion_run",
        seed=2026,
        
        # Data
        id_col="id",
        target_col="satisfaction",
        group_col=None,
        time_col=None,
        drop_cols=[],
        
        # Task & Metrik
        task_type="binary",
        eval_metric="roc_auc",             # Metrik kompetisi
        optimize_threshold=False,          # Output probabilitas kontinu
        
        # Validasi
        cv_strategy="stratified_kfold",
        n_splits=5,
        shuffle=True,
        
        # Fitur
        run_generic_features=False,        # Pakai fitur domain terarah saja
        create_missing_flags=False,
        create_freq_encoding=False,
        create_group_aggregations=False,
        
        # Preprocessing & Model
        categorical_encoder="ordinal",
        handle_missing="auto",
        run_adversarial_validation=True,
        
        # Model Parameters (Tuned untuk 700k data)
        models=["lgbm", "xgb", "catboost"],
        model_params={
            "lgbm": {
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "num_leaves": 127,
                "min_child_samples": 200,
                "colsample_bytree": 0.70,
                "subsample": 0.80,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
            },
            "xgb": {
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "max_depth": 7,
                "min_child_weight": 50,
                "colsample_bytree": 0.70,
                "subsample": 0.80,
                "tree_method": "hist",
            },
            "catboost": {
                "iterations": 1000,
                "learning_rate": 0.07,
                "depth": 6,
                "l2_leaf_reg": 5.0,
            }
        },
        
        # Turnamen Ensembling
        ensemble_methods=["mean", "rank", "optimal_weights", "stacking", "hill_climbing"],
        save_oof_predictions=True,
        save_test_predictions=True,
    )

    print("[2/3] Menjalankan Pipeline Tabular...")
    pipeline = TabularPipeline(config=config, feature_hook=AirlineFeatures())
    results = pipeline.run(train_df=train_df, test_df=test_df)

    print("\n[3/3] Selesai!")
    print(f"Champion Ensemble : {results['champion_model']}")
    print(f"Skor ROC-AUC OOF  : {results['champion_score']:.5f}")
    print(f"File Hasil        : {os.path.join(config.output_dir, 'submission.csv')}")


if __name__ == "__main__":
    main()
