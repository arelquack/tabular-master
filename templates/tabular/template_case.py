"""
Template Skrip untuk Kasus Tabular Baru (Kaggle / Gemastik)
Cukup duplikasi skrip ini, sesuaikan nama kolom dan metrik di CONFIG, lalu jalankan!
"""

import pandas as pd
from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook

# ==========================================
# 1. DOMAIN FEATURE ENGINEERING (OPSIONAL)
# ==========================================
class CaseDomainFeatures(BaseFeatureHook):
    """
    Tulis logika domain knowledge khusus kasus ini di sini.
    Jika tidak ada atau belum sempat buat fitur custom,
    kosongkan saja (framework tetap akan auto-generate fitur generik).
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]
        for df in dfs:
            # Contoh ide fitur domain:
            # df["rasio_keuangan"] = df["hutang"] / (df["pendapatan"] + 1)
            pass
        return train_df, test_df


# ==========================================
# 2. RUN PIPELINE
# ==========================================
def main():
    # Load dataset kasus Anda
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    config = PipelineConfig(
        project_name="my_case_experiment",
        output_dir="experiments/exp_01",
        
        # Kolom kunci
        id_col="id",                   # Nama kolom ID (misal: "id", "PassengerId", "SK_ID_CURR")
        target_col="target",           # Nama kolom target
        drop_cols=[],                  # Kolom teks mentah / uninformative yang mau di-drop
        
        # Tipe tugas & evaluasi
        task_type="binary",            # "binary" | "multiclass" | "regression"
        eval_metric="roc_auc",         # "roc_auc" | "accuracy" | "f1_macro" | "logloss" | "rmse" | "mae"
        
        # Strategi validasi
        cv_strategy="stratified_kfold",# "stratified_kfold" | "kfold" | "group_kfold" | "time_series"
        group_col=None,                # Isi jika cv_strategy = group_kfold
        n_splits=5,
        seed=42,
        
        # Model yang dilatih (GBDT Holy Trinity)
        models=["lgbm", "catboost", "xgb"],
        
        # Fitur otomatis & diagnosa
        run_adversarial_validation=True,
        run_generic_features=True,
        optimize_threshold=True,       # Otomatis tuning threshold probabilitas
    )

    # Inisialisasi dan jalankan
    pipeline = TabularPipeline(config=config, feature_hook=CaseDomainFeatures())
    results = pipeline.run(train_df=train_df, test_df=test_df)

    print("\nSummary Hasil:")
    print(f"Model Terbaik : {results['champion_model']}")
    print(f"Skor OOF      : {results['champion_score']:.5f}")


if __name__ == "__main__":
    main()
