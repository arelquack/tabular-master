# Universal Tabular Pipeline Framework
Sebuah framework machine learning modular & production-ready yang dirancang khusus untuk kompetisi tabular (Kaggle, Gemastik) dan tugas data mining industri.

---

## 🎯 Mengapa Menggunakan Framework Ini?
Pada kasus data tabular, **85%+ tahapan selalu repetitif**:
- Pengecekan data leakage & split validasi
- Adversarial validation (deteksi train-test shift)
- Pembuatan fitur generik (frequency encoding, aggregasi grup, is_nan flag)
- Melatih trio model GBDT terbaik (**LightGBM, XGBoost, CatBoost**)
- Out-of-Fold (OOF) cross-validation
- Blending & Ensembling otomatis (Mean, Rank Average, Optimal Weight SLSQP, Stacking)
- Post-processing (Threshold Optimization)
- Pembuatan file submit & log eksperimen

Dengan framework ini, Anda **hanya perlu fokus pada 15% sisanya**, yaitu **Domain Feature Engineering** dan konfigurasi kasus.

---

## 📂 Struktur Arsitektur

```text
tabular_pipeline/
│
├── __init__.py             # Entry point package
├── config.py               # PipelineConfig dataclass
├── data_detector.py        # Otomatis mendeteksi tipe kolom (num, cat, datetime, text, missing)
├── feature_engine.py       # Generic Feature Engine + BaseFeatureHook (Custom domain logic)
├── diagnostics.py          # Adversarial Validation (mengecek train-test drift)
├── encoders.py             # Leak-free Ordinal & Target Encoders (fit hanya di dalam fold train)
├── validation.py           # Cross-validation factory (Stratified, Group, TimeSeries, KFold)
├── metrics.py              # Evaluator metrik lengkap + Binary Threshold Optimizer
│
├── models/                 # Unified Wrapper dengan interface seragam
│   ├── base.py             # Abstract BaseTabularModel
│   ├── lgbm_model.py       # LightGBM Classifier / Regressor
│   ├── xgb_model.py        # XGBoost Classifier / Regressor
│   ├── cat_model.py        # CatBoost Classifier / Regressor
│   ├── sklearn_model.py    # RF, ExtraTrees, Ridge, LogisticRegression
│   └── factory.py          # ModelFactory
│
├── trainer.py              # Out-Of-Fold (OOF) CV training loop
├── ensemble.py             # Blending (Mean, Rank, SLSQP Weights, Stacking)
└── pipeline.py             # Master TabularPipeline orchestrator
```

---

## 🚀 Cara Menggunakan untuk Kasus Baru (3 Langkah Mudah)

### Langkah 1: Buat File Script Kasus Baru (misal: `run_my_task.py`)

```python
import pandas as pd
from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook

# 1. Load Data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Definisikan Domain Features (Jika Ada)
class MyDomainFeatures(BaseFeatureHook):
    def transform(self, train_df, test_df=None):
        dfs = [train_df] if test_df is None else [train_df, test_df]
        for df in dfs:
            # Tulis fitur kreatif Anda di sini:
            if "income" in df.columns and "loan_amount" in df.columns:
                df["debt_to_income"] = df["loan_amount"] / (df["income"] + 1)
        return train_df, test_df

# 3. Konfigurasi Kasus
config = PipelineConfig(
    project_name="my_competition",
    output_dir="experiments/exp1",
    id_col="id",
    target_col="target",
    task_type="binary",             # binary | multiclass | regression
    eval_metric="roc_auc",          # roc_auc | accuracy | f1_macro | rmse | logloss
    cv_strategy="stratified_kfold", # stratified_kfold | group_kfold | time_series
    n_splits=5,
    models=["lgbm", "catboost", "xgb"],
    run_adversarial_validation=True,
    run_generic_features=True,
)

# 4. Jalankan Pipeline!
pipeline = TabularPipeline(config=config, feature_hook=MyDomainFeatures())
results = pipeline.run(train_df=train_df, test_df=test_df)
```

---

## 📊 Output Otomatis di Folder `experiments/<project_name>/`
Setelah pipeline selesai, framework otomatis menghasilkan:
1. `submission.csv`: File hasil prediksi yang langsung siap di-submit ke sistem lomba / Kaggle.
2. `oof_champion.csv`: Out-of-fold predictions untuk verifikasi akurasi lokal.
3. `feature_importances.csv`: Peringkat fitur paling berpengaruh di seluruh fold dan model.
4. `config.json`: Konfigurasi eksperimen untuk reproducibility.
5. `summary.json`: Ringkasan skor per model, skor ensemble terbaik, threshold optimal, dan waktu eksekusi.
