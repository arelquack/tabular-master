"""
SNIPPET 01: Universal Tabular Pipeline Runner
Deskripsi: Skrip serbaguna otomatis untuk semua kasus Tabular (Klasifikasi Biner, Multikelas, Regresi).
Auto-detect target, kolom ID, missing values, Auto-FE, leak-free CV, dan Caruana Ensemble.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Pastikan root workspace terdaftar pada sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook


def auto_detect_schema(df: pd.DataFrame, target_col: str = None, id_col: str = None):
    cols = list(df.columns)
    
    # 1. Detect ID Column
    if not id_col:
        for c in ["id", "index", "passengerid", "sk_id_curr", "customer_id", "row_id"]:
            for col in cols:
                if col.lower() == c:
                    id_col = col
                    break
            if id_col:
                break
        if not id_col:
            id_col = cols[0]

    # 2. Detect Target Column
    if not target_col:
        for c in ["target", "label", "class", "survived", "transported", "default", "churn", "price", "score"]:
            for col in cols:
                if col.lower() == c:
                    target_col = col
                    break
            if target_col:
                break
        if not target_col:
            target_col = cols[-1]

    # 3. Detect Task Type
    target_series = df[target_col].dropna()
    n_unique = target_series.nunique()
    if pd.api.types.is_numeric_dtype(target_series) and n_unique > 20:
        task_type = "regression"
        eval_metric = "rmse"
        cv_strategy = "kfold"
    elif n_unique == 2:
        task_type = "binary"
        eval_metric = "roc_auc"
        cv_strategy = "stratified_kfold"
    else:
        task_type = "multiclass"
        eval_metric = "f1_macro"
        cv_strategy = "stratified_kfold"

    return id_col, target_col, task_type, eval_metric, cv_strategy


def run_universal_tabular(train_path: str, test_path: str = None, output_dir: str = "experiments/tabular_run"):
    print("=" * 65)
    print(" UNIVERSAL TABULAR RUNNER")
    print("=" * 65)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File train tidak ditemukan: {train_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path and os.path.exists(test_path) else None

    id_col, target_col, task_type, eval_metric, cv_strategy = auto_detect_schema(train_df)
    print(f"Dataset Train      : {train_path} ({len(train_df):,} baris)")
    print(f"Dataset Test       : {test_path or 'None'} ({len(test_df) if test_df is not None else 0:,} baris)")
    print(f"Terdeteksi ID Col  : {id_col}")
    print(f"Terdeteksi Target  : {target_col}")
    print(f"Tipe Tugas         : {task_type.upper()} | Metrik: {eval_metric.upper()} | CV: {cv_strategy}")

    os.makedirs(output_dir, exist_ok=True)

    config = PipelineConfig(
        project_name="universal_tabular_run",
        output_dir=output_dir,
        id_col=id_col,
        target_col=target_col,
        task_type=task_type,
        eval_metric=eval_metric,
        cv_strategy=cv_strategy,
        n_splits=5,
        seed=2026,
        models=["lgbm", "catboost", "xgb"],
        run_generic_features=True,
        run_adversarial_validation=True if test_df is not None else False,
        optimize_threshold=True if task_type == "binary" else False
    )

    pipeline = TabularPipeline(config=config)
    results = pipeline.run(train_df=train_df, test_df=test_df)

    print("\nHASIL PELATIHAN UNIVERSAL:")
    print(f"Champion Model : {results.get('champion_model')}")
    print(f"Champion Score : {results.get('champion_score', 0):.5f}")
    if test_df is not None:
        sub_p = os.path.join(output_dir, "submission.csv")
        print(f"Submission CSV : {sub_p} (Siap dikumpulkan!)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Tabular Auto-Pipeline")
    parser.add_argument("--train", type=str, required=True, help="Path ke train.csv")
    parser.add_argument("--test", type=str, default=None, help="Path ke test.csv (opsional)")
    parser.add_argument("--out", type=str, default="experiments/tabular_run", help="Direktori output")
    args = parser.parse_args()

    run_universal_tabular(train_path=args.train, test_path=args.test, output_dir=args.out)
