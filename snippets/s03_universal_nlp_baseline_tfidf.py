"""
SNIPPET 03: Universal NLP Baseline Trainer (TF-IDF + Ridge / Balanced LR)
Deskripsi: Membangun model baseline teks cepat dan solid menggunakan TF-IDF N-Grams
dengan validasi silang Stratified K-Fold serta pencarian parameter otomatis (Ablasi Top 5).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tabular_pipeline.text_engine import NLPBaselineTrainer


def run_baseline_training(
    train_path: str,
    test_path: str = None,
    text_col: str = None,
    label_col: str = None,
    auto_tune: bool = True,
    output_dir: str = "experiments/nlp_baseline_run"
):
    print("=" * 65)
    print(" UNIVERSAL NLP BASELINE TRAINER (TF-IDF + BALANCED LR)")
    print("=" * 65)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path and os.path.exists(test_path) else None

    # Auto detect text & label columns if not passed
    if not text_col:
        for c in ["clean_text", "text", "komentar", "content", "document", "review"]:
            if c in train_df.columns:
                text_col = c
                break
        if not text_col:
            text_col = train_df.columns[1]

    if not label_col:
        for c in ["sentiment", "sentimen", "category", "kategori", "label", "target", "class", "target_biro"]:
            for col in train_df.columns:
                if col.lower() == c:
                    label_col = col
                    break
            if label_col:
                break
        if not label_col:
            candidates = [col for col in train_df.columns if col != text_col]
            label_col = candidates[-1] if candidates else train_df.columns[-1]

    print(f"Data Train : {train_path} ({len(train_df):,} baris)")
    print(f"Text Column: '{text_col}' | Target Column: '{label_col}'")

    os.makedirs(output_dir, exist_ok=True)

    if auto_tune:
        print("\nMenjalankan Auto-Tune Grid Ablation untuk mencari kombinasi parameter terbaik...")
        res = NLPBaselineTrainer.tune_baseline(
            train_df=train_df,
            test_df=test_df,
            text_col=text_col,
            label_col=label_col,
            n_splits=3
        )
        print("\nMATRIKS ABLASI TOP 5 SETTINGAN TERBAIK:")
        print(f"{'Rank':<6} {'Score (F1)':<12} {'Accuracy':<10} {'N-Gram':<10} {'Reg C':<8} {'Max Feat':<10}")
        print("-" * 60)
        for r in res.get("top_5_ablation", []):
            print(f"#{r['rank']:<5} {r['score']*100:<11.2f}% {r['accuracy']*100:<9.1f}% {r['ngram_range']:<10} {r['c_param']:<8} {r['max_features']:<10}")
    else:
        res = NLPBaselineTrainer.train_baseline(
            train_df=train_df,
            test_df=test_df,
            text_col=text_col,
            label_col=label_col,
            n_splits=5,
            max_features=25000,
            ngram_range=(1, 2),
            c_param=1.5
        )

    print("\nHASIL VALIDASI SILANG BASELINE:")
    print(f"Overall Accuracy : {res.get('overall_accuracy', 0)*100:.2f}%")
    print(f"Overall Macro-F1 : {res.get('overall_macro_f1', 0)*100:.2f}%")
    print(f"Jumlah Fitur     : {res.get('features_extracted', 0):,} N-Grams")

    # Simpan array OOF untuk tahap ensemble blending
    oof_mat = res.get("oof_probs_matrix")
    if oof_mat is not None:
        np.save(os.path.join(output_dir, "baseline_oof_probs.npy"), np.array(oof_mat, dtype=np.float32))
        np.save(os.path.join(output_dir, "baseline_oof_indices.npy"), np.array(res.get("oof_indices")))
        print(f"OOF Probabilities tersimpan di '{output_dir}/baseline_oof_probs.npy'.")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal NLP Baseline Trainer")
    parser.add_argument("--train", type=str, required=True, help="Path ke clean train.csv")
    parser.add_argument("--test", type=str, default=None, help="Path ke test.csv (opsional)")
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--no_tune", action="store_true", help="Lewati auto-tune grid")
    args = parser.parse_args()

    run_baseline_training(
        train_path=args.train,
        test_path=args.test,
        text_col=args.text_col,
        label_col=args.label_col,
        auto_tune=not args.no_tune
    )
