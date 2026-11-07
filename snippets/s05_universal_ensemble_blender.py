"""
SNIPPET 05: Universal Probability Blending & Ensemble Optimizer
Deskripsi: Menggabungkan prediksi out-of-fold (OOF) probabilitas dari dua model atau lebih
(misal TF-IDF Baseline + Transformer BERT) untuk mencari bobot optimal dan mencetak skor tertinggi.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def optimize_blend_weights(probs_list, y_true):
    """Mencari bobot optimal via SLSQP yang memaksimalkan Macro-F1."""
    n_models = len(probs_list)
    
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blended = np.zeros_like(probs_list[0])
        for i in range(n_models):
            blended += w[i] * probs_list[i]
        preds = np.argmax(blended, axis=1)
        # Negatif Macro-F1 untuk minimisasi
        return -f1_score(y_true, preds, average="macro", zero_division=0)

    init_weights = [1.0 / n_models] * n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    optimal_w = res.x / np.sum(res.x)
    return optimal_w


def run_ensemble_blender(case_dir: str, custom_weight_model1: float = None):
    print("=" * 65)
    print(" UNIVERSAL ENSEMBLE PROBABILITY BLENDER")
    print("=" * 65)

    baseline_oof_p = os.path.join(case_dir, "baseline_oof_probs.npy")
    baseline_idx_p = os.path.join(case_dir, "baseline_oof_indices.npy")
    trans_oof_p = os.path.join(case_dir, "transformer_val_probs.npy")
    trans_idx_p = os.path.join(case_dir, "transformer_val_indices.npy")
    trans_tgt_p = os.path.join(case_dir, "transformer_val_targets.npy")

    if not os.path.exists(baseline_oof_p) or not os.path.exists(trans_oof_p):
        raise FileNotFoundError(f"Array prediksi OOF belum lengkap di '{case_dir}'. Jalankan Snippet 03 dan 04 terlebih dahulu.")

    baseline_probs = np.load(baseline_oof_p)
    baseline_indices = np.load(baseline_idx_p)
    trans_probs = np.load(trans_oof_p)
    trans_indices = np.load(trans_idx_p)
    trans_targets = np.load(trans_tgt_p)

    # Sinkronisasi indeks validasi
    common_indices, b_idx, t_idx = np.intersect1d(baseline_indices, trans_indices, return_indices=True)
    b_common = baseline_probs[b_idx]
    t_common = trans_probs[t_idx]
    y_common = trans_targets[t_idx]

    b_f1 = f1_score(y_common, np.argmax(b_common, axis=1), average="macro", zero_division=0)
    t_f1 = f1_score(y_common, np.argmax(t_common, axis=1), average="macro", zero_division=0)

    print(f"Sampel Evaluasi Bersama : {len(common_indices)} baris")
    print(f"Skor Model 1 (TF-IDF)   : {b_f1 * 100:.2f}%")
    print(f"Skor Model 2 (Transf)   : {t_f1 * 100:.2f}%")

    if custom_weight_model1 is not None:
        w1 = custom_weight_model1
        w2 = 1.0 - w1
        print(f"\nMenggunakan Custom Weights: Model1={w1:.2f}, Model2={w2:.2f}")
    else:
        print("\nMengoptimasi bobot fusion menggunakan algoritma SLSQP...")
        optimal_w = optimize_blend_weights([b_common, t_common], y_common)
        w1, w2 = optimal_w[0], optimal_w[1]

    blended_probs = w1 * b_common + w2 * t_common
    ensemble_preds = np.argmax(blended_probs, axis=1)
    ens_f1 = f1_score(y_common, ensemble_preds, average="macro", zero_division=0)
    ens_acc = accuracy_score(y_common, ensemble_preds)

    gain = (ens_f1 - max(b_f1, t_f1)) * 100
    print("\nHASIL AKHIR ENSEMBLE CHAMPION:")
    print(f"Optimal Weights : {w1*100:.1f}% Model 1 + {w2*100:.1f}% Model 2")
    print(f"Ensemble Macro-F1 : {ens_f1 * 100:.2f}%")
    print(f"Ensemble Accuracy : {ens_acc * 100:.2f}%")
    print(f"Relative Gain     : +{gain:.2f}% vs single best")

    # Generate test submission jika test probs tersedia
    b_test_p = os.path.join(case_dir, "baseline_test_probs.npy")
    t_test_p = os.path.join(case_dir, "transformer_test_probs.npy")
    classes_p = os.path.join(case_dir, "baseline_classes.npy")

    if os.path.exists(b_test_p) and os.path.exists(t_test_p) and os.path.exists(classes_p):
        b_test = np.load(b_test_p)
        t_test = np.load(t_test_p)
        classes = np.load(classes_p, allow_pickle=True)
        blended_test = w1 * b_test + w2 * t_test
        test_pred_labels = [classes[idx] for idx in np.argmax(blended_test, axis=1)]

        sub_df = pd.DataFrame({"id": range(len(test_pred_labels)), "label": test_pred_labels})
        out_sub = os.path.join(case_dir, "submission_ensemble.csv")
        sub_df.to_csv(out_sub, index=False)
        print(f"\nSubmission akhir tersimpan di '{out_sub}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Ensemble Probability Blender")
    parser.add_argument("--case_dir", type=str, required=True, help="Direktori cases/<case_id>")
    parser.add_argument("--w1", type=float, default=None, help="Bobot kustom model 1 (0.0 - 1.0)")
    args = parser.parse_args()

    run_ensemble_blender(case_dir=args.case_dir, custom_weight_model1=args.w1)
