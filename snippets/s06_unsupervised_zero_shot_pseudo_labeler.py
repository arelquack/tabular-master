"""
SNIPPET 06: Unsupervised to Supervised Converter (Zero-Shot Pseudo-Labeler)
Deskripsi: Mengubah dataset teks tanpa label (unsupervised) menjadi dataset terawasi (supervised)
menggunakan Pretrained RoBERTa Teacher Model dengan penyaringan tingkat keyakinan (Confidence Filter).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train_transformer_universal import resolve_model_path


def run_pseudo_labeler(
    input_csv: str,
    output_csv: str,
    text_col: str = "clean_text",
    model_name: str = "w11wo/indonesian-roberta-base-sentiment-classifier",
    confidence_threshold: float = 0.80,
    batch_size: int = 32
):
    print("=" * 65)
    print(" UNSUPERVISED TO SUPERVISED PSEUDO-LABELER")
    print("=" * 65)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    df = pd.read_csv(input_csv)
    print(f"Data Mentah Unlabelled : {input_csv} ({len(df):,} baris)")
    print(f"Kolom Teks Target      : '{text_col}'")
    print(f"Model Pretrained Guru   : {model_name}")
    print(f"Confidence Threshold   : >= {confidence_threshold * 100:.0f}%")

    resolved_model = resolve_model_path(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device Komputasi       : {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_model).to(device)
    model.eval()

    id2label = model.config.id2label or {0: "Negatif", 1: "Netral", 2: "Positif"}
    texts = df[text_col].fillna("").astype(str).tolist()

    all_preds = []
    all_confs = []

    print("Menjalankan zero-shot teacher inference...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            all_preds.extend([id2label[pid] for pid in pred_ids])
            all_confs.extend(confs)

    df["pseudo_label"] = all_preds
    df["confidence"] = np.round(all_confs, 4)

    # Filter data dengan confidence tinggi untuk mencegah label noise
    high_conf_df = df[df["confidence"] >= confidence_threshold].copy()
    high_conf_df["sentiment"] = high_conf_df["pseudo_label"]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    high_conf_df.to_csv(output_csv, index=False)

    print("\nHASIL GENERASI SUPERVISED DATASET:")
    print(f"Total Sampel Mentah    : {len(df):,} baris")
    print(f"Lolos Filter Keyakinan : {len(high_conf_df):,} baris ({len(high_conf_df)/len(df)*100:.1f}%)")
    print("\nDistribusi Kelas Supervised Baru:")
    print(high_conf_df["pseudo_label"].value_counts())
    print(f"\nDataset Supervised siap pakai tersimpan di '{output_csv}'!")
    print("Sekarang Anda dapat memasukkan file ini ke Tab 1 Studio.")
    return high_conf_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Pseudo-Labeler")
    parser.add_argument("--src", type=str, required=True, help="Path ke CSV teks unlabelled")
    parser.add_argument("--out", type=str, default="data/supervised_train.csv")
    parser.add_argument("--text_col", type=str, default="clean_text")
    parser.add_argument("--model", type=str, default="w11wo/indonesian-roberta-base-sentiment-classifier")
    parser.add_argument("--thresh", type=float, default=0.80)
    args = parser.parse_args()

    run_pseudo_labeler(
        input_csv=args.src,
        output_csv=args.out,
        text_col=args.text_col,
        model_name=args.model,
        confidence_threshold=args.thresh
    )
