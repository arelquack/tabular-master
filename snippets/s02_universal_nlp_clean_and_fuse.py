"""
SNIPPET 02: Universal NLP Cleaner & Field Fusion
Deskripsi: Menggabungkan beberapa kolom teks (Field Fusion) dan membersihkan teks mentah
dari tag HTML, tautan URL, mention, dateline, serta frasa rekomendasi editorial.
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tabular_pipeline.text_engine import UniversalTextNormalizer, UniversalTextCleaner


def run_clean_and_fuse(
    source_path: str,
    output_path: str,
    columns_to_combine: list = None,
    target_clean_col: str = "clean_text",
    prefix_with_tag: bool = True,
    custom_remove_phrases: list = None
):
    print("=" * 65)
    print(" UNIVERSAL NLP FIELD FUSION & CLEANER")
    print("=" * 65)

    # 1. Normalisasi masukan file mentah
    norm_meta = UniversalTextNormalizer.normalize_to_csv(
        source_path=source_path,
        output_csv_path=output_path
    )
    df = pd.read_csv(output_path)
    print(f"Format Terdeteksi : {norm_meta['detected_format']}")
    print(f"Total Dokumen     : {len(df):,} baris")

    # 2. Field Fusion
    if not columns_to_combine:
        # Auto detect semua kolom teks bertipe object/string kecuali ID dan label
        cols_to_fuse = []
        for c in df.columns:
            if c not in [norm_meta.get("detected_id_col"), norm_meta.get("detected_label_col")]:
                if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
                    cols_to_fuse.append(c)
        columns_to_combine = cols_to_fuse or [norm_meta["detected_text_col"]]

    print(f"Kolom Digabungkan : {columns_to_combine} (Prefix Tag: {prefix_with_tag})")

    def fuse_row(row):
        parts = []
        for col in columns_to_combine:
            val = str(row[col]) if pd.notna(row[col]) else ""
            if val.strip():
                if prefix_with_tag:
                    tag = f"[{col.upper()}] "
                    parts.append(f"{tag}{val}")
                else:
                    parts.append(val)
        return " ".join(parts)

    df[target_clean_col] = df.apply(fuse_row, axis=1)

    # 3. Text Cleaning
    phrases = custom_remove_phrases or ["baca juga:...", "simak video:...", "baca selengkapnya:..."]
    print(f"Menjalankan pembersihan korpus & filter frasa: {phrases}...")
    df[target_clean_col] = df[target_clean_col].apply(
        lambda t: UniversalTextCleaner.clean(
            t,
            lower=True,
            strip_urls=True,
            strip_html=True,
            strip_dateline=True,
            reduce_repeated_chars=True,
            custom_remove_phrases=phrases
        )
    )

    df.to_csv(output_path, index=False)
    print(f"SELESAI: Korpus bersih tersimpan di '{output_path}'.")
    print(f"Sampel Teks Hasil:\n{df[target_clean_col].iloc[0][:200]}...")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal NLP Field Fusion & Cleaner")
    parser.add_argument("--src", type=str, required=True, help="Path ke data mentah (CSV, JSONL, folder txt, dll)")
    parser.add_argument("--out", type=str, default="data/clean_train.csv", help="Path CSV output")
    args = parser.parse_args()

    run_clean_and_fuse(source_path=args.src, output_path=args.out)
