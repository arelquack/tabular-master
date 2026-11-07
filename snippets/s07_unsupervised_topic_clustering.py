"""
SNIPPET 07: Unsupervised Topic Clustering & Keyword Extraction (c-TF-IDF)
Deskripsi: Mengelompokkan teks tanpa label ke dalam kluster topik menggunakan
Transformer Dense Embeddings + K-Means, serta mengekstrak kata kunci representatif per topik via c-TF-IDF.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tabular_pipeline.text_engine import TextFeatureExtractor


def extract_ctfidf_keywords(texts, cluster_labels, top_n=8):
    """Menghitung Class-based TF-IDF (c-TF-IDF) untuk mengekstrak kata kunci topik."""
    df_temp = pd.DataFrame({"text": texts, "cluster": cluster_labels})
    # Gabungkan semua dokumen dalam kluster yang sama menjadi 1 teks besar
    cluster_docs = df_temp.groupby("cluster")["text"].apply(lambda t: " ".join(t)).reset_index()

    vec = TfidfVectorizer(max_features=10000, stop_words=None, ngram_range=(1, 2))
    X = vec.fit_transform(cluster_docs["text"])
    features = np.array(vec.get_feature_names_out())

    keywords_per_cluster = {}
    for i, row in cluster_docs.iterrows():
        c_id = row["cluster"]
        row_vals = X[i].toarray().flatten()
        top_indices = np.argsort(row_vals)[::-1][:top_n]
        keywords_per_cluster[c_id] = [features[idx] for idx in top_indices if row_vals[idx] > 0]

    return keywords_per_cluster


def run_topic_clustering(
    input_csv: str,
    output_csv: str,
    text_col: str = "clean_text",
    n_clusters: int = 5,
    model_name: str = "indobenchmark/indobert-base-p1"
):
    print("=" * 65)
    print(" UNSUPERVISED TOPIC CLUSTERING & C-TF-IDF EXTRACTOR")
    print("=" * 65)

    df = pd.read_csv(input_csv)
    print(f"Dataset Input    : {input_csv} ({len(df):,} dokumen)")
    print(f"Target Kluster   : {n_clusters} Kluster Topik")
    print(f"Model Embeddings : {model_name}")

    print("\n[1/3] Mengekstraksi dense embeddings dari Transformer...")
    try:
        train_emb, _ = TextFeatureExtractor.extract_transformer_embeddings(
            train_df=df,
            test_df=None,
            text_col=text_col,
            model_dir_or_name=model_name,
            batch_size=32,
            max_length=128
        )
        X_vecs = train_emb.values
    except Exception as e:
        print(f"Fallback ke TF-IDF + SVD embeddings karena: {e}")
        train_emb, _ = TextFeatureExtractor.extract_tfidf_lsa(
            train_df=df,
            test_df=None,
            text_col=text_col,
            n_components=64
        )
        X_vecs = train_emb.values

    print(f"Dimensi Vektor   : {X_vecs.shape}")

    print("\n[2/3] Menjalankan reduksi dimensi dan K-Means Clustering...")
    svd = TruncatedSVD(n_components=min(20, X_vecs.shape[1] - 1), random_state=42)
    X_reduced = svd.fit_transform(X_vecs)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)
    df["topic_cluster"] = [f"Topic_{c}" for c in clusters]
    df["cluster_id"] = clusters

    print("\n[3/3] Mengekstraksi kata kunci representatif per topik via c-TF-IDF...")
    texts = df[text_col].fillna("").astype(str).values
    topic_keywords = extract_ctfidf_keywords(texts, df["topic_cluster"].values, top_n=6)

    print("\nRINGKASAN TOPIK YANG DITEMUKAN:")
    print("-" * 65)
    for t_name, words in topic_keywords.items():
        count = (df['topic_cluster'] == t_name).sum()
        pct = count / len(df) * 100
        print(f"{t_name:<10} ({count:>4} docs, {pct:>5.1f}%): {', '.join(words)}")
    print("-" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nDataset dengan label kluster tersimpan di '{output_csv}'!")
    print("Sekarang dataset memiliki kolom target 'topic_cluster' dan siap diproses di Tab 1 Studio.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised Topic Clustering")
    parser.add_argument("--src", type=str, required=True, help="Path ke CSV teks unlabelled")
    parser.add_argument("--out", type=str, default="data/topics_clustered_train.csv")
    parser.add_argument("--text_col", type=str, default="clean_text")
    parser.add_argument("--clusters", type=int, default=5)
    args = parser.parse_args()

    run_topic_clustering(
        input_csv=args.src,
        output_csv=args.out,
        text_col=args.text_col,
        n_clusters=args.clusters
    )
