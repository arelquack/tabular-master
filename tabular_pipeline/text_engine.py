import os
import re
import glob
import json
import string
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

class UniversalTextNormalizer:
    """
    Universal Ingestion & Normalizer for Text Datasets.
    Automatically identifies and parses:
    1. CSV / TSV with auto-delimiter detection (comma, tab, semicolon, pipe)
    2. JSON / JSONL / NDJSON
    3. Excel (.xlsx, .xls)
    4. Parquet (.parquet)
    5. Folder of raw text files (.txt) with subfolder class names (e.g. data/train/hoax/*.txt)
    6. Flat folder of .txt files with label mapping file
    """

    SUPPORTED_EXTENSIONS = [".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet"]

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    f.read(4096)
                return enc
            except UnicodeDecodeError:
                continue
        return "latin-1"

    @classmethod
    def load_any_tabular_file(cls, file_path: str) -> pd.DataFrame:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".csv", ".tsv", ".txt"]:
            enc = cls.detect_encoding(file_path)
            # Try sniffing delimiter
            with open(file_path, "r", encoding=enc, errors="ignore") as f:
                sample_lines = [f.readline() for _ in range(5)]
            sample_text = "".join(sample_lines)

            # Detect separator
            sep = ","
            if "\t" in sample_text and sample_text.count("\t") > sample_text.count(","):
                sep = "\t"
            elif ";" in sample_text and sample_text.count(";") > sample_text.count(","):
                sep = ";"
            elif "|" in sample_text and sample_text.count("|") > sample_text.count(","):
                sep = "|"

            try:
                return pd.read_csv(file_path, sep=sep, encoding=enc, on_bad_lines="skip")
            except Exception:
                return pd.read_csv(file_path, encoding=enc, engine="python", on_bad_lines="skip")

        elif ext in [".json", ".jsonl", ".ndjson"]:
            try:
                return pd.read_json(file_path, lines=True)
            except Exception:
                try:
                    return pd.read_json(file_path)
                except Exception:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    elif isinstance(data, dict):
                        # Find key with list of dicts
                        for k, v in data.items():
                            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                                return pd.DataFrame(v)
                        return pd.DataFrame([data])
                    raise ValueError(f"Could not parse JSON structure from {file_path}")

        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)

        elif ext == ".parquet":
            return pd.read_parquet(file_path)

        raise ValueError(f"Unsupported file format: {ext}")

    @classmethod
    def load_from_txt_directory(cls, dir_path: str) -> pd.DataFrame:
        """
        Parses folder of text files.
        If directory contains subdirectories, subfolder names are inferred as class labels!
        """
        subdirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        rows = []

        if subdirs:
            # Case 1: Folder per class (e.g. train/hoax/*.txt, train/fakta/*.txt)
            for sdir in sorted(subdirs):
                label_name = os.path.basename(sdir)
                txt_files = sorted(list(set(glob.glob(os.path.join(sdir, "*.txt")) + glob.glob(os.path.join(sdir, "*.TXT")))))
                for tf in txt_files:
                    enc = cls.detect_encoding(tf)
                    with open(tf, "r", encoding=enc, errors="ignore") as f:
                        content = f.read().strip()
                    rows.append({
                        "id": os.path.splitext(os.path.basename(tf))[0],
                        "text": content,
                        "label": label_name
                    })
        else:
            # Case 2: Flat directory of .txt files
            txt_files = sorted(list(set(glob.glob(os.path.join(dir_path, "*.txt")) + glob.glob(os.path.join(dir_path, "*.TXT")))))
            
            # Check for mapping file
            mapping = {}
            for map_name in ["labels.csv", "label.csv", "mapping.csv", "train_labels.csv", "targets.csv"]:
                mp = os.path.join(dir_path, map_name)
                if os.path.exists(mp):
                    mdf = pd.read_csv(mp)
                    # If 2 columns, assume col0=id/filename, col1=label
                    if len(mdf.columns) >= 2:
                        id_col, lbl_col = mdf.columns[0], mdf.columns[1]
                        mapping = dict(zip(mdf[id_col].astype(str), mdf[lbl_col]))
                    break

            for tf in txt_files:
                doc_id = os.path.splitext(os.path.basename(tf))[0]
                enc = cls.detect_encoding(tf)
                with open(tf, "r", encoding=enc, errors="ignore") as f:
                    content = f.read().strip()
                row = {
                    "id": doc_id,
                    "text": content
                }
                if doc_id in mapping or os.path.basename(tf) in mapping:
                    row["label"] = mapping.get(doc_id, mapping.get(os.path.basename(tf)))
                rows.append(row)

        if not rows:
            raise FileNotFoundError(f"No .txt documents found in directory: {dir_path}")

        return pd.DataFrame(rows)

    @classmethod
    def normalize_to_csv(
        cls,
        source_path: str,
        output_csv_path: str,
        text_col_override: Optional[str] = None,
        label_col_override: Optional[str] = None,
        id_col_override: Optional[str] = None,
        drop_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Universal normalizer entrypoint. Produces clean CSV and returns metadata.
        """
        df = None
        detected_format = ""

        # 1. Multi-file Glob / Wildcards (e.g. data/batch_*.csv or train_*.jsonl)
        if "*" in source_path or "?" in source_path:
            matched_files = sorted(glob.glob(source_path))
            if not matched_files:
                raise FileNotFoundError(f"No files matched glob pattern: {source_path}")
            dfs = [cls.load_any_tabular_file(f) for f in matched_files]
            df = pd.concat(dfs, ignore_index=True)
            detected_format = f"glob_partitions ({len(matched_files)} files)"

        # 2. Directory (Folder of text files or directory of CSV/JSONL partitions)
        elif os.path.isdir(source_path):
            has_txt = any(
                glob.glob(os.path.join(source_path, "*.txt")) or 
                glob.glob(os.path.join(source_path, "*", "*.txt")) or
                glob.glob(os.path.join(source_path, "*.TXT")) or
                glob.glob(os.path.join(source_path, "*", "*.TXT"))
            )
            if has_txt:
                df = cls.load_from_txt_directory(source_path)
                detected_format = "folder_of_text_files"
            else:
                # Check for partitions: csv, tsv, jsonl, parquet, excel
                part_files = []
                for ext in [".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"]:
                    f_list = sorted(glob.glob(os.path.join(source_path, f"*{ext}")))
                    if f_list:
                        part_files = f_list
                        break
                if part_files:
                    dfs = [cls.load_any_tabular_file(f) for f in part_files]
                    df = pd.concat(dfs, ignore_index=True)
                    detected_format = f"partition_folder ({len(part_files)} files)"
                else:
                    raise FileNotFoundError(f"No recognizable tabular or text files found in folder: {source_path}")

        # 3. Single tabular or text document file
        elif os.path.isfile(source_path):
            df = cls.load_any_tabular_file(source_path)
            detected_format = os.path.splitext(source_path)[1].lower()
        else:
            raise FileNotFoundError(f"Path does not exist: {source_path}")

        # Intelligent column mapping
        columns = list(df.columns)
        text_col = text_col_override
        label_col = label_col_override
        id_col = id_col_override

        # Detect text column if not overridden
        if not text_col:
            # Look for exact text names
            for candidate in ["text", "content", "document", "sentence", "body", "tweet", "sms", "review", "comment", "title"]:
                for c in columns:
                    if c.lower() == candidate:
                        text_col = c
                        break
                if text_col:
                    break

            # If still not found, pick the string column with highest average character length
            if not text_col:
                best_len = -1
                for c in columns:
                    if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
                        lens = df[c].dropna().astype(str).str.len()
                        if len(lens) > 0 and lens.mean() > best_len:
                            best_len = lens.mean()
                            text_col = c

        # Detect label column
        if not label_col:
            for candidate in ["label", "target", "category", "class", "sentiment", "y"]:
                for c in columns:
                    if c.lower() == candidate:
                        label_col = c
                        break
                if label_col:
                    break

        # Detect id column
        if not id_col:
            for candidate in ["id", "index", "doc_id", "uid", "identifier"]:
                for c in columns:
                    if c.lower() == candidate:
                        id_col = c
                        break
                if id_col:
                    break

        # If no id col, generate sequential IDs
        if not id_col:
            df.insert(0, "id", np.arange(len(df)))
            id_col = "id"

        # Ensure text column exists
        if not text_col:
            raise ValueError(f"Could not automatically detect text column in columns: {columns}")

        # Drop entirely empty rows
        df[text_col] = df[text_col].fillna("").astype(str)

        # Deduplication
        orig_len = len(df)
        dropped_dups = 0
        if drop_duplicates:
            if "link" in df.columns:
                df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)
            elif text_col and text_col in df.columns:
                df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
            dropped_dups = orig_len - len(df)

        # Save to normalized destination
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding="utf-8")

        # Class distribution if label exists
        classes = {}
        if label_col and label_col in df.columns:
            vc = df[label_col].value_counts()
            classes = {str(k): int(v) for k, v in vc.items()}

        return {
            "source_path": source_path,
            "detected_format": detected_format,
            "output_csv": output_csv_path,
            "total_rows": len(df),
            "original_rows": orig_len,
            "dropped_duplicates": dropped_dups,
            "columns": list(df.columns),
            "detected_text_col": text_col,
            "detected_label_col": label_col,
            "detected_id_col": id_col,
            "class_distribution": classes,
            "sample_rows": df[[id_col, text_col] + ([label_col] if label_col in df.columns else [])].head(3).to_dict(orient="records")
        }


class UniversalTextCleaner:
    """
    Configurable, domain-agnostic text cleaner for any NLP dataset
    (News outlets like CNN/Detik/Kompas, Social Media/Tweets, Product Reviews, Legal, Academic, etc.).
    """
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    MENTION_PATTERN = re.compile(r"@[\w_]+")
    MULTISPACE_PATTERN = re.compile(r"\s+")
    REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
    GENERIC_DATELINE_PATTERN = re.compile(r"^[A-Za-z\s]{2,25}(?:,\s*[A-Za-z0-9\.\s\-]+)?\s*[-–—]{1,2}\s*")

    @classmethod
    def clean(
        cls,
        text: str,
        lower: bool = True,
        strip_urls: bool = True,
        strip_mentions: bool = False,
        strip_html: bool = True,
        reduce_repeated_chars: bool = True,
        strip_dateline: bool = False,
        custom_remove_phrases: Optional[List[str]] = None,
        custom_regex_list: Optional[List[str]] = None,
    ) -> str:
        if not isinstance(text, str):
            return ""

        if strip_html:
            text = cls.HTML_PATTERN.sub(" ", text)

        if strip_urls:
            text = cls.URL_PATTERN.sub(" ", text)

        if strip_mentions:
            text = cls.MENTION_PATTERN.sub(" ", text)

        if strip_dateline:
            text = cls.GENERIC_DATELINE_PATTERN.sub("", text)

        if custom_remove_phrases:
            for phrase in custom_remove_phrases:
                p = phrase.strip()
                if not p:
                    continue
                if p.endswith("...") or p.endswith("*"):
                    clean_prefix = re.escape(p.rstrip(".*").strip())
                    pattern = rf"{clean_prefix}.*?(?:\.|\n|$)"
                    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
                else:
                    text = re.sub(re.escape(p), " ", text, flags=re.IGNORECASE)

        if custom_regex_list:
            for pattern in custom_regex_list:
                if pattern and pattern.strip():
                    try:
                        text = re.sub(pattern.strip(), " ", text, flags=re.IGNORECASE)
                    except Exception:
                        pass

        if lower:
            text = text.lower()

        if reduce_repeated_chars:
            text = cls.REPEAT_CHAR_PATTERN.sub(r"\1\1", text)

        text = cls.MULTISPACE_PATTERN.sub(" ", text).strip()
        return text

    @classmethod
    def clean_news(cls, text: str, lower: bool = True) -> str:
        """Convenience method for news datasets with generic dateline stripping."""
        return cls.clean(text, lower=lower, strip_urls=True, strip_html=True, strip_dateline=True, reduce_repeated_chars=True)


# Backwards compatibility alias
IndonesianTextCleaner = UniversalTextCleaner


class TextFeatureExtractor:
    """
    Extracts high-signal tabular features from text columns:
    1. Stylometric & Text Morphological Statistics (length, lexical diversity, uppercase, digits, puncts)
    2. TF-IDF + TruncatedSVD (LSA components)
    3. Offline Transformer Embeddings (when torch & transformers are present)
    """

    @staticmethod
    def extract_stylometrics(df: pd.DataFrame, text_col: str, prefix: str = "txt_") -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        s = df[text_col].fillna("").astype(str)

        out[f"{prefix}char_count"] = s.str.len()
        out[f"{prefix}word_count"] = s.apply(lambda x: len(x.split()))
        out[f"{prefix}avg_word_len"] = out[f"{prefix}char_count"] / (out[f"{prefix}word_count"].replace(0, 1))

        # Lexical diversity: unique words / total words
        def calc_diversity(x):
            words = x.lower().split()
            return len(set(words)) / max(len(words), 1)

        out[f"{prefix}lexical_diversity"] = s.apply(calc_diversity)
        out[f"{prefix}upper_count"] = s.apply(lambda x: sum(1 for c in x if c.isupper()))
        out[f"{prefix}upper_ratio"] = out[f"{prefix}upper_count"] / (out[f"{prefix}char_count"].replace(0, 1))
        out[f"{prefix}digit_count"] = s.apply(lambda x: sum(1 for c in x if c.isdigit()))
        out[f"{prefix}digit_ratio"] = out[f"{prefix}digit_count"] / (out[f"{prefix}char_count"].replace(0, 1))
        out[f"{prefix}punct_count"] = s.apply(lambda x: sum(1 for c in x if c in string.punctuation))
        out[f"{prefix}exclamation_count"] = s.apply(lambda x: x.count("!"))
        out[f"{prefix}question_count"] = s.apply(lambda x: x.count("?"))

        return out

    @staticmethod
    def extract_tfidf_lsa(
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        text_col: str,
        n_components: int = 50,
        ngram_range: Tuple[int, int] = (1, 2),
        prefix: str = "lsa_"
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Vectorizes text with TF-IDF and compresses with TruncatedSVD (Latent Semantic Analysis).
        Guarantees fast GBDT training without exploding column dimensionality.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        clean_train = train_df[text_col].fillna("").astype(str)
        clean_test = test_df[text_col].fillna("").astype(str) if test_df is not None else None

        # Dual word + char n-gram vectorizer
        tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=25000,
            sublinear_tf=True,
            min_df=2
        )
        X_train_tfidf = tfidf.fit_transform(clean_train)

        max_possible = min(X_train_tfidf.shape[1], len(train_df))
        n_comp = min(n_components, max_possible)
        if n_comp < 1:
            n_comp = 1

        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        train_lsa = svd.fit_transform(X_train_tfidf)

        actual_comp = train_lsa.shape[1]
        col_names = [f"{prefix}{i:02d}" for i in range(actual_comp)]
        train_lsa_df = pd.DataFrame(train_lsa, columns=col_names, index=train_df.index)

        test_lsa_df = None
        if test_df is not None:
            X_test_tfidf = tfidf.transform(clean_test)
            test_lsa = svd.transform(X_test_tfidf)
            test_lsa_df = pd.DataFrame(test_lsa, columns=col_names, index=test_df.index)

        return train_lsa_df, test_lsa_df

    @staticmethod
    def extract_transformer_embeddings(
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        text_col: str,
        model_dir_or_name: str,
        batch_size: int = 32,
        max_length: int = 128,
        device: str = "cuda",
        prefix: str = "emb_"
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Offline Transformer Embeddings Extractor (e.g. IndoBERT / DeBERTa).
        Gracefully executes on GPU (RTX 3070) or CPU with zero internet requirement.
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("PyTorch and Transformers are required for Transformer embeddings. Install them or use extract_tfidf_lsa fallback.")

        dev = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name, local_files_only=os.path.exists(model_dir_or_name))
        model = AutoModel.from_pretrained(model_dir_or_name, local_files_only=os.path.exists(model_dir_or_name)).to(dev)
        model.eval()

        def embed_series(series: pd.Series) -> np.ndarray:
            texts = series.fillna("").astype(str).tolist()
            all_embs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    ).to(dev)
                    
                    outputs = model(**encoded)
                    # Mean pooling over attention mask
                    token_embs = outputs.last_hidden_state
                    attn_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embs.size()).float()
                    sum_embs = torch.sum(token_embs * attn_mask, 1)
                    sum_mask = torch.clamp(attn_mask.sum(1), min=1e-9)
                    pooled = (sum_embs / sum_mask).cpu().numpy()
                    all_embs.append(pooled)
            return np.vstack(all_embs)

        train_vecs = embed_series(train_df[text_col])
        n_dim = train_vecs.shape[1]
        col_names = [f"{prefix}{i:03d}" for i in range(n_dim)]
        train_emb_df = pd.DataFrame(train_vecs, columns=col_names, index=train_df.index)

        test_emb_df = None
        if test_df is not None:
            test_vecs = embed_series(test_df[text_col])
            test_emb_df = pd.DataFrame(test_vecs, columns=col_names, index=test_df.index)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return train_emb_df, test_emb_df


class TextEDAReporter:
    """
    Computes comprehensive EDA diagnostics for text datasets.
    """
    @staticmethod
    def generate_report(df: pd.DataFrame, text_col: str, label_col: Optional[str] = None) -> Dict[str, Any]:
        s = df[text_col].fillna("").astype(str)
        char_lens = s.str.len().values
        word_lens = s.apply(lambda x: len(x.split())).values

        # Token vocabulary
        from collections import Counter
        words = []
        bigrams = []
        for text in s.head(5000): # Sample for fast generation
            toks = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
            words.extend(toks)
            if len(toks) >= 2:
                bigrams.extend([f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)])

        vocab_counter = Counter(words)
        bigram_counter = Counter(bigrams)
        total_tokens = len(words)
        vocab_size = len(vocab_counter)

        # Percentiles
        def get_stats(arr):
            if len(arr) == 0:
                return {}
            return {
                "min": int(np.min(arr)),
                "p25": int(np.percentile(arr, 25)),
                "median": int(np.median(arr)),
                "p75": int(np.percentile(arr, 75)),
                "p95": int(np.percentile(arr, 95)),
                "max": int(np.max(arr)),
                "mean": round(float(np.mean(arr)), 1)
            }

        # Quality metrics
        empty_count = int(np.sum(s.str.strip() == ""))
        duplicates_count = int(s.duplicated().sum())
        html_detected = int(s.str.contains(r"<.*?>", regex=True).sum())

        top_words = [{"word": w, "count": c} for w, c in vocab_counter.most_common(20)]
        top_bigrams = [{"bigram": b, "count": c} for b, c in bigram_counter.most_common(15)]

        class_dist = {}
        if label_col and label_col in df.columns:
            vc = df[label_col].value_counts()
            class_dist = {str(k): int(v) for k, v in vc.items()}

        # 1. Length distribution histogram (10 bins)
        hist, bin_edges = np.histogram(word_lens, bins=10)
        length_histogram = []
        for i in range(len(hist)):
            length_histogram.append({
                "bin_start": int(bin_edges[i]),
                "bin_end": int(bin_edges[i+1]),
                "count": int(hist[i]),
                "pct": round(float(hist[i] / max(len(word_lens), 1) * 100), 1)
            })

        # 2. Per-class length profile
        class_length_profile = []
        if label_col and label_col in df.columns:
            try:
                df_tmp = pd.DataFrame({"_wlen": word_lens, "_cat": df[label_col].astype(str)})
                grp = df_tmp.groupby("_cat")["_wlen"].agg(["mean", "median", "count"]).reset_index()
                for _, r in grp.iterrows():
                    class_length_profile.append({
                        "class": str(r["_cat"]),
                        "mean_words": round(float(r["mean"]), 1),
                        "median_words": int(r["median"]),
                        "count": int(r["count"])
                    })
            except Exception:
                pass

        # 3. Dynamic Actionable Insights
        word_stats = get_stats(word_lens)
        p95_w = word_stats.get("p95", 128)
        rec_seq = 128 if p95_w <= 100 else (256 if p95_w <= 200 else 512)
        insights = [
            f"95% dokumen berada di bawah {p95_w} kata. Menggunakan max_length={rec_seq} pada transformer akan mempertahankan >95% informasi tanpa memboroskan VRAM atau komputasi."
        ]

        if class_dist:
            counts = list(class_dist.values())
            if counts and min(counts) > 0:
                ratio = max(counts) / min(counts)
                if ratio > 3.0:
                    insights.append(f"Rasio ketimpangan kelas mencapai {ratio:.1f}:1. Disarankan mengaktifkan Stratified K-Fold dan class_weight='balanced' untuk mencegah bias model ke kelas mayoritas.")
                else:
                    insights.append(f"Distribusi kelas relatif seimbang (rasio {ratio:.1f}:1). Standard Stratified K-Fold akan memberikan estimasi skor CV yang andal.")

        if duplicates_count > 0 or html_detected > 0:
            insights.append(f"Ditemukan {duplicates_count} duplikasi teks ({round(duplicates_count / max(len(df), 1) * 100, 1)}%) dan {html_detected} tag HTML. Jalankan pembersihan di Tab Preprocessing untuk meningkatkan kualitas korpus.")

        if len(class_length_profile) >= 2:
            sorted_by_len = sorted(class_length_profile, key=lambda x: x["mean_words"])
            shortest = sorted_by_len[0]
            longest = sorted_by_len[-1]
            if longest["mean_words"] > 1.35 * shortest["mean_words"] and shortest["mean_words"] > 0:
                insights.append(f"Terdapat perbedaan panjang teks antar kelas: kelas '{longest['class']}' rata-rata {longest['mean_words']} kata vs '{shortest['class']}' {shortest['mean_words']} kata. Fitur stilometrik (panjang kata/karakter) akan menjadi sinyal prediktif yang kuat bagi model GBDT.")

        return {
            "total_documents": len(df),
            "text_col": text_col,
            "label_col": label_col,
            "char_length_stats": get_stats(char_lens),
            "word_length_stats": word_stats,
            "vocabulary_size": vocab_size,
            "total_tokens": total_tokens,
            "lexical_richness": round(vocab_size / max(total_tokens, 1), 4),
            "quality_metrics": {
                "empty_documents": empty_count,
                "empty_percentage": round((empty_count / max(len(df), 1)) * 100, 2),
                "duplicate_documents": duplicates_count,
                "duplicate_percentage": round((duplicates_count / max(len(df), 1)) * 100, 2),
                "html_tags_detected": html_detected
            },
            "top_words": top_words,
            "top_bigrams": top_bigrams,
            "class_distribution": class_dist,
            "length_histogram": length_histogram,
            "class_length_profile": class_length_profile,
            "actionable_insights": insights
        }


class NLPBaselineTrainer:
    """
    Production-grade baseline trainer for NLP text classification competitions.
    - TF-IDF Vectorizer with customizable n-grams and sublinear TF scaling
    - Robust Stratified K-Fold CV (automatically shields rare classes with < k samples)
    - Regularized Logistic Regression with balanced class weights
    - Fold-by-fold accuracy and macro-F1 metrics
    - Informative keywords extraction per class
    - Test set inference and submission CSV generation
    """

    @classmethod
    def train_baseline(
        cls,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        text_col: str = "text",
        text_col_2: Optional[str] = None,
        label_col: str = "label",
        id_col: str = "id",
        task_type: str = "classification",
        max_features: int = 15000,
        ngram_min: int = 1,
        ngram_max: int = 2,
        n_splits: int = 5,
        c_param: float = 1.5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import StratifiedKFold, KFold
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

        if text_col not in train_df.columns:
            raise ValueError(f"Text column '{text_col}' not found in training dataset.")
        if label_col not in train_df.columns:
            raise ValueError(f"Label column '{label_col}' not found in training dataset.")

        subset_cols = [text_col, text_col_2, label_col] if text_col_2 and text_col_2 in train_df.columns else [text_col, label_col]
        clean_train = train_df.dropna(subset=subset_cols).copy()
        if text_col_2 and text_col_2 in clean_train.columns:
            clean_train["_combined_text"] = clean_train[text_col].astype(str) + " " + clean_train[text_col_2].astype(str)
            active_text_col = "_combined_text"
        else:
            clean_train[text_col] = clean_train[text_col].astype(str)
            active_text_col = text_col

        # Store original index to ensure exact row alignment with other models
        clean_train["_orig_idx"] = clean_train.index.values

        # All classes in dataset sorted alphabetically (includes rare classes)
        all_classes = sorted(clean_train[label_col].astype(str).unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(all_classes)}

        # Identify rare classes
        class_counts = clean_train[label_col].value_counts()
        stratifiable_classes = class_counts[class_counts >= n_splits].index
        
        df_cv = clean_train[clean_train[label_col].isin(stratifiable_classes)].copy()
        rare_df = clean_train[~clean_train[label_col].isin(stratifiable_classes)].copy()

        # TF-IDF Vectorizer
        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max),
            sublinear_tf=True,
            strip_accents="unicode"
        )
        X_all = vec.fit_transform(clean_train[active_text_col])
        feature_names = np.array(vec.get_feature_names_out())

        # Fit full model for test predictions & top words
        full_clf = LogisticRegression(C=c_param, max_iter=300, class_weight="balanced", random_state=random_state)
        full_clf.fit(X_all, clean_train[label_col])

        # Top words per class
        top_words_per_class = {}
        if hasattr(full_clf, "coef_"):
            classes = full_clf.classes_
            for idx, cname in enumerate(classes):
                coef = full_clf.coef_[idx] if len(classes) > 2 else (full_clf.coef_[0] if idx == 1 else -full_clf.coef_[0])
                top_indices = np.argsort(coef)[-10:][::-1]
                top_words_per_class[str(cname)] = [feature_names[i] for i in top_indices]

        # K-Fold CV on stratifiable slice
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_results = []
        oof_preds = []
        oof_true = []

        X_cv = vec.transform(df_cv[text_col])
        y_cv = df_cv[label_col].values
        
        # Initialize oof_probs for the ENTIRE clean_train dataset with all classes
        clean_indices = clean_train.index.to_numpy()
        idx_to_clean_pos = {idx: pos for pos, idx in enumerate(clean_indices)}
        oof_probs = np.zeros((len(clean_train), len(all_classes)), dtype=np.float32)

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
            X_tr, y_tr = X_cv[trn_idx], y_cv[trn_idx]
            X_va, y_va = X_cv[val_idx], y_cv[val_idx]

            # If rare samples exist, include them in training fold
            if len(rare_df) > 0:
                X_rare = vec.transform(rare_df[text_col])
                import scipy.sparse as sp
                X_tr = sp.vstack([X_tr, X_rare])
                y_tr = np.concatenate([y_tr, rare_df[label_col].values])

            clf = LogisticRegression(C=c_param, max_iter=300, class_weight="balanced", random_state=random_state + fold)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_va)
                val_orig_ids = df_cv.iloc[val_idx]["_orig_idx"].values
                pos_in_clean = [idx_to_clean_pos[oi] for oi in val_orig_ids]
                # Map clf classes to all_classes order
                for ci, c in enumerate(clf.classes_):
                    if c in class_to_idx:
                        oof_probs[pos_in_clean, class_to_idx[c]] = probs[:, ci]

            acc = float(accuracy_score(y_va, preds))
            macro_f1 = float(f1_score(y_va, preds, average="macro"))
            fold_results.append({
                "fold": fold + 1,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "val_samples": len(val_idx)
            })
            oof_preds.extend(preds)
            oof_true.extend(y_va)

        # Predict rare samples using full_clf so all clean_train rows have probabilities
        if len(rare_df) > 0 and hasattr(full_clf, "predict_proba"):
            X_rare = vec.transform(rare_df[text_col])
            rare_probs = full_clf.predict_proba(X_rare)
            rare_orig_ids = rare_df["_orig_idx"].values
            pos_rare = [idx_to_clean_pos[oi] for oi in rare_orig_ids]
            for ci, c in enumerate(full_clf.classes_):
                if c in class_to_idx:
                    oof_probs[pos_rare, class_to_idx[c]] = rare_probs[:, ci]

        overall_acc = float(accuracy_score(oof_true, oof_preds)) if oof_true else 0.0
        overall_f1 = float(f1_score(oof_true, oof_preds, average="macro")) if oof_true else 0.0

        # Store oof probabilities with true original indices
        oof_indices = clean_indices
        test_probs_matrix = None

        # Predict test if provided
        test_predictions_sample = []
        test_pred_df = None
        if test_df is not None and len(test_df) > 0:
            test_text = test_df[text_col].fillna("").astype(str) if text_col in test_df.columns else test_df.iloc[:, 1].fillna("").astype(str)
            X_test = vec.transform(test_text)
            preds = full_clf.predict(X_test)
            if hasattr(full_clf, "predict_proba"):
                test_probs_raw = full_clf.predict_proba(X_test)
                probs = np.max(test_probs_raw, axis=1)
                test_probs_matrix = np.zeros((len(test_df), len(all_classes)), dtype=np.float32)
                for ci, c in enumerate(full_clf.classes_):
                    if c in class_to_idx:
                        test_probs_matrix[:, class_to_idx[c]] = test_probs_raw[:, ci]
            else:
                probs = [1.0] * len(preds)
            
            t_ids = test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))
            test_pred_df = pd.DataFrame({
                "id": t_ids,
                "predicted_label": preds,
                "confidence": np.round(probs, 4)
            })
            test_predictions_sample = test_pred_df.head(10).to_dict(orient="records")

        return {
            "model_type": "TF-IDF + Balanced Logistic Regression",
            "features_extracted": int(X_all.shape[1]),
            "n_splits": n_splits,
            "overall_accuracy": round(overall_acc, 4),
            "overall_macro_f1": round(overall_f1, 4),
            "fold_results": fold_results,
            "top_words_per_class": top_words_per_class,
            "test_predictions_sample": test_predictions_sample,
            "total_train_samples": len(clean_train),
            "rare_classes_shielded": len(rare_df),
            "test_pred_df": test_pred_df,
            "oof_probs_matrix": oof_probs,
            "oof_indices": oof_indices,
            "test_probs_matrix": test_probs_matrix,
            "class_names": all_classes
        }


    @classmethod
    def tune_baseline(
        cls,
        train_df: pd.DataFrame,
        text_col: str,
        label_col: str,
        test_df: Optional[pd.DataFrame] = None,
        id_col: Optional[str] = "id",
        n_splits: int = 3,
        random_state: int = 2026
    ) -> Dict[str, Any]:
        """
        Grid ablation search over TF-IDF and regularizer hyperparameters.
        Evaluates CV F1-macro, automatically selects best configuration,
        and returns Top 5 comparison trials for ablation analysis.
        """
        import time

        param_grid = [
            {"ngram_range": (1, 2), "c_param": 1.0, "max_features": 25000},
            {"ngram_range": (1, 2), "c_param": 2.5, "max_features": 35000},
            {"ngram_range": (1, 3), "c_param": 1.5, "max_features": 30000},
            {"ngram_range": (1, 1), "c_param": 1.0, "max_features": 15000},
            {"ngram_range": (1, 2), "c_param": 0.5, "max_features": 20000},
            {"ngram_range": (1, 2), "c_param": 5.0, "max_features": 40000},
            {"ngram_range": (1, 3), "c_param": 2.0, "max_features": 50000},
            {"ngram_range": (1, 1), "c_param": 2.0, "max_features": 25000},
        ]

        results = []
        for p in param_grid:
            t0 = time.time()
            try:
                res = cls.train_baseline(
                    train_df=train_df,
                    text_col=text_col,
                    label_col=label_col,
                    test_df=None,
                    id_col=id_col,
                    n_splits=n_splits,
                    max_features=p["max_features"],
                    ngram_range=p["ngram_range"],
                    c_param=p["c_param"],
                    random_state=random_state
                )
                dt = round(time.time() - t0, 2)
                f1 = res.get("overall_macro_f1", res.get("overall_accuracy", 0))
                acc = res.get("overall_accuracy", 0)
                results.append({
                    "score": f1,
                    "accuracy": acc,
                    "params": p,
                    "duration_seconds": dt,
                    "features": res.get("features_extracted", 0),
                    "full_res": res
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        top_5 = []
        for rank, r in enumerate(results[:5], 1):
            top_5.append({
                "rank": rank,
                "score": round(r["score"], 4),
                "accuracy": round(r["accuracy"], 4),
                "ngram_range": f"{r['params']['ngram_range'][0]}-{r['params']['ngram_range'][1]}",
                "c_param": r["params"]["c_param"],
                "max_features": r["params"]["max_features"],
                "duration_seconds": r["duration_seconds"]
            })

        best = results[0] if results else None
        best_full = None
        if best:
            best_full = cls.train_baseline(
                train_df=train_df,
                text_col=text_col,
                label_col=label_col,
                test_df=test_df,
                id_col=id_col,
                n_splits=n_splits,
                max_features=best["params"]["max_features"],
                ngram_range=best["params"]["ngram_range"],
                c_param=best["params"]["c_param"],
                random_state=random_state
            )
            best_full["top_5_ablation"] = top_5
            best_full["best_tuned_params"] = best["params"]

        return best_full or {}
