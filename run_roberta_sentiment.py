"""
Automated Sentiment & Public Opinion Mining for 'Aksi 28 Oktober'
Using Pretrained Indonesian RoBERTa Sentiment Classifier: w11wo/indonesian-roberta-base-sentiment-classifier
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def main():
    log("================ STARTING RO-BERTA SENTIMENT ANALYSIS ================")
    case_dir = os.path.join("cases", "aksi_28_okt")
    if not os.path.exists(case_dir):
        case_dir = os.path.join("cases", "aksi-28-okt")
    input_csv = os.path.join(case_dir, "train.csv")
    
    if not os.path.exists(input_csv):
        log(f"Error: {input_csv} not found.")
        sys.exit(1)
        
    df = pd.read_csv(input_csv)
    log(f"Loaded {len(df)} comments from {input_csv}")
    
    text_col = "clean_text" if "clean_text" in df.columns else "Komentar"
    texts = df[text_col].fillna("").astype(str).tolist()
    
    # 1. Check Torch & Transformers
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        log("Error: PyTorch and Transformers are required to run RoBERTa.")
        log("If running on PC Dosen, ensure your GPU virtual environment is active.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Computation Device: {device} " + (f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    
    # 2. Resolve Model
    model_id = "w11wo/indonesian-roberta-base-sentiment-classifier"
    local_candidates = [
        os.path.join("models", "indonesian-roberta-base-sentiment"),
        os.path.join("models", "indonesian-roberta-base-sentiment-classifier")
    ]
    model_name = model_id
    for cand in local_candidates:
        if os.path.exists(cand) and os.path.exists(os.path.join(cand, "config.json")):
            model_name = cand
            log(f"Loading from offline local directory: {cand}")
            break
    if model_name == model_id:
        log(f"Loading from Hugging Face Hub: {model_id}")
        
    log("Loading Tokenizer and Model weights...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    id2label = model.config.id2label
    log(f"Model Labels configuration: {id2label}")
    
    # 3. Batch Inference
    batch_size = 32 if device.type == "cuda" else 8
    all_preds = []
    all_confs = []
    all_probs = []
    
    log(f"Running inference across {len(texts)} comments (Batch Size: {batch_size})...")
    t0 = time.time()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            
            for p, c, pr in zip(preds, confs, probs):
                raw_label = id2label.get(p, str(p))
                # Standardize label name in Indonesian
                if "pos" in raw_label.lower() or raw_label == "LABEL_0" and "pos" in str(id2label).lower():
                    clean_lbl = "Positif"
                elif "neg" in raw_label.lower():
                    clean_lbl = "Negatif"
                elif "neu" in raw_label.lower():
                    clean_lbl = "Netral"
                else:
                    clean_lbl = raw_label.capitalize()
                    
                all_preds.append(clean_lbl)
                all_confs.append(round(float(c), 4))
                all_probs.append([round(float(x), 4) for x in pr])
                
            if (i // batch_size) % 20 == 0 or (i + batch_size) >= len(texts):
                pct = min(100.0, round((i + len(batch_texts)) / len(texts) * 100, 1))
                log(f"Progress: {i + len(batch_texts)}/{len(texts)} ({pct}%)")
                
    elapsed = round(time.time() - t0, 2)
    log(f"Inference completed in {elapsed}s ({round(len(texts)/elapsed, 1)} comments/sec)")
    
    # 4. Attach Predictions
    df["sentiment"] = all_preds
    df["confidence"] = all_confs
    
    # Map probability columns
    num_classes = len(id2label)
    for idx in range(num_classes):
        lbl_name = id2label.get(idx, f"class_{idx}")
        col_name = f"prob_{lbl_name.lower()}"
        df[col_name] = [p[idx] for p in all_probs]
        
    # 5. Export Results
    out_xlsx = os.path.join(case_dir, "dataset_labeled_roberta.xlsx")
    out_csv = os.path.join(case_dir, "dataset_labeled_roberta.csv")
    
    try:
        df.to_excel(out_xlsx, index=False)
        log(f"Saved labeled Excel: {out_xlsx}")
    except Exception as e:
        log(f"Notice: Could not write Excel directly ({e}), skipping xlsx.")
        
    df.to_csv(out_csv, index=False, encoding="utf-8")
    df.to_csv(input_csv, index=False, encoding="utf-8") # Update train.csv with sentiment
    log(f"Saved labeled CSV: {out_csv} and updated {input_csv}")
    
    # 6. Analytics & Distribution
    dist = df["sentiment"].value_counts().to_dict()
    total = len(df)
    dist_pct = {k: round((v / total) * 100, 2) for k, v in dist.items()}
    
    log("================ SUMMARY ANALYTICS ================")
    log(f"Total Comments Analyzed: {total}")
    for k, v in dist.items():
        log(f" - {k}: {v} comments ({dist_pct[k]}%)")
        
    # Extract top keywords per sentiment
    from sklearn.feature_extraction.text import TfidfVectorizer
    top_words_per_sentiment = {}
    for sent in df["sentiment"].unique():
        sub = df[df["sentiment"] == sent]
        if len(sub) > 5:
            try:
                vec = TfidfVectorizer(max_features=10, stop_words=None, ngram_range=(1, 2))
                vec.fit(sub[text_col])
                top_words_per_sentiment[sent] = list(vec.get_feature_names_out())
            except Exception:
                pass
                
    insights = {
        "case_id": "aksi_28_okt",
        "total_comments": total,
        "elapsed_seconds": elapsed,
        "distribution_counts": dist,
        "distribution_percentages": dist_pct,
        "top_keywords": top_words_per_sentiment,
        "sample_positive": df[df["sentiment"] == "Positif"].nlargest(3, "confidence")[[text_col, "confidence"]].to_dict(orient="records") if "Positif" in dist else [],
        "sample_negative": df[df["sentiment"] == "Negatif"].nlargest(3, "confidence")[[text_col, "confidence"]].to_dict(orient="records") if "Negatif" in dist else []
    }
    
    insights_path = os.path.join(case_dir, "sentiment_insights.json")
    with open(insights_path, "w", encoding="utf-8") as inf_f:
        json.dump(insights, inf_f, indent=2)
    log(f"Saved public opinion insights: {insights_path}")
    log("================ PROCESS COMPLETED SUCCESSFULLY ================")

if __name__ == "__main__":
    main()
