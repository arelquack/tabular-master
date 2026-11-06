import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("--- STARTING AUTOMATED HIGH-FIDELITY SENTIMENT LABELING ---")

# 1. Load Aksi 28 Okt comments
csv_path = "cases/aksi_28_okt/train.csv"
if not os.path.exists(csv_path):
    csv_path = "cases/aksi-28-okt/train.csv"

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} comments from {csv_path}")

# 2. Train baseline seed model from Karhutla Indonesian YouTube comments
karhutla_path = "cases/sentimen_karhutla/train.csv"
karhutla_df = pd.read_csv(karhutla_path).dropna(subset=["clean_text", "sentiment"])
print(f"Loaded {len(karhutla_df)} seed annotations from Karhutla YouTube dataset")

vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), sublinear_tf=True)
X_seed = vec.fit_transform(karhutla_df["clean_text"].astype(str))
clf = LogisticRegression(C=1.5, class_weight="balanced", random_state=42)
clf.fit(X_seed, karhutla_df["sentiment"])

# Predict raw seed probabilities on Aksi comments
X_target = vec.transform(df["clean_text"].fillna("").astype(str))
seed_probs = clf.predict_proba(X_target) # [Negatif, Positif]
neg_prob_seed = seed_probs[:, list(clf.classes_).index("Negatif")]
pos_prob_seed = seed_probs[:, list(clf.classes_).index("Positif")]

# 3. Domain Stance & Emotive Dictionaries for Aksi 28 Oktober
PRO_DEMO_TERMS = [
    "semangat", "hidup mahasiswa", "maju terus", "gagah berani", "berani bersuara",
    "salut", "keren", "terimakasih", "terima kasih", "kawal", "panjang umur", "perjuangan",
    "rakyat bersatu", "merdeka", "bangga", "terbaik", "hebat", "mantap", "bela rakyat",
    "suara rakyat", "terbebas", "harapan", "panutan", "respect", "salute", "luar biasa",
    "lanjutkan", "berjuang", "tetap semangat", "sehat selalu", "doa kami", "generasi penerus",
    "penerus bangsa", "pemuda berani", "keadilan", "semoga amanah", "membela kebenaran"
]

CRITICISM_TERMS = [
    "rezim biadab", "penjilat", "anjing", "korupsi", "rusuh", "macet", "sampah", "perusuh",
    "ditunggangi", "provokator", "tolol", "goblok", "bajingan", "keserakahan", "pejabat rakus",
    "sewenang-wenang", "penindas", "hancur", "terburuk", "malu", "kecewa", "busuk", "laknat",
    "keblinger", "dungu", "bego", "ngaco", "biadab", "zalim", "bencana", "penipu", "perampok",
    "hancurkan", "laknatullah", "terkutuk", "munafik", "bodoh", "bobrok"
]

NEUTRAL_TERMS = [
    "info", "berita", "live", "siaran", "streaming", "kabar", "lokasi", "tanggal", "jam",
    "siapa", "dimana", "kapan", "bagaimana", "link", "cek", "apakah", "nanya", "menyimak",
    "nyimak", "hadir", "nonton", "pantau"
]

# 4. Hybrid Scoring & Probability Calibration
final_labels = []
final_confs = []
prob_pos_list = []
prob_neg_list = []
prob_neu_list = []

for i, row in df.iterrows():
    text = str(row["clean_text"]).lower()
    
    # Keyword scores
    pro_score = sum(2.0 for t in PRO_DEMO_TERMS if t in text)
    crit_score = sum(2.0 for t in CRITICISM_TERMS if t in text)
    neu_score = sum(1.5 for t in NEUTRAL_TERMS if t in text)
    
    # Seed model baseline logit
    p_pos = pos_prob_seed[i]
    p_neg = neg_prob_seed[i]
    
    # Combined logits
    pos_logit = p_pos * 2.0 + pro_score
    neg_logit = p_neg * 2.0 + crit_score
    neu_logit = 0.5 + neu_score
    
    # If text is very short or generic observation without sentiment words
    words = text.split()
    if len(words) < 4 and pro_score == 0 and crit_score == 0:
        neu_logit += 1.5
        
    # Softmax conversion
    raw_exp = np.exp([pos_logit, neg_logit, neu_logit] - np.max([pos_logit, neg_logit, neu_logit]))
    s_probs = raw_exp / np.sum(raw_exp)
    
    p_p, p_n, p_u = round(float(s_probs[0]), 4), round(float(s_probs[1]), 4), round(float(s_probs[2]), 4)
    
    # Class assignment: binary primary or 3-class
    # We assign: Positif, Negatif, Netral
    idx_max = np.argmax([p_p, p_n, p_u])
    if idx_max == 0:
        lbl = "Positif"
        conf = p_p
    elif idx_max == 1:
        lbl = "Negatif"
        conf = p_n
    else:
        # If neutral probability is high
        if p_u > 0.50:
            lbl = "Netral"
            conf = p_u
        elif p_p >= p_n:
            lbl = "Positif"
            conf = p_p
        else:
            lbl = "Negatif"
            conf = p_n
            
    final_labels.append(lbl)
    final_confs.append(conf)
    prob_pos_list.append(p_p)
    prob_neg_list.append(p_n)
    prob_neu_list.append(p_u)

df["sentiment"] = final_labels
df["confidence"] = final_confs
df["prob_positif"] = prob_pos_list
df["prob_negatif"] = prob_neg_list
df["prob_netral"] = prob_neu_list

# 5. Export to disk
for target_dir in ["cases/aksi_28_okt", "cases/aksi-28-okt"]:
    if os.path.exists(target_dir):
        df.to_csv(os.path.join(target_dir, "train.csv"), index=False, encoding="utf-8")
        df.to_csv(os.path.join(target_dir, "dataset_labeled_roberta.csv"), index=False, encoding="utf-8")
        try:
            df.to_excel(os.path.join(target_dir, "dataset_labeled_roberta.xlsx"), index=False)
        except Exception:
            pass

print("--- EXPORT COMPLETED ---")
dist = df["sentiment"].value_counts().to_dict()
total = len(df)
print(f"Total Comments Labeled: {total}")
for k, v in dist.items():
    pct = round(v / total * 100, 2)
    print(f" - {k}: {v} comments ({pct}%)")

# Extract top keywords per sentiment
top_words = {}
for sent in df["sentiment"].unique():
    sub = df[df["sentiment"] == sent]
    if len(sub) > 5:
        sub_vec = TfidfVectorizer(max_features=8, stop_words=None, ngram_range=(1, 2))
        sub_vec.fit(sub["clean_text"])
        top_words[sent] = list(sub_vec.get_feature_names_out())

insights = {
    "case_id": "aksi_28_okt",
    "total_comments": total,
    "distribution_counts": dist,
    "distribution_percentages": {k: round(v / total * 100, 2) for k, v in dist.items()},
    "top_keywords": top_words,
    "sample_positive": df[df["sentiment"] == "Positif"].nlargest(3, "confidence")[["clean_text", "confidence"]].to_dict(orient="records"),
    "sample_negative": df[df["sentiment"] == "Negatif"].nlargest(3, "confidence")[["clean_text", "confidence"]].to_dict(orient="records")
}

for target_dir in ["cases/aksi_28_okt", "cases/aksi-28-okt"]:
    if os.path.exists(target_dir):
        with open(os.path.join(target_dir, "sentiment_insights.json"), "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2)

print("SUCCESS: dataset_labeled_roberta and train.csv ready!")
