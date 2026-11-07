# Snippets Kode Serbaguna Kompetisi Machine Learning

Folder ini berisi kumpulan 7 snippet kode Python mandiri (*standalone*), fleksibel, dan teruji yang pasti berjalan untuk setiap variasi kasus kompetisi data science (Tabular, NLP Supervised, dan Teks Unsupervised).

Setiap skrip dapat dijalankan langsung melalui terminal CLI atau diimpor ke dalam skrip Python / Jupyter Notebook Anda.

---

## Indeks Snippet

| No | Berkas Snippet | Fokus Kasus | Output yang Dihasilkan |
| :- | :--- | :--- | :--- |
| **01** | [s01_universal_tabular_pipeline.py](file:///d:/POLBAN/Semester%207/gemastik-final/snippets/s01_universal_tabular_pipeline.py) | Semua Kasus Tabular (Biner, Multikelas, Regresi) | Auto-FE, CV 5-Fold, Model Champion GBDT, dan `submission.csv` |
| **02** | [s02_universal_nlp_clean_and_fuse.py](file:///d:/POLBAN/Semester%207/gemastik-final/snippets/s02_universal_nlp_clean_and_fuse.py) | Ingesti & Pembersihan Teks Mentah Multi-Kolom | Korpus bersih terstandarisasi dengan tag `[TAG]` dan filter editorial |
| **03** | [s03_universal_nlp_baseline_tfidf.py](file:///d:/POLBAN/Semester%207/gemastik-final/snippets/s03_universal_nlp_baseline_tfidf.py) | Baseline Cepat NLP (TF-IDF + Ridge/LR) | Skor CV Stratified, Matriks Ablasi Top 5, dan `baseline_oof_probs.npy` |
| **04** | [s04_universal_transformer_finetune.py](file:///d:/POLBAN/Semester%207/gemastik-final/snippets/s04_universal_transformer_finetune.py) | Deep Learning Transformer (GPU PyTorch) | Model bobot fine-tuned dan `transformer_val_probs.npy` |
| **05** | [s05_universal_ensemble_blender.py](file:///d:/POLBAN/Semester%207/gemastik-final/snippets/s05_universal_ensemble_blender.py) | Late-Stage Ensembling & Probability Blending | Bobot optimal SLSQP dan `submission_ensemble.csv` |
| **06** | [s06_unsupervised_zero_shot_pseudo_labeler.py](file:///d:/POLBAN/Semester%207/gemastik-final/snippets/s06_unsupervised_zero_shot_pseudo_labeler.py) | Teks Tanpa Label -> Klasifikasi Supervised | Filter sampel keyakinan tinggi ($\ge 80\%$) dan `supervised_train.csv` |
| **07** | [s07_unsupervised_topic_clustering.py](file:///d:/POLBAN/Semester%207/gemastik-final/snippets/s07_unsupervised_topic_clustering.py) | Teks Tanpa Label -> Eksplorasi Klaster Topik | Pemetaan kluster topik, kata kunci c-TF-IDF, dan `clustered_train.csv` |

---

## Cara Penggunaan Cepat

### 1. Menjalankan Kasus Tabular Otomatis (Snippet 01)
```bash
python snippets/s01_universal_tabular_pipeline.py --train data/train.csv --test data/test.csv --out experiments/my_run
```

### 2. Menggabungkan Field Teks & Pembersihan Editorial (Snippet 02)
```bash
python snippets/s02_universal_nlp_clean_and_fuse.py --src data/raw_news.csv --out data/clean_train.csv
```

### 3. Melatih Baseline Teks dengan Auto-Tune Top 5 (Snippet 03)
```bash
python snippets/s03_universal_nlp_baseline_tfidf.py --train data/clean_train.csv --test data/clean_test.csv
```

### 4. Melatih Deep Learning Transformer dengan GPU (Snippet 04)
```bash
python snippets/s04_universal_transformer_finetune.py --case sentimen_karhutla --model xlm-roberta-base --epochs 3
```

### 5. Menggabungkan Prediksi OOF Menjadi Ensemble Juara (Snippet 05)
```bash
python snippets/s05_universal_ensemble_blender.py --case_dir cases/sentimen_karhutla
```

### 6. Mengubah Teks Unsupervised Menjadi Supervised (Snippet 06)
```bash
python snippets/s06_unsupervised_zero_shot_pseudo_labeler.py --src data/unlabelled_comments.csv --out data/supervised_train.csv --thresh 0.85
```

### 7. Memetakan Kluster Topik Teks Tanpa Label (Snippet 07)
```bash
python snippets/s07_unsupervised_topic_clustering.py --src data/aspirasi_warga.csv --out data/topics_train.csv --clusters 5
```
