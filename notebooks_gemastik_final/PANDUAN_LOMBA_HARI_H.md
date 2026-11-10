# PANDUAN EKSEKUSI HARI-H FINAL GEMASTIK XIX 2026
## DIVISI III: PENAMBANGAN DATA (DATA MINING)
---

Berdasarkan **Technical Meeting (TM)** resmi dan strategi penambangan data teks, berikut panduan operasional lengkap untuk tim.

---

### 1. RINGKASAN ATURAN KRUSIAL DARI TM (SLIDE 1–15)
1. **Jadwal Eksekusi**:
   - **Penyelesaian Kasus**: Rabu, 11 Nov 2026, **13.00 – 18.00 WIB (5 Jam Non-Stop, Tanpa Break)**.
   - **Koneksi Internet**: **HANYA DIBUKA pukul 17.00 – 18.00 WIB** (1 jam terakhir untuk unduh submit Kaggle dan upload deliverables).
   - **Batas Pengumpulan PPT Final**: Paling lambat pukul **21.00 WIB**.
   - **Presentasi Luring**: Kamis, 12 Nov 2026, **08.00 – 17.00 WIB** (10 menit presentasi + 10 menit tanya jawab).
2. **Kriteria Penilaian**:
   - **30%**: Skor Kinerja Leaderboard Kaggle
   - **25%**: Ketepatan Metode (Kesesuaian metode penambangan data dengan kasus)
   - **25%**: Presentasi (Kejelasan visual, alur metodologi, kelengkapan PPT)
   - **20%**: Nilai Technical Report Penyisihan
   *(Perhatikan: 50% nilai ada di Metode + Presentasi!)*
3. **Ketentuan Larangan Keras**:
   - **NILAI 0** jika menggunakan *zero-shot* atau *few-shot prompting* LLM.
   - Hasil akhir **WAJIB** berasal dari model hasil pembelajaran (*training*) yang dibangun oleh tim.
   - Maksimal **3 kali submission** ke Kaggle leaderboard.
   - Dataset bersifat rahasia (NDA). Dilarang menyalin dataset keluar dari PC Panitia.

---

### 2. STRATEGI HARDWARE & LIBRARY TERSEDIA (PLAN B: CPU MASTERY)
- **Kondisi Perangkat**:
  - CPU: Intel Core i7-12700 (12 Cores / 20 Threads)
  - RAM: 32 GB DDR5
  - Library Siap Pakai: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `transformers`.
  - Library Missing: `autogluon`, `nltk`, `spacy`, `gensim`.
- **Keputusan Strategis**:
  - Jangan pusing mencari GPU/PyTorch CUDA jika environment terbatas di CPU.
  - Kombinasi **TF-IDF + Linear Classifier (Balanced) + TruncatedSVD (LSA) + LightGBM** pada CPU i7-12700 (20 threads) selesai dalam **<60 detik**, sangat stabil, dan tidak ada risiko *memory error* atau *training crash*.

---

### 3. DAFTAR MASTER NOTEBOOKS (SUDAH DISIAPKAN)

Seluruh notebook berada di folder: `notebooks_gemastik_final/`

| Nama Notebook | Deskripsi & Tugas | Output yang Dihasilkan |
| :--- | :--- | :--- |
| **`01_task1_text_classification_master.ipynb`** | **Task 1 (Klasifikasi Teks)**: Pembersihan teks mandiri, Stratified 5-Fold CV, Model Linear Balanced, Model Tree (LightGBM on LSA + Meta Features), Soft-Voting Blend. | `output_submissions/submission_task1.csv`<br>`output_charts/classification_confusion_matrix.png`<br>`output_charts/top_informative_terms_per_class.png` |
| **`02_task2_topic_modeling_master.ipynb`** | **Task 2 (Topic Modeling)**: Ekstraksi topik unsupervised berbasis TF-IDF + NMF (Non-Negative Matrix Factorization), tabel kata kunci tematik, proyeksi 2D klaster dokumen. | `output_topics/topic_modeling_summary.csv`<br>`output_charts/nmf_topic_keywords_barchart.png`<br>`output_charts/topic_clusters_projection_2d.png` |
| **`03_master_integrated_solution.ipynb`** | **Unified Pipeline (Fusi Dua Task)**: Menjalankan Topic Modeling untuk mengekstrak fitur distribusi topik, lalu diinjeksikan sebagai fitur prediktif ke Klasifikasi Teks. | `output_submissions/submission_final_gemastik.csv`<br>Hasil validasi silang tertinggi |

---

### 4. ALUR KERJA DI PC PANITIA PUKUL 13.00 WIB

1. **Buka Visual Studio Code / Jupyter Notebook**:
   - Arahkan ke folder proyek `gemastik-final`.
   - Pastikan kernel Jupyter menggunakan Python Virtual Environment proyek (`.venv`).
2. **Cek Lokasi Dataset Panitia**:
   - Buka notebook `01_task1_text_classification_master.ipynb`.
   - Di Sel 2, ubah path dataset:
     ```python
     TRAIN_PATH = "path/ke/train.csv_panitia"
     TEST_PATH = "path/ke/test.csv_panitia"
     TEXT_COL = "nama_kolom_teks"
     TARGET_COL = "nama_kolom_target"
     ```
3. **Jalankan Run All (Shift + Enter per sel)**:
   - Selesai dalam 1–2 menit!
   - File `output_submissions/submission_task1.csv` langsung siap disubmit saat internet dibuka pukul 17.00.
4. **Buka Notebook 02 untuk Topic Modeling**:
   - Jalankan notebook `02_task2_topic_modeling_master.ipynb`.
   - Ambil tabel ringkasan topik dan grafik bar chart untuk disalin ke slide PowerPoint.
5. **Jalankan Notebook 03 (Integrasi)**:
   - Menghasilkan submission final dengan skor F1 tertinggi.

---

### 5. NARASI ILMIAH & PRESENTASI JURI (BOBOT 50%!)

Gunakan poin-poin penjelasan berikut saat presentasi 10 menit di hadapan dewan juri:

1. **Kenapa Memilih NMF untuk Topic Modeling (Task 2)?**
   - *"Kami menggunakan Non-Negative Matrix Factorization (NMF) di atas representasi TF-IDF dengan inisialisasi NNDSVDA. Pada dokumen teks berukuran pendek hingga menengah, NMF menghasilkan topik yang lebih koheren secara semantik dan minim overlapping antar topik dibandingkan LDA tradisional, serta bersifat deterministik dan cepat."*
2. **Kenapa Memadukan Linear-SVM/Logistic Regression dengan Gradient Boosting (Task 1)?**
   - *"Teks berada di ruang berdimensi tinggi sparse. Linear classifier dengan L2-regularization bekerja sangat optimal menemukan hyperplane pemisah margin tanpa overfitting. Sementara itu, LightGBM di atas komponen LSA (Latent Semantic Analysis) menangkap interaksi non-linear dan struktur meta teks. Melalui soft-voting blend, kami mencapai trade-off bias-variance optimal."*
3. **Nilai Plus Integrasi Dua Task (Killer Feature):**
   - *"Kedua tugas ini tidak kami kerjakan secara terisolasi. Matriks distribusi probabilitas topik dokumen dari Task 2 kami injeksikan sebagai fitur semantik tambahan pada Task 1. Ini membuktikan sinergi nyata penambangan data teks secara komprehensif."*
4. **Efisiensi Komputasi & Sustainable AI:**
   - *"Seluruh pipeline dirancang hemat daya (sustainable), konvergen dalam waktu kurang dari 90 detik di CPU multi-core, bebas ketergantungan koneksi jaringan, dan mudah diinterpretasikan fitur terpentingnya (transparent AI).*
