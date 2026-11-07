# Tabular & NLP Master Studio

Platform data mining dan kompetisi machine learning otonom (*offline-first* dan *leak-free*) yang dirancang untuk perlombaan data science (Kaggle, GEMASTIK, Data Mining Datathons). Mendukung pemrosesan data tabular terstruktur maupun teks (NLP) dalam satu antarmuka web terintegrasi.

---

## 1. Persyaratan Sistem

- **Sistem Operasi**: Windows 10/11, Linux (Ubuntu/Debian), atau macOS.
- **Python**: Versi 3.10 atau lebih baru.
- **Hardware Rekomendasi**:
  - RAM minimal 8 GB (disarankan 16 GB).
  - GPU NVIDIA dengan dukungan CUDA (opsional namun sangat disarankan untuk akselerasi pelatihan model Transformer).

---

## 2. Instalasi Cepat (One-Command Setup)

Platform telah dilengkapi skrip instalasi otomatis yang membuat virtual environment, menginstal seluruh dependensi, dan mengunduh pre-trained model agar aplikasi siap digunakan 100% offline di ruangan lomba:

### Windows
Jalankan file `setup.bat` (atau melalui Command Prompt/PowerShell):
```bash
setup.bat
```

### Linux / macOS
Jalankan skrip shell berikut:
```bash
chmod +x setup.sh launch_studio.sh
./setup.sh
```

Skrip di atas secara otomatis mengeksekusi tiga tahapan:
1. Membuat virtual environment lokal `.venv`.
2. Menginstal seluruh pustaka dari `requirements.txt`.
3. Menjalankan `scripts/precache_nlp_models.py --preset all` untuk mengunduh bobot tokenizer dan model NLP (IndoBERT, IndoBERTweet, Indonesian RoBERTa, XLM-RoBERTa, RoBERTa, dan DistilBERT) ke cache lokal komputer.

---

## 3. Menjalankan Aplikasi

Setelah proses instalasi selesai, luncurkan aplikasi melalui:

### Windows
Cukup klik ganda pada berkas:
```bash
launch_studio.bat
```

### Linux / macOS / Terminal
```bash
./launch_studio.sh
# Atau jalankan manual via uvicorn
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Browser akan terbuka secara otomatis ke alamat `http://localhost:8000`. Frontend disajikan langsung oleh backend FastAPI dari folder `frontend/dist/` (tidak memerlukan instalasi Node.js pada komputer pengguna).

---

## 4. Panduan Alur Kerja Tabular Studio

Untuk kasus berbasis tabel (angka, kategori, tanggal):

1. **Pendaftaran Kasus (Register New Case)**:
   - Klik **New Case** di pojok kiri atas.
   - Pilih berkas data latih (`train.csv`) dan data uji (`test.csv`). Sistem otomatis mendeteksi kolom ID, target, dan tipe tugas (*binary*, *multiclass*, atau *regression*).
2. **Tab 1 - Dataset & EDA**:
   - Memeriksa sebaran statistik, persentase nilai kosong (*missing values*), rasio target, dan korelasi antar fitur.
3. **Tab 2 - Feature Engineering**:
   - Mengaktifkan Auto-FE: ringkasan statistik baris, kombinasi kategori berpasangan, agregasi grup (*group-by mean/std*), dan dekomposisi tanggal. Seluruh transformasi dihitung ketat pada lipatan latih (*Zero Data Leakage*).
4. **Tab 3 - Model Training**:
   - Melatih model berbasis GBDT: LightGBM, XGBoost, dan CatBoost dengan validasi silang (Stratified K-Fold, Group K-Fold, atau Time Series).
5. **Tab 4 - Hyperparameter Optimization**:
   - Menjalankan tuning Bayesian berbasis Optuna (TPE) dan menampilkan matriks komparasi 5 pengaturan parameter terbaik.
6. **Tab 5 - Ensemble Laboratory**:
   - Menggabungkan probabilitas prediksi out-of-fold (OOF) menggunakan algoritma Caruana Forward Stepwise Hill-Climbing atau regresi Nelder-Mead dalam waktu kurang dari 50 milidetik.
   - Mengunduh berkas `submission.csv`.

---

## 5. Panduan Alur Kerja Text & NLP Studio

Untuk kasus berbasis teks (sentimen, klasifikasi berita, ulasan produk, tweet, atau teks berpasangan):

1. **Ingesti & Normalisasi Format Universal**:
   - Buka modal **New Case** -> Pilih mode **Text / NLP Dataset**.
   - Sistem menerima berkas tunggal (`.csv`, `.tsv`, `.parquet`, `.xlsx`, `.jsonl`), partisi banyak berkas (`part_*.csv` via wildcard), maupun folder berkas mentah `.txt` terpisah per subfolder label kelas. Data otomatis dinormalisasi ke `cases/<case_id>/train.csv`.
2. **Tab 1 - Preprocessing & Field Fusion**:
   - **Penggabungan Kolom**: Centang kolom yang ingin disatukan (misal: `judul` + `ringkasan` + `isi`) dengan penanda konteks `[TAG]`.
   - **Pembersihan Teks**: Gunakan chip preset satu-klik (Hapus Angka, Hapus Tanda Baca, Hapus Kurung Editorial, Hapus Hashtag).
   - **Pemotongan Frasa Editorial**: Ketik kata untuk menghapus kata saja, atau akhiri dengan tanda titik tiga `...` (misal `baca juga:...` atau `simak video:...`) untuk memotong frasa beserta seluruh kalimat rekomendasi lanjutannya.
3. **Tab 2 - Corpus & Visual EDA**:
   - Menampilkan histogram sebaran panjang dokumen, garis batas **P95 Threshold** (rekomendasi *max sequence length* transformer), disparitas panjang kata antar kelas, serta grafik frekuensi unigram dan bigram.
4. **Tab 3 - TF-IDF & Baseline CV**:
   - Melatih baseline cepat TF-IDF + Balanced Ridge/Logistic Regression.
   - Klik **Auto-Tune Baseline** untuk menjalankan pencarian parameter otomatis dan meninjau tabel ablasi 5 konfigurasi terbaik.
5. **Tab 4 - Deep Learning (Transformers)**:
   - Pilih arsitektur pretrained yang sesuai:
     - `xlm-roberta-base`: Untuk teks multibahasa, campuran Indonesia-Inggris, atau bahasa gaul/slang.
     - `indobenchmark/indobert-base-p1`: Untuk bahasa Indonesia formal dan berita.
     - `indolem/indobertweet-base-uncased`: Untuk media sosial dan komentar informal.
     - `roberta-base` / `microsoft/deberta-v3-small`: Untuk bahasa Inggris.
   - Learning rate dan batch size otomatis disesuaikan secara cerdas sesuai model yang dipilih.
6. **Tab 5 - Model Arena, Training Logs & Blending**:
   - Meninjau grafik visual komparasi performa antara Baseline, Transformer, dan Ensemble.
   - Mengatur pembobotan probabilitas *lexical + semantic* via slider interaktif.
   - Meninjau log pelatihan live GPU terminal, log metrik validasi, serta mengunduh berkas hasil akhir `submission_ensemble.csv`.

---

## 6. Penanganan Kasus Teks Tanpa Label (Unsupervised)

Jika panitia lomba memberikan dataset teks mentah tanpa kolom target:

1. **Kasus Sentimen / Kategori Terselubung (Zero-Shot Pseudo-Labeling)**:
   - Gunakan model pretrained klasifikasi (misal `w11wo/indonesian-roberta-base-sentiment-classifier`).
   - Saring sampel yang memiliki tingkat keyakinan (*confidence score*) $\ge 0.85$ sebagai target label awal.
   - Simpan hasilnya ke `cases/<case_id>/train.csv` dan lanjutkan proses secara terawasi (*supervised*) mulai dari Tab 1.
2. **Kasus Eksplorasi Topik / Pemetaan Masalah Murni**:
   - Ekstrak representasi embedding padat menggunakan fitur `extract_transformer_embeddings`.
   - Lakukan pengelompokan (*clustering*) berbasis K-Means atau HDBSCAN.
   - Tetapkan nomor kluster sebagai target label dan analisis kata kunci representatif per kelompok.

---

## 7. Protokol Kolaborasi Tim 3 Orang (Distributed Offline Training)

Untuk menghemat waktu saat batas pengumpulan mendekati akhir:

1. **Distribusi Mesin**:
   - Anggota 1 (Laptop A): Melatih model LightGBM (`models: ["lgbm"]`).
   - Anggota 2 (Laptop B): Melatih model CatBoost (`models: ["catboost"]`).
   - Anggota 3 (PC Dosen / Laptop C GPU): Melatih model Transformer (IndoBERT/XLM-RoBERTa).
2. **Penggabungan Sub-Detik**:
   - Salin berkas matriks prediksi out-of-fold (`.npy`) dari masing-masing komputer ke folder `experiments/<case_id>_run/artifacts/` pada salah satu komputer utama.
   - Buka Tab Ensemble pada aplikasi dan klik **Auto-Find Optimal Blend**. Platform akan menghitung bobot gabungan terbaik dan mencetak file submission akhir tanpa perlu melatih ulang model dari awal.

---

## 8. Bantuan & Panduan Terintegrasi dalam Aplikasi

Panduan taktis interaktif dan rangkuman alur kompetisi juga dapat diakses kapan saja langsung dari dalam aplikasi dengan mengklik tombol **User Guide** pada navigasi atas web studio.

---

## 9. Lisensi

Platform ini dikembangkan untuk keperluan riset data mining, pengajaran akademik, dan kompetisi data science nasional (GEMASTIK) maupun internasional. Lisensi di bawah MIT License.
