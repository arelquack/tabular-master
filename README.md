# Tabular Master Studio

Tabular Master Studio is an autonomous, leak-free, offline-first tabular data mining and competitive machine learning platform. Designed for data science competitions (Kaggle, Gemastik, Datathons) and enterprise tabular tasks, it automates exploratory data analysis, generic feature engineering, cross-validation, Bayesian hyperparameter optimization (Optuna), and sub-second ensembling.

---

## Key Capabilities

1. **Automated Leak-Free Pipeline**:
   - Strict validation isolation across Stratified K-Fold, Group K-Fold, and Time Series splits.
   - Encodings, aggregations, and statistics computed exclusively on training folds to prevent target and data leakage.

2. **Universal Automated Feature Engineering (Auto-FE)**:
   - Domain-agnostic mathematical primitives:
     - Row-wise summary statistics (Mean, Std, Min, Max, Zero-counts across homogeneous numeric groups).
     - Pairwise categorical cross-interactions with rarity/frequency mapping.
     - Group-by aggregations and relative difference/Z-score features.
     - Temporal datetime decomposition.
     - Missingness signature flags and variance/collinearity pruning (r > 0.98).

3. **Interactive Visual EDA Studio**:
   - Dynamic Feature Pattern & Target Response Explorer.
   - Binned population frequency histograms coupled with conditional target probability distributions.
   - Target correlation ranking and schema diagnostics.

4. **Automated Hyperparameter Optimization (Optuna Engine)**:
   - Integrated Bayesian optimization powered by Tree-structured Parzen Estimator (TPE).
   - Optimizes tree architecture, learning rate, and L1/L2 regularization for LightGBM, XGBoost, and CatBoost.
   - One-click configuration export and persistence.

5. **Sub-Second Ensemble Laboratory**:
   - Caruana Forward Stepwise Hill-Climbing with replacement.
   - Nelder-Mead / SLSQP Continuous Rank & Probability Blending.
   - Interactive weight sliders with instantaneous recalculation (< 50ms) from pre-cached out-of-fold prediction matrices.

---

## Distributed Team Collaboration Workflow (3-Person Protocol)

In competitive environments with strict time limits and compute constraints, a 3-person team can distribute training across machines and merge predictions in sub-seconds:

### The Problem
Training full 5-fold cross-validation on 3 model families (LightGBM, XGBoost, CatBoost) sequentially on a single laptop CPU can take 45-60 minutes.

### The Solution: Distributed Parallel Training
Because `TabularPipeline` relies on deterministic random seeding and identical fold generation, the team divides model execution:

1. **Prerequisite**:
   All 3 team members clone the repository and use identical `case.json` settings (e.g. `seed: 2026`, `cv_strategy: "stratified_kfold"`, `n_splits: 5`).

2. **Execution Partitioning**:
   - **Member 1 (Machine A)**: Runs LightGBM (`models: ["lgbm"]`).
   - **Member 2 (Machine B)**: Runs XGBoost (`models: ["xgb"]`).
   - **Member 3 (Machine C)**: Runs CatBoost (`models: ["catboost"]`).
   
   *Result*: Total team training time is cut from 45 minutes down to ~15 minutes (or ~2 minutes with GPU).

3. **Artifact Sharing**:
   Each member only exports their generated cache files from `experiments/<case_id>_run/artifacts/`:
   - Member 1 provides: `oof_lgbm.npy` and `test_lgbm.npy`
   - Member 2 provides: `oof_xgb.npy` and `test_xgb.npy`
   - Member 3 provides: `oof_catboost.npy` and `test_catboost.npy`

4. **Zero-Lag Ensembling**:
   Place all 6 `.npy` files into the `experiments/<case_id>_run/artifacts/` directory on any member machine. 
   Open Tabular Studio -> **Tab 4 (Ensemble Laboratory)**:
   - Click **"Optimize Weights" (Caruana Hill Climbing)** or adjust manual weight sliders.
   - The platform evaluates the ensemble out-of-fold score and generates `submission.csv` in less than 50 milliseconds without retraining.

---

## Directory Structure

```
gemastik-final/
├── cases/                          # Competition case definitions and dataset configs
│   ├── airline_s6e10/              # Playground s6e10 Airline Satisfaction
│   ├── aksi_28_okt/                # Opinion Mining & Public Sentiment (Aksi 28 Okt)
│   ├── campus_complaints_routing/  # Multiclass NLP Bureau Routing
│   ├── credit_card_fraud/          # Extreme Imbalance Tabular Fraud Detection
│   ├── f1_podium/                  # Relational Star-Schema F1 Prediction
│   ├── home_credit/                # Home Credit Default Risk
│   ├── nfl_big_data_bowl_2026/     # GroupKFold Spatial-Temporal NFL Regression
│   ├── nlp_kompas/                 # Multiclass News Classification
│   ├── sentimen_karhutla/          # Environmental Sentiment Classification
│   └── spaceship_titanic/          # Classic Tabular Benchmark
├── docs/                           # Official guides, competition rules & assets
│   ├── Panduan-GEMASTIK-2026.pdf   # GEMASTIK competition guideline
│   └── assets/                     # EDA visual assets & sample charts
├── experiments/                    # Model outputs, checkpoints, OOF caches & submissions
├── frontend/                       # React 19 + TypeScript + Tailwind CSS UI
│   ├── dist/                       # Pre-built static distribution bundle
│   ├── src/                        # React UI components
│   └── vite.config.ts
├── scripts/                        # Standalone utility and automation scripts
│   ├── download_models.py          # Offline HuggingFace model cache downloader
│   ├── auto_label_aksi.py          # Zero-shot pseudo-labeler & validator
│   └── runners/                    # Standalone CLI runners for cases
│       ├── run_case1_spaceship.py
│       ├── run_case2_airline.py
│       ├── run_case3_homecredit.py
│       ├── run_s6e10.py
│       └── run_roberta_sentiment.py
├── tabular_pipeline/               # Core Python Data Mining & ML Framework
│   ├── models/                     # GBDT & linear model wrappers (LGBM, XGB, CatBoost)
│   ├── config.py                   # Strict pipeline dataclass configuration
│   ├── data_detector.py            # Automated column schema & cardinality detector
│   ├── diagnostics.py              # Adversarial validation drift tester
│   ├── ensemble.py                 # Caruana, SLSQP, Stacking, and Rank optimizers
│   ├── feature_engine.py           # Universal Auto-FE and BaseFeatureHook
│   ├── metrics.py                  # Task-aligned metric calculation & threshold tuning
│   ├── pipeline.py                 # TabularPipeline orchestrator
│   ├── trainer.py                  # Leak-free cross-validator
│   └── tuner.py                    # Optuna hyperparameter optimization engine
├── templates/                      # Standardized project starters
│   ├── tabular/                    # Starter template for tabular challenges
│   └── nlp/                        # 5 modular starter templates for NLP tasks
├── launch_studio.bat               # Windows one-click desktop launcher
├── run.bat                         # Quick CLI execution launcher
├── server.py                       # FastAPI unified backend server
├── train_transformer_universal.py  # Universal PyTorch / Transformers training CLI
├── requirements.txt                # Production Python dependencies
└── README.md
```

---

## Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/arelquack/tabular-master.git
cd tabular-master
```

### 2. Setup Python Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Tabular Studio
On Windows, double-click `launch_studio.bat` or run:
```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```
Open your browser at `http://127.0.0.1:8000`. The frontend is served directly by FastAPI from the pre-built `frontend/dist` directory (no Node.js installation required).

---

## Adding a Custom Competition Case

1. Create a directory under `cases/<case_id>/`.
2. Add a `case.json` definition:
```json
{
    "id": "scholarship_eligibility",
    "name": "Student Scholarship Eligibility",
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "id_col": "student_id",
    "target_col": "eligible",
    "task_type": "binary",
    "eval_metric": "roc_auc",
    "cv_strategy": "stratified_kfold",
    "n_splits": 5,
    "seed": 2026,
    "models": ["lgbm", "xgb", "catboost"]
}
```
3. (Optional) Provide a `feature_hook.py` inheriting from `BaseFeatureHook` for domain-specific transformations.
4. Refresh the studio dashboard or click **"New Case"** in the web interface.

---

## License

MIT License. Designed for academic and competition data mining excellence.
