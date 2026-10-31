import os
import sys
import json
import glob
import importlib.util
import threading
import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook, HyperparameterTuner
from tabular_pipeline.ensemble import EnsembleOptimizer
from tabular_pipeline.metrics import MetricEvaluator

app = FastAPI(title="Tabular Mining Studio API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CASES_DIR = os.path.join(ROOT_DIR, "cases")
os.makedirs(CASES_DIR, exist_ok=True)

# In-memory execution state, EDA cache, and tuning state
job_state = {}
eda_cache = {}
tune_state = {}

class CreateCaseRequest(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    train_path: str
    test_path: Optional[str] = None
    auxiliary_paths: Optional[List[str]] = []
    id_col: Optional[str] = None
    target_col: str
    task_type: str = "binary"
    eval_metric: str = "roc_auc"
    cv_strategy: str = "stratified_kfold"
    n_splits: int = 5
    seed: int = 2026
    models: List[str] = ["lgbm", "xgb", "catboost"]
    hook_code: Optional[str] = None

class TuneRequest(BaseModel):
    model: str = "lgbm"
    n_trials: int = 15

class BlendRequest(BaseModel):
    weights: Dict[str, float]

class InspectDatasetRequest(BaseModel):
    path: str

def load_hook_class(case_dir: str):
    hook_file = os.path.join(case_dir, "feature_hook.py")
    if not os.path.exists(hook_file):
        return None
    try:
        spec = importlib.util.spec_from_file_location("custom_feature_module", hook_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseFeatureHook) and attr is not BaseFeatureHook:
                return attr()
    except Exception as e:
        print(f"[ERROR] Failed to load hook from {hook_file}: {e}")
    return None

@app.get("/api/cases")
def list_cases():
    cases = []
    case_dirs = glob.glob(os.path.join(CASES_DIR, "*"))
    for c_dir in case_dirs:
        case_json = os.path.join(c_dir, "case.json")
        if os.path.exists(case_json):
            try:
                with open(case_json, "r") as f:
                    meta = json.load(f)
                
                out_dir = meta.get("output_dir", f"experiments/{meta['id']}_run")
                summary_file = os.path.join(ROOT_DIR, out_dir, "summary.json")
                summary_data = None
                if os.path.exists(summary_file):
                    with open(summary_file, "r") as sf:
                        summary_data = json.load(sf)
                
                meta["status"] = job_state.get(meta["id"], {}).get("status", "completed" if summary_data else "ready")
                meta["summary"] = summary_data
                cases.append(meta)
            except Exception as e:
                print(f"[ERROR] Reading case {c_dir}: {e}")
    return {"cases": cases}

@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    out_dir = meta.get("output_dir", f"experiments/{meta['id']}_run")
    full_out_dir = os.path.join(ROOT_DIR, out_dir)
    summary_file = os.path.join(full_out_dir, "summary.json")
    
    summary_data = {}
    if os.path.exists(summary_file):
        with open(summary_file, "r") as sf:
            summary_data = json.load(sf)

    # Feature Importances
    fi_file = os.path.join(full_out_dir, "feature_importances.csv")
    feature_importances = []
    if os.path.exists(fi_file):
        fi_df = pd.read_csv(fi_file)
        agg_fi = fi_df.groupby("feature")["importance"].mean().reset_index()
        agg_fi = agg_fi.sort_values("importance", ascending=False).head(30)
        feature_importances = agg_fi.to_dict(orient="records")

    # Hook code
    hook_file = os.path.join(case_dir, "feature_hook.py")
    hook_code = ""
    if os.path.exists(hook_file):
        with open(hook_file, "r") as hf:
            hook_code = hf.read()

    # Cached models for ensembling
    artifacts_dir = os.path.join(full_out_dir, "artifacts")
    all_models = ["lgbm", "xgb", "catboost"]
    available_cached = [m for m in all_models if os.path.exists(os.path.join(artifacts_dir, f"oof_{m}.npy"))]

    return {
        "meta": meta,
        "summary": summary_data,
        "feature_importances": feature_importances,
        "hook_code": hook_code,
        "available_cached": available_cached,
        "job_state": job_state.get(case_id, {"status": "completed" if summary_data else "ready", "progress": 0, "logs": []})
    }

@app.get("/api/cases/{case_id}/eda")
def get_case_eda(case_id: str):
    if case_id in eda_cache:
        return eda_cache[case_id]

    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    train_path = os.path.join(ROOT_DIR, meta["train_path"])
    if not os.path.exists(train_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {train_path}")

    # 1. Exact Total Rows Count (Fast line count in binary)
    with open(train_path, "rb") as f:
        total_rows = sum(1 for _ in f) - 1

    # 2. Sample data for in-depth statistical profiling (up to 50k rows)
    df_sample = pd.read_csv(train_path, nrows=50000)
    target_col = meta.get("target_col")
    id_col = meta.get("id_col")

    # 3. Summary metrics
    total_cols = len(df_sample.columns)
    total_missing_cells = int(df_sample.isna().sum().sum())
    total_cells = len(df_sample) * total_cols
    raw_pct = (total_missing_cells / total_cells) * 100
    missing_pct_display = "< 0.01%" if 0 < raw_pct < 0.01 else f"{round(raw_pct, 2)}%"
    file_size_mb = round(os.path.getsize(train_path) / (1024 * 1024), 2)

    null_cols = [c for c in df_sample.columns if df_sample[c].isna().sum() > 0]
    null_cols_note = f"in {', '.join(null_cols)}" if null_cols else "No null cells"

    # 4. Target Distribution
    target_dist = []
    if target_col and target_col in df_sample.columns:
        vc = df_sample[target_col].value_counts(dropna=False)
        for val, count in vc.items():
            pct = round((count / len(df_sample)) * 100, 1)
            target_dist.append({
                "label": str(val),
                "count": int(count),
                "percentage": pct
            })

    # 5. Correlations with Target
    correlations = []
    if target_col and target_col in df_sample.columns:
        y_s = df_sample[target_col]
        if y_s.dtype == bool or not np.issubdtype(y_s.dtype, np.number):
            y_num = pd.Series(y_s).astype("category").cat.codes
        else:
            y_num = y_s

        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            if c != id_col and c != target_col:
                corr_val = df_sample[c].corr(y_num)
                if pd.notnull(corr_val):
                    correlations.append({
                        "feature": c,
                        "correlation": round(corr_val, 4)
                    })
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # 6. Detailed Column Profiles
    columns_profile = []
    for c in df_sample.columns:
        s = df_sample[c]
        is_num = np.issubdtype(s.dtype, np.number)
        missing_count = int(s.isna().sum())
        missing_rate = round((missing_count / len(s)) * 100, 2)
        nunique = int(s.nunique())

        col_info = {
            "name": c,
            "is_target": c == target_col,
            "is_id": c == id_col,
            "type": "Numerical" if is_num else ("Datetime" if "date" in c.lower() or "time" in c.lower() else "Categorical"),
            "dtype": str(s.dtype),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "unique_count": nunique
        }

        if is_num:
            col_info["stats"] = {
                "min": round(float(s.min()), 2) if pd.notnull(s.min()) else None,
                "mean": round(float(s.mean()), 2) if pd.notnull(s.mean()) else None,
                "median": round(float(s.median()), 2) if pd.notnull(s.median()) else None,
                "max": round(float(s.max()), 2) if pd.notnull(s.max()) else None
            }
        else:
            top_val = s.mode().iloc[0] if not s.dropna().empty else "N/A"
            col_info["stats"] = {
                "top_value": str(top_val)
            }

        # Calculate rich visual distribution buckets with target breakdown
        dist_bins = []
        try:
            if nunique <= 12 or not is_num:
                val_counts = s.value_counts(dropna=False).head(10)
                for cat_val, count in val_counts.items():
                    cat_str = str(cat_val) if pd.notnull(cat_val) else "Missing"
                    b_dict = {
                        "label": cat_str,
                        "count": int(count),
                        "pct": round((count / len(s)) * 100, 1)
                    }
                    if target_col and target_col in df_sample.columns and len(target_dist) == 2:
                        subset = df_sample[s == cat_val] if pd.notnull(cat_val) else df_sample[s.isna()]
                        if not subset.empty:
                            pos_label = target_dist[1]["label"]
                            pos_count = int((subset[target_col].astype(str) == pos_label).sum())
                            b_dict["target_rate"] = round((pos_count / len(subset)) * 100, 1)
                    dist_bins.append(b_dict)
            else:
                s_clean = s.dropna()
                if not s_clean.empty and s_clean.nunique() > 1:
                    counts, bin_edges = np.histogram(s_clean, bins=10)
                    for i in range(len(counts)):
                        b_low = round(float(bin_edges[i]), 1)
                        b_high = round(float(bin_edges[i+1]), 1)
                        b_cnt = int(counts[i])
                        b_dict = {
                            "label": f"{b_low} - {b_high}",
                            "count": b_cnt,
                            "pct": round((b_cnt / len(s)) * 100, 1)
                        }
                        if target_col and target_col in df_sample.columns and len(target_dist) == 2:
                            in_bin = df_sample[(s >= bin_edges[i]) & (s <= bin_edges[i+1])]
                            if not in_bin.empty:
                                pos_label = target_dist[1]["label"]
                                pos_count = int((in_bin[target_col].astype(str) == pos_label).sum())
                                b_dict["target_rate"] = round((pos_count / len(in_bin)) * 100, 1)
                        dist_bins.append(b_dict)
        except Exception:
            pass

        col_info["distribution"] = dist_bins
        columns_profile.append(col_info)

    preview_rows = df_sample.head(25).fillna("NaN").to_dict(orient="records")

    result = {
        "summary": {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "file_size_mb": file_size_mb,
            "missing_cells_sample": total_missing_cells,
            "missing_pct_sample": missing_pct_display,
            "null_cols_note": null_cols_note,
            "target_col": target_col
        },
        "target_distribution": target_dist,
        "correlations": correlations[:12],
        "columns_profile": columns_profile,
        "preview_rows": preview_rows
    }
    eda_cache[case_id] = result
    return result

@app.post("/api/cases")
def create_case(req: CreateCaseRequest):
    case_dir = os.path.join(CASES_DIR, req.id)
    os.makedirs(case_dir, exist_ok=True)
    
    case_data = req.dict(exclude={"hook_code"})
    case_data["output_dir"] = f"experiments/{req.id}_run"
    
    with open(os.path.join(case_dir, "case.json"), "w") as f:
        json.dump(case_data, f, indent=4)

    if req.hook_code:
        with open(os.path.join(case_dir, "feature_hook.py"), "w") as f:
            f.write(req.hook_code)

    return {"status": "created", "case_id": req.id}

@app.post("/api/inspect-dataset")
def inspect_dataset(req: InspectDatasetRequest):
    file_path = os.path.join(ROOT_DIR, req.path) if not os.path.isabs(req.path) else req.path
    if not os.path.exists(file_path):
        return {"exists": False, "error": f"File does not exist: {file_path}"}
    
    try:
        df_preview = pd.read_csv(file_path, nrows=50)
        col_stats = []
        for c in df_preview.columns:
            dtype_str = str(df_preview[c].dtype)
            missing = int(df_preview[c].isna().sum())
            nunique = int(df_preview[c].nunique())
            col_stats.append({
                "column": c,
                "dtype": dtype_str,
                "missing_in_sample": missing,
                "unique_in_sample": nunique
            })
        return {
            "exists": True,
            "total_columns": len(df_preview.columns),
            "columns": col_stats,
            "preview_rows": df_preview.fillna("NaN").to_dict(orient="records")
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}

def run_pipeline_thread(case_id: str, meta: dict, case_dir: str):
    job_state[case_id] = {
        "status": "running",
        "progress": 5,
        "current_model": "Preprocessing",
        "current_fold": "Setup",
        "logs": [f"[{time.strftime('%H:%M:%S')}] Starting pipeline execution..."]
    }
    def log(msg):
        job_state[case_id]["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        print(f"[{case_id}] {msg}")

    try:
        train_path = os.path.join(ROOT_DIR, meta["train_path"])
        test_path = os.path.join(ROOT_DIR, meta["test_path"]) if meta.get("test_path") else None
        
        log(f"Loading dataset from {meta['train_path']}...")
        job_state[case_id]["progress"] = 12
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path and os.path.exists(test_path) else None

        # Augment with auxiliary data if configured (e.g. Kaggle original.csv)
        if meta.get("auxiliary_paths"):
            for aux_p in meta["auxiliary_paths"]:
                full_aux = os.path.join(ROOT_DIR, aux_p)
                if os.path.exists(full_aux):
                    log(f"Loading auxiliary dataset: {aux_p}...")
                    aux_df = pd.read_csv(full_aux)
                    common_cols = [c for c in train_df.columns if c in aux_df.columns]
                    train_df["is_original"] = 0
                    aux_df["is_original"] = 1
                    if test_df is not None:
                        test_df["is_original"] = 0
                    train_df = pd.concat([train_df, aux_df[common_cols + ["is_original"]]], ignore_index=True)
                    log(f"Dataset augmented with original data: {len(train_df):,} total rows.")

        job_state[case_id]["progress"] = 20
        log(f"Final training matrix: {len(train_df):,} rows, {len(train_df.columns)} columns.")

        cfg = PipelineConfig(
            project_name=meta.get("name", case_id),
            output_dir=meta.get("output_dir", f"experiments/{case_id}_run"),
            seed=meta.get("seed", 2026),
            id_col=meta.get("id_col"),
            target_col=meta.get("target_col"),
            task_type=meta.get("task_type", "binary"),
            eval_metric=meta.get("eval_metric", "roc_auc"),
            cv_strategy=meta.get("cv_strategy", "stratified_kfold"),
            n_splits=meta.get("n_splits", 5),
            models=meta.get("models", ["lgbm", "xgb", "catboost"]),
            model_params=meta.get("model_params", {}),
            ensemble_methods=["mean", "rank", "opt_weights", "hill_climbing", "stacking"] if len(meta.get("models", [])) > 1 else [],
            run_adversarial_validation=True,
            save_oof_predictions=True,
            save_test_predictions=True
        )

        hook = load_hook_class(case_dir)
        if hook:
            log("Loaded custom feature engineering hook.")

        total_models = len(cfg.models)
        total_folds = cfg.n_splits
        total_units = max(1, total_models * total_folds)
        completed_units = 0

        def on_progress(info: dict):
            nonlocal completed_units
            stage = info.get("stage", "")
            msg = info.get("message", "")

            if stage == "training_fold_done":
                completed_units += 1
                pct = int(22 + (completed_units / total_units) * 66)
                job_state[case_id]["progress"] = min(90, pct)
                job_state[case_id]["current_model"] = info.get("model", "").upper()
                job_state[case_id]["current_fold"] = f"{info.get('fold')}/{info.get('total_folds')}"
                log(msg)
            elif stage == "training_fold_start":
                job_state[case_id]["current_model"] = info.get("model", "").upper()
                job_state[case_id]["current_fold"] = f"{info.get('fold')}/{info.get('total_folds')}"
                log(msg)
            elif stage == "ensembling":
                job_state[case_id]["progress"] = 92
                job_state[case_id]["current_model"] = "Ensemble"
                job_state[case_id]["current_fold"] = "Caruana"
                log(msg)
            else:
                log(msg)

        log(f"Starting {cfg.n_splits}-fold CV on {', '.join(cfg.models).upper()}...")
        job_state[case_id]["progress"] = 22

        pipeline = TabularPipeline(cfg, feature_hook=hook, progress_callback=on_progress)
        results = pipeline.run(train_df, test_df)

        job_state[case_id]["progress"] = 100
        job_state[case_id]["status"] = "completed"
        job_state[case_id]["current_model"] = "Completed"
        job_state[case_id]["current_fold"] = "Done"
        champ_name = results.get('champion_model', 'N/A')
        champ_score = results.get('champion_score', 0.0)
        log(f"All stages finished! Champion: {champ_name} (Score: {champ_score:.5f})")

    except Exception as e:
        job_state[case_id]["status"] = "error"
        log(f"Error during execution: {str(e)}")

@app.post("/api/cases/{case_id}/run")
def run_case_pipeline(case_id: str, background_tasks: BackgroundTasks):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    if job_state.get(case_id, {}).get("status") == "running":
        return {"status": "already_running"}

    thread = threading.Thread(target=run_pipeline_thread, args=(case_id, meta, case_dir))
    thread.daemon = True
    thread.start()

    return {"status": "started", "case_id": case_id}

@app.get("/api/cases/{case_id}/status")
def get_case_status(case_id: str):
    return job_state.get(case_id, {"status": "idle", "progress": 0, "logs": []})

def run_tune_thread(case_id: str, model_name: str, n_trials: int, meta: dict, case_dir: str):
    tune_state[case_id] = {
        "status": "running",
        "model": model_name,
        "current_trial": 0,
        "total_trials": n_trials,
        "best_score": None,
        "best_params": None,
        "trials": [],
        "logs": [f"[{time.strftime('%H:%M:%S')}] Initializing Optuna study for {model_name.upper()}..."]
    }
    def tlog(msg):
        tune_state[case_id]["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    try:
        train_path = os.path.join(ROOT_DIR, meta["train_path"])
        train_df = pd.read_csv(train_path)

        target_col = meta["target_col"]
        id_col = meta.get("id_col")
        y_raw = train_df[target_col].values
        if meta["task_type"] == "binary":
            y = (y_raw == 1) | (y_raw == True) | (y_raw == "True") if y_raw.dtype == bool or not np.issubdtype(y_raw.dtype, np.number) else y_raw
            if not np.issubdtype(y.dtype, np.number):
                y = pd.Series(y_raw).astype("category").cat.codes.values
            y = y.astype(int)
        else:
            y = y_raw

        drop_cols = [c for c in [target_col, id_col] if c and c in train_df.columns]
        X = train_df.drop(columns=drop_cols)

        # Check feature hook
        hook = load_hook_class(case_dir)
        if hook:
            X, _ = hook.transform(X, None)

        # Sanitize column names
        import re
        clean_map = {c: re.sub(r'[\[\]<>, "]+', '_', str(c)) for c in X.columns}
        X = X.rename(columns=clean_map)

        # Convert object columns to category codes for speed
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category").cat.codes

        tlog(f"Starting {n_trials} Bayesian Optimization trials on {len(X):,} rows x {len(X.columns)} features...")

        def on_tune_progress(pinfo: dict):
            trial_num = pinfo["trial"]
            sc = pinfo["score"]
            bsc = pinfo["best_score"]
            is_best = pinfo["is_new_best"]
            tune_state[case_id]["current_trial"] = trial_num
            tune_state[case_id]["best_score"] = bsc
            tune_state[case_id]["trials"].append({
                "trial": trial_num,
                "score": sc,
                "best_score": bsc,
                "is_new_best": is_best,
                "params": pinfo["params"]
            })
            marker = " [NEW BEST]" if is_best else ""
            tlog(f"Trial {trial_num}/{n_trials} -> Score: {sc:.5f} (Best: {bsc:.5f}){marker}")

        tuner = HyperparameterTuner(
            task_type=meta["task_type"],
            eval_metric=meta["eval_metric"],
            seed=meta.get("seed", 2026),
            progress_callback=on_tune_progress
        )

        res = tuner.tune_model(
            model_name=model_name,
            X=X,
            y=y,
            n_trials=n_trials,
            sample_size=60000,
        )

        tune_state[case_id]["status"] = "completed"
        tune_state[case_id]["best_score"] = res["best_score"]
        tune_state[case_id]["best_params"] = res["best_params"]
        tlog(f"Optuna Optimization Completed! Best {meta['eval_metric'].upper()}: {res['best_score']:.5f}")

    except Exception as e:
        tune_state[case_id]["status"] = "failed"
        tlog(f"Error during Optuna tuning: {str(e)}")

@app.post("/api/cases/{case_id}/tune")
def start_tuning(case_id: str, req: TuneRequest):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    if tune_state.get(case_id, {}).get("status") == "running":
        return {"status": "already_running"}

    t = threading.Thread(target=run_tune_thread, args=(case_id, req.model, req.n_trials, meta, case_dir))
    t.daemon = True
    t.start()
    return {"status": "started", "model": req.model, "n_trials": req.n_trials}

@app.get("/api/cases/{case_id}/tune-status")
def get_tune_status(case_id: str):
    return tune_state.get(case_id, {"status": "idle", "trials": [], "logs": []})

@app.post("/api/cases/{case_id}/tune-apply")
def apply_tuned_params(case_id: str):
    t_info = tune_state.get(case_id)
    if not t_info or not t_info.get("best_params"):
        raise HTTPException(status_code=400, detail="No tuned parameters available to apply.")

    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    with open(case_json, "r") as f:
        meta = json.load(f)

    if "model_params" not in meta:
        meta["model_params"] = {}

    m_name = t_info["model"]
    meta["model_params"][m_name] = t_info["best_params"]

    with open(case_json, "w") as f:
        json.dump(meta, f, indent=4)

    return {"status": "applied", "model": m_name, "best_params": t_info["best_params"]}

@app.post("/api/cases/{case_id}/ensemble/hill-climb")
def run_hill_climb(case_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    with open(case_json, "r") as f:
        meta = json.load(f)

    out_dir = os.path.join(ROOT_DIR, meta.get("output_dir", f"experiments/{case_id}_run"))
    all_models = ["lgbm", "xgb", "catboost"]
    artifacts_dir = os.path.join(out_dir, "artifacts")
    available_cached = [m for m in all_models if os.path.exists(os.path.join(artifacts_dir, f"oof_{m}.npy"))]

    if len(available_cached) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 cached models to optimize ensemble.")

    oofs, tests = TabularPipeline.load_cached_predictions(out_dir, available_cached)
    
    y_true_path = os.path.join(artifacts_dir, "y_true.npy")
    if os.path.exists(y_true_path):
        y_true = np.load(y_true_path)
    else:
        train_df = pd.read_csv(os.path.join(ROOT_DIR, meta["train_path"]), usecols=[meta["target_col"]])
        y_true = train_df[meta["target_col"]].values
        if y_true.dtype == bool or not np.issubdtype(y_true.dtype, np.number):
            y_true = pd.Series(y_true).astype("category").cat.codes.values

    opt = EnsembleOptimizer(task_type=meta["task_type"], eval_metric=meta["eval_metric"])
    hill_weights, hill_oof, hill_score = opt.hill_climbing(y_true, oofs)

    single_scores = {m: MetricEvaluator.calculate(y_true, oofs[m], meta["eval_metric"], meta["task_type"]) for m in available_cached}
    higher_better = MetricEvaluator.is_higher_better(meta["eval_metric"])
    best_single = max(single_scores.values()) if higher_better else min(single_scores.values())

    return {
        "score": hill_score,
        "weights": hill_weights,
        "single_best": best_single,
        "gain": hill_score - best_single,
        "single_scores": single_scores
    }

@app.post("/api/cases/{case_id}/ensemble/blend")
def run_custom_blend(case_id: str, req: BlendRequest):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    with open(case_json, "r") as f:
        meta = json.load(f)

    out_dir = os.path.join(ROOT_DIR, meta.get("output_dir", f"experiments/{case_id}_run"))
    models = list(req.weights.keys())
    oofs, tests = TabularPipeline.load_cached_predictions(out_dir, models)

    artifacts_dir = os.path.join(out_dir, "artifacts")
    y_true_path = os.path.join(artifacts_dir, "y_true.npy")
    if os.path.exists(y_true_path):
        y_true = np.load(y_true_path)
    else:
        train_df = pd.read_csv(os.path.join(ROOT_DIR, meta["train_path"]), usecols=[meta["target_col"]])
        y_true = train_df[meta["target_col"]].values
        if y_true.dtype == bool or not np.issubdtype(y_true.dtype, np.number):
            y_true = pd.Series(y_true).astype("category").cat.codes.values

    total_w = sum(req.weights.values())
    if total_w <= 0:
        raise HTTPException(status_code=400, detail="Total weights must be > 0")

    norm_w = {k: v / total_w for k, v in req.weights.items()}
    blended_oof = sum(w * oofs[k] for k, w in norm_w.items())
    score = MetricEvaluator.calculate(y_true, blended_oof, meta["eval_metric"], meta["task_type"])

    if tests and meta.get("test_path") and meta.get("id_col"):
        blended_test = sum(w * tests[k] for k, w in norm_w.items())
        test_df = pd.read_csv(os.path.join(ROOT_DIR, meta["test_path"]), usecols=[meta["id_col"]])
        sub_df = pd.DataFrame({
            meta["id_col"]: test_df[meta["id_col"]],
            meta["target_col"]: blended_test
        })
        sub_df.to_csv(os.path.join(out_dir, "custom_blend_submission.csv"), index=False)

    return {
        "score": score,
        "normalized_weights": norm_w
    }

@app.get("/api/cases/{case_id}/download/submission")
def download_submission(case_id: str, custom: bool = False):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    with open(case_json, "r") as f:
        meta = json.load(f)

    out_dir = os.path.join(ROOT_DIR, meta.get("output_dir", f"experiments/{case_id}_run"))
    filename = "custom_blend_submission.csv" if custom else "submission.csv"
    sub_path = os.path.join(out_dir, filename)

    if not os.path.exists(sub_path):
        raise HTTPException(status_code=404, detail="Submission file not found. Run pipeline first.")

    return FileResponse(sub_path, media_type="text/csv", filename=f"{case_id}_{filename}")

@app.get("/api/cases/{case_id}/download/oof")
def download_oof(case_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    with open(case_json, "r") as f:
        meta = json.load(f)

    out_dir = os.path.join(ROOT_DIR, meta.get("output_dir", f"experiments/{case_id}_run"))
    oof_path = os.path.join(out_dir, "oof_champion.csv")

    if not os.path.exists(oof_path):
        raise HTTPException(status_code=404, detail="OOF predictions file not found. Run pipeline first.")

    return FileResponse(oof_path, media_type="text/csv", filename=f"{case_id}_oof_champion.csv")

@app.get("/api/cases/{case_id}/export-script")
def export_standalone_script(case_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    with open(case_json, "r") as f:
        meta = json.load(f)

    hook_file = os.path.join(case_dir, "feature_hook.py")
    hook_code = ""
    if os.path.exists(hook_file):
        with open(hook_file, "r") as hf:
            hook_code = hf.read()
    else:
        hook_code = """
class CustomFeatures(BaseFeatureHook):
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        return train_df, test_df
"""

    script = f'''"""
Standalone Competitive Tabular Script for: {meta.get('name', case_id)}
Auto-generated by Tabular Studio
"""
import os
import pandas as pd
import numpy as np
from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook

{hook_code}

def main():
    print("[1/3] Loading Datasets...")
    train_df = pd.read_csv("{meta['train_path']}")
    test_df = pd.read_csv("{meta['test_path']}") if "{meta.get('test_path')}" else None

    print("[2/3] Configuring Tabular Pipeline...")
    cfg = PipelineConfig(
        project_name="{meta.get('name', case_id)}",
        output_dir="{meta.get('output_dir', f'experiments/{case_id}_run')}",
        seed={meta.get('seed', 2026)},
        id_col="{meta.get('id_col')}",
        target_col="{meta.get('target_col')}",
        task_type="{meta.get('task_type', 'binary')}",
        eval_metric="{meta.get('eval_metric', 'roc_auc')}",
        cv_strategy="{meta.get('cv_strategy', 'stratified_kfold')}",
        n_splits={meta.get('n_splits', 5)},
        models={meta.get('models', ['lgbm', 'xgb', 'catboost'])},
        ensemble_methods=["mean", "rank", "opt_weights", "hill_climbing", "stacking"],
        run_adversarial_validation=True,
        save_oof_predictions=True,
        save_test_predictions=True
    )

    print("[3/3] Training Models & Ensembling...")
    pipeline = TabularPipeline(cfg, feature_hook=CustomFeatures())
    results = pipeline.run(train_df, test_df)

    print("\\n============================================================")
    print(f"CHAMPION: {{results['champion']['model']}} | SCORE: {{results['champion']['score']:.5f}}")
    print(f"Submission saved to: {{results['champion'].get('submission_path')}}")
    print("============================================================\\n")

if __name__ == "__main__":
    main()
'''
    return Response(content=script, media_type="text/plain")

# Mount static frontend build if dist folder exists
frontend_dist = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
