import os
import sys
import json
import glob
import re
import shutil
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

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    p = request.url.path
    if p.endswith(".html") or p == "/" or p.endswith(".js") or p.endswith(".css"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

CASES_DIR = os.path.join(ROOT_DIR, "cases")
os.makedirs(CASES_DIR, exist_ok=True)

# In-memory execution state, EDA cache, and tuning state
job_state = {}
eda_cache = {}
tune_state = {}
stop_events = {}

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
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    n_splits: int = 5
    seed: int = 2026
    models: List[str] = ["lgbm", "xgb", "catboost"]
    hook_code: Optional[str] = None

class TuneRequest(BaseModel):
    model: str = "lgbm"
    n_trials: int = 15

class RunPipelineRequest(BaseModel):
    models: Optional[List[str]] = None

class SaveHookRequest(BaseModel):
    hook_code: str

class BlendRequest(BaseModel):
    weights: Dict[str, float]

class InspectDatasetRequest(BaseModel):
    path: str

class UpdateCasePathsRequest(BaseModel):
    train_path: str
    test_path: Optional[str] = None

class UpdateRunLBRequest(BaseModel):
    lb_score: Optional[float] = None
    notes: Optional[str] = None

class UpdateParamsRequest(BaseModel):
    architecture_preset: str = "standard"  # "shallow", "standard", "custom"
    model_params: Optional[Dict[str, Any]] = None

class MergeDatasetsRequest(BaseModel):
    case_id: str
    base_path: str
    join_path: str
    join_keys: List[str]
    how: str = "inner"  # "left", "inner", "right", "outer"
    aggregation: str = "none"  # "none", "last", "first"
    group_by_keys: Optional[List[str]] = None
    filter_col: Optional[str] = None
    filter_val: Optional[str] = None
    output_filename: str = "train_merged.csv"
    set_as_target: Optional[str] = None  # "train", "test", None

class ConcatPartitionsRequest(BaseModel):
    case_id: str
    pattern: str
    max_files: Optional[int] = None
    output_filename: str = "concatenated.csv"
    set_as_target: Optional[str] = None

class PreviewMergeRequest(BaseModel):
    base_path: str
    join_path: str
    join_keys: List[str]
    how: str = "inner"
    aggregation: str = "none"
    group_by_keys: Optional[List[str]] = None
    filter_col: Optional[str] = None
    filter_val: Optional[str] = None

ARCH_PRESETS = {
    "shallow": {
        "lgbm": {
            "num_leaves": 4,
            "max_depth": 3,
            "learning_rate": 0.04,
            "n_estimators": 5000,
            "min_child_samples": 100,
            "colsample_bytree": 0.5,
            "subsample": 0.8
        },
        "xgb": {
            "max_depth": 2,
            "learning_rate": 0.04,
            "n_estimators": 5000,
            "min_child_weight": 20,
            "colsample_bytree": 0.5,
            "subsample": 0.8
        },
        "catboost": {
            "depth": 3,
            "learning_rate": 0.05,
            "iterations": 4000,
            "l2_leaf_reg": 5.0
        }
    },
    "standard": {
        "lgbm": {
            "num_leaves": 45,
            "learning_rate": 0.03,
            "n_estimators": 2000,
            "colsample_bytree": 0.75,
            "subsample": 0.8
        },
        "xgb": {
            "max_depth": 6,
            "learning_rate": 0.03,
            "n_estimators": 2000,
            "colsample_bytree": 0.75,
            "subsample": 0.8
        },
        "catboost": {
            "depth": 6,
            "learning_rate": 0.03,
            "iterations": 2000,
            "l2_leaf_reg": 5.0
        }
    }
}

def resolve_dataset_path(raw_path: str, case_id: Optional[str] = None) -> Optional[str]:
    if not raw_path:
        return None

    clean_p = str(raw_path).strip().strip('"').strip("'")
    if not clean_p:
        return None

    # Candidate 1: Exact path (absolute or relative to CWD)
    if os.path.exists(clean_p):
        return os.path.abspath(clean_p)

    # Candidate 2: Relative to ROOT_DIR
    p2 = os.path.join(ROOT_DIR, clean_p)
    if os.path.exists(p2):
        return os.path.abspath(p2)

    # Candidate 3: Normalized slashes
    norm_p = clean_p.replace("\\", os.sep).replace("/", os.sep)
    if os.path.exists(norm_p):
        return os.path.abspath(norm_p)
    p3 = os.path.join(ROOT_DIR, norm_p)
    if os.path.exists(p3):
        return os.path.abspath(p3)

    # Candidate 4: Right-to-left sub-path matching (handles absolute paths copied from another machine)
    # E.g. "D:/POLBAN/Semester 7/gemastik-final/playground-series-s6e11/train.csv" -> matches "playground-series-s6e11/train.csv"
    parts = [seg for seg in re.split(r'[\\/]', clean_p) if seg and not seg.endswith(':')]
    if len(parts) >= 2:
        for depth in range(min(len(parts), 4), 1, -1):
            sub_rel = os.path.join(*parts[-depth:])
            p_sub = os.path.join(ROOT_DIR, sub_rel)
            if os.path.exists(p_sub):
                return os.path.abspath(p_sub)

    # Candidate 5: Direct folder or workspace matching
    filename = os.path.basename(clean_p)
    if filename:
        p_root = os.path.join(ROOT_DIR, filename)
        if os.path.exists(p_root):
            return os.path.abspath(p_root)

        if case_id:
            p_case = os.path.join(CASES_DIR, case_id, filename)
            if os.path.exists(p_case):
                return os.path.abspath(p_case)
            p_case_root = os.path.join(ROOT_DIR, case_id, filename)
            if os.path.exists(p_case_root):
                return os.path.abspath(p_case_root)

        # Check immediate parent directory match
        parent_dir = os.path.basename(os.path.dirname(clean_p))
        if parent_dir:
            p_parent = os.path.join(ROOT_DIR, parent_dir, filename)
            if os.path.exists(p_parent):
                return os.path.abspath(p_parent)

        # Recursive search in workspace (filter out system/build folders)
        found_matches = []
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"venv", ".venv", "node_modules", "dist", "experiments", "experiments_backup_archive", "catboost_info"}]
            if filename in files:
                found_matches.append(os.path.abspath(os.path.join(root, filename)))

        if found_matches:
            if parent_dir:
                for match in found_matches:
                    if parent_dir.lower() in match.lower():
                        return match
            if case_id:
                for match in found_matches:
                    if case_id.lower() in match.lower():
                        return match
            return found_matches[0]

    return None

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
    has_hook = False
    if os.path.exists(hook_file):
        has_hook = True
        with open(hook_file, "r") as hf:
            hook_code = hf.read()

    # Cached models for ensembling
    artifacts_dir = os.path.join(full_out_dir, "artifacts")
    all_models = ["lgbm", "xgb", "catboost"]
    available_cached = [m for m in all_models if os.path.exists(os.path.join(artifacts_dir, f"oof_{m}.npy"))]

    # Path resolution status
    resolved_train = resolve_dataset_path(meta.get("train_path", ""), case_id=case_id)
    resolved_test = resolve_dataset_path(meta.get("test_path", ""), case_id=case_id) if meta.get("test_path") else None
    train_exists = bool(resolved_train and os.path.exists(resolved_train))
    test_exists = bool(resolved_test and os.path.exists(resolved_test)) if resolved_test else True

    # History
    history_file = os.path.join(case_dir, "history.json")
    history_runs = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as hf:
                history_runs = json.load(hf).get("runs", [])
        except Exception:
            pass

    return {
        "meta": meta,
        "summary": summary_data,
        "feature_importances": feature_importances,
        "hook_code": hook_code,
        "has_hook": has_hook,
        "train_exists": train_exists,
        "test_exists": test_exists,
        "resolved_train_path": resolved_train,
        "resolved_test_path": resolved_test,
        "available_cached": available_cached,
        "history": history_runs,
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

    train_path = resolve_dataset_path(meta.get("train_path", ""), case_id=case_id)
    if not train_path or not os.path.exists(train_path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {meta.get('train_path')}")

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
    if not null_cols:
        null_cols_note = "Zero missing values"
    elif len(null_cols) <= 3:
        null_cols_note = f"in {', '.join(null_cols)}"
    else:
        null_cols_note = f"Across {len(null_cols)} columns (e.g. {', '.join(null_cols[:2])}, +{len(null_cols) - 2} more)"

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

@app.post("/api/cases/{case_id}/paths")
def update_case_paths(case_id: str, req: UpdateCasePathsRequest):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    meta["train_path"] = req.train_path
    if req.test_path is not None:
        meta["test_path"] = req.test_path

    with open(case_json, "w") as f:
        json.dump(meta, f, indent=4)

    # Invalidate EDA cache so next read recalculates
    if case_id in eda_cache:
        del eda_cache[case_id]

    resolved_train = resolve_dataset_path(req.train_path, case_id=case_id)
    resolved_test = resolve_dataset_path(req.test_path, case_id=case_id) if req.test_path else None
    train_exists = bool(resolved_train and os.path.exists(resolved_train))
    test_exists = bool(resolved_test and os.path.exists(resolved_test)) if resolved_test else True

    return {
        "status": "updated",
        "meta": meta,
        "resolved_train_path": resolved_train,
        "resolved_test_path": resolved_test,
        "train_exists": train_exists,
        "test_exists": test_exists
    }

@app.get("/api/scan-csvs")
def scan_workspace_csvs():
    """Scans the workspace recursively for CSV datasets."""
    csv_list = []
    ignored_dirs = {".venv", "venv", "node_modules", "experiments", "catboost_info", ".git", "experiments_backup_archive", "dist"}
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
        for f in files:
            if f.endswith(".csv"):
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, ROOT_DIR).replace("\\", "/")
                size_mb = round(os.path.getsize(full_p) / (1024 * 1024), 2)
                csv_list.append({
                    "path": rel_p,
                    "filename": f,
                    "size_mb": size_mb
                })
    csv_list.sort(key=lambda x: x["path"])
    return {"files": csv_list}

@app.get("/api/browse-file")
def browse_file(title: str = "Select CSV Dataset"):
    """Opens the native Windows Explorer File Dialog to choose a CSV dataset."""
    import subprocess
    ps_cmd = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Filter = "CSV Files (*.csv)|*.csv|All Files (*.*)|*.*"
$dialog.Title = "{title}"
$dialog.InitialDirectory = [System.IO.Path]::GetFullPath(".")
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
    Write-Output $dialog.FileName
}}
"""
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True, timeout=120)
        selected_file = proc.stdout.strip()
        if selected_file:
            try:
                rel = os.path.relpath(selected_file, ROOT_DIR)
                if not rel.startswith(".."):
                    return {"path": rel.replace("\\", "/"), "full_path": selected_file.replace("\\", "/")}
            except Exception:
                pass
            return {"path": selected_file.replace("\\", "/"), "full_path": selected_file.replace("\\", "/")}
    except Exception as e:
        print(f"Error in browse_file: {e}")
    return {"path": ""}

@app.post("/api/inspect-dataset")
def inspect_dataset(req: InspectDatasetRequest):
    file_path = resolve_dataset_path(req.path)
    if not file_path or not os.path.exists(file_path):
        return {"exists": False, "error": f"File does not exist: {req.path}"}
    
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

@app.get("/api/tools/inspect-columns")
def inspect_dataset_columns(path: str):
    file_p = resolve_dataset_path(path)
    if not file_p or not os.path.exists(file_p):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        sample = pd.read_csv(file_p, nrows=20)
        return {
            "columns": list(sample.columns),
            "sample": sample.head(3).fillna("NaN").to_dict(orient="records"),
            "total_sample_cols": len(sample.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _prepare_join_df(join_df: pd.DataFrame, req_filter_col, req_filter_val, req_aggregation, group_keys):
    if req_filter_col and req_filter_col in join_df.columns and req_filter_val is not None:
        val_str = str(req_filter_val).strip()
        if val_str.lower() == "true":
            join_df = join_df[join_df[req_filter_col] == True]
        elif val_str.lower() == "false":
            join_df = join_df[join_df[req_filter_col] == False]
        else:
            join_df = join_df[join_df[req_filter_col].astype(str) == val_str]

    if req_aggregation == "last" and group_keys:
        valid_groups = [k for k in group_keys if k in join_df.columns]
        if valid_groups:
            sort_c = [c for c in ["frame_id", "timestamp", "time", "date", "id"] if c in join_df.columns]
            if sort_c:
                join_df = join_df.sort_values(sort_c[0])
            join_df = join_df.groupby(valid_groups).last().reset_index()
    elif req_aggregation == "first" and group_keys:
        valid_groups = [k for k in group_keys if k in join_df.columns]
        if valid_groups:
            sort_c = [c for c in ["frame_id", "timestamp", "time", "date", "id"] if c in join_df.columns]
            if sort_c:
                join_df = join_df.sort_values(sort_c[0])
            join_df = join_df.groupby(valid_groups).first().reset_index()
    return join_df

@app.post("/api/tools/preview-merge")
def preview_merge_datasets(req: PreviewMergeRequest):
    base_p = resolve_dataset_path(req.base_path)
    join_p = resolve_dataset_path(req.join_path)
    if not base_p or not os.path.exists(base_p):
        raise HTTPException(status_code=404, detail=f"Base dataset not found: {req.base_path}")
    if not join_p or not os.path.exists(join_p):
        raise HTTPException(status_code=404, detail=f"Join dataset not found: {req.join_path}")

    try:
        base_df = pd.read_csv(base_p, nrows=200)
        join_df = pd.read_csv(join_p, nrows=1000)

        group_keys = req.group_by_keys or req.join_keys
        join_df = _prepare_join_df(join_df, req.filter_col, req.filter_val, req.aggregation, group_keys)

        merged = base_df.merge(join_df, on=req.join_keys, how=req.how, suffixes=("_base", "_join"))
        return {
            "status": "ok",
            "columns": list(merged.columns),
            "preview_rows": merged.head(5).fillna("NaN").to_dict(orient="records"),
            "preview_count": len(merged)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/tools/merge-datasets")
def execute_merge_datasets(req: MergeDatasetsRequest):
    base_p = resolve_dataset_path(req.base_path)
    join_p = resolve_dataset_path(req.join_path)
    if not base_p or not os.path.exists(base_p):
        raise HTTPException(status_code=404, detail=f"Base dataset not found: {req.base_path}")
    if not join_p or not os.path.exists(join_p):
        raise HTTPException(status_code=404, detail=f"Join dataset not found: {req.join_path}")

    case_dir = os.path.join(CASES_DIR, req.case_id)
    if not os.path.exists(case_dir):
        raise HTTPException(status_code=404, detail=f"Case not found: {req.case_id}")

    try:
        base_df = pd.read_csv(base_p)
        join_df = pd.read_csv(join_p)

        group_keys = req.group_by_keys or req.join_keys
        join_df = _prepare_join_df(join_df, req.filter_col, req.filter_val, req.aggregation, group_keys)

        merged = base_df.merge(join_df, on=req.join_keys, how=req.how, suffixes=("_base", "_join"))

        out_fname = req.output_filename.strip() if req.output_filename.strip() else "train_merged.csv"
        if not out_fname.endswith(".csv"):
            out_fname += ".csv"
        out_p = os.path.join(case_dir, out_fname)
        merged.to_csv(out_p, index=False)

        # Update case.json if requested
        rel_saved_path = os.path.relpath(out_p, ROOT_DIR).replace("\\", "/")
        case_json_p = os.path.join(case_dir, "case.json")
        if os.path.exists(case_json_p) and req.set_as_target in ["train", "test"]:
            with open(case_json_p, "r") as f:
                meta = json.load(f)
            if req.set_as_target == "train":
                meta["train_path"] = rel_saved_path
            elif req.set_as_target == "test":
                meta["test_path"] = rel_saved_path
            with open(case_json_p, "w") as f:
                json.dump(meta, f, indent=4)

        return {
            "status": "success",
            "saved_path": rel_saved_path,
            "total_rows": len(merged),
            "total_columns": len(merged.columns),
            "columns": list(merged.columns),
            "applied_target": req.set_as_target
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/tools/concat-partitions")
def execute_concat_partitions(req: ConcatPartitionsRequest):
    case_dir = os.path.join(CASES_DIR, req.case_id)
    if not os.path.exists(case_dir):
        raise HTTPException(status_code=404, detail=f"Case not found: {req.case_id}")

    pattern = req.pattern.strip().strip('"').strip("'")
    full_pattern = os.path.join(ROOT_DIR, pattern) if not os.path.isabs(pattern) else pattern
    files = sorted(glob.glob(full_pattern))
    if not files:
        raise HTTPException(status_code=404, detail=f"No files matched pattern: {pattern}")

    if req.max_files and req.max_files > 0:
        files = files[:req.max_files]

    try:
        dfs = []
        for fp in files:
            dfs.append(pd.read_csv(fp))
        concatenated = pd.concat(dfs, ignore_index=True)

        out_fname = req.output_filename.strip() if req.output_filename.strip() else "concatenated.csv"
        if not out_fname.endswith(".csv"):
            out_fname += ".csv"
        out_p = os.path.join(case_dir, out_fname)
        concatenated.to_csv(out_p, index=False)

        rel_saved_path = os.path.relpath(out_p, ROOT_DIR).replace("\\", "/")
        case_json_p = os.path.join(case_dir, "case.json")
        if os.path.exists(case_json_p) and req.set_as_target in ["train", "test"]:
            with open(case_json_p, "r") as f:
                meta = json.load(f)
            if req.set_as_target == "train":
                meta["train_path"] = rel_saved_path
            elif req.set_as_target == "test":
                meta["test_path"] = rel_saved_path
            with open(case_json_p, "w") as f:
                json.dump(meta, f, indent=4)

        return {
            "status": "success",
            "files_matched": len(files),
            "saved_path": rel_saved_path,
            "total_rows": len(concatenated),
            "total_columns": len(concatenated.columns),
            "applied_target": req.set_as_target
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def record_run_history(case_id: str, meta: dict, results: dict, hook_present: bool, case_dir: str):
    history_file = os.path.join(case_dir, "history.json")
    history_data = {"case_id": case_id, "runs": []}
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)
        except Exception:
            pass

    existing_runs = history_data.get("runs", [])
    run_num = len(existing_runs) + 1
    run_id = f"run_{run_num:03d}"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    out_dir = os.path.join(ROOT_DIR, meta.get("output_dir", f"experiments/{case_id}_run"))
    fi_file = os.path.join(out_dir, "feature_importances.csv")
    feat_count = 0
    if os.path.exists(fi_file):
        try:
            fi_df = pd.read_csv(fi_file)
            feat_count = int(fi_df["feature"].nunique())
        except Exception:
            pass

    # Archive submission
    sub_source = os.path.join(out_dir, "submission.csv")
    sub_archive_dir = os.path.join(out_dir, "submissions")
    os.makedirs(sub_archive_dir, exist_ok=True)
    archive_name = f"submission_{run_id}.csv"
    sub_dest = os.path.join(sub_archive_dir, archive_name)
    has_sub = False
    if os.path.exists(sub_source):
        shutil.copyfile(sub_source, sub_dest)
        has_sub = True

    champ_score = float(results.get("champion_score", 0.0))
    metric = results.get("metric", meta.get("eval_metric", "roc_auc"))
    higher_better = MetricEvaluator.is_higher_better(metric)

    # Baseline comparison (run 1 is baseline reference)
    if existing_runs:
        baseline_score = existing_runs[0].get("cv_score", champ_score)
        delta_vs_baseline = round(champ_score - baseline_score, 5) if higher_better else round(baseline_score - champ_score, 5)
    else:
        delta_vs_baseline = 0.0

    # Tag & name
    is_ensemble = len(results.get("all_scores", {})) > len(meta.get("models", []))
    if is_ensemble:
        tag = "ensemble"
        name = f"Ensemble ({results.get('champion_model', 'Blend')})"
    elif hook_present:
        tag = "feature_engineering"
        name = f"Feature Engineering ({feat_count} features)"
    else:
        tag = "baseline"
        name = f"Pure Baseline ({feat_count} features)"

    run_entry = {
        "run_id": run_id,
        "run_number": run_num,
        "timestamp": timestamp,
        "name": name,
        "tag": tag,
        "has_hook": hook_present,
        "features_count": feat_count,
        "models": meta.get("models", []),
        "cv_score": round(champ_score, 5),
        "all_scores": {k: round(v, 5) for k, v in results.get("all_scores", {}).items()},
        "metric": metric,
        "cv_delta": delta_vs_baseline,
        "lb_score": None,
        "champion_model": results.get("champion_model", ""),
        "submission_filename": archive_name if has_sub else None,
        "notes": ""
    }

    existing_runs.append(run_entry)
    history_data["runs"] = existing_runs

    with open(history_file, "w") as f:
        json.dump(history_data, f, indent=4)

def run_pipeline_thread(case_id: str, meta: dict, case_dir: str):
    stop_event = threading.Event()
    stop_events[case_id] = stop_event

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
        train_path = resolve_dataset_path(meta.get("train_path", ""), case_id=case_id)
        test_path = resolve_dataset_path(meta.get("test_path", ""), case_id=case_id) if meta.get("test_path") else None
        
        if not train_path or not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset not found: {meta.get('train_path')}")

        log(f"Loading dataset from {train_path}...")
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

        hook = load_hook_class(case_dir)
        if hook:
            log("Loaded custom feature engineering hook.")

        auto_fe_preset = meta.get("auto_fe_preset", "off" if hook else "conservative")

        cfg = PipelineConfig(
            project_name=meta.get("name", case_id),
            output_dir=meta.get("output_dir", f"experiments/{case_id}_run"),
            seed=meta.get("seed", 2026),
            id_col=meta.get("id_col") or None,
            target_col=meta.get("target_col"),
            group_col=meta.get("group_col") or None,
            time_col=meta.get("time_col") or None,
            task_type=meta.get("task_type", "binary"),
            eval_metric=meta.get("eval_metric", "roc_auc"),
            cv_strategy=meta.get("cv_strategy", "stratified_kfold"),
            n_splits=meta.get("n_splits", 5),
            models=meta.get("models", ["lgbm", "xgb", "catboost"]),
            model_params=meta.get("model_params", {}),
            auto_fe_preset=auto_fe_preset,
            run_generic_features=(auto_fe_preset != "off"),
            ensemble_methods=["mean", "rank", "opt_weights", "hill_climbing", "stacking"] if len(meta.get("models", [])) > 1 else [],
            run_adversarial_validation=True,
            save_oof_predictions=True,
            save_test_predictions=True
        )

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

        pipeline = TabularPipeline(cfg, feature_hook=hook, progress_callback=on_progress, is_aborted=stop_event.is_set)
        results = pipeline.run(train_df, test_df)

        job_state[case_id]["progress"] = 100
        job_state[case_id]["status"] = "completed"
        job_state[case_id]["current_model"] = "Completed"
        job_state[case_id]["current_fold"] = "Done"
        champ_name = results.get('champion_model', 'N/A')
        champ_score = results.get('champion_score', 0.0)
        log(f"All stages finished! Champion: {champ_name} (Score: {champ_score:.5f})")

        # Record run in experiment history ledger
        try:
            record_run_history(case_id, meta, results, hook_present=bool(hook), case_dir=case_dir)
        except Exception as he:
            log(f"Warning: Failed to log run history: {he}")

    except InterruptedError:
        job_state[case_id]["status"] = "stopped"
        job_state[case_id]["current_model"] = "Stopped"
        job_state[case_id]["current_fold"] = "Cancelled"
        log("Pipeline execution successfully halted by user.")
    except Exception as e:
        job_state[case_id]["status"] = "error"
        log(f"Error during execution: {str(e)}")

@app.post("/api/cases/{case_id}/stop")
def stop_case_pipeline(case_id: str):
    if case_id in stop_events:
        stop_events[case_id].set()
    if case_id in job_state:
        job_state[case_id]["status"] = "stopped"
        job_state[case_id]["current_model"] = "Stopped"
        job_state[case_id]["current_fold"] = "Cancelled"
        job_state[case_id]["logs"].append(f"[{time.strftime('%H:%M:%S')}] Stop signal sent by user. Halting...")
    return {"status": "stopped", "case_id": case_id}

@app.post("/api/cases/{case_id}/run")
def run_case_pipeline(case_id: str, req: Optional[RunPipelineRequest] = None):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    if req and req.models and len(req.models) > 0:
        meta["models"] = req.models

    if job_state.get(case_id, {}).get("status") == "running":
        return {"status": "already_running"}

    thread = threading.Thread(target=run_pipeline_thread, args=(case_id, meta, case_dir))
    thread.daemon = True
    thread.start()

    return {"status": "started", "case_id": case_id, "models": meta["models"]}

@app.post("/api/cases/{case_id}/hook")
def save_case_hook(case_id: str, req: SaveHookRequest):
    case_dir = os.path.join(CASES_DIR, case_id)
    if not os.path.exists(case_dir):
        raise HTTPException(status_code=404, detail="Case not found")
    hook_file = os.path.join(case_dir, "feature_hook.py")
    with open(hook_file, "w", encoding="utf-8") as f:
        f.write(req.hook_code)
    return {"status": "saved", "path": hook_file}

@app.post("/api/cases/{case_id}/hook/test")
def test_case_hook(case_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    hook_file = os.path.join(case_dir, "feature_hook.py")
    if not os.path.exists(hook_file):
        return {"status": "none", "message": "No feature_hook.py found"}

    try:
        hook = load_hook_class(case_dir)
        if not hook:
            return {"status": "error", "message": "Failed to load BaseFeatureHook subclass in feature_hook.py"}

        train_path = resolve_dataset_path(meta.get("train_path", ""), case_id=case_id)
        if not train_path or not os.path.exists(train_path):
            return {"status": "error", "message": f"Training dataset not found: {meta.get('train_path')}"}
        sample_df = pd.read_csv(train_path, nrows=1000)
        orig_cols = list(sample_df.columns)
        orig_count = len(orig_cols)

        t0 = time.time()
        res_train, _ = hook.transform(sample_df.copy(), None)
        duration_ms = round((time.time() - t0) * 1000, 1)

        new_cols = list(res_train.columns)
        added_cols = [c for c in new_cols if c not in orig_cols]
        new_nulls = int(res_train[added_cols].isnull().sum().sum()) if added_cols else 0

        return {
            "status": "valid",
            "original_cols_count": orig_count,
            "transformed_cols_count": len(new_cols),
            "added_features_count": len(added_cols),
            "added_features": added_cols,
            "new_nulls_created": new_nulls,
            "duration_ms": duration_ms,
            "message": f"Successfully validated: {len(added_cols)} new features generated in {duration_ms}ms"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/cases/{case_id}/status")
def get_case_status(case_id: str):
    return job_state.get(case_id, {"status": "idle", "progress": 0, "logs": []})

def run_tune_thread(case_id: str, model_name: str, n_trials: int, meta: dict, case_dir: str):
    models_to_tune = ["lgbm", "xgb", "catboost"] if model_name.lower() == "all" else [model_name.lower()]
    total_trials = len(models_to_tune) * n_trials
    tune_state[case_id] = {
        "status": "running",
        "model": model_name,
        "current_model": models_to_tune[0].upper(),
        "current_trial": 0,
        "total_trials": total_trials,
        "trials_per_model": n_trials,
        "best_score": None,
        "best_params": None,
        "all_best_params": {},
        "all_best_scores": {},
        "trials": [],
        "logs": [f"[{time.strftime('%H:%M:%S')}] Initializing Optuna study ({'All 3 Models (Sequential)' if model_name.lower() == 'all' else model_name.upper()})..."]
    }
    def tlog(msg):
        tune_state[case_id]["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    try:
        train_path = resolve_dataset_path(meta.get("train_path", ""), case_id=case_id)
        if not train_path or not os.path.exists(train_path):
            raise FileNotFoundError(f"Training dataset not found: {meta.get('train_path')}")
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

        tlog(f"Preparing Optuna search space on {len(X):,} rows x {len(X.columns)} features ({total_trials} total trials scheduled)...")

        higher_is_better = MetricEvaluator.is_higher_better(meta["eval_metric"])
        overall_best_score = -float("inf") if higher_is_better else float("inf")
        all_best_params = {}
        all_best_scores = {}
        trial_offset = 0

        tuner = HyperparameterTuner(
            task_type=meta["task_type"],
            eval_metric=meta["eval_metric"],
            seed=meta.get("seed", 2026),
            progress_callback=None
        )

        for m in models_to_tune:
            tune_state[case_id]["current_model"] = m.upper()
            tlog(f"Starting {n_trials} Bayesian Optimization trials for {m.upper()}...")

            def make_callback(curr_m, offset):
                def on_tune_progress(pinfo: dict):
                    trial_num = pinfo["trial"]
                    overall_num = offset + trial_num
                    sc = pinfo["score"]
                    bsc = pinfo["best_score"]
                    is_best = pinfo["is_new_best"]
                    tune_state[case_id]["current_trial"] = overall_num
                    tune_state[case_id]["current_model"] = curr_m.upper()
                    
                    curr_bs = tune_state[case_id]["best_score"]
                    if curr_bs is None:
                        tune_state[case_id]["best_score"] = sc
                    elif higher_is_better and sc > curr_bs:
                        tune_state[case_id]["best_score"] = sc
                    elif not higher_is_better and sc < curr_bs:
                        tune_state[case_id]["best_score"] = sc

                    tune_state[case_id]["trials"].append({
                        "trial": overall_num,
                        "model": curr_m.upper(),
                        "score": sc,
                        "best_score": bsc,
                        "is_new_best": is_best,
                        "params": pinfo["params"]
                    })
                    marker = " [NEW BEST]" if is_best else ""
                    tlog(f"[{curr_m.upper()}] Trial {trial_num}/{n_trials} -> Score: {sc:.5f} (Best: {bsc:.5f}){marker}")
                return on_tune_progress

            tuner.progress_callback = make_callback(m, trial_offset)
            res = tuner.tune_model(
                model_name=m,
                X=X,
                y=y,
                n_trials=n_trials,
                sample_size=60000,
            )
            all_best_params[m] = res["best_params"]
            all_best_scores[m] = res["best_score"]
            tlog(f"Completed {m.upper()} study! Best {meta['eval_metric'].upper()}: {res['best_score']:.5f}")

            if higher_is_better:
                if res["best_score"] > overall_best_score:
                    overall_best_score = res["best_score"]
            else:
                if res["best_score"] < overall_best_score:
                    overall_best_score = res["best_score"]

            trial_offset += n_trials

        tune_state[case_id]["status"] = "completed"
        tune_state[case_id]["best_score"] = overall_best_score
        tune_state[case_id]["best_params"] = all_best_params if model_name.lower() == "all" else all_best_params[models_to_tune[0]]
        tune_state[case_id]["all_best_params"] = all_best_params
        tune_state[case_id]["all_best_scores"] = all_best_scores
        tlog(f"Optuna Optimization Finished! Highest Single Model Score: {overall_best_score:.5f}")

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
    if not t_info or (not t_info.get("best_params") and not t_info.get("all_best_params")):
        raise HTTPException(status_code=400, detail="No tuned parameters available to apply.")

    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    with open(case_json, "r") as f:
        meta = json.load(f)

    if "model_params" not in meta:
        meta["model_params"] = {}

    m_name = t_info.get("model", "").lower()
    if m_name == "all" and "all_best_params" in t_info:
        for m, params in t_info["all_best_params"].items():
            meta["model_params"][m] = params
    else:
        meta["model_params"][m_name] = t_info["best_params"]

    with open(case_json, "w") as f:
        json.dump(meta, f, indent=4)

    return {"status": "applied", "model": m_name, "model_params": meta["model_params"]}

@app.get("/api/cases/{case_id}/params")
def get_case_params(case_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")
    with open(case_json, "r") as f:
        meta = json.load(f)
    preset = meta.get("architecture_preset", "standard")
    params = meta.get("model_params", ARCH_PRESETS.get(preset, ARCH_PRESETS["standard"]))
    return {
        "architecture_preset": preset,
        "model_params": params,
        "presets_catalog": ARCH_PRESETS
    }

@app.post("/api/cases/{case_id}/params")
def update_case_params(case_id: str, req: UpdateParamsRequest):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")
    with open(case_json, "r") as f:
        meta = json.load(f)

    meta["architecture_preset"] = req.architecture_preset
    if req.architecture_preset in ARCH_PRESETS and (not req.model_params or req.model_params == {}):
        meta["model_params"] = ARCH_PRESETS[req.architecture_preset]
    elif req.model_params is not None:
        meta["model_params"] = req.model_params
    else:
        meta["model_params"] = ARCH_PRESETS.get(req.architecture_preset, {})

    with open(case_json, "w") as f:
        json.dump(meta, f, indent=4)

    return {
        "status": "updated",
        "architecture_preset": meta["architecture_preset"],
        "model_params": meta["model_params"]
    }

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
        train_path = resolve_dataset_path(meta.get("train_path", ""), case_id=case_id)
        if not train_path or not os.path.exists(train_path):
            raise HTTPException(status_code=404, detail="Training dataset not found")
        train_df = pd.read_csv(train_path, usecols=[meta["target_col"]])
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
        train_path = resolve_dataset_path(meta.get("train_path", ""), case_id=case_id)
        if not train_path or not os.path.exists(train_path):
            raise HTTPException(status_code=404, detail="Training dataset not found")
        train_df = pd.read_csv(train_path, usecols=[meta["target_col"]])
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
        test_path = resolve_dataset_path(meta.get("test_path", ""), case_id=case_id)
        if test_path and os.path.exists(test_path):
            blended_test = sum(w * tests[k] for k, w in norm_w.items())
            test_df = pd.read_csv(test_path, usecols=[meta["id_col"]])
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

@app.get("/api/cases/{case_id}/history")
def get_case_history_api(case_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    history_file = os.path.join(case_dir, "history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading history.json for {case_id}: {e}")

    # Fallback / Seed initial run if summary.json exists
    out_dir = os.path.join(ROOT_DIR, meta.get("output_dir", f"experiments/{case_id}_run"))
    summary_file = os.path.join(out_dir, "summary.json")
    runs = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as sf:
                smry = json.load(sf)
            fi_file = os.path.join(out_dir, "feature_importances.csv")
            feat_cnt = 0
            if os.path.exists(fi_file):
                fi_df = pd.read_csv(fi_file)
                feat_cnt = int(fi_df["feature"].nunique())

            hook_file = os.path.join(case_dir, "feature_hook.py")
            has_hook = os.path.exists(hook_file)

            sub_source = os.path.join(out_dir, "submission.csv")
            sub_archive_dir = os.path.join(out_dir, "submissions")
            os.makedirs(sub_archive_dir, exist_ok=True)
            archive_name = "submission_run_001.csv"
            sub_dest = os.path.join(sub_archive_dir, archive_name)
            if os.path.exists(sub_source) and not os.path.exists(sub_dest):
                shutil.copyfile(sub_source, sub_dest)

            runs.append({
                "run_id": "run_001",
                "run_number": 1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(summary_file))),
                "name": "Current Baseline / Champion",
                "tag": "feature_engineering" if has_hook else "baseline",
                "has_hook": has_hook,
                "features_count": feat_cnt,
                "models": meta.get("models", []),
                "cv_score": round(smry.get("champion_score", 0.0), 5),
                "all_scores": {k: round(v, 5) for k, v in smry.get("all_scores", {}).items()},
                "metric": smry.get("metric", meta.get("eval_metric", "roc_auc")),
                "cv_delta": 0.0,
                "lb_score": None,
                "champion_model": smry.get("champion_model", ""),
                "submission_filename": archive_name if os.path.exists(sub_dest) else None,
                "notes": "Initial benchmark run"
            })
            history_data = {"case_id": case_id, "runs": runs}
            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=4)
        except Exception as e:
            print(f"Error seeding history: {e}")

    return {"case_id": case_id, "runs": runs}

@app.post("/api/cases/{case_id}/history/{run_id}/lb")
def update_case_history_lb(case_id: str, run_id: str, req: UpdateRunLBRequest):
    case_dir = os.path.join(CASES_DIR, case_id)
    history_file = os.path.join(case_dir, "history.json")
    if not os.path.exists(history_file):
        raise HTTPException(status_code=404, detail="No history found for case")

    with open(history_file, "r") as f:
        history_data = json.load(f)

    updated = False
    for r in history_data.get("runs", []):
        if r.get("run_id") == run_id:
            if req.lb_score is not None:
                r["lb_score"] = req.lb_score
            if req.notes is not None:
                r["notes"] = req.notes
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found in history")

    with open(history_file, "w") as f:
        json.dump(history_data, f, indent=4)

    return {"status": "updated", "history": history_data}

@app.get("/api/cases/{case_id}/download/submission/run/{run_id}")
def download_submission_run(case_id: str, run_id: str):
    case_dir = os.path.join(CASES_DIR, case_id)
    case_json = os.path.join(case_dir, "case.json")
    if not os.path.exists(case_json):
        raise HTTPException(status_code=404, detail="Case not found")

    with open(case_json, "r") as f:
        meta = json.load(f)

    out_dir = os.path.join(ROOT_DIR, meta.get("output_dir", f"experiments/{case_id}_run"))
    sub_dir = os.path.join(out_dir, "submissions")
    run_file = os.path.join(sub_dir, f"submission_{run_id}.csv")

    if not os.path.exists(run_file):
        run_file = os.path.join(out_dir, "submission.csv")
        if not os.path.exists(run_file):
            raise HTTPException(status_code=404, detail=f"Submission file for {run_id} not found")

    return FileResponse(run_file, media_type="text/csv", filename=f"{case_id}_{run_id}_submission.csv")

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
        id_col={repr(meta.get('id_col') or None)},
        target_col="{meta.get('target_col')}",
        group_col={repr(meta.get('group_col') or None)},
        time_col={repr(meta.get('time_col') or None)},
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
    print(f"CHAMPION: {results.get('champion_model')} | SCORE: {results.get('champion_score', 0):.5f}")
    print(f"Submission saved to: {results.get('submission_path', 'submission.csv')}")
    print("============================================================\\n")

if __name__ == "__main__":
    main()
'''
    return Response(content=script, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename={case_id}_solution.py"})

@app.get("/api/cases/{case_id}/export-notebook")
def export_standalone_notebook(case_id: str):
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
        hook_code = """class CustomFeatures(BaseFeatureHook):
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        return train_df, test_df"""

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Machine Learning Solution: {meta.get('name', case_id)}\n",
                    "### Competitive Leak-Free Tabular Pipeline & Multi-Model Ensembling\n",
                    f"- **Task Type**: `{meta.get('task_type', 'binary')}`\n",
                    f"- **Target Column**: `{meta.get('target_col')}`\n",
                    f"- **Evaluation Metric**: `{meta.get('eval_metric', 'roc_auc').upper()}`\n",
                    f"- **Cross-Validation**: `{meta.get('n_splits', 5)}-Fold {meta.get('cv_strategy', 'stratified_kfold')}`\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Setup & Environment Dependencies\n",
                    "Import core tabular machine learning packages."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "import time\n",
                    "import json\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import lightgbm as lgb\n",
                    "import xgboost as xgb\n",
                    "import catboost as cb\n",
                    "import sklearn\n",
                    "from tabular_pipeline import PipelineConfig, TabularPipeline, BaseFeatureHook\n",
                    "\n",
                    "print(f\"Pandas     : {pd.__version__}\")\n",
                    "print(f\"LightGBM   : {lgb.__version__}\")\n",
                    "print(f\"XGBoost    : {xgb.__version__}\")\n",
                    "print(f\"CatBoost   : {cb.__version__}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dataset Ingestion & Integrity Audit\n",
                    "Load train and test datasets."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"train_path = '{meta['train_path']}'\n",
                    f"test_path = '{meta.get('test_path', '')}'\n",
                    "\n",
                    "train_df = pd.read_csv(train_path)\n",
                    "test_df = pd.read_csv(test_path) if test_path and os.path.exists(test_path) else None\n",
                    "\n",
                    "print(f\"Training Matrix : {train_df.shape[0]:,} rows x {train_df.shape[1]} columns\")\n",
                    "if test_df is not None:\n",
                    "    print(f\"Testing Matrix  : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns\")\n",
                    "train_df.head(5)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Domain Feature Engineering Hook\n",
                    "Custom domain transformations (contextual imputations, interaction terms, ratio features)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    hook_code
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Pipeline Execution & Leak-Free Cross-Validation\n",
                    "Execute 5-fold cross-validation on LightGBM, XGBoost, and CatBoost with automated ensembling."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"cfg = PipelineConfig(\n",
                    f"    project_name='{meta.get('name', case_id)}',\n",
                    f"    output_dir='experiments/{case_id}_notebook_run',\n",
                    f"    seed={meta.get('seed', 2026)},\n",
                    f"    id_col={repr(meta.get('id_col') or None)},\n",
                    f"    target_col='{meta.get('target_col')}',\n",
                    f"    group_col={repr(meta.get('group_col') or None)},\n",
                    f"    time_col={repr(meta.get('time_col') or None)},\n",
                    f"    task_type='{meta.get('task_type', 'binary')}',\n",
                    f"    eval_metric='{meta.get('eval_metric', 'roc_auc')}',\n",
                    f"    cv_strategy='{meta.get('cv_strategy', 'stratified_kfold')}',\n",
                    f"    n_splits={meta.get('n_splits', 5)},\n",
                    f"    models={json.dumps(meta.get('models', ['lgbm', 'xgb', 'catboost']))},\n",
                    f"    model_params={json.dumps(meta.get('model_params', {}))},\n",
                    f"    ensemble_methods=['mean', 'rank', 'opt_weights', 'hill_climbing', 'stacking'],\n",
                    f"    run_adversarial_validation=True,\n",
                    f"    save_oof_predictions=True,\n",
                    f"    save_test_predictions=True\n",
                    f")\n",
                    "\n",
                    "pipeline = TabularPipeline(cfg, feature_hook=CustomFeatures())\n",
                    "results = pipeline.run(train_df, test_df)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Validation Leaderboard & Champion Model Evaluation\n",
                    "Display Out-Of-Fold validation scores and gain across evaluated models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "scores = results.get('all_scores', {})\n",
                    "leaderboard = pd.DataFrame(list(scores.items()), columns=['Strategy', 'OOF Score']).sort_values('OOF Score', ascending=False).reset_index(drop=True)\n",
                    "leaderboard['Rank'] = np.arange(1, len(leaderboard) + 1)\n",
                    "print(f\"CHAMPION: {results.get('champion_model')} | SCORE: {results.get('champion_score', 0):.5f}\")\n",
                    "leaderboard[['Rank', 'Strategy', 'OOF Score']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Submission File Sanity Audit\n",
                    "Verify shape, missing values, and value ranges for competition submission."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sub_path = os.path.join(cfg.output_dir, 'submission.csv')\n",
                    "if os.path.exists(sub_path):\n",
                    "    sub = pd.read_csv(sub_path)\n",
                    "    print(f\"Submission Shape : {sub.shape}\")\n",
                    "    print(f\"Missing Values   : {sub.isna().sum().sum()}\")\n",
                    "    print(sub.head())\n",
                    "else:\n",
                    "    print('No test dataset provided. OOF models ready for inference.')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_json = json.dumps(nb, indent=2)
    return Response(
        content=nb_json,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f"attachment; filename={case_id}_solution.ipynb"}
    )

# Mount static frontend build if dist folder exists
frontend_dist = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
