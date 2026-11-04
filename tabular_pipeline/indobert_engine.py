import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

class IndoBERTEngine:
    """
    IndoBERT (indobenchmark/indobert-base-p1) Neural Representation Engine with
    Downstream GBDT Classifiers (LightGBM, CatBoost, XGBoost) and CUDA Acceleration.
    """
    DEFAULT_MODEL = "indobenchmark/indobert-base-p1"

    @staticmethod
    def get_gpu_status() -> Dict[str, Any]:
        """Check GPU availability, VRAM, and device information via PyTorch & system."""
        status = {
            "cuda_available": False,
            "device_name": "CPU Only",
            "device_count": 0,
            "current_device": "cpu",
            "total_memory_mb": 0.0,
            "free_memory_mb": 0.0,
            "cuda_version": None,
            "torch_version": None,
            "details": "PyTorch not yet loaded."
        }
        try:
            import torch
            status["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                status["cuda_available"] = True
                status["device_count"] = torch.cuda.device_count()
                status["current_device"] = f"cuda:{torch.cuda.current_device()}"
                status["device_name"] = torch.cuda.get_device_name(0)
                status["cuda_version"] = torch.version.cuda
                
                # Memory info
                free_b, total_b = torch.cuda.mem_get_info()
                status["free_memory_mb"] = round(free_b / (1024 * 1024), 1)
                status["total_memory_mb"] = round(total_b / (1024 * 1024), 1)
                status["details"] = f"{status['device_name']} (CUDA {status['cuda_version']}) - {status['free_memory_mb']} MB / {status['total_memory_mb']} MB VRAM Free"
            else:
                status["details"] = "No CUDA GPU detected; running on CPU."
        except Exception as e:
            status["details"] = f"Torch check error: {str(e)}"
        return status

    def __init__(self, model_name: str = DEFAULT_MODEL, use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def _init_model(self, log_callback: Optional[Callable[[str], None]] = None):
        """Lazy loader for IndoBERT model and tokenizer onto optimal device."""
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModel

        if self.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            if log_callback:
                log_callback(f"[Hardware] GPU Acceleration Active: {gpu_name} ({self.device})")
        else:
            self.device = "cpu"
            if log_callback:
                log_callback("[Hardware] Using CPU Execution.")

        if log_callback:
            log_callback(f"[IndoBERT] Loading tokenizer & weights for '{self.model_name}'...")
        
        start_t = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        dur = round(time.time() - start_t, 2)
        if log_callback:
            log_callback(f"[IndoBERT] Model successfully loaded on {self.device} in {dur}s.")

    def extract_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 256,
        pooling: str = "cls",
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Extract dense 768-dimensional semantic embeddings from IndoBERT.
        Supports 'cls' token or 'mean' pooling with attention mask.
        """
        import torch

        self._init_model(log_callback)
        clean_texts = [str(t).strip() if pd.notnull(t) and str(t).strip() else "[PAD]" for t in texts]
        total = len(clean_texts)
        all_embeddings = []

        if log_callback:
            log_callback(f"[Embedding] Extracting {total:,} IndoBERT representations (batch_size={batch_size}, max_len={max_length}, pooling={pooling})...")

        start_time = time.time()
        for idx in range(0, total, batch_size):
            batch_texts = clean_texts[idx : idx + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state  # shape: (B, seq_len, 768)

                if pooling == "mean":
                    # Mean pooling masking out padded tokens
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_emb = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    # CLS token representation (first token)
                    batch_emb = last_hidden[:, 0, :].cpu().numpy()

            all_embeddings.append(batch_emb)
            current_processed = min(idx + batch_size, total)
            if progress_callback:
                progress_callback(current_processed, total)
            if log_callback and (current_processed % (batch_size * 5) == 0 or current_processed == total):
                elapsed = time.time() - start_time
                speed = current_processed / (elapsed + 1e-5)
                log_callback(f"[Embedding] Processed {current_processed:,}/{total:,} texts ({speed:.1f} texts/sec)")

        # Clear GPU cache after embedding extraction
        if self.device == "cuda":
            torch.cuda.empty_cache()

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        total_time = round(time.time() - start_time, 2)
        if log_callback:
            log_callback(f"[Embedding] Done in {total_time}s! Output shape: {embeddings.shape}")
        return embeddings

    def release(self):
        """Free the transformer model and its CUDA memory before GBDT training."""
        self.model = None
        self.tokenizer = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _build_classifier(m_key: str, n_classes: int, is_binary: bool, seed: int, gpu_avail: bool = False):
        """Construct a fresh GBDT classifier with the studio's fixed hyperparameters."""
        if m_key == "lgbm":
            n_est = 180 if n_classes > 10 else 250
            params = {
                "n_estimators": n_est,
                "learning_rate": 0.07,
                "num_leaves": 25,
                "random_state": seed,
                "verbose": -1,
                "n_jobs": -1
            }
            if is_binary:
                return lgb.LGBMClassifier(**params, objective="binary")
            return lgb.LGBMClassifier(**params, objective="multiclass", num_class=n_classes)

        if m_key == "catboost":
            return CatBoostClassifier(
                iterations=250 if n_classes > 10 else 350,
                learning_rate=0.08,
                depth=5,
                random_seed=seed,
                verbose=0,
                task_type="GPU" if gpu_avail else "CPU"
            )

        if m_key == "xgb":
            params = {
                "n_estimators": 180 if n_classes > 10 else 250,
                "learning_rate": 0.07,
                "max_depth": 5,
                "tree_method": "hist",
                "device": "cuda" if gpu_avail else "cpu",
                "random_state": seed,
                "verbosity": 0
            }
            if is_binary:
                return xgb.XGBClassifier(**params, objective="binary:logistic", eval_metric="logloss")
            return xgb.XGBClassifier(**params, objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")

        raise ValueError(f"Unsupported classifier: {m_key}")

    @staticmethod
    def _fit_classifier(
        m_key: str,
        n_classes: int,
        is_binary: bool,
        seed: int,
        gpu_avail: bool,
        X,
        y,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """Fit a classifier, retrying on CPU when GPU VRAM runs out."""
        clf = IndoBERTEngine._build_classifier(m_key, n_classes, is_binary, seed, gpu_avail)
        try:
            clf.fit(X, y)
            return clf
        except Exception as e:
            message = str(e).lower()
            is_gpu_error = any(k in message for k in ["memory", "alloc", "cuda", "gpu"])
            if not (gpu_avail and is_gpu_error):
                raise
            if log_callback:
                log_callback(f"[GPU] {m_key.upper()} gagal di GPU ({type(e).__name__}). Fallback ke CPU...")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            clf = IndoBERTEngine._build_classifier(m_key, n_classes, is_binary, seed, False)
            clf.fit(X, y)
            return clf

    @staticmethod
    def fit_full_classifiers(
        X: np.ndarray,
        y_raw: np.ndarray,
        models_to_train: List[str] = ["lgbm", "catboost", "xgb"],
        seed: int = 2026,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Train final classifiers on the full dataset so the sandbox can run real inference."""
        unique_labels = sorted(list(pd.Series(y_raw).dropna().unique()))
        label2idx = {l: idx for idx, l in enumerate(unique_labels)}
        y = pd.Series(y_raw).map(label2idx).values
        n_classes = len(unique_labels)
        is_binary = n_classes == 2

        gpu_avail = False
        try:
            import torch
            gpu_avail = torch.cuda.is_available()
        except Exception:
            pass

        models = {}
        for m_name in models_to_train:
            m_key = m_name.lower().strip()
            if m_key not in ["lgbm", "catboost", "xgb"]:
                continue
            if log_callback:
                log_callback(f"[Final Model] Fitting {m_key.upper()} on all {len(y):,} training rows...")
            clf = IndoBERTEngine._fit_classifier(
                m_key, n_classes, is_binary, seed, gpu_avail, X, y, log_callback
            )
            models[m_key] = clf

        return {
            "models": models,
            "classes": [str(c) for c in unique_labels],
            "label2idx": {str(k): int(v) for k, v in label2idx.items()},
        }

    @staticmethod
    def train_downstream_classifiers(
        X_emb: np.ndarray,
        y_raw: np.ndarray,
        models_to_train: List[str] = ["lgbm", "catboost", "xgb"],
        n_splits: int = 5,
        seed: int = 2026,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Train downstream GBDT classifiers (LightGBM, CatBoost, XGBoost) on top of IndoBERT embeddings.
        Executes Stratified K-Fold CV, computes out-of-fold predictions, single scores, and an ensemble.
        """
        # Encode target labels
        unique_labels = sorted(list(pd.Series(y_raw).dropna().unique()))
        label2idx = {l: idx for idx, l in enumerate(unique_labels)}
        idx2label = {idx: l for idx, l in enumerate(unique_labels)}
        y = pd.Series(y_raw).map(label2idx).values
        n_classes = len(unique_labels)
        is_binary = n_classes == 2

        if log_callback:
            log_callback(f"[IndoBERT ML] Target classes ({n_classes}): {unique_labels}")
            log_callback(f"[IndoBERT ML] Cross-Validation: {n_splits}-Fold StratifiedKFold (Seed: {seed})")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # GPU acceleration check for GBDTs
        gpu_avail = False
        try:
            import torch
            gpu_avail = torch.cuda.is_available()
        except Exception:
            pass

        results: Dict[str, Any] = {
            "classes": [str(c) for c in unique_labels],
            "n_classes": n_classes,
            "models": {},
            "ensemble": None
        }

        oof_preds_dict = {}

        for m_name in models_to_train:
            m_key = m_name.lower().strip()
            if m_key not in ["lgbm", "catboost", "xgb"]:
                continue

            if log_callback:
                log_callback(f"\n[IndoBERT ML] >>> Training Classifier: {m_key.upper()} <<<")

            oof_probs = np.zeros((len(y), n_classes)) if not is_binary else np.zeros(len(y))
            fold_accuracies = []
            fold_f1_macros = []
            start_m_time = time.time()

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_emb, y)):
                X_tr, y_tr_orig = X_emb[train_idx], y[train_idx]
                X_va, y_va = X_emb[val_idx], y[val_idx]

                # Map fold classes to contiguous 0..K-1 to handle any rare classes missing in this fold
                fold_classes, y_tr = np.unique(y_tr_orig, return_inverse=True)
                k_classes = len(fold_classes)
                fold_is_binary = is_binary and (k_classes == 2)

                clf = IndoBERTEngine._fit_classifier(
                    m_key, k_classes, fold_is_binary, seed + fold, gpu_avail, X_tr, y_tr, log_callback
                )
                probs = clf.predict_proba(X_va)

                # Assign fold probabilities to global oof_probs matrix
                if is_binary and k_classes == 2:
                    p1 = probs[:, 1] if probs.ndim == 2 else probs
                    oof_probs[val_idx] = p1
                    preds = (p1 >= 0.5).astype(int)
                else:
                    for p_idx, global_cls in enumerate(fold_classes):
                        oof_probs[val_idx, int(global_cls)] = probs[:, p_idx]
                    preds = np.argmax(oof_probs[val_idx], axis=1)

                acc = accuracy_score(y_va, preds)
                f1_mac = f1_score(y_va, preds, average="macro", zero_division=0)
                fold_accuracies.append(acc)
                fold_f1_macros.append(f1_mac)
                if log_callback:
                    log_callback(f"  Fold {fold + 1}/{n_splits} | Accuracy: {acc * 100:.2f}% | Macro F1: {f1_mac * 100:.2f}%")

            duration = round(time.time() - start_m_time, 2)
            oof_preds_dict[m_key] = oof_probs

            # Overall OOF metrics
            if is_binary:
                final_preds = (oof_probs >= 0.5).astype(int)
            else:
                final_preds = np.argmax(oof_probs, axis=1)

            overall_acc = accuracy_score(y, final_preds)
            overall_f1_macro = f1_score(y, final_preds, average="macro", zero_division=0)
            overall_f1_weighted = f1_score(y, final_preds, average="weighted", zero_division=0)
            overall_precision = precision_score(y, final_preds, average="macro", zero_division=0)
            overall_recall = recall_score(y, final_preds, average="macro", zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y, final_preds).tolist()

            # Classification report per class
            rep = classification_report(y, final_preds, target_names=[str(c) for c in unique_labels], output_dict=True, zero_division=0)
            per_class_metrics = {}
            for c_str in [str(c) for c in unique_labels]:
                if c_str in rep:
                    per_class_metrics[c_str] = {
                        "precision": round(rep[c_str]["precision"] * 100, 2),
                        "recall": round(rep[c_str]["recall"] * 100, 2),
                        "f1_score": round(rep[c_str]["f1-score"] * 100, 2),
                        "support": int(rep[c_str]["support"])
                    }

            results["models"][m_key] = {
                "name": "LightGBM" if m_key == "lgbm" else ("CatBoost" if m_key == "catboost" else "XGBoost"),
                "accuracy": round(overall_acc * 100, 2),
                "f1_macro": round(overall_f1_macro * 100, 2),
                "f1_weighted": round(overall_f1_weighted * 100, 2),
                "precision_macro": round(overall_precision * 100, 2),
                "recall_macro": round(overall_recall * 100, 2),
                "fold_accuracies": [round(a * 100, 2) for a in fold_accuracies],
                "fold_f1_macros": [round(f * 100, 2) for f in fold_f1_macros],
                "training_time_sec": duration,
                "confusion_matrix": cm,
                "per_class": per_class_metrics
            }

            if log_callback:
                log_callback(f"[IndoBERT ML] {m_key.upper()} Completed: Accuracy = {overall_acc * 100:.2f}% | Macro F1 = {overall_f1_macro * 100:.2f}% ({duration}s)")

        # -------------------------------------------------------------
        # Multi-Model Soft Voting Ensemble
        # -------------------------------------------------------------
        if len(oof_preds_dict) >= 2:
            if log_callback:
                log_callback("\n[IndoBERT ML] >>> Computing Soft Voting Ensemble of All Trained Models <<<")

            avg_oof = sum(oof_preds_dict.values()) / len(oof_preds_dict)
            if is_binary:
                ens_preds = (avg_oof >= 0.5).astype(int)
            else:
                ens_preds = np.argmax(avg_oof, axis=1)

            ens_acc = accuracy_score(y, ens_preds)
            ens_f1_mac = f1_score(y, ens_preds, average="macro", zero_division=0)
            ens_f1_w = f1_score(y, ens_preds, average="weighted", zero_division=0)
            ens_cm = confusion_matrix(y, ens_preds).tolist()
            ens_rep = classification_report(y, ens_preds, target_names=[str(c) for c in unique_labels], output_dict=True, zero_division=0)

            ens_per_class = {}
            for c_str in [str(c) for c in unique_labels]:
                if c_str in ens_rep:
                    ens_per_class[c_str] = {
                        "precision": round(ens_rep[c_str]["precision"] * 100, 2),
                        "recall": round(ens_rep[c_str]["recall"] * 100, 2),
                        "f1_score": round(ens_rep[c_str]["f1-score"] * 100, 2),
                        "support": int(ens_rep[c_str]["support"])
                    }

            # Find best individual model
            best_m = max(results["models"].items(), key=lambda x: x[1]["f1_macro"])
            gain = round((ens_f1_mac * 100) - best_m[1]["f1_macro"], 2)

            results["ensemble"] = {
                "name": f"IndoBERT + Multi-Model Ensemble ({' + '.join([m.upper() for m in oof_preds_dict.keys()])})",
                "accuracy": round(ens_acc * 100, 2),
                "f1_macro": round(ens_f1_mac * 100, 2),
                "f1_weighted": round(ens_f1_w * 100, 2),
                "gain_over_best_single": gain,
                "best_single_model": best_m[1]["name"],
                "confusion_matrix": ens_cm,
                "per_class": ens_per_class
            }

            if log_callback:
                log_callback(f"[IndoBERT ML] Ensemble Score: Accuracy = {ens_acc * 100:.2f}% | Macro F1 = {ens_f1_mac * 100:.2f}% (Gain: {gain:+.2f}%)")

        return results

