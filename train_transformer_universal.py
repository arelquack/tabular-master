import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding
    )
except ImportError:
    print("[ERROR] HuggingFace transformers not installed. Run: pip install transformers accelerate")
    sys.exit(1)


def resolve_model_path(model_identifier: str) -> str:
    """Resolve HuggingFace model path from local offline cache or repo ID."""
    if "wirasana" in model_identifier.lower():
        model_identifier = "w11wo/indonesian-roberta-base-sentiment-classifier"

    # 1. Exact path check
    if os.path.exists(model_identifier) and os.path.exists(os.path.join(model_identifier, "config.json")):
        return model_identifier

    # 2. Check models/ directory
    basename = model_identifier.split("/")[-1]
    candidates = [
        os.path.join("models", model_identifier),
        os.path.join("models", basename),
        os.path.join("models", "indobert-base-p1") if "indobert" in basename.lower() and "tweet" not in basename.lower() else None,
        os.path.join("models", "indobertweet-base-uncased") if "tweet" in basename.lower() else None,
        os.path.join("models", "indonesian-roberta-base-sentiment") if "roberta" in basename.lower() else None,
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.exists(os.path.join(c, "config.json")):
            print(f"[OFFLINE CACHE] Resolved '{model_identifier}' to local folder '{c}'")
            return c
    return model_identifier


class PyTorchTextDataset(torch.utils.data.Dataset):
    """Native PyTorch Dataset that eliminates dependencies on datasets package."""
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) if not isinstance(val[idx], torch.Tensor) else val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx]) if not isinstance(self.labels[idx], torch.Tensor) else self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class UniversalTrainer(Trainer):
    """Custom Trainer implementing weighted CrossEntropyLoss (Classification) or MSELoss (Regression)."""
    def __init__(self, class_weights=None, is_regression=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.is_regression = is_regression

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.is_regression:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.view(-1), labels.view(-1).float())
        else:
            if self.class_weights is not None:
                weights = self.class_weights.to(logits.device)
                loss_fn = nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def run_training(
    case_id: str,
    model_name: str = "indobenchmark/indobert-base-p1",
    task_type: str = "classification",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = None,
    text_col: str = None,
    text_col_2: str = None,
    label_col: str = None,
    val_size: float = 0.2,
    seed: int = 2026,
    log_callback=None
):
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    model_name = resolve_model_path(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    log(f"============================================================")
    log(f" UNIVERSAL TRANSFORMER PIPELINE: {case_id}")
    log(f" Active Device: {device.upper()} | GPU: {gpu_name}")
    log(f" Backbone Model: {model_name}")
    log(f" Mode: {task_type.upper()}")
    log(f"============================================================")

    case_dir = os.path.join("cases", case_id)
    if not os.path.exists(case_dir):
        alt_id = case_id.replace("_", "-") if "_" in case_id else case_id.replace("-", "_")
        alt_dir = os.path.join("cases", alt_id)
        if os.path.exists(alt_dir):
            case_dir = alt_dir
        else:
            raise FileNotFoundError(f"Case directory '{case_dir}' not found.")

    case_json_path = os.path.join(case_dir, "case.json")
    meta = {}
    if os.path.exists(case_json_path):
        with open(case_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # 1. Load Train Dataset
    train_csv_path = os.path.join(case_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv not found in '{case_dir}'")

    df = pd.read_csv(train_csv_path)
    log(f"Loaded dataset: {len(df)} rows.")

    # 2. Resolve Text and Label Columns
    if not text_col:
        text_col = meta.get("text_column") or "clean_text"
        if text_col not in df.columns:
            for candidate in ["clean_text", "comment", "text", "content", "document", "teks", "sentence", "premise"]:
                if candidate in df.columns:
                    text_col = candidate
                    break

    if not text_col_2 and meta.get("text_column_2"):
        text_col_2 = meta.get("text_column_2")
    if not text_col_2:
        for candidate in ["text_2", "sentence_2", "hypothesis", "comment_2", "text_b"]:
            if candidate in df.columns and candidate != text_col:
                text_col_2 = candidate
                break

    if not label_col:
        label_col = meta.get("target_column") or meta.get("target_col") or "sentiment"
        if label_col not in df.columns:
            for candidate in ["sentiment", "category", "target", "label", "target_biro", "score", "rating"]:
                if candidate in df.columns:
                    label_col = candidate
                    break

    is_pairwise = bool(text_col_2 and text_col_2 in df.columns)
    if is_pairwise:
        log(f"Pairwise Cross-Encoder Mode Active: '{text_col}' <-> '{text_col_2}' | Target: '{label_col}'")
        df = df.dropna(subset=[text_col, text_col_2, label_col]).copy()
        df[text_col] = df[text_col].astype(str)
        df[text_col_2] = df[text_col_2].astype(str)
    else:
        log(f"Single-Text Mode Active: '{text_col}' | Target: '{label_col}'")
        df = df.dropna(subset=[text_col, label_col]).copy()
        df[text_col] = df[text_col].astype(str)

    # Auto-detect task type
    if task_type == "classification" and meta.get("task_type") == "regression":
        task_type = "regression"
    elif task_type == "classification" and pd.api.types.is_float_dtype(df[label_col]) and df[label_col].nunique() > 20:
        task_type = "regression"

    is_regression = (task_type.lower() == "regression")
    log(f"Task Type: {task_type.upper()}")

    # 3. Label Preparation
    if is_regression:
        df["label"] = df[label_col].astype(float)
        unique_labels = ["target"]
        num_labels = 1
        label2id = {"target": 0}
        id2label = {0: "target"}
        log(f"Regression Range: Min = {df['label'].min():.4f}, Max = {df['label'].max():.4f}, Mean = {df['label'].mean():.4f}")
    else:
        unique_labels = sorted(df[label_col].unique().tolist())
        num_labels = len(unique_labels)
        label2id = {l: idx for idx, l in enumerate(unique_labels)}
        id2label = {idx: l for idx, l in enumerate(unique_labels)}
        df["label"] = df[label_col].map(label2id)
        log(f"Total Classes: {num_labels} -> {unique_labels}")
        dist = df[label_col].value_counts().to_dict()
        log(f"Class Distribution: {dist}")

    # 4. Adaptive Sequence Length
    if max_length is None or max_length <= 0:
        if is_pairwise:
            words_per_doc = df[text_col].apply(lambda s: len(s.split())) + df[text_col_2].apply(lambda s: len(s.split()))
        else:
            words_per_doc = df[text_col].apply(lambda s: len(s.split()))
        p95_words = int(np.percentile(words_per_doc, 95))
        adaptive_len = min(512, max(64, int(p95_words * 1.35)))
        if adaptive_len <= 96:
            max_length = 96
        elif adaptive_len <= 128:
            max_length = 128
        elif adaptive_len <= 256:
            max_length = 256
        else:
            max_length = 512
        log(f"Adaptive Max Sequence Length (P95: {p95_words} words): {max_length}")
    else:
        log(f"Manual Max Sequence Length: {max_length}")

    # 5. Train-Val Split
    val_size = min(0.25, max(0.1, val_size))
    if is_regression:
        train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed)
        class_weights_tensor = None
    else:
        class_counts = df["label"].value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        if rare_classes:
            log(f"Shielding rare classes (< 2 samples) to train set: {rare_classes}")
            rare_mask = df["label"].isin(rare_classes)
            rare_df = df[rare_mask]
            splittable_df = df[~rare_mask]
            train_df, val_df = train_test_split(
                splittable_df,
                test_size=val_size,
                random_state=seed,
                stratify=splittable_df["label"]
            )
            train_df = pd.concat([train_df, rare_df], ignore_index=True)
        else:
            train_df, val_df = train_test_split(
                df,
                test_size=val_size,
                random_state=seed,
                stratify=df["label"]
            )
        weights_arr = compute_class_weight(class_weight="balanced", classes=np.arange(num_labels), y=train_df["label"].values)
        class_weights_tensor = torch.tensor(weights_arr, dtype=torch.float)
        log(f"Balanced Class Weights: {np.round(weights_arr, 3).tolist()}")

    val_orig_indices = val_df.index.to_numpy()
    log(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # 6. Tokenizer & Native PyTorch Dataset Preparation
    log(f"Initializing tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if is_pairwise:
        train_encodings = tokenizer(
            train_df[text_col].tolist(),
            train_df[text_col_2].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length
        )
        val_encodings = tokenizer(
            val_df[text_col].tolist(),
            val_df[text_col_2].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length
        )
    else:
        train_encodings = tokenizer(
            train_df[text_col].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length
        )
        val_encodings = tokenizer(
            val_df[text_col].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length
        )

    train_labels = train_df["label"].values.astype(float) if is_regression else train_df["label"].values.astype(int)
    val_labels = val_df["label"].values.astype(float) if is_regression else val_df["label"].values.astype(int)

    train_tokenized = PyTorchTextDataset(train_encodings, train_labels)
    val_tokenized = PyTorchTextDataset(val_encodings, val_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. Model Initialization
    log(f"Loading transformer model ({'Regression' if is_regression else 'Classification'})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="regression" if is_regression else "single_label_classification",
        id2label=id2label,
        label2id=label2id
    )

    # 8. Compute Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if is_regression:
            preds = logits.squeeze()
            labels = labels.squeeze()
            rmse = float(np.sqrt(mean_squared_error(labels, preds)))
            mae = float(mean_absolute_error(labels, preds))
            ss_res = np.sum((labels - preds) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-8))) if ss_tot > 0 else 0.0
            return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
        else:
            preds = np.argmax(logits, axis=1)
            acc = float(accuracy_score(labels, preds))
            macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
            weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
            return {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4)
            }

    # 9. Training Arguments
    output_checkpoint_dir = os.path.join(case_dir, "transformer_checkpoints")
    best_model_dir = os.path.join(case_dir, "best_transformer_model")
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="rmse" if is_regression else "macro_f1",
        greater_is_better=False if is_regression else True,
        logging_steps=max(5, len(train_df) // (batch_size * 4)),
        report_to="none",
        seed=seed
    )

    trainer = UniversalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        is_regression=is_regression
    )

    log(f"Starting Fine-Tuning on {device.upper()} for {epochs} epochs...")
    trainer.train()

    # 10. Final Evaluation & OOF Persistence
    log("Executing final validation evaluation...")
    eval_results = trainer.evaluate()
    val_preds_raw = trainer.predict(val_tokenized)

    if is_regression:
        val_preds_continuous = val_preds_raw.predictions.squeeze()
        y_true = val_preds_raw.label_ids.squeeze()
        np.save(os.path.join(case_dir, "transformer_val_probs.npy"), val_preds_continuous)
        np.save(os.path.join(case_dir, "transformer_val_targets.npy"), y_true)
        np.save(os.path.join(case_dir, "transformer_val_indices.npy"), val_orig_indices)
        report_dict = {
            "rmse": eval_results.get("eval_rmse", 0),
            "mae": eval_results.get("eval_mae", 0),
            "r2": eval_results.get("eval_r2", 0)
        }
        smart_diagnostics = {
            "status": "healthy",
            "val_loss": eval_results.get("eval_loss", 0),
            "advice": f"Model regression fine-tuning selesai dengan RMSE = {eval_results.get('eval_rmse', 0):.4f} dan R2 = {eval_results.get('eval_r2', 0):.4f}."
        }
    else:
        def softmax_np(x, axis=-1):
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / np.sum(e_x, axis=axis, keepdims=True)

        val_probs = softmax_np(val_preds_raw.predictions, axis=1)
        y_preds = np.argmax(val_preds_raw.predictions, axis=1)
        y_true = val_preds_raw.label_ids

        np.save(os.path.join(case_dir, "transformer_val_probs.npy"), val_probs)
        np.save(os.path.join(case_dir, "transformer_val_targets.npy"), y_true)
        np.save(os.path.join(case_dir, "transformer_val_indices.npy"), val_orig_indices)
        np.save(os.path.join(case_dir, "transformer_classes.npy"), np.array(unique_labels))

        report_dict = classification_report(
            y_true,
            y_preds,
            target_names=[str(c) for c in unique_labels],
            labels=np.arange(num_labels),
            output_dict=True,
            zero_division=0
        )
        smart_diagnostics = {
            "status": "healthy",
            "val_loss": eval_results.get("eval_loss", 0),
            "advice": f"Model klasifikasi selesai dengan Akurasi = {eval_results.get('eval_accuracy', 0)*100:.2f}% dan Macro F1 = {eval_results.get('eval_macro_f1', 0)*100:.2f}%."
        }

    # 11. Save Best Model Weights
    log(f"Exporting best model checkpoint to '{best_model_dir}'...")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # 12. Inference on Test Set
    test_csv_path = os.path.join(case_dir, "test.csv")
    test_preds_sample = []
    if os.path.exists(test_csv_path):
        log(f"Test dataset detected at '{test_csv_path}'. Running inference...")
        test_df = pd.read_csv(test_csv_path)
        id_col = meta.get("id_col") or ("id" if "id" in test_df.columns else test_df.columns[0])

        if text_col in test_df.columns:
            test_df[text_col] = test_df[text_col].fillna("").astype(str)
            if is_pairwise and text_col_2 in test_df.columns:
                test_df[text_col_2] = test_df[text_col_2].fillna("").astype(str)
                test_encodings = tokenizer(test_df[text_col].tolist(), test_df[text_col_2].tolist(), padding=True, truncation=True, max_length=max_length)
            else:
                test_encodings = tokenizer(test_df[text_col].tolist(), padding=True, truncation=True, max_length=max_length)

            test_tokenized = PyTorchTextDataset(test_encodings)
            test_preds_raw = trainer.predict(test_tokenized)

            if is_regression:
                test_preds = test_preds_raw.predictions.squeeze()
                sub_df = pd.DataFrame({id_col: test_df[id_col], label_col: np.round(test_preds, 4)})
            else:
                test_probs = softmax_np(test_preds_raw.predictions, axis=1)
                test_preds = np.argmax(test_probs, axis=1)
                sub_df = pd.DataFrame({id_col: test_df[id_col], label_col: [id2label[i] for i in test_preds]})
                np.save(os.path.join(case_dir, "transformer_test_probs.npy"), test_probs)

            sub_path = os.path.join(case_dir, "submission_transformer.csv")
            sub_df.to_csv(sub_path, index=False)
            log(f"Saved submission to '{sub_path}' ({len(sub_df)} rows).")
            test_preds_sample = sub_df.head(5).to_dict(orient="records")

    summary_results = {
        "case_id": case_id,
        "model_name": model_name,
        "task_type": task_type,
        "is_pairwise": is_pairwise,
        "device": device,
        "gpu_name": gpu_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "eval_loss": eval_results.get("eval_loss", 0),
        "accuracy": eval_results.get("eval_accuracy", 0) if not is_regression else None,
        "macro_f1": eval_results.get("eval_macro_f1", 0) if not is_regression else None,
        "rmse": eval_results.get("eval_rmse", 0) if is_regression else None,
        "mae": eval_results.get("eval_mae", 0) if is_regression else None,
        "r2": eval_results.get("eval_r2", 0) if is_regression else None,
        "classification_report": report_dict if not is_regression else None,
        "test_predictions_sample": test_preds_sample,
        "smart_diagnostics": smart_diagnostics,
        "status": "completed"
    }

    results_file = os.path.join(case_dir, "transformer_results.json")
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump(summary_results, rf, indent=2)

    log(f"Results successfully saved to '{results_file}'.")
    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Transformer Fine-Tuning Runner")
    parser.add_argument("--case", type=str, default="sentimen_karhutla", help="Case ID in cases/ folder")
    parser.add_argument("--model", type=str, default="indobenchmark/indobert-base-p1", help="Pretrained model identifier")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification", "regression"], help="Task type")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length (None for auto P95)")
    parser.add_argument("--text_col", type=str, default=None, help="Primary text column name")
    parser.add_argument("--text_col_2", type=str, default=None, help="Second text column for pairwise matching / NLI")
    parser.add_argument("--label_col", type=str, default=None, help="Target/label column name")
    args = parser.parse_args()

    run_training(
        case_id=args.case,
        model_name=args.model,
        task_type=args.task_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        text_col=args.text_col,
        text_col_2=args.text_col_2,
        label_col=args.label_col
    )
