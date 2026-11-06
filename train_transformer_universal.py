import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding
    )
    from datasets import Dataset
except ImportError:
    print("[ERROR] HuggingFace transformers or datasets not installed. Run: pip install transformers datasets accelerate")
    sys.exit(1)


class WeightedTrainer(Trainer):
    """Custom Trainer implementing weighted CrossEntropyLoss to resolve class imbalance."""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
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
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = None,
    val_size: float = 0.2,
    seed: int = 2026,
    log_callback=None
):

def resolve_model_path(model_identifier: str) -> str:
    # Auto-correct typo from previous snippet
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
    text_col = meta.get("text_column") or "clean_text"
    if text_col not in df.columns:
        for candidate in ["clean_text", "comment", "text", "content", "document", "teks"]:
            if candidate in df.columns:
                text_col = candidate
                break

    label_col = meta.get("target_column") or meta.get("target_col") or "sentiment"
    if label_col not in df.columns:
        for candidate in ["sentiment", "category", "target", "label", "target_biro"]:
            if candidate in df.columns:
                label_col = candidate
                break

    log(f"Text Column: '{text_col}' | Label Column: '{label_col}'")

    # Clean missing values
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)

    # 3. Label Encoding
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
        words_per_doc = df[text_col].apply(lambda s: len(s.split()))
        p95_words = int(np.percentile(words_per_doc, 95))
        # Add 30% margin for subword tokenization
        adaptive_len = min(512, max(64, int(p95_words * 1.35)))
        # Snap to powers or friendly sizes
        if adaptive_len <= 96:
            max_length = 96
        elif adaptive_len <= 128:
            max_length = 128
        elif adaptive_len <= 256:
            max_length = 256
        else:
            max_length = 512
        log(f"Adaptive Max Sequence Length computed from P95 ({p95_words} words): {max_length}")
    else:
        log(f"Manual Max Sequence Length: {max_length}")

    # 5. Train-Val Split with Rare Class Shielding
    val_size = min(0.25, max(0.1, val_size))
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

    val_orig_indices = val_df.index.to_numpy()
    log(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # 6. Balanced Class Weights
    weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_df["label"].values
    )
    class_weights_tensor = torch.tensor(weights_arr, dtype=torch.float)
    log(f"Balanced Class Weights: {np.round(weights_arr, 3).tolist()}")

    # 7. Tokenizer & Dataset Preparation
    log(f"Initializing tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch[text_col],
            padding=False,
            truncation=True,
            max_length=max_length
        )

    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[[text_col, "label"]].reset_index(drop=True))

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8. Model Initialization
    log(f"Loading sequence classification model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # 9. Compute Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4)
        }

    # 10. Training Arguments
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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=max(10, len(train_df) // (batch_size * 4)),
        report_to="none",
        seed=seed
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor
    )

    log(f"Starting Fine-Tuning on {device.upper()} for {epochs} epochs...")
    trainer.train()

    # 11. Final Evaluation & Classification Report
    log("Executing final validation evaluation...")
    eval_results = trainer.evaluate()
    val_preds_raw = trainer.predict(val_tokenized)
    y_preds = np.argmax(val_preds_raw.predictions, axis=1)
    y_true = val_preds_raw.label_ids

    # Pure numpy softmax to extract calibrated probabilities
    def softmax_np(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    val_probs = softmax_np(val_preds_raw.predictions, axis=1)
    np.save(os.path.join(case_dir, "transformer_val_probs.npy"), val_probs)
    np.save(os.path.join(case_dir, "transformer_val_targets.npy"), y_true)
    np.save(os.path.join(case_dir, "transformer_val_indices.npy"), val_orig_indices)
    np.save(os.path.join(case_dir, "transformer_classes.npy"), np.array(unique_labels))
    log(f"Saved validation probabilities matrix: {val_probs.shape}")

    report_dict = classification_report(
        y_true,
        y_preds,
        labels=np.arange(num_labels),
        target_names=unique_labels,
        output_dict=True,
        zero_division=0
    )

    log("================ FINAL VALIDATION METRICS ================")
    log(f"Accuracy:    {eval_results.get('eval_accuracy', 0):.4f}")
    log(f"Macro-F1:    {eval_results.get('eval_macro_f1', 0):.4f}")
    log(f"Weighted-F1: {eval_results.get('eval_weighted_f1', 0):.4f}")
    log("==========================================================")

    # 12. Save Best Model and Tokenizer
    log(f"Saving best transformer model weights to '{best_model_dir}'...")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # 13. Check if test.csv exists and generate blind inference
    test_csv_path = os.path.join(case_dir, "test.csv")
    test_preds_sample = []
    if os.path.exists(test_csv_path):
        log(f"Found blind test set '{test_csv_path}'. Running inference...")
        test_df = pd.read_csv(test_csv_path)
        if text_col in test_df.columns:
            test_df[text_col] = test_df[text_col].fillna("").astype(str)
            test_ds = Dataset.from_pandas(test_df[[text_col]].reset_index(drop=True))
            test_tokenized = test_ds.map(tokenize_function, batched=True)
            test_preds_out = trainer.predict(test_tokenized)
            test_pred_indices = np.argmax(test_preds_out.predictions, axis=1)
            test_df["prediction"] = [id2label[i] for i in test_pred_indices]
            test_probs = softmax_np(test_preds_out.predictions, axis=1)
            np.save(os.path.join(case_dir, "transformer_test_probs.npy"), test_probs)

            id_col = meta.get("id_col") or "id"
            if id_col not in test_df.columns:
                test_df[id_col] = range(len(test_df))

            sub_path = os.path.join(case_dir, "submission_transformer.csv")
            test_df[[id_col, "prediction"]].to_csv(sub_path, index=False)
            log(f"Saved transformer test predictions to '{sub_path}'.")
            test_preds_sample = test_df[[id_col, "prediction"]].head(10).to_dict(orient="records")

    # 14. Automated Smart Loss & Sweet Spot Diagnostics
    eval_loss = float(eval_results.get("eval_loss", 0.0))
    random_guess_loss = round(float(np.log(num_labels)), 4) if num_labels > 1 else 0.6931
    loss_reduction_pct = round(float(max(0.0, (1.0 - eval_loss / random_guess_loss) * 100)), 2) if random_guess_loss > 0 else 0.0

    ratio = eval_loss / random_guess_loss if random_guess_loss > 0 else 1.0
    if ratio < 0.25:
        learning_phase = "Sweet Spot Optimal"
        phase_color = "emerald"
        advice = f"Validation loss ({eval_loss:.4f}) berhasil terpangkas {loss_reduction_pct}% dari tebakan acak ({random_guess_loss:.4f})! Konvergensi model berada di titik optimal. Siap untuk dievaluasi dan di-blend."
    elif ratio < 0.45:
        learning_phase = "Healthy Convergence"
        phase_color = "blue"
        advice = f"Validation loss ({eval_loss:.4f}) turun {loss_reduction_pct}% dari baseline acak ({random_guess_loss:.4f}) pada {num_labels} kelas. Model belajar dengan sehat dan stabil. Coba naikkan 1-2 epoch untuk mendorong loss lebih dalam jika belum overfit."
    elif ratio < 0.75:
        learning_phase = "Moderate Progress"
        phase_color = "amber"
        advice = f"Validation loss ({eval_loss:.4f}) turun {loss_reduction_pct}% dari tebakan acak ({random_guess_loss:.4f}). Model sedang belajar membedakan {num_labels} kelas. Disarankan menambah epoch atau menaikkan learning rate sedikit."
    else:
        learning_phase = "High Loss / Noise Warning"
        phase_color = "red"
        advice = f"Validation loss ({eval_loss:.4f}) masih dekat dengan tebakan acak ({random_guess_loss:.4f}). Periksa apakah ada noise pada label data atau model membutuhkan lebih banyak epoch."

    smart_diagnostics = {
        "num_classes": num_labels,
        "eval_loss": round(eval_loss, 4),
        "random_guess_loss": random_guess_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "learning_phase": learning_phase,
        "phase_color": phase_color,
        "advice": advice
    }

    # 15. Save Results JSON
    summary_results = {
        "case_id": case_id,
        "model_name": model_name,
        "device": device,
        "gpu_name": gpu_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "accuracy": eval_results.get("eval_accuracy", 0),
        "macro_f1": eval_results.get("eval_macro_f1", 0),
        "weighted_f1": eval_results.get("eval_weighted_f1", 0),
        "eval_loss": eval_results.get("eval_loss", 0),
        "classification_report": report_dict,
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
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length (None for auto P95)")
    args = parser.parse_args()

    run_training(
        case_id=args.case,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length
    )
