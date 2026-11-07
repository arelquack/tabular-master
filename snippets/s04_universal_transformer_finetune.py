"""
SNIPPET 04: Universal Transformer Fine-Tuner (PyTorch + HuggingFace)
Deskripsi: Melatih model Deep Learning Transformer (IndoBERT, XLM-RoBERTa, RoBERTa)
dengan native PyTorch Dataset (bebas dependensi library 'datasets') dan akselerasi GPU CUDA/FP16.
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train_transformer_universal import run_training, resolve_model_path


def finetune_transformer(
    case_id: str,
    model_name: str = "xlm-roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    text_col: str = None,
    label_col: str = None
):
    print("=" * 65)
    print(" UNIVERSAL TRANSFORMER FINE-TUNING RUNNER")
    print("=" * 65)
    print(f"Kasus ID   : {case_id}")
    print(f"Model Name : {model_name} (Resolved: {resolve_model_path(model_name)})")
    print(f"Hyperparams: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, MaxLen={max_length}")

    results = run_training(
        case_id=case_id,
        model_name=model_name,
        task_type="classification",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        text_col=text_col,
        label_col=label_col,
        weight_decay=0.01,
        warmup_ratio=0.1,
        grad_accum_steps=1
    )

    print("\nHASIL EVALUASI TRANSFORMER:")
    print(f"Macro-F1 Validasi: {results.get('macro_f1', 0)*100:.2f}%")
    print(f"Akurasi Validasi : {results.get('accuracy', 0)*100:.2f}%")
    print(f"OOF Probabilities tersimpan di 'cases/{case_id}/transformer_val_probs.npy'.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Transformer Fine-Tuning")
    parser.add_argument("--case", type=str, required=True, help="Nama folder kasus di cases/<case_id>")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Pretrained model (xlm-roberta-base, indobenchmark/indobert-base-p1, roberta-base)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    finetune_transformer(
        case_id=args.case,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_len
    )
