"""
Universal Offline NLP Model Downloader & Verification Tool
Designed for Gemastik Final Competition (PC Panitia / PC Dosen)
Ensures 100% offline capability without internet connectivity.
"""

import os
import sys
import argparse

ALL_MODELS = {
    # Indonesian Models
    "indobert": {
        "repo_id": "indobenchmark/indobert-base-p1",
        "local_dir": "models/indobert-base-p1",
        "model_type": "encoder",
        "category": "indonesian",
        "description": "Standard Indonesian BERT (Formal & General News, Kompas, Berita)"
    },
    "indobertweet": {
        "repo_id": "indolem/indobertweet-base-uncased",
        "local_dir": "models/indobertweet-base-uncased",
        "model_type": "encoder",
        "category": "indonesian",
        "description": "Indonesian Twitter & Social Media BERT (Slang, Informal, Komentar)"
    },
    "indonesian-roberta": {
        "repo_id": "w11wo/indonesian-roberta-base-sentiment-classifier",
        "local_dir": "models/indonesian-roberta-base-sentiment",
        "model_type": "sequence_classification",
        "category": "indonesian",
        "description": "Pre-tuned Indonesian RoBERTa Sentiment Classifier (SmSA benchmark)"
    },
    
    # Multilingual & Mixed Code-Switching
    "xlm-roberta": {
        "repo_id": "xlm-roberta-base",
        "local_dir": "models/xlm-roberta-base",
        "model_type": "encoder",
        "category": "multilingual",
        "description": "XLM-RoBERTa Multilingual (100 languages, Mixed Indo-Eng, Code-Switching, Slang)"
    },

    # English & Fast Benchmark Models
    "distilbert": {
        "repo_id": "distilbert-base-uncased",
        "local_dir": "models/distilbert-base-uncased",
        "model_type": "encoder",
        "category": "english",
        "description": "DistilBERT English (Lightweight, Ultra-Fast 260MB, Monolingual English)"
    },
    "roberta-base": {
        "repo_id": "roberta-base",
        "local_dir": "models/roberta-base",
        "model_type": "encoder",
        "category": "english",
        "description": "Standard RoBERTa English (Kaggle Benchmark Standard)"
    },
    "deberta-v3-small": {
        "repo_id": "microsoft/deberta-v3-small",
        "local_dir": "models/deberta-v3-small",
        "model_type": "encoder",
        "category": "english",
        "description": "DeBERTa v3 Small (Disentangled Attention, Top SOTA Competition Model)"
    }
}

ALL_MODELS.update({
    "sdd-sentiment-general": {
        "repo_id": "AzrilFahmiardi/sdd-sentiment-general",
        "local_dir": "models/sdd-sentiment-general",
        "category": "indonesian",
        "model_type": "sequence_classification",
        "description": "IndoBERT 3-class sentiment classifier for Indonesian news and social media"
    },
    "indobertweet-sentiment": {
        "repo_id": "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis",
        "local_dir": "models/indobertweet-base-Indonesian-sentiment-analysis",
        "category": "indonesian",
        "model_type": "sequence_classification",
        "description": "IndoBERTweet fine-tuned for Indonesian sentiment classification"
    },
    "indonesian-roberta-base": {
        "repo_id": "flax-community/indonesian-roberta-base",
        "local_dir": "models/indonesian-roberta-base",
        "category": "indonesian",
        "model_type": "encoder",
        "description": "Indonesian RoBERTa base encoder for feature extraction"
    },
})

def verify_offline_load(local_dir: str, model_type: str = "encoder") -> bool:
    """Verify the checkpoint with the model architecture it is intended to provide."""
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        if model_type == "sequence_classification":
            mod = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
        else:
            mod = AutoModel.from_pretrained(local_dir, local_files_only=True)
        return tok is not None and mod is not None
    except Exception as error:
        print(f"      [VERIFY ALERT]: {error}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Transformer models for offline competition deployment.")
    parser.add_argument(
        "--preset",
        type=str,
        default="all",
        choices=["all", "indonesian", "core", "english"],
        help="Model preset to download: 'all', 'core' (Indonesian + XLM-R), 'indonesian', or 'english'"
    )
    args, unknown = parser.parse_known_args()

    if args.preset == "all":
        selected_keys = list(ALL_MODELS.keys())
    elif args.preset == "core":
        selected_keys = ["indobert", "indobertweet", "indonesian-roberta", "sdd-sentiment-general", "indobertweet-sentiment", "indonesian-roberta-base", "xlm-roberta"]
    elif args.preset == "indonesian":
        selected_keys = ["indobert", "indobertweet", "indonesian-roberta", "sdd-sentiment-general", "indobertweet-sentiment", "indonesian-roberta-base"]
    elif args.preset == "english":
        selected_keys = ["distilbert", "roberta-base", "deberta-v3-small"]
    else:
        selected_keys = list(ALL_MODELS.keys())

    print("=" * 75)
    print(" OFFLINE MODEL PACKAGER & DOWNLOADER (GEMASTIK FINAL DEPLOYMENT)")
    print(f" Active Preset: '{args.preset.upper()}' ({len(selected_keys)} Models Selected)")
    print("=" * 75)
    print("Target Destination: ./models/ inside project root directory\n")

    try:
        from huggingface_hub import snapshot_download
        has_hub = True
    except ImportError:
        has_hub = False

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
        has_hf = True
    except ImportError:
        has_hf = False

    if not has_hub and not has_hf:
        print("[ERROR] Neither 'huggingface_hub' nor 'transformers' is installed.")
        print("Please run: pip install huggingface_hub transformers")
        sys.exit(1)

    os.makedirs("models", exist_ok=True)
    all_success = True

    for key in selected_keys:
        item = ALL_MODELS[key]
        repo_id = item["repo_id"]
        local_dir = item["local_dir"]
        desc = item["description"]

        print(f"\n[+] Target Model : '{repo_id}'")
        print(f"    Target Dir   : '{local_dir}'")
        print(f"    Description  : {desc}")

        # Check if already downloaded and verified
        if os.path.exists(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
            print("    [CHECK] Checking existing local cache...")
            if verify_offline_load(local_dir, item["model_type"]):
                print(f"    [VERIFIED OFFLINE READY] Model '{repo_id}' is ready in '{local_dir}'!")
                continue
            else:
                print("    [WARN] Existing folder incomplete. Resuming download...")

        print("    [DOWNLOADING] Downloading complete weights and tokenizer files...")
        success = False

        if has_hub:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "optimizer.pt", "scheduler.pt", "training_args.bin", "events.out.tfevents*"]
                )
                print(f"    [DOWNLOAD OK] Snapshot downloaded successfully via huggingface_hub.")
                success = True
            except Exception as e:
                print(f"    [WARN] snapshot_download issue: {e}. Trying transformers fallback...")

        if not success and has_hf:
            try:
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                tokenizer.save_pretrained(local_dir)
                if item["model_type"] == "sequence_classification":
                    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
                else:
                    model = AutoModel.from_pretrained(repo_id)
                model.save_pretrained(local_dir)
                print(f"    [DOWNLOAD OK] Saved successfully via transformers API.")
                success = True
            except Exception as e:
                print(f"    [ERROR] transformers fallback failed: {e}")

        if success:
            print("    [VERIFYING] Testing offline loading (local_files_only=True)...")
            if verify_offline_load(local_dir, item["model_type"]):
                print(f"    [SUCCESS] '{repo_id}' verified 100% OFFLINE READY!")
            else:
                print(f"    [WARNING] Model downloaded but offline verification produced an alert.")
        else:
            print(f"    [FAILED] Failed to download '{repo_id}'. Check internet connection on PC Dosen.")
            all_success = False

    print("\n" + "=" * 75)
    if all_success:
        print(" [SUCCESS] ALL REQUESTED PRETRAINED MODELS READY IN './models/'")
        print(" You can now safely copy the entire project folder to a Flashdisk")
        print(" and run 100% OFFLINE on PC Panitia with zero internet connection!")
    else:
        print(" [WARNING] Some models encountered download issues. Re-run script to complete.")
    print("=" * 75)

if __name__ == "__main__":
    main()
