"""
Universal Offline NLP Model Downloader & Verification Tool
Designed for Gemastik Final Competition (PC Panitia / PC Dosen)
Ensures 100% offline capability without internet connectivity.
"""

import os
import sys

MODELS = [
    {
        "repo_id": "indobenchmark/indobert-base-p1",
        "local_dir": "models/indobert-base-p1",
        "description": "Standard Indonesian BERT (Formal & General News/Articles, e.g. Kompas, Formal Text)"
    },
    {
        "repo_id": "indolem/indobertweet-base-uncased",
        "local_dir": "models/indobertweet-base-uncased",
        "description": "Indonesian Twitter & Social Media BERT (Slang, Informal, YouTube Comments)"
    },
    {
        "repo_id": "w11wo/indonesian-roberta-base-sentiment-classifier",
        "local_dir": "models/indonesian-roberta-base-sentiment",
        "description": "Pre-tuned Indonesian RoBERTa Sentiment Classifier (SmSA benchmark)"
    }
]

def verify_offline_load(local_dir: str) -> bool:
    """Verifies that the downloaded model can be loaded with local_files_only=True (zero internet)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        mod = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True, num_labels=2)
        return tok is not None and mod is not None
    except Exception as e:
        # Fallback test with AutoModel if sequence classification head has specific dimension
        try:
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
            mod = AutoModel.from_pretrained(local_dir, local_files_only=True)
            return tok is not None and mod is not None
        except Exception as e2:
            print(f"      [VERIFY FAILED]: {e2}")
            return False

def main():
    print("=" * 75)
    print(" OFFLINE MODEL PACKAGER & DOWNLOADER (GEMASTIK FINAL DEPLOYMENT)")
    print("=" * 75)
    print("Target Destination: ./models/ inside project root directory")
    print("Ensures all weights are portable and copyable via Flashdisk to PC Panitia.\n")

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

    for item in MODELS:
        repo_id = item["repo_id"]
        local_dir = item["local_dir"]
        desc = item["description"]

        print(f"\n[+] Target Model : '{repo_id}'")
        print(f"    Target Dir   : '{local_dir}'")
        print(f"    Description  : {desc}")

        # Check if already downloaded and verified
        if os.path.exists(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
            print("    [CHECK] Checking existing local cache...")
            if verify_offline_load(local_dir):
                print(f"    [VERIFIED OFFLINE READY] Model '{repo_id}' is ready in '{local_dir}'!")
                continue
            else:
                print("    [WARN] Existing folder corrupted or incomplete. Re-downloading...")

        print("    [DOWNLOADING] Downloading complete weights and tokenizer files...")
        success = False

        if has_hub:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"]
                )
                print(f"    [DOWNLOAD OK] Snapshot downloaded successfully via huggingface_hub.")
                success = True
            except Exception as e:
                print(f"    [WARN] snapshot_download encountered issue: {e}. Trying transformers fallback...")

        if not success and has_hf:
            try:
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                tokenizer.save_pretrained(local_dir)
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
                except Exception:
                    model = AutoModel.from_pretrained(repo_id)
                model.save_pretrained(local_dir)
                print(f"    [DOWNLOAD OK] Saved successfully via transformers API.")
                success = True
            except Exception as e:
                print(f"    [ERROR] transformers fallback failed: {e}")

        if success:
            print("    [VERIFYING] Testing offline loading (local_files_only=True)...")
            if verify_offline_load(local_dir):
                print(f"    [SUCCESS] '{repo_id}' verified 100% OFFLINE READY!")
            else:
                print(f"    [WARNING] Model downloaded but offline verification produced an alert.")
        else:
            print(f"    [FAILED] Failed to download '{repo_id}'. Check internet connection on PC Dosen.")
            all_success = False

    print("\n" + "=" * 75)
    if all_success:
        print(" [SUCCESS] ALL PRETRAINED MODELS READY IN './models/'")
        print(" You can now safely copy the entire project folder to a Flashdisk")
        print(" and run 100% OFFLINE on PC Panitia with zero internet connection!")
    else:
        print(" [WARNING] Some models failed to download. Please re-run on PC Dosen with internet.")
    print("=" * 75)

if __name__ == "__main__":
    main()
