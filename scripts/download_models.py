import os
import sys

MODELS = [
    {
        "repo_id": "indobenchmark/indobert-base-p1",
        "local_dir": "models/indobert-base-p1",
        "description": "Standard Indonesian BERT (Formal & General News/Articles, e.g. Kompas)"
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

def main():
    print("=" * 70)
    print(" OFFLINE MODEL PACKAGER & DOWNLOADER (FOR PC PANITIA / PC DOSEN)")
    print("=" * 70)

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

    for item in MODELS:
        repo_id = item["repo_id"]
        local_dir = item["local_dir"]
        desc = item["description"]

        print(f"\n[+] Target: '{repo_id}'")
        print(f"    Folder: '{local_dir}'")
        print(f"    Desc  : {desc}")

        if os.path.exists(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"    [SKIP] Already exists locally at '{local_dir}'.")
            continue

        print(f"    [DOWNLOADING] Fetching weights and config files...")
        success = False

        # Method 1: snapshot_download (fastest, complete, handles all files & safetensors)
        if has_hub:
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"]
                )
                print(f"    [SUCCESS] Downloaded via huggingface_hub to '{local_dir}'!")
                success = True
            except Exception as e:
                print(f"    [WARN] snapshot_download failed: {e}. Trying transformers fallback...")

        # Method 2: transformers AutoModel fallback
        if not success and has_hf:
            try:
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                tokenizer.save_pretrained(local_dir)
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
                except Exception:
                    model = AutoModel.from_pretrained(repo_id)
                model.save_pretrained(local_dir)
                print(f"    [SUCCESS] Downloaded via transformers to '{local_dir}'!")
                success = True
            except Exception as e:
                print(f"    [ERROR] transformers download failed: {e}")

        if not success:
            print(f"    [FAILED] Could not download '{repo_id}'. Check internet connection.")

    print("\n" + "=" * 70)
    print(" ALL PRETRAINED MODELS READY IN './models/'")
    print(" You can now copy the entire './models/' folder to a flashdisk")
    print(" to use offline on PC Panitia / PC Dosen without internet!")
    print("=" * 70)

if __name__ == "__main__":
    main()
