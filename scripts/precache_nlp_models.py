import os
import sys
import argparse

PRESETS = {
    "id": [
        "indobenchmark/indobert-base-p1",
        "indolem/indobertweet-base-uncased",
        "w11wo/indonesian-roberta-base-sentiment-classifier"
    ],
    "multi": [
        "xlm-roberta-base",
        "bert-base-multilingual-cased"
    ],
    "en": [
        "roberta-base",
        "distilbert-base-uncased"
    ],
    "all": [
        "indobenchmark/indobert-base-p1",
        "indolem/indobertweet-base-uncased",
        "w11wo/indonesian-roberta-base-sentiment-classifier",
        "xlm-roberta-base",
        "roberta-base",
        "distilbert-base-uncased"
    ]
}

def precache_model(model_name: str):
    print(f"\n========================================================")
    print(f" Pre-caching model: {model_name}")
    print(f"========================================================")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"[1/2] Downloading & caching tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"      Tokenizer ready: {tokenizer.__class__.__name__}")

        print(f"[2/2] Downloading & caching model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        print(f"      Model weights ready: {model.__class__.__name__}")
        print(f"SUCCESS: '{model_name}' is fully cached for 100% offline usage.")
        return True
    except Exception as e:
        print(f"ERROR caching '{model_name}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Pre-cache HuggingFace NLP models for offline competition readiness")
    parser.add_argument("--preset", type=str, default="id", choices=["id", "multi", "en", "all"], help="Predefined model suite")
    parser.add_argument("--model", type=str, default=None, help="Specific model repo ID to cache")
    args = parser.parse_args()

    models_to_cache = []
    if args.model:
        models_to_cache = [args.model]
    else:
        models_to_cache = PRESETS.get(args.preset, PRESETS["id"])

    print(f"Selected {len(models_to_cache)} model(s) to cache: {models_to_cache}")
    success_count = 0
    for m in models_to_cache:
        if precache_model(m):
            success_count += 1

    print(f"\nSummary: {success_count}/{len(models_to_cache)} models successfully pre-cached.")

if __name__ == "__main__":
    main()
