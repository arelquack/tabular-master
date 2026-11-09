import os
import sys

def main():
    print("======================================================================")
    print("  MEMERIKSA KESIAPAN ENVIRONMENT & MODEL OFFLINE...")
    print("======================================================================")
    print(f"Python Interpreter : {sys.executable} ({sys.version.split()[0]})")

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Multi-Core Mode"
        print(f"Akselerator Compute: PyTorch {torch.__version__} | CUDA: {has_cuda} ({dev_name})")
    except Exception as e:
        print(f"Akselerator Compute: Gagal memeriksa PyTorch: {e}")

    try:
        import transformers
        print(f"Transformers Engine: {transformers.__version__}")
    except Exception as e:
        print(f"Transformers Engine: Gagal import transformers: {e}")

    try:
        import sklearn, lightgbm, xgboost, catboost, optuna, fastapi, uvicorn
        print("ML & Web Libraries : Scikit-Learn, LightGBM, XGBoost, CatBoost, Optuna, FastAPI -> OK")
    except Exception as e:
        print(f"ML & Web Libraries : Ada library yang belum terpasang: {e}")

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(script_dir, "models")
    if os.path.exists(models_dir):
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        print(f"Model Offline Siap : {len(subdirs)} model -> {subdirs}")
    else:
        print("Model Offline      : Folder ./models/ belum ada.")
    print("======================================================================")

if __name__ == "__main__":
    main()
