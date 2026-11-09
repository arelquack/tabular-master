#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "  PACKAGE FULL OFFLINE BUNDLE (RUN ON LINUX/PC DOSEN WITH INTERNET)"
echo "======================================================================"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 tidak ditemukan pada sistem PATH!"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[1/4] Membuat Virtual Environment (.venv)..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual Environment (.venv) sudah tersedia."
fi

echo ""
echo "[2/4] Menginstal dependencies dari requirements.txt..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "[3/4] Mengunduh seluruh Python Wheels (.whl) ke ./wheels/ ..."
mkdir -p wheels
"$VENV_DIR/bin/pip" download -r requirements.txt -d "./wheels"

echo ""
echo "[4/4] Mengunduh Pre-Trained NLP Models ke ./models/ ..."
"$VENV_DIR/bin/python" scripts/download_models.py

echo ""
echo "======================================================================"
echo "  MEMERIKSA KELENGKAPAN BUNDLE..."
echo "======================================================================"
"$VENV_DIR/bin/python" -c "
import os, glob
wheels = glob.glob('wheels/*.whl')
models = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))] if os.path.exists('models') else []
print(f'Total Wheels Cached : {len(wheels)}')
print(f'Total Models Cached : {len(models)} -> {models}')
if len(wheels) > 0 and len(models) >= 3:
    print('\n>>> STATUS: 100% SUKSES! BUNDLE SIAP DISALIN KE FLASHDISK! <<<')
"
echo "======================================================================"
