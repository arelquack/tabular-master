#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "  OFFLINE ENVIRONMENT INSTALLER (PC PANITIA / ZERO INTERNET REQUIRED)"
echo "======================================================================"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 tidak ditemukan pada sistem PATH!"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[1/3] Membuat Virtual Environment lokal (.venv)..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/3] Virtual environment (.venv) sudah terdeteksi."
fi

echo ""
echo "[2/3] Menginstal dependencies dari folder lokal ./wheels/ (100% OFFLINE)..."
if [ -d "$SCRIPT_DIR/wheels" ]; then
    "$VENV_DIR/bin/pip" install --no-index --find-links="$SCRIPT_DIR/wheels" -r "$SCRIPT_DIR/requirements.txt"
else
    echo "[WARNING] Folder ./wheels/ tidak ditemukan! Mencoba online pip..."
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

echo ""
echo "[3/3] Memeriksa kesiapan Hardware (GPU/CUDA) dan Model Offline..."
"$VENV_DIR/bin/python" -c "
import sys, os
print(f'Python Interpreter : {sys.executable}')
try:
    import torch
    has_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if has_cuda else 'CPU Multi-Core Mode'
    print(f'Akselerator Compute: PyTorch {torch.__version__} | CUDA: {has_cuda} ({dev_name})')
except Exception as e:
    print('Akselerator Compute:', e)

models_dir = os.path.join('$SCRIPT_DIR', 'models')
if os.path.exists(models_dir):
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f'Model Offline Siap : {len(subdirs)} model -> {subdirs}')
"

echo ""
echo "======================================================================"
echo "  INSTALASI OFFLINE SELESAI DENGAN SUKSES!"
echo "  Jalankan ./launch_studio.sh untuk memulai platform."
echo "======================================================================"
