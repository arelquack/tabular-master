#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  TABULAR & NLP STUDIO - AUTOMATED ENVIRONMENT SETUP"
echo "============================================================"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 tidak ditemukan pada sistem PATH!"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[1/3] Membuat Python Virtual Environment (.venv)..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/3] Virtual environment (.venv) sudah tersedia."
fi

echo ""
echo "[2/3] Menginstal dependencies dari requirements.txt..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "[3/3] Memeriksa dan mengunduh pre-trained NLP models ke folder ./models/ ..."
"$VENV_DIR/bin/python" scripts/download_models.py

echo ""
echo "============================================================"
echo "  INSTALASI DAN PRE-CACHING MODEL SELESAI DENGAN SUKSES!"
echo "  Jalankan ./launch_studio.sh untuk memulai platform."
echo "============================================================"
