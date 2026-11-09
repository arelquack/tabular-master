#!/usr/bin/env bash
# ============================================================
#   TABULAR & NLP STUDIO - Competitive Machine Learning Platform
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  TABULAR & NLP STUDIO - Competitive Machine Learning Platform"
echo "============================================================"
echo ""

# Offline environment setup
if [ -d "$SCRIPT_DIR/models" ]; then
    export HF_HOME="$SCRIPT_DIR/models"
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "[OFFLINE READY] HuggingFace cache diatur ke folder lokal: $SCRIPT_DIR/models"
fi

# Matikan proses lama yang memakai port 8000 (jika ada)
if command -v lsof &> /dev/null; then
    OLD_PID=$(lsof -ti tcp:8000 2>/dev/null || true)
    if [ -n "$OLD_PID" ]; then
        echo "Port 8000 sudah dipakai oleh PID $OLD_PID. Mematikan proses lama..."
        kill -9 $OLD_PID 2>/dev/null || true
        sleep 1
    fi
fi

# Buka browser secara background
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000" &
elif command -v gnome-open &> /dev/null; then
    gnome-open "http://localhost:8000" &
elif command -v firefox &> /dev/null; then
    firefox "http://localhost:8000" &
fi

# Cari Python environment dengan urutan prioritas
VENV_PY=""
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/spaceship-titanic/.venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/spaceship-titanic/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/venv/bin/python"
elif command -v python3 &> /dev/null; then
    VENV_PY="$(command -v python3)"
elif command -v python &> /dev/null; then
    VENV_PY="$(command -v python)"
else
    echo "Error: Python environment tidak ditemukan. Menjalankan install_offline.sh..."
    if [ -f "$SCRIPT_DIR/install_offline.sh" ]; then
        bash "$SCRIPT_DIR/install_offline.sh"
    fi
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
fi

echo "Using Python: $VENV_PY"
echo "Starting FastAPI Engine on http://localhost:8000 ..."
echo "Press Ctrl+C in this console to stop the platform."
echo ""
"$VENV_PY" -m uvicorn server:app --host 127.0.0.1 --port 8000
