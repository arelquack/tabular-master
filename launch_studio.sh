#!/usr/bin/env bash
# ============================================================
#   TABULAR STUDIO - Competitive Machine Learning Platform
# ============================================================

# Pindah ke direktori script ini berada
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  TABULAR STUDIO - Competitive Machine Learning Platform"
echo "============================================================"
echo ""
echo "Starting FastAPI Engine on http://localhost:8000 ..."
echo "Press Ctrl+C in this console to stop the platform."
echo ""

# Matikan proses lama yang memakai port 8000 (jika ada)
OLD_PID=$(lsof -ti tcp:8000 2>/dev/null)
if [ -n "$OLD_PID" ]; then
    echo "⚠  Port 8000 sudah dipakai oleh PID $OLD_PID. Mematikan proses lama..."
    kill -9 $OLD_PID 2>/dev/null
    sleep 1
    echo "✓  Proses lama dihentikan."
    echo ""
fi

# Buka browser secara background (coba beberapa perintah yang umum di Linux)
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000" &
elif command -v gnome-open &> /dev/null; then
    gnome-open "http://localhost:8000" &
elif command -v firefox &> /dev/null; then
    firefox "http://localhost:8000" &
fi

# Jalankan server menggunakan Python dari venv
VENV_PY="$SCRIPT_DIR/spaceship-titanic/.venv/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "Error: Python environment tidak ditemukan di: $VENV_PY"
    exit 1
fi

"$VENV_PY" -m uvicorn server:app --host 127.0.0.1 --port 8000
