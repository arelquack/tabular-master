#!/usr/bin/env bash
set -e

echo "========================================================"
echo "Starting Tabular & NLP Studio Nexus via Docker Compose"
echo "========================================================"

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed."
    echo "Please install Docker or run locally via Python."
    exit 1
fi

docker compose up --build -d
echo ""
echo "Nexus Studio is running at: http://localhost:8000"
echo "To view logs: docker compose logs -f"
