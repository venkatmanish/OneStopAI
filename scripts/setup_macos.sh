#!/usr/bin/env bash
set -euo pipefail

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

brew install postgresql@16 redis qdrant tesseract
brew services start postgresql@16
brew services start redis
brew services start qdrant

echo "Create the database/user if they do not exist:"
echo "  /opt/homebrew/opt/postgresql@16/bin/createuser -s rag_user || true"
echo "  /opt/homebrew/opt/postgresql@16/bin/createdb agentic_rag -O rag_user || true"
echo "Copy .env.example to .env and set API keys."
