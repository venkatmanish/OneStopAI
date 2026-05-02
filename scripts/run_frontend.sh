#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:."
streamlit run frontend/app.py
