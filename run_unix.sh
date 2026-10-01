#!/usr/bin/env bash
# KeenForge one-click launcher (Linux / macOS)
set -e
cd "$(dirname "$0")"

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[ERROR] Python 3 was not found. Install Python 3.8+ and re-run."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

if [ ! -f ".venv/.kf_ready" ]; then
  echo "Upgrading pip ..."
  python -m pip install --upgrade pip
  echo "Installing dependencies (first run only; this can take several minutes) ..."
  python -m pip install -r requirements.txt
  touch .venv/.kf_ready
fi

echo "Launching KeenForge ..."
exec python src/main.py
