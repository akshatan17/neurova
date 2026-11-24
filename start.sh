#!/bin/sh
# Run Neurova on macOS or Linux:   ./start.sh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Setting up Neurova for the first time. This takes a minute or two..."
  python3 -m venv .venv && .venv/bin/python -m pip install --quiet --upgrade pip || { echo "Setup did not finish."; rm -rf .venv; exit 1; }
fi
# Install packages when requirements.txt is new or has changed since the last install.
if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
  echo "Installing the packages Neurova needs..."
  .venv/bin/python -m pip install -r requirements.txt && cp requirements.txt .venv/installed-requirements.txt \
    || { echo "Setup did not finish. Delete the .venv folder and try again."; exit 1; }
fi
exec .venv/bin/python run.py
