#!/usr/bin/env bash
# Run ONCE:  bash setup_mac_linux.sh
set -e
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Done. Next:  source venv/bin/activate && python -m pytest tests -q"
