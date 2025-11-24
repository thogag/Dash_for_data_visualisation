#!/bin/bash

cd "$(dirname "$0")"

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install -r sports_analysis/requirements.txt

cd sports_analysis
python3 mon_app/app.py

