@echo off
call venv\Scripts\activate.bat
python scripts\run_demo.py --use-survey-thresholds
pause
