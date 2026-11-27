@echo off
REM Run this ONCE (double-click). It makes a private Python "box" called venv and installs everything.
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Done. Now double-click run_tests_windows.bat to check that everything works.
pause
