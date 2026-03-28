@echo off
setlocal
cd /d "%~dp0"
python -m venv venv
venv\Scripts\pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt
echo Python environment ready.
