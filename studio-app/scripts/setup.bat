@echo off
setlocal
cd /d "%~dp0"
python -m venv venv
venv\Scripts\pip install --upgrade pip
echo Windows detected -- installing onnxruntime-directml for GPU acceleration
venv\Scripts\pip install onnxruntime-directml
venv\Scripts\pip install -r requirements.txt
echo Python environment ready.
