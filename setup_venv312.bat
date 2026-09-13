@echo off
REM setup_venv312.bat — Install all dependencies into the Python 3.12 CUDA venv
REM Run this once after creating venv312

SET SCRIPT_DIR=%~dp0
SET PIP=%SCRIPT_DIR%venv312\Scripts\pip.exe

IF NOT EXIST "%SCRIPT_DIR%venv312\Scripts\python.exe" (
    echo Creating Python 3.12 venv...
    py -3.12 -m venv venv312
)

echo Installing PyTorch with CUDA 12.1 support...
"%PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing ML and simulation packages...
"%PIP%" install stable-baselines3[extra] gymnasium numpy scipy matplotlib
"%PIP%" install faiss-cpu onnxruntime-directml streamlit
"%PIP%" install scikit-learn pandas

echo.
echo Verifying installation...
"%SCRIPT_DIR%venv312\Scripts\python.exe" -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
"%SCRIPT_DIR%venv312\Scripts\python.exe" -c "import stable_baselines3; print('sb3:', stable_baselines3.__version__)"

echo.
echo Setup complete! Use train_gpu.bat to launch GPU-accelerated training.
pause
