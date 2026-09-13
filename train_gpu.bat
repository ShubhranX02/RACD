@echo off
REM train_gpu.bat — Launch transistor RL training under Python 3.12 + CUDA PyTorch
REM
REM Usage:
REM   train_gpu.bat                        # Full auto: build surrogate + train 50k steps
REM   train_gpu.bat --steps 200000         # Longer run
REM   train_gpu.bat --skip-datagen         # Skip SPICE data gen (use existing data)
REM   train_gpu.bat --spice --steps 500    # Force pure SPICE (slow, ground truth)

SET SCRIPT_DIR=%~dp0
SET VENV=%SCRIPT_DIR%venv312\Scripts\python.exe

IF NOT EXIST "%VENV%" (
    echo [ERROR] venv312 not found. Run setup first:
    echo   py -3.12 -m venv venv312
    echo   venv312\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo   venv312\Scripts\pip install stable-baselines3 gymnasium faiss-cpu onnxruntime-directml numpy
    pause
    exit /b 1
)

echo ============================================================
echo  RACD2 GPU Training Launcher
echo ============================================================
"%VENV%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ============================================================
echo.

"%VENV%" "%SCRIPT_DIR%src\train_transistor.py" %*

echo.
echo ============================================================
echo  Training complete. Run validate_pvt.py to verify:
echo    venv312\Scripts\python src\validate_pvt.py --target 6.0
echo ============================================================
pause
