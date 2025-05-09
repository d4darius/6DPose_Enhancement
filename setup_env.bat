@echo off
SETLOCAL ENABLEEXTENSIONS

REM Set the virtual environment directory
SET VENV_DIR=.venv

REM Check if Python 3.10 is installed
where python3.10 >nul 2>nul
IF ERRORLEVEL 1 (
    echo ❌ Python 3.10 not found. Please install Python 3.10 and ensure it's in your PATH.
    goto END
)

REM Create virtual environment if not already present
IF NOT EXIST %VENV_DIR% (
    echo 🔧 Creating virtual environment at %VENV_DIR%...
    python3.10 -m venv %VENV_DIR%
) ELSE (
    echo 🔁 Virtual environment already exists at %VENV_DIR%.
)

REM Activate the virtual environment
CALL %VENV_DIR%\Scripts\activate.bat

REM Upgrade pip
echo 🔼 Upgrading pip...
python -m pip install --upgrade pip

REM Install base PyTorch stack with CUDA 12.1
echo 🚀 Installing PyTorch stack...
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

REM Install DenseFusion + Ultralytics requirements
echo 📦 Installing DenseFusion and Ultralytics dependencies...
pip install open3d==0.19.0 ultralytics==8.3.129 matplotlib==3.10.3 scipy==1.15.3 seaborn==0.13.2 tqdm kaleido pyquaternion opencv-python

REM Install extra utilities
echo 🧰 Installing additional Python utilities...
pip install filelock fsspec jinja2 networkx sympy typing_extensions numpy==1.26.4 wandb

REM Reminder for future usage
echo.
echo ✅ Setup complete!
echo 💡 To activate the environment in the future, run:
echo    %VENV_DIR%\Scripts\activate.bat
echo.

:END
ENDLOCAL
pause