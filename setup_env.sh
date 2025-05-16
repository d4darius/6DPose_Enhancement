#!/bin/bash

set -e

echo "🔧 Setting up environment on macOS..."

# === [1] Check for Homebrew ===
if ! command -v brew &> /dev/null; then
    echo "🍺 Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# === [2] Install Python 3.10 if not available ===
if ! command -v python3.10 &> /dev/null; then
    echo "🐍 Installing Python 3.10..."
    brew install python@3.10
fi

# === [3] Create virtual environment ===
echo "📦 Creating virtual environment with Python 3.10..."
python3.10 -m venv densefusion-env

# === [4] Activate virtual environment ===
source densefusion-env/bin/activate

# === [5] Upgrade pip ===
pip install --upgrade pip

# === [6] Install PyTorch (macOS CPU version) ===
echo "🔥 Installing PyTorch for macOS (CPU-only)..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# === [7] DenseFusion & YOLOv8 deps ===
pip install open3d==0.19.0 ultralytics==8.3.129 matplotlib==3.10.3 scipy==1.15.3 seaborn==0.13.2 tqdm kaleido pyquaternion opencv-python

# === [8] Utilities ===
pip install filelock fsspec jinja2 networkx sympy typing_extensions numpy==1.26.4 wandb

# === [9] Done ===
echo "✅ Environment setup complete!"
echo "Run: source densefusion-env/bin/activate to activate the environment"
