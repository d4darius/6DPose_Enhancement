#!/bin/bash

set -e

echo "🔧 Setting up environment on macOS or Linux..."

# === [1] Check for Homebrew (macOS only) ===
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo "🍺 Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # === [2] Install Python 3.10 if not available ===
    if ! command -v python3.10 &> /dev/null; then
        echo "🐍 Installing Python 3.10..."
        brew install python@3.10
    fi

    PYTHON_BIN=python3.10
else
    # Linux fallback
    PYTHON_BIN=python3
fi

# === [3] Create virtual environment ===
echo "📦 Creating virtual environment..."
$PYTHON_BIN -m venv densefusion-env

# === [4] Activate virtual environment ===
source densefusion-env/bin/activate

# === [5] Upgrade pip ===
pip install --upgrade pip

# === [6] Install PyTorch ===
echo "🔥 Installing PyTorch 2.2.2..."

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# === [7] Detect CUDA version (if available) ===
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oE "release ([0-9]+\.[0-9]+)" | awk '{print $2}')
    echo "🧠 CUDA version detected: $CUDA_VERSION"

    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        CU_TAG="cu118"
    elif [[ "$CUDA_VERSION" == "11.7" ]]; then
        CU_TAG="cu117"
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        CU_TAG="cu121"
    else
        echo "⚠️ Unsupported or unknown CUDA version: $CUDA_VERSION — defaulting to CPU-only"
        CU_TAG="cpu"
    fi
else
    echo "❄️ No CUDA detected — installing CPU-only version"
    CU_TAG="cpu"
fi

# === [8] Install PyG dependencies ===
PYG_URL="https://data.pyg.org/whl/torch-2.2.2+${CU_TAG}.html"

echo "📦 Installing PyTorch Geometric dependencies from: $PYG_URL"

pip install torch-scatter -f "$PYG_URL"
pip install torch-sparse -f "$PYG_URL"
pip install torch-cluster -f "$PYG_URL"
pip install torch-spline-conv -f "$PYG_URL"
pip install torch-geometric

# === [9] DenseFusion & YOLOv8 deps ===
pip install open3d==0.19.0 ultralytics==8.3.129 matplotlib==3.10.3 scipy==1.15.3 seaborn==0.13.2 tqdm kaleido pyquaternion opencv-python

# === [10] Utilities ===
pip install filelock fsspec jinja2 networkx sympy typing_extensions numpy==1.26.4 wandb

# === [11] Done ===
echo "✅ Environment setup complete!"
echo "Run: source densefusion-env/bin/activate to activate the environment"
