# 6DPose_Enhancement

This is a Computer Vision project devoted to the enhancement of a 6D pose extimation model with deep machine learning models

Here is the updated **“Initial Setup”** section for your `README.md`, now covering **both Linux/macOS (Bash)** and **Windows (CMD)** setup scripts with virtual environments:

---

### Initial Setup

This project is tested and recommended with:

- **Python 3.10**
- **CUDA 12.1** (for GPU-enabled setups with PyTorch)

#### Automated Setup Options

You can use the provided setup scripts to create a virtual environment and install all dependencies automatically:

#### For Linux/macOS:

```bash
# From the project root directory
bash setup_env.sh
```

This will create a `.venv` folder outside the repo (in the parent directory) to avoid cluttering and activate the Python 3.10 environment.

- It also install PyTorch 2.2.2 with CUDA 12.1, DenseFusion, Ultralytics, and additional dependencies.

To activate it later:

```bash
source ../.venv/bin/activate
```

#### For Windows (CMD):

```cmd
:: From the project root directory
setup_env.bat
```

This will create a `.venv` folder inside the project that uses `python3.10` if it's available in your PATH.

- It also installs all required packages including PyTorch, Ultralytics, and related libraries.

To activate the environment later:

```cmd
.venv\Scripts\activate.bat
```

## Dataset Testing

The correct dataset loading can be verified by running the dataset_test.py file found in the dataload folder using the command

`python3.10 dataload/dataset_tests.py`

The output will be stored in the plot/testing directory

## YOLO model Training and Testing

This section describes how to train a YOLO11n model for object detection on the Linemod dataset and evaluate its performance.

In all the 3 following scripts there are many optional args, check them!

### 1. Export Dataset to YOLO Format

Before training, convert the preprocessed Linemod dataset into the format required by YOLO.

`python dataload/yolo_export.py`

### 2. Finetune the model

`python models/yolo/train.py`

### 3. Evaluate

will save a detailed report in models/yolo/test_results

`python models/yolo/test.py --weights models/yolo/runs/detect/linemod_finetune/weights/best.pt`
