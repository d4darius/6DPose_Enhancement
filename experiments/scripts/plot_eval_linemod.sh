#!/bin/bash

set -x
set -e


# Activate virtual environment
source densefusion-env/bin/activate

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3.10 ./models/denseBase/plot_inference.py --dataset_root ../dataset/linemod/DenseFusion/Linemod_preprocessed\
  --model checkpoints/linemod/pose_model_current.pth