#!/bin/bash

set -x
set -e

# Activate virtual environment
source densefusion-env/bin/activate

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3.10 ./models/denseBase/train.py --dataset linemod\
  --dataset_root ../dataset/linemod/DenseFusion/Linemod_preprocessed