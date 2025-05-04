#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./models/denseBase/eval_linemod.py --dataset_root ../dataset/linemod/DenseFusion/Linemod_preprocessed\
  --model checkpoints/linemod/pose_model_9_0.01310166542980859.pth  #\
#  --refine_model trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth