# 6DPose_Enhancement

This is a Computer Vision project devoted to the enhancement of a 6D pose extimation model with deep machine learning models

### Initial Setup

The python version recommended is

- python 3.10.2

It is suggested to follow these steps:

- Create a virtual environment using `python3.10 -m venv .venv`
- Launch it using: `source .venv/bin/activate`
- To verify that it is using the correct version we can use `python --version`

To properly start the project it is necessary to ensure the correct installation of all the necessary libraries. To do so, we run the following command:

`pip3.10 install -r requirements.txt`

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