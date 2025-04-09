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
