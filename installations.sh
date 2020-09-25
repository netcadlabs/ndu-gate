#!/bin/bash

# sudo apt install python3-opencv

python3 -m pip install torch==1.6.0

python3 -c "import cv2; print(cv2.__version__)"