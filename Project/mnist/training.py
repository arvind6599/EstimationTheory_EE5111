import sys
import os
sys.path.append('/usr/local/lib/python3.7/site-packages')
import numpy as np
import cv2
import mnist_dataset

x_train, t_train, x_test, t_test = mnist_dataset.load()
