import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocess(path):
    img_size = 150
    img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(img_arr, (img_size, img_size))
    arr = np.array(arr)/255
    return arr


