from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
import cv2
import os
import json
import numpy as np
from prepare import preprocess

print("Please enter file path")
loaded_model = load_model('model2.h5')
img_size = 150
input_data = []
path = input()
input_data.append(preprocess(path))
input_data = np.array(input_data)
input_data = input_data.reshape(-1,150,150,1)

def make_prediction(input_data):
    prediction_probability = loaded_model.predict(input_data)
    prediction_classes = (prediction_probability > 0.5).astype(int)
    label_dict = {0: "Pneumonia Detected", 1: "Your Lung seems normal"}
    replace_func = np.vectorize(lambda x: label_dict[x])
    new_arr = replace_func(prediction_classes)
    return new_arr

result = make_prediction(input_data)
with open('out.json', 'w') as outfile:
    json.dump(result.tolist(), outfile)


