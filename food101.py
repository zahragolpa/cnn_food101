'''this code is written to classify different types of dishes using cnn on food 101 dataset'''

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import keras.utils
import cv2
from skimage.transform import resize


DATASET_DIRECTORY = "../instagram_dp_keras/input/food-101/food-101/food-101/images/"

def show_image(image_name):
    '''plot image using this'''
    plt.imshow(image_name)
    plt.show()

def shrink_dataset(remaining_items):
    '''shrink dataset to remaining_items
    didn't test it after moving it to this function'''
    for root, dirs, files in os.walk(DATASET_DIRECTORY):
        deleted = 0
        for elements in files:
            if deleted > len(files) - remaining_items:
                break
            if elements == '.DS_Store':
                continue
            if os.path.exists(root + '/' + elements):
                os.remove(root + '/' + elements)
                deleted += 1

def save_train_data():
    '''save train data in x_train and y_train files for further use'''
    x_train = np.array(np.zeros(1))
    y_train = np.array(np.zeros(1))
    for root, dirs, files in os.walk(DATASET_DIRECTORY):
        label = str(root).split('/')[-1]
        for elements in files:
            if elements == '.DS_Store':
                continue
            x_train = np.append(x_train, elements)
            y_train = np.append(y_train, label)

    np.save('x_train', x_train)
    np.save('y_train', y_train)


x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
