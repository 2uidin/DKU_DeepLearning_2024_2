# load MNIST image files

import numpy as np
from keras.utils import to_categorical  # type:ignore

#import pandas as pd
#import matplotlib.pyplot as plt
#import keras
 
import os     # operating system interfaces
#import shutil # high-level file operations
 
 # The path to the directory where the original dataset was uncompressed

img_dir_train = 'D:\\data\\mnist\\training'
img_dir_test = 'D:\\data\\mnist\\testing'
 
flist_train = os.listdir(img_dir_train)  # get file names
flist_test  = os.listdir(img_dir_test)  # get file names

# Preprocess the image into a 4D tensor using keras.preprocessing
from keras.preprocessing import image   # type:ignore

# load train images
X_train = np.zeros(shape=(len(flist_train), 28,28,3))
y_train = np.zeros(shape=(len(flist_train)))

for idx, fname in enumerate(flist_train):  
    img_path = os.path.join(img_dir_train, fname)
    img = image.load_img(img_path, target_size=(28,28))

    img_array_train = image.img_to_array(img)
    img_array_train2 = np.expand_dims(img_array_train, axis=0)
    X_train[idx] = img_array_train2
    y_train[idx] =  flist_train[idx][:1]

# scaling into [0, 1]
X_train = X_train / 255.0

# load test images
X_test = np.zeros(shape=(len(flist_test), 28,28,3))
y_test = np.zeros(shape=(len(flist_test)))

for idx, fname in enumerate(flist_test):  
    img_path = os.path.join(img_dir_test, fname)
    img = image.load_img(img_path, target_size=(28,28))
    img_array_test = image.img_to_array(img)
    img_array_test = np.expand_dims(img_array_test, axis=0)
    X_test[idx] = img_array_test
    y_test[idx] =  flist_test[idx][:1]

# scaling into [0, 1]
X_test = X_test / 255.0

# one hot encoding
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)



