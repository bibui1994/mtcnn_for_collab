# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:35:25 2022

@author: habib
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import image_dataset_from_directory, array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.optimizers import Adam
from Face_mask_detection_img_load import train_ds
from Face_mask_detection_img_load import validation_ds  



#  la construction du model CNN
input_dim = (256, 256, 3)
cnn_model = Sequential()

# Normalisation (ramener la taille des zones de l'image à 256/256)
cnn_model.add(Rescaling(1./255, input_shape=input_dim)) 
# amélioration de l'arch
#cnn_model.add(RandomFlip('horizontal')) 
#cnn_model.add(RandomRotation(0.2))
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_dim))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(2, activation='sigmoid'))   # pour classification binaire 

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.summary()

# history_cnn = cnn_model.fit(train_ds, validation_data=validation_ds, epochs=5)


history_cnn = cnn_model.fit(
 train_ds,
 steps_per_epoch=100,
 epochs=30,
 validation_data=validation_ds)



# Save the Model
cnn_model.save("1_MTCNN_Face_Mask_Detection/Outputs/cnn_model.h5")