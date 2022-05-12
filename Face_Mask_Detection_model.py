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
nb_epochs= 30
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

cnn_model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

cnn_model.add(Flatten())
# cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(2, activation='sigmoid'))   # pour classification binaire 
#cnn_model.add(Dense(2, activation='softmax'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.summary()

# history_cnn = cnn_model.fit(train_ds, validation_data=validation_ds, epochs=5)


history_cnn = cnn_model.fit(
 train_ds,
 epochs=nb_epochs,
 validation_data=validation_ds)



# Save the Model
cnn_model.save("1_MTCNN_Face_Mask_Detection/Outputs/cnn_model_30_313.h5")
path_figure='1_MTCNN_Face_Mask_Detection/figures/'
def history_plot(history_cnn):
      custom_params = {"axes.spines.right": False, "axes.spines.top": False}
      sns.set_theme(context='talk', style="ticks", rc=custom_params)
      fig = plt.figure(figsize=(12, 6))
      plt.plot(history_cnn.history['accuracy'])
      plt.plot(history_cnn.history['val_accuracy'])
      plt.title('Model Dice', y=1.05)
      plt.ylabel('Dice')
      plt.xlabel('Epoch')
      plt.legend(['Train Dice', 'Validation Dice'], loc = 'lower right')
      plt.grid(linestyle='-', linewidth=0.5)
      #plt.ylim([0.6, 1]); 
      plt.ylim(top=1)
      plt.xlim([0, nb_epochs])
      plt.savefig(path_figure +  'accuracy_30_313.png',dpi=300,bbox_inches='tight')  
      #plt.show()

def loss_plot(history_cnn):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='talk', style="ticks", rc=custom_params)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(history_cnn.history['loss'])
    plt.plot(history_cnn.history['val_loss'])
    plt.title('Model Loss', y=1.05)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc = 'upper right')
    plt.grid(linestyle='-', linewidth=0.5)
    #plt.ylim([0, 1]); 
    plt.xlim([0, nb_epochs])
    plt.savefig(path_figure +  'loss_30_313.png',dpi=300,bbox_inches='tight')  
    #plt.show()
 

    
    
    
history_plot(history_cnn)
loss_plot(history_cnn)


