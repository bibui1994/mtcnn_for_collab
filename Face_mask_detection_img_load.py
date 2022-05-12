# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:53:18 2022

@author: habib
"""

# Afficher la structure hiérarchique d'une dataset
import pandas as pd
import os # getting path
import glob
from tensorflow.keras.utils import image_dataset_from_directory, array_to_img, img_to_array, load_img
import numpy as np


# Set the path to dataset:
path = "1_MTCNN_Face_Mask_Detection/Face_Mask_Dataset"
train_path = path+"/Train"
validation_path = path+"/Validation"
test_path = path+"/Test"

path1 = "1_MTCNN_Face_Mask_Detection/Face_Mask_Dataset/"
dataset = {"Image_Path":[],"Mask_Status":[],"Data_Type":[]}
for Data_Type in os.listdir(path1):
    for Mask_Status in os.listdir(path1+"/"+Data_Type):
        for image in glob.glob(path1+Data_Type+"/"+Mask_Status+"/"+"*.png"):
            dataset["Image_Path"].append(image)
            dataset["Mask_Status"].append(Mask_Status)
            dataset["Data_Type"].append(Data_Type)
Mask_dataset = pd.DataFrame(dataset)
Mask_dataset.sample(5)

Mask_dataset.value_counts("Mask_Status")


Mask_dataset.value_counts("Data_Type")


# Obtain a tf.data.Dataset that yields batches of images from the 
# subdirectories ['WithoutMask', 'WithMask']
train_ds = image_dataset_from_directory(directory=train_path, 
                                        class_names = ['WithoutMask', 'WithMask'], 
                                        label_mode = 'categorical', seed=22)
validation_ds = image_dataset_from_directory(directory = validation_path, 
                                             class_names = ['WithoutMask', 'WithMask'], 
                                             label_mode = 'categorical', seed = 22)
test_ds = image_dataset_from_directory(directory = test_path, 
                                       class_names = ['WithoutMask', 'WithMask'], 
                                       label_mode = 'categorical', seed = 22)
     

# Extract test images and labels
def extract_images(test_ds):
  x_test = []
  y_test = []
  for x, y in test_ds.unbatch():
    x_test.append(x.numpy())
    y_test.append(y.numpy())

  x_test = np.array(x_test)
  y_test = np.array(y_test)
  
  def flat(x):
    if x[0] == 1:
      return 0
    else:
      return 1
  y_test = np.apply_along_axis(flat, 1, y_test)
  return x_test, y_test


x_test, y_test = extract_images(test_ds)