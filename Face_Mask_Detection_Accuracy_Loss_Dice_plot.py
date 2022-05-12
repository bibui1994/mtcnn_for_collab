# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:00:40 2022

@author: habib
"""

import seaborn as sns
import matplotlib.pyplot as plt


from Face_Mask_Detection_model import history_cnn

# Plot accuracy, dice and loss
def history_plot(history_cnn):
  custom_params = {"axes.spines.right": False, "axes.spines.top": False}
  sns.set_theme(context='talk', style="ticks", rc=custom_params)
  fig = plt.figure(figsize=(10, 6))
  plt.plot(history_cnn.history['accuracy'])
  plt.plot(history_cnn.history['val_accuracy'])
  plt.title('Model Dice', y=1.05)
  plt.ylabel('Dice')
  plt.xlabel('Epoch')
  plt.legend(['Train Dice', 'Validation Dice'], loc = 'lower right')
  plt.grid(linestyle='-', linewidth=0.5)
  #plt.ylim([0.6, 1]); 
  plt.ylim(top=1)
  plt.xlim([0, 11])
  plt.show()
  plt.savefig("1_MTCNN_Face_Mask_Detection/figures/accuracy.png")  

def loss_plot(history_cnn):
  custom_params = {"axes.spines.right": False, "axes.spines.top": False}
  sns.set_theme(context='talk', style="ticks", rc=custom_params)
  fig = plt.figure(figsize=(10, 6))
  plt.plot(history_cnn.history['loss'])
  plt.plot(history_cnn.history['val_loss'])
  plt.title('Model Loss', y=1.05)
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train Loss', 'Validation Loss'], loc = 'upper right')
  plt.grid(linestyle='-', linewidth=0.5)
  #plt.ylim([0, 1]); 
  plt.xlim([0, 11])
  plt.show()
  plt.savefig("1_MTCNN_Face_Mask_Detection/figures/loss.png")  
  
  def dice_plot(history_cnn):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(context='talk', style="ticks", rc=custom_params)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history_cnn.history['dice_coef'])
    plt.plot(history_cnn.history['val_dice_coef'])
    plt.title('Model Dice', y=1.05)
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train Dice', 'Validation Dice'], loc = 'upper right')
    plt.grid(linestyle='-', linewidth=0.5)
    #plt.ylim([0, 1]); 
    plt.xlim([0, 11])
    plt.show()
    plt.savefig("1_MTCNN_Face_Mask_Detection/figures/dice.png")  
    
    
  history_plot(history_cnn)
  
  dice_plot(history_cnn)
  
  loss_plot(history_cnn)
