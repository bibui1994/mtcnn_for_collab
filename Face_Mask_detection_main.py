# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:49:18 2022

@author: habib
"""

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import smart_resize
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
import glob

test_path = "1_MTCNN_Face_Mask_Detection/Test_Images"

model = keras.models.load_model("1_MTCNN_Face_Mask_Detection/Outputs/cnn_model.h5")


def mask_detect(img):
  sns.set_theme(style="white", palette=None)
  img_ary = img_to_array(img)
  detector = MTCNN()
  faces = detector.detect_faces(img_ary);
  plt.figure(figsize=(15, 12))
  plt.imshow(img);
  ax = plt.gca()
  for face in faces:
    x1, y1, width, height = face['box']
    x2, y2 = x1+width, y1+height

    pred = model.predict(smart_resize(img_ary[y1:y2, x1:x2], (256, 256)).reshape(1, 256, 256, 3));
    #print("prediction")
    #print(pred)
    if pred[0][0] >= pred[0][1]:
      txt = 'Without Mask'
      color = 'red'
    else:
      txt = 'With Mask'
      color = 'lime'
    
    rect = Rectangle((x1, y1), width, height, fill=False, color=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x1, y1-2, txt, fontsize=14, fontweight='bold', color=color)
  plt.show()
  
  
#images = [cv2.imread(file) for file in glob.glob(test_path+"/*")]

images = [cv2.imread(test_path+"/out.png")]

print('There are {} test images'.format(len(images)))

print(test_path)

print(images)

# Resize the faces zone
scaled_images =[]
for img in images:
  dim = (256,256)
  if img.shape<(256, 256):
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('resized')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  scaled_images.append(img)

for image in scaled_images:
  mask_detect(image);