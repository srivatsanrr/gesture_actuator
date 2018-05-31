# -*- coding: utf-8 -*-
"""
Created on Wed May 30 06:48:30 2018

@author: Srivatsan
"""

from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, load_model
import numpy as np
from keras import backend as K

import cv2

im_w = 28
im_h = 28

def classtoalpha(clsno):
    return chr(clsno+65)

modelname = "LeNet5_rmsprop.h5"
model = load_model(modelname)

if K.image_data_format() == 'channels_first':
    req_shape = (1, 1,28,28)
else:
    req_shape = (1, 28, 28, 1)

hand_cascade = cv2.CascadeClassifier('hand.xml')
cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) #Lateral inversion. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (im_w, im_h))
    cv2.imshow('Reduced image', gray)
    gray= np.reshape(gray, req_shape)
    gray = gray.astype('float32')
    gray/= 255  #Normalizing pixels 
    res = np.argmax(model.predict(gray))
    op = classtoalpha(res)
    print(op)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,op, (300, 300),font,0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('img', frame)
    k = cv2.waitKey(50) & 0xff
    if(k==27):
        break

cap.release()
cv2.destroyAllWindows() 
