# -*- coding: utf-8 -*-
"""
Created on Mon May 28 03:02:28 2018

@author: Srivatsan
"""
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential 
from pandas import read_csv
import numpy as np
from keras import backend as K

nClasses = 25 #A-Y; Z is not a static gesture hence excluded
train_data = read_csv('train.csv').values
test_data = read_csv('test.csv').values

#Preparing train data 
train_labels_one_hot = train_data[:, 0]
train_labels_one_hot = to_categorical(train_labels_one_hot, 25)

#Preparing test data
test_labels = test_data[:, 0]
test_labels_onehot = to_categorical(test_labels, 25)

train_data = train_data[:, 1:]
test_data = test_data[:, 1:]

#Checking Backend config to prepare image set accordingly.
if K.image_data_format() == 'channels_first':
    train_data = np.reshape(train_data, (27455,1, 28, 28))
    test_data = np.reshape(test_data, (7172,1, 28, 28))
    input_shape = (1, 28, 28)
else:
    train_data = np.reshape(train_data, (27455, 28, 28, 1))
    test_data = np.reshape(test_data, (7172, 28, 28, 1))
    input_shape = (28, 28, 1)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

#Normalising pixel values
train_data/=255
test_data/=255

#Creating model/ Computational graph
#Graph inspired by LeNet-5 But Made some hyperparameter changes according to dataset of gestures
#Also, Dropout not added as it resulted in heavy underfitting. Appropriate dropout will be figured out if required later on. 
def createModel(modelname):
    model = Sequential()
#   Conv Layer 1
    model.add(Conv2D(32, (5,5), padding = 'same', activation = 'relu', input_shape =input_shape))
    model.add(MaxPooling2D(pool_size = (2,2)))
#   Conv Layer 2
    model.add(Conv2D(64, (5,5), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Flatten())
#   FC3 
    model.add(Dense(4*4*64, activation = 'relu'))
#   FC4 
    model.add(Dense(180, activation = 'relu'))
#   Output Layer
    model.add(Dense(nClasses, activation = 'softmax'))
    return model



#Training the network 
modelname = "LeNet5_rmsprop.h5"
model1 = createModel(modelname)
epochs = 5
batch_size = 10
#Training model
model1.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model1.fit(train_data, train_labels_one_hot, batch_size = batch_size,epochs = epochs, verbose =1, validation_data = (test_data, test_labels_onehot))
model1.evaluate(test_data, test_labels_onehot)
model1.save(modelname)