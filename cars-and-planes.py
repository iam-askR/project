#!/usr/bin/env python
# coding: utf-8

# In[5]:


#import all necessary libraries for the CNN architecture 

import keras

import numpy as np 

from keras import backend as K 

from keras import optimizers

from keras.models import Sequential 

from keras.layers import Conv2D, MaxPooling2D 

from keras.layers import Activation, Dropout, Flatten, Dense 

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy 

from keras.preprocessing.image import ImageDataGenerator 

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import *

from matplotlib import pyplot as plt 

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

#with refernce to the location of your images; create path based on your defined parameters for the folders
train_path = 'cars-and-planes/train'
valid_path = 'cars-and-plans/valid'
test_path  = 'cars-and-planes/test'

#number of samples
nb_train_samples = 20
nb_test_samples = 10
nb_valid_samples =20
epochs = 10

if K.image_data_format() == 'channels_first': 
    input_shape = (3, 224, 224) 
else: 
    input_shape = (224, 224, 3)
    
#Models 
#build and Train a CNN 
#convolution is 2D image 
#input shape = height,width,channel dimension(rgb) 
#output filter = (3,3)
#flatten layer =from rgb to 1D tensor then fed into dense layer
#Conv2D is the layer to convolve the image into multiple images
#Activation is the activation function.
#MaxPooling2D is used to max pool the value from the given size matrix and same is used for the next 2 layers. 
#then, Flatten is used to flatten the dimensions of the image obtained after convolving it.
#Dense is used to make this a fully connected model and is the hidden layer.
#Dropout is used to avoid overfitting on the dataset.
#Dense is the output layer contains only one neuron which decide to which category image belongs.


model = Sequential() 
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Conv2D(32, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Conv2D(64, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 

model.save_weights('model_saved.h5')


# In[ ]:




