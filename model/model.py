import numpy as np
import keras
import tensorflow as tf

from keras import models, backend, Model
from keras.layers import Input, Conv2D, MaxPool2D, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Dense, Reshape

def get_1000fps_model(input_size,filters=64):
    input_shape = (*input_size,3)
    x = Input(input_shape,name="input")
    #First Convolution Block
    block_name = "conv1-"
    inode = ZeroPadding2D(padding=(2,2),name=block_name+"padding")(x)
    inode = Conv2D(filters, (5,5), strides=(1,1), activation='relu',name=block_name+"conv")(inode)
    inode = MaxPool2D(pool_size=(3,3),strides=2,name=block_name+"pool")(inode)
    inode = BatchNormalization(name=block_name+"batchnorm")(inode)

    #Second Convolution Block
    block_name = "conv2-"
    inode = ZeroPadding2D(padding=(2,2),name=block_name+"padding")(inode)
    inode = Conv2D(filters, (5,5), strides=(1,1), activation='relu',name=block_name+"conv")(inode)
    inode = MaxPool2D(pool_size=(3,3),strides=2,name=block_name+"pool")(inode)
    inode = BatchNormalization(name=block_name+"batchnorm")(inode)

    #Third Convolution Block
    block_name = "conv3-"
    inode = Conv2D(filters, (3,3), strides=(1,1), activation='relu',name=block_name+"conv")(inode)

    #First FC layer
    block_name = "fc1"
    inode = Flatten(name='flatten')(inode)
    inode = Dense(100, activation='relu',name=block_name)(inode)

    #Second FC layer
    block_name = "fc2"
    inode = Dense(400, activation='relu',name=block_name)(inode)

    #output layer
    block_name = "out-"
    inode = Dense(input_size[0]*input_size[1], activation='sigmoid',name=block_name+"fc")(inode)

    y = Reshape(input_size,name=block_name+'reshape')(inode)
    return Model(x,y)
