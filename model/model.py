import numpy as np
import keras
import tensorflow as tf

from keras import models, backend, Model
from keras.layers import Input, Conv2D, MaxPool2D, ZeroPadding2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Flatten, Dense, Reshape, concatenate

'''
    1000fps simple-seg net 모델
    simple-seg-net ACC : 62.70% (논문 상)
'''
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

'''
    U-NET Architecture
    목표 ACC : 80%
    -> Dense-Net으로 바꾸면서 좀더 높은 ACC 기대
'''
def unet_convBlock(x, filters, block_name):
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same', name=block_name+"conv1") (x)
    conv = Conv2D(filters, (3, 3), activation='relu', padding='same', name=block_name+"conv2") (conv)
    out = MaxPooling2D((2, 2), name=block_name+"pool") (conv)
    return conv, out

def unet_upconvBlock(x, connect_layer, filters, block_name):
    upconv = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=block_name+"upconv1") (x)
    concat = concatenate([upconv, connect_layer], axis=3, name=block_name+"concat")
    out = Conv2D(filters, (3, 3), activation='relu', padding='same', name=block_name+"conv1") (concat)
    out = Conv2D(filters, (3, 3), activation='relu', padding='same', name=block_name+"conv2") (out)
    return out

def get_basic_unet_model(input_size, depth =4, filters=8):
    input_shape = (*input_size,3)
    x = Input(input_shape,name="input")

    # Down part
    p = None
    convs = []
    for i in range(depth):
        if p is None:
            c, p = unet_convBlock(x, filters, block_name="conv{}-".format(i))
        else:
            c, p = unet_convBlock(p, filters, block_name="conv{}-".format(i))
        convs.append(c)
        filters*=2

    # Bottom part
    p = Conv2D(filters, (3, 3), activation='relu', padding='same', name='mid-1') (p)
    p = Conv2D(filters, (3, 3), activation='relu', padding='same', name='mid-2') (p)

    # Up part
    for i in range(depth-1,-1,-1):
        filters //= 2
        if i == depth-1:
            c = unet_upconvBlock(p, convs[i], filters, block_name="upconv{}-".format(i))
        else:
            c = unet_upconvBlock(c, convs[i], filters, block_name="upconv{}-".format(i))

    c = Conv2D(1, (1, 1), activation='sigmoid') (c)
    y = Reshape(input_size)(c)

    model = Model(inputs=x, outputs=y)
    return model
