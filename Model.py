import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
from keras.callbacks import TensorBoard
import os

from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers.merge import Concatenate




generator_batch = 3000
model_batch = 150

imSize = (128, 128)
theShape = (32, 32, 3)
theShape2 = (32, 32, 1)


def autoEncoderGen(path, input_shape=theShape):
    if os.path.exists(path):
        print('loading: ' + str(sorted(os.listdir('models/'))[-1]))
        autoencoder = load_model(os.path.join('./models', sorted(os.listdir('models/'))[-1]))
        name = str(sorted(os.listdir('models/'))[-1])
        print('loaded: ' + name)
    else:
        print('No previous model found.')
        print('Building a new model.')
        input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format

        x = UpSampling2D((2, 2))(input_img)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = x

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.save('models/autoencoder.h5')
        name = 'autoencoderV.h5'
    eye_temp = name.replace('.h5', '')
    try:
        eye = int(eye_temp.replace('autoencoderV', '')) - 10000000
    except ValueError:
        eye = 0

    return autoencoder, name, eye

def autoEncoderGen2(path, input_shape=theShape):
    if os.path.exists(path):
        print('loading: ' + str(sorted(os.listdir('models2/'))[-1]))
        autoencoder = load_model(os.path.join('./models2', sorted(os.listdir('models2/'))[-1]))
        name = str(sorted(os.listdir('models2/'))[-1])
        print('loaded: ' + name)
    else:
        print('No previous model found.')
        print('Building a new model.')
        input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format

        x = UpSampling2D((4, 4))(input_img)
        x = Conv2D(30, (2, 2), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
        x = Conv2D(24, (2, 2), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
        x = Conv2D(21, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
        x = Conv2D(9, (5, 5), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
        x = Conv2D(9, (7, 7), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
        x = Conv2D(3, (9, 9), activation='relu', padding='same', kernel_initializer='glorot_normal')(x)
        decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
        autoencoder.save('models2/autoencoder.h5')
        name = 'autoencoderV.h5'
    eye_temp = name.replace('.h5', '')
    try:
        eye = int(eye_temp.replace('autoencoderV', '')) - 10000000
    except ValueError:
        eye = 0

    return autoencoder, name, eye


def autoEncoderGen3(path, input_shape=theShape):
    if os.path.exists(path):
        print('loading: ' + str(sorted(os.listdir('models2/'))[-1]))
        autoencoder = load_model(os.path.join('./models2', sorted(os.listdir('models2/'))[-1]))
        name = str(sorted(os.listdir('models2/'))[-1])
        print('loaded: ' + name)
    else:
        print('No previous model have been found.')
        print('Building a new model.')
        input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format

        x = UpSampling2D((4, 4))(input_img)
        print(x.shape)
        xr = x[:, :, :, 0]
        xg = x[:, :, :, 1]
        xb = x[:, :, :, 2]
        print(xr.shape)
        xr = xr[:, :, :, np.newaxis]
        xg = xg[:, :, :, np.newaxis]
        xb = xb[:, :, :, np.newaxis]
        print(xr.shape)
        xr = Conv2D(12, (4, 4), activation='relu', padding='same')(xr)
        xg = Conv2D(12, (4, 4), activation='relu', padding='same')(xg)
        xb = Conv2D(12, (4, 4), activation='relu', padding='same')(xb)

        x = Concatenate((xr, xg, xg))

        print('final shape of x: ', x.shape)
        decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)


        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.save('models2/autoencoder.h5')
        name = 'autoencoderV.h5'
    eye_temp = name.replace('.h5', '')
    try:
        eye = int(eye_temp.replace('autoencoderV', '')) - 10000000
    except ValueError:
        eye = 0

    return autoencoder, name, eye


