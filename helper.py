import tensorflow as tf
import tensorflow.contrib.keras as keras
#import tensorflow.contrib.keras.backend as K
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize


def vimresize (arr, size=(128, 128)):
    arr2 = []
    for i in range(len(arr)):
        arr2.append(imresize(arr[i], size))
    return arr2


def Picgenerator(directory, batch_size=32, target=(256, 256)):
    generator_mod = ImageDataGenerator()
    generator = generator_mod.flow_from_directory(directory=directory, batch_size=batch_size,
                                                  target_size=(target[0], target[1]),
                                                  color_mode='rgb', class_mode=None)
    while True:
        batch = generator.next()
        y = batch.astype('float32') / 255.
        y_train = y  # np.array(vimresize(y, size=target))
        x_train = np.array(vimresize(y, size=(int(target[0]*0.25), int(target[1]*0.25))))
        yield x_train, y_train
