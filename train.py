import os

import numpy as np
from keras.layers import Conv2D, Dense
from keras.layers import Input
from keras.layers import UpSampling2D
from keras.models import Model, load_model, save_model
from keras.preprocessing.image import ImageDataGenerator
import h5py

from Model import autoEncoderGen, imSize, theShape, autoEncoderGen2
from enhance import picEnhance
from helper import Picgenerator

generator_batch = 3000
model_batch = 100



print(h5py.version)

#autoencoder, autoencoder_name, eye = autoEncoderGen('models/autoencoder.h5')

autoencoder, autoencoder_name, eye = autoEncoderGen2('models2/autoencoder.h5')


gen = Picgenerator(directory='./256_ObjectCategories/', batch_size=generator_batch, target=imSize)
print('The type is: ', type(gen))

for i in range(eye, 20000):
    x_test, y_test = next(gen)
    print(np.shape(x_test))
    autoencoder.fit(x_test, y_test, batch_size= model_batch, epochs=1, shuffle=True)
    if i % 10 == 0:
        autoencoder.save('models2/autoencoderV%d.h5' % (i+10000000))
    print('The I is: ', i)

picEnhance('Test.jpg', autoencoder)
print('Done.')