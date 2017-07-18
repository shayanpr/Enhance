import os

import numpy as np
from keras.layers import Conv2D, Dense
from keras.layers import Input
from keras.layers import UpSampling2D
from keras.models import Model, load_model, save_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import h5py

from Model import autoEncoderGen, imSize, theShape, autoEncoderGen2
from enhance import picEnhance
from helper import Picgenerator


generator_batch = 3000
model_batch = 10



print(h5py.version)

#autoencoder, autoencoder_name, eye = autoEncoderGen('models/autoencoder.h5')

autoencoder, autoencoder_name, eye = autoEncoderGen2('models2/autoencoder.h5')
print(autoencoder.summary())


gen = Picgenerator(directory='./Data/', batch_size=model_batch, target=imSize)
print('The type is: ', type(gen))
# chkpnt = ModelCheckpoint()

# for i in range(eye, 20000):
#     x_test, y_test = next(gen)
#     print(np.shape(x_test))
#     autoencoder.fit(x_test, y_test, batch_size= model_batch, epochs=1, shuffle=True)
#     if i % 10 == 0:
#         autoencoder.save('models2/autoencoderV%d.h5' % (i+10000000))
#     print('The I is: ', i)
for i in range(eye, 20000):
    print('Epoch #: %d is underway.'%i)
    history = autoencoder.fit_generator(gen, steps_per_epoch=30607, epochs=1)
    autoencoder.save('models2/autoencoderV%d.h5' % (i+10000000))
    print('Done with epoch #: %d'%i)
    print('-----------------------------------------------------')
    print()


print('Done.')