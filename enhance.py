import random

from resampy import interp

import numpy as np
from scipy.misc import imread, imsave
import os
from keras.models import Model, load_model
import h5py
from scipy.misc.pilutil import imresize
from Model import theShape, imSize
import glob


def picEnhance(file_name, autoencoder, shape=theShape, F = 4):

    Sh, Sw = shape[0], shape[1]  # Shape height and shape width.
    Sh0, Sw0 = int(Sh/2), int(Sw/2)  # number of pixels to move the window in each patch. Not implemented.
    Sh2, Sw2 = Sh*F, Sw*F  # The new shape height and shape width.
    file = imread(file_name, flatten=False, mode='RGB')
    name = file_name.replace('.jpg', '-Enhanced.jpg')
    name2 = file_name.replace('.jpg', '-nearest.jpg')
    name3 = file_name.replace('.jpg', '-bicubic.jpg')
    name4 = file_name.replace('.jpg', '4together.jpg')
    h, w, colorChannel = file.shape
    p, q = h // Sh, w // Sw
    t = min(p, q)
    picEnh = np.zeros((p * Sh2, q * Sw2, 3))

    x = np.zeros((p*q, Sh, Sw, 3))
    pic4 = np.zeros((4*Sh2+3, 4*Sw2+3, 3))
    for i in range(p):
        for j in range(q):
            row1, row2 = (i*Sh), (i + 1)*Sh
            col1, col2 = j*Sw, (j+1)*Sw
            x[i * q + j] = file[row1: row2, col1: col2, :]

    x = x.astype('float32')  # / 255. ## For some reason, doing this step makes the picture black.
    # (not in relu(seems like even in relu))
    y = autoencoder.predict(x, 150, verbose=1)* 255  # So does omitting this part.
    y_two = y.astype('uint8')
    for i in range(p):
        for j in range(q):
            row1, row2 = i*Sh2, (i+1)*Sh2
            col1, col2 = j*Sw2, (j+1)*Sw2
            picEnh[row1: row2, col1: col2, :] = y_two[i * q + j]


    nearest = imresize(file, size=400, interp='nearest')
    bicubic = imresize(file, size=400, interp='bicubic')
    bilinear = imresize(file, size=400, interp='bilinear')
    ran = random.randint(0, t-2)
    x, y = ran*Sh2, ran*Sw2

    pic4[0:2*Sh2, 0:2*Sw2, :] = nearest[x:x + 2 * Sh2, y: y + 2 * Sw2, :]
    pic4[3+2*Sh2:, 0:2*Sw2, :] = bicubic[x:x + 2 * Sh2, y: y + 2 * Sw2, :]
    pic4[0:2*Sh2, 3+2*Sw2:, :] = bilinear[x:x + 2 * Sh2, y: y + 2 * Sw2, :]
    pic4[3+2*Sh2:, 3+2*Sw2:, :] = picEnh[x:x + 2 * Sh2, y: y + 2 * Sw2, :]

    imsave(name, picEnh)
    imsave(name2, nearest)
    imsave(name3, bicubic)
    imsave(name4, pic4)

    del bicubic, picEnh, pic4, bilinear, nearest
    print(ran)

def main(*args):
    if os.path.exists('models2/autoencoder.h5'):
        print('loading: ' + str(sorted(os.listdir('models2/'))[-1]))
        autoencoder = load_model(os.path.join('./models2', sorted(os.listdir('models2/'))[-1]))
        print('loaded: ' + str(sorted(os.listdir('models2/'))[-1]))
        autoencoder.summary()
        for pic in glob.glob('./test/*jpg'):
            picEnhance(pic, autoencoder, theShape)
        print('All Done.')
        print()
    else:
        print('No Model found.')

if __name__ == "__main__":
    main()
