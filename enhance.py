import argparse
import random
from argparse import ArgumentParser

from builtins import str

from google.protobuf.unittest_custom_options_pb2 import required_enum_opt
from resampy import interp

import numpy as np
from scipy.misc import imread, imsave
import os
from keras.models import Model, load_model
import h5py
from scipy.misc.pilutil import imresize
from Model import theShape, imSize
import glob

os.makedirs('./models2', exist_ok=True)


def picEnhance(file_name, autoencoder, shape=theShape, F = 4, save_path="enhanced"):

    Sh, Sw = shape[0], shape[1]  # Shape height and shape width.
    Sh0, Sw0 = int(Sh/2), int(Sw/2)  # number of pixels to move the window in each patch. Not implemented.
    Sh2, Sw2 = Sh*F, Sw*F  # The new shape height and shape width.
    file = imread(file_name, flatten=False, mode='RGB')
    if file.shape[0] > 5000 or file.shape[1] > 5000:
        print(file_name,': File too big.', file.shape)
        return
    save_path = os.path.join(os.path.split(file_name)[0], save_path)
    os.makedirs(save_path, exist_ok=True)
    _, file_name = os.path.split(file_name)

    name = file_name.replace('.jpg', '-Enhanced.jpg')
    name2 = file_name.replace('.jpg', '-nearest.jpg')
    name3 = file_name.replace('.jpg', '-bicubic.jpg')
    name4 = file_name.replace('.jpg', '-4together.jpg')
    h, w, colorChannel = file.shape
    p, q = h // Sh, w // Sw
    t = min(p, q)
    ran = random.randint(0, t-2)
    X, Y = ran*Sh2, ran*Sw2

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

    pic4[0:2*Sh2, 0:2*Sw2, :] = nearest[X:X + 2 * Sh2, Y: Y + 2 * Sw2, :]
    pic4[3+2*Sh2:, 0:2*Sw2, :] = bicubic[X:X + 2 * Sh2, Y: Y + 2 * Sw2, :]
    pic4[0:2*Sh2, 3+2*Sw2:, :] = bilinear[X:X + 2 * Sh2, Y: Y + 2 * Sw2, :]
    pic4[3+2*Sh2:, 3+2*Sw2:, :] = picEnh[X:X + 2 * Sh2, Y: Y + 2 * Sw2, :]

    imsave(os.path.join(save_path, name), picEnh)
    imsave(os.path.join(save_path, name2), nearest)
    imsave(os.path.join(save_path, name3), bicubic)
    imsave(os.path.join(save_path, name4), pic4)

    del bicubic, picEnh, pic4, bilinear, nearest


def main(*args, **kwargs):
    image_path = kwargs.get('image_path', './test/')

    parser = argparse.ArgumentParser()
    parser.add_argument("-image_path", help="Specify the image folder.", type=str, required=False, default='./')
    args = parser.parse_args()
    print("args is : ", args)
    image_path = os.path.join(args.image_path, '*.jpg')  # This ensures all the jpg files in the path is listed.

    if os.path.exists('models2/autoencoder.h5'):
        print('loading: ' + str(sorted(os.listdir('models2/'))[-1]))
        autoencoder = load_model(os.path.join('./models2', sorted(os.listdir('models2/'))[-1]))
        print('loaded: ' + str(sorted(os.listdir('models2/'))[-1]))
        autoencoder.summary()
        print(image_path)
        pictures = glob.glob(image_path)
        print(glob.glob(str(image_path)))
        for pic in pictures:
            try:
                print(pic, ":")
                picEnhance(pic, autoencoder, theShape)
            except MemoryError:
                print("File too big.")
            except Exception as err:
                print(pic, " produced an error of the form:")
                print(err)
        if len(pictures) == 0:
            print("No pictures in this folder.")

        print('All Done.')
        print()
    else:
        print('No Model found.')

if __name__ == "__main__":
    main()