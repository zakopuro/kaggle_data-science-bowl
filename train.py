import pandas as pd
import numpy as np
import random
import sys
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from load_data import load_train_data,load_data_ids
from skimage.io import imread, imshow, imread_collection, concatenate_images
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

IMG_WIDTH		= 256
IMG_HEIGHT		= 256
IMG_CHANNELS	= 3
DATA_DIR	= '/home/zakopuro/kaggle_data-science-bowl/input/'
TRAIN_PATH	= os.path.join(DATA_DIR,'stage1_train/')
TEST_PATH	= os.path.join(DATA_DIR,'stage2_test_final/')

seed = 42
random.seed = seed
np.random.seed = seed


pi = 3
x = np.linspace(0, 2*pi, 100)
y = np.sin(x)
plt.plot(x,y)
plt.show()

# X_train = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
# # img = Image.open('/home/zakopuro/kaggle_data-science-bowl/input/stage1_test/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732/images/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png')

# img = imread('/home/zakopuro/kaggle_data-science-bowl/input/stage1_test/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732/images/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png')[:,:,:IMG_CHANNELS]
# X_train[0] = img
# im_list = np.asarray(img)
# plt.imshow(im_list)
# img.show()
	# X_train,Y_train = load_train_data(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
	# train_ids = load_data_ids(TRAIN_PATH)
	# ix = random.randint(0, len(train_ids))
	# imshow(X_train[ix])
	# imshow(np.squeeze(Y_train[ix]))