import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

DATA_DIR	= '/home/zakopuro/kaggle_data-science-bowl'
TRAIN_PATH	= os.path.join(DATA_DIR,'stage1_train/')
TEST_PATH	= os.path.join(DATA_DIR,'stage2_test_final/')

def load_data_ids(path):
	ids = next(os.walk(path))[1]
	return ids

def load_train_data(img_height,img_width,img_channels):
	train_ids = load_data_ids(TRAIN_PATH)
	X_train = np.zeros((len(train_ids), img_height, img_width, img_channels), dtype=np.uint8)
	Y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)
	print('Getting and resizing train images and masks ... ')
	sys.stdout.flush()
	for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
		path = TRAIN_PATH + id_
		img = imread(path + '/images/' + id_ + '.png')[:,:,:img_channels]
		img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
		X_train[n] = img
		mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
		for mask_file in next(os.walk(path + '/masks/'))[2]:
			mask_ = imread(path + '/masks/' + mask_file)
			mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant',
											preserve_range=True), axis=-1)
			mask = np.maximum(mask, mask_)
	Y_train[n] = mask
	return X_train,Y_train




if __name__ == '__main__':
	train_ids = load_data_ids(TRAIN_PATH)
	print(train_ids[0:5])