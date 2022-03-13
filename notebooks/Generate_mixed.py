import numpy as np
import pandas as pd
import os
import cv2
from skimage import measure
import glob
import imutils
import matplotlib.pyplot as plt

from imutils import contours
from PIL import Image as Img
from IPython.display import Image

from joblib import Parallel, delayed
from tqdm import  tqdm
import matplotlib.pyplot as plt


class CFG:
	train_path = r'F:\Pycharm_projects\UltraMNIST\data\train/'
	test_path = r'F:\Pycharm_projects\UltraMNIST\data\test/'
	train_out_path = r'F:\Pycharm_projects\UltraMNIST\data\train_patches/'
	test_out_path = r'F:\Pycharm_projects\UltraMNIST\data\test_patches/'
	size = 512


# create patches for a given image - adapted from https://www.kaggle.com/remekkinas/step-2-find-numbers-no-model-required
def create_patches(fname, outdir):
	image = cv2.imread(fname, 0)
	blurred = cv2.GaussianBlur(image, (11, 11), 0)
	thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=4)

	labels = measure.label(thresh, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	for label in np.unique(labels):
		if label == 0:
			continue

		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if numPixels > 300:
			mask = cv2.add(mask, labelMask)

	bbox_list = []

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]
	# print(f'Found {len(cnts)} contours / numbers')
	backtorgb = cv2.cvtColor(image.astype('float32'), cv2.COLOR_GRAY2RGB)

	for (i, c) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)
		bbox_list.append([x, y, w, h])
		cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 0, 0), 5)

	# print(f'BBoxes coordinates: {bbox_list}')
	# display(Img.fromarray(backtorgb.astype(np.uint8)).resize((480,480)))
	image_concated = None
	for (i, bbox) in enumerate(bbox_list):
		bbox_img = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
		padding = 50
		bbox_img = cv2.copyMakeBorder(bbox_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

		bbox_img = cv2.resize(bbox_img, (CFG.size, CFG.size), interpolation=cv2.INTER_AREA)
		bbox_img = bbox_img[:, :, np.newaxis]

		if image_concated is None:
			image_concated = bbox_img
		else:

			image_concated = np.concatenate((image_concated, bbox_img), axis=1)
	#    display(Img.fromarray((bbox_resized).astype(np.uint8)))


	cv2.imwrite(outdir + fname.split('\\')[-1][:-5] + '.jpeg', image_concated)

	return 0


train_list = glob.glob(os.path.join(CFG.train_path, '*.jpeg'))
test_list = glob.glob(os.path.join(CFG.test_path, '*.jpeg'))
res1 = Parallel(n_jobs=8, backend='threading')(delayed(
	create_patches)(i, CFG.train_out_path) for i in tqdm(train_list, total=len(train_list)))
res2 = Parallel(n_jobs=8, backend='threading')(delayed(
	create_patches)(i, CFG.test_out_path) for i in tqdm(test_list, total=len(test_list)))
