import matplotlib.pyplot as plt
import glob
import shutil
import numpy as np
import cv2
import os

ids = []
for i in glob.glob(r'F:\Pycharm_projects\UltraMNIST\data\train_patches/*.jpeg'):
	id = i.split('\\')[5]
	id = id.split('.')[0]
	ids.append(id)

	plt.imshow(cv2.imread(i))
	plt.show()
	try:
		label = int(input())
		shutil.move(i, fr'F:\Pycharm_projects\UltraMNIST\data\labelled_patches/{id}.jpeg')
		labels = np.load(r"F:\Pycharm_projects\UltraMNIST\data/labels.npy")
		labels = np.append(labels, label)
		np.save(r"F:\Pycharm_projects\UltraMNIST\data/labels.npy", labels)
	except:
		os.remove(i)
		print(i)

labels = np.load(r"F:\Pycharm_projects\UltraMNIST\data/labels.npy")
print(labels)
