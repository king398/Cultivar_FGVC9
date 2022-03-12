import numpy as np
import torch
import timm
import cv2
import glob
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, CosineAnnealingLR
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# Metrics
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import get_valid_transforms
import matplotlib.pyplot as plt
train_path = r'F:\Pycharm_projects\UltraMNIST\data\train_patches/*.jpeg'
model_pretrained = timm.create_model('resnet50d',
                                     pretrained=False,  # model pre trained
                                     in_chans=1,  # number of channel
                                     num_classes=10,  # number of class
                                     global_pool='max').cuda()


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.cnn = model_pretrained

	def forward(self, x):
		x = self.cnn(x)
		return x


model_pretrained.load_state_dict(torch.load(r'F:\Pycharm_projects\UltraMNIST\models\best_model.pt'), strict=False)
model_pretrained.eval()
x = torch.randn(1, 1, 512, 512).cuda()


class MNIST_Data(Dataset):
	def __init__(self, image_files, aug):
		self.image_files = image_files
		self.aug = aug

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		image_path = self.image_files[idx]
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		if self.aug is not None:
			image = self.aug(image=image)['image']
		return image


dataset = MNIST_Data(image_files=[r'F:\Pycharm_projects\UltraMNIST\data\train_patches\aakrmobtqc_p1.jpeg'],
                     aug=get_valid_transforms(1024))
for i,x in enumerate(dataset):
	x = np.array(x)
	x = np.transpose(x)
	plt.imshow(x)
	plt.show()