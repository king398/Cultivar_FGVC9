######## Helper Functions #############
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import os
import random
import gc
import yaml
from pathlib import Path
import argparse
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import glob

from sklearn import preprocessing
############# Deep learning Stuff #################
import torch
from torch.nn import functional as F
from torch import nn
import torch.optim as optim
####### Function Created by me ###############
from utils import *
from augmentations import *
from dataset import *
from model import *
from train_func import *


def main(cfg):
	train_df = pd.read_csv(cfg['train_file_path'])
	predicted_labels = None
	seed_everything(cfg['seed'])
	gc.enable()
	device = return_device()
	label_encoder = preprocessing.LabelEncoder()
	train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])
	test_dataset = Cultivar_data_inference(image_path=glob.glob(f"{cfg['test_dir']}/*.png"),
	                                       transform=get_test_transforms(cfg['image_size']))
	test_loader = DataLoader(
		test_dataset, batch_size=cfg['batch_size'], shuffle=False,
		num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
	)

	for path in glob.glob(f"{cfg['model_path']}/*.pth"):
		model = BaseModel(cfg)
		model.load_state_dict(torch.load(path))
		model.to(device)
		model.eval()
