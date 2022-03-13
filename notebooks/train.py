from sklearn import utils
from dataset import *

from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import functional as F
from torch import nn
import os
from pathlib import Path
import argparse
import numpy as np
import random
import torch.optim as optim
import albumentations
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import os
import random
import gc
import cv2
import glob
import yaml
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument("--file", type=Path)
args = parser.parse_args()
with open(str(args.file), "r") as stream:
	cfg = yaml.safe_load(stream)


def main(cfg):
	train_df = pd.read_csv(cfg['train_file_path'])
	RANDOM_SEED = 42
	seed_everything(RANDOM_SEED)
	for i in cfg.folds:
		train = train_df[train_df['kfold'] != fold]
		valid = train_df[train_df['kfold'] == fold]
