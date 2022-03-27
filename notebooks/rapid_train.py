#   import cuml
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
from sklearn import preprocessing
from PIL import Image

############# Deep learning Stuff #################
import torch
from torch.nn import functional as F
from torch import nn
import torch.optim as optim
import clip

####### Function Created by me ###############
from utils import *
from augmentations import *
from dataset import *
from model import *
from train_func import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/mithil/PycharmProjects/Cultivar_FGVC9/clip_models/ViT-B-16.pt", device=device)
image = preprocess(Image.open("/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train/2017-06-01__10-26-42-473.jpeg")).unsqueeze(0).to(device)

with torch.no_grad():
    image_feat = model.encode_image(image)
print(image_feat.shape)