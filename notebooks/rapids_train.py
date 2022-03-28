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

    train_df['file_path'] = train_df['image'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    features = np.load(cfg['features_path'])

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = np.load(cfg['label_encoder_path'], allow_pickle=True)
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])
    train_labels = train_df['cultivar']
    nn_dataset = NN_data(features=features, labels=train_labels)
    loader = DataLoader(nn_dataset, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg['num_workers'])
    model = NN_model()
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    for i in range(cfg['epochs']):
        nn_train(model, loader, device, loss, optimizer, i)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)
