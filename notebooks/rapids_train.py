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
from sklearn.model_selection import train_test_split
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
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = np.load(cfg['label_encoder_path'], allow_pickle=True)
    train_df['file_path'] = train_df['image'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])

    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()

    features = np.load(cfg['features_path'])
    features_train, features_valid, labels_train, labels_valid = train_test_split(features, train_df['cultivar'].values,
                                                                                  test_size=0.2,
                                                                                  random_state=cfg['seed'])
    print(len(features_train))
    nn_dataset_train = NN_data(features=features_train, labels=labels_train)
    nn_dataset_valid = NN_data(features=features_valid, labels=labels_valid)
    train_loader = DataLoader(nn_dataset_train, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'])
    valid_loader = DataLoader(nn_dataset_valid, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'])
    model = NN_model()

    model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    best_accuracy = - np.inf
    best_model_name = None
    for i in range(cfg['epochs']):
        nn_train(model, train_loader, device, loss, optimizer, i)
        acc = nn_valid(model, valid_loader, device, loss, i)
        if acc > best_accuracy:
            best_accuracy = acc
            if best_model_name is not None:
                os.remove(best_model_name)
            torch.save(model.state_dict(),
                       f"/home/mithil/PycharmProjects/Cultivar_FGVC9/models/nn/nn_{acc:.4f}_epoch_{i}.pth")
            best_model_name = f"/home/mithil/PycharmProjects/Cultivar_FGVC9/models/nn/nn_{acc:.4f}_epoch_{i}.pth"
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)
