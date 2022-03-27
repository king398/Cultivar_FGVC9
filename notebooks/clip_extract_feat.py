import cuml
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
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

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


def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['image'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    model, preprocess = clip.load(cfg['model_path'], device=device)

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = np.load(cfg['label_encoder_path'], allow_pickle=True)
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])
    train_path = train_df['file_path'][:100]
    train_labels = train_df['cultivar'][:100]
    dataset = Clip_data(image_path=train_path, preprocess=preprocess, device=device)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)

    features = clip_extract(train_loader, model, device)
    features = features.astype(np.float32)
    torch.cuda.empty_cache()
    gc.collect()
    del train_loader
    del dataset
    del model
    rapid_model = cuml.svm.SVC()
    rapid_model.fit(features, train_labels)
    print(accuracy_score(train_labels, rapid_model.predict(features)))


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)
