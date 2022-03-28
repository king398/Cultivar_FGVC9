import cuml
import numpy as np
import argparse
import yaml
from pathlib import Path
import cuml
import pandas as pd
from utils import *
import gc
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as acc
import cupy


def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['image'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    features = np.load(cfg['features_path'])

    label_encoder = preprocessing.LabelEncoder()
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])
    train_labels = train_df['cultivar']
    print(max(train_labels))
    rapid_model = cuml.svm.SVR(C=10, kernel='rbf', gamma=1)

    rapid_model.fit(features, train_labels)
    x = rapid_model.predict(features)
    x = cupy.clip(x, 0, 101)
    print(x)

    print(acc(train_labels, x))


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)
