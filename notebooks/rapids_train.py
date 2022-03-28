import cuml
import numpy as np
import argparse
import yaml
from pathlib import Path
import pandas as pd
from utils import *
import gc
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc
import cupy
from sklearn.model_selection import RandomizedSearchCV
from sklearnex import patch_sklearn


def main(cfg):
    patch_sklearn()
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
    param_grid = {
        'hidden_layer_sizes': np.arange(100, 10000, step=100)

    }

    rapid_model = MLPClassifier(hidden_layer_sizes=(1000), verbose=True, max_iter=10000)

    rapid_model.fit(features, train_labels)
    x = rapid_model.predict(features)

    print(acc(train_labels, x))


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)
