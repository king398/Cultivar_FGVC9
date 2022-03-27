import cuml
import numpy as np

import pandas as pd
from utils import *
import gc
from sklearn import preprocessing


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
    train_path = train_df['file_path'][:100]
    train_labels = train_df['cultivar'][:100]
    rapid_model = cuml.svm.SVC()
    rapid_model.fit(features, train_labels)
    print(accuracy_score(train_labels, rapid_model.predict(features)))
