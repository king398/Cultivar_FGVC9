import pandas as pd
import os
from fastai.vision import *
from fastai.vision.data import *
from fastai.vision.all import *

import numpy as np
from fastai import *
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os
import torch

path = Path('../input/small-jpegs-fgvc')

device = torch.device("cuda:0")


def return_filpath(name, folder='/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train/'):
    path = os.path.join(folder, f'{name}')
    return path


def return_filepath_id(name, folder='/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/test/test/'):
    name = name.split('.')[0]
    path = os.path.join(folder, f'{name}.jpeg')
    return path


train_df = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train_cultivar_mapping.csv')
train_ids = train_df['image'].values
train_path = list(map(return_filpath, train_ids))
train_labels = np.zeros(len(train_ids))
test_df = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/sample_submission.csv')
test_ids = test_df['filename'].values
test_path = list(map(return_filepath_id, test_ids))
test_labels = np.ones(len(test_path))
df = pd.DataFrame.from_dict(
    {'path': np.concatenate((train_path, test_path)), 'label': np.concatenate((train_labels, test_labels))})
df = df.sample(frac=1).reset_index(drop=True)
df.head()
dls = ImageDataLoaders.from_df(df)
learn = cnn_learner(dls, models.resnet34, metrics=AccumMetric(func=RocAucBinary))
print(learn.loss_func)
lr = learn.lr_find()
learn.fit_one_cycle(1, lr_max=lr)
