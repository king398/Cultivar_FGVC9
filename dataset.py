import os
import os.path as osp
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import cv2
from random import shuffle
from torchtoolbox.transform import Cutout, RandomErasing
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
import copy
import random
import torch
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import Sampler
import pandas as pd
from tqdm import tqdm


def loader_fold(mode, t2t=False):
    root_dir = '/home/weiqi/dataset/cvpr_cultivar/data/'
    df_all = pd.read_csv(root_dir + 'train_cultivar_mapping.csv')
    df_all.dropna(inplace=True)
    unique_cultivars = list(df_all["cultivar"].unique())

    if mode == 'train':
        df_all["file_path"] = df_all["image"].apply(lambda image: root_dir + 'train_images/' +image)
        df_all["cultivar_index"] = df_all["cultivar"].map(lambda item: unique_cultivars.index(item))
        df_all["is_exist"] = df_all["file_path"].apply(lambda file_path: os.path.exists(file_path))
        df_all = df_all[df_all.is_exist == True]

        if t2t:
            df_t2t = pd.read_csv(root_dir + 'test_to_train.csv')
            df_t2t['cultivar_index'] = df_t2t["cultivar"].map(lambda item: unique_cultivars.index(item))
            df_t2t["filename"] = df_t2t["filename"].apply(lambda image: root_dir + image)

            return df_all, df_t2t
        else:
            return df_all

    elif mode == 'test':
        folders = glob.glob((root_dir + 'test/*.png'))
        return folders, unique_cultivars

    else:
        ValueError('Error Mode')


class cultivar(Dataset):
    def __init__(self, args, mode, aug_mode, image_paths, labels=None):
       self.image_paths = image_paths
       self.labels = labels
       self.mode = mode
       self.aug_mode = aug_mode

       self.transform_val = Compose([
           A.Resize(args.img_size, args.img_size),
           A.Normalize(mean=[0.3511794, 0.37462908, 0.2873578], std=[0.20823358, 0.2117826, 0.16226698]),
           ToTensorV2(),
       ])

       self.transform_test = Compose([
           A.Resize(args.img_size, args.img_size),
           A.Normalize(mean=[0.3511794, 0.37462908, 0.2873578], std=[0.20823358, 0.2117826, 0.16226698]),
           ToTensorV2(),
       ])

       self.transform_train = A.Compose([
           A.RandomResizedCrop(height=args.img_size, width=args.img_size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CoarseDropout(),
            A.Normalize(mean=[0.3511794, 0.37462908, 0.2873578], std=[0.20823358, 0.2117826, 0.16226698]),
            ToTensorV2()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        if self.mode == 'test':
            root = self.image_paths[item]
            image = cv2.imread(root)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.transform_val(image=image)["image"]
            return root, image

        elif self.mode == 'val':
            label = self.labels[item]

            root = self.image_paths[item]
            image = cv2.imread(root)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.transform_val(image=image)["image"]
            return image, label

        elif self.mode == 'train':
            label = self.labels[item]
            root = self.image_paths[item]

            # image = Image.open(root).convert('RGB')
            # image = self.transform_train(image)
            image = cv2.imread(root)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.transform_train(image=image)["image"]
            return image, label

        else:
            ValueError('Error mode in dataloader!')

