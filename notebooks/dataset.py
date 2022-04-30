import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image


class Cultivar_data(Dataset):

    def __init__(self, image_path, cfg, targets, transform=None):
        self.image_path = image_path
        self.cfg = cfg
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):

        image_path_single = self.image_path[idx]
        if self.cfg['in_channels'] == 1:
            image = cv2.imread(image_path_single, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path_single)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.targets[idx])
        return image, label


class Cultivar_data_inference(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image


class Clip_data(Dataset):
    def __init__(self, image_path, preprocess, device):
        self.image_path = image_path
        self.preprocess = preprocess
        self.device = device

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        return self.preprocess(Image.open(self.image_path[item]))


class NN_data(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


class Cultivar_data_oof(Dataset):
    def __init__(self, image_path, cfg, targets, ids, transform=None):
        self.image_path = image_path
        self.cfg = cfg
        self.transform = transform
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):

        image_path_single = self.image_path[idx]
        if self.cfg['in_channels'] == 1:
            image = cv2.imread(image_path_single, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path_single)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = self.targets[idx]
        id = self.ids[idx]
        return image, label, id


class Cultivar_data_inference_tta(Dataset):
    def __init__(self, image_path, transform=None, transform_2=None, transform_3=None, transform_4=None,
                 transform_5=None, transform_6=None, transform_7=None):
        self.image_path = image_path
        self.transform = transform
        self.transform_2 = transform_2
        self.transform_3 = transform_3
        self.transform_4 = transform_4
        self.transform_5 = transform_5
        self.transform_6 = transform_6
        self.transform_7 = transform_7

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)

        if self.transform is not None:
            image_1 = self.transform(image=image)['image']
        if self.transform_2 is not None:
            image_2 = self.transform_2(image=image)['image']
        if self.transform_3 is not None:
            image_3 = self.transform_3(image=image)['image']
        if self.transform_4 is not None:
            image_4 = self.transform_4(image=image)['image']
        if self.transform_5 is not None:
            image_5 = self.transform_5(image=image)['image']
        if self.transform_6 is not None:
            image_6 = self.transform_6(image=image)['image']
        if self.transform_7 is not None:
            image_7 = self.transform_7(image=image)['image']

        return image_1, image_2, image_3, image_4, image_5, image_6, image_7


class Cultivar_data_tta_oof(Dataset):
    def __init__(self, image_path, targets, ids, transform=None, transform_2=None, transform_3=None, transform_4=None,
                 transform_5=None, transform_6=None, transform_7=None):
        self.image_path = image_path
        self.targets = targets
        self.ids = ids
        self.transform = transform
        self.transform_2 = transform_2
        self.transform_3 = transform_3
        self.transform_4 = transform_4
        self.transform_5 = transform_5
        self.transform_6 = transform_6
        self.transform_7 = transform_7

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)

        if self.transform is not None:
            image_1 = self.transform(image=image)['image']
        if self.transform_2 is not None:
            image_2 = self.transform_2(image=image)['image']
        if self.transform_3 is not None:
            image_3 = self.transform_3(image=image)['image']
        if self.transform_4 is not None:
            image_4 = self.transform_4(image=image)['image']
        if self.transform_5 is not None:
            image_5 = self.transform_5(image=image)['image']
        if self.transform_6 is not None:
            image_6 = self.transform_6(image=image)['image']
        if self.transform_7 is not None:
            image_7 = self.transform_7(image=image)['image']
        label = self.targets[idx]
        id = self.ids[idx]
        return image_1, image_2, image_3, image_4, image_5, image_6, image_7, label, id
