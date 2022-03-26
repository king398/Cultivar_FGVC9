import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch


def get_train_transforms(DIM):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM, DIM),
            albumentations.Normalize(
                mean=[0.3511794, 0.37462908, 0.2873578],
                std=[0.20823358, 0.2117826, 0.16226698],
            ),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            ToTensorV2(p=1.0)
        ]
    )


def get_valid_transforms(DIM):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM, DIM),
            albumentations.Normalize(
                mean=[0.3511794, 0.37462908, 0.2873578],
                std=[0.20823358, 0.2117826, 0.16226698],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_test_transforms(DIM):
    return albumentations.Compose(
        [
            albumentations.Resize(DIM, DIM),
            albumentations.Normalize(
                mean=[0.3511794, 0.37462908, 0.2873578],
                std=[0.20823358, 0.2117826, 0.16226698],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def mixup_data(x, y, alpha=1.0, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
