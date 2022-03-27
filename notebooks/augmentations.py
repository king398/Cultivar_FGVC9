import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch


def get_train_transforms(DIM):
    return A.Compose(
        [
            A.RandomResizedCrop(height=DIM, width=DIM),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Blur(p=0.1),
                A.GaussianBlur(p=0.1),
                A.MotionBlur(p=0.1),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(p=0.1),
                A.ISONoise(p=0.1),
                A.GridDropout(ratio=0.5, p=0.2),
                A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8,
                                p=0.2)
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(DIM):
    return A.Compose(
        [
            A.Resize(DIM, DIM),
            A.Normalize(
                mean=[0.3511794, 0.37462908, 0.2873578],
                std=[0.20823358, 0.2117826, 0.16226698],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_test_transforms(DIM):
    return A.Compose(
        [
            A.Resize(DIM, DIM),
            A.Normalize(
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
