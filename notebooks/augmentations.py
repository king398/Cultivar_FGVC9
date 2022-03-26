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


def mixup_data(x, z, y, params):
    if params['mixup_alpha'] > 0:
        lam = np.random.beta(
            params['mixup_alpha'], params['mixup_alpha']
        )
    else:
        lam = 1

    batch_size = x.size()[0]
    if params['device'].type == 'cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_z = lam * z + (1 - lam) * z[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_z, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
