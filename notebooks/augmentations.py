import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
from ttach.base import Compose
import torch.nn.functional as F
import ttach as tta


def ten_crop_hflip_vflip_transform(crop_height, crop_width):
    return Compose([tta.HorizontalFlip(), tta.VerticalFlip(), tta.FiveCrops(crop_height, crop_width)])


def get_train_transforms(DIM):
    return A.Compose(
        [
            A.RandomResizedCrop(height=DIM, width=DIM),
            A.CLAHE(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(),
            A.CoarseDropout(),

            A.ColorJitter(brightness=0.2, contrast=0.05, saturation=0.1),

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
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_test_transforms(DIM):
    return A.Compose(
        [
            A.Resize(DIM, DIM),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_test_transforms_flip(DIM):
    return A.Compose(
        [
            A.RandomResizedCrop(DIM, DIM),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_test_transforms_shift_scale(DIM):
    return A.Compose(
        [
            A.RandomResizedCrop(DIM, DIM),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_test_transforms_brightness(DIM):
    return A.Compose(
        [
            A.Resize(DIM, DIM),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(p=1.0)
        ]
    )


def get_test_transforms_all(DIM):
    return A.Compose(
        [
            A.RandomResizedCrop(height=DIM, width=DIM),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),

            ToTensorV2(),
        ]
    )


def randaugment(dim, n=5):
    return A.Compose(
        [
            A.RandomResizedCrop(dim, dim),
            A.SomeOf([A.HorizontalFlip(),
                      A.VerticalFlip(),
                      A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                      A.CoarseDropout(),
                      A.Sharpen(),
                      A.ColorJitter(),
                      A.Equalize(),
                      A.Affine(),
                      A.InvertImg(),
                      A.Posterize(),
                      A.Solarize()], n)
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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.r
    andperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    return new_data, target, shuffled_target, lam


def get_spm(input, target, model):
    imgsize = (512, 512)
    bs = input.size(0)
    with torch.no_grad():
        output, fms = model(input)
        clsw = model.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea), 1)
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i, target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i, target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps, clslogit


def snapmix(input, target, alpha, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if True:
        wfmaps, _ = get_spm(input, target, model)
        bs = input.size(0)
        lam = np.random.beta(alpha, alpha)
        lam1 = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index, :, :]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

        area = (bby2 - bby1) * (bbx2 - bbx1)
        area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

        if area1 > 0 and area > 0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (wfmaps.sum(2).sum(1) + 1e-8)
            lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (wfmaps_b.sum(2).sum(1) + 1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1 - lam

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()
