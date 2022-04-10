import timm
import torch

from augmentations import *
from model import *
import cv2

cfg = {"model": 'tf_efficientnet_b3_ns', 'pretrained': False, 'in_channels': 3, 'target_size': 100,
       'snapmix_alpha': 1.0, 'image_size': 512, 'netname': 'effnet'}





image = cv2.imread('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train/2017-06-12__14-52-59-958.jpeg')
image = get_train_transforms(512)(image=image)['image']
image = image.unsqueeze(0)
model = BaseModelEffNet(cfg)
x = snapmix(image, torch.tensor(7), conf=cfg, model=model)
print(x)
