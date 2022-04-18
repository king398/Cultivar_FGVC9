from augmentations import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timm
import os
import sys
import numpy as np
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cv2
from efficientnet_pytorch import EfficientNet

import torchvision
import pdb

dimdict = {'efficientnet-b0': 1536, 'efficientnet-b1': 1536, 'efficientnet-b2': 1536, 'efficientnet-b3': 1536,
           'efficientnet-b4': 1792, 'efficientnet-b5': 2048, 'efficientnet-b6': 2304, 'efficientnet-b7': 2560}


class config:
    netname = 'efficientnet-b0'
    num_class = 100


class Snapmix_net(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model('tf_efficientnet_b0', pretrained=False)
        n_features = backbone.classifier.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats


model = Snapmix_net()
aug = get_test_transforms(512)
image = cv2.imread('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/test/526928.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = aug(image=image)['image']
image = image.unsqueeze(0)
image_2 = cv2.imread('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/test/754146.jpeg')
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

image_2 = aug(image=image_2)['image']
image_2 = image_2.unsqueeze(0)
image = torch.cat((image, image_2))
image, target, target_b, lam_a = cutmix(image, torch.tensor([3, 4]), 1)
image = np.array(image.detach().cpu())

plt.imshow((image[0].transpose() * 255).astype(np.uint8))
print(target[0])
print(target_b[0])
plt.show()
