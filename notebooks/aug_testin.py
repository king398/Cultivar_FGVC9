from augmentations import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

aug = RandAugment(5, 9, 512)
image = cv2.imread('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train/2017-06-01__10-26-42-473.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = aug(image)
image = image * 255
image = image.astype(np.uint8)
plt.imshow(image)
plt.show()
print(image.shape)
