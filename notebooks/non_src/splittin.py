import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

image_size = [1024, 1024]
n = 2
x0 = int(image_size[0] / n)
y0 = int(image_size[1] / n)
image = np.array(
    cv2.imread('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train/2017-06-01__10-26-28-944.jpeg'))
image_cropped = [image[x0 * x:x0 * (x + 1), y0 * y:y0 * (y + 1)] for x in range(n) for y in range(n)]
random.shuffle(image_cropped)
plt.imshow(image)
plt.show()
for i in image_cropped:
    plt.imshow(i)
    plt.show()
