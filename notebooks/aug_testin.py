from augmentations import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from model import *
import yaml

aug = RandAugment(4, 9, 1024)
image = cv2.imread('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/test/754146.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = aug(image)['image']
image = np.array(image).transpose()
image = image * 255
image = image.astype(np.uint8)
plt.imshow(image)
plt.show()
from accelerate import Accelerator

cfg = yaml.safe_load(open('/home/mithil/PycharmProjects/Cultivar_FGVC9/cfg/tf_efficientnet_b5_ns_colab.yaml', 'r'))
model = BaseModel(cfg)
accelerate = Accelerator()
model = accelerate.prepare(model)


model1 = accelerate.unwrap_model(model)
print(model.state_dict())
accelerate.save(model.state_dict(), '/home/mithil/PycharmProjects/Cultivar_FGVC9/models/efficientnet_b5_ns_colab.pth')
