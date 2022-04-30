import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def return_filpath(name, folder):
    path = os.path.join(folder, f'{name}')
    return path


device = torch.device('cpu')

# TODO: add GradCam ++
oof = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof.csv')
oof_probability = np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof_probablity.npy')
print(oof_probability.shape)
oof['file_path'] = oof['image_id'].apply(
    lambda x: return_filpath(x, folder='/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train'))
images_path = oof['file_path'].values
ids = {}
for i, (pred, truth, pred_int) in enumerate(zip(oof['prediction'].values, oof['cultivar'].values, oof['cultivar_int'])):
    if pred != truth:
        ids.update({i: oof['file_path'].values[i]})
print(len(ids))
for key, values in ids.items():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(1, 2)
    img = cv2.imread(values)
    ax[0].imshow(img)
    ax[0].set_title(f'Predicted: {oof["prediction"].values[key]}')
    ax[1].imshow(img)
    ax[1].set_title(f'Truth: {oof["cultivar"].values[key]}')
    break

########################################################################################################################
"""from sklearn.manifold import TSNE
from matplotlib import cm
tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(oof_probability)
# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 100
for lab in range(num_categories):
    indices = oof['cultivar_int'].values==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()
"""
