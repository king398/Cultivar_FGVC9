import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import numpy as np

def return_filpath(name, folder):
    path = os.path.join(folder, f'{name}')
    return path

# TODO: add GradCam ++
oof = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof.csv')
oof_probability = np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof_probablity.npy')
print(oof_probability.shape)
oof['file_path'] = oof['image_id'].apply(
    lambda x: return_filpath(x, folder='/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train'))
images_path = oof['file_path'].values
ids = {}
for i, (pred, truth,pred_int) in enumerate(zip(oof['prediction'].values, oof['cultivar'].values,oof['cultivar_int'])):
    if pred != truth and oof_probability[i, pred_int] > 0.5:
        ids.update({i: oof['file_path'].values[i]})
print(len(ids))
for key,values in ids.items():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(1, 2)
    img = cv2.imread(values)
    ax[0].imshow(img)
    ax[0].set_title(f'Predicted: {oof["prediction"].values[key]}')
    ax[1].imshow(img)
    ax[1].set_title(f'Truth: {oof["cultivar"].values[key]}')
    break