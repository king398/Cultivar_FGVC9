from sklearn.linear_model import RidgeCV
import pandas as pd
import numpy as np
from sklearn import preprocessing
import time
from multiprocessing import Pool

label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/classes.npy',
                                 allow_pickle=True)

effnet_v2_m = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof.csv')
seresnet_50 = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/seresnext_50_tta_oof.csv')
target = seresnet_50['target_int'].values
effnet_v2_m_probablity = np.load(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof_probablity.npy')
seresnext_50_probablity = np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/seresnext_50_probablity.npy')

print(np.argmax(seresnext_50_probablity, axis=1)[:10])
print(target[:10])
def weight(x):
    weight = np.random.random(size=2)
    weight /= np.sum(weight)
    return weight


x = list(map(weight, range(1000000)))
best_weight = None
for i in x:
    preds = i[0] * effnet_v2_m_probablity + i[1] * seresnext_50_probablity
    print(np.argmax(preds[0]))
    break