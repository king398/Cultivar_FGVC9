from sklearn.linear_model import RidgeCV
import pandas as pd
import numpy as np
from sklearn import preprocessing
import time
from multiprocessing import Pool
from joblib import Parallel, delayed
from tqdm import tqdm

label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/classes.npy',
                                 allow_pickle=True)

effnet_v2_m = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof.csv')
seresnet_50 = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/seresnext_50_tta_oof.csv')
target = seresnet_50['target_int'].values
effnet_v2_m_probablity = np.load(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/tf_efficientnetv2_m_tta_oof_probablity.npy')
seresnext_50_probablity = np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/oof/seresnext_50_probablity.npy')


def weight(x):
    weight = np.random.random(size=2)
    weight /= np.sum(weight)
    return weight


x = list(map(weight, range(10000)))
best_weight = None
best_score = -np.inf


def get_score(weight):
    preds = weight[0] * effnet_v2_m_probablity + weight[1] * seresnext_50_probablity
    preds = np.argmax(preds, axis=1)
    accuracy = np.mean(preds == target)
    return accuracy


accuracy = Parallel(n_jobs=100)(delayed(get_score)(x) for x in tqdm(x, colour="blue"))
accuracy = np.array(accuracy)
print(f"Found Best Accuracy Blend on Cv: {np.max(accuracy)} with weight {x[np.argmax(accuracy)]}")
