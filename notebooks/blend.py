import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing

train_csv = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train_cultivar_mapping.csv')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(train_csv['cultivar'])

effnet = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/tf_efficientnet_b5_ns_mixup_more_epoch_tta_probablity.npy'))
print(effnet.shape)
effnet_big = torch.tensor(
    np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tf_efficientnet_b4_ns_mixup_probablity.npy',
            allow_pickle=True),
)
ids = pd.read_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/tf_efficientnet_b5_ns_mixup_more_epoch_tta_submission.csv')[
    'filename']
probablity = effnet * 0.4 + effnet_big * 0.6
preds = torch.argmax(probablity, 1).numpy()
sub = pd.DataFrame({"filename": ids, "cultivar": label_encoder.inverse_transform(preds)})
sub.to_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/blend/Effnetb5_effnetb3_blend_tta_20_epoch',
    index=False)
