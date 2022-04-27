import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing

train_csv = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train_cultivar_mapping.csv')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(train_csv['cultivar'])

effnet = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/effnetv2_b3_CLAHE_augs_640_7xtta_probablity.npy'))

effnet_big = torch.tensor(
    np.load(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/tf_effnetv2_b3_cutmix_35_own_tta_meanstd_640_probablity.npy',
        allow_pickle=True),
)

ids = pd.read_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tf_efficientnet_b3_ns_mixup_more_epoch_tta_submission.csv')[
    'filename']
probablity = effnet * 0.5 + effnet_big * 0.5
preds = torch.argmax(probablity, 1).numpy()
sub = pd.DataFrame({"filename": ids, "cultivar": label_encoder.inverse_transform(preds)})
sub.to_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/blend/effnetv2_b3_diffmean_augs_640_7x_probablity_tf_effnetv2_b3_cutmix_35_own_tta_640_probablity.csv',
    index=False)
