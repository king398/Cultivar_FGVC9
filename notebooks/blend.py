import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing

train_csv = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train_cultivar_mapping.csv')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(train_csv['cultivar'])

effnet = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/tf_efficientnet_b3_AUGS_ns_mixup_more_epoch_tta_probablity.npy'))

effnet_big = torch.tensor(
    np.load(
        '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/tf_efficientnet_b3_ns_cutmix_probablity.npy',
        allow_pickle=True),
)

ids = pd.read_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tf_efficientnet_b3_ns_mixup_more_epoch_tta_submission.csv')[
    'filename']

probablity = effnet * 0.2    + effnet_big * 0.8
preds = torch.argmax(probablity, 1).numpy()
sub = pd.DataFrame({"filename": ids, "cultivar": label_encoder.inverse_transform(preds)})
sub.to_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/blend/tf_efficientnet_b3_ns_mixup_more_epoch_tta_submission_weights_0_3_07.csv',
    index=False)
