import numpy as np
import pandas as pd

effnet_b7 = pd.read_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/others/tf_efficientnet_b7_ns_800_cutmix_newSet (is(800-_640) 7xTTa pseudo_label)__wo_out2.csv')[
    'cultivar'].values
effnet_xl = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/others/tf_efficientnetv2_xl_512.csv')[
    'cultivar'].values
x = 0
true = [effnet_b7 == effnet_xl]
for i in true:
    print(i)
    if i == np.False_:
        x += 1
