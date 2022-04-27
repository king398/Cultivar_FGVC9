import numpy as np

probs = np.load(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/effnetv2_b3_CLAHE_augs_640_7xtta_probablity.npy')
for i in probs:
    print(i)
    break
