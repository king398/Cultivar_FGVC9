import numpy as np

probs = np.load(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/tta/effnetv2_b3_CLAHE_augs_640_7xtta_probablity.npy')
max_value = list(map(lambda x: np.max(x), probs))
low_confidence = 0
for i in max_value:
    if i < 0.9:
        low_confidence += 1
print(low_confidence)







