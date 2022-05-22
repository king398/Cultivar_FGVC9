import pandas as pd
import numpy as np
import torch
import tqdm
from sklearn import preprocessing


def read2logits(name):
    output_list = []
    df = pd.read_csv(name)
    df['output'] = df['output'].apply(lambda x: x.split('[')[1].split(']')[0])
    df_output = df['output'].values.tolist()
    for i in tqdm.tqdm(range(23639)):
        output = df_output[i].split(',')
        output_list.append(torch.tensor([float(x) for x in output]).reshape(1, -1))

    output_list = torch.cat(output_list, dim=0)
    return output_list


efficinet_b7_logits = read2logits(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/final/tf_efficientnet_b7_ns_800_cutmix_newSet (is(800-_640) 7xTTa pseudo_label)__w_out.csv')
print(efficinet_b7_logits.shape)
efficinet_l_logits = read2logits(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/final/tresnet_m_640_cutmix_newSet (640_7TTa_pseudo_label)__w_out.csv')
filenames = pd.read_csv(
    '/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/final/tf_efficientnetv2_l_in21ft1k_800_cutmix_h_1(is800-_640 TTa pseudo_label)__w_out.csv')
cultivar = filenames['cultivar'].values.tolist()

probablity = efficinet_l_logits * 0.4 + efficinet_b7_logits * 0.6
probablity = torch.argmax(probablity, dim=1)
probablity = probablity.numpy()
label_to_str = {}
for i, j in zip(cultivar, probablity):
    if j not in label_to_str:
        label_to_str.update({j: i})

print(len(label_to_str))
probablity = list(map(lambda x: label_to_str[x], probablity))
print(probablity[:10])

submissions = pd.DataFrame.from_dict({'filename': filenames['filename'].values.tolist(), 'cultivar': probablity})
submissions.to_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/submissions/final/ensemble_tresnet_efficitnet.csv', index=False)
