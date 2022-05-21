import pandas as pd
import numpy as np
import torch
import tqdm


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

efficinet_logits = read2logits('efficientnet-b0_output.csv')