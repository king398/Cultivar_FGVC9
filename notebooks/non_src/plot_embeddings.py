import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import preprocessing
import pandas as pd
import torch



def plot(embeds, labels, fig_path='./example.pdf'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    # plt.savefig(fig_path)


df = pd.read_csv('/home/mithil/PycharmProjects/Cultivar_FGVC9/plot_files/ensemble_logit.csv')
label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.load('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/classes.npy',
                                 allow_pickle=True)
df['cultivar'] = label_encoder.fit_transform(df['cultivar'])

df['filename'] = df['filename'].apply(lambda x: x)
df['output'] = df['output'].apply(lambda x: x.split('[')[1].split(']')[0])
outputs = df['output'].values

arr = np.zeros((len(outputs), 100))
for i in range(len(outputs)):
    out = outputs[i]
    values = out.split(',')
    for j in range(100):
        arr[i, j] = float(values[j])

x = torch.from_numpy(arr)
x = torch.softmax(x, dim=1).numpy()

p = 0
for i in x:
    i = np.max(i)
    if i < 0.3:
        p += 1
print(p)