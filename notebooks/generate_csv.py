import pandas as pd
import os
train = pd.read_csv(r"F:\Pycharm_projects\Cultivar_FGVC9\data\train_cultivar_mapping.csv")
train["file_path"] = train["image"].apply(lambda image: fr"F:\Pycharm_projects\Cultivar_FGVC9\data\train/" + image)
train["is_exist"] = train["file_path"].apply(lambda file_path: os.path.exists(file_path))
print("Total Training Samples:", len(train))
train = train[train.is_exist==True]
print("Valid Training Samples:", len(train))
def return_split(x):
	return str(x).split('.')[0]
train['image'] = train['image'].apply(lambda x: return_split(x))
train.reset_index(drop=True)

train.to_csv('F:\Pycharm_projects\Cultivar_FGVC9\data/train.csv',index=False)