import argparse
from pathlib import Path

############# Deep learning Stuff #################
import clip
import pandas as pd
import yaml
from sklearn import preprocessing

####### Function Created by me ###############
from dataset import *
from train_func import *


def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['image'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    model, preprocess = clip.load(cfg['model_path'], device=device)

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = np.load(cfg['label_encoder_path'], allow_pickle=True)
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])
    train_path = train_df['file_path']
    train_labels = train_df['cultivar']
    dataset = Clip_data(image_path=train_path, preprocess=preprocess, device=device)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)

    features = clip_extract(train_loader, model, device)
    features = features.astype(np.float32)
    torch.cuda.empty_cache()
    gc.collect()
    del train_loader
    del dataset
    del model
    np.save(cfg['features_path'], features)


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)
