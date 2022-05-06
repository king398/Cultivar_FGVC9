import argparse
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
# Deep learning Stuff
import yaml
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch.optim import *
from accelerate import Accelerator

# Function Created by me
from dataset import *
from model import *
from train_func import *


# Passing Argument to get filepath to load our file which in our case is a yaml file


# Main Function to do all our training with help of helper functions
def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['image'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    accelerator = Accelerator()
    device = accelerator.device
    skf = StratifiedKFold(n_splits=cfg['n_fold'], random_state=cfg['seed'], shuffle=True)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = np.load(cfg['label_encoder_path'], allow_pickle=True)
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])

    for fold, (trn_index, val_index) in enumerate(skf.split(train_df, train_df.cultivar)):

        if fold in cfg['folds']:
            best_model_name = None
            best_accuracy = - np.inf

            train = train_df.iloc[trn_index]

            valid = train_df.iloc[val_index]
            train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)

            train_path = train['file_path']
            train_labels = train['cultivar']
            valid_path = valid['file_path']
            valid_labels = valid['cultivar']

            train_dataset = Cultivar_data(image_path=train_path,
                                          cfg=cfg,
                                          targets=train_labels,
                                          transform=get_train_transforms(cfg['image_size']))
            valid_dataset = Cultivar_data(image_path=valid_path,
                                          cfg=cfg,
                                          targets=valid_labels,
                                          transform=get_valid_transforms(cfg['image_size']))
            train_loader = DataLoader(
                train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )

            val_loader = DataLoader(
                valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
                num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
            )
            if cfg['triplet']:
                model = TripletModel(cfg)
            if cfg['snapmix']:
                model = Snapmix_net_effnet(cfg)
            else:
                model = BaseModel(cfg)

            model.to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = eval(cfg['optimizer'])(model.parameters(), lr=float(cfg['lr']))

            scheduler = get_scheduler(optimizer, cfg, train_loader)
            train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)

            for epoch in range(cfg['epochs']):
                train_fn(train_loader, model, criterion, optimizer, epoch, cfg, scheduler, accelerator)
                accuracy = validate_fn(val_loader, model, criterion, epoch, cfg)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if best_model_name is not None:
                        os.remove(best_model_name)
                    accelerator.save(model.state_dict(),
                                     f"{cfg['model_dir']}/{cfg['model']}_fold{fold}_epoch{epoch}_accuracy_{round(accuracy, 4)}.pth")
                    best_model_name = f"{cfg['model_dir']}/{cfg['model']}_fold{fold}_epoch{epoch}_accuracy_{round(accuracy, 4)}.pth"

            gc.collect()
            torch.cuda.empty_cache()
            del model
            del optimizer
            del scheduler
            del train_dataset
            del valid_dataset
            del train_loader
            del val_loader


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    os.makedirs(cfg['model_dir'], exist_ok=True)
    main(cfg)
