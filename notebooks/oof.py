import argparse
import glob
from pathlib import Path

import pandas as pd
############# Deep learning Stuff #################
import ttach as tta
import yaml
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

####### Function Created by me ###############
from dataset import *
from model import *
from train_func import *


def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])

    train_df['file_path'] = train_df['image'].apply(lambda x: return_filpath(x, folder=cfg['train_dir']))
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    skf = StratifiedKFold(n_splits=cfg['n_fold'], random_state=cfg['seed'], shuffle=True)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.classes_ = np.load(cfg['label_encoder_path'], allow_pickle=True)
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])
    oof_preds = None
    oof_probablity = None
    oof_ids = []
    oof_targets = []

    acc_list = []
    for fold, (trn_index, val_index) in enumerate(skf.split(train_df, train_df.cultivar)):
        print(f"Fold: {fold}")
        train = train_df.iloc[trn_index]

        valid = train_df.iloc[val_index]
        train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
        valid_path = valid['file_path']
        valid_labels = valid['cultivar']
        valid_id = valid['image']
        valid_dataset = Cultivar_data_tta_oof(image_path=valid_path,
                                              targets=valid_labels,
                                              ids=valid_id,
                                              transform=get_test_transforms(cfg['image_size']),
                                              transform_2=get_test_transforms_flip(cfg['image_size']),
                                              transform_3=get_test_transforms_shift_scale(cfg['image_size']),
                                              transform_4=get_test_transforms_brightness(cfg['image_size']),
                                              transform_5=get_test_transforms_all(cfg['image_size']),
                                              transform_6=get_test_transforms_vflip(cfg['image_size']),
                                              transform_7=get_test_transforms_crop(cfg['image_size'])
                                              )
        val_loader = DataLoader(
            valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
            num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
        )
        path = glob.glob(f"{cfg['model_dir']}/{cfg['model']}_fold{fold}*.pth")
        model = BaseModelEffNet(cfg)
        model = model.to(device)
        model.load_state_dict(torch.load(path[0]))

        ids, target, preds, probablity, accuracy = oof_fn(val_loader, model, cfg)
        print(f"Fold: {fold} Accuracy: {accuracy}")
        oof_preds = np.concatenate([oof_preds, preds]) if oof_preds is not None else preds
        oof_probablity = np.concatenate([oof_probablity, probablity]) if oof_probablity is not None else probablity
        oof_ids.extend(ids)
        oof_targets.extend(target)
        acc_list.append(accuracy)
        del model
        del val_loader
        del valid_dataset
        del ids, target, preds, probablity, accuracy
        torch.cuda.empty_cache()
        gc.collect()
    oof_pred_real = label_encoder.inverse_transform(oof_preds)
    oof_targets_real = label_encoder.inverse_transform(oof_targets)
    oof_df = pd.DataFrame.from_dict(
        {'image_id': oof_ids, 'cultivar': oof_targets_real, 'prediction': oof_pred_real, 'cultivar_int': oof_preds,
         'target_int': oof_targets})
    oof_df.to_csv(cfg['oof_file_path'], index=False)
    np.save(cfg['oof_probablity_path'], oof_probablity)
    print(f"Mean Accuracy: {np.mean(acc_list)}")


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)
