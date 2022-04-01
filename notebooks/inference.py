# Helper Functions for Inference
import argparse
import glob
from pathlib import Path

import pandas as pd
# Deep learning Stuff
import ttach as tta
import yaml
from sklearn import preprocessing

# Function Created by me
from dataset import *
from model import *
from train_func import *


def main(cfg):
    train_df = pd.read_csv(cfg['train_file_path'])
    probabilitys = None
    seed_everything(cfg['seed'])
    gc.enable()
    device = return_device()
    label_encoder = preprocessing.LabelEncoder()
    train_df['cultivar'] = label_encoder.fit_transform(train_df['cultivar'])
    paths = glob.glob(f"{cfg['test_dir']}/*.jpeg")
    test_dataset = Cultivar_data_inference(image_path=paths,
                                           transform=get_test_transforms(cfg['image_size']))

    test_loader = DataLoader(
        test_dataset, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
    )
    ids = list(map(lambda string: string.split('/')[-1], paths))
    ids = list(map(lambda string: string.split('.')[0], ids))
    ids = list(map(lambda string: string + '.png', ids))

    for path in glob.glob(f"{cfg['model_path']}/*.pth"):
        model = BaseModelEffNet(cfg)
        model.load_state_dict(torch.load(path))
        model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform())

        model.to(device)
        model.eval()
        probablity = inference_fn(test_loader, model, cfg)

        if probabilitys is None:
            probabilitys = probablity / 5
        else:
            probabilitys += probablity / 5
        del model
        gc.collect()
        torch.cuda.empty_cache()
    preds = torch.argmax(probabilitys, 1).numpy()
    sub = pd.DataFrame({"filename": ids, "cultivar": label_encoder.inverse_transform(preds)})
    probablitys = probabilitys.numpy()
    np.save(cfg['probablity_file'], probablitys, allow_pickle=True)
    sub.to_csv(cfg['submission_file'], index=False)


if __name__ == '__main__' and '__file__' in globals():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    with open(str(args.file), "r") as stream:
        cfg = yaml.safe_load(stream)
    main(cfg)
