import glob
import cv2
from joblib import Parallel, delayed
import glob
import os
import numpy as np
from tqdm import tqdm

os.makedirs('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/normalized/train', exist_ok=True)
os.makedirs('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/normalized/test', exist_ok=True)


def remove_background(filename, save_path):
    id = filename.split('/')[-1].split('.')[0]
    image = cv2.imread(filename)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([10, 0, 0])
    brown_hi = np.array([20, 255, 255])

    # Mask image to only select browns
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    # Change image to red where we found brown
    image[mask > 0] = (0, 0, 0)
    cv2.imwrite(f'{save_path}/{id}.jpeg', image)


train_files = glob.glob('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/train/*.jpeg')
test_files = glob.glob('/home/mithil/PycharmProjects/Cultivar_FGVC9/data/archive/test/*.jpeg')
res1 = Parallel(n_jobs=16, backend='threading')(delayed(
    remove_background)(i, '/home/mithil/PycharmProjects/Cultivar_FGVC9/data/normalized/train') for i in
                                                tqdm(train_files, total=len(train_files)))
res2 = Parallel(n_jobs=16, backend='threading')(delayed(
    remove_background)(i, '/home/mithil/PycharmProjects/Cultivar_FGVC9/data/normalized/test') for i in
                                                tqdm(test_files, total=len(test_files)))
