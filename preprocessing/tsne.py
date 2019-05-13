import scipy as sp
import json
from tqdm import tqdm
from tsnecuda import TSNE
import pandas as pd
from argparse import Namespace
from PIL import Image
import numpy as np

import time
from sys import exit

def get_parameters():
    args = Namespace()

    args.train_csv_path = '/host/ssd/dataset/iFood2019/train_labels.csv'
    args.val_csv_path = '/host/ssd/dataset/iFood2019/val_labels.csv'

    args.train_set_path = '/host/ssd/dataset/iFood2019/train_set'
    args.val_set_path = '/host/ssd/dataset/iFood2019/val_set'
    args.test_set_path = '/host/ssd/dataset/iFood2019/test_set'

    args.output_dir = './result_tsne'

    return args



def parse_info(csv_path):
    df = pd.read_csv(csv_path)
    img_name = df['img_name'].tolist()
    label = df['label'].tolist()
    return (np.array(img_name), np.array(label))


def load_img(root, img_path_array):
    def loading(root, path):
        img_path = '{}/{}'.format(root, path)
        img = np.array(Image.open(img_path))
        return img

    img_np_array = []
    for path in tqdm(img_path_array):
        img_np_array.append(loading(root, path))

    return np.array(img_np_array)

def train_set(args):
    img_ar, label_ar = parse_info(args.train_csv_path)
    model = TSNE(n_components=2, perplexity=30.0, theta=0.5, n_iter=1000)

    img_np_ar = load_img(args.train_set_path, img_ar)
    print(img_np_ar.shape)

    predicted = 




def main(args):
    train_set(args)



if __name__ == '__main__':
    args = get_parameters()
    main(args)