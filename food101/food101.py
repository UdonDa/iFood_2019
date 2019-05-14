import os, sys, time
import argparse
import glob
from PIL import Image
import random
import torch
from torchvision import transforms
import numpy as np
import torch.utils.data
from parameter import get_parameters
import pandas as pd

def arg():
    parser = argparse.ArgumentParser(description='tmplate')
    #parser.add_argument('--essential', '-e', required=True, type=str)
    parser.add_argument('--option',    '-o', default=0, type=int)
    parser.add_argument('--flag',      '-f', action='store_const', const=True, default=False )
    return parser.parse_args()


def parse_info(csv_path):
        df = pd.read_csv(csv_path)
        img_name = df['img_name'].tolist()
        label = df['label'].tolist()
        return (np.array(img_name), np.array(label))

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Food101Dataset(torch.utils.data.Dataset):
    
    def __init__(self, train=True, transform=None):
        if train:
            self.img_name, self.label = parse_info('/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/preprocessing/food101/train.csv')
        else:
            self.img_name, self.label = parse_info('/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/preprocessing/food101/test.csv')
                
        self.transform = transform
        
    def __len__(self):
        return len(self.img_name)
        
    def __getitem__(self, idx):
        img, label = self.img_name[idx], self.label[idx]
        img = pil_loader(img)
        img = self.transform(img)
        
        return img, label

def get_food101_dataloader(args):

    transform_train = transforms.Compose([
        transforms.Resize(args.image_min_size),
        transforms.RandomRotation(180),
        transforms.TenCrop(args.nw_input_size),
        transforms.Lambda(lambda crops: crops[np.random.randint(len(crops))]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=0.02), # 5
        transforms.ToTensor(),
        transforms.Normalize(mean=args.pretrain_dset_mean,
                                        std=args.pretrain_dset_std),
    ])


    transform_test = transforms.Compose([
                transforms.Resize(args.image_min_size),
                transforms.TenCrop(args.nw_input_size), # TTA: TenCrops
                transforms.Lambda(lambda crops: torch.stack([
                    transforms.Normalize(mean=args.pretrain_dset_mean,std=args.pretrain_dset_std)(transforms.ToTensor()(crop)) for crop in crops])),
                ])


    trainset = Food101Dataset(train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

    testset = Food101Dataset(train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    return trainloader, testloader


if __name__ == "__main__":
    args = get_parameters()
    trainloader , testloader = get_food101_dataloader(args)
    
    from sys import exit
    for image, label in trainloader:
        print('image.size(): ', image.size())
        print('label.size(): ', label.size())

        exit()
    
