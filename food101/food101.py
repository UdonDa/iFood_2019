import os, sys, time
import argparse
import glob
from PIL import Image
import random
import torch
from torchvision import transforms
import numpy as np
import torch.utils.data

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
            self.img_name, self.label = parse_info(csv_path)
        else:
            with open('/host/data/dataset/food101/meta/test.txt', 'r') as f:
                self.ids = f.read().strip().split('\n')
                
        self.transform = transform
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        img_path, label = self.ids[idx].split()
        image = Image.open(img_path)
        label = int(label)
        
        if self.transform:
            ### blur (rescale: 0.3~1.3)
            #rate  = 0.3 + random.random()
            #image = image.resize([int(x // (1/rate)) for x in image.size]).resize(image.size)
            ### preprocessing
            image = self.transform(image)
        
        return image, label




def get_uecfood_dataloader(args):

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


    trainset = UECFood100Dataset(train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

    testset = UECFood100Dataset(train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    return trainloader, testloader


if __name__ == "__main__":
    
    #args = arg()
    output_dir = 'UECFOOD100'
    uecfood100_crop(output_dir)
    #trates()
    #category()
