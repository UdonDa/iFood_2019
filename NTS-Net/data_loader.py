# import torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import PIL
from PIL import Image
import os
import os.path
import numpy as np
import sys
from copy import copy
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


import torchvision.transforms as transforms
from augments import RandomErasing

import torch


def create_transforms(args):
    train_tform = None
    if args.test_overfit:
        train_tform = transforms.Compose([
                                    transforms.Resize(args.image_min_size),
                                    transforms.CenterCrop(args.nw_input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=args.pretrain_dset_mean,
                                    std=args.pretrain_dset_std)
                                         ])
    else:
        train_tform = transforms.Compose([
                    transforms.Resize(args.image_min_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.TenCrop(args.nw_input_size),
                    transforms.Lambda(lambda crops: crops[np.random.randint(len(crops))]),
                    transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=0.02),
                    transforms.RandomRotation(180, resample=PIL.Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=args.pretrain_dset_mean,
                                        std=args.pretrain_dset_std),
                    # RandomErasing(probability = args.random_erasing_p, sh = args.random_erasing_sh, r1 = args.random_erasing_r1)
                    ])

    val_tform = transforms.Compose([
                    transforms.Resize(args.image_min_size),
                    transforms.TenCrop(args.nw_input_size), # TTA: TenCrops
                    transforms.Lambda(lambda crops: torch.stack([
                        transforms.Normalize(mean=args.pretrain_dset_mean,std=args.pretrain_dset_std)(transforms.ToTensor()(crop)) for crop in crops])),
                    ])
    return (train_tform, val_tform)


import torch.utils.data as data
from copy import copy
import numpy as np

def parse_split_info(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1)
    df_1, df_2 = df[:80000], df[80000:]

    img_name_1 = df_1['img_name'].tolist()
    img_name_2 = df_2['img_name'].tolist()

    label_1 = df_1['label'].tolist()
    label_2 = df_2['label'].tolist()
    return (np.array(img_name_1), np.array(label_1)), (np.array(img_name_2), np.array(label_2))

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        # i = 0
        i = self.num_samples - 1
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

      
class FoodDataset(data.Dataset):
        def __init__(self, root, csv_path, num_labels=250, transform=None, target_transform=None, l=None):
            self.root = root
            self.img_name, self.label = l
            self.num_labels = num_labels
            self.transform = transform

        def __getitem__(self, index):
            img, label = self.img_name[index], self.label[index]
            img = '{}/{}'.format(self.root, img)
        
            # Make img
            img = pil_loader(img)
            img = self.transform(img)
            
            # return img, zeros
            return img, label

        def __len__(self):
            return len(self.img_name)

class FoodDatasetTest(data.Dataset):
        def __init__(self, root=None, num_labels=251, transform=None):
            self.root = root
            self.img_name = sorted(glob('{}/*.jpg'.format(root)))
            self.num_labels = num_labels

            self.transform = transform

        def __getitem__(self, index):
            img = self.img_name[index]
            # Make img
            img = pil_loader(img)
            img = self.transform(img)
            
            return img

        def __len__(self):
            return len(self.img_name)
        
def imshow(img):
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = denorm(img).clamp_(0, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imsave(img):
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = denorm(img).clamp_(0, 1)
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))

    pil_img_f = Image.fromarray(npimg.astype(np.uint8))
    pil_img_f.save('a.png')


def get_data_loader(args):
    train_tform, val_tform = create_transforms(args)

    train_l, train_ul = parse_split_info(args.train_labels_csv)

    train_dset = FoodDataset(args.train_dir, args.train_labels_csv, args.num_labels, transform=train_tform, l=train_l)
    train_ul_dset = FoodDataset(args.train_dir, args.train_labels_csv, args.num_labels, transform=train_tform, l=train_ul)
    val_dset = FoodDataset(args.val_dir, args.val_labels_csv, args.num_labels, transform=val_tform)
    test_dset = FoodDatasetTest(root=args.test_dir, num_labels=args.num_labels, transform=val_tform)

    train_loader = torch.utils.data.DataLoader(train_dset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                        #    pin_memory=True,
                                            sampler=InfiniteSampler(len(train_l[0]))
                                          )
    train_loader = torch.utils.data.DataLoader(train_dset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                        #    pin_memory=True,
                                            sampler=InfiniteSampler(len(train_l[0]))
                                          )
    val_loader = torch.utils.data.DataLoader(val_dset,
                                         batch_size=args.val_batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                        #  pin_memory=True
                                        )

    test_loader = torch.utils.data.DataLoader(test_dset,
                                         batch_size=args.val_batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                        #  pin_memory=True
                                        )
    return train_loader, val_loader, test_loader, test_dset


if __name__ == '__main__':
    from parameter import get_parameters
    args = get_parameters()

    # df = parse_info(args.train_labels_csv)
    df = parse_info('/Users/daichi/Downloads/ifood/train_labels.csv')