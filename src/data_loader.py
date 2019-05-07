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
#         train_tform = transforms.Compose([transforms.RandomResizedCrop(args.image_min_size,
#                                                                        scale=(1.0, 1.0),
#                                                                        scale=(0.8, 1.2),
#                                                                        scale=(0.08, 1.0),
#                                                                        ratio=(3. / 4., 4. / 3.),
#                                                                        ratio=(float(args.nw_input_size) / float(args.image_min_size),
#                                                                               float(args.image_min_size) / float(args.nw_input_size)
#                                                                              ),
#                                                                        ratio=(1.0, 1.0),
#                                                                        interpolation=Image.BILINEAR
#                                                                       ),
#         train_tform = transforms.Compose([transforms.RandomAffine(0,
#                                                                   translate=(0., 0.),
#                                                                   translate=(0.25, 0.25),
# #                                                                   scale=(3. /4., 4. / 3.),
#                                                                   scale=(1., 1.),
#                                                                   shear=0,
#                                                                   shear=20,
#                                                                   resample=PIL.Image.BILINEAR,
#                                                                   fillcolor=0),
        train_tform = transforms.Compose([
                    transforms.Resize(args.image_min_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.TenCrop(args.nw_input_size),
                    transforms.Lambda(lambda crops: crops[np.random.randint(len(crops))]),
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=args.pretrain_dset_mean,
                                        std=args.pretrain_dset_std),
                    RandomErasing(probability = args.random_erasing_p, sh = args.random_erasing_sh, r1 = args.random_erasing_r1)])

    val_tform = transforms.Compose([transforms.Resize(args.image_min_size),
                                    transforms.CenterCrop(args.nw_input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=args.pretrain_dset_mean,
                                                         std=args.pretrain_dset_std)
                                   ])
    return (train_tform, val_tform)


import torch.utils.data as data
from copy import copy
import numpy as np

def parse_info(csv_path):
        df = pd.read_csv(csv_path)
        img_name = df['img_name'].tolist()
        label = df['label'].tolist()
        return (np.array(img_name), np.array(label))

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
      
class FoodDataset(data.Dataset):
        def __init__(self, root, csv_path, num_labels=250, transform=None, target_transform=None, test=False):
            self.root = root
            if not test:
                self.img_name, self.label = parse_info(csv_path)
            else:
                self.img_name = sorted(glob('{}/*.jpg'.format(root)))
            self.num_labels = num_labels
            self.transform = transform
            self.test = test

        def __getitem__(self, index):
            if not self.test:
                img, label = self.img_name[index], self.label[index]
                img = '{}/{}'.format(self.root, img)
            else:
                img = self.img_name[index]
                # print('img path in test_dst: ', img)
            
            # Make label
            zeros = torch.zeros(self.num_labels)
            if not self.test:
                zeros[label] = 1
            
            # Make img
            img = pil_loader(img)
            img = self.transform(img)
            
            return img, zeros

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
    train_dset = FoodDataset(args.train_dir, args.train_labels_csv, args.num_labels, transform=train_tform)
    val_dset = FoodDataset(args.val_dir, args.val_labels_csv, args.num_labels, transform=val_tform)
    test_dset = FoodDataset(args.test_dir, None, args.num_labels, transform=val_tform, test=True)

    train_loader = torch.utils.data.DataLoader(train_dset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                          )
    val_loader = torch.utils.data.DataLoader(val_dset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True
                                        )

    test_loader = torch.utils.data.DataLoader(test_dset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True
                                        )
    return train_loader, val_loader, test_loader, test_dset


if __name__ == '__main__':
    from parameter import get_parameters
    args = get_parameters()
    
    train_dset, val_dset, test_dset = get_data_loader(args)

    from random import randint
    train_dset, val_dset, test_dset = iter(train_dset), iter(val_dset), iter(test_dset)
    a, b = train_dset.next()
    # imshow(torchvision.utils.make_grid(a))
    # a, b = val_dset.next()
    # imshow(torchvision.utils.make_grid(a))
    # a, b = test_dset.next()
    # imshow(torchvision.utils.make_grid(a))
    # imsave(torchvision.utils.make_grid(a))