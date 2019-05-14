#!/usr/bin/env python
#coding: utf-8

import os, sys, time
import argparse
import glob
from PIL import Image
import random
import torch
from torchvision import transforms

def arg():
    parser = argparse.ArgumentParser(description='tmplate')
    #parser.add_argument('--essential', '-e', required=True, type=str)
    parser.add_argument('--option',    '-o', default=0, type=int)
    parser.add_argument('--flag',      '-f', action='store_const', const=True, default=False )
    return parser.parse_args()

import torch.utils.data
class UECFood100Dataset(torch.utils.data.Dataset):
    
    def __init__(self, train=True, transform=None):
        """
        Args:
            train : trainset or testset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if train:
            with open('/host/space0/ege-t/works/works/classification_pytorch/uecfood100/UECFOOD100/UECFOOD100_train0.txt', 'r') as f:
                self.ids = f.read().strip().split('\n')
        else:
            with open('/host/space0/ege-t/works/works/classification_pytorch/uecfood100/UECFOOD100/UECFOOD100_val0.txt', 'r') as f:
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

def uecfood100_crop(output_dir):
    
    uecfood_dir = '/export/data/dataset/UECFOOD/'
    uecfood100_dir = uecfood_dir+'UECFOOD100'
    uecfood100_imageset = uecfood_dir+'BB'

    files = glob.glob(uecfood100_dir+'/*')
    dirs = []
    for f in files:
        if os.path.isdir(f):
            dirs.append(f)
    dirs = sorted(dirs)

    ### mkdir directories
    os.system('mkdir {0}'.format(output_dir))
    for i in dirs:
        class_dir = i.strip().split('/')[-1]
        os.system('mkdir {0}/{1}'.format(output_dir,class_dir))
    
    ### cp image list
    os.system('cp {}/UECFOOD100_train0.txt {}'.format(uecfood100_imageset, output_dir))
    os.system('cp {}/UECFOOD100_val0.txt {}'.format(uecfood100_imageset, output_dir))
    
    ### crop and save images
    for i in dirs:
        with open(i+'/bb_info.txt') as f:
            bb_info = f.read().strip().split('\n')
        bb_info = map(lambda x:x.strip().split(), bb_info)
        output_path = output_dir + '/' + i.strip().split('/')[-1]
        
        for bb in bb_info[1:]:
            img_path = i + '/' + bb[0] + '.jpg'

            if True:
                bbox = map(int, bb[1:])
                img = Image.open(img_path).crop(bbox)
                img.save(output_path + '/' + bb[0] + '.jpg')
            else:
                bbox = ' '.join(bb[1:])
                with open(output_path+'/'+bb[0]+'.txt','w') as f:
                    f.write(bbox)
            
            #print(img_path, bbox)

def category():
    
    with open('/export/data/dataset/UECFOOD/UECFOOD100/category_ja_utf8.txt','r') as f:
        category_ja = f.read().strip().split('\n')
    category_ja = [i.split()[1] for i in category_ja[1:]]
        
    print(category_ja)
    
    with open('/export/data/dataset/UECFOOD/UECFOOD100/category.txt','r') as f:
        category_ja = f.read().strip().split('\n')
    category = [' '.join(i.split()[1:]) for i in category_ja[1:]]
    
    print(category)

import numpy as np
def trates():

    with open('UECFOOD100/UECFOOD100_train0.txt','r') as f:
        tmp = f.read().strip().split('\n')
    img_path = map(lambda x:x.split()[0], tmp)

    with open('UECFOOD100/UECFOOD100_val0.txt','r') as f:
        tmp = f.read().strip().split('\n')
    img_path += map(lambda x:x.split()[0], tmp)

    print(len(img_path))
    for i in img_path:
        img = np.array(Image.open(i))
        print(img.shape)

classes = ('ごはん', 'うな重', 'ピラフ', '親子丼', 'カツ丼', 'カレーライス', '寿司', 'チキンライス', 'チャーハン', '天丼', 'ビビンバ', 'トースト', 'クロワッサン', 'ロールパン', 'ぶとうパン', '惣菜パン', 'ハンバーガー', 'ピザ', 'サンドウィッチ', 'かけうどん', '天ぷらうどん', 'ざるそば', 'ラーメン', 'チャーシューメン', '天津麺', '焼きそば', 'スパゲッティ', 'お好み焼き', 'たこ焼き', 'グラタン', '野菜炒め', 'コロッケ', 'なすの油味噌', 'ほうれん草炒め', '野菜の天ぷら', '味噌汁', 'コーンスープ', 'ウィンナーのソテー', 'おでん', 'オムレツ', 'がんもどきの煮物', '餃子', 'シチュー', '魚の照り焼き', '魚のフライ', '鮭の塩焼', '鮭のムニエル', '刺身', 'さんまの塩焼', 'すき焼き', '酢豚', 'たたき', '茶碗蒸し', '天ぷら盛り合わせ', '鶏の唐揚げ', '豚カツ', '南蛮漬け', '煮魚', '肉じゃが', 'ハンバーグ', 'ビーフステーキ', '干物', '豚肉の生姜焼き', '麻婆豆腐', '焼き鳥', 'ロールキャベツ', '卵焼き', '目玉焼き', '納豆', '冷奴', '春巻き', '冷やし中華', 'チンジャオロース', '角煮', '筑前煮', '海鮮丼', 'ちらし寿司', 'たい焼き', 'エビチリ', 'ローストチキン', 'シュウマイ', 'オムライス', 'カツカレー', 'スパゲッティミートソース', 'エビフライ', 'ポテトサラダ', 'グリーンサラダ', 'マカロニサラダ', 'けんちん汁', '豚汁', '中華スープ', '牛丼', 'きんぴらごぼう', 'おにぎり', 'ピザトースト', 'つけ麺', 'ホットドッグ', 'フライドポテト', '炊き込みご飯', 'ゴーヤチャンプル')
classes_en = ('rice', 'eels on rice', 'pilaf', "chicken-'n'-egg on rice", 'pork cutlet on rice', 'beef curry', 'sushi', 'chicken rice', 'fried rice', 'tempura bowl', 'bibimbap', 'toast', 'croissant', 'roll bread', 'raisin bread', 'chip butty', 'hamburger', 'pizza', 'sandwiches', 'udon noodle', 'tempura udon', 'soba noodle', 'ramen noodle', 'beef noodle', 'tensin noodle', 'fried noodle', 'spaghetti', 'Japanese-style pancake', 'takoyaki', 'gratin', 'sauteed vegetables', 'croquette', 'grilled eggplant', 'sauteed spinach', 'vegetable tempura', 'miso soup', 'potage', 'sausage', 'oden', 'omelet', 'ganmodoki', 'jiaozi', 'stew', 'teriyaki grilled fish', 'fried fish', 'grilled salmon', 'salmon meuniere', 'sashimi', 'grilled pacific saury', 'sukiyaki', 'sweet and sour pork', 'lightly roasted fish', 'steamed egg hotchpotch', 'tempura', 'fried chicken', 'sirloin cutlet', 'nanbanzuke', 'boiled fish', 'seasoned beef with potatoes', 'hambarg steak', 'beef steak', 'dried fish', 'ginger pork saute', 'spicy chili-flavored tofu', 'yakitori', 'cabbage roll', 'rolled omelet', 'egg sunny-side up', 'fermented soybeans', 'cold tofu', 'egg roll', 'chilled noodle', 'stir-fried beef and peppers', 'simmered pork', 'boiled chicken and vegetables', 'sashimi bowl', 'sushi bowl', 'fish-shaped pancake with bean jam', 'shrimp with chill source', 'roast chicken', 'steamed meat dumpling', 'omelet with fried rice', 'cutlet curry', 'spaghetti meat sauce', 'fried shrimp', 'potato salad', 'green salad', 'macaroni salad', 'Japanese tofu and vegetable chowder', 'pork miso soup', 'chinese soup', 'beef bowl', 'kinpira-style sauteed burdock', 'rice ball', 'pizza toast', 'dipping noodles', 'hot dog', 'french fries', 'mixed rice', 'goya chanpuru')


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
