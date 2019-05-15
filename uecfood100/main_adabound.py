'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
#import torchvision.models as models

import os, sys
import argparse

import models
from utils import progress_bar, display, get_mean_and_std
from uecfood100 import *

parser = argparse.ArgumentParser(description='PyTorch UECFOOD100 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--gpu', '-g', default='0', type=str, help='gpu id')
# parser.add_argument('--net', '-n', default='resnet18', type=str, help='model')
parser.add_argument('--net', '-n', default='inceptionv4', type=str, help='model')

parser.add_argument('--out', '-o', default='', type=str, help='output')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import tensorboardX as tbx


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if 'polynet' in args.net:
    rsize, csize = 363, 331
elif 'inception' in args.net:
    rsize, csize = 331, 299
else:
    rsize, csize = 256, 224
print('rsize:{} csize:{}'.format(rsize, csize))
    
# transform_train = transforms.Compose([
#     transforms.RandomRotation(180),
#     transforms.Resize((rsize,rsize)),
#     #transforms.RandomResizedCrop(csize, scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=2),
#     transforms.RandomResizedCrop(csize, scale=(0.3, 1.0), ratio=(0.75, 1.333), interpolation=2),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2,hue=0.02), # 5
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # imagenet is better
#     #transforms.Normalize((0.5898, 0.4888, 0.3634), (0.2088, 0.2219, 0.2339)),  # uecfood100
# ])

# transform_test = transforms.Compose([
#     transforms.Resize((rsize,rsize)),
#     transforms.TenCrop(csize, vertical_flip=False),
#     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#     transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(crop) for crop in crops])),
# ])

# trainset = UECFood100Dataset(train=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)
# #print(get_mean_and_std(trainset))
# #display(trainloader, classes_en)

# testset = UECFood100Dataset(train=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
# #display(testloader, classes_en)


# print('train:{} test:{}'.format(trainset.__len__(), testset.__len__()))

# Model
print('==> Building model..')
from parameter import get_parameters
from pre_models import create_model
config = get_parameters()
trainloader, testloader = get_uecfood_dataloader(config)
net, classifier_layers = create_model(config)


# net = models.__dict__[args.net](pretrained='imagenet', num_classes=1000)
# net.num_classes = num_classes = 100
# #for p in net.parameters():
# #    p.requires_grad = False

# if 'alexnet' in args.net:
#     net.classifier = nn.Sequential(
#         nn.Dropout(),
#         nn.Linear(256 * 6 * 6, 4096),
#         nn.ReLU(inplace=True),
#         nn.Dropout(),
#         nn.Linear(4096, 4096),
#         nn.ReLU(inplace=True),
#         nn.Linear(4096, num_classes),
#     )
# elif 'resnet' in args.net:
#     if 'se_' in args.net:
#         num_ftrs = net.last_linear.in_features
#         net.last_linear = nn.Linear(num_ftrs, num_classes)
#     else:
#         num_ftrs = net.fc.in_features
#         net.fc = nn.Linear(num_ftrs, num_classes)
# elif 'vgg' in  args.net:
#     if 'fcn' not in args.net:
#         '''
#         net.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )
#         '''
#         net.classifier[6] = nn.Linear(4096, num_classes)
#     else:
#         '''
#         net.classifier = nn.Sequential(
#             nn.Conv2d(512, 4096, kernel_size=7, padding=0),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Conv2d(4096, num_classes, kernel_size=1, padding=0),
#         )
#         '''
#         net.classifier[6] = nn.Conv2d(4096, num_classes, kernel_size=1, padding=0)
# elif 'inception' in args.net:
#     if 'v4' not in args.net:
#         num_ftrs = net.AuxLogits.fc.in_features
#         net.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
#         num_ftrs = net.fc.in_features
#         net.fc = nn.Linear(num_ftrs, num_classes)
#     else:
#         num_ftrs = net.last_linear.in_features
#         net.last_linear = nn.Linear(num_ftrs, num_classes)
# elif 'squeezenet' in args.net:
#     final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
#     net.classifier = nn.Sequential(
#         nn.Dropout(p=0.5),
#         final_conv,
#         nn.ReLU(inplace=True),
#         nn.AvgPool2d(13, stride=1)
#     )
# elif 'densenet' in args.net:
#     num_ftrs = net.classifier.in_features
#     net.classifier = nn.Linear(num_ftrs, num_classes)
# elif 'senet' in args.net:
#     num_ftrs = net.last_linear.in_features
#     net.last_linear = nn.Linear(num_ftrs, num_classes)
# elif 'xception' in args.net:
#     num_ftrs = net.last_linear.in_features
#     net.last_linear = nn.Linear(num_ftrs, num_classes)
# elif 'poly' in args.net:
#     num_ftrs = net.last_linear.in_features
#     net.last_linear = nn.Linear(num_ftrs, num_classes)
# print(device)
# print(type(net))

# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
    
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(args.resume)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
criterion = nn.CrossEntropyLoss()
# if 'alexnet' in args.net:
#     classifier_layers = ['classifier.1.weight','classifier.1.bias',
#                          'classifier.4.weight','classifier.4.bias',
#                          'classifier.6.weight','classifier.6.bias']
# if 'resnet' in args.net:
#     if 'se_' in args.net:
#         classifier_layers = ['last_linear.weight','last_linear.bias']
#     else:
#         classifier_layers = ['fc.weight','fc.bias']
# elif 'vgg' in  args.net:
#     classifier_layers = ['classifier.0.weight','classifier.0.bias',
#                          'classifier.3.weight','classifier.3.bias',
#                          'classifier.6.weight','classifier.6.bias']
#     #optimizer = optim.SGD([
#     #    {'params': net.features.parameters(), 'lr': args.lr},
#     #    {'params': net.classifier.parameters(), 'lr': args.lr*10.}
#     #], momentum=0.9, weight_decay=5e-4)
# elif 'inception' in args.net:
#     if 'v4' in args.net:
#         classifier_layers = ['last_linear.weight','last_linear.bias']
#     else:
#         classifier_layers = ['AuxLogits.fc.weight','AuxLogits.fc.bias',
#                              'fc.weight','fc.bias']
# elif 'squeezenet' in args.net:
#     classifier_layers = ['classifier.1.weight','classifier.1.bias']
# elif 'densenet' in args.net:
#     classifier_layers = ['classifier.weight','classifier.bias']
# elif 'senet' in args.net:
#     classifier_layers = ['last_linear.weight','last_linear.bias']
# elif 'xception' in args.net:
#     classifier_layers = ['fc.weight','fc.bias']
# elif 'poly' in args.net:
#     classifier_layers = ['last_linear.weight','last_linear.bias']
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    # if key in classifier_layers:
    #     params += [{'params':[value],'lr':args.lr*10.}]
    params += [{'params':[value],'lr':args.lr}]
    # else:
    #     params += [{'params':[value],'lr':args.lr}] # args.lr*0.

### adabound        
from lib import *
optimizer = adabound.AdaBound(params, lr=1e-3, final_lr=0.1)
# optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            100,
                            )
    
# Training
# configure("runs/run-{}".format(args.out), flush_secs=5)
writer = tbx.SummaryWriter("runs/run-{}".format(args.out))


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """mixup criterion"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    log_step = epoch * trainloader.__len__()
    print('\nEpoch: %d' % epoch)
    #if True:
    #if epoch in [150,250]:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] *= .5
    #    log_value('lr', param_group['lr'], epoch)
    #elif epoch in [275]:
    #    for (key, value), param_group in zip(params_dict.items(),optimizer.param_groups):            
    #        if key not in classifier_layers:
    #            param_group['lr'] = args.lr*.5*.5
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, config.alpha)

        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = 0
        if type(outputs) is tuple:
            # loss = 0
            for output in outputs:
                # loss += criterion(output, targets)
                loss += mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            outputs = outputs[-1]
        else:
            # loss = criterion(outputs, targets)
            loss += mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # if batch_idx % 20 == 0:
        #     break
        
    # log_value()
    # log_value()
    writer.add_scalar('Train/Loss', train_loss/(batch_idx+1), epoch)
    writer.add_scalar('Train/Acc', 100.*correct/total, epoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            bs, ncrops, c, h, w = inputs.size()

            outputs = net(inputs.view(-1, c, h, w))
            outputs = torch.mean(outputs, dim=0, keepdim=True)
            
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            # if batch_idx % 20 == 0:
            #     break
        
    # log_value('val los', )
    # log_value('val acc', )

    writer.add_scalar('Val/Loss', test_loss/(batch_idx+1), epoch)
    writer.add_scalar('Val/Acc', 100.*correct/total, epoch)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving....')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{}_epoch_{}_acc_{}.t7'.format(args.out, epoch, acc))
        best_acc = acc

    # from sys import exit
    # exit()

for epoch in range(start_epoch, start_epoch+500):
    lr_scheduler.step()
    train(epoch)
    test(epoch)
    sys.stdout.flush()
