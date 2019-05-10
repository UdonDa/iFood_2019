import os
from torch.backends import cudnn
from parameter import get_parameters
from solver import start_train, get_model, get_criterion, get_optimizer
from data_loader import get_data_loader
import time


import torch
import torch.nn as nn

class Identity(nn.Module):

  def __init__(self, out_channels, stride):
    super().__init__()
    self.out_channels = out_channels
    self.stride = stride

  def forward(self, x):
    if self.stride != 1:
      x = x[:, :, ::self.stride, ::self.stride]

    if self.out_channels < x.shape[1]:
      return x[:, :self.out_channels]
    elif self.out_channels > x.shape[1]:
      return nn.functional.pad(x, (0, 0, 0, 0, 0, self.out_channels - x.shape[1]), 'constant', 0)
    else:
      return x


class ResBlock(nn.Module):

  def __init__(self, in_channels, out_channels, stride):
    super().__init__()

    self.identity = Identity(out_channels, stride)
    self.op = nn.Sequential(
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                groups=in_channels, bias=False),
      nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                groups=out_channels, bias=False),
      nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(out_channels))

  def forward(self, x):
    return self.identity(x) + self.op(x)


class ResSection(nn.Module):

  def __init__(self, blocks, in_channels, out_channels, stride):
    super().__init__()

    self.blocks = nn.Sequential(
      ResBlock(in_channels, out_channels, stride),
      *[ResBlock(out_channels, out_channels, 1) for _ in range(blocks - 1)])

  def forward(self, x):
    return self.blocks(x)


class ResNet(nn.Module):

  def __init__(self):
    super().__init__()

    channels_list = [16] + [16 * (2 ** i) for i in range(3)]

    self.input = nn.Sequential(
      nn.Conv2d(3, channels_list[0], 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(channels_list[0]))

    self.sections = nn.Sequential(*[
      ResSection(6, ic, oc, 1 if i == 0 else 2)
      for i, (ic, oc) in enumerate(zip(channels_list[:-1], channels_list[1:]))])

    self.pool = nn.Sequential(
      nn.BatchNorm2d(channels_list[-1]),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(1))

    self.output = nn.Linear(channels_list[-1], 251, bias=False)

  def forward(self, x):
    x = self.input(x)
    x = self.sections(x)
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = self.output(x)
    return x


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output, target):
  with torch.no_grad():
    return output.argmax(dim=1).eq(target).float().sum() / target.size(0)


def perform(model, criterion, loader, optimizer=None):
  loss_avg = AverageMeter()
  acc_avg = AverageMeter()

  for i, (x, t) in enumerate(loader):
    x = x.cuda(non_blocking=True)
    t = t.cuda(non_blocking=True)

    # forward
    y = model(x)

    loss = criterion(y, t)
    acc = accuracy(y, t)

    # update parameters
    if optimizer is not None:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # update results
    loss_avg.update(float(loss), x.size(0))
    acc_avg.update(float(acc), x.size(0))

    if i % 10 == 0:
        print('Loss: {:.4f}, Acc: {:.4f}'.format(loss_avg.avg, acc_avg.avg))

  return loss_avg.avg, acc_avg.avg


def main(args):
    cudnn.benchmark = True

    # model = get_model(args)
    model = ResNet().cuda()
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model)

    torch.torch.backends.cudnn.benchmark = True
    torch.torch.backends.cudnn.enabled = True

    EPOCH = 200

    train_loader, valid_loader, _, _ = get_data_loader(args)

    schesuler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)

    train_time = AverageMeter()
    valid_time = AverageMeter()
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    for epoch in range(1, EPOCH+1):
        schesuler.step()

        start_time = time.time()
        model.train()
        loss, acc = perform(model, criterion,train_loader, optimizer)
        train_loss.append(loss)
        train_acc.append(acc)
        train_time.update(time.time() - start_time)
        print('[{}] train: loss={:.4f}, accuracy={:.4f}'.format(epoch, loss, acc))

        start_time = time.time()
        model.eval()
        with torch.no_grad():
            loss, acc = perform(model, criterion, valid_loader)
        valid_loss.append(loss)
        valid_acc.append(acc)
        valid_time.update(time.time() - start_time)
        print('[{}] valid: loss={:.4f}, accuracy={:.4f}'.format(epoch, loss, acc))

    print('train time/epoch: {:.4f} sec'.format(train_time.avg))
    print('valid time/epoch: {:.4f} sec'.format(valid_time.avg))




if __name__ == '__main__':
    args = get_parameters()
    args.batch_size = 50
    main(args)