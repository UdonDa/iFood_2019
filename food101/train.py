import os
from torch.backends import cudnn
from parameter import get_parameters
from solver import start_train as train

from solver import start_train as train_mixup



def main(args):
    cudnn.benchmark = True
    if args.mixup:
        print('=> Train start with mixup...')
        train_mixup(args)
    else:
        print('=> Train start...')
        train(args)


if __name__ == '__main__':
    args = get_parameters()
    main(args)