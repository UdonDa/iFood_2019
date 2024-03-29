import os
from torch.backends import cudnn
from parameter import get_parameters
from solver import start_train as train

import argparse

parser = argparse.ArgumentParser(description='iFood 2019 challenge')
parser.add_argument('--gpu', '-g', default='0', type=str, help='gpu id')
parser.add_argument('--name', '-n', default='', type=str, help='gpu id')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main(args):
    cudnn.benchmark = True
    print('=> Train start with NTS...')
    train(args)


if __name__ == '__main__':
    args = get_parameters()
    main(args)