import os
from torch.backends import cudnn
from parameter import get_parameters
from solver import start_train


def main(args):
    cudnn.benchmark = True

    start_train(args)


if __name__ == '__main__':
    args = get_parameters()
    main(args)