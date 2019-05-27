from argparse import Namespace
import os
import socket
import re
from datetime import datetime
from collections import OrderedDict
import datetime
import torch


def get_output_fname(args):
    return "%s_%s" % (get_hostname_timestamp_id(), args.arch)

def get_hostname_timestamp_id():
    return str(datetime.now())[:16].replace(' ', '-').replace(':', '').replace('-', '')

def mkdir_p(d):
    os.makedirs(d, exist_ok=True)

def save_exp_info(args):

    logfile = os.path.join(args.save_dir, 'parameters.txt')
    log_file = open(logfile, 'w')
    p = OrderedDict()
    p['arch'] = args.arch
    p['image_min_size'] = args.image_min_size
    p['nw_input_size'] = args.nw_input_size
    p['loss_type'] = args.loss_type
    p['epoch'] = args.epochs
    p['all_parameter_freeze'] = args.all_parameter_freeze
    p['mixup'] = args.mixup
    p['random_erasing']= args.random_erasing

    p['lr'] = args.lr
    p['lr_scheduler'] = args.lr_scheduler
    p['optimizer'] = args.optimizer
    if args.optimizer == 'Adam':
        p['lr'] = args.lr
        p['beta1'] = args.beta1
        p['beta2'] = args.beta2
        p['amsgrad'] = args.amsgrad
        p['weight_decay'] = args.weight_decay
    elif args.optimizer == 'Sgd':
        p['lr'] = args.lr
        p['momentum'] = args.momentum
        p['weight_decay'] = args.weight_decay
        p['nesterov'] = args.nesterov
    elif args.optimizer == 'AdaBound':
        p['lr'] = args.lr
        p['beta1'] = args.beta1
        p['beta2'] = args.beta2
        p['final_lr'] = args.final_lr
        p['gamma'] = args.gamma


    for key, val in p.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


def get_parameters():

    args = Namespace()
    """Dataset path"""
    # args.data_dir = '/host/space/horita-d/dataset/ifoodchallenge2019'
    args.data_dir = '/export/ssd/dataset/iFood2019'
    # args.data_dir = '/Users/daichi/Downloads/ifood'

    args.train_dir = args.data_dir + os.sep + 'train_set'
    args.val_dir = args.data_dir + os.sep + 'val_set'
    args.test_dir = args.data_dir + os.sep + 'test_set'

    args.train_labels_csv = args.data_dir + os.sep + 'train_labels.csv'
    args.val_labels_csv = args.data_dir + os.sep + 'val_labels.csv'


    """Number of class labels"""
    args.num_labels = 251
    args.epochs = 300
    args.num_output_labels = 3
    args.test_overfit = False
    args.num_labels_uecfood100 = 100
    args.num_labels_food101 = 101

    args.num_workers = 8

    args.pretrained = True
    args.start_epoch = 1

    """NTSNET"""
    args.PROPOSAL_NUM = 6
    args.CAT_NUM = 4


    """Resume"""
    args.resume_model_path = None
    #args.resume_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/NTS-Net/results/uecfood/resnet50-20190518_003738/ckpt/resnet50-27-0.7001'
    #args.resume_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/NTS-Net/results/food101/resnet152-20190518_010610/ckpt/resnet152-12-0.6916'
    #args.resume_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/NTS-Net/results/scratch/inceptionresnetv2-20190518_011052/ckpt/inceptionresnetv2-12-0.6579'
    #args.resume_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/NTS-Net/results/scratch/resnet50-20190518_002349/ckpt/resnet50-9-0.6537'
    #args.resume_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/NTS-Net/results/scratch/resnet152-20190518_002731/ckpt/resnet152-6-0.6265'
    #args.resume_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/NTS-Net/results/uecfood/resnet152-20190518_003345/ckpt/resnet152-6-0.6187'


    """mixup"""
    # args.mixup = True
    args.mixup = False
    args.alpha = 1.

    """Random Erasing"""
    args.random_erasing = True
    args.random_erasing_p = 0.5
    args.random_erasing_sh = 0.3
    args.random_erasing_r1 = 0.2
    # args.random_erasing = False


    """Only linear?"""
    args.all_parameter_freeze = False
    # args.all_parameter_freeze = True


    """Image size"""
    # args.resolution = 1
    args.resolution = 2


    """Pretrained UECFOOD100 or FOOD101"""
    args.pre_learned = True
    args.pre_dataset = 'UECFOOD100'
    # args.pre_dataset = 'FOOD101'

    # args.pre_learned = False
    # args.pre_dataset = 'imagenet'


    """Model architecture"""
    # args.library_type = 'Pretrainedmodels'
    # # args.arch = 'default'
    # args.arch = 'pnasnet5large' # rankinglossがnanになる

    # args.arch = 'nasnetalarge'# works
    # args.arch = 'resnext10132x4d' # works
    # args.arch = 'senet154' # works
    # args.arch = 'polynet'
    #args.arch = 'inceptionresnetv2' #works
    #args.arch = 'inceptionv4' #works

    args.library_type = 'torchvisions'
    # args.arch = 'resnet18'
    args.arch = 'resnet50' # works
    # args.arch = 'resnet152' # works
    # args.arch = 'densenet161'
    # args.arch = 'densenet169'

    """Save Path"""
    # args.save_path = './results/debug'
    # args.save_path = './results/uecfood'
    # args.save_path = './results/scratch'
    # args.save_path = './results/food101'

    args.save_path = './results2/debug'
    # args.save_path = './results2/uecfood'
    # args.save_path = './results2/scratch'
    # args.save_path = './results2/food101'


    """Optimizer"""
    # args.optimizer = 'Adam'
    args.optimizer = 'Sgd'
    # args.optimizer = 'AdaBound'


    """Lr Scheduler"""
    # args.lr_scheduler = 'ReduceLROnPlateau' # [ReduceLROnPlateau, ]
    args.lr_scheduler = 'CosineAnnealingLR'
    # args.lr_scheduler = 'MultiStepLR'


    """Loss"""
    # args.loss_type = 'BCEWithLogitsLoss'
    args.loss_type = 'CrossEntropyLoss'




    args.save_dir = os.path.join(args.save_path, '{}-{}'.format(args.arch, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

    args.ckpt_dir = os.path.join(args.save_dir, 'ckpt')
    args.submission_dir = os.path.join(args.save_dir, 'submission')

    if os.path.exists(args.save_dir):
        raise NameError('model dir exists!')
    os.makedirs(args.save_dir)
    os.makedirs(args.ckpt_dir)
    os.makedirs(args.submission_dir)

    num_of_gpus = torch.cuda.device_count()
    args.batch_size = 8
    """Model Architecture"""
    if args.arch == 'pnasnet5large':
        args.fv_size = 4320
        args.imagenet_mean = [0.5, 0.5, 0.5]
        args.imagenet_std = [0.5, 0.5, 0.5]
        if args.all_parameter_freeze:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                args.batch_size = 160
                args.val_batch_size = 16
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                args.batch_size = 80
                args.val_batch_size = 8
        else:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                if num_of_gpus == 2:
                    args.batch_size = 16
                if num_of_gpus == 4:
                    args.batch_size = 32
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 1:
                    args.batch_size = 4
                if num_of_gpus == 2:
                    args.batch_size = 2
                if num_of_gpus == 4:
                    args.batch_size = 4
                if num_of_gpus == 8:
                    args.batch_size = 8
            args.val_batch_size = args.batch_size

    elif args.arch == 'nasnetalarge':
        args.fv_size = 4032
        args.imagenet_mean = [0.5, 0.5, 0.5]
        args.imagenet_std = [0.5, 0.5, 0.5]
        if args.all_parameter_freeze:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                args.batch_size = 100
                args.val_batch_size = 8
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                args.batch_size = 100
                args.val_batch_size = 8
        else:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                if num_of_gpus == 4:
                    args.batch_size = 28
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 2:
                    args.batch_size = 8
                if num_of_gpus == 4:
                    args.batch_size = 8
                if num_of_gpus == 8:
                    args.batch_size = 14
            args.val_batch_size = args.batch_size

    elif args.arch == 'resnext10132x4d':
        args.fv_size = 2048
        args.imagenet_mean = [0.485, 0.456, 0.406]
        args.imagenet_std = [0.229, 0.224, 0.225]
        if args.all_parameter_freeze:
            if args.resolution == 1:
                args.image_min_size = 256
                args.nw_input_size = 224
                args.batch_size = 80
                args.val_batch_size = 10
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                args.batch_size = 80
                args.val_batch_size = 10
        else:
            if args.resolution == 1:
                args.image_min_size = 256
                args.nw_input_size = 224
                if num_of_gpus == 2:
                    args.batch_size = 20
                elif num_of_gpus == 4:
                    args.batch_size = 100
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 2:
                    args.batch_size = 8
                if num_of_gpus == 4:
                    args.batch_size = 12
                if num_of_gpus == 10:
                    args.batch_size = 64
            args.val_batch_size = args.batch_size

    elif args.arch == 'senet154':
        args.fv_size = 2048
        args.imagenet_mean = [0.485, 0.456, 0.406]
        args.imagenet_std = [0.229, 0.224, 0.225]
        if args.all_parameter_freeze:
            if args.resolution == 1:
                args.image_min_size = 256
                args.nw_input_size = 224
                args.batch_size = 100
                args.val_batch_size = 10
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                args.batch_size = 100
                args.val_batch_size = 10
        else:
            if args.resolution == 1:
                args.image_min_size = 256
                args.nw_input_size = 224
                if num_of_gpus == 4:
                    args.batch_size = 80
                elif num_of_gpus == 10:
                    args.batch_size = 128
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 2:
                    args.batch_size = 4
                if num_of_gpus == 4:
                    args.batch_size = 8
                elif num_of_gpus == 8:
                    args.batch_size = 16
            args.val_batch_size = args.batch_size

    elif args.arch == 'polynet':
        args.fv_size = 2048
        args.imagenet_mean = [0.485, 0.456, 0.406]
        args.imagenet_std = [0.229, 0.224, 0.225]
        if args.all_parameter_freeze:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                args.batch_size = 160
                args.val_batch_size = 10
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                args.batch_size = 160
                args.val_batch_size = 10
        else:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                if num_of_gpus == 2:
                    args.batch_size = 20
                elif num_of_gpus == 4:
                    args.batch_size = 80
                elif num_of_gpus == 10:
                    args.batch_size = 128
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 4:
                    args.batch_size = 20
                elif num_of_gpus == 10:
                    args.batch_size = 44
            args.val_batch_size = args.batch_size

    elif args.arch == 'inceptionresnetv2':
        args.fv_size = 1536
        args.imagenet_mean = [0.5, 0.5, 0.5]
        args.imagenet_std = [0.5, 0.5, 0.5]
        if args.all_parameter_freeze:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                args.batch_size = 160
                args.val_batch_size = 10
            elif args.resolution == 2:
                args.image_min_size = 498
                args.nw_input_size = 448
                args.batch_size = 160
                args.val_batch_size = 10
        else:
            if args.resolution == 1:
                args.image_min_size = 339
                args.nw_input_size = 299
                if num_of_gpus == 2:
                    args.batch_size = 64
                elif num_of_gpus == 4:
                    args.batch_size = 80
                elif num_of_gpus == 10:
                    args.batch_size = 128
            if args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 2:
                    args.batch_size = 14
                if num_of_gpus == 4:
                    args.batch_size = 24
                elif num_of_gpus == 10:
                    args.batch_size = 44
            args.val_batch_size = args.batch_size

    elif args.arch == 'inceptionv4':
        args.fv_size = 1536
        args.imagenet_mean = [0.5, 0.5, 0.5]
        args.imagenet_std = [0.5, 0.5, 0.5]
        if args.all_parameter_freeze:
            if args.resolution == 1:
                args.image_min_size = 363
                args.nw_input_size = 331
                args.batch_size = 200
                args.val_batch_size = 10
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                args.batch_size = 200
                args.val_batch_size = 10
        else:
            if args.resolution == 1:
                args.image_min_size = 339
                args.nw_input_size = 299
                if num_of_gpus == 2:
                    args.batch_size = 90
                elif num_of_gpus == 4:
                    args.batch_size = 80
                elif num_of_gpus == 10:
                    args.batch_size = 128
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 2:
                    args.batch_size = 12
                if num_of_gpus == 4:
                    args.batch_size = 36
                elif num_of_gpus == 10:
                    args.batch_size = 44
            args.val_batch_size = args.batch_size

    elif args.arch == 'resnet18':
        args.fv_size = 512
        args.imagenet_mean = [0.485, 0.456, 0.406]
        args.imagenet_std = [0.229, 0.224, 0.225]
        if args.resolution == 1:
            args.image_min_size = 256
            args.nw_input_size = 224
            if num_of_gpus == 2:
                args.batch_size = 300
            elif num_of_gpus == 4:
                args.batch_size = 80
        elif args.resolution == 2:
            args.image_min_size = 478
            args.nw_input_size = 448
            if num_of_gpus == 2:
                args.batch_size = 100
            elif num_of_gpus == 4:
                args.batch_size = 20
            elif num_of_gpus == 10:
                args.batch_size = 44
            args.val_batch_size = args.batch_size
    elif args.arch == 'resnet50':
        args.fv_size = 2048
        args.imagenet_mean = [0.485, 0.456, 0.406]
        args.imagenet_std = [0.229, 0.224, 0.225]
        if args.resolution == 1:
            args.image_min_size = 256
            args.nw_input_size = 224
            if num_of_gpus == 2:
                args.batch_size = 300
            elif num_of_gpus == 4:
                args.batch_size = 80
        elif args.resolution == 2:
            args.image_min_size = 478
            args.nw_input_size = 448
            if num_of_gpus == 1:
                args.batch_size = 10
            if num_of_gpus == 2:
                args.batch_size = 20
            elif num_of_gpus == 4:
                args.batch_size = 44
            elif num_of_gpus == 10:
                args.batch_size = 80
        print('args.batch_size: ', args.batch_size)
        args.val_batch_size = args.batch_size
    elif args.arch == 'resnet152':
        args.fv_size = 2048
        args.imagenet_mean = [0.485, 0.456, 0.406]
        args.imagenet_std = [0.229, 0.224, 0.225]
        if args.resolution == 1:
            args.image_min_size = 256
            args.nw_input_size = 224
            if num_of_gpus == 2:
                args.batch_size = 300
                args.val_batch_size = 10
            elif num_of_gpus == 4:
                args.batch_size = 80
                args.val_batch_size = 10
        elif args.resolution == 2:
            args.image_min_size = 478
            args.nw_input_size = 448
            if num_of_gpus == 2:
                args.batch_size = 8
            elif num_of_gpus == 4:
                args.batch_size = 20
            elif num_of_gpus == 10:
                args.batch_size = 50
            args.val_batch_size = args.batch_size

    elif args.arch == 'densenet161':
        args.fv_size = 2208
        args.imagenet_mean = [0.485, 0.456, 0.406]
        args.imagenet_std = [0.229, 0.224, 0.225]
        if args.resolution == 1:
            args.image_min_size = 256
            args.nw_input_size = 224
            if num_of_gpus == 2:
                args.batch_size = 300
                args.val_batch_size = 10
            elif num_of_gpus == 4:
                args.batch_size = 80
                args.val_batch_size = 10
        elif args.resolution == 2:
            args.image_min_size = 478
            args.nw_input_size = 448
            if num_of_gpus == 2:
                args.batch_size = 8
            elif num_of_gpus == 4:
                args.batch_size = 12
            elif num_of_gpus == 10:
                args.batch_size = 50
            args.val_batch_size = args.batch_size

    args.pretrain_dset_mean = args.imagenet_mean
    args.pretrain_dset_std = args.imagenet_std

    """Optimizer"""
    if args.optimizer == 'Adam':
        args.lr = 1e-3
        args.beta1 = 0.9
        args.beta2 = 0.999
        args.amsgrad = True
        args.weight_decay = 5e-4
    elif args.optimizer == 'Sgd':
        args.lr = 0.001 # 0.1はダメ -> 0.001が良さげ
        #args.lr = 0.0001
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.nesterov = True
    elif args.optimizer == 'AdaBound':
        args.lr = 1e-3
        # args.lr = 1e-4
        args.beta1 = 0.9
        args.beta2 = 0.999
        # args.final_lr = 0.1-------------------------------------
        args.final_lr = 0.01
        args.gamma = 1e-3

    """Lr scheduler"""
    if args.lr_scheduler == 'ReduceLROnPlateau':
        args.scheduler_patience = 1              # Number of epochs with no improvement after which learning rate will be reduced
        args.scheduler_threshold = 1e-6          # learning rate scheduler threshold for measuring the new optimum, to only focus on significant changes
        args.scheduler_factor = 0.1        # learning rate scheduler factor by which the learning rate will be reduced. new_lr = lr * factor
    elif args.lr_scheduler == 'CosineAnnealingLR':
        args.T_max = args.epochs
        args.eta_min = 0.05
        args.last_epoch = 4e-4

    save_exp_info(args)
    # print('Batch size: ', args.batch_size)
    return args


if __name__ == '__main__':
    args = get_parameters()
    print(args.output_id)
0