from argparse import Namespace
import os
import socket
import re
from datetime import datetime
from collections import OrderedDict
import torch


def get_output_fname(args):
    return "%s_%s" % (get_hostname_timestamp_id(), args.arch)

def get_hostname_timestamp_id():
    return str(datetime.now())[:16].replace(' ', '-').replace(':', '').replace('-', '')
    
def mkdir_p(d):
    os.makedirs(d, exist_ok=True)

def mkdir_exp_dir(args):
    mkdir_p(args.exp_dir)
    mkdir_p(args.ckpt_dir)
    mkdir_p(args.sub_dir)

    
    logfile = os.path.join(args.exp_dir, 'parameters.txt')
    log_file = open(logfile, 'w')
    p = OrderedDict()
    p['arch'] = args.arch
    p['image_min_size'] = args.image_min_size
    p['nw_input_size'] = args.nw_input_size
    p['lr'] = args.lr
    p['lr_scheduler'] = args.lr_scheduler
    p['loss_type'] = args.loss_type
    p['epoch'] = args.epochs
    p['all_parameter_freeze'] = args.all_parameter_freeze
    p['mixup'] = args.mixup
    p['tta'] = args.tta
    p['random_erasing']= args.random_erasing

    for key, val in p.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def get_parameters():

    args = Namespace()
    # args.data_dir = '/host/space/horita-d/dataset/ifoodchallenge2019'
    args.data_dir = '/export/ssd/dataset/iFood2019'
    # args.data_dir = '/Users/daichi/Downloads/ifood'

    args.output_dir = '../results2'

    args.train_dir = args.data_dir + os.sep + 'train_set'
    args.val_dir = args.data_dir + os.sep + 'val_set'
    args.test_dir = args.data_dir + os.sep + 'test_set'

    args.train_labels_csv = args.data_dir + os.sep + 'train_labels.csv'
    args.val_labels_csv = args.data_dir + os.sep + 'val_labels.csv'

    args.debug_weights = False
    args.test_overfit = False
    args.num_labels = 100
    
    args.num_workers = 8

    args.pretrained = True
    args.resume = False
    args.start_epoch = 1
    args.small = 1e-12


    """mixup"""
    args.mixup = True
    args.alpha = 1.
    args.random_erasing = False

    """model architecture"""
    args.all_parameter_freeze = False
    # args.all_parameter_freeze = True


    # args.resolution = 1
    args.resolution = 2

    # Pretrainedmodels
    # args.arch = 'pnasnet5large'
    # args.arch = 'resnext10132x4d'
    # args.arch = 'nasnetalarge'
    # args.arch = 'senet154'
    args.arch = 'polynet'
    # args.arch = 'inceptionresnetv2'
    # args.arch = 'inceptionv4'

    # Torchvisions
    # args.arch = 'resnet18'
    # args.arch = 'resnet152'

    """Optimizer"""
    # args.optimizer = 'Adam'
    args.optimizer = 'Sgd'
    # args.optimizer = 'AdaBound'

    """Lr Scheduler"""
    # args.lr_scheduler = 'ReduceLROnPlateau' # [ReduceLROnPlateau, ]
    args.lr_scheduler = 'CosineAnnealingLR'

    """Loss"""
    # args.loss_type = 'BCEWithLogitsLoss'
    args.loss_type = 'CrossEntropyLoss'

    args.earlystopping_patience = 1          # early stopping patience is the number of epochs with no improvement after which training will be stopped
    args.earlystopping_min_delta = 1e-5      # minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement

    args.evaluate = False
    args.epochs = 300
    
    args.output_dir += '/reso{}-{}-ParamFreeze-{}-mixup-{}-randomErasing-{}'.format(args.resolution, args.optimizer, args.all_parameter_freeze, args.mixup, args.random_erasing)

    """Random Erasing"""
    args.random_erasing_p = 0.5
    args.random_erasing_sh = 0.4
    args.random_erasing_r1 = 0.3

    args.tta = True

    args.output_id = get_output_fname(args)
    args.exp_dir = '{}/{}'.format(args.output_dir, args.output_id)
    args.ckpt_dir = '{}/ckpt'.format(args.exp_dir)
    args.ckpt = args.ckpt_dir + os.sep + 'ckpt_%s.pth.tar' % (args.arch,)
    args.best = args.ckpt_dir + os.sep + 'best_%s.pth.tar' % (args.arch,)
    args.num_output_labels = 3
    args.sub_dir = args.exp_dir + os.sep + 'submissions'
    
    args.output_file = args.sub_dir + os.sep + 'output_%s_%s.csv' %  (args.output_id, args.output_id)
    args.params_file = args.sub_dir + os.sep + 'params_%s.json' % args.output_id

    args.log_dir = args.exp_dir

    args.edata_json = '/home/yanai-lab/horita-d/ifood/src/edafa/imagenet.json'

    
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
                if num_of_gpus == 4:
                    args.batch_size = 16
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
                if num_of_gpus == 4:
                    args.batch_size = 12
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
                if num_of_gpus == 4:
                    args.batch_size = 32
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
                if num_of_gpus == 4:
                    args.batch_size = 20
                elif num_of_gpus == 10:
                    args.batch_size = 44
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
            elif args.resolution == 2:
                args.image_min_size = 478
                args.nw_input_size = 448
                if num_of_gpus == 4:
                    args.batch_size = 20
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
                if num_of_gpus == 4:
                    args.batch_size = 20
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
                args.batch_size = 100
            elif num_of_gpus == 4:
                args.batch_size = 20
            elif num_of_gpus == 10:
                args.batch_size = 44
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
        args.lr = 0.1
        args.momentum = 0.9
        args.weight_decay = 5e-4
        args.nesterov = True
    elif args.optimizer == 'AdaBound':
        args.lr = 1e-3
        args.beta1 = 0.9
        args.beta2 = 0.999
        args.final_lr = 0.1
        args.gamma = 1e-3

    """Lr scheduler"""
    if args.lr_scheduler == 'ReduceLROnPlateau':
        args.scheduler_patience = 1              # Number of epochs with no improvement after which learning rate will be reduced
        args.scheduler_threshold = 1e-6          # learning rate scheduler threshold for measuring the new optimum, to only focus on significant changes
        args.scheduler_factor = 0.1        # learning rate scheduler factor by which the learning rate will be reduced. new_lr = lr * factor
    elif args.lr_scheduler == 'CosineAnnealingLR':
        args.T_max = args.epochs
        args.eta_min = 0
        args.last_epoch = -1
    args.batch_size = 8
    return args


if __name__ == '__main__':
    args = get_parameters()
    print(args.output_id)