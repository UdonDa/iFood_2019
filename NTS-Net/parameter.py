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
    p['loss_type'] = args.loss_type
    p['epoch'] = args.epochs
    p['all_parameter_freeze'] = args.all_parameter_freeze
    p['mixup'] = args.mixup
    p['tta'] = args.tta
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
    # args.data_dir = '/host/space/horita-d/dataset/ifoodchallenge2019'
    args.data_dir = '/export/ssd/dataset/iFood2019'
    # args.data_dir = '/Users/daichi/Downloads/ifood'

    # args.output_dir = './results/uecfood101foods100-adabound'
    args.output_dir = './results/debug'

    args.train_dir = args.data_dir + os.sep + 'train_set'
    args.val_dir = args.data_dir + os.sep + 'val_set'
    args.test_dir = args.data_dir + os.sep + 'test_set'

    args.train_labels_csv = args.data_dir + os.sep + 'train_labels.csv'
    args.val_labels_csv = args.data_dir + os.sep + 'val_labels.csv'

    args.debug_weights = False
    args.test_overfit = False
    args.num_labels = 251
    
    args.num_workers = 8

    args.pretrained = True
    args.start_epoch = 1
    args.small = 1e-12

    """NTSNET"""
    args.PROPOSAL_NUM = 6
    args.CAT_NUM = 4

    """ams"""
    args.prof = True


    """Resume"""
    args.resume = False
    args.pretrained_model_path = None
    # args.resume = True # Pretrained model

    # RESNEXT
    ## args.pretrained_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/results/201905101755_resnext10132x4d/ckpt/best-19-0.6727-resnext10132x4d.pth.tar'
    # args.pretrained_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/results2/reso2-Sgd-ParamFreeze-False-mixup-True-randomErasing-False-resume-True/201905131418_resnext10132x4d/ckpt/ckpt-69-0.7329-resnext10132x4d.pth.tar'
    # NASNET
    # args.pretrained_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/results/201905101757_nasnetalarge/ckpt/best-8-0.6174-nasnetalarge.pth.tar'
    # SENET
    # args.pretrained_model_path = '/host/space/horita-d/programing/python/conf/cvpr2020/ifood_challenge2019/results/201905101757_senet154/ckpt/best-7-0.6845-senet154.pth.tar'

    """Pretrained UECFOOD100 or FOOD101"""
    args.pre_learned = True
    args.pre_dataset = 'UECFOOD'
    # args.pre_learned = False

    """mixup"""
    args.mixup = True
    # args.mixup = False
    args.alpha = 1.
    # args.random_erasing = True
    args.random_erasing = False

    """model architecture"""
    args.all_parameter_freeze = False
    # args.all_parameter_freeze = True

    # args.resolution = 1
    args.resolution = 2

    # # Pretrainedmodels
    # args.library_type = 'Pretrainedmodels'
    # # args.arch = 'default'
    # args.arch = 'pnasnet5large' # rankinglossがnanになる

    # args.arch = 'nasnetalarge'# works
    # args.arch = 'resnext10132x4d' # works
    # args.arch = 'senet154' # works
    # args.arch = 'polynet'
    # args.arch = 'inceptionresnetv2' #works
    # args.arch = 'inceptionv4' #works

    # # Torchvisions
    args.library_type = 'torchvisions'
    # args.arch = 'resnet18'
    args.arch = 'resnet50' # works
    # args.arch = 'resnet152' # works

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

    args.earlystopping_patience = 1          # early stopping patience is the number of epochs with no improvement after which training will be stopped
    args.earlystopping_min_delta = 1e-5      # minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement

    args.evaluate = False
    args.epochs = 500
    
    # args.output_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    args.output_dir = os.path.join(args.output_dir, 'debug')


    """Random Erasing"""
    args.random_erasing_p = 0.5
    args.random_erasing_sh = 0.4
    args.random_erasing_r1 = 0.3

    args.tta = True

    args.output_id = args.arch
    args.exp_dir = '{}-{}'.format(args.output_dir, args.output_id)
    args.ckpt_dir = '{}/ckpt'.format(args.exp_dir)

    args.ckpt = 'ckpt_%s.pth.tar' % (args.arch,)
    args.best = 'best_%s.pth.tar' % (args.arch,)

    args.num_output_labels = 3
    args.sub_dir = args.exp_dir + os.sep + 'submissions'
    
    args.output_file = args.sub_dir + os.sep + 'output_%s_%s.csv' %  (args.output_id, args.output_id)
    args.params_file = args.sub_dir + os.sep + 'params_%s.json' % args.output_id

    args.log_dir = args.exp_dir

    args.edata_json = '/home/yanai-lab/horita-d/ifood/src/edafa/imagenet.json'

    
    num_of_gpus = torch.cuda.device_count()
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
                    args.batch_size = 16
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
                args.batch_size = 40
            elif num_of_gpus == 10:
                args.batch_size = 44
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
                args.batch_size = 12
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
        args.lr = 0.001 # 0.1はダメ -> 0.001が良さげ
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.nesterov = True
    elif args.optimizer == 'AdaBound':
        # args.lr = 1e-3
        args.lr = 1e-4
        args.beta1 = 0.9
        args.beta2 = 0.999
        # args.final_lr = 0.1
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

    return args


if __name__ == '__main__':
    args = get_parameters()
    print(args.output_id)
