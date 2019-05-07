from argparse import Namespace
import os
import socket
import re
from datetime import datetime
from collections import OrderedDict


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

    for key, val in p.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def get_parameters():

    args = Namespace()
    # args.data_dir = '/host/space/horita-d/dataset/ifoodchallenge2019'
    args.data_dir = '/Users/daichi/Downloads/ifood'

    args.output_dir = '../results'

    args.train_dir = args.data_dir + os.sep + 'train_set'
    args.val_dir = args.data_dir + os.sep + 'val_set'
    args.test_dir = args.data_dir + os.sep + 'test_set'

    args.train_labels_csv = args.data_dir + os.sep + 'train_labels.csv'
    args.val_labels_csv = args.data_dir + os.sep + 'val_labels.csv'

    args.debug_weights = False
    args.test_overfit = False
    args.num_labels = 251
    args.batch_size = 64
    # args.image_min_size = 256
    args.image_min_size = 384
    # args.nw_input_size = 224
    args.nw_input_size = 331
    args.num_workers = 32
    args.imagenet_mean = [0.485, 0.456, 0.406]
    args.imagenet_std = [0.229, 0.224, 0.225]
    args.pretrain_dset_mean = args.imagenet_mean
    args.pretrain_dset_std = args.imagenet_std

    # args.arch = 'resnet101'
    # args.arch = 'resnet152'
    args.arch = 'pnasnet5large'
    # args.fv_size = 2048
    args.fv_size = 4320
    args.pretrained = True
    args.resume = False
    args.start_epoch = 0

    args.small=1e-12                         # small value used for avoiding div by zero
    args.lr = 1e-4      # Adam optimizer initial learning rate
    # args.lr = 1e-3      # Adam optimizer initial learning rate
    args.lr_scheduler = '' # []
    args.scheduler_patience = 1              # Number of epochs with no improvement after which learning rate will be reduced
    args.scheduler_threshold = 1e-6          # learning rate scheduler threshold for measuring the new optimum, to only focus on significant changes
    args.scheduler_factor = 0.1        # learning rate scheduler factor by which the learning rate will be reduced. new_lr = lr * factor
    args.earlystopping_patience = 1          # early stopping patience is the number of epochs with no improvement after which training will be stopped
    args.earlystopping_min_delta = 1e-5      # minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement
    args.evaluate = False
    args.epochs = 50
    args.print_details = False
    args.print_freq = args.batch_size

    args.loss_type = 'ce' # ['ce', 'focal']

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


    return args


if __name__ == '__main__':
    args = get_parameters()
    print(args.output_id)