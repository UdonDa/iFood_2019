from argparse import Namespace

def get_output_fname():
    return "%s_%s_%s" % (args.author, args.arch, get_hostname_timestamp_id())

def get_hostname_timestamp_id():
    return socket.gethostname() + '_' + re.sub(r'\W+', '', str(datetime.now()))

def get_parameters():

    args = Namespace()
    args.perm_dir = '/mnt/disks/imaterialist_fashion'
    args.base_dir = '/mnt/ram-disk/imaterialist_fashion'
    # args.data_dir = '/host/space/horita-d/dataset/ifoodchallenge2019'
    args.data_dir = '/Users/daichi/Downloads/ifood'

    args.input_dir = args.data_dir + os.sep + 'input'
    args.output_dir = args.data_dir + os.sep + 'output'

    args.train_dir = args.data_dir + os.sep + 'train_set'
    args.val_dir = args.data_dir + os.sep + 'val_set'
    args.test_dir = args.data_dir + os.sep + 'test_set'

    args.train_labels_csv = args.data_dir + os.sep + 'train_labels.csv'
    args.val_labels_csv = args.data_dir + os.sep + 'val_labels.csv'
    # args.test_labels_csv = args.input_dir + os.sep + 'test_.csv'
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
    args.optimizer_learning_rate = 1e-4      # Adam optimizer initial learning rate
    # args.optimizer_learning_rate = 1e-3      # Adam optimizer initial learning rate
    args.scheduler_patience = 1              # Number of epochs with no improvement after which learning rate will be reduced
    args.scheduler_threshold = 1e-6          # learning rate scheduler threshold for measuring the new optimum, to only focus on significant changes
    args.scheduler_factor = 0.1        # learning rate scheduler factor by which the learning rate will be reduced. new_lr = lr * factor
    args.earlystopping_patience = 1          # early stopping patience is the number of epochs with no improvement after which training will be stopped
    args.earlystopping_min_delta = 1e-5      # minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement
    args.evaluate = False
    args.epochs = 50
    args.print_details = False
    args.print_freq = args.batch_size
    args.ckpt_dir = args.output_dir + os.sep + 'ckpt'
    args.ckpt = args.ckpt_dir + os.sep + 'ckpt_%s.pth.tar' % (args.arch,)
    args.best = args.ckpt_dir + os.sep + 'best_%s.pth.tar' % (args.arch,)
    args.num_output_labels = 3
    args.sub_dir = args.output_dir + os.sep + 'submissions'
    args.author = 'deccanlearners'
    args.output_id = get_output_fname()
    args.output_file = args.sub_dir + os.sep + 'output_%s.csv' %  args.output_id
    args.params_file = args.sub_dir + os.sep + 'params_%s.json' % args.output_id
    args.min_img_bytes = 4792

    return args


if __name__ == '__main__':
    args = get_parameters()
    print(args)