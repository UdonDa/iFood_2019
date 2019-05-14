'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision

import numpy as np        
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
#import seaborn as sn
from sklearn.metrics import confusion_matrix

import numbers
class GridCrop(object):
    def __init__(self, size, num):
        self.size = size
        self.num = num
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        if isinstance(num, numbers.Number):
            self.num = (int(num), int(num))
        else:
            assert len(num) == 2, "Please provide only two dimensions (h, w) for num."
            self.num = num
    
    def __call__(self, img):
        w, h = img.size
        crop_h, crop_w = self.size
        num_h, num_w = self.num
        if crop_w > w or crop_h > h:
            raise ValueError("Requested crop size {} is bigger than input size {}".format(size,(h, w)))        
        interval_x = int((w-crop_w)/float(num_w-1))
        interval_y = int((h-crop_h)/float(num_h-1))
        
        crop_imgs = []
        for i in range(num_h):
            for j in range(num_w):
                left  = interval_x*j
                upper = interval_y*i
                crop_imgs.append(img.crop((left, upper, left+crop_w, upper+crop_h)))
                
        return tuple(crop_imgs)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0} interval={0})'.format(self.size, self.interval)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def display(trainloader, classes_en):
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465]) 
        std = np.array([0.2023, 0.1994, 0.2010])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        #plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        #plt.show()
        plt.imsave('disp.jpg', inp)
        sys.exit()
    # Get a batch of training data
    inputs, labels = next(iter(trainloader))
    #bs, ncrops, c, h, w = inputs.size()
    #inputs = inputs.view(-1, c, h, w)
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[classes_en[x] for x in labels])

def print_cmx(y_pred, y_true):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sn.heatmap(df_cmx, annot=True)
    plt.show()

def eval_cmx(pred, true):
    '''
    input: confusion matrix
    '''
    cmx_data = confusion_matrix(true, pred) # confusion matrix
    num_classes = cmx_data.shape[0]

    eval_data = {}
    eval_data['micro average'] = np.sum(np.diag(cmx_data))/np.sum(cmx_data)*100

    eval_data['Ps'] = []
    eval_data['Rs'] = []
    eval_data['F1s'] = []
    sum_precisions = 0
    sum_recalls = 0
    sum_f1s = 0
    for i in range(num_classes):
        tp = cmx_data[i,i]
        tpfp = np.sum(cmx_data[:,i]) # for precision
        tpfn = np.sum(cmx_data[i,:]) # for recall
        
        precision = tp/tpfp
        recall    = tp/tpfn
        f1        = 2./(1/precision+1/recall)
        
        sum_precisions += precision
        sum_recalls    += recall
        sum_f1s        += f1
        
        eval_data['Ps'].append(precision)
        eval_data['Rs'].append(recall)
        eval_data['F1s'].append(f1)
            
        #print('Class{} P:{:.3f} R:{:.3f} F1:{:.3f}'.format(i+1, precision*100, recall*100, f1*100))
        
    eval_data['macro average precision'] = sum_precisions/num_classes
    eval_data['macro average recall']    = sum_recalls/num_classes
    eval_data['macro average f1']        = sum_f1s/num_classes

    return eval_data
