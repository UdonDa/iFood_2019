import os
import torch.utils.data
import torch
import torch.nn as nn
from torch.backends import cudnn
import datetime
import tensorboardX as tbx
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
# from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import modelNTS
from core.utils import init_log, progress_bar
from data_loader import get_data_loader
from parameter import get_parameters, save_exp_info
from util import AverageMeter, adjust_learning_rate, TopKAccuracyMicroAverageMeter, F1MicroAverageMeter, F1MicroAverageMeterByTopK
import time
import argparse
from sys import exit
from adabound_optim import AdaBound
from collections import OrderedDict
from vat import VAT


# Parser
parser = argparse.ArgumentParser(description='iFood 2019 challenge')
parser.add_argument('--gpu', '-g', default='0', type=str, help='gpu id')
parser.add_argument('--name', '-n', default='', type=str, help='gpu id')
config = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
device = 'cuda'
args = get_parameters()


def get_optimizers(args, net):
    raw_parameters = list(net.pretrained_model.parameters())
    part_parameters = list(net.proposal_net.parameters())
    concat_parameters = list(net.concat_net.parameters())
    partcls_parameters = list(net.partcls_net.parameters())

    raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer = None, None, None, None
    if args.optimizer == 'Sgd':
        raw_optimizer = torch.optim.SGD(raw_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        concat_optimizer = torch.optim.SGD(concat_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        part_optimizer = torch.optim.SGD(part_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'AdaBound':
        raw_optimizer = AdaBound(raw_parameters, lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)
        concat_optimizer = AdaBound(concat_parameters, lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)
        part_optimizer = AdaBound(part_parameters-----, lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)
        partcls_optimizer = AdaBound(partcls_parameters, lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)

    return raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer


def load_checkpoint(args, net, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer):
    start_epoch = 1
    best_acc = 0
    ckpt = args.resume_model_path
    if ckpt is not None:
        if os.path.isfile(ckpt):
            print('==> Loading checkpoint...\n{}'.format(ckpt))
            from core.modelNTS import convert_state_dict
            ckpt = torch.load(ckpt)
            # net
            net.load_state_dict(convert_state_dict(ckpt['state_dict']))

            # optimizer
            #raw_optimizer.load_state_dict(ckpt['raw_optimizer'])
            #concat_optimizer.load_state_dict(ckpt['concat_optimizer'])
            #part_optimizer.load_state_dict(ckpt['part_optimizer'])
            #partcls_optimizer.load_state_dict(ckpt['partcls_optimizer'])

            start_epoch = ckpt['epoch']
            best_acc = ckpt['best_acc']
        else:
            print('==> No checkpoint found at {}'.format(ckpt))
    return net, start_epoch, best_acc, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer


def save_checkpoint(state, args, epoch, best_acc):
    best_acc = str(best_acc)[:6]
    filename = '{args.arch}-{epoch}-{best_acc}'.format(
        args=args, epoch=epoch, best_acc=best_acc
    )
    save_path = os.path.join(args.ckpt_dir, filename)
    torch.save(state, save_path)
    print('==> Sucess to save model: ', save_path)


# read dataset
trainloader, val_loader, test_loader, test_dset = get_data_loader(args)

# define model
net = modelNTS.attention_net(args)

# define optimizer
raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer = get_optimizers(args, net)

# Resume
# net, start_epoch, best_acc, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer = load_checkpoint(args, net, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer)

# Criterion
criterion = nn.CrossEntropyLoss().cuda()
eps = 1.0
xi = 10.0
use_entmin = True
vat_criterion = VAT(device, eps, xi, use_entmin=use_entmin)

#define optimizers
raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer = get_optimizers(args, net)

net, start_epoch, best_acc, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer = load_checkpoint(args, net, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer)

# define schedulers
schedulers = [MultiStepLR(raw_optimizer, milestones=[40, 50, 60, 70, 80, 90, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[40, 50, 60, 70, 80, 90, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[40, 50, 60, 70, 80, 90, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[40, 50, 60, 70, 80, 90, 100], gamma=0.1)]

# schedulers = [CosineAnnealingLR(raw_optimizer, 300),
#               CosineAnnealingLR(concat_optimizer, 300),
#               CosineAnnealingLR(part_optimizer, 300),
#               CosineAnnealingLR(partcls_optimizer, 300)]


# cuda and data parallel
net = net.cuda()
net = nn.DataParallel(net)

start_time = time.time()

# Tensoroard
writer = tbx.SummaryWriter(args.save_dir)

for epoch in range(start_epoch, args.epochs):

    """Training part"""
    net.train()

    # define average meters
    total_loss_meter = AverageMeter()
    raw_loss_meter = AverageMeter()
    rank_loss_meter = AverageMeter()
    concat_loss_meter = AverageMeter()
    partcls_loss_meter = AverageMeter()
    correct, total, acc = 0, 0, 0

    for i, (img, label) in enumerate(trainloader):
        img, label = img.cuda(), label.cuda().long()

        # zero grad
        raw_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        part_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        # training
        batch_size = img.size(0)
        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
        part_loss = modelNTS.list_loss(part_logits.view(batch_size * args.PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, args.PROPOSAL_NUM).view(-1)).view(batch_size, args.PROPOSAL_NUM)
        raw_loss = criterion(raw_logits, label)
        concat_loss = criterion(concat_logits, label)
        rank_loss = modelNTS.ranking_loss(top_n_prob, part_loss)
        partcls_loss = criterion(part_logits.view(batch_size * args.PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, args.PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()

        raw_optimizer.step()
        concat_optimizer.step()
        part_optimizer.step()
        partcls_optimizer.step()

        # Average meter
        total_loss_meter.update(total_loss.item(), batch_size)
        raw_loss_meter.update(raw_loss.item(), batch_size)
        rank_loss_meter.update(rank_loss.item(), batch_size)
        concat_loss_meter.update(concat_loss.item(), batch_size)
        partcls_loss_meter.update(partcls_loss.item(), batch_size)

        # Accuracy
        _, predicted = torch.max(concat_logits, 1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        acc = correct / total

        if i % 10 == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            print('Train: [{0}][{1}/{2}]\t'
                'Time: {time}\t'
                'Total Loss: {total_loss_meter.avg:.4f}\t'
                'Raw Loss: {raw_loss_meter.avg:.4f}\t'
                'Rank Loss: {rank_loss_meter.avg:.4f}\t'
                'Concat Loss: {concat_loss_meter.avg:.4f}\t'
                'Partcls Loss: {partcls_loss_meter.avg:.4f}\t'
                'Acc {acc:.4f}'.format(
                    epoch,
                    i,
                    len(trainloader),
                    time=et,
                    total_loss_meter=total_loss_meter,
                    raw_loss_meter=raw_loss_meter,
                    rank_loss_meter=rank_loss_meter,
                    concat_loss_meter=concat_loss_meter,
                    partcls_loss_meter=partcls_loss_meter,
                    acc=acc
                )
            )

        # if i % 11 == 0:
        #     break # TODO: Debug

    writer.add_scalar('Train/TotalLoss', total_loss_meter.avg, epoch)
    writer.add_scalar('Train/RawLoss', raw_loss_meter.avg, epoch)
    writer.add_scalar('Train/RankLoss', rank_loss_meter.avg, epoch)
    writer.add_scalar('Train/ConcatLoss', concat_loss_meter.avg, epoch)
    writer.add_scalar('Train/PartclsLoss', partcls_loss_meter.avg, epoch)
    writer.add_scalar('Train/Acc', acc, epoch)

    """Validation part"""
    loss_meter = AverageMeter()
    acc, correct, total = 0, 0, 0
    # switch to evaluate mode
    net.eval()

    with torch.no_grad():
        start_time = time.time()
        loss_avg_epoch = 0.0

        for i, (input, target) in enumerate(val_loader):
            input, label = input.to(device), target.to(device).long()

            bs, ncrops, c, h, w = input.size() # When you use Ten crops.
            _, concat_logits, _, _, _  = net(input.view(-1, c, h, w))
            concat_logits = concat_logits.view(bs, ncrops, -1).mean(1)

            loss = criterion(concat_logits, label)

            # compute accuracy
            _, predicted = torch.max(concat_logits, 1)
            total += target.size(0)
            correct += predicted.eq(label).sum().item()
            acc = correct / total

            # Average meter
            loss_meter.update(loss.item(), input.size(0))

            # Accuracy
            _, predicted = torch.max(concat_logits, 1)
            total += target.size(0)
            correct += predicted.eq(label).sum().item()
            acc = correct / total

            if i % 10 == 0:

                # measure elapsed time
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]

                print('Val: [{0}][{1}/{2}]\t'
                    'Time: {time}\t'
                    'Loss: {loss_meter.avg:.4f}\t'
                    'Acc {acc:.4f}'.format(
                        epoch,
                        i,
                        len(val_loader),
                        time=et,
                        loss_meter=loss_meter,
                        acc=acc
                    )
                )

                # if i % 10 == 0:
                #     break # TODO: Debug

    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/Acc', acc, epoch)


    is_best = acc >= best_acc
    print('acc: {:.4f}, best_acc {:.4f}'.format(acc, best_acc))
    print('is_best: ', is_best)
    best_acc = max(acc, best_acc)
    if is_best:
        print("BEST at epoch: ", epoch)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'raw_optimizer' : raw_optimizer.state_dict(),
            'concat_optimizer' : concat_optimizer.state_dict(),
            'part_optimizer' : part_optimizer.state_dict(),
            'partcls_optimizer' : partcls_optimizer.state_dict(),
            }, args, epoch, best_acc)


        """Testing part"""
        res = OrderedDict()
        net.eval()
        ofname_ = '{}/{}.csv'.format(args.submission_dir, epoch)
        with open(ofname_, "w") as ofd:
            ofd.write("img_name,label\n")
            with torch.no_grad():
                end = time.time()
                index = 0
                for i, input in enumerate(test_loader):
                    input = input.cuda()

                    bs, ncrops, c, h, w = input.size() # When you use Ten crops.
                    _, output, _, _, _ = net(input.view(-1, c, h, w))
                    output = output.view(bs, ncrops, -1).mean(1)
                    res = torch.exp(output).topk(args.num_output_labels, dim=1)[1].cpu().numpy().tolist()

                    for j, resj in enumerate(res):
                        result = "%s,%s\n" % (os.path.basename(test_dset.img_name[index]), " ".join(map(str, resj)))
                        ofd.write(result)
                        index += 1

                    if i % 10 == 0:
                        # measure elapsed time
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]

                        print('Test: [{0}][{1}/{2}]\t'
                            'Time: {time}\t'.format(
                            epoch, i, len(test_loader), time=et))

                    # if i % 11 == 0: # TODO: debug
                    #     break
