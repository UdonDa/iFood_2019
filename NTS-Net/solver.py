import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import sys
from sys import exit
import time
import shutil
from collections import OrderedDict
import os
import json
import tensorboardX as tbx
import datetime
from core import modelNTS

from util import AverageMeter, adjust_learning_rate, TopKAccuracyMicroAverageMeter, F1MicroAverageMeter, F1MicroAverageMeterByTopK
from data_loader import get_data_loader
from parameter import mkdir_exp_dir
device = 'cuda'
from torch.optim.lr_scheduler import MultiStepLR



def get_criterion(args):
    criterion = None
    if args.loss_type == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()    

    return criterion


def get_NTS(args):
    # Model
    net = modelNTS.attention_net(args)
    # # Params
    raw_parameters = list(net.pretrained_model.parameters())
    part_parameters = list(net.proposal_net.parameters())
    concat_parameters = list(net.concat_net.parameters())
    partcls_parameters = list(net.partcls_net.parameters())

    # Optimizer
    raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer = None, None, None, None
    if args.optimizer == 'Sgd':
        raw_optimizer = torch.optim.SGD(raw_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        concat_optimizer = torch.optim.SGD(concat_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        part_optimizer = torch.optim.SGD(part_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Scheduler
    schedulers = [
        MultiStepLR(raw_optimizer, milestones=[60, 100, 300], gamma=0.1),
        MultiStepLR(concat_optimizer, milestones=[60, 100, 300], gamma=0.1),
        MultiStepLR(part_optimizer, milestones=[60, 100, 300], gamma=0.1),
        MultiStepLR(partcls_optimizer, milestones=[60, 100, 300], gamma=0.1)]

    net = net.to(device)
    net = nn.DataParallel(net)
    return net, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer, schedulers




def load_checkpoint(model, args, resume=True, ckpt=None):
    """optionally resume from a checkpoint."""
    best_top3 = 0
    best_acc = 0
    if args.resume:
        if os.path.isfile(ckpt):
            print("=> loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            args.start_epoch = checkpoint['epoch']
            best_top3 = checkpoint['best_top3']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt))
            best_top3 = 0
    return (model, args, best_top3, best_acc)


def save_checkpoint(state, is_best, args, epoch, best_acc):
    best_top3 = str(best_acc)[:6]
    a, b, c = args.ckpt.split('_')
    filename = '{}_{}-{}-{}-{}'.format(a, b, epoch, best_top3, c)
    a, b, c = args.best.split('_')
    best_model_filename = '{}_{}-{}-{}-{}'.format(a, b, epoch, best_top3, c)
    torch.save(state, filename)
    print('Sucess to save model: ', filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)


def train(args, train_loader, model, criterion, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer, epoch):
    total_loss_meter = AverageMeter()
    raw_loss_meter = AverageMeter()
    rank_loss_meter = AverageMeter()
    concat_loss_meter = AverageMeter()
    partcls_loss_meter = AverageMeter()

    correct = 0
    total = 0
    acc = 0
    model.train()
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, label = input.to(device), target.to(device).long()
        batch_size = input.size(0)
        
        raw_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        part_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        raw_logits, concat_logits, part_logits, _, top_n_prob = model(input)
        
        # Loss
        part_loss = modelNTS.list_loss(part_logits.view(batch_size * args.PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, args.PROPOSAL_NUM).view(-1)).view(batch_size, args.PROPOSAL_NUM)
        raw_loss = criterion(raw_logits, label)
        concat_loss = criterion(concat_logits, label)
        rank_loss = modelNTS.ranking_loss(top_n_prob, part_loss)
        partcls_loss = criterion(part_logits.view(batch_size * args.PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, args.PROPOSAL_NUM).view(-1))
        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()

        raw_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        part_optimizer.zero_grad()
        partcls_optimizer.zero_grad()
        
        # Average meter
        total_loss_meter.update(total_loss.item(), input.size(0))
        raw_loss_meter.update(raw_loss.item(), input.size(0))
        rank_loss_meter.update(rank_loss.item(), input.size(0))
        concat_loss_meter.update(concat_loss.item(), input.size(0))
        partcls_loss_meter.update(partcls_loss.item(), input.size(0))

        # Accuracy
        _, predicted = torch.max(concat_logits, 1)
        total += target.size(0)
        correct += predicted.eq(label).sum().item()
        acc = correct / total

        if i % 10 == 0:
            # measure elapsed time
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
                    len(train_loader),
                    time=et,
                    total_loss_meter=total_loss_meter,
                    raw_loss_meter=raw_loss_meter,
                    rank_loss_meter=rank_loss_meter,
                    concat_loss_meter=concat_loss_meter,
                    partcls_loss_meter=partcls_loss_meter,
                    acc=acc
                )
            )

        # if i % 10 == 0:
        #     break # TODO: Debug
            
    writer.add_scalar('Train/TotalLoss', total_loss_meter.avg, epoch)
    writer.add_scalar('Train/RawLoss', raw_loss_meter.avg, epoch)
    writer.add_scalar('Train/RankLoss', rank_loss_meter.avg, epoch)
    writer.add_scalar('Train/ConcatLoss', concat_loss_meter.avg, epoch)
    writer.add_scalar('Train/PartclsLoss', partcls_loss_meter.avg, epoch)
    writer.add_scalar('Train/Acc', acc, epoch)


def validate(args, val_loader, model, criterion, epoch):
    loss_meter = AverageMeter()
    acc = 0
    correct = 0
    total = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        loss_avg_epoch = 0.0

        for i, (input, target) in enumerate(val_loader):
            input, label = input.to(device), target.to(device).long()

            bs, ncrops, c, h, w = input.size() # When you use Ten crops.
            _, concat_logits, _, _, _  = model(input.view(-1, c, h, w))
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
                
                print('Val: [{0}/{1}]\t'
                    'Time: {time}\t'
                    'Loss: {loss_meter.avg:.4f}\t'
                    'Acc {acc:.4f}'.format(
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

    return acc


def test(ofname, pfname, args, test_dset,
        test_loader, best_model_ckpt, model,
        num_output_labels=3, epoch=0):
    
    batch_time = AverageMeter()
    res = OrderedDict()

    model.eval()
    
    res = OrderedDict()

    ofname_ = '{}/{}.csv'.format(args.sub_dir, epoch)
    with open(ofname_, "w") as ofd:
        ofd.write("img_name,label\n")
        with torch.no_grad():
            end = time.time()
            index = 0
            for i, input in enumerate(test_loader):
                input = input.cuda()
                # output = model(input) # When you do not use Ten crops.

                bs, ncrops, c, h, w = input.size() # When you use Ten crops.
                output = model(input.view(-1, c, h, w))
                output = output.view(bs, ncrops, -1).mean(1)

                res = torch.exp(output).topk(num_output_labels, dim=1)[1].cpu().numpy().tolist()

                batch_time.update(time.time() - end)
                end = time.time()
            
                if i % 10 == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time))

                for j, resj in enumerate(res):
                    result = "%s,%s\n" % (os.path.basename(test_dset.img_name[index]), " ".join(map(str, resj)))
                    ofd.write(result)
                    index += 1

                # if i > 10: # TODO: debug
                #     break
                    
            
            print('TEST: [{epoch}]\t'
                'Time {epoch_time:.3f})\t'.format(epoch=epoch, epoch_time=batch_time.sum))


def train_loop(train_loader=None, val_loader=None, test_loader=None, test_dset=None, args=None, raw_optimizer=None,
            concat_optimizer=None,
            part_optimizer=None,
            partcls_optimizer=None, lr_scheduler=None, model=None, criterion=None, num_output_labels=3):
    if args.evaluate:
        validate(val_loader, model, criterion)
    else:
        model, args, best_top3, best_acc = load_checkpoint(model, args, resume=args.resume, ckpt=args.pretrained_model_path)
        for epoch in range(args.start_epoch, args.epochs):
            train(args, train_loader, model, criterion,
                            raw_optimizer,
                            concat_optimizer,
                            part_optimizer,
                            partcls_optimizer,
                            epoch)

            acc = validate(args, val_loader, model, criterion, epoch)

            is_best = acc > best_acc
            print('acc: {:.4f}, best_acc {:.4f}'.format(acc, best_acc))
            print('is_best: ', is_best)
            best_acc = max(acc, best_acc)
            mkdir_exp_dir(args)
            if is_best:
                print("BEST at epoch: ", epoch)
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_top3': best_top3,
                    'best_acc': best_acc,
                    'raw_optimizer' : raw_optimizer.state_dict(),
                    'concat_optimizer' : concat_optimizer.state_dict(),
                    'part_optimizer' : part_optimizer.state_dict(),
                    'partcls_optimizer' : partcls_optimizer.state_dict(),
                    }, is_best, args, epoch, best_acc)
                test(args.output_file, args.params_file, args, test_dset, test_loader, args.best, model, num_output_labels=args.num_output_labels, epoch=epoch)

            if epoch < 2:
                pfname_ = '{}/param.json'.format(args.exp_dir)
                with open(pfname_, "w") as pfd:
                    json.dump(vars(args), pfd, sort_keys=True, indent=4)

def start_train(args):
    criterion = get_criterion(args)
    model, raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer, lr_scheduler = get_NTS(args)

    train_loader, val_loader, test_loader, test_dset = get_data_loader(args)
    global writer
    writer = tbx.SummaryWriter(args.log_dir)

    train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            test_dset=test_dset,
            args=args,
            raw_optimizer=raw_optimizer,
            concat_optimizer=concat_optimizer,
            part_optimizer=part_optimizer,
            partcls_optimizer=partcls_optimizer,
            lr_scheduler=lr_scheduler,
            model=model,
            criterion=criterion,
    )