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
import numpy as np

from models import create_model, WeightUpdateTracker
from util import AverageMeter, adjust_learning_rate, TopKAccuracyMicroAverageMeter, F1MicroAverageMeter, F1MicroAverageMeterByTopK, MyPredictor
from data_loader import get_data_loader
from parameter import mkdir_exp_dir
from logger import Logger


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """mixup criterion"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
def get_model(args):
    model, clasify = create_model(args)
    return model, clasify


def get_criterion(args):
    criterion = None
    if args.loss_type == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.loss_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()    

    return criterion


def get_optimizer(args, model, classifier_layers):
    optimizer = None

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key in classifier_layers:
            params += [{'params':[value],'lr':args.lr*10.}]
        else:
            params += [{'params':[value],'lr':args.lr}] # args.lr*0.


    print('=> Use optimizer: ', args.optimizer)
    if args.optimizer == 'Adam':
        if args.all_parameter_freeze:
            if args.arch == 'resnet18' or args.arch == 'resnet152':
                optimizer = torch.optim.Adam(model.module.fc.parameters(),amsgrad=args.amsgrad,lr=args.lr,betas=(args.beta1, args.beta2),eps=args.small,weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(model.module.last_linear.parameters(),amsgrad=args.amsgrad,lr=args.lr,betas=(args.beta1, args.beta2),eps=args.small,weight_decay=args.weight_decay)
        else:
            # optimizer = torch.optim.Adam(model.parameters(),amsgrad=args.amsgrad,lr=args.lr,betas=(args.beta1, args.beta2),eps=args.small,weight_decay=args.weight_decay)
            optimizer = torch.optim.Adam(params, amsgrad=args.amsgrad,lr=args.lr,betas=(args.beta1, args.beta2),eps=args.small,weight_decay=args.weight_decay)

    elif args.optimizer == 'Sgd':
        if args.all_parameter_freeze:
            if args.arch == 'resnet18' or args.arch == 'resnet152':
                optimizer = torch.optim.SGD(model.module.fc.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=args.nesterov)
            else:
                optimizer = torch.optim.SGD(model.module.last_linear.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=args.nesterov)
        else:
            # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=args.nesterov)
            optimizer = torch.optim.SGD(params,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=args.nesterov)

    elif args.optimizer == 'AdaBound':
        from adabound_optim import AdaBound
        if args.all_parameter_freeze:
            if args.arch == 'resnet18' or args.arch == 'resnet152':
                optimizer = AdaBound(model.module.fc.parameters(),lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)
            else:
                optimizer = AdaBound(model.module.last_linear.parameters(),lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)
        else:
            # optimizer = AdaBound(model.parameters(),lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)
            optimizer = AdaBound(params,lr=args.lr,betas=(args.beta1, args.beta2),final_lr=args.final_lr,gamma=args.gamma)
    return optimizer


def get_lr_scheduler(args, optimizer):
    lr_scheduler = None
    if args.lr_scheduler == 'ReduceLROnPlateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            # mode='max',   # accuracy measure
                            mode='min',   # loss
                            patience=args.scheduler_patience,
                            threshold=args.scheduler_threshold,
                            factor=args.scheduler_factor,
                            verbose=1
                            )
    if args.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.T_max,
                            eta_min=args.eta_min,
                            last_epoch=args.last_epoch
                            )
    return lr_scheduler


def load_checkpoint(model, optimizer, lr_scheduler, args, resume=True, ckpt=None):
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
        #  scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt))
            best_top3 = 0
    return (model, optimizer, lr_scheduler, args, best_top3, best_acc)


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


def train(args, train_loader, model, criterion, optimizer, epoch, writer):
    loss_meter = AverageMeter()
    top3 = TopKAccuracyMicroAverageMeter(k=3)
    correct = 0
    total = 0
    acc = 0

    model.train()

    start_time = time.time()

    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda().long()

        # Mixup
        input, target_a, target_b, lam = mixup_data(input, target, args.alpha)

        output = model(input)

        # loss = criterion(output, target)
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)

        loss_meter.update(loss.item(), input.size(0))
        top3.update(target, torch.exp(output))

        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
        acc = correct / total

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]

        if i % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                'Top-3 {top3.accuracy:.3f}\t Acc {acc:.4f}'.format(
                epoch, i, len(train_loader), loss_meter=loss_meter, top3=top3, acc=acc))

        # if i % 10 == 0:
        #     break # TODO: Debug
            
            
    print('TRAIN: [{epoch}]\t'
        'Loss {loss_meter.avg:.4f}\t'
        'Top-3 {top3.accuracy:.3f}\t'
        'Acc {acc:.4f}'.format(epoch=epoch,
                            loss_meter=loss_meter,
                            top3=top3,
                            acc=acc))
    writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/top3', top3.accuracy, epoch)
    writer.add_scalar('Train/Acc', acc, epoch)
    return writer


def validate(args, val_loader, model, criterion, epoch, writer):
    loss_meter = AverageMeter()
    top3 = TopKAccuracyMicroAverageMeter(k=3)
    acc = 0
    correct = 0
    total = 0


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        loss_avg_epoch = 0.0

        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda().long()

            # output = model(input) # When you do not use Ten crops.

            bs, ncrops, c, h, w = input.size() # When you use Ten crops.
            output = model(input.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1)

            loss = criterion(output, target)

            # compute accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = correct / total

            # measure F1 and record loss
            loss_meter.update(loss.item(), input.size(0))
            top3.update(target, torch.exp(output))
            loss_avg_epoch = float(i * loss.item()) / float(i + 1)
            
            # measure elapsed time
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]

            if i % 10 == 0:
                print('Val: [{0}/{1}]\t'
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    'Top-3 {top3.accuracy:.3f}\t'
                    'Acc {acc:.4f}'.format(
                    i, len(val_loader), loss_meter=loss_meter,
                    top3=top3, acc=acc))

                # if i % 10 == 0:
                #     break # TODO: Debug

#         print(' * Top-3 Accuracy {top3.accuracy:.3f}'
#               .format(top3=top3))
        print('VAL: [{epoch}]\t'
                    'Loss {loss_meter.avg:.4f}\t'
                    'Top-3 {top3.accuracy:.3f}\t'
                    'Acc {acc:.4f}'.format(epoch=epoch,
                                        loss_meter=loss_meter,
                                        top3=top3,
                                        acc=acc))
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/top3', top3.accuracy, epoch)
    writer.add_scalar('Val/Acc', acc, epoch)

    return top3.accuracy, loss_avg_epoch, acc, writer


def test(ofname, pfname, args, test_dset,
        test_loader, best_model_ckpt, model,
        num_output_labels=3, epoch=0):
    
#     checkpoint = torch.load(best_model_ckpt)
#     model.load_state_dict(checkpoint['state_dict'])
    
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


def train_loop(train_loader=None, val_loader=None, test_loader=None, test_dset=None, args=None, optimizer=None, lr_scheduler=None, model=None, criterion=None, num_output_labels=3, writer=None):
    if args.evaluate:
        validate(val_loader, model, criterion)
    else:
        model, optimizer, lr_scheduler, args, best_top3, best_acc = load_checkpoint(model, optimizer, lr_scheduler, args, resume=args.resume, ckpt=args.pretrained_model_path)
        wut = None
        writer = writer
        if args.debug_weights:
            wut = WeightUpdateTracker(model)
        for epoch in range(args.start_epoch, args.epochs):
            writer = train(args, train_loader, model, criterion, optimizer, epoch, writer)

            if args.debug_weights:
                wut.track(model)
                print('wut: ', wut)

            top3, val_loss, acc,writer = validate(args, val_loader, model, criterion, epoch, writer)

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
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : lr_scheduler.state_dict(),
                    }, is_best, args, epoch, best_acc)
                test(args.output_file, args.params_file, args, test_dset, test_loader, args.best, model, num_output_labels=args.num_output_labels, epoch=epoch)
            adjust_learning_rate(optimizer, lr_scheduler, epoch, val_loss, args)

            if epoch < 2:
                pfname_ = '{}/param.json'.format(args.exp_dir)
                with open(pfname_, "w") as pfd:
                    json.dump(vars(args), pfd, sort_keys=True, indent=4)


def start_train(args):
    model, clasify = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model, clasify)
    lr_scheduler = get_lr_scheduler(args, optimizer)

    train_loader, val_loader, test_loader, test_dset = get_data_loader(args)
    writer = tbx.SummaryWriter(args.log_dir)

    train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            test_dset=test_dset,
            args=args,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            model=model,
            criterion=criterion,
            writer=writer
    )