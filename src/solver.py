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


from models import create_model, WeightUpdateTracker
from util import AverageMeter, adjust_learning_rate, TopKAccuracyMicroAverageMeter, F1MicroAverageMeter, F1MicroAverageMeterByTopK, MyPredictor
from data_loader import get_data_loader
from parameter import mkdir_exp_dir
from logger import Logger



    
def get_model(args):
    model = create_model(args)
    return model


def get_criterion(args):
    criterion = None
    if args.loss_type == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()

    return criterion


def get_optimizer(args, model):
    optimizer = None
    print('Use optimizer: ', args.optimizer)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
                        model.parameters(),
                        amsgrad=args.amsgrad,
                        lr=args.lr,
                        betas=(args.beta1, args.beta2),
                        eps=args.small,
                        weight_decay=args.weight_decay
                        )

    elif args.optimizer == 'Sgd':
        optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=args.nesterov
        )
    elif args.optimizer == 'AdaBound':
        from adabound import AdaBound
        optimizer = AdaBound(
                        model.parameters(),
                        lr=args.lr,
                        betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr,
                        gamma=args.gamma
        )

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
    return lr_scheduler


def load_checkpoint(model, optimizer, lr_scheduler, args, resume=True, ckpt=None):
    """optionally resume from a checkpoint."""
    best_top3 = 0
    if args.resume:
        if os.path.isfile(ckpt):
            print("=> loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            args.start_epoch = checkpoint['epoch']
            best_top3 = checkpoint['best_top3']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        #  scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt))
            best_top3 = 0
    return (model, optimizer, lr_scheduler, args, best_top3)


def save_checkpoint(state, is_best, args, epoch, best_top3):
    best_top3 = str(best_top3)[:6]
    a, b, c = args.ckpt.split('_')
    filename = '{}_{}-{}-{}-{}'.format(a, b, epoch, best_top3, c)
    a, b, c = args.best.split('_')
    best_model_filename = '{}_{}-{}-{}-{}'.format(a, b, epoch, best_top3, c)
    torch.save(state, filename)
    print('Sucess to save model: ', filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)


def train(args, train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    top3 = TopKAccuracyMicroAverageMeter(k=3)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()
        # compute output

        # # TODO: save image
        # torchvision.utils.save_image(input, './sample.png', normalize=True)
        # print('Success save image')
        
        print('input.size(): ', input.size())
        # output = model.module.features(input)
        output = model(input)
        print(output.size())

        exit()
        loss = criterion(output, target)

        # measure top-3 accuracy and record loss
        loss_meter.update(loss.item(), input.size(0))
        top3.update(target, torch.exp(output))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                'Top-3 {top3.accuracy:.3f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss_meter=loss_meter, top3=top3))
            # break # TODO: Debug
            
            
    print('TRAIN: [{epoch}]\t'
        'Time {epoch_time:.3f}\t'
        'Data {epoch_data_time:.3f}\t'
        'Loss {loss_meter.avg:.4f}\t'
        'Top-3 {top3.accuracy:.3f}'.format(epoch=epoch,
                                            epoch_time=batch_time.sum,
                                            epoch_data_time=data_time.sum,
                                            loss_meter=loss_meter,
                                            top3=top3))
    writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/top3', top3.accuracy, epoch)
    return writer


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    top3 = TopKAccuracyMicroAverageMeter(k=3)


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        loss_avg_epoch = 0.0
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure F1 and record loss
            loss_meter.update(loss.item(), input.size(0))
            top3.update(target, torch.exp(output))
            loss_avg_epoch = float(i * loss.item()) / float(i + 1)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Val: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    'Top-3 {top3.accuracy:.3f}'.format(
                    i, len(val_loader), batch_time=batch_time, loss_meter=loss_meter,
                    top3=top3))
                # break # TODO: Debug

#         print(' * Top-3 Accuracy {top3.accuracy:.3f}'
#               .format(top3=top3))
        print('VAL: [{epoch}]\t'
                    'Time {epoch_time:.3f}\t'
                    'Loss {loss_meter.avg:.4f}\t'
                    'Top-3 {top3.accuracy:.3f}'.format(epoch=epoch,
                                                        epoch_time=batch_time.sum,
                                                        loss_meter=loss_meter,
                                                        top3=top3))
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/top3', top3.accuracy, epoch)

    return top3.accuracy, loss_avg_epoch, writer


def test(ofname, pfname, args, test_dset,
        test_loader, best_model_ckpt, model,
        num_output_labels=3, epoch=0):
    
#     checkpoint = torch.load(best_model_ckpt)
#     model.load_state_dict(checkpoint['state_dict'])

    # predictor = MyPredictor(model, args.edata_json)
    
    batch_time = AverageMeter()
    res = OrderedDict()

    # switch to evaluate mode
    model.eval()
    
    res = OrderedDict()

    # ofname_ = "%s%s%03d_%s" % (os.path.dirname(ofname), os.sep, epoch, os.path.basename(ofname))
    ofname_ = '{}/{}.csv'.format(args.sub_dir, epoch)
    with open(ofname_, "w") as ofd:
        ofd.write("img_name,label\n")
        with torch.no_grad():
            end = time.time()
            index = 0
            for i, (input, _) in enumerate(test_loader):
                # compute output
                input = input.cuda()
                # TODO: TTAに変える
                output = model(input)
                # print('input.size(): ', input.size())
                # preds_with = predictor.predict_images(input)

                res = torch.exp(output).topk(num_output_labels, dim=1)[1].cpu().numpy().tolist()
                # measure elapsed time
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
        model, optimizer, lr_scheduler, args, best_top3 = load_checkpoint(model, optimizer, lr_scheduler, args, resume=args.resume, ckpt=args.ckpt)
        wut = None
        writer = writer
        if args.debug_weights:
            wut = WeightUpdateTracker(model)
        for epoch in range(args.start_epoch, args.epochs):
            writer = train(args, train_loader, model, criterion, optimizer, epoch, writer)

            if args.debug_weights:
                wut.track(model)
                print('wut: ', wut)

            top3, val_loss, writer = validate(val_loader, model, criterion, epoch, writer)

            is_best = top3 > best_top3
            print('is_best: ', is_best)
            best_top3 = max(top3, best_top3)
            mkdir_exp_dir(args)
            if is_best:
                print("BEST at epoch: ", epoch)
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_top3': best_top3,
                    'optimizer' : optimizer.state_dict(),
                    # 'scheduler' : scheduler.state_dict(),
                    }, is_best, args, epoch, best_top3)
            test(args.output_file, args.params_file, args, test_dset, test_loader, args.best, model, num_output_labels=args.num_output_labels, epoch=epoch)

            adjust_learning_rate(optimizer, lr_scheduler, epoch, val_loss, args)

            if epoch < 2:
                pfname_ = '{}/param.json'.format(args.exp_dir)
                with open(pfname_, "w") as pfd:
                    json.dump(vars(args), pfd, sort_keys=True, indent=4)

def start_train(args):
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model)
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