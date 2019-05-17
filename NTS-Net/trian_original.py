import os
import torch.utils.data
from torch.nn import DataParallel
import datetime
from torch.optim.lr_scheduler import MultiStepLR
# from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import modelNTS as model
from core.utils import init_log, progress_bar
from data_loader import get_data_loader
from parameter import get_parameters, save_exp_info
from util import AverageMeter, adjust_learning_rate, TopKAccuracyMicroAverageMeter, F1MicroAverageMeter, F1MicroAverageMeterByTopK
import time
import argparse
from sys import exit


# Parser
parser = argparse.ArgumentParser(description='iFood 2019 challenge')
parser.add_argument('--gpu', '-g', default='0', type=str, help='gpu id')
parser.add_argument('--name', '-n', default='', type=str, help='gpu id')
config = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

args = get_parameters()


# read dataset
trainloader, valloader, testloader, test_dset = get_data_loader(args)

# Tensoroard
writer = tbx.SummaryWriter(save_dir)

# define model
net = model.attention_net(args)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)

start_time = time.time()
for epoch in range(start_epoch, 500):
    for scheduler in schedulers:
        scheduler.step()
    correct = 0
    total = 0
    acc = 0

    # begin training
    _print('--' * 50)
    net.train()
    total_loss_meter = AverageMeter()
    raw_loss_meter = AverageMeter()
    rank_loss_meter = AverageMeter()
    concat_loss_meter = AverageMeter()
    partcls_loss_meter = AverageMeter()

    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
        part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = model.ranking_loss(top_n_prob, part_loss)
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
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
        # progress_bar(i, len(trainloader), 'train')

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size
                progress_bar(i, len(trainloader), 'eval train set')

        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))

	# evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(testloader), 'eval test set')

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))

	# save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

print('finishing training')
