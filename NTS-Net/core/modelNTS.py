from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM
from core.premodels.models import *



def convert_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict



class ProposalNet(nn.Module):
    def __init__(self, args):
        super(ProposalNet, self).__init__()
        self.args = args
        self.down1 = nn.Conv2d(args.fv_size, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)
        self.fc = nn.Linear(1161, 1614) # TODO: inceptionresnetv2, inceptionv4

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        f = torch.cat((t1, t2, t3), dim=1)

        if self.args.arch == 'inceptionresnetv2' or self.args.arch == 'inceptionv4':
            f = self.fc(f) # TODO: inceptionresnetv2

        return f



def create_model_uecfood(args):
    model = None
    if args.library_type == 'torchvisions':
        if args.arch == 'resnet50':
            model = resnet.resnet50(pretrained=True)
        elif args.arch == 'resnet152':
            model = resnet.resnet152(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        if args.pre_learned:
            model.fc = nn.Linear(args.fv_size, args.num_labels_uecfood100)
        else:
            model.fc = nn.Linear(args.fv_size, args.num_labels)

    elif args.library_type == 'Pretrainedmodels':
        if args.arch == 'pnasnet5large':
            model = pnasnet5large(pretrained=False)
        elif args.arch == 'nasnetalarge':
            model = nasnetalarge(pretrained=False)
        elif args.arch == 'senet154':
            # model = senet154(pretrained=None)
            model = senet154(pretrained=None)
        elif args.arch == 'polynet':
            model = polynet(pretrained=False)
        elif args.arch == 'inceptionresnetv2':
            model = inceptionresnetv2(pretrained=False)
        elif args.arch == 'inceptionv4':
            model = inceptionv4(pretrained=False)
        elif args.arch == 'resnext10132x4d':
            model = resnext101_32x4d(pretrained=None)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

        if args.pre_learned:
            model.last_linear = nn.Linear(args.fv_size, args.num_labels_uecfood100)
        else:
            model.last_linear = nn.Linear(args.fv_size, args.num_labels)

    if args.pre_learned:
        print('==> Resuming from checkpoint..')
        pretrained_path = '/home/yanai-lab/horita-d/ifood/uecfood100/checkpoint/'
        if args.arch == 'pnasnet5large':
            pretrained_path += 'ckpt_pnas-adabound-2_epoch_38_acc_82.04600068657741.t7'
        elif args.arch == 'resnext10132x4d':
            pretrained_path += 'ckpt_resnext-adabound-2-vv3_epoch_82_acc_84.72365259182973.t7'
        elif args.arch == 'nasnetalarge':
            pretrained_path += 'ckpt_nas-adabound-2_epoch_41_acc_80.63851699279094.t7'
        elif args.arch == 'senet154':
            # pretrained_path = '/host/space0/ege-t/works/works/classification_pytorch/uecfood100/checkpoint/ckpt_se154_b8_e300.t7'
            pretrained_path += 'ckpt_senet-adabound-2_epoch_32_acc_83.7624442155853.t7'
        elif args.arch == 'inceptionresnetv2':
            pretrained_path += 'ckpt_incepresv2-adabound-2_epoch_104_acc_81.66838311019568.t7'
        elif args.arch == 'inceptionv4':
            pretrained_path += 'ckpt_incepv4-adabound-2_epoch_113_acc_80.32955715756951.t7'
        elif args.arch == 'resnet50':
            pretrained_path = '/host/space0/ege-t/works/works/classification_pytorch/uecfood100/checkpoint/ckpt_res50_b8_e300.t7'
        elif args.arch == 'resnet152':
            pretrained_path = '/host/space0/ege-t/works/works/classification_pytorch/uecfood100/checkpoint/ckpt_res152_b8_e300.t7'

        checkpoint = torch.load(pretrained_path)
        state = convert_state_dict(checkpoint['net'])
        # state = checkpoint['net']
        model.load_state_dict(state)
        # best_acc = checkpoint['acc']

        if args.library_type == 'torchvisions':
            model.fc = nn.Linear(args.fv_size, args.num_labels)
        else:
            model.last_linear = nn.Linear(args.fv_size, args.num_labels)

        print('=> Loaded UECFOOD100 model...')
    return model



def create_model_food101(args):
    # Food101
    model = None
    if args.library_type == 'torchvisions':
        if args.arch == 'resnet50':
            model = resnet.resnet50(pretrained=True)
        elif args.arch == 'resnet152':
            model = resnet.resnet152(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        if args.pre_learned:
            model.fc = nn.Linear(args.fv_size, args.num_labels_food101)
        else:
            model.fc = nn.Linear(args.fv_size, args.num_labels)
    
    elif args.library_type == 'Pretrainedmodels':
        if args.arch == 'pnasnet5large':
            model = pnasnet5large(pretrained=False)
        elif args.arch == 'nasnetalarge':
            model = nasnetalarge(pretrained=False)
        elif args.arch == 'senet154':
            model = senet154()
        elif args.arch == 'polynet':
            model = polynet(pretrained=False)
        elif args.arch == 'inceptionresnetv2':
            print('IMagenet!!!!!!!!!!!!!!')
            model = inceptionresnetv2()
        elif args.arch == 'inceptionv4':
            model = inceptionv4()
        elif args.arch == 'resnext10132x4d':
            model = resnext101_32x4d(pretrained=None)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)

        if args.pre_learned:
            model.last_linear = nn.Linear(args.fv_size, args.num_labels_food101)
        else:
            model.last_linear = nn.Linear(args.fv_size, args.num_labels)

    if args.pre_learned:
        print('==> Resuming from checkpoint..')
        pretrained_path = '/home/yanai-lab/horita-d/ifood/food101/checkpoint/'
        if args.arch == 'pnasnet5large':
            pretrained_path += 'ckpt_pnas-adabound-2_epoch_10_acc_86.0910891089109.t7'
        elif args.arch == 'resnext10132x4d':
            pretrained_path += 'ckpt_resnext-ada_epoch_23_acc_87.70991761723701.t7'
        elif args.arch == 'nasnetalarge':
            pretrained_path += 'ckpt_nas-adabound-2_epoch_10_acc_84.52673267326733.t7'
        elif args.arch == 'senet154':
            pretrained_path += 'ckpt_senet-adabound-2_epoch_8_acc_87.2910891089109.t7'
        elif args.arch == 'inceptionresnetv2':
            pretrained_path += 'ckpt_incepresv2-adabound-2_epoch_24_acc_84.62574257425743.t7'
        elif args.arch == 'inceptionv4':
            pretrained_path += 'ckpt_incepv4-adaound-2_epoch_31_acc_85.15643564356435.t7'
        elif args.arch == 'resnet152':
            pretrained_path += 'ckpt_resnet152-weaa_epoch_29_acc_88.39540412044374.t7'
        elif args.arch == 'resnet50':
            pretrained_path += 'ckpt_resnet50-vv6_epoch_34_acc_87.0079365079365.t7'

        checkpoint = torch.load(pretrained_path)
        state = convert_state_dict(checkpoint['net'])
        # state = checkpoint['net']
        model.load_state_dict(state)
        # best_acc = checkpoint['acc']

        if args.library_type == 'torchvisions':
            model.fc = nn.Linear(args.fv_size, 251)
        else:
            model.last_linear = nn.Linear(args.fv_size, 251)

        print('=> Loaded food101 model...')
    return model


class attention_net(nn.Module):
    def __init__(self, args):
        super(attention_net, self).__init__()

        if args.pre_dataset == 'UECFOOD100':
            print('=> Using {} with UECFOOD'.format(args.arch))
            self.pretrained_model = create_model_uecfood(args)
        elif args.pre_dataset == 'FOOD101':
            print('=> Using {} with FOOD101'.format(args.arch))
            self.pretrained_model = create_model_food101(args)
        elif args.pre_dataset == 'imagenet':
            print('=> Using {} with imagenet'.format(args.arch))
            args.pre_learned = False
            self.pretrained_model = create_model_food101(args)

        self.proposal_net = ProposalNet(args)
        self.topN = args.PROPOSAL_NUM
        self.concat_net = nn.Linear(args.fv_size * (CAT_NUM + 1), args.num_labels)
        self.partcls_net = nn.Linear(args.fv_size, args.num_labels)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size
