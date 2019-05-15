import torch
import torch.nn as nn
import torch.nn.init as weight_init
import pretrainedmodels
from torchvision.models import resnet18, resnet152


class FCWithLogSigmoid(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FCWithLogSigmoid, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.logsigmoid = nn.LogSigmoid()
    
    def forward(self, x):
        return self.logsigmoid(self.linear(x))

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


def create_model(args):
    models = pretrainedmodels
    if args.pretrained:
        print('=> Using pre-trained model `{}` - {}'.format(args.arch, args.resolution))

        if args.arch == 'pnasnet5large' or args.arch == 'nasnetalarge':
            model = models.__dict__[args.arch](pretrained='imagenet+background')
        elif args.arch == 'senet154' or args.arch == 'polynet' or args.arch == 'inceptionresnetv2' or args.arch == 'inceptionv4':
            model = models.__dict__[args.arch](pretrained='imagenet')
        elif args.arch == 'resnext10132x4d':
            model = models.__dict__["resnext101_32x4d"](pretrained='imagenet')

        elif args.arch == 'resnet18':
            model = resnet18(pretrained=args.pretrained)
        elif args.arch == 'resnet152':
            model = resnet152(pretrained=args.pretrained)
    else:
        print('=> From schratch model `{}`'.format(args.arch))
        if args.arch == 'resnet18':
            model = resnet18(pretrained=args.pretrained)
        else:
            model = models.__dict__[arch]()

    if args.resolution > 1:
        if args.arch == 'resnet18' or args.arch == 'resnet152': # Torchvision
            model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    # Freeze parameter
    model = freeeze_parameter(model, args)
    classifier_layers = None

    if args.loss_type == 'BCEWithLogitsLoss':
        model.last_linear = FCWithLogSigmoid(args.fv_size, args.num_labels)
    if args.loss_type == 'CrossEntropyLoss':
        if args.arch == 'resnet18' or args.arch == 'resnet152':
            if args.pre_learned:
                model.fc = nn.Linear(args.fv_size, 100)
            else:
                model.fc = nn.Linear(args.fv_size, args.num_labels)
            classifier_layers = ['fc.weight','fc.bias']
        else:
            if args.pre_learned:
                model.last_linear = nn.Linear(args.fv_size, 100)
            else:
                model.last_linear = nn.Linear(args.fv_size, args.num_labels)
            classifier_layers = ['last_linear.weight','last_linear.bias']

    # if args.arch.startswith('alexnet') or args.arch.starswith('vgg'):
    #     model.features = nn.DataParallel(model.features).cuda()
    # else:
    model = nn.DataParallel(model)
    # model = model.to('cuda')


    if args.pre_learned:
        print('==> Resuming from checkpoint..')
        pretrained_path = '/home/yanai-lab/horita-d/ifood/uecfood100/checkpoint/'
        if args.arch == 'pnasnet5large':
            pretrained_path += 'ckpt_pnas-adabound-2_epoch_38_acc_82.04600068657741.t7'
        elif args.arch == 'resnext10132x4d':
            pretrained_path += 'ckpt_resnext-adabound-2_epoch_41_acc_73.66975626501888.t7'
        elif args.arch == 'nasnetalarge':
            pretrained_path += 'ckpt_nas-adabound-2_epoch_41_acc_80.63851699279094.t7'
        elif args.arch == 'senet154':
            pretrained_path += 'ckpt_senet-adabound-2_epoch_32_acc_83.7624442155853.t7'
        elif args.arch == 'inceptionresnetv2':
            pretrained_path += 'ckpt_incepresv2-adabound-2_epoch_104_acc_81.66838311019568.t7'
            # pretrained_path += 'ckpt_incepv4-adabound-2_epoch_113_acc_80.32955715756951.t7'
        elif args.arch == 'inceptionv4':
            pretrained_path += 'ckpt_incepv4-adabound-2_epoch_113_acc_80.32955715756951.t7'
            # pretrained_path += 'ckpt_incepresv2-adabound-2_epoch_104_acc_81.66838311019568.t7'

        checkpoint = torch.load(pretrained_path)
        # state = convert_state_dict(checkpoint['net'])
        state = checkpoint['net']
        model.load_state_dict(state)
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']

        if args.arch == 'resnet18' or args.arch == 'resnet152':
            model.module.fc = nn.Linear(args.fv_size, args.num_labels)
        else:
            model.module.last_linear = nn.Linear(args.fv_size, args.num_labels)

        # print(model)
        
        print('=> Loaded UECFOOD100 model...')


    return model.to('cuda'), classifier_layers


def freeeze_parameter(model, args):
    if args.all_parameter_freeze:
        print('=> Freeze all parameters')
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("=> Train all layers.")
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WeightUpdateTracker:
    
    def __init__(self, model):
        with torch.no_grad():
            self.num_param_tensors = len(list(model.parameters()))
            self.prev_pnorms = torch.zeros(self.num_param_tensors) 
            self.curr_pnorms = self.parameter_norms(model) 

    def parameter_norms(self, model):
        with torch.no_grad():
            pnorms = torch.zeros(self.num_param_tensors)
            for i, x in enumerate(list(model.parameters())):
                pnorms[i] = x.norm().item()
            return pnorms
        
    def track(self, model):
        with torch.no_grad():
            self.prev_pnorms = self.curr_pnorms.clone()
            self.curr_pnorms = self.parameter_norms(model)
            self.delta = (self.curr_pnorms - self.prev_pnorms) / self.prev_pnorms

            
    def __repr__(self):
        with torch.no_grad():
            return self.delta.__repr__()

if __name__ == '__main__':
    from parameter import get_parameters
    args = get_parameters()
    model = create_model(args)

    print("Neural Network has ", count_parameters(model), " trainable parameters")

    