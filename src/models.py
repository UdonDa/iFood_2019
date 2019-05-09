import torch
import torch.nn as nn
import torch.nn.init as weight_init
import pretrainedmodels


class FCWithLogSigmoid(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FCWithLogSigmoid, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.logsigmoid = nn.LogSigmoid()
    
    def forward(self, x):
        return self.logsigmoid(self.linear(x))

def create_model(args):
    models = pretrainedmodels
    if args.pretrained:
        print('=> Using pre-trained model `{}`'.format(args.arch))

        if args.arch == 'pnasnet5large' or args.arch == 'nasnetalarge':
            model = models.__dict__[args.arch](pretrained='imagenet+background')
            
        elif args.arch == 'senet154':            
            model = models.__dict__[args.arch](pretrained='imagenet')
        elif args.arch == 'resnext10132x4d':
            model = models.__dict__["resnext101_32x4d"](pretrained='imagenet')

    else:
        print('=> From schratch model `{}`'.format(args.arch))
        model = models.__dict__[arch]()

    if args.resolution > 1:
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    if args.loss_type == 'BCEWithLogitsLoss':
        model.last_linear = FCWithLogSigmoid(args.fv_size, args.num_labels)
    if args.loss_type == 'CrossEntropyLoss':
        model.last_linear = nn.Linear(args.fv_size, args.num_labels)

    # if args.arch.startswith('alexnet') or args.arch.starswith('vgg'):
    #     model.features = nn.DataParallel(model.features).cuda()
    # else:
    model = nn.DataParallel(model).cuda()
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

    