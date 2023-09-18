from .completionformer import CompletionFormer
import torch
import torch.nn as nn
from .backbone_vpt_v1 import BackboneVPTV1

class CompletionFormerVPTV1(nn.Module):
    def __init__(self, args):
        super(CompletionFormerVPTV1, self).__init__()

        self.args = args

        self.foundation = CompletionFormer(args)
        self.foundation.load_state_dict(torch.load(args.pretrained_completionformer)['net'])
        # for param in self.foundation.parameters():
        #     param.requires_grad = False

        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1
        # print(self.foundation.keys())
        # print(self.foundation['net'].keys())
        self.backbone = BackboneVPTV1(args, self.foundation.backbone, mode='rgbd')

        if self.prop_time > 0:
            self.prop_layer = self.foundation.prop_layer

        cnt = 0
        for param in self.backbone.parameters():
            print(param.requires_grad)
            cnt+=1
            if cnt > 100:
                break
    
    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        pol = sample['pol']
        print("--> Pol shape {}".format(pol.shape))
        pred_init, guide, confidence = self.backbone(rgb, dep, pol)
        pred_init = pred_init + dep

        # Diffusion
        y_inter = [pred_init, ]
        conf_inter = [confidence, ]
        if self.prop_time > 0:
            y, y_inter, offset, aff, aff_const = \
                self.prop_layer(pred_init, guide, confidence, dep, rgb)
        else:
            y = pred_init
            offset, aff, aff_const = torch.zeros_like(y), torch.zeros_like(y), torch.zeros_like(y).mean()

        # Remove negative depth
        y = torch.clamp(y, min=0)
        # best at first
        y_inter.reverse()
        conf_inter.reverse()
        if not self.args.conf_prop:
            conf_inter = None

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': conf_inter}

        return output