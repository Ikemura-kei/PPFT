import torch
import torch.nn as nn
from .comp.completionformer import CompletionFormer
from .sfp_wild.transunet import TransUnet


class PDNE(nn.Module):
    def __init__(self, args):
        super(PDNE, self).__init__()
        self.completion_model = CompletionFormer(args)
        self.normal_model = TransUnet(args)

        self.completion_model

        self.fusion = nn.Conv2d(9, 4, 1)

        if args.fix_completion:
            for param in self.completion_model.parameters():
                param.requires_grad = False
        
        if args.fix_normal:
            for param in self.mormal_model.parameters():
                param.requires_grad = False

    def forward(self, net_in, coarse_depth, images):
        pol_norm = self.normal_model(net_in)
        comp_depth = self.completion_model(torch.cat(images, coarse_depth, dim=1))
        out = self.fusion(torch.cat(comp_depth, coarse_depth, pol_norm))
        depth = out[:, 0:1, :, :]
        norm = out[:, 1:4, :, :]

        return depth, norm

