import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
"""
    CompletionFormer
    ======================================================================

    CompletionFormer implementation
"""

from model.completionformer.nlspn_module import NLSPN

class SimMat(nn.Module):
    def __init__(self, args):
        from collections import OrderedDict
        super(SimMat, self).__init__()

        self.args = args
        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        pretrain = torch.load(args.pretrained_completionformer, map_location='cpu')['net']

        weights = OrderedDict()
        weights['0.weight'] = pretrain['backbone.conv1_rgb.0.weight']
        weights['0.bias'] = pretrain['backbone.conv1_rgb.0.bias']
        self.backbone = SimMatBackbone(args, weights, mode='rgbd')

        if self.prop_time > 0:
            self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                    self.args.prop_kernel)

    def forward(self, sample):
        rgb = sample['rgb']
        pol = sample['pol']
        dep = sample['dep']

        pred_init, guide, confidence = self.backbone(pol, dep)
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

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine=True) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels)[None,:, None,None])
            self.bias = nn.Parameter(torch.zeros(num_channels)[None,:, None,None])
        else:
            self.weight = 1
            self.bias = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers
    
class SimMatBackbone(nn.Module):
    def __init__(self, args, pretrain, mode='rgbd'):
        from model.completionformer.resnet_cbam import BasicBlock
        from model.completionformer.pvt import PVT
        super(SimMatBackbone, self).__init__()
        self.args = args
        self.mode = mode
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1
        
        # -- SimMat stuff --
        pretrained_weights = copy.deepcopy(pretrain)
        
        target_channels = 7 # assumes using leichenyang representation (Iun(1) + DoP(1) + AoP(2) + vd(3) = input(7))
        self.moe = nn.Sequential(nn.Conv2d(target_channels, target_channels * 4, 3, 1, padding=1),
                                 LayerNorm2d(target_channels * 4),
                                 nn.ReLU(),
                                 nn.Conv2d(target_channels * 4, target_channels, 3, 1, padding=1),
                                 nn.AdaptiveAvgPool2d((1, 1))
                                 )

        # frozen vision branch
        self.zip_rgbx = nn.Conv2d(target_channels, 3, 1, 1)
        self.conv1_rgb_frozen = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                          bn=False)  # same to pretrain
        # print(self.conv1_rgb_frozen.state_dict())
        # exit()
        self.conv1_rgb_frozen.load_state_dict(pretrained_weights, strict=True)
        self.conv1_rgb_frozen.requires_grad_(False)

        # learnable modality branch
        self.conv1_rgb_learnable = nn.ModuleList()
        pretrained_weights['0.weight'] = pretrained_weights['0.weight'].sum(1, keepdim=True) # squash the conv layer
        for _ in range(target_channels):
            module = conv_bn_relu(1, 48, kernel=3, stride=1, padding=1,
                                          bn=False)
            module.load_state_dict(pretrained_weights, strict=True)
            self.conv1_rgb_learnable.append(module)
        # zero-init. conv
        self.fc = nn.Linear(target_channels, 1, bias=False)
        nn.init.zeros_(self.fc.weight)

        # Encoder
        # assumes mode is rgbd
        # self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
        #                                 bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                        bn=False)
        self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                    bn=False)

        self.former = PVT(in_chans=64, patch_size=2, pretrained='ckpts/pvt.pth',)

        channels = [64, 128, 64, 128, 320, 512]
        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            convt_bn_relu(channels[5], 256, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(256, 256, stride=1, downsample=None, ratio=16),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            convt_bn_relu(256+channels[4], 128, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(128, 128, stride=1, downsample=None, ratio=8),

        )
        # 1/4
        self.dec4 = nn.Sequential(
            convt_bn_relu(128 + channels[3], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/2
        self.dec3 = nn.Sequential(
            convt_bn_relu(64 + channels[2], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/1
        self.dec2 = nn.Sequential(
            convt_bn_relu(64 + channels[1], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )


        # Init Depth Branch
        # 1/1
        self.dep_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                     padding=1)
        self.dep_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                     padding=1, bn=False, relu=True)
        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(64+channels[0], 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+channels[0], 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Sigmoid()
            )

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, pol=None, depth=None):
        # Encoding
        # fe1_rgb = self.conv1_rgb(rgb)
        weights = torch.sigmoid(self.moe(pol))
        frozen_branch, learnable_branch = pol * weights, pol * (1 - weights)
        # frozen vision branch
        frozen_embedding = self.conv1_rgb_frozen(self.zip_rgbx(frozen_branch))
        # learnable modality branch
        learnable_embedding = []
        bs, c, h, w = pol.shape
        for idx in range(c):
            embedding = self.conv1_rgb_learnable[idx](learnable_branch[:, idx:idx+1])
            learnable_embedding.append(embedding)
        learnable_embedding = torch.stack(learnable_embedding,dim=-1)
        learnable_embedding = self.fc(learnable_embedding)[..., 0]
        fe1_rgb = frozen_embedding + learnable_embedding
        
        fe1_dep = self.conv1_dep(depth)
        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        fe1 = self.conv1(fe1)

        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        dep_fd1 = self.dep_dec1(self._concat(fd2, fe2))
        init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        return init_depth, guide, confidence