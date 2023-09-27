import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cbam import BasicBlock
from .pvt_vpt_v1 import PVTVPTV1
from .backbone import Backbone


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

class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output

class BackboneVPTV1(nn.Module):
    def __init__(self, args, foundation, mode='rgbd'):
        super(BackboneVPTV1, self).__init__()
        self.args = args
        self.mode = mode
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        self.conv0_0 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=128, out_channels=48, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea()

        for param in foundation.parameters():
            param.requires_grad = False

        # Encoder
        if mode == 'rgbd':
            self.conv1_rgb = foundation.conv1_rgb
            self.conv1_dep = foundation.conv1_dep
            self.conv1 = foundation.conv1
            # -- the following two are the only updating layers --
            self.prompt = nn.Parameter(torch.zeros([1, 48, 208, 272])) # TODO: make the dimensions flexible
            nn.init.uniform_(self.prompt)

            self.dep_prompt = nn.Parameter(torch.zeros([1, 16, 208, 272])) # TODO: make the dimensions flexible
            nn.init.uniform_(self.dep_prompt)

            if self.args.pol_rep == 'grayscale-4':
                self.conv1_pol_for_rgb = conv_bn_relu(4, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
                self.conv2_pol_for_rgb = conv_bn_relu(16, 128, kernel=3, stride=1, padding=1,
                                          bn=False)
                self.conv3_pol_for_rgb = conv_bn_relu(128, 256, kernel=3, stride=1, padding=1,
                                          bn=False)
                self.conv4_pol_for_rgb = conv_bn_relu(256, 48, kernel=3, stride=1, padding=1,
                                          bn=False)
                # self.conv6_pol_for_rgb = conv_bn_relu(48, 64, kernel=1, stride=1, padding=0,
                #                           bn=False)
                # self.conv7_pol_for_rgb = conv_bn_relu(64, 128, kernel=1, stride=1, padding=0,
                #                           bn=False)
                # self.conv8_pol_for_rgb = conv_bn_relu(128, 256, kernel=1, stride=1, padding=0,
                #                           bn=False)
                self.conv1_pol_for_dep = conv_bn_relu(4, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
                self.conv2_pol_for_dep = conv_bn_relu(16, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
            elif self.args.pol_rep == 'rgb-12':
                self.conv1_pol_for_rgb = conv_bn_relu(12, 48, kernel=3, stride=1, padding=1,
                                          bn=False)
                self.conv1_pol_for_dep = conv_bn_relu(12, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
            elif self.args.pol_rep == 'leichenyang-7':
                self.conv1_pol_for_rgb = conv_bn_relu(7, 48, kernel=3, stride=1, padding=1,
                                          bn=False)
                self.conv1_pol_for_dep = conv_bn_relu(7, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        elif mode == 'rgb':
            self.conv1 = foundation.conv1
        elif mode == 'd':
            self.conv1 = foundation.conv1
        else:
            raise TypeError(mode)
        
        # self.former = foundation.former
        self.former = PVTVPTV1(in_chans=64, patch_size=2, pretrained='./model/completionformer_original/pretrained/pvt.pth', foundation=foundation.former)

        # Shared Decoder
        # 1/16
        self.dec6 = foundation.dec6
        # 1/8
        self.dec5 = foundation.dec5
        # 1/4
        self.dec4 = foundation.dec4

        # 1/2
        self.dec3 = foundation.dec3

        # 1/1
        self.dec2 = foundation.dec2

        # Init Depth Branch
        # 1/1
        self.dep_dec1 = foundation.dep_dec1
        self.dep_dec0 = foundation.dep_dec0
        # Guidance Branch
        # 1/1
        self.gd_dec1 = foundation.gd_dec1
        self.gd_dec0 = foundation.gd_dec0

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = foundation.cf_dec1
            self.cf_dec0 = foundation.cf_dec0 

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb=None, depth=None, pol=None):
        B = rgb.shape[0]
        # Encoding
        if self.mode == 'rgbd':
            self.conv1_rgb.eval()
            fe1_rgb = self.conv1_rgb(rgb)

            # -- v1.0x: prompt parameters --
            # fe1_rgb = fe1_rgb + self.prompt.expand(B, -1, -1, -1)

            # -- v1.1x: polarization feature with gradient update --
            # fe1_pol_for_rgb = self.conv4_pol_for_rgb(\
            #                 self.conv3_pol_for_rgb(\
            #                 self.conv2_pol_for_rgb(\
            #                 self.conv1_pol_for_rgb(pol))))
            # -- v1.12: also add prompt --
            fe1_pol_for_rgb = self.conv4_pol_for_rgb(\
                            self.conv3_pol_for_rgb(\
                            self.conv2_pol_for_rgb(\
                            self.conv1_pol_for_rgb(pol)))) + self.prompt

            # -- v1.11: direct addition of the polarization feature with the rgb feature --
            fe1_rgb = fe1_rgb + fe1_pol_for_rgb


            # -- v1.13: fusion via ViPT way --
            # x0 = fe1_rgb.contiguous()
            # x0 = self.conv0_0(x0)
            # x1 = fe1_pol_for_rgb.contiguous()
            # x1 = self.conv0_1(x1)
            # x0 = self.fovea(x0) + x1
            # fe1_rgb = self.conv1x1(x0) + fe1_rgb
            self.conv1_dep.eval()
            fe1_dep = self.conv1_dep(depth)

            # -- v1.14: also with pol features added to dep
            # -- v1.15: add prompt to dep feature as well
            fe1_pol_for_dep = self.conv2_pol_for_dep(self.conv1_pol_for_dep(pol)) + self.dep_prompt
            # fe1_dep = fe1_dep + fe1_pol_for_dep

            # print("--> Dimension of fe1_rgb {}".format(fe1_rgb.shape))
            fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

            self.conv1.eval()
            fe1 = self.conv1(fe1)
        elif self.mode == 'rgb':
            fe1 = self.conv(rgb)
        elif self.mode == 'd':
            fe1 = self.conv(depth)
        else:
            raise TypeError(self.mode)

        self.former.eval()
        # fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1, torch.cat([fe1_pol_for_rgb, fe1_pol_for_dep], dim=1))
        # Shared Decoding
        self.dec6.eval()
        self.dec5.eval()
        self.dec4.eval()
        self.dec3.eval()
        self.dec2.eval()
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        self.dep_dec1.eval()
        self.dep_dec0.eval()
        dep_fd1 = self.dep_dec1(self._concat(fd2, fe2))
        init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))

        # Guidance Decoding
        self.gd_dec1.eval()
        self.gd_dec0.eval()
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            self.cf_dec1.eval()
            self.cf_dec0.eval()
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        return init_depth, guide, confidence

