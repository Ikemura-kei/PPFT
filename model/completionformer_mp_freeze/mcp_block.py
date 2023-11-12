import torch
import torch.nn as nn

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

class MCPBlock(nn.Module):
    def __init__(self, in_dim1, in_dim2, hidden_dim):
        super(MCPBlock, self).__init__()

        self.fovea = Fovea()
        self.conv0_0 = nn.Conv2d(in_channels=in_dim1, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=in_dim2, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hidden_dim, out_channels=in_dim1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, complementary):
        x0 = x.contiguous()
        x0 = self.conv0_0(x0)

        x1 = complementary.contiguous()
        x1 = self.conv0_1(x1)

        x2 = self.fovea(x0) + x1

        return self.conv1x1(x2) 