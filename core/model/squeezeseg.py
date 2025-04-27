import torch
import torch.nn as nn

import easydict

from . import MODEL


class FireConv(nn.Module):
    def __init__(self, in_channels, sq1x1, ex1x1, ex3x3, freeze=False, stddev=0.001):
        super().__init__()
        self.sq1x1 = nn.Conv2d(in_channels, sq1x1, kernel_size=1, stride=1, padding=0)
        self.ex1x1 = nn.Conv2d(sq1x1,       ex1x1, kernel_size=1, stride=1, padding=0)
        self.ex3x3 = nn.Conv2d(sq1x1,       ex3x3, kernel_size=3, stride=1, padding=1) # keep output the original shape
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        # 初始化权重
        nn.init.normal_(self.sq1x1.weight, std=stddev)
        nn.init.normal_(self.ex1x1.weight, std=stddev)
        nn.init.normal_(self.ex3x3.weight, std=stddev)

    def forward(self, x):
        sq1x1 = self.sq1x1(x)
        ex1x1 = self.ex1x1(sq1x1)
        ex3x3 = self.ex3x3(sq1x1)
        return torch.cat([ex1x1, ex3x3], dim=1)

class FireDeconv(nn.Module):
    def __init__(self, in_channels, sq1x1, ex1x1, ex3x3, factors=[2, 2], freeze=False, stddev=0.001):
        super().__init__()
        self.sq1x1 = nn.Conv2d(in_channels, sq1x1, kernel_size=1, stride=1, padding=0)
        ksize_h = factors[0]
        ksize_w = factors[1]
        self.deconv = nn.ConvTranspose2d(sq1x1, sq1x1, kernel_size=(ksize_h, ksize_w), stride=factors, padding=0)
        self.ex1x1 = nn.Conv2d(sq1x1, ex1x1, kernel_size=1, stride=1, padding=0)
        self.ex3x3 = nn.Conv2d(sq1x1, ex3x3, kernel_size=3, stride=1, padding=1)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        # 初始化权重
        nn.init.normal_(self.sq1x1.weight, std=stddev)
        # deconv初始化为双线性插值
        self.deconv.weight.data = self.bilinear_init(self.deconv.weight.data)
        nn.init.normal_(self.ex1x1.weight, std=stddev)
        nn.init.normal_(self.ex3x3.weight, std=stddev)

    def forward(self, x):
        sq1x1 = self.sq1x1(x)
        deconv = self.deconv(sq1x1)
        ex1x1 = self.ex1x1(deconv)
        ex3x3 = self.ex3x3(deconv)
        return torch.cat([ex1x1, ex3x3], dim=1)

    def bilinear_init(self, weight):
        import math
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        n = fan_in
        for i in range(weight.size(2)):
            for j in range(weight.size(3)):
                weight[0, 0, i, j] = (1 - math.fabs(i - (weight.size(2)-1)/2.0)/(weight.size(2)/2)) * (1 - math.fabs(j - (weight.size(3)-1)/2.0)/(weight.size(3)/2))
        return weight
    

@MODEL.register
class SqueezeSeg(nn.Module):
    def __init__(self, *args, **kwds):
        super().__init__()
        kwds = easydict.EasyDict(kwds)
        self.args = kwds

        in_channels = kwds.in_channels
        out_channels = kwds.num_classes

        self.num_classes = out_channels

        self.conv1       = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_skip  = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.pool1       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2       = FireConv( 64, 16, 64, 64)
        self.conv3       = FireConv(128, 16, 64, 64)
        self.pool3       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4       = FireConv(128, 32, 128, 128)
        self.conv5       = FireConv(256, 32, 128, 128)
        self.pool5       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv6       = FireConv(256, 48, 192, 192)
        self.conv7       = FireConv(384, 48, 192, 192)
        self.conv8       = FireConv(384, 64, 256, 256)
        self.conv9       = FireConv(512, 64, 256, 256)
        self.deconv10    = FireDeconv(512, 64, 128, 128, factors=[2, 2])
        self.deconv11    = FireDeconv(256, 32,  64,  64, factors=[2, 2])
        self.deconv12    = FireDeconv(128, 16,  32,  32, factors=[2, 2])
        self.deconv13    = FireDeconv( 64, 16,  32,  32, factors=[2, 2])
        self.drop13      = nn.Dropout2d(p=self.args.dropout)
        self.conv14_prob = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        # where is CRF refine RNN layer ?

    def forward(self, data: torch.Tensor):   # [N,   5, 64, 512]
        # U-Net(FCN)
        conv1_out   = self.conv1(data)              # [N,  64, 32, 256]
        skip1_out   = self.conv1_skip(data)         # [N,  64, 64, 512]
        pool1_out   = self.pool1(conv1_out)         # [N,  64, 16, 128]
        conv2_out   = self.conv2(pool1_out)         # [N, 128, 16, 128]
        conv3_out   = self.conv3(conv2_out)         # [N, 256, 16, 128]
        pool3_out   = self.pool3(conv3_out)         # [N, 256,  8,  64]
        conv4_out   = self.conv4(pool3_out)         # [N, 256,  8,  64]
        conv5_out   = self.conv5(conv4_out)         # [N, 256,  8,  64]
        pool5_out   = self.pool5(conv5_out)         # [N, 256,  4,  32]
        conv6_out   = self.conv6(pool5_out)         # [N, 384,  4,  32]
        conv7_out   = self.conv7(conv6_out)         # [N, 384,  4,  32]
        conv8_out   = self.conv8(conv7_out)         # [N, 512,  4,  32]
        conv9_out   = self.conv9(conv8_out)         # [N, 512,  4,  32]
        decv10_out  = self.deconv10(conv9_out)      # [N, 256,  8,  64]
        fuse_10_5   = decv10_out + conv5_out        # [N, 256,  8,  64]
        decv11_out  = self.deconv11(fuse_10_5)      # [N, 128, 16, 128]
        fuse_11_3   = decv11_out + conv3_out        # [N, ]
        decv12_out  = self.deconv12(fuse_11_3)
        fuse_12_1   = decv12_out + conv1_out
        decv13_out  = self.deconv13(fuse_12_1)
        fuse_13_1   = decv13_out + skip1_out
        drop13_out  = self.drop13(fuse_13_1)
        conv14_prob = self.conv14_prob(drop13_out)
        # bilateral_filter_weights = self._bilateral_filter_layer(self.lidar_input[:, :, :, :3])
        # output_prob = self._recurrent_crf_layer(conv14_prob, bilateral_filter_weights)

        return conv14_prob
