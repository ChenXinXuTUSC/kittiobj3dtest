import torch
import torch.nn as nn

class FireConv(nn.Module):
    def __init__(self, in_channels, sq1x1, ex1x1, ex3x3, freeze=False, stddev=0.001):
        super(FireConv, self).__init__()
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
    def __init__(self, in_channels, sq1x1, ex1x1, ex3x3, factors=[1,2], freeze=False, stddev=0.001):
        super(FireDeconv, self).__init__()
        self.sq1x1 = nn.Conv2d(in_channels, sq1x1, kernel_size=1, stride=1, padding=0)
        ksize_h = factors[0] * 2 - factors[0] % 2
        ksize_w = factors[1] * 2 - factors[1] % 2
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
    

class SqueezeSeg(nn.Module):
    def __init__(self, conf):
        super(SqueezeSeg, self).__init__()

        in_channels = conf.in_channels  # 假设定义了输入通道数

        self.conv1       = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_skip  = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0)
        self.pool1       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire2       = FireConv( 64, 16, 64, 64)
        self.fire3       = FireConv(128, 16, 64, 64)
        self.pool3       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire4       = FireConv(256, 32, 128, 128)
        self.fire5       = FireConv(256, 32, 128, 128)
        self.pool5       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire6       = FireConv(512, 48, 192, 192)
        self.fire7       = FireConv(384, 48, 192, 192)
        self.fire8       = FireConv(384, 64, 256, 256)
        self.fire9       = FireConv(512, 64, 256, 256)
        self.fire10      = FireDeconv(512, 64, 128, 128, factors=[1, 2])
        self.fire11      = FireDeconv(256, 32,  64,  64, factors=[1, 2])
        self.fire12      = FireDeconv(128, 16,  32,  32, factors=[1, 2])
        self.fire13      = FireDeconv( 64, 16,  32,  32, factors=[1, 2])
        self.drop13      = nn.Dropout2d(p=conf.dropout)
        self.conv14_prob = nn.Conv2d(64, conf.num_cls, kernel_size=3, stride=1, padding=1)
        # where is CRF refine RNN layer ?

    def forward(self, x):
        conv1       = self.conv1(x)
        conv1_skip  = self.conv1_skip(x)
        pool1       = self.pool1(conv1)
        fire2       = self.fire2(pool1)
        fire3       = self.fire3(fire2)
        pool3       = self.pool3(fire3)
        fire4       = self.fire4(pool3)
        fire5       = self.fire5(fire4)
        pool5       = self.pool5(fire5)
        fire6       = self.fire6(pool5)
        fire7       = self.fire7(fire6)
        fire8       = self.fire8(fire7)
        fire9       = self.fire9(fire8)
        fire10      = self.fire10(fire9)
        fire10_fuse = fire10 + fire5
        fire11      = self.fire11(fire10_fuse)
        fire11_fuse = fire11 + fire3
        fire12      = self.fire12(fire11_fuse)
        fire12_fuse = fire12 + conv1
        fire13      = self.fire13(fire12_fuse)
        fire13_fuse = fire13 + conv1_skip
        drop13      = self.drop13(fire13_fuse)
        conv14_prob = self.conv14_prob(drop13)
        # bilateral_filter_weights = self._bilateral_filter_layer(self.lidar_input[:, :, :, :3])
        # output_prob = self._recurrent_crf_layer(conv14_prob, bilateral_filter_weights)
        return conv14_prob
