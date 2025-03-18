import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class RepConv(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier=2, use_bn=False, deploy=False):
        super(RepConv, self).__init__()
        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1,
                                     padding=1, dilation=1, groups=self.groups, bias=True)
        else:
            if self.use_bn:
                self.bn = nn.BatchNorm2d(num_features=self.out_planes)

            if self.type == 'conv3x3':
                conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, stride=1, padding=1)
                self.k0 = conv0.weight
                self.b0 = conv0.bias
            elif self.type == 'conv1x1-conv3x3':
                self.mid_planes = int(out_planes * depth_multiplier)
                conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
                self.k0 = conv0.weight
                self.b0 = conv0.bias

                conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, padding=1)
                self.k1 = conv1.weight
                self.b1 = conv1.bias

            elif self.type == 'conv1x1-conv1x3':
                self.mid_planes = int(out_planes * depth_multiplier)
                conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
                self.k0 = conv0.weight
                self.b0 = conv0.bias

                conv1x3_padding = [0, 1]
                conv1 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=(1, 3), padding=conv1x3_padding, bias=True, padding_mode='zeros')
                self.k1 = conv1.weight
                self.b1 = conv1.bias

            elif self.type == 'conv1x1-conv3x1':
                self.mid_planes = int(out_planes * depth_multiplier)
                conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
                self.k0 = conv0.weight
                self.b0 = conv0.bias

                conv3x1_padding = [1, 0]
                conv1 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=(3, 1), padding=conv3x1_padding,
                                         bias=True, padding_mode='zeros')
                self.k1 = conv1.weight
                self.b1 = conv1.bias

            elif self.type == 'conv1x1-laplacian':
                conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
                self.k0 = conv0.weight
                self.b0 = conv0.bias

                # init scale & bias
                scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
                self.scale = nn.Parameter(torch.FloatTensor(scale))
                bias = torch.randn(self.out_planes) * 1e-3
                bias = torch.reshape(bias, (self.out_planes,))
                self.bias = nn.Parameter(torch.FloatTensor(bias))
                # init mask
                self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
                self.laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                # self.laplacian_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
                for i in range(self.out_planes):
                    self.mask[i, 0, :, :] = self.laplacian_kernel
                self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            elif self.type == 'conv1x1-scharr':
                conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
                self.k0 = conv0.weight
                self.b0 = conv0.bias

                # init scale & bias
                scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
                self.scale = nn.Parameter(torch.FloatTensor(scale))
                # bias = 0.0
                # bias = [bias for c in range(self.out_planes)]
                # bias = torch.FloatTensor(bias)
                bias = torch.randn(self.out_planes) * 1e-3
                bias = torch.reshape(bias, (self.out_planes,))
                self.bias = nn.Parameter(torch.FloatTensor(bias))
                # init mask
                self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
                assert self.out_planes - self.out_planes / 8 * 8 == 0, "if use scharr kernel, the out_planes must be devided by 8"
                self.repeat_kernel_num = self.out_planes // 8
                self.scharr_kernel = torch.tensor([[[3, 10, 3], [0, 0, 0], [-3, -10, -3]], [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                                                   [[3, 0, -3], [10, 0, -10], [3, 0, -3]], [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                                                   [[0, 3, 10], [-3, 0, 3], [-10, -3, 0]], [[-10, -3, 0], [-3, 0, 3], [0, 3, 10]],
                                                   [[10, 3, 0], [3, 0, -3], [0, -3, -10]], [[0, -3, -10], [3, 0, -3], [10, 3, 0]]])
                # self.scharr_kernel = torch.tensor([[[0, 3, 10], [-3, 0, 3], [-10, -3, 0]], [[-10, -3, 0], [-3, 0, 3], [0, 3, 10]],
                #                                    [[10, 3, 0], [3, 0, -3], [0, -3, -10]], [[0, -3, -10], [3, 0, -3], [10, 3, 0]]])
                self.repeat_kernel_num = self.out_planes // len(self.scharr_kernel)
                for i in range(len(self.scharr_kernel)):
                    self.mask[self.repeat_kernel_num * i:self.repeat_kernel_num * (i+1), 0, :, :] = self.scharr_kernel[i]

                self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            else:
                raise ValueError('the type of repconv is not supported!')

    def forward(self, x):
        if self.deploy:
            return self.repconv(x)
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-1x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
            if self.use_bn:
                y1 = self.bn(y1)

        elif self.type =='conv1x1-conv1x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            pad = (1, 1, 0, 0)
            # explicitly padding with bias
            y0 = F.pad(y0, pad, 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
            if self.use_bn:
                y1 = self.bn(y1)
        elif self.type =='conv1x1-conv3x1':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            pad = (0, 0, 1, 1)
            # explicitly padding with bias
            y0 = F.pad(y0, pad, 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            # conv-3x1
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
            if self.use_bn:
                y1 = self.bn(y1)
        elif self.type == 'conv1x1-laplacian':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
            if self.use_bn:
                y1 = self.bn(y1)
        elif self.type == 'conv1x1-scharr':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
            if self.use_bn:
                y1 = self.bn(y1)
        else:
            y0 = F.pad(x, (1, 1, 1, 1), 'constant', 0)
            y1 = F.conv2d(input=y0, weight=self.k0, bias=self.b0, stride=1)
            if self.use_bn:
                y1 = self.bn(y1)
        return y1

    def get_equivalent_kernel_bias(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1

        elif self.type == 'conv3x3':
            RK, RB = self.k0, self.b0

        elif self.type == 'conv1x1-laplacian':
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1

        elif self.type == 'conv1x1-scharr':
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1

        else:
            square_k = torch.zeros((self.out_planes, self.mid_planes, 3, 3), device=device)
            asym_h = self.k1.size(2)
            asym_w = self.k1.size(3)
            square_h = square_k.size(2)
            square_w = square_k.size(3)
            square_k[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h, square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] = self.k1

            RK = F.conv2d(input=square_k, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=square_k).view(-1, ) + self.b1
        if self.use_bn:
            RK, RB = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def fuse_bn_tensor(self, weight, bias, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True


class activation_bn(nn.ReLU):
    def __init__(self, dim, act_num=3, use_bn=False, deploy=False):
        super(activation_bn, self).__init__()
        self.use_bn = use_bn
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        if self.use_bn:
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        torch.nn.init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation_bn, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            if self.use_bn:
                return self.bn(torch.nn.functional.conv2d(
                    super(activation_bn, self).forward(x),
                    self.weight, padding=self.act_num, groups=self.dim))
            else:
                return torch.nn.functional.conv2d(
                    super(activation_bn, self).forward(x),
                    self.weight, padding=self.act_num, groups=self.dim)

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        if self.use_bn:
            kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        else:
            kernel, bias = self.weight, self.bias
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        #self.bias.data = bias
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class block_activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(block_activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.dim = dim
        self.act_num = act_num
        torch.nn.init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(block_activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return torch.nn.functional.conv2d(
                super(block_activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim)

    def switch_to_deploy(self):
        kernel, bias = self.weight, self.bias
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        #self.bias.data = bias
        self.deploy = True


class MutilBranchConv(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=2, num_conv=4, with_idt=False, use_bn= False, deploy=False):
        super(MutilBranchConv, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        # self.num_conv_left = num_conv // 2
        # self.num_conv_right = num_conv - num_conv // 2
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.use_bn = use_bn
        self.deploy = deploy
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        if self.deploy:
            self.repconv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1,
                                     padding=1, dilation=1, groups=1, bias=True)
        else:
            conv1x1_3x3 = list()
            conv1x1_1x3 = list()
            conv1x1_3x1 = list()


            for _ in range(self.num_conv):
                conv1x1_3x3.append(RepConv('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier, self.use_bn))
                conv1x1_1x3.append(RepConv('conv1x1-conv1x3', self.inp_planes, self.out_planes, self.depth_multiplier, self.use_bn))
                conv1x1_3x1.append(RepConv('conv1x1-conv3x1', self.inp_planes, self.out_planes, self.depth_multiplier, self.use_bn))

            self.conv1x1_3x3 = nn.ModuleList(conv1x1_3x3)
            self.conv1x1_1x3 = nn.ModuleList(conv1x1_1x3)
            self.conv1x1_3x1 = nn.ModuleList(conv1x1_3x1)
            self.conv1x1_scharr = RepConv('conv1x1-scharr', self.inp_planes, self.out_planes, -1, self.use_bn)

    def forward(self, x):
        if self.deploy:
            y = self.repconv(x)
            return y
        else:
            conv1x1_1x3_out = 0
            conv1x1_3x3_out = 0
            conv1x1_3x1_out = 0
            conv_scharr_out = 0
            for i in range(self.num_conv):
                conv1x1_3x3_out += self.conv1x1_3x3[i](x)
                conv1x1_1x3_out += self.conv1x1_1x3[i](x)
                conv1x1_3x1_out += self.conv1x1_3x1[i](x)

            conv_scharr_out += self.conv1x1_scharr(x)
            y =  conv1x1_3x3_out + conv1x1_1x3_out + conv1x1_3x1_out + conv_scharr_out
            if self.with_idt:
                y += x
        return y

    def get_equivalent_kernel_bias(self):
        K0, B0, K1, B1, K2, B2, K3, B3 = 0, 0, 0, 0, 0, 0, 0, 0
        # for i in range(self.num_conv):
        #     tempk0, tempb0 = self.conv3x3[i].get_equivalent_kernel_bias()
        #     K0 += tempk0
        #     B0 += tempb0
        for i in range(self.num_conv):
            tempk1, tempb1 = self.conv1x1_3x3[i].get_equivalent_kernel_bias()
            K1 += tempk1
            B1 += tempb1
        for i in range(self.num_conv):
            tempk2, tempb2 = self.conv1x1_1x3[i].get_equivalent_kernel_bias()
            K2 += tempk2
            B2 += tempb2
        for i in range(self.num_conv):
            tempk3, tempb3 = self.conv1x1_3x1[i].get_equivalent_kernel_bias()
            K3 += tempk3
            B3 += tempb3
        RK, RB = (K0 + K1 + K2 + K3), (B0 + B1 + B2 + B3)

        # R_l, B_l = self.conv1x1_lpl.get_equivalent_kernel_bias()
        # RK += R_l
        # RB += B_l

        R_s, B_s = self.conv1x1_scharr.get_equivalent_kernel_bias()
        RK += R_s
        RB += B_s

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        #self.__delattr__('conv3x3')
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1_1x3')
        self.__delattr__('conv1x1_3x1')
        self.__delattr__('conv1x1_scharr')
        self.deploy = True

class MutilBranchGroupConv(nn.Module):
    def __init__(self, inp_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, num_conv=4, with_idt=False, use_bn=False, deploy=False):
        super(MutilBranchGroupConv, self).__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_conv = 2 * num_conv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(in_channels=self.inp_planes,
                                      out_channels=self.out_planes,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=True)
        else:
            rbr_conv = list()
            for _ in range(self.num_conv):
                rbr_conv.append(self._conv_bn(kernel_size=self.kernel_size, padding=self.padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)
            if self.kernel_size != 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x):
        if self.deploy:
            x_clone = x.clone()
            out = self.repconv(x)
            if self.inp_planes == self.out_planes and self.with_idt:
                out = out + x_clone
            return out
        else:
            x_clone = x.clone()
            rbr_scale_out = 0
            if self.kernel_size != 1:
                rbr_scale_out += self.rbr_scale(x)
            rbr_conv_out = 0
            for i in range(self.num_conv):
                rbr_conv_out += self.rbr_conv[i](x)
            out = rbr_conv_out + rbr_scale_out
            if self.inp_planes == self.out_planes and self.with_idt:
                out = out + x_clone
            return out

    def get_equivalent_kernel_bias(self):
        kernel_conv, bias_conv = 0, 0
        for i in range(self.num_conv):
            _kernel, _bias = self.rbr_conv[i].conv.weight, self.rbr_conv[i].conv.bias
            if self.use_bn:
                _kernel, _bias = self._fuse_bn_tensor(_kernel, _bias, self.rbr_conv[i].bn)
            kernel_conv += _kernel
            bias_conv += _bias
        if self.kernel_size != 1:
            kernel_scale, bias_scale = self.rbr_scale.conv.weight, self.rbr_scale.conv.bias
            if self.use_bn:
                kernel_scale, bias_scale = self._fuse_bn_tensor(kernel_scale, bias_scale, self.rbr_scale.bn)
            pad = self.kernel_size // 2
            kernel_scale = nn.functional.pad(kernel_scale, [pad, pad, pad, pad])
            kernel_final = kernel_conv + kernel_scale
            bias_final = bias_conv + bias_scale
        else:
            kernel_final = kernel_conv
            bias_final = bias_conv
        # if self.inp_planes == self.out_planes and self.with_idt:
        #     kernel_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3)
        #     for i in range(self.out_planes):
        #         kernel_idt[i, i, 1, 1] = 1.0
        #     bias_idt = 0.0
        #     kernel_final += kernel_idt
        #     bias_final += bias_idt
        return kernel_final, bias_final

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=self.kernel_size,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.__delattr__('rbr_conv')
        if self.kernel_size != 1:
            self.__delattr__('rbr_scale')
        self.deploy = True

    def _conv_bn(self, kernel_size, padding):
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.inp_planes,
                                              out_channels=self.out_planes,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=True))
        if self.use_bn:
            mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_planes))
        return mod_list

    def _fuse_bn_tensor(self, weight, bias, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

class MutilBranchGroupConvTest(nn.Module):
    def __init__(self, inp_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, num_conv=4, with_idt=False, use_bn=False, deploy=False):
        super(MutilBranchGroupConvTest, self).__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_conv = num_conv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(in_channels=self.inp_planes,
                                      out_channels=self.out_planes,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=True)
        else:
            rbr_conv = list()
            for _ in range(self.num_conv):
                rbr_conv.append(self._conv_bn(kernel_size=self.kernel_size, padding=self.padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)
            if self.kernel_size != 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x):
        if self.deploy:
            x_clone = x.clone()
            out = self.repconv(x)
            if self.inp_planes == self.out_planes and self.with_idt:
                out = out + x_clone
            return out
        else:
            x_clone = x.clone()
            rbr_scale_out = 0
            if self.kernel_size != 1:
                rbr_scale_out += self.rbr_scale(x)
            rbr_conv_out = 0
            for i in range(self.num_conv):
                rbr_conv_out += self.rbr_conv[i](x)
            out = rbr_conv_out + rbr_scale_out
            if self.inp_planes == self.out_planes and self.with_idt:
                out = out + x_clone
            return out

    def get_equivalent_kernel_bias(self):
        kernel_conv, bias_conv = 0, 0
        for i in range(self.num_conv):
            _kernel, _bias = self.rbr_conv[i].conv.weight, self.rbr_conv[i].conv.bias
            if self.use_bn:
                _kernel, _bias = self._fuse_bn_tensor(_kernel, _bias, self.rbr_conv[i].bn)
            kernel_conv += _kernel
            bias_conv += _bias
        if self.kernel_size != 1:
            kernel_scale, bias_scale = self.rbr_scale.conv.weight, self.rbr_scale.conv.bias
            if self.use_bn:
                kernel_scale, bias_scale = self._fuse_bn_tensor(kernel_scale, bias_scale, self.rbr_scale.bn)
            pad = self.kernel_size // 2
            kernel_scale = nn.functional.pad(kernel_scale, [pad, pad, pad, pad])
            kernel_final = kernel_conv + kernel_scale
            bias_final = bias_conv + bias_scale
        else:
            kernel_final = kernel_conv
            bias_final = bias_conv
        # if self.inp_planes == self.out_planes and self.with_idt:
        #     kernel_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3)
        #     for i in range(self.out_planes):
        #         kernel_idt[i, i, 1, 1] = 1.0
        #     bias_idt = 0.0
        #     kernel_final += kernel_idt
        #     bias_final += bias_idt
        return kernel_final, bias_final

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=self.kernel_size,
                                 stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.__delattr__('rbr_conv')
        if self.kernel_size != 1:
            self.__delattr__('rbr_scale')
        self.deploy = True

    def _conv_bn(self, kernel_size, padding):
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.inp_planes,
                                              out_channels=self.out_planes,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=True))
        if self.use_bn:
            mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_planes))
        return mod_list

    def _fuse_bn_tensor(self, weight, bias, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std




if __name__ == '__main__':
    # # test rep-conv
    x = torch.randn(1, 8, 16, 16)
    conv = RepConv('conv1x1-scharr', 8, 16, 4)
    conv.eval()
    y0 = conv(x)
    conv.switch_to_deploy()
    y1 = conv(x)
    #RK, RB = conv.rep_params()
    #y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print(y0 - y1)

    # test mbc
    x = torch.randn(1, 3, 5, 5) * 2000
    mbc = MutilBranchConv(3, 6, 2, 4, with_idt=True, use_bn=False)
    mbc.eval()
    start = time.time()
    y0 = mbc(x)
    print(time.time()-start)
    mbc.switch_to_deploy()
    start1 = time.time()
    y1 = mbc(x)
    print(time.time()-start1)
    print(y0 - y1)
    #
    # # test mbc
    # x = torch.randn(1, 6, 5, 5) * 2000
    # mbc = MutilBranchGroupConv(6, 6, kernel_size=1, stride=1, padding=0, dilation=1, groups=6, num_conv=8, with_idt=True, use_bn=True, deploy=False)
    # mbc.eval()
    # start = time.time()
    # y0 = mbc(x)
    # print(time.time()-start)
    # mbc.switch_to_deploy()
    # start1 = time.time()
    # y1 = mbc(x)
    # print(time.time()-start1)
    # print(y0 - y1)