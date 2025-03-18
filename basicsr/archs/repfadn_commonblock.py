import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .repfadn_layers import MutilBranchConv, block_activation, MutilBranchGroupConv, MutilBranchGroupConvTest


class Activate_Function(nn.Module):
    def __init__(self, input_channels, act_type='gelu', block_act_num=3, deploy=False):
        super(Activate_Function, self).__init__()
        self.input_channels = input_channels
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.deploy = deploy
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.input_channels)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'block_act':
            self.act = block_activation(self.input_channels, self.block_act_num, self.deploy)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'linear':
            self.act = nn.Identity()
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.act_type != 'linear':
            x = self.act(x)
        return x

    def switch_to_deploy(self):
        if self.act_type == 'block_act':
            self.act.switch_to_deploy()
        self.deploy = True



class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class PSUpsample(nn.Sequential):
    def __init__(self, up_scale, feature_num):
        m=[]
        if (up_scale & (up_scale - 1)) == 0:
            for _ in range(int(math.log(up_scale, 2))):
                m.append(nn.Conv2d(feature_num, 4 * feature_num, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif up_scale == 3:
            m.append(nn.Conv2d(feature_num, 9 * feature_num, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'up_scale:{up_scale} is not support! up_sacle is only support 2^n and 3!')
        super(PSUpsample, self).__init__(*m)

class PSMBUpsample(nn.Module):
    def __init__(self, up_scale, feature_num, depth_multiplier=2, num_conv=4, with_idt=False, use_bn= True, deploy=False):
        super(PSMBUpsample, self).__init__()
        self.up_scale = up_scale
        self.feature_num = feature_num
        if self.up_scale==2:
            self.upsample1 = MutilBranchConv(feature_num, 4 * feature_num, depth_multiplier, num_conv, with_idt, use_bn, deploy)
            self.pixshuffle1 = nn.PixelShuffle(2)
        elif self.up_scale==3:
            self.upsample1 = MutilBranchConv(feature_num, 9 * feature_num, depth_multiplier, num_conv, with_idt, use_bn,
                                             deploy)
            self.pixshuffle1 = nn.PixelShuffle(3)
        elif self.up_scale==4:
            self.upsample1 = MutilBranchConv(feature_num, 4 * feature_num, depth_multiplier, num_conv, with_idt, use_bn, deploy)
            self.pixshuffle1 = nn.PixelShuffle(2)
            self.upsample2 = MutilBranchConv(feature_num, 4 * feature_num, depth_multiplier, num_conv, with_idt, use_bn, deploy)
            self.pixshuffle2 = nn.PixelShuffle(2)
        elif self.up_scale == 8:
            self.upsample1 = MutilBranchConv(feature_num, 4 * feature_num, depth_multiplier, num_conv, with_idt, use_bn, deploy)
            self.pixshuffle1 = nn.PixelShuffle(2)
            self.upsample2 = MutilBranchConv(feature_num, 4 * feature_num, depth_multiplier, num_conv, with_idt, use_bn, deploy)
            self.pixshuffle2 = nn.PixelShuffle(2)
            self.upsample3 = MutilBranchConv(feature_num, 4 * feature_num, depth_multiplier, num_conv, with_idt, use_bn, deploy)
            self.pixshuffle3 = nn.PixelShuffle(2)
        else:
            raise ValueError(f'up_scale:{up_scale} is not support! up_sacle is only support 2^n and 3!')
    def forward(self, x):
        if self.up_scale==2:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
        elif self.up_scale==3:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
        elif self.up_scale==4:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
            x = self.upsample2(x)
            x = self.pixshuffle2(x)
        else:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
            x = self.upsample2(x)
            x = self.pixshuffle2(x)
            x = self.upsample3(x)
            x = self.pixshuffle3(x)

        return x
    def switch_to_deploy(self):
        if self.up_scale == 2:
            self.upsample1.switch_to_deploy()
        elif self.up_scale == 3:
            self.upsampleb1.switch_to_deploy()
        elif self.up_scale == 4:
            self.upsample1.switch_to_deploy()
            self.upsample2.switch_to_deploy()
        else:
            self.upsample1.switch_to_deploy()
            self.upsample2.switch_to_deploy()
            self.upsample3.switch_to_deploy()

class PSPC3C1Upsample(nn.Module):
    def __init__(self, up_scale, feature_num, depth_multiplier=2, num_conv=4, with_idt=False, use_bn= True, deploy=False):
        super(PSPC3C1Upsample, self).__init__()
        self.up_scale = up_scale
        self.feature_num = feature_num
        if self.up_scale==2:
            self.upsample1 = Parallel_MBConv3x3_MBConv1x1_Bn_Act(feature_num, 4 * feature_num, depth_multiplier,
                                                             num_conv, 'linear', 3, True, with_idt, use_bn, deploy)
            self.pixshuffle1 = nn.PixelShuffle(2)
        elif self.up_scale==3:
            self.upsample1 = Parallel_MBConv3x3_MBConv1x1_Bn_Act(feature_num, 9 * feature_num, depth_multiplier,
                                                             num_conv, 'linear', 3, True, with_idt, use_bn, deploy)
            self.pixshuffle1 = nn.PixelShuffle(3)
        elif self.up_scale==4:
            self.upsample1 = Parallel_MBConv3x3_MBConv1x1_Bn_Act(feature_num, 4 * feature_num, depth_multiplier,
                                                             num_conv, 'linear', 3, True, with_idt, use_bn, deploy)
            self.pixshuffle1 = nn.PixelShuffle(2)
            self.upsample2 = Parallel_MBConv3x3_MBConv1x1_Bn_Act(feature_num, 4 * feature_num, depth_multiplier,
                                                             num_conv, 'linear', 3, True, with_idt, use_bn, deploy)
            self.pixshuffle2 = nn.PixelShuffle(2)
        elif self.up_scale == 8:
            self.upsample1 = Parallel_MBConv3x3_MBConv1x1_Bn_Act(feature_num, 4 * feature_num, depth_multiplier,
                                                             num_conv, 'linear', 3, True, with_idt, use_bn, deploy)
            self.pixshuffle1 = nn.PixelShuffle(2)
            self.upsample2 = Parallel_MBConv3x3_MBConv1x1_Bn_Act(feature_num, 4 * feature_num, depth_multiplier,
                                                             num_conv, 'linear', 3, True, with_idt, use_bn, deploy)
            self.pixshuffle2 = nn.PixelShuffle(2)
            self.upsample3 = Parallel_MBConv3x3_MBConv1x1_Bn_Act(feature_num, 4 * feature_num, depth_multiplier,
                                                             num_conv, 'linear', 3, True, with_idt, use_bn, deploy)
            self.pixshuffle3 = nn.PixelShuffle(2)
        else:
            raise ValueError(f'up_scale:{up_scale} is not support! up_sacle is only support 2^n and 3!')
    def forward(self, x):
        if self.up_scale==2:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
        elif self.up_scale==3:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
        elif self.up_scale==4:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
            x = self.upsample2(x)
            x = self.pixshuffle2(x)
        else:
            x = self.upsample1(x)
            x = self.pixshuffle1(x)
            x = self.upsample2(x)
            x = self.pixshuffle2(x)
            x = self.upsample3(x)
            x = self.pixshuffle3(x)
        return x

    def switch_to_deploy(self):
        if self.up_scale == 2:
            self.upsample1.switch_to_deploy()
        elif self.up_scale == 3:
            self.upsampleb1.switch_to_deploy()
        elif self.up_scale == 4:
            self.upsample1.switch_to_deploy()
            self.upsample2.switch_to_deploy()
        else:
            self.upsample1.switch_to_deploy()
            self.upsample2.switch_to_deploy()
            self.upsample3.switch_to_deploy()

class CONVUpsamplex2(nn.Module):
    def __init__(self, channels, use_conv):
        super().init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bicubic")
        if self.use_conv:
            x = self.conv(x)
        return x

class MBConv1x1Block(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='gelu', block_act_num=3, use_mbconv=True, num_conv=2, with_idt=False, use_bn=False, deploy=False):
        super(MBConv1x1Block, self).__init__()
        self.input_channels = inp_planes
        self.output_channels = out_planes
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.num_conv = num_conv
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv = MutilBranchGroupConv(self.input_channels, self.output_channels, kernel_size=1, padding=0,
                                                 num_conv=self.num_conv, with_idt=self.with_idt, use_bn=self.use_bn, deploy=self.deploy)
            else:
                self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.input_channels, eps=1e-6)

        self.act = Activate_Function(self.output_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
            return y
        else:
            y = self.conv(inputs)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
            return y

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK, RB = self.conv.get_equivalent_kernel_bias()
        else:
            RK, RB = self.conv.weight.data, self.conv.bias.data
        if self.use_bn:
            RK, RB  = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        self.act.switch_to_deploy()
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias

        self.__delattr__('conv')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True


class MBConv1x1BlockTest(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='gelu', block_act_num=3, use_mbconv=True, num_conv=2, with_idt=False, use_bn=False, deploy=False):
        super(MBConv1x1BlockTest, self).__init__()
        self.input_channels = inp_planes
        self.output_channels = out_planes
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.num_conv = num_conv
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv = MutilBranchGroupConvTest(self.input_channels, self.output_channels, kernel_size=1, padding=0,
                                                 num_conv=self.num_conv, with_idt=self.with_idt, use_bn=self.use_bn, deploy=self.deploy)
            else:
                self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.input_channels, eps=1e-6)

        self.act = Activate_Function(self.output_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
            return y
        else:
            y = self.conv(inputs)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
            return y

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK, RB = self.conv.get_equivalent_kernel_bias()
        else:
            RK, RB = self.conv.weight.data, self.conv.bias.data
        if self.use_bn:
            RK, RB  = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        self.act.switch_to_deploy()
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias

        self.__delattr__('conv')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True



class FFNConv1x1Block(nn.Module):
    def __init__(self, inp_planes, FFN_Expand=2, act_type='gelu', block_act_num=3, train_act=True, use_bn=False, deploy=False):
        super(FFNConv1x1Block, self).__init__()
        self.FFN_Expand = FFN_Expand
        self.input_channels = inp_planes
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.ffn_channels = int(self.input_channels * self.FFN_Expand)
        self.act_learn = 1
        self.train_act = train_act
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        else:
            if self.train_act:
                self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.ffn_channels, kernel_size=1, stride=1,
                                      bias=True)
                self.conv2 = nn.Conv2d(in_channels=self.ffn_channels, out_channels=self.input_channels, kernel_size=1, stride=1,
                                    bias=True)
            else:
                self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.input_channels, eps=1e-6)

        self.act = Activate_Function(self.input_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
            return y
        else:
            if self.train_act:
                y = self.conv1(inputs)
                y = torch.nn.functional.leaky_relu(y, self.act_learn)
                y = self.conv2(y)
            else:
                y = self.conv(inputs)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
            return y

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

    def get_equivalent_kernel_bias(self):
        if self.train_act:
            RK1 = self.conv1.weight.data
            RB1 = self.conv1.bias.data
            RK2 = self.conv2.weight.data
            RB2 = self.conv2.bias.data
            RK = torch.matmul(RK2.transpose(1, 3), RK1.squeeze(3).squeeze(2)).transpose(1, 3)
            RB = RB2 + (RB1.view(1, -1, 1, 1)*RK2).sum(3).sum(2).sum(1)
        else:
            RK, RB = self.conv.weight.data, self.conv.bias.data
        if self.use_bn:
            RK, RB  = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        self.act.switch_to_deploy()
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        if self.train_act:
            self.__delattr__('conv1')
            self.__delattr__('conv2')
        else:
            self.__delattr__('conv')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class Conv1x1Block(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='gelu', block_act_num=3, train_act=True, use_bn=False, deploy=False):
        super(Conv1x1Block, self).__init__()
        self.input_channels = inp_planes
        self.output_channels = out_planes
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.act_learn = 1
        self.train_act = train_act
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        else:
            if self.train_act:
                self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=1, stride=1,
                                      bias=True)
                self.conv2 = nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=1, stride=1,
                                    bias=True)
            else:
                self.conv = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.output_channels, eps=1e-6)

        self.act = Activate_Function(self.output_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
            return y
        else:
            if self.train_act:
                y = self.conv1(inputs)
                y = torch.nn.functional.leaky_relu(y, self.act_learn)
                y = self.conv2(y)
            else:
                y = self.conv(inputs)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
            return y

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

    def get_equivalent_kernel_bias(self):
        if self.train_act:
            RK1 = self.conv1.weight.data
            RB1 = self.conv1.bias.data
            RK2 = self.conv2.weight.data
            RB2 = self.conv2.bias.data
            RK = torch.matmul(RK2.transpose(1, 3), RK1.squeeze(3).squeeze(2)).transpose(1, 3)
            RB = RB2 + (RB1.view(1, -1, 1, 1)*RK2).sum(3).sum(2).sum(1)
        else:
            RK, RB = self.conv.weight.data, self.conv.bias.data
        if self.use_bn:
            RK, RB  = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        self.act.switch_to_deploy()
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        if self.train_act:
            self.__delattr__('conv1')
            self.__delattr__('conv2')
        else:
            self.__delattr__('conv')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class Stem(nn.Module):
    def __init__(self, in_chans=3, num_features=32, depth_multiplier=2, num_conv=4, act_type='gelu', block_act_num=3, use_mbconv=True, with_idt=False, use_bn=False, train_act=True, deploy=False):
        super(Stem, self).__init__()
        self.act_learn = 1
        self.in_channels = in_chans
        self.out_channels = num_features
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.train_act = train_act
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv1 = MutilBranchConv(self.in_channels, self.out_channels, self.depth_multiplier, self.num_conv, self.with_idt, False, self.deploy)
            else:
                self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            if self.use_mbconv:
                self.conv2 = MutilBranchGroupConv(self.out_channels, self.out_channels, kernel_size=1, padding=0,
                                                    num_conv=self.num_conv, with_idt=self.with_idt, use_bn=False,
                                                    deploy=self.deploy)
            else:
                self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(self.out_channels, eps=1e-6)
                #self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-6)
        self.act = Activate_Function(self.out_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
        else:
            y = self.conv1(inputs)
            # if self.use_bn:
            #     y = self.bn1(y)
            if self.train_act:
                y = torch.nn.functional.leaky_relu(y, self.act_learn)
            y = self.conv2(y)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
        return y

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK1, RB1 = self.conv1.get_equivalent_kernel_bias()
        else:
            RK1 = self.conv1.weight.data
            RB1 = self.conv1.bias.data
        # if self.use_bn:
        #     RK1, RB1 = self.fuse_bn_tensor(RK1, RB1, self.bn1)
        if self.use_mbconv:
            RK2, RB2 = self.conv2.get_equivalent_kernel_bias()
        else:
            RK2 = self.conv2.weight.data
            RB2 = self.conv2.bias.data
        RK = torch.einsum('oi,icjk->ocjk', RK2.squeeze(3).squeeze(2), RK1)
        RB = RB2 + (RB1.view(1, -1, 1, 1)*RK2).sum(3).sum(2).sum(1)

        if self.use_bn:
            RK, RB = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.act.switch_to_deploy()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class ParallelStem(nn.Module):
    def __init__(self, in_chans=3, num_features=32, depth_multiplier=2, num_conv=4, act_type='gelu', block_act_num=3, use_mbconv=True, with_idt=False, use_bn=False, train_act=True, deploy=False):
        super(ParallelStem, self).__init__()
        self.act_learn = 1
        self.in_channels = in_chans
        self.out_channels = num_features
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.train_act = train_act
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.conv = Parallel_MBConv3x3_MBConv1x1_Bn_Act(self.in_channels, self.out_channels, self.depth_multiplier,
                                                             self.num_conv, 'linear', self.block_act_num, self.use_mbconv,
                                                             self.with_idt, self.use_bn, self.deploy)

        self.act = Activate_Function(self.out_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
        else:
            y = self.conv(inputs)
            y = self.act(y)
        return y

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.conv.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.act.switch_to_deploy()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class Conv_Bn_Act(nn.Module):
    def __init__(self, in_chans, out_chans, depth_multiplier=2, num_conv=4, act_type='gelu', block_act_num=3, use_mbconv=True, with_idt=False, use_bn=False, deploy=False):
        super(Conv_Bn_Act, self).__init__()
        self.in_channels = in_chans
        self.out_channels = out_chans
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv = MutilBranchConv(self.in_channels, self.out_channels, self.depth_multiplier, self.num_conv, self.with_idt, False, self.deploy)
            else:
                self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(self.out_channels, eps=1e-6)
        self.act = Activate_Function(self.out_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
        else:
            y = self.conv(inputs)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
        return y

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK, RB = self.conv.get_equivalent_kernel_bias()
        else:
            RK = self.conv.weight.data
            RB = self.conv.bias.data
        if self.use_bn:
            RK, RB = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.act.switch_to_deploy()
        self.__delattr__('conv')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class MBConv3x3_MBConv1x1_Bn_Act(nn.Module):
    def __init__(self, in_chans, out_chans, depth_multiplier=2, num_conv=4, act_type='gelu', block_act_num=3, use_mbconv=True, with_idt=False, use_bn=False, train_act=True, deploy=False):
        super(MBConv3x3_MBConv1x1_Bn_Act, self).__init__()
        self.act_learn = 1
        self.in_channels = in_chans
        self.out_channels = out_chans
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.train_act = train_act
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv3x3 = MutilBranchConv(self.in_channels, self.out_channels, self.depth_multiplier, self.num_conv, self.with_idt, False, self.deploy)
            else:
                self.conv3x3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            if self.use_mbconv:
                self.conv1x1 = MutilBranchGroupConv(self.out_channels, self.out_channels, kernel_size=1, padding=0, num_conv=self.num_conv, with_idt=self.with_idt, use_bn=False, deploy=self.deploy)
            else:
                self.conv1x1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(self.out_channels, eps=1e-6)
                #self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-6)
        self.act = Activate_Function(self.out_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
        else:
            y = self.conv3x3(inputs)
            # if self.use_bn:
            #     y = self.bn1(y)
            if self.train_act:
                y = torch.nn.functional.leaky_relu(y, self.act_learn)
            y = self.conv1x1(y)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
        return y

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK1, RB1 = self.conv3x3.get_equivalent_kernel_bias()
        else:
            RK1 = self.conv3x3.weight.data
            RB1 = self.conv3x3.bias.data
        if self.use_mbconv:
            RK2, RB2 = self.conv1x1.get_equivalent_kernel_bias()
        else:
            RK2 = self.conv1x1.weight.data
            RB2 = self.conv1x1.bias.data
        # if self.use_bn:
        #     RK1, RB1 = self.fuse_bn_tensor(RK1, RB1, self.bn1)
        # if self.use_bn:
        #     RK2, RB2 = self.fuse_bn_tensor(RK2, RB2, self.bn2)
        RK = torch.einsum('oi,icjk->ocjk', RK2.squeeze(3).squeeze(2), RK1)
        RB = RB2 + (RB1.view(1, -1, 1, 1)*RK2).sum(3).sum(2).sum(1)
        if self.use_bn:
            RK, RB = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.act.switch_to_deploy()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class MBConv3x3_Conv1x1_Bn_Act(nn.Module):
    def __init__(self, in_chans, out_chans, depth_multiplier=2, num_conv=4, act_type='gelu', block_act_num=3, use_mbconv=True, with_idt=False, use_bn=False, train_act=True, deploy=False):
        super(MBConv3x3_Conv1x1_Bn_Act, self).__init__()
        self.act_learn = 1
        self.in_channels = in_chans
        self.out_channels = out_chans
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.train_act = train_act
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv3x3 = MutilBranchConv(self.in_channels, self.out_channels, self.depth_multiplier, self.num_conv, self.with_idt, False, self.deploy)
            else:
                self.conv3x3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            if self.train_act:
                self.conv1x1 = Conv1x1Block(self.out_channels, self.out_channels, self.act_type, self.block_act_num, self.train_act, self.use_bn, self.deploy)
            else:
                self.conv1x1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(self.out_channels, eps=1e-6)
                #self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-6)
        self.act = Activate_Function(self.out_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, inputs):
        if self.deploy:
            y = self.repconv(inputs)
            y = self.act(y)
        else:
            y = self.conv3x3(inputs)
            # if self.use_bn:
            #     y = self.bn1(y)
            if self.train_act:
                y = torch.nn.functional.leaky_relu(y, self.act_learn)
            y = self.conv1x1(y)
            if self.use_bn:
                y = self.bn(y)
            y = self.act(y)
        return y

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK1, RB1 = self.conv3x3.get_equivalent_kernel_bias()
        else:
            RK1 = self.conv3x3.weight.data
            RB1 = self.conv3x3.bias.data
        if self.train_act:
            RK2, RB2 = self.conv1x1.get_equivalent_kernel_bias()
        else:
            RK2 = self.conv1x1.weight.data
            RB2 = self.conv1x1.bias.data
        # if self.use_bn:
        #     RK1, RB1 = self.fuse_bn_tensor(RK1, RB1, self.bn1)
        # if self.use_bn:
        #     RK2, RB2 = self.fuse_bn_tensor(RK2, RB2, self.bn2)
        RK = torch.einsum('oi,icjk->ocjk', RK2.squeeze(3).squeeze(2), RK1)
        RB = RB2 + (RB1.view(1, -1, 1, 1)*RK2).sum(3).sum(2).sum(1)
        if self.use_bn:
            RK, RB = self.fuse_bn_tensor(RK, RB, self.bn)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.act.switch_to_deploy()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if self.use_bn:
            self.__delattr__('bn')
        self.deploy = True

class Res_MBConv3x3_MBConv1x1_Bn_Act(nn.Module):
    def __init__(self, channels, depth_multiplier=2, num_conv=4, act_type='gelu', block_act_num=3, use_mbconv=True, with_idt=False, use_bn=False, deploy=False):
        super(Res_MBConv3x3_MBConv1x1_Bn_Act, self).__init__()
        self.act_learn = 1
        self.channels = channels
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv3x3 = MutilBranchConv(self.channels, self.channels, self.depth_multiplier, self.num_conv,
                                               False, False, self.deploy)
            else:
                self.conv3x3 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=True)
            if self.use_mbconv:
                self.conv1x1 = MutilBranchGroupConv(self.channels, self.channels, kernel_size=1, padding=0,
                                                    num_conv=self.num_conv, with_idt=False, use_bn=False,
                                                    deploy=self.deploy)
            else:
                self.conv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0,
                                         bias=True)
            if self.use_bn:
                self.bn1 = nn.BatchNorm2d(self.channels)
                self.bn2 = nn.BatchNorm2d(self.channels)
        self.act = Activate_Function(self.channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, x):
        if self.deploy:
            x = self.repconv(x)
            x = self.act(x)
            return x
        else:
            conv3x3 = self.conv3x3(x)
            conv1x1 = self.conv1x1(x)
            if self.use_bn:
                conv3x3 = self.bn1(conv3x3)
                conv1x1 = self.bn2(conv1x1)
            return self.act(conv3x3 + conv1x1 + x)

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK1, RB1 = self.conv3x3.get_equivalent_kernel_bias()
        else:
            RK1 = self.conv3x3.weight.data
            RB1 = self.conv3x3.bias.data
        if self.use_mbconv:
            RK2, RB2 = self.conv1x1.get_equivalent_kernel_bias()
        else:
            RK2 = self.conv1x1.weight.data
            RB2 = self.conv1x1.bias.data

        RK2 = torch.nn.functional.pad(RK2, [1, 1, 1, 1])

        if self.use_bn:
            RK1, RB1 = self.fuse_bn_tensor(RK1, RB1, self.bn1)
            RK2, RB2 = self.fuse_bn_tensor(RK2, RB2, self.bn2)
        identity = torch.zeros(self.channels, self.channels, 3, 3, device=RK2.device)
        for i in range(self.channels):
            identity[i, i, 1, 1] = 1.0
        RK = RK1 + RK2 + identity
        RB = RB1 + RB2
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.act.switch_to_deploy()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if self.use_bn:
            self.__delattr__('bn1')
            self.__delattr__('bn2')
        self.deploy = True

#并联结构
class Parallel_MBConv3x3_MBConv1x1_Bn_Act(nn.Module):
    def __init__(self, in_channels, out_channels, depth_multiplier=2, num_conv=4, act_type='gelu', block_act_num=3, use_mbconv=True, with_idt=False, use_bn=False, deploy=False):
        super(Parallel_MBConv3x3_MBConv1x1_Bn_Act, self).__init__()
        self.act_learn = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.use_mbconv = use_mbconv
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            if self.use_mbconv:
                self.conv3x3 = MutilBranchConv(self.in_channels, self.out_channels, self.depth_multiplier, self.num_conv,
                                               False, False, self.deploy)
            else:
                self.conv3x3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            if self.use_mbconv:
                self.conv1x1 = MutilBranchGroupConv(self.in_channels, self.out_channels, kernel_size=1, padding=0,
                                                    num_conv=self.num_conv, with_idt=False, use_bn=False,
                                                    deploy=self.deploy)
            else:
                self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                         bias=True)
            if self.use_bn:
                self.bn1 = nn.BatchNorm2d(self.out_channels)
                self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.act = Activate_Function(self.out_channels, self.act_type, self.block_act_num, self.deploy)

    def forward(self, x):
        if self.deploy:
            x = self.repconv(x)
            x = self.act(x)
            return x
        else:
            conv3x3 = self.conv3x3(x)
            conv1x1 = self.conv1x1(x)
            if self.use_bn:
                conv3x3 = self.bn1(conv3x3)
                conv1x1 = self.bn2(conv1x1)
            return self.act(conv3x3 + conv1x1)

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

    def get_equivalent_kernel_bias(self):
        if self.use_mbconv:
            RK1, RB1 = self.conv3x3.get_equivalent_kernel_bias()
        else:
            RK1 = self.conv3x3.weight.data
            RB1 = self.conv3x3.bias.data
        if self.use_mbconv:
            RK2, RB2 = self.conv1x1.get_equivalent_kernel_bias()
        else:
            RK2 = self.conv1x1.weight.data
            RB2 = self.conv1x1.bias.data

        RK2 = torch.nn.functional.pad(RK2, [1, 1, 1, 1])

        if self.use_bn:
            RK1, RB1 = self.fuse_bn_tensor(RK1, RB1, self.bn1)
            RK2, RB2 = self.fuse_bn_tensor(RK2, RB2, self.bn2)
        RK = RK1 + RK2
        RB = RB1 + RB2
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.act.switch_to_deploy()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if self.use_bn:
            self.__delattr__('bn1')
            self.__delattr__('bn2')
        self.deploy = True

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops

class MBUpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, num_conv, deploy=False):
        super(MBUpsampleOneStep, self).__init__()
        self.scale = scale
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        self.num_conv = num_conv
        self.deploy = deploy
        if self.deploy:
            self.repconv = nn.Conv2d(self.num_feat, self.num_out_ch * (self.scale ** 2), kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv = MutilBranchGroupConv(self.num_feat, self.num_out_ch * (self.scale ** 2), kernel_size=3, padding=1,
                                                    num_conv=self.num_conv, with_idt=False, use_bn=False,
                                                    deploy=self.deploy)
        self.pixelshuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        if self.deploy:
            x = self.repconv(x)
        else:
            x = self.conv(x)
        x = self.pixelshuffle(x)
        return x

    def get_equivalent_kernel_bias(self):
        RK, RB = self.conv.get_equivalent_kernel_bias()
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.repconv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.repconv.weight.data = kernel
        self.repconv.bias.data = bias
        self.__delattr__('conv')
        self.deploy = True
