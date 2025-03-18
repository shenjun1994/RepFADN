import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
    def __init__(self, input_channels, SE_Factor):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=input_channels // SE_Factor, kernel_size=1, stride=1, bias=False)
        self.up = nn.Conv2d(in_channels=input_channels // SE_Factor, out_channels=input_channels, kernel_size=1, stride=1, bias=False)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=(inputs.size(2), inputs.size(3)))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class SimpleSEBlock(nn.Module):
    def __init__(self, inp_planes, SE_Factor=2, train_act=True, deploy=False):
        super(SimpleSEBlock, self).__init__()
        self.input_channels = inp_planes
        self.SE_Factor = SE_Factor
        self.mid_channels = int(self.input_channels // self.SE_Factor)
        self.act_learn = 1
        self.train_act = train_act
        self.deploy = deploy

        if self.deploy:
            self.repconv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=True)
        else:
            self.down = nn.Conv2d(in_channels=self.input_channels, out_channels=self.mid_channels, kernel_size=1, stride=1,
                                  bias=True)
            self.up = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.input_channels, kernel_size=1, stride=1,
                                bias=True)
    def forward(self, inputs):
        if self.deploy:
            y = F.avg_pool2d(inputs, kernel_size=(inputs.size(2), inputs.size(3)))
            if self.train_act:
                y = self.repconv(y)
            else:
                y = self.down(y)
                y = torch.nn.functional.relu(y, inplace=True)
                y = self.up(y)
            y = torch.sigmoid(y)
            y = y.view(-1, self.input_channels, 1, 1)
            return inputs * y
        else:
            y = F.avg_pool2d(inputs, kernel_size=(inputs.size(2), inputs.size(3)))
            y = self.down(y)
            if self.train_act:
                y = torch.nn.functional.leaky_relu(y, self.act_learn)
            else:
                y = torch.nn.functional.relu(y, inplace=True)
            y = self.up(y)
            y = torch.sigmoid(y)
            y = y.view(-1, self.input_channels, 1, 1)
            return inputs * y

    def get_equivalent_kernel_bias(self):
        device = self.up.weight.get_device()
        if device < 0:
            device = None
        RK1 = self.down.weight.data
        RB1 = self.down.bias.data
        RK2 = self.up.weight.data
        RB2 = self.up.bias.data
        RK = torch.matmul(RK2.transpose(1, 3), RK1.squeeze(3).squeeze(2)).transpose(1, 3)
        RB = RB2 + (RB1.view(1, -1, 1, 1)*RK2).sum(3).sum(2).sum(1)
        return RK, RB

    def switch_to_deploy(self):
        if hasattr(self, 'repconv'):
            return
        if self.train_act:
            kernel, bias = self.get_equivalent_kernel_bias()
            self.repconv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
            self.repconv.weight.data = kernel
            self.repconv.bias.data = bias
            self.__delattr__('down')
            self.__delattr__('up')
        self.deploy = True

class ChannelAttention(nn.Module):
    '''CBAM混合注意力的通道'''
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out

class SpatitalAttention(nn.Module):
    '''CBAM混合注意力机制的空间注意力'''
    def __init__(self, kernel_size=7):
        super(SpatitalAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must in [3,7]'
        padding = 3 if kernel_size==7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    '''CBAM混合注意力机制'''
    def __init__(self, in_channels, ratio=4, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatitalAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x

class GlobalAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, window_size=8):
        super(GlobalAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        assert channels % num_heads == 0
        self.norm = nn.GroupNorm(4, channels)
        self.qkv = nn.Linear(channels, channels*3, bias=False)
        self.proj = nn.Linear(channels, channels)
        #self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1, bias=False)
        #self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def window_partition(self, x, windows_size=8):
        B, H, W, C = x.shape
        pad_h = (windows_size - H % windows_size) % windows_size
        pad_w = (windows_size - W % windows_size) % windows_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // windows_size, windows_size, Wp // windows_size, windows_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, windows_size, windows_size, C)
        return windows, (Hp, Wp)

    def window_unpartition(self, windows, window_size, pad_hw, hw):
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp //window_size //window_size)
        x = windows.view(B, Hp //window_size, Wp //window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     x = self.norm(x)
    #     x = x.permute(0, 2, 3,1)
    #     windows, (Hp, Wp) = self.window_partition(x, self.window_size)
    #     qkv = self.qkv(windows)
    #     q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
    #     scale = 1./math.sqrt(math.sqrt(C // self.num_heads))
    #     attn = torch.einsum('bct,bcs->bts', q*scale, k*scale)
    #     attn = attn.softmax(dim=-1)
    #     h = torch.einsum('bts,bcs->bct', attn, v)
    #     h = self.window_unpartition(h, self.window_size, (Hp, Wp), (H, W))
    #     h = h.reshape(B, -1, H, W)
    #     h = self.proj(h)
    #     return h+x

    def forward(self, x):
        Br, Cr, Hr, Wr = x.shape
        y = self.norm(x)
        y = y.permute(0, 2, 3,1)
        windows, (Hp, Wp) = self.window_partition(y, self.window_size)
        B, H, W, C = windows.shape
        qkv = self.qkv(windows).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B*self.num_heads, H*W, -1).unbind(0)
        scale = (C // self.num_heads)**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        h = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        h = self.proj(h)
        h = self.window_unpartition(h, self.window_size, (Hp, Wp), (Hr, Wr))
        h = h.permute(0, 3, 1, 2)
        return h+x

class PixelAttention(nn.Module):
    '''像素注意力模块'''
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.fc(x)
        return x + x * y
        #return x * y

class SimpleChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimpleChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.fc(x)
        attn = attn.view(-1, self.in_channels, 1, 1)
        return x * attn

if __name__ == '__main__':
    x = torch.randn(8, 32, 120, 160)
    import time
    ga = GlobalAttentionBlock(32, 1, 8)
    start = time.time()
    x = ga(x)
    print(time.time()-start)
