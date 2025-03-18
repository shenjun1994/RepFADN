import glob
import math
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Input : X :(B,C,H,W)
        Return : X :(B,H*W,C)
        """
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)  # --Layernorm 1
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        """
        Input : X : (B,H*W,C)  X_size: (H,W) Return : X (B,C,H,W)
        """
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B,C, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class Mlp(nn.Module):
    def __init__(self,in_features ,hidden_features=None,out_features = None,act_layer=nn.GELU):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features,out_features)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ChannelAttention4(nn.Module):
    '''
    input : X[B,N,C]
    output : [B,N,C]

    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2] # ( B , N , C )
        k = k * self.scale
        attention = k.transpose(-1, -2) @ v #(B,C,N)* (B,N,C) = (B,C,C)
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2) #(B,C,C) * (B,C,N) = (B,C,N)
        x = self.proj(x)
        return x


class ChannelAttention2(nn.Module):
    '''
    CA
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.window_size = 8

    def forward(self, x, size):
        '''
        x:[B,L,C] , size = H,W
        :param x:windows[B,Nw,N,C] N =win * win
        :return:
        '''
        H, W = size
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C).view(B, -1,self.window_size * self.window_size,C)
        B, Nw, N, C = x.shape
        x = x.view(B, -1, C)
        qkv = self.qkv(x).reshape(B, Nw, N, 3, C).permute(3, 0, 1, 2, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        # b,Nw,C,C
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        # [B,Nw,C,N->B,Nw,N,C]
        x = x.transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        return x




class ChannelAttention(nn.Module):
    '''
    Input : x [B.L.C]
    Output : [B,L,C]

    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ChannelBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 ffn_mode = Mlp, CA_mode = ChannelAttention4 ):
        super().__init__()
        self.ffn = ffn_mode
        self.norm1 = norm_layer(dim)
        self.attn = CA_mode(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        if self.ffn is not None:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = ffn_mode(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
        self.CA = CA_mode

    def forward(self, x,size):
        xc = x
        cur = self.norm1(x)
        if self.CA == ChannelAttention2:
            cur = self.attn(cur,size)
        else :
            cur = self.attn(cur)
        x = xc + cur
        if self.ffn:
            x = x + self.mlp(self.norm2(x))
        return x


def window_partition(x,window_size):
    """
    this function will change the input and spare it to many windows
    Input :
        x : Tensor ( B,H,W,C)
        windows_size : the size of the windows
    Return :
        Windows: (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = windows.view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    this function will change the input from windows to a picture
    Input:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self,dim, window_size , num_heads ,qkv_bias = True ,qk_scale = None):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** (-0.5)

        # relative position bias :1,relative position bias table 2.relative position index
        # 1.relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # 2.relative position index
        # 2-1 absolute position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # 2-2 2D relative position index
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 2-3 1D relative position index
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,mask = None):
        B_, N, C = x.shape # B_ = Num_Win * B ,N =Win_size * Win_size
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)  （B_,nh,N,Cd）
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.to(x.device)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class Spacial_Block(nn.Module):
    def __init__(self,dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(Spacial_Block, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution # what is this
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        #control the input size min
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        if norm_layer is not None:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
        else:
            self.norm1 = None
            self.norm2 = None
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio), act_layer=act_layer)

        if self.shift_size > 0:
            self.attn_mask = self.calculate_mask(self.input_resolution)
        else :
            self.attn_mask = None

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self,x,x_size):
        """
        Input :X (B,H*W ,C)
        x.size = H,W --Window_reverse
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        #x = self.channel_attention(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = x.view(B, H, W, C)
        #the image roll of that for shift windows SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size*window_size, C

        if self.input_resolution == x_size:  # mask = None if shife_size = 0 else calalar_mask(input_resolution)
            mask = self.attn_mask
        else:
            mask = self.calculate_mask(x_size).to(x.device)

        attn_windows = self.attn(x_windows,mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)


        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class DATransformer(nn.Module):
    def __init__(self,dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm, CA = ChannelAttention4 ):
        super(DATransformer, self).__init__()
        self.shift_size = shift_size

        self.SA_BLOCK = Spacial_Block(dim,
                                      input_resolution,
                                      num_heads,
                                      window_size=window_size,
                                      shift_size=shift_size,
                                     mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                    act_layer=act_layer,
                                    norm_layer=norm_layer)

        self.CA_BLOCK = ChannelBlock(dim,
                                     num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer,
                                     ffn_mode= Mlp,
                                     CA_mode = CA)

    def forward(self , x,x_size):
        '''
        x = [B,L,C] -> [B,L,C]
        '''
        out = self.SA_BLOCK(x,x_size)
        out = self.CA_BLOCK(out,x_size)
        return out

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,act_layer= nn.ReLU,
                  norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        # build blocks
        self.blocks = nn.ModuleList([
            DATransformer(dim=dim,
                            input_resolution=input_resolution,
                           num_heads=num_heads,
                            window_size=window_size,
                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            act_layer=act_layer,
                            norm_layer=norm_layer)
            for i in range(depth)])
    def forward(self, x, x_size):
        ''' Input  [B,L,C] --> Output [B,L,C]'''
        for blk in self.blocks:
            x = blk(x, x_size)
        return x

class RSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         act_layer=act_layer,
                                         norm_layer=norm_layer)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(embed_dim=dim,norm_layer=None)
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x, x_size):
        res = x
        x = self.residual_group(x, x_size)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x) + res
        return x


@ARCH_REGISTRY.register()
class DASR(nn.Module):
    def __init__(self, in_chans=3,
                 embed_dim=96,
                 depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 train_patchsize = (48,48),
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.ReLU,
                 patch_norm=True,
                 upscale=2,
                 img_range=1.,
                 resi_connection='1conv',
                 **kwargs):
        super(DASR, self).__init__()
        #print(train_patchsize)

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None)
        self.patch_unembed = PatchUnEmbed()

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=train_patchsize,
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,# no impact on SR results
                         act_layer=act_layer,
                         norm_layer=norm_layer,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)

        if norm_layer is not None :
            self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # for classical SR
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                              nn.ReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)
        if self.norm is not None:
            x = self.norm(x)  # B L C   ---Layernorm
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        mid = self.forward_features(x)
        #torch.save(mid,"/home/liangshubo/Image_SR/Code/Test/CNN_VIT_Compare/DASR1.pt")
        x = self.conv_after_body(mid) + x



        x = self.conv_before_upsample(x)

        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x[:, :, :H*self.upscale, :W*self.upscale]



def make_model(args, parent=False):
    if args.sw_mode == 'S':
        return DASR(in_chans=3,upscale=args.scale[0],
                 embed_dim=60,
                 depths=[2,2,2,2,2,2],
                 num_heads=[6,6,6,6,6,6],
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 train_patchsize = args.input_size ,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.ReLU,
                 patch_norm=True,
                 img_range=1.,
                 resi_connection='1conv')

    elif args.sw_mode == 'B':
        return DASR(in_chans=3,upscale=4,
                 embed_dim=120,
                 depths=[2, 4, 6, 6,4,2],
                 num_heads=[6, 6, 6, 6,6,6],
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 train_patchsize = args.input_size,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.ReLU,
                 patch_norm=True,
                 img_range=1.,
                 resi_connection='1conv')
    elif args.sw_mode == 'L':
        return DASR(in_chans=3,upscale=4,
                 embed_dim=180,
                 depths=[6, 6, 6, 6,6,6],
                 num_heads=[6, 6, 6, 6,6,6],
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 train_patchsize = args.input_size,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.ReLU,
                 patch_norm=True,
                 img_range=1.,
                 resi_connection='1conv')
    else:
        return DASR(upscale=4, img_size=(32,32),
                      window_size=8, img_range=255., depths=[6, 6, 6, 6,6,6],
                      embed_dim=180, num_heads=[6, 6, 6, 6,6,6], mlp_ratio=2)
