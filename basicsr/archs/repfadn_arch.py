import torch
import torch.nn as nn
import torch.nn.functional as F
from .repfadn_blocks import Block
from .repfadn_layers import MutilBranchConv
from .repfadn_commonblock import Stem, PSUpsample, Conv1x1Block, MBConv3x3_Conv1x1_Bn_Act, PSMBUpsample, Parallel_MBConv3x3_MBConv1x1_Bn_Act, PSPC3C1Upsample, UpsampleOneStep, MBUpsampleOneStep
from basicsr.utils.registry import ARCH_REGISTRY
import time

@ARCH_REGISTRY.register()
class RepFADN(nn.Module):
    def __init__(self, img_channels, feature_channels, up_scale=4, depth_multiplier=2, num_conv=2, se_factor=2, ffn_expand=2, res_scale=1, dropout_rate=0, use_mbconv=True,
                 act_type='linear', block_act_num=3, block_num=4, with_idt=False, use_bn= False, train_act=False, deploy=False, img_range=255., rgb_mean=[0.4488, 0.4371, 0.4040]):
        super().__init__()
        self.img_channels = img_channels
        self.feature_channels = feature_channels
        self.up_scale = up_scale
        self.depth_multiplier = depth_multiplier
        self.num_conv = num_conv
        self.se_factor = se_factor
        self.ffn_expand = ffn_expand
        self.res_scale = res_scale
        self.block_num = block_num
        self.dropout_rate = dropout_rate
        self.use_mbconv = use_mbconv
        self.act_type = act_type
        self.block_act_num = block_act_num
        self.with_idt = with_idt
        self.use_bn = use_bn
        self.train_act = train_act
        self.deploy = deploy
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.stem = Stem(self.img_channels, self.feature_channels, self.depth_multiplier, self.num_conv, self.act_type,
                         self.block_act_num, self.use_mbconv, self.with_idt, self.use_bn, self.train_act, self.deploy)
        # self.stem = Parallel_MBConv3x3_MBConv1x1_Bn_Act(self.img_channels, self.feature_channels, self.depth_multiplier,
        #                                                      self.num_conv, self.act_type, self.block_act_num, self.use_mbconv,
        #                                                      self.with_idt, self.use_bn, self.deploy)
        self.bicubic_upsample = nn.Upsample(scale_factor=self.up_scale, mode='bicubic')
        # self.upsample = PSUpsample(self.up_scale, self.feature_channels)

        # self.upsample = PSMBUpsample(self.up_scale, self.feature_channels, self.depth_multiplier, self.num_conv,
        #                              self.with_idt, self.use_bn, self.deploy)
        # self.upsample_lastconv = nn.Sequential(nn.Conv2d(self.feature_channels, self.img_channels, 3, padding=1, bias=True),
        #                                        nn.GELU(),
        #                                        nn.Conv2d(self.img_channels, self.img_channels, 3, padding=1, bias=True))

        # self.upsample = UpsampleOneStep(self.up_scale, self.feature_channels, self.img_channels)
        self.upsample = MBUpsampleOneStep(self.up_scale, self.feature_channels, self.img_channels, self.num_conv, self.deploy)

        blocks = []

        channels = self.feature_channels
        for _ in range(self.block_num):
            blocks += [Block(channels, self.depth_multiplier, self.num_conv, self.se_factor, self.ffn_expand, self.res_scale,
                                                       self.dropout_rate, self.use_mbconv, self.act_type, self.block_act_num,
                                                       self.with_idt, self.use_bn, self.train_act, self.deploy)]
        self.blocks = nn.Sequential(*blocks)
        self.apply(self._init_weight)

    def _init_weight(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            #nn.init.constant_(m.bias, 0)

    def change_act_learn(self, m):
        self.stem.act_learn = m
        #self.upsample_conv1.act_learn = m
        #self.upsample_conv2.act_learn = m
        for block in self.blocks:
            block.act_learn = m

    def forward(self, inputs):
        #self.mean = self.mean.type_as(inputs)

        #x = (inputs - self.mean) * self.img_range
        if self.img_channels == 1:
            inputs = inputs[:, 0:1, :, :]
        x = self.stem(inputs)
        x = self.blocks(x)
        x = self.upsample(x)
        # x = self.upsample_lastconv(x)
        x = x + self.bicubic_upsample(inputs)
        #x = x / self.img_range + self.mean + self.bicubic_upsample(inputs)
        if self.img_channels == 1:
            x = torch.concat([x, x, x], dim=1)
        return x


    def switch_to_deploy(self):
        self.stem.switch_to_deploy()
        for block in self.blocks:
            block.switch_to_deploy()
        self.upsample.switch_to_deploy()
        # self.upsample_lastconv.switch_to_deploy()
        # self.upsample_lastconv1.switch_to_deploy()
        # self.upsample_lastconv2.switch_to_deploy()
        self.deploy = True


if __name__ == '__main__':
    # test block

    x = (torch.randn(1, 1, 120, 160)*200).cuda()
    net = RepFADN(1, 16, block_num=4, act_type='gelu', use_bn=False, train_act=False, deploy=False).cuda()
    net.eval()
    with torch.no_grad():
        start = time.time()
        for i in range(100):
            y0 = net(x)
        print((time.time()-start)/100)
        net.switch_to_deploy()
        start = time.time()
        for i in range(100):
            y1 = net(x)
        print((time.time()-start)/100)
        #RK, RB = ecb.rep_params()
        #y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        print(y0 - y1)
