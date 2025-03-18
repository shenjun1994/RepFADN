import torch
import torch.nn as nn
import torch.nn.functional as F
from .repfadn_commonblock import FFNConv1x1Block, Conv1x1Block, Conv_Bn_Act, MBConv3x3_MBConv1x1_Bn_Act, LayerNorm2d, \
    MBConv1x1Block, MBConv3x3_Conv1x1_Bn_Act, Res_MBConv3x3_MBConv1x1_Bn_Act, MBConv1x1BlockTest
from .repfadn_attention import SimpleSEBlock, GlobalAttentionBlock, PixelAttention, SEBlock, SimpleChannelAttention
import time

class Block(nn.Module):
    def __init__(self, channels, depth_multiplier=2, num_conv=4, se_factor=2, ffn_expand=2, res_scale=1, dropout_rate=0, use_mbconv=True,
                 act_type='linear', block_act_num=3, with_idt=False, use_bn= True, train_act=True, deploy=False):
        super().__init__()
        self.act_learn = 1
        self.train_act = train_act
        self.deploy = deploy
        self.use_bn = use_bn
        self.use_mbconv = use_mbconv
        self.res_scale = res_scale
        self.dropout_rate = dropout_rate
        self.with_idt = with_idt
        self.act_type = act_type
        self.se_factor = se_factor
        self.ffn_expand = ffn_expand
        self.channels = channels
        self.depth_multiplier = depth_multiplier
        self.block_act_num = block_act_num
        self.num_conv = num_conv

        self.conv1 = Res_MBConv3x3_MBConv1x1_Bn_Act(self.channels, self.depth_multiplier, self.num_conv,
                                            self.act_type, self.block_act_num, self.use_mbconv, self.with_idt,
                                            self.use_bn, self.deploy)

        self.conv2 = Res_MBConv3x3_MBConv1x1_Bn_Act(self.channels, self.depth_multiplier, self.num_conv,
                                            self.act_type, self.block_act_num, self.use_mbconv, self.with_idt,
                                            self.use_bn, self.deploy)

        # self.sca = SEBlock(self.channels, 4)
        self.sca = SimpleChannelAttention(self.channels)
        self.pa = PixelAttention(self.channels)

        self.conv1x1_1 = MBConv1x1Block(self.channels, 2*self.channels, 'linear', self.block_act_num, self.use_mbconv,
                                        self.num_conv, self.with_idt, self.use_bn, self.deploy)

        self.conv1x1_2 = MBConv1x1Block(2*self.channels, self.channels, 'linear', self.block_act_num, self.use_mbconv,
                                        self.num_conv, self.with_idt, self.use_bn, self.deploy)

        # self.conv1x1_1 = MBConv1x1BlockTest(self.channels, 2*self.channels, 'linear', self.block_act_num, self.use_mbconv,
        #                                 5, self.with_idt, self.use_bn, self.deploy)
        #
        # self.conv1x1_2 = MBConv1x1BlockTest(2*self.channels, self.channels, 'linear', self.block_act_num, self.use_mbconv,
        #                                 5, self.with_idt, self.use_bn, self.deploy)


        #self.ga = GlobalAttentionBlock(self.channels, 8, 8)
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv1x1_1(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.sca(x1)
        x1 = self.pa(x1)
        x2 = self.conv2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1_2(x)
        #x = self.ga(x)
        y = x * self.res_scale + inputs
        # y = self.sca(y)
        # y = self.pa(y)
        return y

    def switch_to_deploy(self):
        self.conv1.switch_to_deploy()
        self.conv2.switch_to_deploy()
        self.conv1x1_1.switch_to_deploy()
        self.conv1x1_2.switch_to_deploy()
        self.deploy = True









if __name__ == '__main__':
    # # test block
    # x = torch.randn(1, 8, 5, 5) * 2000
    # block = SimpleSEBlock(8, 4)
    # block.eval()
    # y0 = block(x)
    #
    # block.switch_to_deploy()
    # y1 = block(x)
    # #RK, RB = ecb.rep_params()
    # #y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    # print(y0 - y1)

    # # test stem
    # x = torch.randn(1, 3, 5, 5) * 200
    # stem = Stem(3, 32, 3, use_bn=True)
    # stem.eval()
    # y0 = stem(x)
    #
    # stem.switch_to_deploy()
    # y1 = stem(x)
    # #RK, RB = ecb.rep_params()
    # #y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    # print(y0 - y1)


    # test block
    x = torch.randn(1, 16, 120, 160) * 200
    block = Block(16, 2, use_bn=False, train_act=False, deploy=False)
    block.eval()
    start = time.time()
    y0 = block(x)
    print(time.time()-start)
    block.switch_to_deploy()
    start = time.time()
    y1 = block(x)
    print(time.time()-start)
    #RK, RB = ecb.rep_params()
    #y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print(y0 - y1)
