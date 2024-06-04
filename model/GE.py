import math, torch
from torch import nn as nn
import torch.nn.functional as F

from timm.models.layers.create_act import create_act_layer, get_act_layer
from timm.models.layers.create_conv2d import create_conv2d
from timm.models.layers import make_divisible
from timm.models.layers.mlp import ConvMlp
from timm.models.layers.norm import LayerNorm2d

device = torch.device("cuda:2")

class GatherExcite(nn.Module):
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1./16, rd_channels=None,  rd_divisor=1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module(
                    'conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f'conv{i + 1}',
                        create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge) 
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)

class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class SelfAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if out_channels == None:
            self.out_channels = in_channels

        # self.key_channels = 1
        self.key_channels = in_channels // 8
        self.value_channels = in_channels

        self.f_key = nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False)
        self.f_query = nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False)

        self.f_value = nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1, bias=False)

        self.W = nn.Sequential(
            nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        # v1:
        # self.bias_conv_1 = nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False)
        # self.bias_conv_2 = nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False)
        # v2:
        self.bias_conv_1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.bias_conv_2 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.get_bias = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=3 // 2, stride=1),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.get_bias_sigmoid = nn.Sigmoid()
        # self.ge = GatherExcite(in_channels) 
        self.ge = GatherExcite(in_channels)     
        # self.bias_gamma = nn.Parameter(torch.zeros(1)) 
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.ge(x)
        key = self.f_key(key).view(batch_size, self.key_channels, -1)
        # w/o GE
        # key = self.f_key(x).view(batch_size, self.key_channels, -1)      

        sim_map = torch.matmul(query, key)
        bias_1, bias_2 = self.bias_conv_1(x), self.bias_conv_2(x)
        bias_1 = bias_1.contiguous().view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        bias_2 = bias_2.contiguous().view(batch_size, self.key_channels, -1)
        bias = torch.matmul(bias_1, bias_2)
        bias = self.get_bias(bias.unsqueeze(1)).squeeze(1)
        sim_map = F.softmax(sim_map * bias, dim=-1)
        # w/o M
        # sim_map = F.softmax(sim_map, dim=-1)       

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.W(context)

        return self.gamma * context + x


class GlobalSDAware(nn.Module):
    def __init__(self, channel):
        super(GlobalSDAware, self).__init__()
        self.conv_block_1 = nn.Sequential(
            SelfAttentionBlock2D(channel),
            # nn.Conv2d(channel, channel, 1),
            LayerNorm((1,channel,1,1),dim_index=1),
            nn.GELU()
        )
        self.conv_block_2 = nn.Sequential(
            SelfAttentionBlock2D(channel),
            # nn.Conv2d(channel, channel, 1),
            LayerNorm((1,channel,1,1),dim_index=1),
            nn.GELU()
        )


    def forward(self, x):
        # print('ge')
        out1 = self.conv_block_1(x)
        out1 = self.conv_block_2(out1)
        return out1


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    GE = GatherExcite(512)
    output=GE(input)
    print(output.shape)
