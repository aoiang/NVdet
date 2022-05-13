# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import copy
from ..builder import BACKBONES
from ..utils import CSPLayer


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='ReLU'),
                 # act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@BACKBONES.register_module()
class NASCSPDarknet(BaseModule):



    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='ReLU'),
                 # act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)

        from .network_json import network, out_dataflow
        self.arch = network['backbone']
        # self.layers = []
        self.dataflow = out_dataflow
        self.out_indices = []
        self.layers = nn.ModuleDict()

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        for i in range(len(self.arch['id'])):
            if self.arch['id'][i][1]['type'] == 'Focus':
                self.layers[str(i)] = Focus(
                            int(self.arch['id'][i][1]['in_channel']),
                            int(self.arch['id'][i][1]['out_channel']),
                            kernel_size=self.arch['id'][i][1]['kernel'],
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=dict(type=self.arch['id'][i][1]['activation']))
            elif self.arch['id'][i][1]['type'] == 'ConvModule':
                padding = 1 if self.arch['id'][i][1]['kernel'] == 3 else 0
                self.layers[str(i)] = conv(
                            int(self.arch['id'][i][1]['in_channel']),
                            int(self.arch['id'][i][1]['out_channel']),
                            kernel_size=self.arch['id'][i][1]['kernel'],
                            stride=self.arch['id'][i][1]['stride'],
                            padding=padding,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=dict(type=self.arch['id'][i][1]['activation']))
            elif self.arch['id'][i][1]['type'] == 'SPPBottleneck':
                self.layers[str(i)] = SPPBottleneck(
                    int(self.arch['id'][i][1]['in_channel']),
                    int(self.arch['id'][i][1]['out_channel']),
                    kernel_sizes=self.arch['id'][i][1]['kernel'],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type=self.arch['id'][i][1]['activation']))

            #
            # elif self.arch['id'][i][1]['type'] == 'concat':
            #     self.layers.append('concat')

            if 'final_out' in self.arch['id'][i][1]:
                self.out_indices.append(i)







        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(NASCSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        print("===========here is the size of backbone input===================")
        print(x.size())


        outs = []
        data = {}

        dataflow = copy.deepcopy(self.dataflow)


        for i in range(len(self.arch['id'])):
            # print(self.layers[i])

            # print(i, x.size(), 'heihei')
            if i == 0:
                x = self.layers[str(i)](x)
                data[i] = x
            else:
                if self.arch['id'][i][1]['type'] == 'add':
                    x = data[self.arch['id'][i][1]['prev'][0]]
                    for j in range(1, len(self.arch['id'][i][1]['prev'])):
                        x += data[self.arch['id'][i][1]['prev'][j]]
                        # dataflow[self.arch['id'][i][1]['prev'][j]].remove(i)
                        # if len(dataflow[self.arch['id'][i][1]['prev'][j]]) == 0:
                        #     try:
                        #         del data[dataflow[self.arch['id'][i][1]['prev'][j]]]
                        #     except:
                        #         print('error i is', dataflow[self.arch['id'][i][1]['prev'][j]])
            #
                elif self.arch['id'][i][1]['type'] == 'concat':
                    temp = []
                    for j in range(len(self.arch['id'][i][1]['prev'])):
                        temp.append(data[self.arch['id'][i][1]['prev'][j]])
                        # dataflow[self.arch['id'][i][1]['prev'][j]].remove(i)
                        # if len(dataflow[self.arch['id'][i][1]['prev'][j]]) == 0:
                        #     try:
                        #         del data[dataflow[self.arch['id'][i][1]['prev'][j]]]
                        #     except:
                        #         print('error i is', dataflow[self.arch['id'][i][1]['prev'][j]])
                    x = torch.cat(tuple(temp), dim=1)

                else:
                    x = self.layers[str(i)](data[self.arch['id'][i][1]['prev'][0]])
                    # try:
                    #     dataflow[self.arch['id'][i][1]['prev'][0]].remove(i)
                    # except:
                    #     print('error i is', self.arch['id'][i][1]['prev'][0], i)
                    # if len(dataflow[self.arch['id'][i][1]['prev'][0]]) == 0:
                    #     try:
                    #         del data[dataflow[self.arch['id'][i][1]['prev'][0]]]
                    #     except:
                    #         print('error i is', dataflow[self.arch['id'][i][1]['prev'][0]])
                data[i] = x

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)






