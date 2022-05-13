# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import CSPLayer


@NECKS.register_module()
class YOLOXPAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 # act_cfg=dict(type='Swish'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    # def forward(self, inputs):

    def forward(self, x1, x2, x3):
        inputs = tuple([x1, x2, x3])
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """



        # print("===========here is the size of neck input===================")
        # # print(inputs.size())
        # # print(inputs)
        # print(type(inputs))
        # for d in inputs:
        #     print(d.size())
        # print('=====length of input is inputs======')
        # print(len(inputs))
        #
        # print('=====length of input is in_clannels is======')
        # print(len(self.in_channels))

        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            print("I cant believe=========")
            print(inner_out.size())
            inner_outs.insert(0, inner_out)
            print("I cant believe222222222222222=========")
            for i in range(len(inner_outs)):
                print(inner_outs[i].size())

        # bottom-up path
        outs = [inner_outs[0]]


        print('======here is the output size of top_down==========')
        for i in range(len(inner_outs)):
            print(inner_outs[i].size())

        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)


@NECKS.register_module()
class NASYOLOXPAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 # act_cfg=dict(type='Swish'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(NASYOLOXPAFPN, self).__init__(init_cfg)

        from .network_json import network, out_dataflow
        self.arch = network['neck']


        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.top_down_blcoks = nn.ModuleDict()
        self.bottom_up_blocks = nn.ModuleDict()
        self.final_outs = nn.ModuleDict()

        for i in range(len(self.arch['top_down']['id'])):
            # print(self.arch['top_down']['id'][i][1]['type'])
            if self.arch['top_down']['id'][i][1]['type'] == 'ConvModule':
                padding = 1 if self.arch['top_down']['id'][i][1]['kernel'] == 3 else 0
                # print(int(self.arch['top_down']['id'][i][1]['in_channel']))
                self.top_down_blcoks[str(i)] = conv(
                    int(self.arch['top_down']['id'][i][1]['in_channel']),
                    int(self.arch['top_down']['id'][i][1]['out_channel']),
                    kernel_size=self.arch['top_down']['id'][i][1]['kernel'],
                    stride=self.arch['top_down']['id'][i][1]['stride'],
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type=self.arch['top_down']['id'][i][1]['activation']))
            elif self.arch['top_down']['id'][i][1]['type'] == 'upsample':
                self.top_down_blcoks[str(i)] = nn.Upsample(**dict(scale_factor=2,
                                                                  mode=self.arch['top_down']['id'][i][1]['mode']))

        bottom_up_id = len(self.arch['top_down']['id'])

        # for i in range(bottom_up_id, len(self.arch['bottom_up']['id']) + bottom_up_id):
        for i in range(len(self.arch['bottom_up']['id'])):
            # print('=====i is ', i)
            if self.arch['bottom_up']['id'][i][1]['type'] == 'ConvModule':
                padding = 1 if self.arch['bottom_up']['id'][i][1]['kernel'] == 3 else 0
                self.bottom_up_blocks[str(i)] = conv(
                    int(self.arch['bottom_up']['id'][i][1]['in_channel']),
                    int(self.arch['bottom_up']['id'][i][1]['out_channel']),
                    kernel_size=self.arch['bottom_up']['id'][i][1]['kernel'],
                    stride=self.arch['bottom_up']['id'][i][1]['stride'],
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type=self.arch['bottom_up']['id'][i][1]['activation']))

        final_id = bottom_up_id + len(self.arch['bottom_up']['id'])

        # for i in range(final_id, len(self.arch['final']['id']) + final_id):
        for i in range(len(self.arch['final']['id'])):
            if self.arch['final']['id'][i][1]['type'] == 'ConvModule':
                padding = 1 if self.arch['final']['id'][i][1]['kernel'] == 3 else 0
                self.final_outs[str(i)] = conv(
                    int(self.arch['final']['id'][i][1]['in_channel']),
                    int(self.arch['final']['id'][i][1]['out_channel']),
                    kernel_size=self.arch['final']['id'][i][1]['kernel'],
                    stride=self.arch['final']['id'][i][1]['stride'],
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type=self.arch['final']['id'][i][1]['activation']))


    # def forward(self, inputs):

    def forward(self, x1, x2, x3):
        inputs = tuple([x1, x2, x3])
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """



        print("===========here is the size of neck input===================")

        for d in inputs:
            print(d.size())
        assert len(inputs) == len(self.in_channels)
        data = {}

        # top-down path
        inner_outs = [inputs[-1]]
        data[-1] = inputs[0]
        data[-2] = inputs[1]
        data[-3] = inputs[2]


        for i in range(len(self.arch['top_down']['id'])):
            if self.arch['top_down']['id'][i][1]['type'] == 'add':
                x = data[self.arch['top_down']['id'][i][1]['prev'][0]]
                for j in range(1, len(self.arch['top_down']['id'][i][1]['prev'])):
                    x += data[self.arch['top_down']['id'][i][1]['prev'][j]]

            elif self.arch['top_down']['id'][i][1]['type'] == 'concat':
                temp = []
                for j in range(len(self.arch['top_down']['id'][i][1]['prev'])):
                    temp.append(data[self.arch['top_down']['id'][i][1]['prev'][j]])
                x = torch.cat(tuple(temp), dim=1)
            else:
                try:
                    x = self.top_down_blcoks[str(i)](data[self.arch['top_down']['id'][i][1]['prev'][0]])
                except:
                    print(self.arch['top_down']['id'][i][1]['prev'][0])
                    raise ValueError
            data[i] = x

        stage_1_id = len(self.arch['top_down']['id'])

        # for i in range(stage_1_id, len(self.arch['bottom_up']['id']) + stage_1_id):
        for i in range(len(self.arch['bottom_up']['id'])):
            if self.arch['bottom_up']['id'][i][1]['type'] == 'add':
                x = data[self.arch['bottom_up']['id'][i][1]['prev'][0]]
                for j in range(1, len(self.arch['bottom_up']['id'][i][1]['prev'])):
                    x += data[self.arch['bottom_up']['id'][i][1]['prev'][j]]

            elif self.arch['bottom_up']['id'][i][1]['type'] == 'concat':
                temp = []
                for j in range(len(self.arch['bottom_up']['id'][i][1]['prev'])):
                    temp.append(data[self.arch['bottom_up']['id'][i][1]['prev'][j]])
                try:
                    x = torch.cat(tuple(temp), dim=1)
                except:
                    print(self.arch['bottom_up']['id'][i][1]['prev'])
                    raise ValueError
            else:
                try:
                    x = self.bottom_up_blocks[str(i)](data[self.arch['bottom_up']['id'][i][1]['prev'][0]])
                except:
                    print(self.arch['bottom_up']['id'][i][1]['prev'][0])
                    raise ValueError
            data[i + stage_1_id] = x

        stage_2_id = len(self.arch['bottom_up']['id'])

        outs = []
        for i in range(len(self.arch['final']['id']) ):
        # for i in range(stage_2_id, len(self.arch['final']['id']) + stage_2_id):
            x = self.final_outs[str(i)](data[self.arch['final']['id'][i][1]['prev'][0]])
            outs.append(x)

        return tuple(outs)





