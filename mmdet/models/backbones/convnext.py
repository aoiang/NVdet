import timm
import torch.nn as nn
from ..builder import BACKBONES
from ..builder import MODELS

@BACKBONES.register_module()
class ConvNeXt(nn.Module):
    def __init__(self, num_stages=4, model_name='convnext_small'):
        super(ConvNeXt, self).__init__()
        model = timm.create_model(model_name, pretrained=True, num_classes=80)
        self.network = model
        self.num_stages = num_stages
        for i, layer in enumerate(self.network.stages):
            print(i, layer)

        self.out_indices = [0, 1, 2, 3]

    def forward(self, x):
        outs = []
        x = self.network.stem(x)
        for i, layer in enumerate(self.network.stages):
            x = layer(x)
            # print(layer)
            print('dafdasfdasfadsfadsfadsfadsfadsfadsfads', x.size())
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


