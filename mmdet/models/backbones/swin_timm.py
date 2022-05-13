import timm
import torch.nn as nn
from ..builder import BACKBONES
from ..builder import MODELS
import torchvision.transforms.functional as TF

@BACKBONES.register_module()
class Swin(nn.Module):
    def __init__(self, num_stages=4, model_name='swin_tiny_patch4_window7_224'):
        super(Swin, self).__init__()
        model = timm.create_model(model_name, pretrained=True, num_classes=80)
        self.network = model
        self.num_stages = num_stages
        for i, layer in enumerate(self.network.layers):
            print(i, layer)

        self.out_indices = [0, 1, 2, 3]

    def forward(self, x):
        outs = []
        print(x.size())
        x = TF.crop(x, 1, 3, 224, 224)
        print(x.size())
        x = self.network.patch_embed(x)
        if self.network.absolute_pos_embed is not None:
            x = x + self.network.absolute_pos_embed
        x = self.network.pos_drop(x)
        print(x.size())
        for i, layer in enumerate(self.network.layers):
            x = layer(x)
            # print(layer)
            # print('dafdasfdasfadsfadsfadsfadsfadsfadsfads', x.size())
            # print(int(x[0][1] ** 0.5))

            # print('dafdasfdasfadsfadsfadsfadsfadsfadsfads', x.size())
            # x.reshape(1, x.size(2), int(x.size(1) ** 0.5), -1).contiguous()
            print(x.size())
            if i in self.out_indices:
                outs.append(x.reshape(1, x.size(2), int(x.size(1) ** 0.5), -1).contiguous())
            # x = x.reshape(1, -1, x.size(1)).contiguous()

        return tuple(outs)


