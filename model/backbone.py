from torchvision.ops import FrozenBatchNorm2d
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBackbone(nn.Module):
    def __init__(self, resnet_model: str, return_layers: dict, dcn=False):
        super(ResnetBackbone, self).__init__()
        self.backbone = getattr(models, resnet_model)(pretrained=True, replace_stride_with_dilation=[False, False, dcn], norm_layer=FrozenBatchNorm2d)
        self.return_layers = return_layers
        self.num_channels = 2048 if resnet_model in ['resnet50', 'resnet101', 'resnet152'] else 512
        del self.backbone.avgpool 
        del self.backbone.fc
        
    def forward(self, x, mask=None):
        outputs = {}
        for name, module in self.backbone.named_children():
            x = module(x)
            if name in self.return_layers:
                outputs[self.return_layers[name]] = x
                
        if mask is not None:
            outputs["mask"] = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        return outputs