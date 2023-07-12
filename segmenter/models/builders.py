import segmentation_models_pytorch as smp
import torch.nn as nn
from typing import Callable, Union

class BuilderSMP:

    builders = {'unet': smp.Unet, 'linknet': smp.Linknet, 'pspnet': smp.PSPNet, 'pan': smp.PAN}
    image_sizes = {'unet': (416, 640), 'linknet': (416, 640), 'pspnet': (416, 640), 'pan': (416, 640)}

    def __init__(self, builder: Union[str, Callable], encoder_name: str = 'resnet50', encoder_weights: str = 'imagenet'):
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.builder = BuilderSMP.builders[builder] if isinstance(builder, str) else builder

    def __call__(self):
        model = self.builder(in_channels=1, classes=1, encoder_name=self.encoder_name, encoder_weights=self.encoder_weights)
        return model




