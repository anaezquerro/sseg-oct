import torch.nn as nn
import torch
from modules import DeformableConv2d
from torchvision.transforms import Pad

def deform_conv(in_channels, out_channels, kernel_size: int, padding: int = 0, stride: int = 1):
    return nn.Sequential(
        DeformableConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.ReLU(inplace=True),
    )


class DeformUNet(nn.Module):
    IMAGE_SIZE = (416, 624)
    def __init__(self, input_channels: int = 1):
        super().__init__()

        self.dconv_down1 = deform_conv(input_channels, 64, 11, stride=2)
        self.dconv_down2 = deform_conv(64, 128, 7, stride=2)
        self.dconv_down3 = deform_conv(128, 256, 7, stride=2)
        self.dconv_down4 = deform_conv(256, 512, 5)


        self.maxpool = nn.MaxPool2d(3)
        self.upsample = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)

        self.conv_up3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(5, 5))
        self.conv_up2 = nn.ConvTranspose2d(in_channels=256+512, out_channels=256, kernel_size=(7, 7), stride=(2, 2))
        self.conv_up1 = nn.ConvTranspose2d(in_channels=256+128, out_channels=256, kernel_size=(7, 7), stride=(2, 2))
        self.conv_up0 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(11, 11), stride=(2, 2))

        self.act = nn.Sigmoid()

    def forward(self, x, verbose: bool = False):
        # forward pass of the encoder
        down1 = self.dconv_down1(x)
        down2 = self.dconv_down2(down1)
        down3 = self.dconv_down3(down2)
        down4 = self.dconv_down4(down3)
        if verbose:
            print(down1.shape, down2.shape, down3.shape, down4.shape)

        # forward pass of the decoder
        up3 = self.conv_up3(down4)
        up2 = self.conv_up2(torch.concat([up3, down3], dim=1))
        up1 = self.conv_up1(torch.concat([up2, down2], dim=1))

        out = self.conv_up0(up1)
        out = Pad(padding=(0, 1, 1, 0))(out)

        if verbose:
            print(up3.shape, up2.shape, up1.shape, out.shape)

        return self.act(out)

    @classmethod
    def build(cls, input_channels: int = 1):
        model = cls(input_channels=input_channels)
        return model


if __name__ == '__main__':
    from utils import count_parameters
    model = DeformUNet()
    inputs = torch.rand((10, 1, 416, 640))
    model(inputs)