import torch
import torch.nn as nn
from typing import Tuple
from utils import update_size

class Discriminator(nn.Module):
    def __init__(self, in_channels: int, img_size: Tuple[int]):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size

        dilation1, stride1, kernel_size1 = 4, 4, 7
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size1, stride=stride1, dilation=dilation1)
        img_size = update_size(size=img_size, kernel_size=kernel_size1, stride=stride1, dilation=dilation1)

        dilation2, stride2, kernel_size2 = 2, 4, 7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size2, stride=stride2, dilation=dilation2)
        img_size = update_size(size=img_size, kernel_size=kernel_size2, stride=stride2, dilation=dilation2)

        dilation3, stride3, kernel_size3 = 2, 4, 7
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size3, stride=stride3, dilation=dilation3)
        img_size = update_size(size=img_size, kernel_size=kernel_size3, stride=stride3, dilation=dilation3)

        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=img_size[0]*img_size[1]*32, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        y = self.cls(x3)
        return y



