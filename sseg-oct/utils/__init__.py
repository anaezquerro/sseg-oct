from typing import Tuple, Union
import numpy as np
import torch.nn as nn

def update_size(
        size: Tuple[int], kernel_size: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        stride: Union[int, Tuple[int]] = 1
    ):
    size = np.array(size)
    kernel_size = np.array(kernel_size) if isinstance(kernel_size, tuple) else kernel_size
    padding = np.array(padding) if isinstance(padding, tuple) else padding
    dilation = np.array(dilation) if isinstance(dilation, tuple) else dilation
    stride = np.array(stride) if isinstance(stride, tuple) else stride
    return np.floor((size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1).astype(int)

def count_parameters(model: nn.Module, only_trainable: bool = False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
