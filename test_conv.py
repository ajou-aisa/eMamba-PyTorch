import torch

from models.mamba.block import MambaConv1D


conv = MambaConv1D(d_inner=64)

x = torch.randn(2, 16, 64)
y = conv(x)

print("input shape :", x.shape)
print("output shape:", y.shape)
print("groups      :", conv.conv1d.groups)
print("kernel size :", conv.conv1d.kernel_size)
print("padding     :", conv.conv1d.padding)