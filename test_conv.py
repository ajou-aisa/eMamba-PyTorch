import torch
from models.mamba.block import MambaConv1D

# conv = MambaConv1D(d_inner=64)

# x = torch.randn(2, 16, 64)
# y = conv(x)

# print("input shape :", x.shape)
# print("output shape:", y.shape)
# print("groups      :", conv.conv1d.groups)
# print("kernel size :", conv.conv1d.kernel_size)
# print("padding     :", conv.conv1d.padding)

torch.manual_seed(0)

conv = MambaConv1D(d_inner=1, d_conv=4)

# 값을 해석하기 쉽게 weight와 bias 고정
with torch.no_grad():
    conv.conv1d.weight.fill_(1.0)
    conv.conv1d.bias.zero_()

# 입력 : [1, 6, 1]
x1 = torch.tensor([
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
    ]
])

# 미래 시점 하나만 변경
x2 = x1.clone()
x2[0, 4, 0] = 100.0


y1 = conv(x1)
y2 = conv(x2)

print("x1:")
print(x1.squeeze())

print("\nx2:")
print(x2.squeeze())

print("\ny1:")
print(y1.squeeze())

print("\ny2:")
print(y2.squeeze())

print("\ndifference:")
print((y2 - y1).squeeze())

assert torch.allclose(y1[:, :4, :], y2[:, :4, :])

print("\nCausal test passed.")