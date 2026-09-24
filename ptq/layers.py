import torch
import torch.nn.functional as functional
from brevitas import nn as qnn
from torch import Tensor, nn

from .affine import affine_output, conv_accumulator, linear_accumulator
from .brevitas import fixed_weight_arguments
from .quant import QuantRuntime

# fp32 -> int64 -> accumulator -> fp32
class QLinear(nn.Module):
    def __init__(
        self, source: nn.Linear, prefix: str, input_name: str,
        output_name: str, runtime: QuantRuntime,
    ) -> None:
        super().__init__()
        self.prefix, self.input_name = prefix, input_name
        self.output_name, self.runtime = output_name, runtime
        if runtime.is_frozen:
            self.layer = qnn.QuantLinear(
                source.in_features, source.out_features, bias=False,
                device=source.weight.device, dtype=source.weight.dtype,
                **fixed_weight_arguments(runtime.entry(f"{prefix}.weight").exponent),
            )
            self.layer.weight.data.copy_(source.weight)
        else:
            self.layer = nn.Linear(source.in_features, source.out_features, bias=False)
            self.layer.to(device=source.weight.device, dtype=source.weight.dtype)
            self.layer.weight.data.copy_(source.weight)
        self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().clone())

    def forward(self, value: Tensor) -> Tensor:
        value = self.runtime.boundary(self.input_name, value)
        self.runtime.boundary(f"{self.prefix}.weight", self.layer.weight)
        if self.runtime.is_frozen:
            weights = self.layer.quant_weight().int().to(torch.int64)
            accumulator = linear_accumulator(self.runtime.codes(self.input_name, value), weights)
            bias = None if self.bias is None else self.runtime.boundary(f"{self.prefix}.bias", self.bias)
            result = affine_output(accumulator, bias, self.runtime, prefix=self.prefix,
                                   input_name=self.input_name, output_name=self.output_name)
            return self.runtime.boundary(self.output_name, result)
        result = self.layer(value)
        if self.bias is not None:
            result = result + self.runtime.boundary(f"{self.prefix}.bias", self.bias)
        return self.runtime.boundary(self.output_name, result)


class QConv1d(nn.Module):
    def __init__(self, source: nn.Conv1d, prefix: str, runtime: QuantRuntime) -> None:
        super().__init__()
        self.prefix, self.runtime = prefix, runtime
        if runtime.is_frozen:
            self.layer = qnn.QuantConv1d(
                source.in_channels, source.out_channels, source.kernel_size,
                stride=source.stride, padding=source.padding, groups=source.groups,
                bias=False, device=source.weight.device, dtype=source.weight.dtype,
                **fixed_weight_arguments(runtime.entry(f"{prefix}.weight").exponent),
            )
        else:
            self.layer = nn.Conv1d(
                source.in_channels, source.out_channels, source.kernel_size,
                stride=source.stride, padding=source.padding, groups=source.groups,
                bias=False, device=source.weight.device, dtype=source.weight.dtype,
            )
        self.layer.weight.data.copy_(source.weight)
        self.bias = None if source.bias is None else nn.Parameter(source.bias.detach().clone())

    def forward(self, value: Tensor) -> Tensor:
        value = self.runtime.boundary(f"{self.prefix}.input", value)
        self.runtime.boundary(f"{self.prefix}.weight", self.layer.weight)
        bias = None
        if self.bias is not None:
            bias = self.runtime.boundary(f"{self.prefix}.bias", self.bias)
        if self.runtime.is_frozen:
            input_name = f"{self.prefix}.input"
            weights = self.layer.quant_weight().int().to(torch.int64)
            accumulator = conv_accumulator(self.runtime.codes(input_name, value), weights, self.layer)
            bias = None if bias is None else bias.view(1, -1, 1)
            output_name = f"{self.prefix.rsplit('.', 1)[0]}.conv"
            return affine_output(accumulator, bias, self.runtime, prefix=self.prefix,
                                 input_name=input_name, output_name=output_name)
        return functional.conv1d(
            value, self.layer.weight,
            bias, self.layer.stride, self.layer.padding, self.layer.dilation, self.layer.groups,
        )
