from collections.abc import Mapping

from torch import Tensor, nn

# Import the quantization runtime from ptq/quant.py.
from ptq.quant import QuantRuntime

from .emamba import EMamba


class QEMamba(nn.Module):
    def __init__(
        self,

        source: EMamba,
        runtime: QuantRuntime,
        *,
        continuous_a: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        from ptq.model import prepare_components

        self.d_model = source.d_model
        self.expand = source.expand
        self.patch_size = source.patch_size
        self.num_blocks = source.num_blocks
        self.d_state = source.d_state
        self.runtime = runtime
        self.patch_embedding, self.blocks, self.head = prepare_components(
            source, runtime, continuous_a=continuous_a,
        )

    def forward(self, frames: Tensor) -> Tensor:

        # quantize input frames ->  x scale -> dtype : fp32
        # use an object from patch_embedding.py
        # replace proj(nn.linear) -> QLinear (ptq/layers.py)
        tokens = self.patch_embedding(self.runtime.boundary("input", frames))

        # # use an object from q_block.py
        for block in self.blocks:
            tokens = block(tokens)

        # use an object from output_head.py
        # replace proj(nn.linear) -> QLinear (ptq/layers.py)
        return self.head(tokens)
