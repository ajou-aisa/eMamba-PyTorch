from collections.abc import Mapping

from torch import Tensor, nn

from models.emamba import EMamba

from .components import prepare_components as prepare_components
from .quant import QuantRuntime


class QEMamba(nn.Module):
    def __init__(
        self, source: EMamba, runtime: QuantRuntime, *,
        continuous_a: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
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
        tokens = self.patch_embedding(self.runtime.boundary("input", frames))
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens)


def prepare_model(
    source: EMamba, runtime: QuantRuntime, *,
    continuous_a: Mapping[str, Tensor] | None = None,
) -> QEMamba:
    return QEMamba(source, runtime, continuous_a=continuous_a)
