"""Output stage shown in Figure 5."""

from torch import Tensor, nn


class OutputHead(nn.Module):
    """Interface: [B, L, D] -> [B, out_dim]."""

    def __init__(self, d_model: int, out_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim

    def forward(self, tokens: Tensor) -> Tensor:
        # TODO:
        # 1. Reduce the token sequence [B, L, D] to one frame representation.
        # 2. Project the representation to the final output dimension.
        #
        # For MARS, the output is [B, 57] for 19 joints x 3 coordinates.
        # The paper does not specify the exact sequence readout method
        # (e.g., last token, mean pooling, or flattening).
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, out_dim={self.out_dim}"