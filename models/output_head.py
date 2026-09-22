"""Frame-level regression head for the MARS reproduction."""

from torch import Tensor, nn


class OutputHead(nn.Module):
    """Interface: [B, L, D] -> [B, out_dim]."""

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        readout: str = "flatten",
        *,
        num_tokens: int = 16,
    ) -> None:
        super().__init__()

        if d_model <= 0 or out_dim <= 0 or num_tokens <= 0:
            message = "d_model, out_dim and num_tokens must be positive."
            raise ValueError(message)

        if readout != "flatten":
            raise ValueError(f"Unsupported readout: {readout!r}")

        self.d_model = d_model
        self.out_dim = out_dim
        self.readout = readout
        self.num_tokens = num_tokens
        self.flatten = nn.Flatten(start_dim=1)

        self.proj = nn.Sequential(
            nn.Linear(num_tokens * d_model, d_model, bias=True),
            nn.ReLU(),
            nn.Linear(d_model, out_dim, bias=True),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.num_tokens, self.d_model):
            raise ValueError(
                f"Expected [B, {self.num_tokens}, {self.d_model}], got {tuple(tokens.shape)}"
            )
        return self.proj(self.flatten(tokens))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, out_dim={self.out_dim}, "
            f"readout={self.readout!r}, num_tokens={self.num_tokens}"
        )


# Reproduction notes:
# The eMamba paper does not specify the exact output-head architecture.
