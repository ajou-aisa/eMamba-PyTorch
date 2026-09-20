"""Frame-level regression head for the MARS reproduction."""

from torch import Tensor, nn


class OutputHead(nn.Module):
    """Interface: [B, L, D] -> [B, out_dim]."""

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        readout: str = "mean",
    ) -> None:
        super().__init__()

        if d_model <= 0 or out_dim <= 0:
            raise ValueError("d_model and out_dim must be positive.")

        if readout not in ("mean", "last"):
            raise ValueError(f"Unsupported readout: {readout!r}")

        self.d_model = d_model
        self.out_dim = out_dim
        self.readout = readout

        # Reproduction assumption: one affine projection with bias.
        self.proj = nn.Linear(d_model, out_dim, bias=True)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected [B, L, {self.d_model}], got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] == 0:
            raise ValueError("The token sequence must not be empty.")

        if self.readout == "mean":
            frame = tokens.mean(dim=1)
        else:
            frame = tokens[:, -1, :]

        return self.proj(frame)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, out_dim={self.out_dim}, "
            f"readout={self.readout!r}"
        )


# Reproduction notes:
# The eMamba paper does not specify the exact output-head architecture.
# Use mean pooling as the baseline and last-token readout as an alternative.
# Compare separately trained models using the validation split.