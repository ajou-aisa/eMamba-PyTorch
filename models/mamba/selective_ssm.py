"""Selective SSM boundary from Section 4.3."""

from torch import Tensor, nn


class SelectiveSSM(nn.Module):
    """Interface: [B, L, ED] -> [B, L, ED]."""

    def __init__(self, d_inner: int, d_state: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

    def forward(self, tokens: Tensor) -> Tensor:
        # TODO:
        # 1. Generate input-dependent Delta, B, and C from each token.
        # 2. Apply ReLU to Delta instead of Softplus.
        # 3. Discretize the continuous-time parameters A and B.
        # 4. Update the recurrent state h_t for each token.
        # 5. Compute y_t = C_t h_t + D x_t.
        #
        # FP32 training uses the regular exponential function.
        # Piecewise exponential approximation is applied only for inference.
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"d_inner={self.d_inner}, d_state={self.d_state}"