"""Selective SSM boundary from Section 4.3."""

import math

import torch
from torch import Tensor, nn

from ..piecewise import piecewise_exp


class SelectiveSSM(nn.Module):
    """Interface: [B, L, ED] -> [B, L, ED]."""

    def __init__(self, d_inner: int, d_state: int, dt_rank: int, *, use_pwl: bool = True) -> None:
        super().__init__()
        if d_inner <= 0 or d_state <= 0 or dt_rank <= 0:
            raise ValueError("d_inner, d_state, and dt_rank must be positive")

        #ED
        self.d_inner = d_inner
        #N
        self.d_state = d_state
        self.use_pwl = use_pwl

        # Intermediate low-rank width of the delta projection.
        self.dt_rank = dt_rank
        self.delta_activation = "post_projection_relu"
        self.dt_init_min = 1e-3
        self.dt_init_max = 1e-1
        self.dt_init_scale = 1e-3

        #proj 3 simultaneously to get delta, b, c
        self.ssm_param_proj = nn.Linear(
            d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )

        #proj to delta
        self.delta_proj = nn.Linear(
            self.dt_rank,
            d_inner,
            bias=True,
        )
        self._init_delta_parameters()

        # 연속시간 상태 전이 파라미터 A의 초기 크기이다.
        #
        # shape: [ED,N]
        initial_a = torch.arange(
            1,
            d_state + 1,
            dtype=torch.float32,
        ).repeat(d_inner, 1)

        # 실제 A는 forward에서 -exp(a_log)로 계산한다.
        self.a_log = nn.Parameter(torch.log(initial_a))

        # y_t = C_t h_t + D x_t 수식의 D이다.
        # shape: [ED]
        self.d_skip = nn.Parameter(torch.ones(d_inner))        

    def _init_delta_parameters(self) -> None:
        bound = self.dt_init_scale / math.sqrt(self.dt_rank)
        with torch.no_grad():
            self.delta_proj.weight.uniform_(-bound, bound)
            bias = torch.empty_like(self.delta_proj.bias).uniform_(
                math.log(self.dt_init_min), math.log(self.dt_init_max)
            ).exp_()
            self.delta_proj.bias.copy_(bias)

    def compute_delta(self, tokens: Tensor) -> Tensor:
        """Return final delta for transient diagnostics; caller detaches for logging."""
        delta_features = self.ssm_param_proj(tokens)[..., :self.dt_rank]
        return self._project_delta(delta_features)

    def _project_delta(self, delta_features: Tensor) -> Tensor:
        # Provisional choice, not a confirmed author detail: ReLU after projection.
        return torch.relu(self.delta_proj(delta_features))

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.d_inner:
            raise ValueError(f"tokens must have shape [B, L, {self.d_inner}]")
        if tokens.shape[1] == 0:
            raise ValueError("tokens must have a nonempty sequence")

        # token: [B,L,ED]
        batch_size, sequence_length, _ = tokens.shape

        #proj 3 simultaneously
        # [B,L,ED] -> [B,L,dt_rank + 2N]
        ssm_params = self.ssm_param_proj(tokens)

        #divide ssm_params
        delta_features, input_b, output_c = ssm_params.split(
            (
                self.dt_rank,
                self.d_state,  #B
                self.d_state,  #C
            ),
            dim=-1,
        )

        # [B,L,dt_rank] -> [B,L,ED]
        delta = self._project_delta(delta_features)

        # A = -exp(A_log)
        # continuous_a = A : [ED,N]
        continuous_a = -torch.exp(self.a_log)

        # h_0=0
        # state: [B,ED,N]
        state = tokens.new_zeros(
            batch_size,
            self.d_inner,
            self.d_state,
        )

        # store outputs for each token
        outputs: list[Tensor] = []

        #process L tokens sequentially
        for step in range(sequence_length):
            # value of current token
            x_t = tokens[:, step]          # [B,ED]
            delta_t = delta[:, step]       # [B,ED]
            b_t = input_b[:, step]         # [B,N]
            c_t = output_c[:, step]        # [B,N]

            # A_bar_t = exp(Delta_t A)
            # [B,ED,1] * [ED,N] -> [B,ED,N]
            exp_input = delta_t.unsqueeze(-1) * continuous_a
            a_bar = piecewise_exp(exp_input) if self.use_pwl else torch.exp(exp_input)

            # B_bar_t = Delta_t B_t
            # [B,ED,1] * [B,1,N] -> [B,ED,N]
            b_bar = (
                delta_t.unsqueeze(-1) * b_t.unsqueeze(1)
            )

            # h_t = A_bar_t h_(t-1) + B_bar_t x_t
            # state: [B,ED,N]
            state = (
                a_bar * state + b_bar * x_t.unsqueeze(-1)
            )

            # C_t h_t
            # [B,1,N] * [B,ED,N] -> [B,ED]
            y_t = (
                c_t.unsqueeze(1) * state
            ).sum(dim=-1)

            # y_t = C_t h_t + D x_t
            y_t = y_t + self.d_skip * x_t

            outputs.append(y_t)

        # Combine L outputs of shape [B, ED] into a tensor of shape [B, L, ED]
        return torch.stack(outputs, dim=1)


    def extra_repr(self) -> str:
        return f"d_inner={self.d_inner}, d_state={self.d_state}, dt_rank={self.dt_rank}"
