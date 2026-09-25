import torch
from torch import Tensor, nn

from ptq.ops import dequantize_codes, integer_ssm_step
from ptq.quant import QuantizationError, QuantRuntime
from ..piecewise import piecewise_exp


class QSelectiveSSM(nn.Module):
    def __init__(
        self, ssm_param_proj: nn.Module, delta_proj: nn.Module,
        continuous_a: Tensor, d_skip: Tensor, dt_rank: int,
        prefix: str, runtime: QuantRuntime,
    ) -> None:
        super().__init__()
        self.ssm_param_proj = ssm_param_proj
        self.delta_proj = delta_proj
        self.register_buffer("continuous_a", continuous_a)
        self.continuous_a: Tensor = continuous_a.detach().clone()
        self.d_skip = nn.Parameter(d_skip.detach().clone(), requires_grad=False)
        self.dt_rank = dt_rank
        self.d_state = continuous_a.shape[-1]
        self.prefix = prefix
        self.runtime = runtime

    def _project_parameters(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        # project 3 both to get delta, b, c
        projected = self.ssm_param_proj(tokens)

        # split
        delta_input, input_b, output_c = projected.split(
            (self.dt_rank, self.d_state, self.d_state), dim=-1,
        )

        # delta projection + relu
        delta = torch.relu(self.delta_proj(delta_input))

        # Quantization
        delta = self.runtime.boundary(f"{self.prefix}.delta", delta)
        input_b = self.runtime.boundary(f"{self.prefix}.B", input_b)
        output_c = self.runtime.boundary(f"{self.prefix}.C", output_c)
        return delta, input_b, output_c, self.runtime.boundary(
            f"{self.prefix}.D", self.d_skip,
        )

    def _float_forward(
        self, tokens: Tensor, delta: Tensor, input_b: Tensor,
        output_c: Tensor, d_skip: Tensor,
    ) -> Tensor:
        continuous_a = self.runtime.boundary(f"{self.prefix}.A", self.continuous_a)
        state = tokens.new_zeros(tokens.shape[0], tokens.shape[2], self.d_state)
        outputs: list[Tensor] = []
        for step in range(tokens.shape[1]):
            x_t, delta_t = tokens[:, step], delta[:, step]
            exp_input = self.runtime.boundary(
                f"{self.prefix}.expInput", delta_t.unsqueeze(-1) * continuous_a,
            )
            a_bar = piecewise_exp(exp_input) if self.runtime.use_pwl else torch.exp(exp_input)
            b_bar = delta_t.unsqueeze(-1) * input_b[:, step].unsqueeze(1)
            a_bar = self.runtime.boundary(f"{self.prefix}.Abar", a_bar)
            b_bar = self.runtime.boundary(f"{self.prefix}.Bbar", b_bar)
            state = a_bar * state + b_bar * x_t.unsqueeze(-1)
            self.runtime.boundary(f"{self.prefix}.currentState", state, 24)
            y_t = (output_c[:, step].unsqueeze(1) * state).sum(-1)
            y_t = y_t + d_skip * x_t
            outputs.append(self.runtime.boundary(f"{self.prefix}.y", y_t))
        return torch.stack(outputs, dim=1)

    def _integer_forward(
        self, tokens: Tensor, delta: Tensor, input_b: Tensor,
        output_c: Tensor, d_skip: Tensor,
    ) -> Tensor:

        # Validate Abar and state scales for the 7-bit state storage shift
        abar_entry = self.runtime.entry(f"{self.prefix}.Abar")
        current_entry = self.runtime.entry(f"{self.prefix}.currentState")
        state_entry = self.runtime.entry(f"{self.prefix}.state")
        if abar_entry.exponent != -7 or current_entry.exponent + 7 != state_entry.exponent:
            raise QuantizationError("SSM Abar/state exponent relation is invalid")

        # quantization of continuous A
        a = self.runtime.boundary(f"{self.prefix}.A", self.continuous_a)

        names = {key: f"{self.prefix}.{key}" for key in ("x", "Abar", "Bbar", "C", "D", "state", "y")}
        x_codes = self.runtime.codes(names["x"], tokens)
        c_codes = self.runtime.codes(names["C"], output_c)
        d_codes = self.runtime.codes(names["D"], d_skip)
        state = torch.zeros(
            tokens.shape[0], tokens.shape[2], self.d_state, dtype=torch.int64, device=tokens.device,
        )
        outputs: list[Tensor] = []
        for step in range(tokens.shape[1]):
            delta_t = delta[:, step]
            # delta x A
            exp_input = self.runtime.boundary(
                f"{self.prefix}.expInput", delta_t.unsqueeze(-1) * a,
            )
            # a_bar
            a_bar = piecewise_exp(exp_input) if self.runtime.use_pwl else torch.exp(exp_input)
            # b_bar
            b_bar = delta_t.unsqueeze(-1) * input_b[:, step].unsqueeze(1)
            # Quantization of Abar and Bbar
            a_bar = self.runtime.boundary(names["Abar"], a_bar)
            b_bar = self.runtime.boundary(names["Bbar"], b_bar)
            output, _, state = integer_ssm_step(
                x_codes[:, step], self.runtime.codes(names["Abar"], a_bar),
                self.runtime.codes(names["Bbar"], b_bar), c_codes[:, step], d_codes, state,
                x_exponent=self.runtime.entry(names["x"]).exponent,
                b_exponent=self.runtime.entry(names["Bbar"]).exponent,
                c_exponent=self.runtime.entry(names["C"]).exponent,
                d_exponent=self.runtime.entry(names["D"]).exponent,
                state_exponent=state_entry.exponent,
                output_exponent=self.runtime.entry(names["y"]).exponent,
                recorder=self.runtime.record_codes, prefix=self.prefix,
            )
            outputs.append(output)

        result = dequantize_codes(torch.stack(outputs, 1), self.runtime.entry(names["y"]).exponent, tokens.dtype)
        return self.runtime.boundary(names["y"], result)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.runtime.boundary(f"{self.prefix}.x", tokens)
        delta, input_b, output_c, d_skip = self._project_parameters(tokens)
        if self.runtime.is_frozen:
            return self._integer_forward(tokens, delta, input_b, output_c, d_skip)
        return self._float_forward(tokens, delta, input_b, output_c, d_skip)
