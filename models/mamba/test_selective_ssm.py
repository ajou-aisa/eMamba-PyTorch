from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from models.emamba import EMamba

RESULT_PATH: Final[Path] = Path(__file__).with_name("selective_ssm_test_result.txt")


def check_shape(
    name: str,
    tensor: Tensor,
    expected: tuple[int, ...],
) -> str:
    actual = tuple(tensor.shape)
    assert actual == expected, f"{name}: expected {expected}, got {actual}"
    return f"{name:<28}: {actual}"


@torch.no_grad()
def test_selective_ssm_forward() -> None:
    # Given: EMamba의 기본 MARS 설정으로 SelectiveSSM 입력을 구성한다.
    torch.manual_seed(0)

    emamba = EMamba()
    first_block = emamba.blocks[0]
    model = first_block.ssm
    model.eval()

    batch_size = 1
    frame_height, frame_width = 8, 8
    patch_size = emamba.patch_size
    sequence_length = (frame_height // patch_size) * (frame_width // patch_size)
    d_model = emamba.d_model
    expand = emamba.expand
    d_inner = first_block.d_inner
    d_state = emamba.d_state
    dt_rank = first_block.dt_rank

    assert d_inner == d_model * expand

    # tokens: [B,L,ED]
    tokens = torch.randn(batch_size, sequence_length, d_inner)

    state_shape = (batch_size, d_inner, d_state)
    step_output_shape = (batch_size, d_inner)
    shape_lines: list[str] = []
    step_value_lines: list[str] = []

    # When 1: 하나의 Linear로 Delta 특징, B, C를 동시에 생성한다.
    # [B,L,ED] -> [B,L,dt_rank + 2N]
    ssm_params = model.ssm_param_proj(tokens)
    shape_lines.append(
        check_shape(
            "ssm_param_proj output",
            ssm_params,
            (batch_size, sequence_length, dt_rank + 2 * d_state),
        )
    )

    delta_features, input_b, output_c = ssm_params.split(
        (dt_rank, d_state, d_state),
        dim=-1,
    )

    # When 2: Delta 경로의 ReLU를 적용한다.
    # [B,L,dt_rank] -> [B,L,dt_rank]
    relu_features = torch.relu(delta_features)
    shape_lines.append(
        check_shape(
            "ReLU output",
            relu_features,
            (batch_size, sequence_length, dt_rank),
        )
    )
    assert torch.all(relu_features >= 0), "ReLU output contains a negative value"

    # When 3: dt_rank 차원을 ED 차원으로 확장하여 Delta를 만든다.
    # [B,L,dt_rank] -> [B,L,ED]
    delta = model.delta_proj(relu_features)
    shape_lines.append(
        check_shape(
            "delta",
            delta,
            (batch_size, sequence_length, d_inner),
        )
    )

    # A = -exp(A_log)이므로 A의 모든 원소는 음수이다.
    # A: [ED,N]
    continuous_a = -torch.exp(model.a_log)
    shape_lines.append(check_shape("A", continuous_a, (d_inner, d_state)))
    assert torch.all(continuous_a < 0), "A contains a non-negative value"

    # h_0 = 0으로 시작하여 L개의 토큰을 순차적으로 처리한다.
    state = tokens.new_zeros(batch_size, d_inner, d_state)
    manual_outputs: list[Tensor] = []

    for step in range(sequence_length):
        x_t = tokens[:, step]
        delta_t = delta[:, step]
        b_t = input_b[:, step]
        c_t = output_c[:, step]

        # A_bar_t = exp(Delta_t A), B_bar_t = Delta_t B_t
        a_bar = torch.exp(delta_t.unsqueeze(-1) * continuous_a)
        b_bar = delta_t.unsqueeze(-1) * b_t.unsqueeze(1)

        # h_t = A_bar_t h_(t-1) + B_bar_t x_t
        state = a_bar * state + b_bar * x_t.unsqueeze(-1)

        # y_t = sum_N(C_t h_t) + D x_t
        c_times_h = c_t.unsqueeze(1) * state
        y_t = c_times_h.sum(dim=-1) + model.d_skip * x_t

        shape_lines.extend(
            (
                check_shape(f"step {step} A_bar", a_bar, state_shape),
                check_shape(f"step {step} B_bar", b_bar, state_shape),
                check_shape(f"step {step} h_t", state, state_shape),
                check_shape(f"step {step} C_t x h_t", c_times_h, state_shape),
                check_shape(f"step {step} y_t", y_t, step_output_shape),
            )
        )

        step_value_lines.append(
            f"step={step}\n"
            f"state =\n{state}\n"
            f"a_bar =\n{a_bar}\n"
            f"b_bar =\n{b_bar}\n"
            f"C_t =\n{c_t}\n"
            f"y_t =\n{y_t}"
        )

        manual_outputs.append(y_t)

    # Then: 위에서 수동 계산한 출력과 실제 forward 출력을 비교한다.
    manual_output = torch.stack(manual_outputs, dim=1)
    forward_output = model(tokens)

    shape_lines.append(
        check_shape(
            "forward output",
            forward_output,
            (batch_size, sequence_length, d_inner),
        )
    )
    torch.testing.assert_close(forward_output, manual_output)

    with RESULT_PATH.open("w", encoding="utf-8") as result_file:
        print("0. 테스트 조건", file=result_file)
        print(f"B (test batch size)       = {batch_size}", file=result_file)
        print(f"H × W (frame size)        = {frame_height} × {frame_width}", file=result_file)
        print(f"P (patch_size)            = {patch_size}", file=result_file)
        print(f"L ((H/P)×(W/P))           = {sequence_length}", file=result_file)
        print(f"D (d_model)               = {d_model}", file=result_file)
        print(f"E (expand)                = {expand}", file=result_file)
        print(f"ED (d_inner=D×E)          = {d_inner}", file=result_file)
        print(f"N (d_state)               = {d_state}", file=result_file)
        print(f"dt_rank (ceil(D/16))      = {dt_rank}", file=result_file)

        print("1. 각 지점의 shape", file=result_file)
        print("\n".join(shape_lines), file=result_file)

        print("\n2. ReLU와 A 값", file=result_file)
        print(f"ReLU output =\n{relu_features}", file=result_file)
        print(f"ReLU minimum = {relu_features.min().item():.6f}", file=result_file)
        print(f"A =\n{continuous_a}", file=result_file)
        print(f"A maximum = {continuous_a.max().item():.6f}", file=result_file)

        print("\n3. 시간별 state, a_bar, b_bar, C_t, y_t", file=result_file)
        print("\n\n".join(step_value_lines), file=result_file)

        print("\n4. 수동 계산과 forward 결과 비교", file=result_file)
        print("PASS: manual calculation matches forward output", file=result_file)

    print(f"PASS: results saved to {RESULT_PATH}")


if __name__ == "__main__":
    test_selective_ssm_forward()
