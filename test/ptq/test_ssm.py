import torch

from ptq.ops import integer_ssm


def test_ssm_computes_output_before_state_storage_shift() -> None:
    x = torch.tensor([[[1]]], dtype=torch.int64)
    a = torch.tensor([[[[0]]]], dtype=torch.int64)
    b = torch.tensor([[[[129]]]], dtype=torch.int64)
    c = torch.tensor([[[1]]], dtype=torch.int64)
    d = torch.tensor([0], dtype=torch.int64)
    output, current, stored = integer_ssm(
        x, a, b, c, d, x_exponent=0, b_exponent=0, c_exponent=0,
        d_exponent=0, state_exponent=7, output_exponent=0,
    )
    assert output.item() == 127
    assert current.item() == 129
    assert stored.item() == 1


def test_ssm_state_storage_is_arithmetic_shift() -> None:
    x = torch.tensor([[[-1]]], dtype=torch.int64)
    zeros = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    _, current, stored = integer_ssm(
        x, zeros, torch.full_like(zeros, 129),
        torch.zeros(1, 1, 1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        x_exponent=0, b_exponent=0, c_exponent=0,
        d_exponent=0, state_exponent=7, output_exponent=0,
    )
    assert current.item() == -129
    assert stored.item() == -2


def test_ssm_records_preclip_current_state() -> None:
    recorded: dict[str, tuple[int, int]] = {}
    def record(name: str, value: torch.Tensor, bits: int) -> None:
        recorded[name] = (int(value.item()), bits)
    value = torch.tensor([[[1]]], dtype=torch.int64)
    zeros = torch.zeros(1, 1, 1, 1, dtype=torch.int64)
    integer_ssm(
        value, zeros, torch.full_like(zeros, 1 << 23),
        torch.zeros(1, 1, 1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
        x_exponent=0, b_exponent=0, c_exponent=0, d_exponent=0,
        state_exponent=7, output_exponent=0, recorder=record, prefix="block.ssm",
    )
    assert recorded["block.ssm.currentState"] == (1 << 23, 24)
    assert recorded["block.ssm.state"] == ((1 << 23) - 1 >> 7, 17)
