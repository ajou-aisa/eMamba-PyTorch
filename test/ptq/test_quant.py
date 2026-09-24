import pytest
import torch

from ptq.ops import checked_add, checked_mul, dequantize_codes, quantize_codes, round_shift, saturate
from ptq.quant import QuantEntry, QuantProfile, QuantRuntime


def test_quantize_codes_uses_full_signed_range_and_ties_even() -> None:
    values = torch.tensor([-200.0, -1.5, -0.5, 0.5, 1.5, 200.0])
    codes = quantize_codes(values, exponent=0, bits=8)
    assert codes.tolist() == [-128, -2, 0, 0, 2, 127]


def test_round_shift_uses_ties_to_even_but_state_shift_is_arithmetic() -> None:
    values = torch.tensor([-129, -128, -64, 64, 128, 192], dtype=torch.int64)
    assert round_shift(values, shift=7).tolist() == [-1, -1, 0, 0, 1, 2]
    assert int(values[0] >> 7) == -2


def test_round_shift_rejects_int64_overflow() -> None:
    with pytest.raises(OverflowError):
        round_shift(torch.tensor([1 << 62], dtype=torch.int64), shift=-1)


def test_round_shift_keeps_exact_int64_boundaries() -> None:
    minimum = -(1 << 63)
    assert round_shift(torch.tensor([minimum]), 1).tolist() == [minimum // 2]
    assert round_shift(torch.tensor([-(1 << 62)]), -1).tolist() == [minimum]
    assert round_shift(torch.tensor([-1]), -63).tolist() == [minimum]
    assert round_shift(torch.tensor([0]), -63).tolist() == [0]


def test_checked_arithmetic_allows_exact_int64_results() -> None:
    maximum, minimum = (1 << 63) - 1, -(1 << 63)
    assert checked_add(torch.tensor([maximum]), torch.tensor([-maximum])).tolist() == [0]
    assert checked_add(torch.tensor([maximum, -maximum]), torch.tensor([-1, 1])).tolist() == [maximum - 1, -maximum + 1]
    assert checked_mul(torch.tensor([minimum]), torch.tensor([0])).tolist() == [0]
    assert checked_mul(torch.tensor([-(1 << 62)]), torch.tensor([2])).tolist() == [minimum]
    assert dequantize_codes(torch.tensor([-128, 127], dtype=torch.int8), 0).tolist() == [-128.0, 127.0]
    with pytest.raises(OverflowError):
        dequantize_codes(torch.tensor([minimum]), 0)
    with pytest.raises(OverflowError):
        checked_add(torch.tensor([maximum]), torch.tensor([1]))
    with pytest.raises(OverflowError):
        checked_mul(torch.tensor([1 << 62]), torch.tensor([2]))


def test_saturate_int24() -> None:
    values = torch.tensor([-(1 << 24), 0, 1 << 24], dtype=torch.int64)
    assert saturate(values, bits=24).tolist() == [-(1 << 23), 0, (1 << 23) - 1]


def test_profile_entries_reject_unsupported_exponents_and_mixed_runtime() -> None:
    with pytest.raises(ValueError):
        QuantEntry(128)
    with pytest.raises(ValueError):
        QuantEntry(121)
    profile = QuantProfile.from_entries({"x": QuantEntry(0)})
    with pytest.raises(ValueError):
        QuantRuntime(profile, lambda _name, _value, _bits: None)


def test_quantization_contract_rejects_malformed_input() -> None:
    with pytest.raises(ValueError):
        quantize_codes(torch.tensor([0.0]), exponent=128)
    with pytest.raises(ValueError):
        quantize_codes(torch.tensor([0.0]), exponent=0, bits=1)
    with pytest.raises(ValueError):
        QuantEntry(0, zero_point=1)


def test_frozen_runtime_records_clipping_and_required_width() -> None:
    runtime = QuantRuntime.frozen(QuantProfile.from_entries({"x": QuantEntry(0)}))
    runtime.boundary("x", torch.tensor([-200.0, 0.0, 200.0]))
    statistics = runtime.statistics()["x"]
    assert statistics.values == 3
    assert statistics.clipped == 2
    assert statistics.required_signed_bits == 9


def test_profile_rejects_invalid_ssm_scale_relations() -> None:
    with pytest.raises(ValueError, match="Abar"):
        QuantProfile.from_entries({"ssm.Abar": QuantEntry(-6)})
    with pytest.raises(ValueError, match="state"):
        QuantProfile.from_entries({
            "ssm.currentState": QuantEntry(0, bits=24),
            "ssm.state": QuantEntry(6, bits=17),
        })
