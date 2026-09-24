from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset

import ptq.calibrate as calibration
from ptq.quant import QuantRuntime


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[Tensor] = []

    def forward(self, value: Tensor) -> Tensor:
        self.seen.extend(value.detach().clone())
        return value


def test_indices_are_seeded_ordered_and_hash_order_sensitive() -> None:
    # Given: a ten-frame training partition and a dedicated seed.
    expected = tuple(torch.randperm(10, generator=torch.Generator().manual_seed(0))[:4].tolist())
    # When: calibration identities are selected repeatedly.
    first = calibration.select_indices(10, count=4, seed=0)
    second = calibration.select_indices(10, count=4, seed=0)
    # Then: order and its identity are exact and reproducible.
    assert first == second == expected
    assert calibration.index_hash(first) != calibration.index_hash(tuple(reversed(first)))


def test_observer_rejects_empty_and_nonfinite_values() -> None:
    # Given: a fresh observer.
    observer = calibration.ProfileObserver(sample_limit=8)
    # When/Then: invalid boundaries fail at collection time.
    with pytest.raises(calibration.CalibrationError):
        observer("empty", torch.empty(0), 8)
    with pytest.raises(calibration.CalibrationError):
        observer("nan", torch.tensor([math.nan]), 8)
    with pytest.raises(calibration.CalibrationError):
        calibration.candidate_profiles({})
    observer("huge", torch.tensor([2.0**200], dtype=torch.float64), 8)
    with pytest.raises(calibration.CalibrationError):
        calibration.candidate_profiles(observer.snapshot())
    observer("bounded", torch.arange(20.0), 8)
    bounded = observer.snapshot()["bounded"]
    assert bounded.count == 20 and bounded.sample.tolist() == [0, 2, 5, 8, 10, 13, 16, 19]


def test_candidates_use_max_for_parameters_and_state_rules() -> None:
    # Given: activation, parameter, Abar, and FP32 state observations.
    observer = calibration.ProfileObserver(sample_limit=65_536)
    observer("layer.output", torch.cat((torch.zeros(1000), torch.tensor([200.0]))), 8)
    observer("layer.weight", torch.tensor([200.0]), 8)
    observer("block.Abar", torch.tensor([1.0]), 8)
    observer("block.currentState", torch.tensor([65_536.0]), 24)
    observer("block.state", torch.tensor([1e12]), 17)
    # When: max and linear 99.9-percentile candidates are frozen.
    candidates = calibration.candidate_profiles(observer.snapshot())
    # Then: only activations differ; state and paper-fixed Abar obey their scales.
    assert candidates["max"].entries["layer.output"].exponent == 1
    assert candidates["percentile"].entries["layer.output"].exponent == 0
    assert candidates["percentile"].entries["layer.weight"].exponent == 1
    assert candidates["max"].entries["block.Abar"].exponent == -7
    assert candidates["max"].entries["block.currentState"].exponent == -6
    assert candidates["max"].entries["block.state"].exponent == 1
    runtime = QuantRuntime.frozen(candidates["max"])
    before = calibration.profile_hash(candidates["max"])
    runtime.boundary("layer.output", torch.tensor([1e9]))
    assert calibration.profile_hash(candidates["max"]) == before


def test_collection_uses_only_features_in_selected_order() -> None:
    # Given: labels that differ from features and non-sorted train indices.
    model = RecordingModel()
    dataset = TensorDataset(torch.arange(6.0).unsqueeze(1), torch.arange(100.0, 106.0))
    # When: the calibration pass runs.
    calibration.collect_calibration(model, dataset, (4, 1, 5), batch_size=2)
    # Then: only ordered features reached the model.
    assert [item.item() for item in model.seen] == [4.0, 1.0, 5.0]
