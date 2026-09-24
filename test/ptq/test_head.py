import torch
from torch import nn

from models.emamba import EMamba
from models.q_emamba import QEMamba
from ptq.calibrate import ProfileObserver
from ptq.quant import QuantRuntime


def test_head_observes_relu_output_once_at_next_linear() -> None:
    observer = ProfileObserver()
    model = QEMamba(EMamba(), QuantRuntime.profiling(observer))
    frames = torch.randn(2, 8, 8, 5)
    result = model(frames)
    observation = observer.snapshot()["head.activation"]
    assert observation.count == frames.shape[0] * model.d_model
    assert observation.bits == 8
    assert isinstance(model.head.proj[1], nn.ReLU)
    assert result.shape == (2, 57)
