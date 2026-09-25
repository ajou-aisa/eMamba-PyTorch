from collections.abc import Mapping

from torch import Tensor

from models.emamba import EMamba
from models.q_emamba import QEMamba

from .components import prepare_components as prepare_components
from .quant import QuantRuntime


def prepare_model(
    source: EMamba, runtime: QuantRuntime, *,
    continuous_a: Mapping[str, Tensor] | None = None,
) -> QEMamba:
    return QEMamba(source, runtime, continuous_a=continuous_a)
