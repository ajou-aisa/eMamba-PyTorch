import torch
from torch import Tensor

INT64_MAX = (1 << 63) - 1


def checked_mul(left: Tensor, right: Tensor) -> Tensor:
    minimum = -(1 << 63)
    left_minimum, right_minimum = left == minimum, right == minimum
    minimum_overflow = (
        (left_minimum & (right != 0) & (right != 1))
        | (right_minimum & (left != 0) & (left != 1))
    )
    if minimum_overflow.any():
        raise OverflowError("integer multiplication would overflow INT64")
    positive_left = torch.where(left > 0, left, 1)
    positive_right = torch.where(right > 0, right, 1)
    negative_right = torch.where(right < 0, right, -1)
    overflow = (
        ((left > 0) & (right > 0) & (left > torch.div(INT64_MAX, positive_right, rounding_mode="trunc")))
        | ((left > 0) & (right < 0) & (right < torch.div(minimum, positive_left, rounding_mode="trunc")))
        | ((left < 0) & (right > 0) & (left < torch.div(minimum, positive_right, rounding_mode="trunc")))
        | ((left < 0) & (right < 0) & (left < torch.div(INT64_MAX, negative_right, rounding_mode="trunc")))
    )
    if overflow.any():
        raise OverflowError("integer multiplication would overflow INT64")
    return left * right


def checked_add(left: Tensor, right: Tensor) -> Tensor:
    minimum = -(1 << 63)
    overflow = (
        ((right > 0) & (left > INT64_MAX - right))
        | ((right < 0) & (left < minimum - right))
    )
    if overflow.any():
        raise OverflowError("integer addition would overflow INT64")
    return left + right
