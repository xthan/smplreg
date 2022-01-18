import numpy as np


def rel_change(prev_val: float, curr_val: float) -> float:
    """
    Compute relative change. Code from https://github.com/vchoutas/smplify-x
    Args:
        prev_val (float): Previous value
        curr_val (float): Current value
    Returns:
        float: Relative change
    """
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])
