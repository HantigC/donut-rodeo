import numpy as np


def min_max(x: np.ndarray, axis=(0, 1)) -> np.ndarray:
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)
