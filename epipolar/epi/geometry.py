from abc import ABC, abstractmethod
from typing import List, Union
from functools import wraps
import numpy as np
import math


def normalize(vec):
    return vec / np.linalg.norm(vec)


def vector_from_euler(yaw, pitch):
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    x = math.cos(yaw_rad) * math.cos(pitch_rad)
    z = math.sin(yaw_rad) * math.cos(pitch_rad)
    y = math.sin(pitch_rad)
    return np.array([x, y, z])


class _homo_dim_check(ABC):

    def call(self, xs: Union[np.ndarray, List], axis: int = 0) -> np.ndarray:

        if isinstance(xs, list):
            xs = np.array(xs)

        if xs.ndim == 2:
            if axis == 0:
                return self.on_two_zero(xs)
            elif axis == 1:
                return self.on_two_one(xs)
            else:
                raise ValueError(
                    f"When `xs` is two-dimentional, `axis` should 0 or 1, not {axis}"
                )
        elif xs.ndim == 1:
            if axis != 0:
                raise ValueError(
                    f"When `xs` is one-dimentional, `axis` should 0, not {axis}"
                )
            return self.on_zero(xs)

        raise ValueError(f"")

    @abstractmethod
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class _drop_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return xs[:, :-1]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1]


class _scale_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return xs / xs[:, -1, None]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs / xs[-1]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs / xs[:-1]


class _from_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return xs[:, :-1] / xs[:, -1, None]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1] / xs[-1]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return xs[:-1] / xs[-1]


class _to_homogenous(_homo_dim_check):
    def on_two_one(self, xs: np.ndarray) -> np.ndarray:
        return np.r_["1", xs, np.ones((xs.shape[0], 1))]

    def on_two_zero(self, xs: np.ndarray) -> np.ndarray:
        return np.r_["0", xs, np.ones((1, xs.shape[1]))]

    def on_zero(self, xs: np.ndarray) -> np.ndarray:
        return np.r_[xs, 1]


drop_homogenous = _drop_homogenous().call
scale_homogenous = _scale_homogenous().call
from_homogenous = _from_homogenous().call
to_homogenous = _to_homogenous().call


def in_bounds(xs: np.ndarray, mins: List[float], maxs: List[float]):
    mask = np.stack([((x >= l) & (x <= g)) for l, g, x in zip(mins, maxs, xs.T)])
    mask = mask.all(axis=0)
    return mask


def add_col(array: np.array, x):
    xs = np.full((array.shape[0], 1), x)
    return np.r_["1", array, xs]


def add_row(array: np.array, x):
    xs = np.full((1, array.shape[1]), x)
    return np.r_["0", array, xs]
