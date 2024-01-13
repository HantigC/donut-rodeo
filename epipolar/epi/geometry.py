from typing import List, Union
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


def to_homogenous(xs: Union[np.ndarray, List]) -> np.ndarray:
    if isinstance(xs, list):
        xs = np.array(xs)
    if xs.ndim == 2:
        if xs.shape[1] in (2, 3):
            return np.r_["1", xs, np.ones((xs.shape[0], 1))]
        elif xs.shape[0] in (2, 3):
            return np.r_["0", xs, np.ones((1, xs.shape[1]))]
        else:
            raise ValueError("vectors should be 2d or 3d")
    elif xs.ndim == 1:
        return np.r_[xs, 1]


def drop_homogenous(xs: Union[np.ndarray, List]) -> np.ndarray:
    if isinstance(xs, list):
        xs = np.array(xs)
    if xs.ndim == 2:
        if xs.shape[1] in (3, 4):
            return xs[:, :-1]
        elif xs.shape[0] in (3, 4):
            return xs[:-1]
        else:
            raise ValueError("vectors should be 3d or 4d")
    elif xs.ndim == 1:
        return xs[:-1]


def scale_homogenous(xs: Union[np.ndarray, List]) -> np.ndarray:
    if isinstance(xs, list):
        xs = np.array(xs)
    if xs.ndim == 2:
        if xs.shape[1] in (3, 4):
            return xs / xs[:, -1, None]
        elif xs.shape[0] in (3, 4):
            return xs / xs[-1]
        else:
            raise ValueError("vectors should be 3d or 4d")
    elif xs.ndim == 1:
        return xs / xs[-1]


def from_homogenous(xs: Union[np.ndarray, List]) -> np.ndarray:
    if isinstance(xs, list):
        xs = np.array(xs)
    if xs.ndim == 2:
        if xs.shape[1] in (3, 4):
            return xs[:, :-1] / xs[:, -1]
        elif xs.shape[0] in (3, 4):
            return xs[:-1] / xs[-1]
        else:
            raise ValueError("vectors should be 3d or 4d")
    elif xs.ndim == 1:
        return xs[:-1] / xs[-1]


def in_bounds(xs: np.ndarray, mins: List[float], maxs: List[float]):
    mask = np.stack([((x >= l) & (x <= g)) for l, g, x in zip(mins, maxs, xs.T)])
    mask = mask.all(axis=0)
    return mask
