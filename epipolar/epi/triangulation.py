from typing import List, Union, Optional, Callable, Tuple
import numpy as np
from epi import geometry as geom


def to_2d(x: np.ndarray):
    if x.ndim == 1:
        return x[np.newaxis, :]
    elif x.ndim == 2:
        return x

    else:
        raise ValueError(f"`x` should have either 1 or 2 dims, not {x.ndim} dims")


def pairwise_dot(x: np.array, y: np.array, axis: int = 1) -> np.array:
    return (x * y).sum(axis=axis)


def mid_point(
    p: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    return_on_rays: bool = False,
) -> np.ndarray:
    p = to_2d(p)
    r = to_2d(r)
    q = to_2d(q)
    s = to_2d(s)

    ss = pairwise_dot(s, s, axis=1)
    rr = pairwise_dot(r, r, axis=1)
    rs = pairwise_dot(r, s, axis=1)

    qr = pairwise_dot(q, r, axis=1)
    pr = pairwise_dot(p, r, axis=1)
    ps = pairwise_dot(p, s, axis=1)
    qs = pairwise_dot(q, s, axis=1)

    miu = (ss * (qr - pr) + rs * (ps - qs)) / (rr * ss - rs**2)
    sigma = (rr * (ps - qs) + rs * (qr - pr)) / (rr * ss - rs**2)
    P = p + r * sigma[:, np.newaxis]
    Q = q + s * miu[:, np.newaxis]
    H = (P + Q) / 2
    if return_on_rays:
        return H, P, Q
    return H


class _Linear:

    @staticmethod
    def two_cameras(
        points1: np.ndarray,
        points2: np.ndarray,
        cam1: np.ndarray,
        cam2: np.ndarray,
    ) -> np.ndarray:
        triangulated_points = _Linear.multi_cameras([points1, points2], [cam1, cam2])
        return triangulated_points

    @staticmethod
    def multi_cameras(
        points: Union[np.ndarray, List[np.ndarray]],
        cameras: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[np.ndarray, List[np.ndarray]]:

        points = np.stack(points, axis=1)
        cams = np.stack(cameras)

        sistems = points[..., np.newaxis] * cams[:, 2, np.newaxis] - cams[:, :2]
        sistems = sistems.reshape(-1, sistems.shape[-2] * sistems.shape[-3], 4)

        triangulated_points = np.linalg.svd(sistems)
        triangulated_points = triangulated_points.Vh[:, 3, :]
        triangulated_points = geom.from_homogenous(triangulated_points, axis=1)
        return triangulated_points


linear = _Linear


class _NonLinear:

    def _jr(self,p1, p2, cam1, cam2, p3d):
        p1_proj = cam1 @ p3d
        p2_proj = cam2 @ p3d
        r = np.r_[
            p1 - geom.from_homogenous(p1_proj), p2 - geom.from_homogenous(p2_proj)
        ]
        r11, r12, r13 = cam1[:3, :3]
        r21, r22, r23 = cam2[:3, :3]
        J = np.stack(
            [
                r11 * p1_proj[2] - r13 * p1_proj[0],
                r12 * p1_proj[2] - r13 * p1_proj[1],
                r21 * p2_proj[2] - r23 * p2_proj[0],
                r22 * p2_proj[2] - r23 * p2_proj[1],
            ]
        )
        return J, r

    def __call__(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        cam1: np.ndarray,
        cam2: np.ndarray,
        p3d: np.ndarray,
        threshold: float = 0.001,
        max_iterations: int = 10,
        return_error: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        error = np.inf
        idx = 0

        while error > threshold and idx < max_iterations:
            J, residual_vector = self._jr(p1, p2, cam1, cam2, geom.to_homogenous(p3d))
            p3d = p3d - np.linalg.inv((J.T @ J)) @ J.T @ residual_vector
            error = np.sum(residual_vector**2) / 2
            idx += 1

        if return_error:
            return p3d, error
        return p3d

    def multiple_points(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        cam1: np.ndarray,
        cam2: np.ndarray,
        p3ds: np.ndarray,
        threshold: float = 0.001,
        max_iterations: int = 10,
        return_error: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        results = []
        for p1, p2, p3d in zip(pts1, pts2, p3ds):
            result = self(
                p1,
                p2,
                cam1,
                cam2,
                p3d,
                threshold,
                max_iterations,
                return_error,
            )
            results.append(result)

        if return_error:
            p3ds, residuals = list(zip(*results))
            return np.array(p3ds), np.array(residuals)

        return np.array(results)


non_linear = _NonLinear()
