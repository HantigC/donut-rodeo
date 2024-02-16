from typing import ClassVar, Union, List, NamedTuple
from functools import cached_property
from dataclasses import dataclass, field
import numpy as np

from .geometry import (
    vector_from_euler,
    normalize,
    to_homogenous,
    scale_homogenous,
    drop_homogenous,
    from_homogenous,
    in_bounds,
    add_col,
)
from .model import Model
import plotly.graph_objects as go


class CoordinateSystem(NamedTuple):
    center: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    right: np.ndarray


@dataclass(init=False)
class ProjCamera:
    UP: ClassVar[np.ndarray] = np.array([0, 1, 0])

    position: np.ndarray
    yaw: float = 0
    pitch: float = 0
    focal_length: float
    width: int
    height: int
    xpixel_mm: float
    ypixel_mm: float

    forward: np.ndarray = field(init=False)
    right: np.ndarray = field(init=False)
    up: np.ndarray = field(init=False)

    def __init__(
        self,
        position,
        focal_length,
        width,
        height,
        yaw=0,
        pitch=0,
        xpixel_mm: float = 1,
        ypixel_mm: float = 1,
    ) -> None:
        self.position = position
        self.focal_length = focal_length
        self.f_x = self.focal_length / xpixel_mm
        self.f_y = self.focal_length / ypixel_mm
        self.width = width
        self.height = height
        self._yaw = yaw
        self._pitch = pitch
        self._update_vector(self._yaw, self._pitch)
        self.xpixel_mm = xpixel_mm
        self.ypixel_mm = ypixel_mm

    def _update_vector(self, yaw, pitch):
        self.forward = vector_from_euler(yaw, pitch)
        self.right = normalize(np.cross(self.forward, self.UP))
        self.up = normalize(np.cross(self.right, self.forward))

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = value
        self._update_vector(self._yaw, self._pitch)

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value
        self._update_vector(self._yaw, self._pitch)

    @cached_property
    def K(self):
        return np.array(
            [
                [self.f_x, 0, self.width // 2],
                [0, self.f_y, self.height // 2],
                [0, 0, 1],
            ]
        )

    @cached_property
    def Kinv(self):
        return np.linalg.inv(self.K)

    @cached_property
    def viewinv(self):
        return np.linalg.inv(self.view)

    @cached_property
    def view(self):
        view = np.array(
            [
                [
                    self.right[0],
                    self.right[1],
                    self.right[2],
                    np.dot(self.right, -self.position),
                ],
                [self.up[0], self.up[1], self.up[2], np.dot(self.up, -self.position)],
                [
                    self.forward[0],
                    self.forward[1],
                    self.forward[2],
                    np.dot(self.forward, -self.position),
                ],
                [0, 0, 0, 1],
            ]
        )
        return view

    @cached_property
    def rotation(self):
        rotation = np.array(
            [
                [
                    self.right[0],
                    self.right[1],
                    self.right[2],
                    0,
                ],
                [self.up[0], self.up[1], self.up[2], 0],
                [
                    self.forward[0],
                    self.forward[1],
                    self.forward[2],
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )
        return rotation

    @cached_property
    def translation(self):
        view = np.array(
            [
                [
                    0,
                    0,
                    0,
                    np.dot(self.right, -self.position),
                ],
                [0, 0, 0, np.dot(self.up, -self.position)],
                [
                    0,
                    0,
                    0,
                    np.dot(self.forward, -self.position),
                ],
                [0, 0, 0, 1],
            ]
        )
        return view

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    @property
    def fx(self):
        return self.forward[0]

    @property
    def fy(self):
        return self.forward[1]

    @property
    def fz(self):
        return self.forward[2]

    @property
    def rx(self):
        return self.right[0]

    @property
    def ry(self):
        return self.right[1]

    @property
    def rz(self):
        return self.right[2]

    @property
    def ux(self):
        return self.up[0]

    @property
    def uy(self):
        return self.up[1]

    @property
    def uz(self):
        return self.up[2]

    def project_vertices(
        self, vertices: np.ndarray, scale_z=True, drop_last=False
    ) -> np.ndarray:
        view_pts = self.view @ to_homogenous(vertices, axis=1).T
        K = add_col(self.K, 0)
        img_proj = K @ view_pts

        if scale_z:
            if drop_last:
                return from_homogenous(img_proj, axis=0).T
            return scale_homogenous(img_proj, axis=0).T

        return img_proj.T

    def render_img(
        self,
        vertices,
        color: Union[np.ndarray, int, float],
        background_color: float = 0,
    ) -> np.ndarray:
        img_coords = self.project_vertices(vertices).astype(np.int32)
        cam_vertices = from_homogenous(
            self.view @ to_homogenous(vertices, axis=1).T, axis=0
        ).T
        mask = in_bounds(img_coords, [0, 0], (self.width - 1, self.height - 1))
        img = np.ones((self.height, self.width)) * background_color
        if isinstance(color, np.ndarray):
            color = color[mask]

        depth = np.full(img.shape, np.nan)
        world_idx = np.full(img.shape, np.nan)

        for idx, (x, y, _) in enumerate(img_coords):
            if not mask[idx]:
                continue
            z = cam_vertices[idx, 2]
            if np.isnan(depth[y, x]) or depth[y, x] > z:
                depth[y, x] = z
                world_idx[y, x] = idx
            img[y, x] = color

        depth = np.where(np.isnan(depth), -1, depth)

        img = np.flipud(img)
        depth = np.flipud(depth)
        world_idx = np.flipud(world_idx)
        return img, depth, world_idx

    def look_at(self, target: np.ndarray) -> None:
        towards = normalize(target - self.position)
        proj_towards = towards.copy()
        proj_towards[self.UP != 1] = 0
        self.pitch = 90 - np.degrees(np.arccos(np.dot(proj_towards, self.UP)))
        proj_towards = towards.copy()
        proj_towards[self.UP == 1] = 0
        proj_towards = normalize(proj_towards)
        self.yaw = np.degrees(np.arccos(np.dot(proj_towards, [1, 0, 0])))

    @cached_property
    def basis(self):
        position = drop_homogenous(self.view @ to_homogenous(self.position))
        forward = drop_homogenous(self.rotation @ to_homogenous(self.forward))
        right = drop_homogenous(self.rotation @ to_homogenous(self.right))
        up = drop_homogenous(self.rotation @ to_homogenous(self.up))
        return CoordinateSystem(position, forward, up, right)
