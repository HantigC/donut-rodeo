from typing import ClassVar, Union
from functools import cached_property
from dataclasses import dataclass, field
import numpy as np

from .geometry import (
    vector_from_euler,
    normalize,
    to_homogenous,
    scale_homogenous,
    in_bounds,
)
import plotly.graph_objects as go


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
            [[self.focal_length, 0, 0, 0], [0, self.focal_length, 0, 0], [0, 0, 1, 0]]
        )

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

    def project_vertices(self, vertices: np.ndarray, scale_z=True) -> np.ndarray:
        view_pts = self.view @ to_homogenous(vertices).T
        img_proj = self.K @ view_pts
        if scale_z:
            return scale_homogenous(img_proj.T)
        return img_proj.T

    def render_img(
        self,
        vertices,
        color: Union[np.ndarray, int, float],
        background_color: float = 0,
    ) -> np.ndarray:
        projected_vertices = self.project_vertices(vertices, scale_z=False)
        img_coords = self.render_vertices(vertices).astype(np.int32)
        mask = in_bounds(img_coords, [0, 0], (self.width - 1, self.height - 1))
        img = np.ones((self.height, self.width)) * background_color
        if isinstance(color, np.ndarray):
            color = color[mask]

        depth = np.full(img.shape, np.nan)

        for idx, (x, y) in enumerate(img_coords):
            if not mask[idx]:
                continue
            z = projected_vertices[idx, 2]
            if np.isnan(depth[y, x]) or depth[y, x] > z:
                depth[y, x] = z
            img[y, x] = color

        img = np.flipud(img)
        depth = np.flipud(depth)
        depth = np.where(np.isnan(depth), -1, depth)
        return img, depth

    def render_vertices(self, vertices: np.ndarray) -> np.ndarray:
        projected_vertices = self.project_vertices(vertices)
        projected_vertices = projected_vertices[..., :-1]
        pixels = projected_vertices / (self.xpixel_mm, self.ypixel_mm)
        pixels = pixels + (self.width / 2, self.height / 2)
        return pixels

    def look_at(self, target: np.ndarray) -> None:
        towards = normalize(target - self.position)
        proj_towards = towards.copy()
        proj_towards[self.UP != 1] = 0
        self.pitch = 90 - np.degrees(np.arccos(np.dot(proj_towards, self.UP)))
        proj_towards = towards.copy()
        proj_towards[self.UP == 1] = 0
        proj_towards = normalize(proj_towards)
        self.yaw = np.degrees(np.arccos(np.dot(proj_towards, [1, 0, 0])))

    def draw(self, fig: go.Figure) -> go.Figure:
        fig = render_camera(fig, self.position, self.forward, self.right, self.up)
        return fig


def render_camera(fig, position, forward, right, up):
    fig.add_trace(
        go.Cone(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            u=[forward[0]],
            v=[forward[1]],
            w=[forward[2]],
            showscale=False,
        ),
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + forward[0]],
            y=[position[1], position[1] + forward[1]],
            z=[position[2], position[2] + forward[2]],
            mode="lines",
            marker=dict(
                color=f"rgb(255, 0, 0)",
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + right[0]],
            y=[position[1], position[1] + right[1]],
            z=[position[2], position[2] + right[2]],
            mode="lines",
            marker=dict(
                color=f"rgb(0, 255, 0)",
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + up[0]],
            y=[position[1], position[1] + up[1]],
            z=[position[2], position[2] + up[2]],
            mode="lines",
            marker=dict(
                color=f"rgb(0, 0, 255)",
            ),
        )
    )
    return fig
