from typing import List, Optional, Dict, Any
import copy
from dataclasses import dataclass
import numpy as np

import plotly.graph_objects as go
from .utils.fig import np_to_plotly
from ..camera import ProjCamera
from ..model import Model
from .. import geometry as geom
from . import express as pxx


@dataclass(init=False)
class CameraCoordinateRenderer:
    camera: ProjCamera
    fig: Optional[go.Figure]
    show_camera: bool = True
    layout: Dict[str, Any] = None

    def __init__(
        self,
        camera: ProjCamera,
        fig: Optional[go.Figure] = None,
        show_camera: bool = True,
        layout: Dict[str, Any] = None,
    ) -> None:
        self.camera = camera
        self.show_camera = show_camera
        if fig is None:
            if layout is not None:
                fig = go.Figure(layout=layout)
        else:
            fig = copy.deepcopy(fig)
            if layout is not None:
                fig.update_layout(**layout)
        self.fig = fig

    def render(
        self,
        models: List[Model],
        fig: Optional[go.Figure] = None,
        use_viewpoint=True,
    ) -> go.Figure:
        if fig is None:
            if self.fig is None:
                raise ValueError(
                    "`fig` is missing. Provide it via object initialization, pass it as an attribute"
                )
            fig = self.fig

        if not isinstance(models, list):
            if not isinstance(models, Model):
                raise ValueError(
                    f"`models` should have Model or List[Model] type, not {models.__class__.__name__}"
                )
            models = [models]
        for model in models:
            vertices = geom.drop_homogenous(
                self.camera.view @ geom.to_homogenous(model.vertices.T, axis=0),
                axis=0,
            ).T
            marker_dict = dict(symbol="x", size=1)
            if model.color is not None:
                marker_dict["color"] = model.color

            fig.add_trace(pxx.scatter_3d(vertices, marker=marker_dict))
        if self.show_camera:
            fig = render_axes(
                fig,
                self.camera.basis.center,
                self.camera.basis.forward,
                self.camera.basis.right,
                self.camera.basis.up,
            )

        if use_viewpoint:
            fig.update_layout(
                scene=dict(
                    camera=dict(
                        up=np_to_plotly(self.camera.basis.up),
                        center=np_to_plotly(self.camera.basis.center),
                        eye=np_to_plotly(-self.camera.gaze),
                    )
                )
            )

        return fig


def to_rect(left_down, right_top):
    ld_x, ld_y, ld_z = left_down
    rt_x, rt_y, rt_z = right_top
    rect_3d = np.array(
        [
            left_down,
            [ld_x, rt_y, ld_z],
            right_top,
            [rt_x, ld_y, rt_z],
        ]
    )
    return rect_3d


def render_frustrum(fig, position, left_down, right_top, scale=None, rgb=None):
    if scale is None:
        scale = 1

    if rgb is None:
        rgb = "rgb(255, 0, 255)"

    rect = scale * to_rect(left_down, right_top)

    fig.add_traces(
        [
            go.Scatter3d(
                x=[position[0], rect[0, 0]],
                y=[position[1], rect[0, 1]],
                z=[position[2], rect[0, 2]],
                mode="lines",
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[1, 0]],
                y=[position[1], rect[1, 1]],
                z=[position[2], rect[1, 2]],
                mode="lines",
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[2, 0]],
                y=[position[1], rect[2, 1]],
                z=[position[2], rect[2, 2]],
                mode="lines",
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[3, 0]],
                y=[position[1], rect[3, 1]],
                z=[position[2], rect[3, 2]],
                mode="lines",
                marker=dict(color=rgb),
            ),
        ]
    )
    rect = np.concatenate(
        [rect, np.expand_dims(rect[0], axis=0)],
        axis=0,
    )

    fig.add_trace(
        go.Scatter3d(
            x=rect[:, 0],
            y=rect[:, 1],
            z=rect[:, 2],
            mode="lines",
            marker=dict(color=rgb),
        )
    )
    return fig


def render_camera_axes(fig, camera: ProjCamera, scale=0.1, rgb=None):
    if scale is not None:
        if rgb is None:
            rgb = "rgb(255, 0, 255)"

        ldtr = (
            camera.Kinv
            @ np.array(
                [
                    [0, 0, 1],
                    [camera.width, camera.height, 1],
                ],
            ).T
        )
        ldtr = ldtr * scale

        left_down, right_top = geom.from_homogenous(
            camera.viewinv @ geom.to_homogenous(ldtr)
        ).T
        render_frustrum(
            fig,
            camera.position,
            left_down,
            right_top,
            rgb=rgb,
        )

    fig = render_axes(
        fig,
        camera.position,
        camera.forward,
        camera.right,
        camera.up,
        scale,
    )

    return fig


def render_axes(fig, position, forward, right, up, scale=None):
    if scale == None:
        scale = 1

    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * forward[0]],
            y=[position[1], position[1] + scale * forward[1]],
            z=[position[2], position[2] + scale * forward[2]],
            mode="lines",
            marker=dict(
                color=f"rgb(255, 0, 0)",
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * right[0]],
            y=[position[1], position[1] + scale * right[1]],
            z=[position[2], position[2] + scale * right[2]],
            mode="lines",
            marker=dict(
                color=f"rgb(0, 255, 0)",
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * up[0]],
            y=[position[1], position[1] + scale * up[1]],
            z=[position[2], position[2] + scale * up[2]],
            mode="lines",
            marker=dict(
                color=f"rgb(0, 0, 255)",
            ),
        )
    )
    return fig
