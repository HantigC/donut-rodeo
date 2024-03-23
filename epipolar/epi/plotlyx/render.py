from typing import List, Optional, Dict, Any, Tuple
import copy
from dataclasses import dataclass
import numpy as np

import plotly.graph_objects as go
from .utils.fig import np_to_plotly
from .utils.color import make_random_rgbs
from ..camera import ProjCamera
from ..model import Model
from .. import geometry as geom
from . import express as pxx
from .frustrum import render_frustrum, render_wh_frustrum


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
            fig = render_camera_basis_axes(fig, self.camera)

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


def render_camera_axes(
    fig: go.Figure,
    camera: ProjCamera,
    scale=0.1,
    name="Camera",
    rgb=None,
) -> go.Figure:
    if scale is not None:
        if rgb is None:
            rgb = "rgb(255, 0, 255)"

        render_wh_frustrum(
            fig,
            camera.position,
            (camera.width, camera.height),
            camera.K,
            camera.view,
            scale,
            name=name,
            rgb=rgb,
        )

    fig = render_axes(
        fig,
        camera.position,
        camera.forward,
        camera.right,
        camera.up,
        scale,
        name=name,
    )

    return fig


def render_camera_basis_axes(
    fig: go.Figure,
    camera: ProjCamera,
    scale=0.1,
    rgb=None,
) -> go.Figure:

    if scale is not None:
        if rgb is None:
            rgb = "rgb(255, 0, 255)"

        render_wh_frustrum(
            fig,
            camera.basis.center,
            (camera.width, camera.height),
            camera.K,
            np.eye(4),
            scale,
            rgb,
        )

    fig = render_axes(
        fig,
        camera.basis.center,
        camera.basis.forward,
        camera.basis.right,
        camera.basis.up,
        scale,
    )

    return fig


def render_od_rays(
    fig: go.Figure,
    origin: np.ndarray,
    directions: np.ndarray,
    scale: float = 1,
    rgbs: List[str] = None,
) -> go.Figure:
    if rgbs is None:
        rgbs = make_random_rgbs(len(directions))

    for (x, y, z), rgb in zip(directions, rgbs):
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], origin[0] + scale * x],
                y=[origin[1], origin[1] + scale * y],
                z=[origin[2], origin[2] + scale * z],
                mode="lines",
                marker=dict(color=rgb),
            ),
        )
    return fig


def render_diff_rays(
    fig: go.Figure,
    origin: np.ndarray,
    points: np.ndarray,
    scale: float = 1,
    rgbs: List[str] = None,
) -> go.Figure:
    directions = geom.normalize(points - origin)
    return render_od_rays(
        fig,
        origin,
        directions,
        scale,
        rgbs,
    )


def render_axes(
    fig,
    position,
    forward,
    right,
    up,
    scale=None,
    name=None,
):
    if scale == None:
        scale = 1

    forward_name = "forward"
    up_name = "up"
    right_name = "right"
    if name is not None:
        forward_name = " ".join([name, forward_name])
        right_name = " ".join([name, right_name])
        up_name = " ".join([name, up_name])

    fig.add_trace(
        go.Scatter3d(
            x=[position[0], position[0] + scale * forward[0]],
            y=[position[1], position[1] + scale * forward[1]],
            z=[position[2], position[2] + scale * forward[2]],
            mode="lines",
            name=forward_name,
            showlegend=False,
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
            name=right_name,
            showlegend=False,
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
            name=up_name,
            showlegend=False,
            marker=dict(
                color=f"rgb(0, 0, 255)",
            ),
        )
    )
    return fig
