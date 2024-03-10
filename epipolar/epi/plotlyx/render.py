from typing import List, Optional, Dict, Any, Tuple
import copy
from dataclasses import dataclass
import numpy as np

import plotly.graph_objects as go
from .utils.fig import np_to_plotly
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
            rgb,
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
