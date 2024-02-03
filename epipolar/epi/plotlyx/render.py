from typing import List, Optional, Dict, Any
import copy
from dataclasses import dataclass

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
                self.camera.view @ geom.to_homogenous(model.vertices).T
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
                        eye=np_to_plotly(self.camera.basis.forward),
                    )
                )
            )

        return fig


def render_camera_axes(fig, camera: ProjCamera):
    fig = render_axes(fig, camera.position, camera.forward, camera.right, camera.up)
    return fig


def render_axes(fig, position, forward, right, up):
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
