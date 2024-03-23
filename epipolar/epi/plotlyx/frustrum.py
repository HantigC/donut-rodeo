from typing import Tuple
import numpy as np
import plotly.graph_objects as go
from .. import geometry as geom


def render_frustrum(
    fig: go.Figure,
    position,
    rect,
    scale=None,
    name="Pyramid",
    rgb=None,
) -> go.Figure:
    if scale is None:
        scale = 1

    if rgb is None:
        rgb = "rgb(255, 0, 255)"

    rect = scale * rect

    fig.add_traces(
        [
            go.Scatter3d(
                x=[position[0], rect[0, 0]],
                y=[position[1], rect[0, 1]],
                z=[position[2], rect[0, 2]],
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[1, 0]],
                y=[position[1], rect[1, 1]],
                z=[position[2], rect[1, 2]],
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[2, 0]],
                y=[position[1], rect[2, 1]],
                z=[position[2], rect[2, 2]],
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(color=rgb),
            ),
            go.Scatter3d(
                x=[position[0], rect[3, 0]],
                y=[position[1], rect[3, 1]],
                z=[position[2], rect[3, 2]],
                mode="lines",
                name=name,
                showlegend=False,
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
            name=name,
            showlegend=False,
            marker=dict(color=rgb),
        )
    )
    return fig


def render_image_frustrum(
    fig: go.Figure,
    position: np.ndarray,
    left_down: Tuple[int, int],
    right_top: Tuple[int, int],
    K: np.ndarray,
    view: np.ndarray,
    scale: float = 0.1,
    name="Image Pyramid",
    rgb=None,
) -> go.Figure:

    left_down, right_top = (
        np.linalg.inv(K)
        @ np.array(
            [
                [left_down[0], left_down[0], 1],
                [right_top[0], right_top[1], 1],
            ],
        ).T
    ).T
    rect = geom.to_rect(left_down, right_top)
    rect = rect * scale
    rect = geom.from_homogenous(
        np.linalg.inv(view) @ geom.to_homogenous(rect.T)
    ).T
    return render_frustrum(
        fig,
        position,
        rect,
        None,
        name=name,
        rgb=rgb,
    )


def render_wh_frustrum(
    fig: go.Figure,
    position: np.ndarray,
    wh: Tuple[int, int],
    K: np.ndarray,
    view: np.ndarray,
    scale: float = 0.1,
    name="Pyramid",
    rgb=None,
) -> go.Figure:
    w, h = wh
    left_down = [0, 0, 1]
    top_right = [w, h, 1]
    return render_image_frustrum(
        fig,
        position,
        left_down,
        top_right,
        K,
        view,
        scale,
        name=name,
        rgb=rgb,
    )
