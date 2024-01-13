from functools import cached_property
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go


@dataclass(init=False)
class Model:
    vertices: np.ndarray
    position: np.ndarray = None
    color: np.ndarray = None
    scale: np.ndarray = field(default=None, init=False)

    def __init__(self, vertices, position=None, color=None) -> None:
        if position is None:
            position = np.array([0, 0, 0])
        self.position = position
        self._vertices = vertices
        self.color = color

    @cached_property
    def vertices(self):
        vertices = self._vertices
        if self.scale is not None:
            vertices *= self.scale
        vertices = vertices + self.position

        return vertices

    @cached_property
    def center(self):
        return np.mean(self.vertices, axis=0)

    def draw(
        self,
        fig: go.Figure,
    ) -> go.Figure:
        vertices = self.vertices

        marker = dict(size=1, symbol="x")
        if self.color is not None:
            marker["color"] = self.color
        fig.add_trace(
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                marker=marker,
            )
        )
        return fig


def read_vertices(filename: str) -> np.ndarray:
    with open(filename) as vertices_file:
        vertices = [
            [float(v) for v in line[3:].split(" ")]
            for line in vertices_file.readlines()
            if line.startswith("v  ")
        ]
    return np.array(vertices)
