import plotly.graph_objects as go


def scatter_3d(xyz, mode="markers", **kwargs):
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode=mode, **kwargs)


def np_to_plotly(xyz):
    return dict(x=xyz[0], y=xyz[1], z=xyz[2])
