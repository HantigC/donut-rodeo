import plotly.graph_objects as go


def scatter_3d(xyz, mode="markers", fig=False, **kwargs):
    scatter_plot = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode=mode, **kwargs
    )
    if fig:
        return go.Figure(data=[scatter_plot])

    return scatter_plot
