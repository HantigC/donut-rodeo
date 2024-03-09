import numpy as np


def nps_to_rgbs(nps: np.ndarray):
    return [f"rgb({r}, {g}, {b})" for r, g, b in nps]

def nps_to_rgbas(nps: np.ndarray):
    return [f"rgba({r}, {g}, {b}, {a})" for r, g, b, a in nps]
