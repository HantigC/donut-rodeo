import numpy as np


def make_random_rgbas(n: int) -> np.ndarray:
    return nps_to_rgbas(np.random.uniform(0, 255, (n, 5)))

def make_random_rgbs(n: int) -> np.ndarray:
    return nps_to_rgbs(np.random.uniform(0, 255, (n, 3)))


def nps_to_rgbs(nps: np.ndarray):
    return [f"rgb({r}, {g}, {b})" for r, g, b in nps]

def nps_to_rgbas(nps: np.ndarray):
    return [f"rgba({r}, {g}, {b}, {a})" for r, g, b, a in nps]
