from typing import Tuple
import numpy as np
from . import geometry as geom


def to_origin_direction(
    origin: np.ndarray,
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    first_origin = np.broadcast_to(origin, points.shape)
    first_directions = geom.normalize(points - first_origin)
    return first_origin, first_directions
