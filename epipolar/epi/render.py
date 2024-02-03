from typing import Tuple
import numpy as np


def render_image(
    vertices: np.ndarray, image_size: Tuple[int, int], pixel_mm: Tuple[float, float]
) -> np.ndarray:
    if vertices.shape[2] == 3:
        vertices = vertices[..., :-1]

    width, height = image_size
    pixels = vertices / pixel_mm
    pixels = pixels + (width / 2, height / 2)
    return pixels