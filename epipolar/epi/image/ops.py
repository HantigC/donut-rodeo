import numpy as np


def to_3channels(img_1chanel: np.ndarray) -> np.ndarray:
    if img_1chanel.ndim == 2:
        img_1chanel = np.expand_dims(img_1chanel, -1)
    img_3channel = np.tile(img_1chanel, (1, 1, 3))
    return img_3channel
