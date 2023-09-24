"""Other utilities"""

import random
from datetime import datetime
import numpy as np
import cv2


def random_float(min: float, max: float):
    return random.random() * (max - min) + min


def get_current_date_and_time() -> str:
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    return dt_string


def find_center_of_mass(mask: np.ndarray) -> tuple[int, int]:
    thresh_map = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    moments = cv2.moments(thresh_map)
    try:
        x_center = int(moments["m10"] / moments["m00"])
        y_center = int(moments["m01"] / moments["m00"])
    except ZeroDivisionError:
        x_center = thresh_map.shape[1] // 2
        y_center = thresh_map.shape[0] // 2
    return int(y_center), int(x_center)
