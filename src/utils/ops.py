import numpy as np
import cv2


def keep_largest_blob(mask: np.ndarray):
    img = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Find largest contour in intermediate image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)

    # Output
    out_mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(out_mask, [largest_contour], -1, 1, cv2.FILLED)
    out_mask = cv2.bitwise_and(mask.astype(np.uint8), out_mask).astype(np.float32)
    return out_mask
