import numpy as np

from PIL import Image


def pixelate(img, severity=1):
    """
    Pixelate. 
    severity=[1, 2, 3, 4, 5] corresponding to sigma=[0.5, 0.4, 0.3, 0.25, 0.2].
    severity mainly refer to Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [0.5, 0.4, 0.3, 0.25, 0.2][severity - 1]
    h, w = np.array(img).shape[:2]
    img = Image.fromarray(img)
    img = img.resize((int(w * c), int(h * c)), Image.BOX)
    img = img.resize((w, h), Image.NEAREST)
    return np.array(img).astype(np.uint8)
