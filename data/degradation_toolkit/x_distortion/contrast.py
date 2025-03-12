import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance


def contrast_weaken_scale(img, severity=1):
    """
    Contrast Weaken by scaling. 
    severity=[1, 2, 3, 4, 5] corresponding to scale=[0.75, 0.6, 0.45, 0.3, 0.2].

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [0.75, 0.6, 0.45, 0.3, 0.2][severity - 1]
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(c)
    img = np.uint8(np.clip(np.array(img), 0, 255))
    return img


def contrast_weaken_stretch(img, severity=1):
    """
    Contrast Weaken by stretching. 
    severity=[1, 2, 3, 4, 5] corresponding to scale=[1.0, 0.9, 0.8, 0.6, 0.4].
    severity mainly refer to PieAPP.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [1.0, 0.9, 0.8, 0.6, 0.4][severity - 1]
    img = np.array(img) / 255.
    img_mean = np.mean(img, axis=(0,1), keepdims=True)
    img = 1. / (1 + (img_mean / (img + 1e-12)) ** c)
    img = np.uint8(np.clip(img, 0, 1) * 255)
    return img


def contrast_strengthen_scale(img, severity=1):
    """
    Contrast Strengthen by scaling. 
    severity=[1, 2, 3, 4, 5] corresponding to scale=[1.4, 1.7, 2.1, 2.6, 4.0].

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [1.4, 1.7, 2.1, 2.6, 4.0][severity - 1]
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(c)
    img = np.uint8(np.clip(np.array(img), 0, 255))
    return img


def contrast_strengthen_stretch(img, severity=1):
    """
    Contrast Strengthen by stretching. 
    severity=[1, 2, 3, 4, 5] corresponding to scale=[2.0, 4.0, 6.0, 8.0, 10.0].
    severity mainly refer to PieAPP.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [2.0, 4.0, 6.0, 8.0, 10.0][severity - 1]
    img = np.array(img) / 255.
    img_mean = np.mean(img, axis=(0,1), keepdims=True)
    img = 1. / (1 + (img_mean / (img + 1e-12)) ** c)
    img = np.uint8(np.clip(img, 0, 1) * 255)
    return img
