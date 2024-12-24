import cv2
import numpy as np

from skimage.filters import gaussian
from .helper import (
    _motion_blur,
    shuffle_pixels_njit, 
    clipped_zoom, 
    gen_disk, 
    gen_lensmask, 
)


def blur_gaussian(img, severity=1):
    """
    Gaussian Blur. 
    severity=[1, 2, 3, 4, 5] corresponding to sigma=[1, 2, 3, 4, 5].
    severity mainly refer to KADID-10K and Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [1, 2, 3, 4, 5][severity - 1]
    img = np.array(img) / 255.
    img = gaussian(img, sigma=c, channel_axis=-1)
    img = np.clip(img, 0, 1) * 255
    return img.round().astype(np.uint8)


def blur_gaussian_lensmask(img, severity=1):
    """
    Gaussian Blur with Lens Mask. 
    severity=[1, 2, 3, 4, 5] corresponding to 
    [gamma, sigma]=[[2.0, 2], [2.4, 4], [3.0, 6], [3.8, 8], [5.0, 10]].
    severity mainly refer to PieAPP.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [(2.0, 2), (2.4, 4), (3.0, 6), (3.8, 8), (5.0, 10)][severity - 1]
    img_orig = np.array(img) / 255.
    h, w = img.shape[:2]
    mask = gen_lensmask(h, w, gamma=c[0])[:, :, None]
    img = gaussian(img_orig, sigma=c[1], channel_axis=-1)
    img = mask * img_orig + (1 - mask) * img
    img = np.clip(img, 0, 1) * 255
    return img.round().astype(np.uint8)


def blur_motion(img, severity=1):
    """
    Motion Blur. 
    severity = [1, 2, 3, 4, 5] corresponding to radius=[5, 10, 15, 15, 20] and
    sigma=[1, 2, 3, 4, 5].
    severity mainly refer to Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [0, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [(5, 3), (10, 5), (15, 7), (15, 9), (20, 12)][severity - 1]
    angle = np.random.uniform(-90, 90)
    img = np.array(img)
    img = _motion_blur(img, radius=c[0], sigma=c[1], angle=angle)
    img = np.clip(img, 0, 255)
    return img.round().astype(np.uint8)


def blur_glass(img, severity=1):
    """
    Glass Blur. 
    severity = [1, 2, 3, 4, 5] corresponding to 
    [sigma, shift, iteration]=[(0.7, 1, 1), (0.9, 2, 1), (1.2, 2, 2), (1.4, 3, 2), (1.6, 4, 2)].
    severity mainly refer to Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [0, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [(0.7, 1, 1), (0.9, 2, 1), (1.2, 2, 2), (1.4, 3, 2), (1.6, 4, 2)][severity - 1]
    img = np.array(img) / 255.
    img = gaussian(img, sigma=c[0], channel_axis=-1)
    img = shuffle_pixels_njit(img, shift=c[1], iteration=c[2])
    img = np.clip(gaussian(img, sigma=c[0], channel_axis=-1), 0, 1) * 255
    return img.round().astype(np.uint8)


def blur_lens(img, severity=1):
    """
    Lens Blur. 
    severity = [1, 2, 3, 4, 5] corresponding to radius=[2, 3, 4, 6, 8].
    severity mainly refer to KADID-10K.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [0, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [2, 3, 4, 6, 8][severity - 1]
    img = np.array(img) / 255.
    kernel = gen_disk(radius=c)
    img_lq = []
    for i in range(3):
        img_lq.append(cv2.filter2D(img[:, :, i], -1, kernel))
    img_lq = np.array(img_lq).transpose((1, 2, 0))
    img_lq = np.clip(img_lq, 0, 1) * 255
    return img_lq.round().astype(np.uint8)


def blur_zoom(img, severity=1):
    """
    Zoom Blur. 
    severity = [1, 2, 3, 4, 5] corresponding to radius=
        [np.arange(1, 1.03, 0.02),
         np.arange(1, 1.06, 0.02),
         np.arange(1, 1.10, 0.02),
         np.arange(1, 1.15, 0.02),
         np.arange(1, 1.21, 0.02)].
    severity mainly refer to Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [0, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [np.arange(1, 1.03, 0.02),
         np.arange(1, 1.06, 0.02),
         np.arange(1, 1.10, 0.02),
         np.arange(1, 1.15, 0.02),
         np.arange(1, 1.21, 0.02)][severity - 1]
    img = (np.array(img) / 255.).astype(np.float32)
    h, w = img.shape[:2]
    img_lq = np.zeros_like(img)
    for zoom_factor in c:
        zoom_layer = clipped_zoom(img, zoom_factor)
        img_lq += zoom_layer[:h, :w, :]
    img_lq = (img + img_lq) / (len(c) + 1)
    img_lq = np.clip(img_lq, 0, 1) * 255
    return img_lq.round().astype(np.uint8)


def blur_jitter(img, severity=1):
    """
    Jitter Blur.
    severity = [1, 2, 3, 4, 5] corresponding to shift=[1, 2, 3, 4, 5]. 
    severity mainly refer to KADID-10K.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [0, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [1, 2, 3, 4, 5][severity - 1]
    img = np.array(img)
    img_lq = shuffle_pixels_njit(img, shift=c, iteration=1)
    return np.uint8(img_lq)
