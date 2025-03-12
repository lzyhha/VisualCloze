import numpy as np
import cv2
from .helper import gen_lensmask


def brightness_brighten_shfit_HSV(img, severity=1):
    """
    The RGB image is mapping to HSV, and then enhance the brightness by V channel
    severity=[1,2,3,4,5] is corresponding to c=[0.1, 0.2, 0.3, 0.4, 0.5]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    img = np.float32(np.array(img) / 255.)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] += c
    img_lq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def brightness_brighten_shfit_RGB(img, severity=1):
    """
    The RGB image is directly enhanced by RGB mean shift
    severity=[1,2,3,4,5] is corresponding to c=[0.1, 0.15, 0.2, 0.27, 0.35]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    c = [0.1, 0.15, 0.2, 0.27, 0.35][severity-1]
    img = np.float32(np.array(img) / 255.)
    img_lq = img + c
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def brightness_brighten_gamma_RGB(img, severity=1):
    """
    The RGB image is enhanced by V channel with a gamma function
    severity=[1,2,3,4,5] is corresponding to gamma=[0.8, 0.7, 0.6, 0.45, 0.3]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    gamma = [0.8, 0.7, 0.6, 0.45, 0.3][severity-1]
    img = np.array(img / 255.)
    img_lq = img ** gamma
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def brightness_brighten_gamma_HSV(img, severity=1):
    """
    The RGB image is enhanced by V channel with a gamma function
    severity=[1,2,3,4,5] is corresponding to gamma=[0.7, 0.55, 0.4, 0.25, 0.1]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    gamma = [0.7, 0.58, 0.47, 0.36, 0.25][severity-1]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv = np.array(img_hsv / 255.)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] ** gamma
    img_lq = np.uint8(np.clip(img_hsv, 0, 1) * 255.)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_HSV2RGB)
    return img_lq


def brightness_darken_shfit_HSV(img, severity=1):
    """
    The RGB image is mapping to HSV, and then darken the brightness by V channel
    severity=[1,2,3,4,5] is corresponding to c=[0.1, 0.2, 0.3, 0.4, 0.5]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    img = np.float32(np.array(img) / 255.)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] -= c
    img_lq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def brightness_darken_shfit_RGB(img, severity=1):
    """
    The RGB image's brightness is directly reduced by RGB mean shift
    severity=[1,2,3,4,5] is corresponding to c=[0.1, 0.15, 0.2, 0.27, 0.35]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    c = [0.1, 0.15, 0.2, 0.27, 0.35][severity-1]
    img = np.float32(np.array(img)/255.)
    img_lq = img - c
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def brightness_darken_gamma_RGB(img, severity=1):
    """
    The RGB image is darkened by V channel with a gamma function
    severity=[1,2,3,4,5] is corresponding to gamma=[1.4, 1.7, 2.1, 2.6, 3.2]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    gamma = [1.4, 1.7, 2.1, 2.6, 3.2][severity-1]
    img = np.array(img / 255.)
    img_lq = img ** gamma
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def brightness_darken_gamma_HSV(img, severity=1):
    """
    The RGB image is enhanced by V channel with a gamma function
    severity=[1,2,3,4,5] is corresponding to gamma=[1.5, 1.8, 2.2, 2.7, 3.5]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    gamma = [1.5, 1.8, 2.2, 2.7, 3.5][severity-1]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv = np.array(img_hsv / 255.)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] ** gamma
    img_lq = np.uint8(np.clip(img_hsv, 0, 1) * 255.)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_HSV2RGB)
    return img_lq


def brightness_vignette(img, severity=1):
    """
    The RGB image is suffered from the vignette effect.
    severity=[1,2,3,4,5] is corresponding to gamma=[0.5, 0.875, 1.25, 1.625, 2]

    @param img: Input image, H x W x RGB, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x RGB, value range [0, 255]
    """
    gamma = [0.5, 0.875, 1.25, 1.625, 2][severity - 1]
    img = np.array(img)
    h, w = img.shape[:2]
    mask = gen_lensmask(h, w, gamma=gamma)[:, :, None]
    img_lq = mask * img
    return np.uint8(np.clip(img_lq, 0, 255))
