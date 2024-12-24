from .blur import *
from .brightness import *
from .quantization import *
from .compression import *
from .contrast import *
from .noise import *
from .oversharpen import *
from .pixelate import *
from .saturate import *


def add_distortion(img, severity=1, distortion_name=None):
    """This function returns a distorted version of the given image.

    @param img (np.ndarray, unit8): Input image, H x W x 3, RGB, [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @distortion_name: 
    @return: Degraded image (np.ndarray, unit8), H x W x 3, RGB, [0, 255]
    """

    if not isinstance(img, np.ndarray):
        raise AttributeError('Expecting type(img) to be numpy.ndarray')
    if not (img.dtype.type is np.uint8):
        raise AttributeError('Expecting img.dtype.type to be numpy.uint8')

    if not (img.ndim in [2, 3]):
        raise AttributeError('Expecting img.shape to be either (h x w) or (h x w x c)')
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)

    h, w, c = img.shape
    if (h < 32 or w < 32):
        raise AttributeError('The (w, h) must be at least 32 pixels')
    if not (c in [1, 3]):
        raise AttributeError('Expecting img to have either 1 or 3 chennels')
    if c == 1:
        img = np.stack((np.squeeze(img),) * 3, axis=-1)

    if severity not in [1, 2, 3, 4, 5]:
        raise AttributeError('The severity must be an integer in [1, 5]')

    if distortion_name:
        img_lq = globals()[distortion_name](img, severity)
    else:
        raise ValueError("The distortion_name must be passed")

    return np.uint8(img_lq)


distortions_dict = {
    "blur": [
        "blur_gaussian", 
        "blur_motion", 
        "blur_glass", 
        "blur_lens", 
        "blur_zoom", 
        "blur_jitter", 
    ],
    "noise": [
        "noise_gaussian_RGB", 
        "noise_gaussian_YCrCb", 
        "noise_speckle", 
        "noise_spatially_correlated", 
        "noise_poisson", 
        "noise_impulse", 
    ], 
    "compression": [
        "compression_jpeg", 
        "compression_jpeg_2000", 
    ], 
    "brighten": [
        "brightness_brighten_shfit_HSV", 
        "brightness_brighten_shfit_RGB", 
        "brightness_brighten_gamma_HSV", 
        "brightness_brighten_gamma_RGB", 
    ], 
    "darken": [
        "brightness_darken_shfit_HSV", 
        "brightness_darken_shfit_RGB", 
        "brightness_darken_gamma_HSV", 
        "brightness_darken_gamma_RGB", 
    ], 
    "contrast_strengthen": [
        "contrast_strengthen_scale",
        "contrast_strengthen_stretch", 
    ],
    "contrast_weaken": [
        "contrast_weaken_scale",
        "contrast_weaken_stretch",
    ],
    "saturate_strengthen": [
        "saturate_strengthen_HSV", 
        "saturate_strengthen_YCrCb", 
    ], 
    "saturate_weaken": [
        "saturate_weaken_HSV", 
        "saturate_weaken_YCrCb", 
    ], 
    "oversharpen": [
        "oversharpen", 
    ], 
    "pixelate": [
        "pixelate", 
    ], 
    "quantization": [
        "quantization_otsu", 
        "quantization_median", 
        "quantization_hist", 
    ], 
    "spatter": [
        "spatter", 
    ], 
}


def get_distortion_names(subset=None):
    if subset in distortions_dict:
        print(distortions_dict[subset])
    else:
        print(distortions_dict)
