import numpy as np

from PIL import Image
from skimage.filters import threshold_multiotsu



def quantization_otsu(img, severity=1):
    """
    Color Quantization using OTSU method.
    severity=[1, 2, 3, 4, 5] corresponding to num_classes=[15, 11, 8, 5, 3].
    severity mainly refer to KADID-10K and Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [15, 11, 8, 5, 3][severity - 1]
    img = np.array(img).astype(np.float32)
    for i in range(img.shape[2]):
        img_gray = img[:, :, i]
        thresholds = threshold_multiotsu(img_gray, classes=c, nbins=30) # modify skimage
        v_max = img_gray.max()
        v_min = img_gray.min()
        img[:, :, i] = np.digitize(img[:, :, i], bins=thresholds) * (v_max - v_min) / c + v_min
    img = np.clip(img, 0, 255)
    return img


def quantization_median(img, severity=1):
    """
    Color Quantization using Histogram Median.
    severity=[1, 2, 3, 4, 5] corresponding to num_classes=[20, 15, 10, 6, 3].
    severity mainly refer to KADID-10K and Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [20, 15, 10, 6, 3][severity - 1]
    for i in range(img.shape[2]):
        img_gray = Image.fromarray(img[:, :, i])
        img_gray = img_gray.quantize(colors=c, method=Image.Quantize.MEDIANCUT).convert("L")
        img[:, :, i] = np.array(img_gray)
    img = np.clip(img, 0, 255)
    return img


def quantization_hist(img, severity=1):
    """
    Color Quantization using Histogram Equalization.
    severity=[1, 2, 3, 4, 5] corresponding to num_classes=[24, 16, 8, 6, 4].
    severity mainly refer to KADID-10K and Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [24, 16, 8, 6, 4][severity - 1]
    hist, _ = np.histogram(img.flatten(), bins=c, range=[0, 255])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img = np.uint8(np.round(img / 255 * (c - 1)))
    img = cdf[img]
    img = np.clip(img, 0, 255)
    return img
