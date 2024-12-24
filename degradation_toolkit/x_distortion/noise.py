import cv2
import numpy as np
import skimage as sk


def noise_gaussian_RGB(img, severity=1):
    """
    Additive Gaussian noise in RGB channels. 
    severity=[1, 2, 3, 4, 5] is corresponding to sigma=[0.05, 0.1, 0.15, 0.2, 0.25]. 
    severity mainly refer to KADID-10K and Imagecorruptions.

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    sigma = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    img = np.array(img) / 255.
    noise = np.random.normal(0, sigma, img.shape)
    img_lq = img + noise
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def noise_gaussian_YCrCb(img, severity=1):
    """
    Additive Gaussian noise with higher noise in color channels. 
    severity=[1, 2, 3, 4, 5] is corresponding to
    sigma_l=[0.05, 0.06, 0.07, 0.08, 0.09], 
    sigma_r=[1, 1.45, 1.9, 2.35, 2.8], 
    sigma_b=[1, 1.45, 1.9, 2.35, 2.8]. 

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    sigma_l = [0.05, 0.06, 0.07, 0.08, 0.09][severity-1]
    sigma_r = sigma_l * [1, 1.45, 1.9, 2.35, 2.8][severity - 1]
    sigma_b = sigma_l * [1, 1.45, 1.9, 2.35, 2.8][severity - 1]
    h, w = img.shape[:2]
    img = np.float32(np.array(img) / 255.)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    noise_l = np.expand_dims(np.random.normal(0, sigma_l, (h, w)), 2)
    noise_r = np.expand_dims(np.random.normal(0, sigma_r, (h, w)), 2)
    noise_b = np.expand_dims(np.random.normal(0, sigma_b, (h, w)), 2)
    noise = np.concatenate((noise_l, noise_r, noise_b), axis=2)
    img_lq = np.float32(img + noise)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_YCR_CB2RGB)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def noise_speckle(img, severity=1):
    """
    Multiplicative Gaussian noise. 
    severity=[1, 2, 3, 4, 5] is corresponding to sigma=[0.14, 0.21, 0.28, 0.35, 0.42].

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [0.14, 0.21, 0.28, 0.35, 0.42][severity - 1]
    img = np.array(img) / 255.
    noise = img * np.random.normal(size=img.shape, scale=c)
    img_lq = img + noise
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def noise_spatially_correlated(img, severity=1):
    """
    Spatially correlated noise. 
    severity=[1, 2, 3, 4, 5] is corresponding to sigma=[0.08, 0.11, 0.14, 0.18, 0.22].

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    sigma = [0.08, 0.11, 0.14, 0.18, 0.22][severity - 1]
    img = np.array(img) / 255.
    noise = np.random.normal(0, sigma, img.shape)
    img_lq = img + noise
    img_lq = cv2.blur(img_lq, [3, 3])
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def noise_poisson(img, severity=1):
    """
    Poisson noise. 
    PieAPP keeps this distortion free of additional parameters.
    The default:
    c =  vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    But Imagecorruptions introduces a extra parameter c
    ranging [60, 25, 12, 5, 3] for sigma = sqrt(I / c).
    severity=[1, 2, 3, 4, 5] is corresponding to c=[80, 60, 40, 25, 15].

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [80, 60, 40, 25, 15][severity - 1]
    img = np.array(img) / 255.
    img_lq = np.random.poisson(img * c) / float(c)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)


def noise_impulse(img, severity=1):
    """
    Impulse noise is also known as salt&pepper noise.
    PieAPP introduce the range [1e-4, 0.045].
    severity=[1, 2, 3, 4, 5] is corresponding to amount=[0.01, 0.03, 0.05, 0.07, 0.10].

    @param img: Input image, H x W x 3, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image, H x W x 3, value range [0, 255]
    """
    c = [0.01, 0.03, 0.05, 0.07, 0.10][severity - 1]
    img = np.array(img) / 255.
    img_lq = sk.util.random_noise(img, mode='s&p', amount=c)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.)
