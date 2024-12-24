import cv2
from scipy.ndimage import zoom as scizoom
from numba import njit, prange
import numpy as np
import math


def gen_lensmask(h, w, gamma):
    """
    Generate lens mask with shape (h, w). 
    For point (i, j), 
    distance = [(i - h // 2)^2 + (j - w // 2)^2] ^ (1/2) / [h // 2)^2 + (w // 2)^2] ^ (1/2)
    mask = scale * (1 - distance) ^ gamma
    
    @param h: height
    @param w: width
    @param gamma: exponential factor
    @return: Mask, H x W
    """
    dist1 = np.array([list(range(w))] * h) - w // 2
    dist2 = np.array([list(range(h))] * w) - h // 2
    dist2 = np.transpose(dist2, (1, 0))
    dist = np.sqrt((dist1 ** 2 + dist2 ** 2)) / np.sqrt((w ** 2 + h ** 2) / 4)
    mask = (1 - dist) ** gamma
    return mask


def gen_disk(radius, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
    else:
        L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    disk /= np.sum(disk)
    return disk


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble,
                                                      array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
        0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    # clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    # clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                  (zoom_factor, zoom_factor, 1), order=1)

    return img


def getOptimalKernelWidth1D(radius, sigma):
    return radius * 2 + 1


def gauss_function(x, mean, sigma):
    return (np.exp(- (x - mean)**2 / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)


def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k/Z


def shift(image, dx, dy):
    if(dx < 0):
        shifted = np.roll(image, shift=image.shape[1]+dx, axis=1)
        shifted[:,dx:] = shifted[:,dx-1:dx]
    elif(dx > 0):
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:,:dx] = shifted[:,dx:dx+1]
    else:
        shifted = image

    if(dy < 0):
        shifted = np.roll(shifted, shift=image.shape[0]+dy, axis=0)
        shifted[dy:,:] = shifted[dy-1:dy,:]
    elif(dy > 0):
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy,:] = shifted[dy:dy+1,:]
    return shifted


def _motion_blur(x, radius, sigma, angle):
    width = getOptimalKernelWidth1D(radius, sigma)
    kernel = getMotionBlurKernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if (np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]):
            # simulated motion exceeded image borders
            break
        shifted = shift(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred


# Numba nopython compilation to shuffle_pixles
@njit()
def shuffle_pixels_njit(img, shift, iteration):
    height, width = img.shape[:2]
    # locally shuffle pixels
    for _ in range(iteration):
        for h in range(height - shift, shift, -1):
            for w in range(width - shift, shift, -1):
                dx, dy = np.random.randint(-shift, shift, size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                img[h, w], img[h_prime, w_prime] = img[h_prime, w_prime], img[h, w]
    return img
