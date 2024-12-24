import cv2
import numpy as np


def oversharpen(img, severity=1):
    """
    OverSharpening filter on a NumPy array.
    severity = [1, 5] corresponding to amount = [2, 4, 6, 8, 10]
    
    @param x: Input image as NumPy array, H x W x C, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image as NumPy array, H x W x C, value range [0, 255]
    """
    assert img.dtype == np.uint8, "Image array should have dtype of np.uint8"
    assert severity in [1, 2, 3, 4, 5], 'Severity must be an integer between 1 and 5.'
    
    amount = [2, 2.8, 4, 6, 8][severity - 1]

    # Setting the kernel size and sigmaX value for Gaussian blur
    # In OpenCV's Size(kernel_width, kernel_height), both kernel_width and kernel_height
    # should be odd numbers; for example, we can use (2*radius+1, 2*radius+1)
    blur_radius = 2  # The radius is the blur radius used to set the size of the Gaussian kernel
    sigmaX = 0

    # Create a blurred/smoothed version of the image
    blurred = cv2.GaussianBlur(img, (2*blur_radius+1, 2*blur_radius+1), sigmaX)

    # Compute the sharpened image with an enhancement factor of 'amount'
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    return sharpened
