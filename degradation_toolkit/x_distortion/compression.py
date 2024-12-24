import numpy as np
from PIL import Image
from io import BytesIO


def compression_jpeg(img, severity=1):
    """
    JPEG compression on a NumPy array.
    severity=[1,2,3,4,5] corresponding to quality=[25,18,15,10,7].
    from https://github.com/bethgelab/imagecorruptions/blob/master/imagecorruptions/corruptions.py

    @param img: Input image as NumPy array, H x W x C, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image as NumPy array, H x W x C, value range [0, 255]
    """
    assert img.dtype == np.uint8, "Image array should have dtype of np.uint8"
    assert severity in [1, 2, 3, 4, 5], 'Severity must be an integer between 1 and 5.'

    quality = [25, 18, 12, 8, 5][severity - 1]
    output = BytesIO()
    gray_scale = False
    if img.shape[2] == 1:  # Check if the image is grayscale
        gray_scale = True
    # Convert NumPy array to PIL Image
    img = Image.fromarray(img)
    if gray_scale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    # Save image to a bytes buffer using JPEG compression
    img.save(output, 'JPEG', quality=quality)
    output.seek(0)
    # Load the compressed image from the bytes buffer
    img_lq = Image.open(output)
    # Convert PIL Image back to NumPy array
    if gray_scale:
        img_lq = np.array(img_lq.convert('L'))
        img_lq = img_lq.reshape((img_lq.shape[0], img_lq.shape[1], 1))  # Maintaining the original shape (H, W, 1)
    else:
        img_lq = np.array(img_lq.convert('RGB'))
    return img_lq


def compression_jpeg_2000(img, severity=1):
    """
    JPEG2000 compression on a NumPy array.
    severity=[1,2,3,4,5] corresponding to quality=[29,27.5,26,24.5,23], quality_mode='dB'.

    @param x: Input image as NumPy array, H x W x C, value range [0, 255]
    @param severity: Severity of distortion, [1, 5]
    @return: Degraded image as NumPy array, H x W x C, value range [0, 255]
    """
    assert img.dtype == np.uint8, "Image array should have dtype of np.uint8"
    assert severity in [1, 2, 3, 4, 5], 'Severity must be an integer between 1 and 5.'

    quality = [29, 27.5, 26, 24.5, 23][severity - 1]
    output = BytesIO()
    gray_scale = False
    if img.shape[2] == 1:  # Check if the image is grayscale
        gray_scale = True
    # Convert NumPy array to PIL Image
    img = Image.fromarray(img)
    if gray_scale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    # Save image to a bytes buffer using JPEG compression
    img.save(output, 'JPEG2000', quality_mode='dB', quality_layers=[quality])
    output.seek(0)
    # Load the compressed image from the bytes buffer
    img_lq = Image.open(output)
    # Convert PIL Image back to NumPy array
    if gray_scale:
        img_lq = np.array(img_lq.convert('L'))
        img_lq = img_lq.reshape((img_lq.shape[0], img_lq.shape[1], 1))  # Maintaining the original shape (H, W, 1)
    else:
        img_lq = np.array(img_lq.convert('RGB'))
    return img_lq
