import os
import numpy as np
import random
import cv2
import math
from scipy import special
from skimage import restoration

import torch
from torch.nn import functional as F
from torchvision.utils import make_grid


def uint2single(img):
    return np.float32(img/255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.
    After clamping to [min, max], values will be normalized to [0, 1].
    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.
    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def get_noise(img, value=10):

    noise = np.random.uniform(0, 256, img.shape[0:2])
    
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise


def rain_blur(noise, length=10, angle=0, w=1):

    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  
    k = cv2.warpAffine(dig, trans, (length, length))  
    k = cv2.GaussianBlur(k, (w, w), 0)  

    blurred = cv2.filter2D(noise, -1, k)  

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    rain = np.expand_dims(blurred, 2)
    blurred = np.repeat(rain, 3, 2)

    return blurred


def add_rain(img,value):
    if np.max(img) > 1:
        pass
    else:
        img = img*255


    w, h, c = img.shape
    h = h - (h % 4)
    w = w - (w % 4)
    img = img[0:w, 0:h, :]


    w = np.random.choice([3, 5, 7, 9, 11], p=[0.2, 0.2, 0.2, 0.2, 0.2])
    length = np.random.randint(30, 41) 
    angle = np.random.randint(-45, 45) 

    noise = get_noise(img, value=value)
    rain = rain_blur(noise, length=length, angle=angle, w=w)

    img = img.astype('float32') + rain
    np.clip(img, 0, 255, out=img)
    img = img/255.0
    return img


def add_rain_range(img, value_min, value_max):
    value = np.random.randint(value_min, value_max)
    if np.max(img) > 1:
        pass
    else:
        img = img*255


    w, h, c = img.shape
    h = h - (h % 4)
    w = w - (w % 4)
    img = img[0:w, 0:h, :]


    w = np.random.choice([3, 5, 7, 9, 11], p=[0.2, 0.2, 0.2, 0.2, 0.2])
    length = np.random.randint(30, 41) 
    angle = np.random.randint(-45, 45) 

    noise = get_noise(img, value=value)
    rain = rain_blur(noise, length=length, angle=angle, w=w)

    img = img.astype('float32') + rain
    np.clip(img, 0, 255, out=img)
    img = img/255.0
    return img


def add_Poisson_noise(img, level=2):
    # input range[0, 1]
    vals = 10**(level)  
    img = np.random.poisson(img * vals).astype(np.float32) / vals
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Gaussian_noise(img, level=20):
    # input range[0, 1]
    noise_level = level / 255.0
    noise_map = np.random.normal(loc=0.0, scale=1.0, size=img.shape)*noise_level
    img += noise_map
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Gaussian_noise_range(img, min_level=10, max_level=50):
    # input range[0, 1]
    level = random.uniform(min_level, max_level)
    noise_level = level / 255.0
    noise_map = np.random.normal(loc=0.0, scale=1.0, size=img.shape)*noise_level
    img += noise_map
    img = np.clip(img, 0.0, 1.0)
    return img


def add_sp_noise(img, snr=0.95, salt_pro=0.5):
    # input range[0, 1]
    output = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < snr:
                output[i][j] = img[i][j]
            else:
                rdn = random.random()
                if rdn < salt_pro:
                    output[i][j] = 1
                else:
                    output[i][j] = 0
    
    return output


def add_JPEG_noise(img, level):

    quality_factor = level
    img = single2uint(img)
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = uint2single(img)
    
    return img


def add_JPEG_noise_range(img, level_min, level_max):

    quality_factor = random.randint(level_min, level_max)
    img = single2uint(img)
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = uint2single(img)
    
    return img


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter, ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)) / ((2 * np.pi * np.sqrt(
            (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)) + 1e-9), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff ** 2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def sinc(img, kernel_size,omega_c):

    sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
    sinc_kernel = torch.FloatTensor(sinc_kernel)

    img = filter2D(img,sinc_kernel)

    return img


def add_ringing(img):
    # input: [0, 1]
    img = img2tensor([img])[0].unsqueeze(0)
    ks = 15
    omega_c = round(1.2, 2)
    img = sinc(img, ks, omega_c)
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    img = tensor2img(img, min_max=(0, 1))
    img = img/255.0
    return img


def low_light(img, lum_scale):
    img = img*lum_scale
    return img


def low_light_range(img):
    lum_scale = random.uniform(0.1, 0.5)
    img = img*lum_scale
    return img


def iso_GaussianBlur(img, window, sigma):
    img = cv2.GaussianBlur(img.copy(), (window, window), sigma)
    return img


def iso_GaussianBlur_range(img, window, min_sigma=2, max_sigma=4):
    sigma = random.uniform(min_sigma, max_sigma)
    img = cv2.GaussianBlur(img.copy(), (window, window), sigma)
    return img


def add_resize(img):
    ori_H, ori_W = img.shape[0], img.shape[1]
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.2, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = cv2.resize(img, (int(ori_W), int(ori_H)), interpolation=random.choice([1, 2, 3]))
    
    img = np.clip(img, 0.0, 1.0)

    return img


def r_l(img):
    img = img2tensor([img],bgr2rgb=False)[0].unsqueeze(0)
    psf = np.ones((1, 1, 5, 5))
    psf = psf / psf.sum()
    img = img.numpy()
    img = np.pad(img, ((0, 0), (0, 0), (7, 7), (7, 7)), 'linear_ramp')
    img = restoration.richardson_lucy(img, psf, 1)
    img = img[:, :, 7:-7, 7:-7]
    img = torch.from_numpy(img)
    img = img.squeeze(0).numpy().transpose(1, 2, 0)
    return img


def inpainting(img,l_num,l_thick):

    ori_h, ori_w = img.shape[0], img.shape[1]
    mask = np.zeros((ori_h, ori_w, 3), np.uint8)
    col = random.choice(['white', 'black'])
    while (l_num):
        x1, y1 = random.randint(0, ori_w), random.randint(0, ori_h)
        x2, y2 = random.randint(0, ori_w), random.randint(0, ori_h)
        pts = np.array([[x1, y1], [x2, y2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        mask = cv2.polylines(mask, [pts], 0, (1, 1, 1), l_thick)
        l_num -= 1

    if col == 'white':
        img = np.clip(img + mask, 0, 1)
    else:
        img = np.clip(img - mask, 0, 1)

    return img
