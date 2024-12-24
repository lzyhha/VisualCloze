import os
import cv2
import numpy as np
# from dataset.L0_smooth import L0Smoothing
import argparse 
from skimage.filters import gaussian
from scipy.ndimage.interpolation import map_coordinates
from tqdm import tqdm

# L0Smoothing_class = L0Smoothing(param_lambda=0.02)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def uint2single(img):
    return np.float32(img/255.)

def Laplacian_edge_detector(img):
    # input: [0, 1]
    # return: [0, 1] (H, W, 3)
    img = np.clip(img*255, 0, 255).astype(np.uint8) # (H, W, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Laplacian(img, cv2.CV_16S) # (H, W)
    img = cv2.convertScaleAbs(img)
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, 2).repeat(3, axis=2) # (H, W, 3)
    return img

def Laplacian_edge_detector_uint8(img):
    # input: [0, 255]
    # return: [0, 255] (H, W, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Laplacian(img, cv2.CV_16S) # (H, W)
    img = cv2.convertScaleAbs(img)
    img = np.expand_dims(img, 2).repeat(3, axis=2) # (H, W, 3)
    return img

def Canny_edge_detector(img):
    # input: [0, 1]
    # return: [0, 1] (H, W, 3)
    img = np.clip(img*255, 0, 255).astype(np.uint8) # (H, W, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 50, 200) # (H, W)
    img = cv2.convertScaleAbs(img)
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, 2).repeat(3, axis=2) # (H, W, 3)
    return img

def Canny_edge_detector_uint8(img):
    # input: [0, 255]
    # return: [0, 255] (H, W, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 50, 200) # (H, W)
    img = cv2.convertScaleAbs(img)
    img = np.expand_dims(img, 2).repeat(3, axis=2) # (H, W, 3)
    return img

def Sobel_edge_detector(img):
    # input: [0, 1]
    # return: [0, 1] (H, W, 3)
    img = np.clip(img*255, 0, 255).astype(np.uint8) # (H, W, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Sobel(img, cv2.CV_16S, 1, 1) # (H, W)
    img = cv2.convertScaleAbs(img)
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, 2).repeat(3, axis=2) # (H, W, 3)
    return img
    
# def L0_smooth(img):
#     img = L0Smoothing_class.run(img)
#     return img


def erosion(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    return img

def dilatation(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def opening(img):
    return dilatation(erosion(img))

def closing(img):
    return erosion(dilatation(img))

def morphological_gradient(img):
    return dilatation(img) - erosion(img)

def top_hat(img):
    return img - opening(img)

def black_hat(img):
    return closing(img) - img

def adjust_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    
    image = single2uint(image)
    # 将图像转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 分割LAB图像的L、A、B通道
    l, a, b = cv2.split(lab)

    # 应用直方图均衡化到亮度通道（L通道）
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    # 合并处理后的L、A、B通道并转换回BGR颜色空间
    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    result = uint2single(result)
    return result

def embossing(img):
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    return cv2.filter2D(img, -1, kernel)

# def harris_corner_detection(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#     img[dst > 0.01 * dst.max()] = [0, 0, 255]
#     return img

def hough_transform_line_detection(img):
    img = single2uint(img)
    dst = cv2.Canny(img, 50, 200, apertureSize=3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 230, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))

            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    
    return uint2single(img)

def hough_circle_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=50, maxRadius=200)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
    return img

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def defocus_blur(image, level=(1, 0.1)):
    c = level
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(image[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x64x64 -> 64x64x3

    return np.clip(channels, 0, 1)

def masks_CFA_Bayer(shape):
    pattern = "RGGB"
    channels = dict((channel, np.zeros(shape)) for channel in "RGB")
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1
    return tuple(channels[c].astype(bool) for c in "RGB")

def cfa4_to_rgb(CFA4):
    # 根据CFA4的构造，重新分配通道以重建RGB图像
    RGB = np.zeros((CFA4.shape[0]*2, CFA4.shape[1]*2, 3), dtype=np.uint8)
    RGB[0::2, 0::2, 0] = CFA4[:, :, 0]  # R
    RGB[0::2, 1::2, 1] = CFA4[:, :, 1]  # G on R row
    RGB[1::2, 0::2, 1] = CFA4[:, :, 2]  # G on B row
    RGB[1::2, 1::2, 2] = CFA4[:, :, 3]  # B

    return RGB

def mosaic_CFA_Bayer(RGB):
    RGB = single2uint(RGB)
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2])
    mask = np.concatenate(
        (R_m[..., np.newaxis], G_m[..., np.newaxis], B_m[..., np.newaxis]), axis=-1
    )
    mosaic = np.multiply(mask, RGB)  # mask*RGB
    CFA = mosaic.sum(2).astype(np.uint8)

    CFA4 = np.zeros((RGB.shape[0] // 2, RGB.shape[1] // 2, 4), dtype=np.uint8)
    CFA4[:, :, 0] = CFA[0::2, 0::2]
    CFA4[:, :, 1] = CFA[0::2, 1::2]
    CFA4[:, :, 2] = CFA[1::2, 0::2]
    CFA4[:, :, 3] = CFA[1::2, 1::2]

    rgb = cfa4_to_rgb(CFA4)
    rgb = uint2single(rgb)
    return rgb

def simulate_barrel_distortion(image, k1=0.02, k2=0.01):
    height, width = image.shape[:2]
    mapx, mapy = np.meshgrid(np.arange(width), np.arange(height))
    mapx = 2 * mapx / (width - 1) - 1
    mapy = 2 * mapy / (height - 1) - 1
    r = np.sqrt(mapx**2 + mapy**2)
    mapx = mapx * (1 + k1 * r**2 + k2 * r**4)
    mapy = mapy * (1 + k1 * r**2 + k2 * r**4)
    mapx = (mapx + 1) * (width - 1) / 2
    mapy = (mapy + 1) * (height - 1) / 2
    distorted_image = cv2.remap(image, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR)
    return distorted_image

def simulate_pincushion_distortion(image, k1=-0.02, k2=-0.01):
    height, width = image.shape[:2]
    mapx, mapy = np.meshgrid(np.arange(width), np.arange(height))
    mapx = 2 * mapx / (width - 1) - 1
    mapy = 2 * mapy / (height - 1) - 1
    r = np.sqrt(mapx**2 + mapy**2)
    mapx = mapx * (1 + k1 * r**2 + k2 * r**4)
    mapy = mapy * (1 + k1 * r**2 + k2 * r**4)
    mapx = (mapx + 1) * (width - 1) / 2
    mapy = (mapy + 1) * (height - 1) / 2
    distorted_image = cv2.remap(image, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR)
    return distorted_image

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x_PIL = x
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)

        if len(x.shape) < 3 or x.shape[2] < 3:
            add_spatter_color = cv2.cvtColor(np.clip(m * color, 0, 1),
                                             cv2.COLOR_BGRA2BGR)
            add_spatter_gray = rgb2gray(add_spatter_color)

            return (np.clip(x + add_spatter_gray, 0, 1) * 255).astype(np.uint8)

        else:

            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return (cv2.cvtColor(np.clip(x + m * color, 0, 1),
                                cv2.COLOR_BGRA2BGR) * 255).astype(np.uint8)
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0

        # x_rgb = np.array(x_PIL.convert('RGB'))
        x_rgb = np.array(x_PIL)

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x_rgb[..., :1]),
                                42 / 255. * np.ones_like(x_rgb[..., :1]),
                                20 / 255. * np.ones_like(x_rgb[..., :1])),
                               axis=2)
        color *= m[..., np.newaxis]
        if len(x.shape) < 3 or x.shape[2] < 3:
            x *= (1 - m)
            return (np.clip(x + rgb2gray(color), 0, 1) * 255).astype(np.uint8)

        else:
            x *= (1 - m[..., np.newaxis])
            return (np.clip(x + color, 0, 1) * 255).astype(np.uint8)

# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=3):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    sigma = np.array(shape_size) * 0.01
    alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][
        severity - 1]
    max_dx = shape[0] * 0.005
    max_dy = shape[0] * 0.005

    dx = (gaussian(np.random.uniform(-max_dx, max_dx, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(
        np.float32)
    dy = (gaussian(np.random.uniform(-max_dy, max_dy, size=shape[:2]),
                   sigma, mode='reflect', truncate=3) * alpha).astype(
        np.float32)

    if len(image.shape) < 3 or image.shape[2] < 3:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    else:
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                              np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx,
                                                          (-1, 1)), np.reshape(
            z, (-1, 1))
    return np.clip(
        map_coordinates(image, indices, order=1, mode='reflect').reshape(
            shape), 0, 1) * 255

def frost(x, severity=2):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]

    idx = np.random.randint(5)
    filename = [os.path.join("degradation_toolkit/frost", 'frost1.png'),
                os.path.join("degradation_toolkit/frost", 'frost2.png'),
                os.path.join("degradation_toolkit/frost", 'frost3.png'),
                os.path.join("degradation_toolkit/frost", 'frost4.jpg'),
                os.path.join("degradation_toolkit/frost", 'frost5.jpg'),
                os.path.join("degradation_toolkit/frost", 'frost6.jpg')][idx]
    frost = cv2.imread(filename)
    frost = uint2single(frost)
    frost_shape = frost.shape
    x_shape = np.array(x).shape

    # resize the frost image so it fits to the image dimensions
    scaling_factor = 1
    if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = 1
    elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = x_shape[0] / frost_shape[0]
    elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
        scaling_factor = x_shape[1] / frost_shape[1]
    elif frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[
        1]:  # If both dims are too small, pick the bigger scaling factor
        scaling_factor_0 = x_shape[0] / frost_shape[0]
        scaling_factor_1 = x_shape[1] / frost_shape[1]
        scaling_factor = np.maximum(scaling_factor_0, scaling_factor_1)

    scaling_factor *= 1.1
    new_shape = (int(np.ceil(frost_shape[1] * scaling_factor)),
                 int(np.ceil(frost_shape[0] * scaling_factor)))
    frost_rescaled = cv2.resize(frost, dsize=new_shape,
                                interpolation=cv2.INTER_CUBIC)

    # randomly crop
    x_start, y_start = np.random.randint(0, frost_rescaled.shape[0] - x_shape[
        0]), np.random.randint(0, frost_rescaled.shape[1] - x_shape[1])

    if len(x_shape) < 3 or x_shape[2] < 3:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]]
        frost_rescaled = rgb2gray(frost_rescaled)
    else:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0],
                         y_start:y_start + x_shape[1]][..., [2, 1, 0]]
    return c[0] * np.array(x) + c[1] * frost_rescaled
    # return np.clip(c[0] * np.array(x) + c[1] * frost_rescaled, 0, 255)

def generate_pincushion_distortion_dataset(image_dir, output_path):
    os.makedirs(output_path, exist_ok=True)
    img_paths = sorted(os.listdir(image_dir))
    for img_name in tqdm(img_paths):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img = uint2single(img)
        distorted_img = simulate_pincushion_distortion(img)
        distorted_img = single2uint(distorted_img)
        cv2.imwrite(os.path.join(output_path, img_name), distorted_img)

def generate_barrel_distortion_dataset(image_dir, output_path):
    os.makedirs(output_path, exist_ok=True)
    img_paths = sorted(os.listdir(image_dir))
    for img_name in tqdm(img_paths):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img = uint2single(img)
        distorted_img = simulate_barrel_distortion(img)
        distorted_img = single2uint(distorted_img)
        cv2.imwrite(os.path.join(output_path, img_name), distorted_img)


def save_image(image, path):
    """
    image: saved image, numpy array, dtype=float
    path: saving path
    """
    # The type of the image is float, and range of the image might not be in [0, 255]
    # Thus, before saving the image, the image needs to be clipped.
    image = np.round(np.clip(image, 0, 255)).astype(np.uint8)

    # save image
    cv2.imwrite(path, image)


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/Enhancement/MIT-fivek/expert_C_train/0001.jpg')
    parser.add_argument('--output_path', type=str, default='example')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    generate_pincushion_distortion_dataset("data/Enhancement/MIT-fivek/expert_C_test", "data/Enhancement/MIT-fivek/expert_C_pincushion_distortion_test")   
    generate_pincushion_distortion_dataset("data/Enhancement/MIT-fivek/expert_C_train", "data/Enhancement/MIT-fivek/expert_C_pincushion_distortion_train")   
    generate_barrel_distortion_dataset("data/Enhancement/MIT-fivek/expert_C_test", "data/Enhancement/MIT-fivek/expert_C_barrel_distortion_test")   
    generate_barrel_distortion_dataset("data/Enhancement/MIT-fivek/expert_C_train", "data/Enhancement/MIT-fivek/expert_C_barrel_distortion_train")    
    # opt = init_parser()
    # os.makedirs(opt.output_path, exist_ok=True)
    # img = cv2.imread(opt.input_path)
    # img = uint2single(img)
    # degraded_img = simulate_pincushion_distortion(img)
    # degraded_img = single2uint(degraded_img)
    # save_image(degraded_img, os.path.join(opt.output_path, "simulate_barrel_distortion.png"))
