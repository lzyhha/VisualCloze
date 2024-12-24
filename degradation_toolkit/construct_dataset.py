import functools
import io
import json
import logging
import os
import queue
import random
import sys
import threading
from threading import Lock

print("load packages")
import basicsr
import cv2
import numpy as np
import torch
from basicsr.utils.matlab_functions import imresize
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from add_degradation_various import *
from data_reader import init_ceph_client_if_needed, read_img_ceph
from image_operators import *
from petrel_client.client import Client  # noqa
from x_distortion import *

print("successfully load packages!")

print("loading client...")
client = Client("../petreloss.conf")
print("successfully load client!")

saved_images = set()
save_lock = Lock()


def read_img(env, path, size=None, float=True):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    if float:
        img = img.astype(np.float32) / 255.0
    return img


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.0).round())


def uint2single(img):
    return np.float32(img / 255.0)


def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k])[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def add_x_distortion_single_images(img_gt1, deg_type):
    # np.uint8, BGR
    x_distortion_dict = distortions_dict
    severity = random.choice([1, 2, 3, 4, 5])
    if deg_type == 'compression' or deg_type == "quantization":
        severity = min(3, severity)
    deg_type = random.choice(x_distortion_dict[deg_type])

    img_gt1 = cv2.cvtColor(img_gt1, cv2.COLOR_BGR2RGB)
    img_lq1 = globals()[deg_type](img_gt1, severity)

    img_gt1 = cv2.cvtColor(img_gt1, cv2.COLOR_RGB2BGR)
    img_lq1 = cv2.cvtColor(img_lq1, cv2.COLOR_RGB2BGR)

    return img_lq1, img_gt1, deg_type


def calculate_operators_single_images(img_gt1, deg_type):
    # np.uint8
    if deg_type == 'Laplacian':
        img_lq1 = img_gt1.copy()
        img_gt1 = Laplacian_edge_detector(img_gt1)
    elif deg_type == 'Canny':
        img_lq1 = img_gt1.copy()
        img_gt1 = Canny_edge_detector(img_gt1)
    # check zero images.
    if np.mean(img_gt1).astype(np.float16) == 0:
        if np.mean(img_gt1).astype(np.float16) == 0:
            print(deg_type, 'prompt&query zero images.')
            img_gt1 = img_lq1.copy()
    return img_lq1, img_gt1


def add_degradation_single_images(img_gt1, deg_type):
    # np.float32
    if deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_lq1 = add_rain(img_gt1, value=value)
    elif deg_type == 'Ringing':
        img_lq1 = add_ringing(img_gt1)
    elif deg_type == 'r_l':
        img_lq1 = r_l(img_gt1)
    elif deg_type == 'Inpainting':
        l_num = random.randint(20, 50)
        l_thick = random.randint(10, 20)
        img_lq1 = inpainting(img_gt1, l_num=l_num, l_thick=l_thick)
    elif deg_type == 'mosaic':
        img_lq1 = mosaic_CFA_Bayer(img_gt1)
    elif deg_type == 'SRx2':
        H, W, _ = img_gt1.shape
        img_lq1 = imresize(img_gt1, 1 / 2)
        img_lq1 = cv2.resize(img_lq1, (W, H), interpolation=cv2.INTER_CUBIC)
    elif deg_type == 'SRx4':
        H, W, _ = img_gt1.shape
        img_lq1 = imresize(img_gt1, 1 / 4)
        img_lq1 = cv2.resize(img_lq1, (W, H), interpolation=cv2.INTER_CUBIC)

    elif deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_lq1 = add_Gaussian_noise(img_gt1, level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_lq1 = iso_GaussianBlur(img_gt1, window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_lq1 = add_JPEG_noise(img_gt1, level=level)
    elif deg_type == 'Resize':
        img_lq1 = add_resize(img_gt1)
    elif deg_type == 'SPNoise':
        img_lq1 = add_sp_noise(img_gt1)
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_lq1 = low_light(img_gt1, lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_lq1 = add_Poisson_noise(img_gt1, level=2)
    elif deg_type == 'gray':
        img_lq1 = cv2.cvtColor(img_gt1, cv2.COLOR_BGR2GRAY)
        img_lq1 = np.expand_dims(img_lq1, axis=2)
        img_lq1 = np.concatenate((img_lq1, img_lq1, img_lq1), axis=2)
    elif deg_type == 'None':
        img_lq1 = img_gt1
    else:
        print('Error!', '-', deg_type, '-')
        exit()

    img_lq1 = np.clip(img_lq1 * 255, 0, 255).round().astype(np.uint8)
    img_lq1 = img_lq1.astype(np.float32) / 255.0

    img_gt1 = np.clip(img_gt1 * 255, 0, 255).round().astype(np.uint8)
    img_gt1 = img_gt1.astype(np.float32) / 255.0

    return img_lq1, img_gt1


class ImageDataset(Dataset):
    def __init__(self, image_path_list=None, json_file=None, patch_size=32, target_size=256):
        """
        初始化函数，可直接通过 image_path_list 传递多张图像路径，或者通过 json_file 传递多张图像路径。
        :param image_path_list: 多张图像路径列表
        :param json_file: 包含图像信息的JSON文件路径
        :param patch_size: 每个patch的大小，用于生成裁剪尺寸
        :param target_size: 目标图像大小
        """
        self.use_json = json_file is not None

        if image_path_list:
            self.image_urls = image_path_list  # 使用多张图像路径列表
            self.descriptions = [{} for _ in image_path_list]  # 为每个图像创建空的描述信息
        elif json_file:
            with open(json_file, 'r') as f:
                self.data = json.load(f)

            self.image_urls = []
            self.descriptions = []
            for item in self.data:
                if 'image_url' in item:
                    self.image_urls.append(item['image_url'])
                    self.descriptions.append(
                        {
                            'blip2_short_cap': item.get('blip2_short_cap', ''),
                            'cogvlm_long': item.get('cogvlm_long', ''),
                            'sharegpt4v_long_cap': item.get('sharegpt4v_long_cap', ''),
                            'llava13b_long_cap': item.get('llava13b_long_cap', ''),
                        }
                    )

            # 确保image_urls和descriptions的长度相同
            assert len(self.image_urls) == len(self.descriptions), "图像URL和描述的数量不匹配"

        # 生成裁剪尺寸列表
        self.patch_size = patch_size
        self.crop_size_list = generate_crop_size_list((target_size // patch_size) ** 2, patch_size)
        logging.info("List of crop sizes:")
        for i in range(0, len(self.crop_size_list), 6):
            logging.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]))

        # 定义图像变换
        self.image_transform = transforms.Compose(
            [
                # transforms.Lambda(functools.partial(var_center_crop, crop_size_list=self.crop_size_list)),
                transforms.ToTensor(),
            ]
        )

        # 初始化退化操作列表
        # self.onthefly_degradation_list1 = [
        #     'blur',
        #     'noise',
        #     'compression',
        #     'brighten',
        #     'darken',
        #     'spatter',
        #     'contrast_strengthen',
        #     'contrast_weaken',
        #     'saturate_strengthen',
        #     'saturate_weaken',
        #     'oversharpen',
        #     'pixelate',
        #     'quantization',
        # ]
        self.onthefly_degradation_list1 = []
        # self.onthefly_degradation_list2 = ['Rain', 'Ringing', 'r_l', 'Inpainting', 'mosaic', 'SRx2', 'SRx4']
        self.onthefly_degradation_list2 = []
        # self.onthefly_degradation_list3 = ['Laplacian', 'Canny']
        self.onthefly_degradation_list3 = []
        self.onthefly_degradation_list4 = [
            'flip',
            'rotate90',
            'rotate180',
            'rotate270',
            'identity',
        ]

        # 将所有退化类型合并成一个列表
        self.all_degradation_types = (
            self.onthefly_degradation_list1 + self.onthefly_degradation_list2 + self.onthefly_degradation_list3
        )

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.image_urls)

    def __getitem__(self, idx):
        """
        根据索引返回裁剪、退化和处理后的图像及其相关信息
        :param idx: 索引
        :return: 返回处理后的图像和描述信息
        """
        # 判断图像来源并读取图像
        if self.use_json:
            img_path = self.image_urls[idx]
            image = read_img_ceph(img_path)
        else:
            img_path = self.image_urls[idx]
            image = read_img(None, img_path, None, float=False)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 图像处理步骤
        image.save('debug.jpg')
        image = var_center_crop(image, crop_size_list=self.crop_size_list)
        image = np.array(image)

        # 保存每种退化后的图像
        degradation_images = {}
        for deg_type in self.all_degradation_types:
            if deg_type in self.onthefly_degradation_list1:
                img_lq1, _, _ = add_x_distortion_single_images(np.copy(image), deg_type)
                img_lq1 = uint2single(img_lq1)
            elif deg_type in self.onthefly_degradation_list2:
                img_lq1, _ = add_degradation_single_images(np.copy(uint2single(image)), deg_type)
            elif deg_type in self.onthefly_degradation_list3:
                _, img_lq1 = calculate_operators_single_images(np.copy(image), deg_type)
                # img_lq1 = uint2single(img_lq1)
            elif deg_type in self.onthefly_degradation_list4:
                img_lq1 = np.copy(uint2single(image))
                if deg_type == 'flip':
                    img_lq1 = np.flip(img_lq1, axis=1)
                elif deg_type == 'rotate90':
                    img_lq1 = np.rot90(img_lq1, k=1)
                elif deg_type == 'rotate180':
                    img_lq1 = np.rot90(img_lq1, k=2)
                elif deg_type == 'rotate270':
                    img_lq1 = np.rot90(img_lq1, k=3)
                elif deg_type == 'identity':
                    pass

            # 转换为张量格式
            img_lq1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq1, (2, 0, 1)))).float()

            # 保存退化后的图像
            degradation_images[deg_type] = img_lq1

        descriptions = self.descriptions[idx] if self.descriptions else {}
        image = self.image_transform(image)
        # 返回干净图像、退化图像列表和描述信息
        return {"img_clean": image, "degradation_images": degradation_images, "description": descriptions}


def save_dataset_images(dataset, n=1, save_dir="saved_images"):
    """
    保存数据集中前n张图像,包括干净图像和所有退化图像
    :param dataset: ImageDataset实例
    :param n: 要保存的图像数量
    :param save_dir: 保存图像的文件夹
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"开始保存数据集中的前 {n} 张图片...")

    # 保存图像的通用函数
    def save_image(img, filename):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(filename)
        print(f"图像已保存: {filename}")

    for index in range(min(n, len(dataset))):
        print(f"正在处理第 {index + 1} 张图片...")
        data = dataset[index]
        img_clean = data["img_clean"]
        degradation_images = data["degradation_images"]
        descriptions = data["description"]

        # 保存描述信息
        with open(os.path.join(save_dir, f"description_{index}.txt"), 'w', encoding='utf-8') as f:
            for key, value in descriptions.items():
                f.write(f"{key}: {value}\n")

        # 保存干净图像
        save_image(img_clean, os.path.join(save_dir, f"clean_image_{index}.png"))

        # 保存退化图像
        for deg_type, img_lq in degradation_images.items():
            save_image(img_lq, os.path.join(save_dir, f"degradation_{deg_type}_{index}.png"))

    print(f"保存完成。共保存了 {min(n, len(dataset))} 张图片。")


def save_dataset_images_threaded(dataset, n=10, save_dir="saved_images", num_threads=2):
    """
    使用多线程保存数据集中前n张图像,包括干净图像和所有退化图像
    :param dataset: ImageDataset实例
    :param n: 要保存的图像数量
    :param save_dir: 保存图像的文件夹
    :param num_threads: 使用的线程数
    """
    import threading
    from queue import Queue

    os.makedirs(save_dir, exist_ok=True)
    print(f"开始使用 {num_threads} 个线程保存数据集中的前 {n} 张图片...")

    # 保存图像的通用函数
    def save_image(img, filename):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(filename)
        print(f"图像已保存: {filename}")

    # 工作线程函数
    def worker():
        while True:
            item = q.get()
            if item is None:
                break
            index, data = item
            img_clean = data["img_clean"]
            degradation_images = data["degradation_images"]
            descriptions = data["description"]

            # 保存描述信息
            with open(os.path.join(save_dir, f"description_{index}.txt"), 'w', encoding='utf-8') as f:
                for key, value in descriptions.items():
                    f.write(f"{key}: {value}\n")

            # 保存干净图像
            save_image(img_clean, os.path.join(save_dir, f"clean_image_{index}.png"))

            # 保存退化图像
            for deg_type, img_lq in degradation_images.items():
                save_image(img_lq, os.path.join(save_dir, f"degradation_{deg_type}_{index}.png"))

            q.task_done()

    # 创建任务队列
    q = Queue()

    # 创建工作线程
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # 添加任务到队列
    for index in range(min(n, len(dataset))):
        q.put((index, dataset[index]))

    # 添加结束标志
    for _ in range(num_threads):
        q.put(None)

    # 等待所有任务完成
    q.join()

    # 等待所有线程结束
    for t in threads:
        t.join()

    print(f"保存完成。共保存了 {min(n, len(dataset))} 张图片。")


def save_dataset_images_chunked(dataset, max_img_num, num_chunks, chunk_id, base_path="genlv_data", num_threads=4):
    """
    将数据集中的图像分块保存到本地指定文件夹中
    :param dataset: ImageDataset实例
    :param max_img_num: 最大保存图像数量
    :param num_chunks: 总块数
    :param chunk_id: 当前处理的块ID（从0开始）
    :param base_path: 本地基础路径
    :param num_threads: 线程数
    """
    chunk_size = max_img_num // num_chunks
    start_index = chunk_id * chunk_size
    end_index = min((chunk_id + 1) * chunk_size, len(dataset))

    print(f"开始处理第 {chunk_id + 1}/{num_chunks} 块，图像范围：{start_index} - {end_index - 1}")

    json_data = {}
    task_queue = queue.Queue()

    def save_image_to_local(img, filename):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img_pil.save(filename, format='PNG')
        return filename

    def process_image(index):
        data = dataset[index]
        img_clean = data["img_clean"]
        degradation_images = data["degradation_images"]
        descriptions = data["description"]

        gt_filename = os.path.join(base_path, "gt_imgs", f"{index:04d}.png")
        save_image_to_local(img_clean, gt_filename)

        json_entries = []
        for deg_type, img_lq in degradation_images.items():
            deg_filename = os.path.join(base_path, deg_type, f"{index:04d}.png")
            save_image_to_local(img_lq, deg_filename)

            json_entry = {
                "input_img_url": deg_filename,
                "target_img_url": gt_filename,
                "task_name": deg_type,
                "descriptions": descriptions,
            }
            json_entries.append(json_entry)

        return json_entries

    def worker():
        while True:
            try:
                index = task_queue.get_nowait()
            except queue.Empty:
                break

            try:
                entries = process_image(index)
                with save_lock:
                    for entry in entries:
                        deg_type = entry["task_name"]
                        if deg_type not in json_data:
                            json_data[deg_type] = []
                        json_data[deg_type].append(entry)
                print(f"处理完成图像 {index}")
            except Exception as exc:
                print(f"处理图像 {index} 时发生错误: {exc}")
            finally:
                task_queue.task_done()

    for index in range(start_index, end_index):
        task_queue.put(index)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    task_queue.join()

    for t in threads:
        t.join()

    # 保存JSON文件
    for deg_type, entries in json_data.items():
        json_filename = os.path.join(base_path, f"{deg_type}_data_chunk_{chunk_id}.json")
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=4, ensure_ascii=False)
        print(f"JSON文件已保存到本地: {json_filename}")

    print(f"第 {chunk_id + 1}/{num_chunks} 块处理完成")


def save_dataset_images_to_ceph(dataset, max_images=100, base_path="genlv_data"):
    """
    将数据集中的图像保存到Ceph存储系统的指定文件夹中
    :param dataset: ImageDataset实例
    :param max_images: 最大保存图像数量
    :param base_path: Ceph中的基础路径
    """
    init_ceph_client_if_needed()
    if client is None:
        raise RuntimeError("Ceph客户端初始化失败。请检查init_ceph_client_if_needed函数。")

    print(f"开始将最多 {max_images} 张图片保存到Ceph...")

    def save_image_to_ceph(img, filename):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        client.put(filename, img_byte_arr)
        return filename

    json_data = {}
    saved_images = set()

    for index in range(min(len(dataset), max_images)):
        data = dataset[index]
        img_clean = data["img_clean"]
        degradation_images = data["degradation_images"]
        descriptions = data["description"]

        gt_filename = f"{base_path}/gt_imgs/{index:04d}.png"

        if gt_filename not in saved_images:
            save_image_to_ceph(img_clean, gt_filename)
            saved_images.add(gt_filename)

        for deg_type, img_lq in degradation_images.items():
            deg_filename = f"{base_path}/{deg_type}/{index:04d}.png"

            if deg_filename not in saved_images:
                save_image_to_ceph(img_lq, deg_filename)
                saved_images.add(deg_filename)

            json_entry = {
                "input_img_url": deg_filename,
                "target_img_url": gt_filename,
                "task_name": deg_type,
                "descriptions": descriptions,
            }

            if deg_type not in json_data:
                json_data[deg_type] = []
            json_data[deg_type].append(json_entry)

        print(f"处理完成图像 {index}")

    # 保存JSON文件
    for deg_type, entries in json_data.items():
        json_filename = f"{base_path}/{deg_type}_data.json"
        json_content = json.dumps(entries, indent=4, ensure_ascii=False)
        client.put(json_filename, json_content.encode('utf-8'))
        print(f"JSON文件已保存到Ceph: {json_filename}")

    print("保存到Ceph完成")


def save_dataset_images_to_ceph_threaded(dataset, max_images=100, base_path="genlv_data", num_threads=4):
    """
    使用多线程将数据集中的图像保存到Ceph存储系统的指定文件夹中
    :param dataset: ImageDataset实例
    :param max_images: 最大保存图像数量
    :param base_path: Ceph中的基础路径
    :param num_threads: 使用的线程数
    """
    init_ceph_client_if_needed()
    if client is None:
        raise RuntimeError("Ceph客户端初始化失败。请检查init_ceph_client_if_needed函数。")

    print(f"开始使用 {num_threads} 个线程将最多 {max_images} 张图片保存到Ceph...")

    def save_image_to_ceph(img, filename):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        client.put(filename, img_byte_arr)
        return filename

    json_data = {}
    saved_images = set()
    save_lock = threading.Lock()

    def process_image(index):
        data = dataset[index]
        img_clean = data["img_clean"]
        degradation_images = data["degradation_images"]
        descriptions = data["description"]

        gt_filename = f"{base_path}/gt_imgs/{index:04d}.png"

        with save_lock:
            if gt_filename not in saved_images:
                save_image_to_ceph(img_clean, gt_filename)
                saved_images.add(gt_filename)

        json_entries = []
        for deg_type, img_lq in degradation_images.items():
            deg_filename = f"{base_path}/{deg_type}/{index:04d}.png"

            with save_lock:
                if deg_filename not in saved_images:
                    save_image_to_ceph(img_lq, deg_filename)
                    saved_images.add(deg_filename)

            json_entry = {
                "input_img_url": deg_filename,
                "target_img_url": gt_filename,
                "task_name": deg_type,
                "descriptions": descriptions,
            }
            json_entries.append(json_entry)

        return json_entries

    def worker():
        while True:
            try:
                index = task_queue.get_nowait()
            except queue.Empty:
                break

            try:
                entries = process_image(index)
                with save_lock:
                    for entry in entries:
                        deg_type = entry["task_name"]
                        if deg_type not in json_data:
                            json_data[deg_type] = []
                        json_data[deg_type].append(entry)
                print(f"处理完成图像 {index}")
            except Exception as exc:
                print(f"处理图像 {index} 时发生错误: {exc}")
            finally:
                task_queue.task_done()

    task_queue = queue.Queue()
    for index in range(min(len(dataset), max_images)):
        task_queue.put(index)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    task_queue.join()

    for t in threads:
        t.join()

    # 保存JSON文件
    for deg_type, entries in json_data.items():
        json_filename = f"{base_path}/{deg_type}_data.json"
        json_content = json.dumps(entries, indent=4, ensure_ascii=False)
        client.put(json_filename, json_content.encode('utf-8'))
        print(f"JSON文件已保存到Ceph: {json_filename}")

    print("保存到Ceph完成")


def save_dataset_images_to_ceph_chunk(dataset, max_img_num, num_chunks, chunk_id, base_path="genlv_data", num_threads=4):
    """
    将数据集中的图像分块保存到Ceph中
    :param dataset: ImageDataset实例
    :param max_img_num: 最大保存图像数量
    :param num_chunks: 总块数
    :param chunk_id: 当前处理的块ID（从0开始）
    :param base_path: Ceph中的基础路径
    :param num_threads: 线程数
    """
    chunk_size = max_img_num // num_chunks
    start_index = chunk_id * chunk_size
    end_index = min((chunk_id + 1) * chunk_size, len(dataset))

    print(f"开始处理第 {chunk_id + 1}/{num_chunks} 块，图像范围：{start_index} - {end_index - 1}")

    json_data = {}
    task_queue = queue.Queue()

    def save_image_to_ceph(img, filename):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        client.put(filename, buffer.getvalue())
        return filename

    def process_image(index):
        data = dataset[index]
        img_clean = data["img_clean"]
        degradation_images = data["degradation_images"]
        descriptions = data["description"]

        gt_filename = f"{base_path}/gt_imgs/{index:04d}.png"
        # save_image_to_ceph(img_clean, gt_filename)

        json_entries = []
        for deg_type, img_lq in degradation_images.items():
            deg_filename = f"{base_path}/{deg_type}/{index:04d}.png"
            if deg_type == 'identity':
                deg_filename = gt_filename
            else:
                save_image_to_ceph(img_lq, deg_filename)

            json_entry = {
                "input_img_url": deg_filename,
                "gt_img_url": gt_filename,
                "task_name": deg_type,
                "description": descriptions,
            }
            json_entries.append(json_entry)

        return json_entries

    def worker():
        while True:
            try:
                index = task_queue.get_nowait()
            except queue.Empty:
                break

            try:
                entries = process_image(index)
                with save_lock:
                    for entry in entries:
                        deg_type = entry["task_name"]
                        if deg_type not in json_data:
                            json_data[deg_type] = []
                        json_data[deg_type].append(entry)
                print(f"处理完成图像 {index}")
            except Exception as exc:
                print(f"处理图像 {index} 时发生错误:")
                print(traceback.format_exc())
            finally:
                task_queue.task_done()

    for index in range(start_index, end_index):
        task_queue.put(index)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    task_queue.join()

    for t in threads:
        t.join()

    # 保存JSON文件到Ceph
    for deg_type, entries in json_data.items():
        json_filename = f"{base_path}/{deg_type}_data_{chunk_id}.json"
        json_content = json.dumps(entries, indent=4, ensure_ascii=False)
        client.put(json_filename, json_content.encode('utf-8'))
        print(f"JSON文件已保存到Ceph: {json_filename}")

    print(f"第 {chunk_id + 1}/{num_chunks} 块保存到Ceph完成")


def parse_args():
    parser = argparse.ArgumentParser(description='构建数据集参数')
    parser.add_argument('--chunk_id', type=int, default=0, help='当前处理的块ID')
    parser.add_argument('--max_img_num', type=int, default=300000, help='最大处理图像数量')
    parser.add_argument('--num_chunks', type=int, default=100, help='总块数')
    parser.add_argument('--base_path', type=str, default='genlv_data', help='保存数据的基础路径')
    parser.add_argument('--num_threads', type=int, default=4, help='处理线程数')
    parser.add_argument('--json_file', type=str, default='omnigen/unsplash.json', help='JSON文件路径')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 创建数据集实例
    print("Building ImageDataset...")
    dataset = ImageDataset(json_file=args.json_file, patch_size=32, target_size=1024)

    # 调用调试函数，调试第 0 张图片并保存图像
    print(f"Start construct data")

    # save_dataset_images_chunked(
    #     dataset,
    #     max_img_num=args.max_img_num,
    #     num_chunks=args.num_chunks,
    #     chunk_id=args.chunk_id,
    #     base_path=args.base_path,
    #     num_threads=args.num_threads,
    # )
    save_dataset_images_to_ceph_chunk(
        dataset,
        max_img_num=args.max_img_num,
        num_chunks=args.num_chunks,
        chunk_id=args.chunk_id,
        base_path=args.base_path,
        num_threads=args.num_threads,
    )
