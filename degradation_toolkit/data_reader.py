import logging
import time
from io import BytesIO
from typing import Union

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


def read_general(path) -> Union[str, BytesIO]:
    if "s3://" in path:
        init_ceph_client_if_needed()
        file_bytes = BytesIO(client.get(path))
        return file_bytes
    else:
        return path

def read_img_general(img_path):
    if "s3://" in img_path:
        init_ceph_client_if_needed()
        img_bytes = client.get(img_path)

        return img_bytes
    else:
        return open('/'+img_path, "rb").read()

def read_img_ceph(img_path):
    init_ceph_client_if_needed()
    img_bytes = client.get(img_path)
    image = Image.open(BytesIO(img_bytes)).convert('RGB')

    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)

    return image_without_exif


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa

        client = Client("../petreloss.conf")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")


client = None
