import PIL.Image
from torch import Tensor, uint8
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage, ToTensor


def to_uint8(tensor):
    scaled = tensor * 255
    return scaled.to(uint8)


def image_to_tensor(file):
    return ToTensor()(PIL.Image.open(file))


def list_to_tensor(l):
    return Tensor(l)


def bounding_box(x, y):
    bounded_image = draw_bounding_boxes(to_uint8(x), y)
    return ToPILImage()(bounded_image)