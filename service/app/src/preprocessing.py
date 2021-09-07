import io
import PIL.Image

from fastapi import File

from torch import stack
from torchvision.transforms import ToTensor

def file_feature(file: bytes = File(...)):
    image = PIL.Image.open(io.BytesIO(file))
    return stack([ToTensor()(image)])