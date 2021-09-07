from torch import Tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

MODEL = fasterrcnn_resnet50_fpn(pretrained=True, progress=False).eval()

def infering(image: Tensor) -> dict:
    return MODEL(image)[0]


def thresholding(y: dict, p: float = .8) -> dict:
    mask = y["scores"] >= p
    return {k: y[k][mask].tolist() for k in y}