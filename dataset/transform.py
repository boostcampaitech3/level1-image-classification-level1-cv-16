import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import *

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class CutoutTransform:
    def __init__(self, augment, resize, cutout, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):    
        if augment:
            self.transform = transforms.Compose([
                CenterCrop(350),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Cutout(1, cutout),
                Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                CenterCrop(350),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])

    def __call__(self, image):
        return self.transform(image)


# 위와 같이 원하는 인자를 넣고, config에도 추가를 하면 custom하여 만들 수 있습니다!
class CustomTransform:
    def __init__(self, augment, resize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):    
        if augment:
            self.transform = transforms.Compose([
                CenterCrop(350),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                CenterCrop(350),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])

    def __call__(self, image):
        return self.transform(image)