import torch
import random
from torchvision.transforms import functional as TF
from PIL import Image


class JointCompose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class JointResize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class JointRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class JointRandomRotation:
    def __init__(self, degrees: float = 10.0):
        self.degrees = degrees

    def __call__(self, image: Image.Image, mask: Image.Image):
        angle = random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class JointRandomResizedCrop:
    def __init__(self, size: int, scale: tuple = (0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, image: Image.Image, mask: Image.Image):
        i, j, h, w = self._get_params(image)
        image = TF.resized_crop(image, i, j, h, w, [self.size, self.size])
        mask = TF.resized_crop(mask, i, j, h, w, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def _get_params(self, image: Image.Image):
        width, height = image.size
        area = height * width
        scale = random.uniform(self.scale[0], self.scale[1])
        target_area = area * scale
        aspect_ratio = width / height

        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))

        w = min(w, width)
        h = min(h, height)

        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        return i, j, h, w


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.adjust_brightness(image, 1 + random.uniform(-self.brightness, self.brightness))
        image = TF.adjust_contrast(image, 1 + random.uniform(-self.contrast, self.contrast))
        image = TF.adjust_saturation(image, 1 + random.uniform(-self.saturation, self.saturation))
        return image, mask


def create_train_transforms(image_size: int):
    return JointCompose([
        JointRandomResizedCrop(image_size, scale=(0.8, 1.0)),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomRotation(degrees=10),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])


def create_val_transforms(image_size: int):
    return JointCompose([
        JointResize(image_size),
    ])

