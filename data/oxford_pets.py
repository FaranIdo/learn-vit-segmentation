import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
from data.transforms import create_train_transforms, create_val_transforms
from config import IMAGENET_MEAN, IMAGENET_STD


class OxfordPetsDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int, transform=None):
        self.image_size = image_size
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        self.transform = transform
        self.samples = self._load_split(root, split)

    def _load_split(self, root: str, split: str) -> list[str]:
        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root, "annotations", split_file)
        with open(split_path, "r") as f:
            lines = f.readlines()
        return [line.strip().split()[0] for line in lines]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_name = self.samples[idx]
        image = self._load_raw_image(sample_name)
        mask = self._load_raw_mask(sample_name)

        if self.transform:
            image, mask = self.transform(image, mask)

        image = self._finalize_image(image)
        mask = self._finalize_mask(mask)
        return image, mask

    def _load_raw_image(self, name: str) -> Image.Image:
        path = os.path.join(self.images_dir, f"{name}.jpg")
        return Image.open(path).convert("RGB")

    def _load_raw_mask(self, name: str) -> Image.Image:
        path = os.path.join(self.masks_dir, f"{name}.png")
        return Image.open(path)

    def _finalize_image(self, pil_image: Image.Image) -> torch.Tensor:
        resized = TF.resize(pil_image, [self.image_size, self.image_size])
        tensor = TF.to_tensor(resized)
        normalized = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return normalized

    def _finalize_mask(self, pil_mask: Image.Image) -> torch.Tensor:
        resized = TF.resize(pil_mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        tensor = torch.tensor(list(resized.getdata()), dtype=torch.long)
        reshaped = tensor.reshape(self.image_size, self.image_size)
        return self._convert_to_binary(reshaped)

    def _convert_to_binary(self, mask: torch.Tensor) -> torch.Tensor:
        binary = torch.zeros_like(mask)
        binary[mask == 1] = 1
        return binary


def create_dataloaders(root: str, image_size: int, batch_size: int):
    train_transform = create_train_transforms(image_size)
    val_transform = create_val_transforms(image_size)

    train_dataset = OxfordPetsDataset(root, "train", image_size, transform=train_transform)
    val_dataset = OxfordPetsDataset(root, "test", image_size, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader
