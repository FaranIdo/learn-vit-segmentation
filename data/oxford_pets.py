import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image


class OxfordPetsDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int):
        self.image_size = image_size
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "annotations", "trimaps")
        
        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root, "annotations", split_file)
        
        self.samples = self._load_split(split_path)
    
    def _load_split(self, split_path: str) -> list[str]:
        with open(split_path, "r") as f:
            lines = f.readlines()
        return [line.strip().split()[0] for line in lines]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_name = self.samples[idx]
        image = self._load_image(sample_name)
        mask = self._load_mask(sample_name)
        return image, mask
    
    def _load_image(self, name: str) -> torch.Tensor:
        path = os.path.join(self.images_dir, f"{name}.jpg")
        image = Image.open(path).convert("RGB")
        image = TF.resize(image, [self.image_size, self.image_size])
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image
    
    def _load_mask(self, name: str) -> torch.Tensor:
        path = os.path.join(self.masks_dir, f"{name}.png")
        mask = Image.open(path)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        mask = torch.tensor(list(mask.getdata()), dtype=torch.long)
        mask = mask.reshape(self.image_size, self.image_size)
        mask = self._convert_to_binary(mask)
        return mask
    
    def _convert_to_binary(self, mask: torch.Tensor) -> torch.Tensor:
        binary = torch.zeros_like(mask)
        binary[mask == 1] = 1  # Foreground (pet)
        return binary


def create_dataloaders(root: str, image_size: int, batch_size: int):
    train_dataset = OxfordPetsDataset(root, "train", image_size)
    val_dataset = OxfordPetsDataset(root, "test", image_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
