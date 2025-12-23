import sys
sys.path.insert(0, ".")

from PIL import Image
import torch
from data.transforms import (
    JointCompose, JointResize, JointRandomHorizontalFlip,
    JointRandomRotation, JointRandomResizedCrop, ColorJitter,
    create_train_transforms, create_val_transforms
)
from data.oxford_pets import OxfordPetsDataset, create_dataloaders
from config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR


def create_dummy_image_and_mask():
    image = Image.new("RGB", (300, 200), color=(100, 150, 200))
    mask = Image.new("L", (300, 200), color=1)
    return image, mask


def test_joint_resize():
    print("Testing JointResize...")
    image, mask = create_dummy_image_and_mask()
    transform = JointResize(224)
    img_out, mask_out = transform(image, mask)
    assert img_out.size == (224, 224), f"Expected (224, 224), got {img_out.size}"
    assert mask_out.size == (224, 224), f"Expected (224, 224), got {mask_out.size}"
    print("  ✓ JointResize works correctly")


def test_joint_flip():
    print("Testing JointRandomHorizontalFlip...")
    image, mask = create_dummy_image_and_mask()
    transform = JointRandomHorizontalFlip(p=1.0)
    img_out, mask_out = transform(image, mask)
    assert img_out.size == image.size
    assert mask_out.size == mask.size
    print("  ✓ JointRandomHorizontalFlip works correctly")


def test_joint_rotation():
    print("Testing JointRandomRotation...")
    image, mask = create_dummy_image_and_mask()
    transform = JointRandomRotation(degrees=15)
    img_out, mask_out = transform(image, mask)
    assert img_out.size == image.size
    assert mask_out.size == mask.size
    print("  ✓ JointRandomRotation works correctly")


def test_joint_resized_crop():
    print("Testing JointRandomResizedCrop...")
    image, mask = create_dummy_image_and_mask()
    transform = JointRandomResizedCrop(224, scale=(0.8, 1.0))
    img_out, mask_out = transform(image, mask)
    assert img_out.size == (224, 224), f"Expected (224, 224), got {img_out.size}"
    assert mask_out.size == (224, 224), f"Expected (224, 224), got {mask_out.size}"
    print("  ✓ JointRandomResizedCrop works correctly")


def test_color_jitter():
    print("Testing ColorJitter...")
    image, mask = create_dummy_image_and_mask()
    transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    img_out, mask_out = transform(image, mask)
    assert img_out.size == image.size
    assert mask_out.size == mask.size
    print("  ✓ ColorJitter works correctly")


def test_train_transforms():
    print("Testing create_train_transforms...")
    image, mask = create_dummy_image_and_mask()
    transform = create_train_transforms(224)
    img_out, mask_out = transform(image, mask)
    assert img_out.size == (224, 224)
    assert mask_out.size == (224, 224)
    print("  ✓ Train transforms pipeline works correctly")


def test_val_transforms():
    print("Testing create_val_transforms...")
    image, mask = create_dummy_image_and_mask()
    transform = create_val_transforms(224)
    img_out, mask_out = transform(image, mask)
    assert img_out.size == (224, 224)
    assert mask_out.size == (224, 224)
    print("  ✓ Val transforms pipeline works correctly")


def test_dataset_with_transforms():
    print("\nTesting OxfordPetsDataset with transforms...")
    train_loader, val_loader = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)

    images, masks = next(iter(train_loader))
    print(f"  Train batch - images: {images.shape}, masks: {masks.shape}")
    assert images.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert masks.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)

    images, masks = next(iter(val_loader))
    print(f"  Val batch - images: {images.shape}, masks: {masks.shape}")
    print("  ✓ Dataset with transforms works correctly")


def test_augmentation_variability():
    print("\nTesting augmentation variability...")
    dataset = OxfordPetsDataset(DATA_DIR, "train", IMAGE_SIZE, transform=create_train_transforms(IMAGE_SIZE))

    img1, mask1 = dataset[0]
    img2, mask2 = dataset[0]

    are_different = not torch.allclose(img1, img2, atol=1e-5)
    print(f"  Same sample, different augmentations: {are_different}")
    if are_different:
        print("  ✓ Augmentations produce different results each time")


def main():
    print("=" * 50)
    print("Transform Tests")
    print("=" * 50)

    test_joint_resize()
    test_joint_flip()
    test_joint_rotation()
    test_joint_resized_crop()
    test_color_jitter()
    test_train_transforms()
    test_val_transforms()
    test_dataset_with_transforms()
    test_augmentation_variability()

    print("\n" + "=" * 50)
    print("All transform tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

