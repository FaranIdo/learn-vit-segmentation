import sys
sys.path.insert(0, ".")

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from data.oxford_pets import create_test_loader
from config import DATA_DIR, IMAGE_SIZE
from visualization.predictions import denormalize_image


def create_test_loader_single_worker(root, image_size, batch_size):
    from torchvision import transforms
    from data.oxford_pets import OxfordIIITPet
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())
    ])
    
    dataset = OxfordIIITPet(root=root, split="test", transform=transform, target_transform=target_transform, download=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return loader


print("Loading a sample from the test set...")
test_loader = create_test_loader_single_worker(root=DATA_DIR, image_size=IMAGE_SIZE, batch_size=1)

for images, masks in test_loader:
    image = images[0]
    mask = masks[0].cpu().numpy()
    
    if mask.sum() > 1000 and mask.sum() < 20000:
        break

print(f"Selected mask with {mask.sum()} pet pixels")

mask_binary = mask > 0

thickness = 3

dilated_steps = [mask_binary.copy()]
for i in range(1, thickness + 1):
    dilated = binary_dilation(mask_binary, iterations=i)
    dilated_steps.append(dilated)

eroded_steps = [mask_binary.copy()]
for i in range(1, thickness + 1):
    eroded = binary_erosion(mask_binary, iterations=i)
    eroded_steps.append(eroded)

boundary = dilated_steps[-1] & ~eroded_steps[-1]
interior = eroded_steps[-1]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

img = denormalize_image(image)
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_binary, cmap="gray")
axes[0, 1].set_title(f"Original Mask\n({mask_binary.sum():,} pixels)")
axes[0, 1].axis("off")

axes[0, 2].imshow(dilated_steps[1], cmap="gray")
axes[0, 2].set_title(f"Dilated (1 iter)\n({dilated_steps[1].sum():,} pixels)")
axes[0, 2].axis("off")

axes[0, 3].imshow(dilated_steps[-1], cmap="gray")
axes[0, 3].set_title(f"Dilated ({thickness} iters)\n({dilated_steps[-1].sum():,} pixels)")
axes[0, 3].axis("off")

axes[1, 0].imshow(eroded_steps[1], cmap="gray")
axes[1, 0].set_title(f"Eroded (1 iter)\n({eroded_steps[1].sum():,} pixels)")
axes[1, 0].axis("off")

axes[1, 1].imshow(eroded_steps[-1], cmap="gray")
axes[1, 1].set_title(f"Eroded ({thickness} iters)\n({eroded_steps[-1].sum():,} pixels)")
axes[1, 1].axis("off")

axes[1, 2].imshow(boundary, cmap="Reds")
axes[1, 2].set_title(f"Boundary Region\n({boundary.sum():,} pixels)")
axes[1, 2].axis("off")

axes[1, 3].imshow(interior, cmap="Greens")
axes[1, 3].set_title(f"Interior Region\n({interior.sum():,} pixels)")
axes[1, 3].axis("off")

plt.suptitle(f"Boundary Extraction Process (thickness={thickness} pixels)", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("outputs/boundary_extraction_process.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/boundary_extraction_process.png")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask_binary, cmap="gray")
axes[1].set_title(f"Ground Truth Mask\n({mask_binary.sum():,} pixels)")
axes[1].axis("off")

axes[2].imshow(boundary, cmap="Reds")
axes[2].set_title(f"Boundary ({thickness}px thick)\n({boundary.sum():,} pixels)")
axes[2].axis("off")

axes[3].imshow(interior, cmap="Greens")
axes[3].set_title(f"Interior\n({interior.sum():,} pixels)")
axes[3].axis("off")

plt.suptitle("Final Boundary vs Interior Regions", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/boundary_extraction_final.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/boundary_extraction_final.png")

print("\nSummary:")
print(f"  Original mask: {mask_binary.sum():,} pixels")
print(f"  Boundary region: {boundary.sum():,} pixels ({100 * boundary.sum() / mask_binary.sum():.1f}%)")
print(f"  Interior region: {interior.sum():,} pixels ({100 * interior.sum() / mask_binary.sum():.1f}%)")
print(f"  Boundary thickness: {thickness} pixels")

