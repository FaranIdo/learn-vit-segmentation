import sys

sys.path.insert(0, "/Users/ido.faran/personal/vis-tutorial")

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from data.oxford_pets import OxfordPetsDataset
from config import DATA_DIR, IMAGE_SIZE
from visualization.predictions import denormalize_image
from torch.utils.data import DataLoader

dataset = OxfordPetsDataset(root=DATA_DIR, split="test", image_size=IMAGE_SIZE)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for images, masks in test_loader:
    image = images[0]
    mask = masks[0].cpu().numpy()

    if mask.sum() > 1000 and mask.sum() < 20000:
        break

mask_binary = mask > 0

dilated_1 = binary_dilation(mask_binary, iterations=1)
dilated_2 = binary_dilation(mask_binary, iterations=2)
dilated_3 = binary_dilation(mask_binary, iterations=3)

eroded_1 = binary_erosion(mask_binary, iterations=1)
eroded_2 = binary_erosion(mask_binary, iterations=2)
eroded_3 = binary_erosion(mask_binary, iterations=3)

boundary = dilated_3 & ~eroded_3
interior = eroded_3

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

img = denormalize_image(image)
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_binary, cmap="gray")
axes[0, 1].set_title("Ground Truth Mask\n(Starting Point)", fontsize=12, fontweight="bold")
axes[0, 1].axis("off")

axes[0, 2].imshow(mask_binary, cmap="gray")
axes[0, 2].set_title("Pixels: " + str(int(mask_binary.sum())), fontsize=10)
axes[0, 2].axis("off")

axes[0, 3].axis("off")

axes[1, 0].imshow(dilated_1, cmap="Greens")
axes[1, 0].set_title("Dilation: Iteration 1\n(Expand by 1 pixel)", fontsize=11)
axes[1, 0].axis("off")

axes[1, 1].imshow(dilated_2, cmap="Greens")
axes[1, 1].set_title("Dilation: Iteration 2\n(Expand by 2 pixels)", fontsize=11)
axes[1, 1].axis("off")

axes[1, 2].imshow(dilated_3, cmap="Greens")
axes[1, 2].set_title("Dilation: Iteration 3\n(Expand by 3 pixels)", fontsize=11, fontweight="bold")
axes[1, 2].axis("off")

axes[1, 3].imshow(dilated_3 & ~mask_binary, cmap="Greens")
axes[1, 3].set_title("Added pixels\n(green = new)", fontsize=10)
axes[1, 3].axis("off")

axes[2, 0].imshow(eroded_1, cmap="Reds")
axes[2, 0].set_title("Erosion: Iteration 1\n(Shrink by 1 pixel)", fontsize=11)
axes[2, 0].axis("off")

axes[2, 1].imshow(eroded_2, cmap="Reds")
axes[2, 1].set_title("Erosion: Iteration 2\n(Shrink by 2 pixels)", fontsize=11)
axes[2, 1].axis("off")

axes[2, 2].imshow(eroded_3, cmap="Reds")
axes[2, 2].set_title("Erosion: Iteration 3\n(Shrink by 3 pixels)", fontsize=11, fontweight="bold")
axes[2, 2].axis("off")

axes[2, 3].imshow(mask_binary & ~eroded_3, cmap="Reds")
axes[2, 3].set_title("Removed pixels\n(red = removed)", fontsize=10)
axes[2, 3].axis("off")

fig.suptitle("Morphological Operations: Dilation & Erosion Process", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("outputs/boundary_extraction_process.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/boundary_extraction_process.png")
plt.close()

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes[0].imshow(img)
axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(mask_binary, cmap="gray")
axes[1].set_title(f"Ground Truth\n({int(mask_binary.sum())} pixels)", fontsize=11)
axes[1].axis("off")

axes[2].imshow(dilated_3, cmap="Greens", alpha=0.7)
axes[2].imshow(img, alpha=0.3)
axes[2].set_title(f"Dilated (expand 3px)\n({int(dilated_3.sum())} pixels)", fontsize=11)
axes[2].axis("off")

axes[3].imshow(eroded_3, cmap="Reds", alpha=0.7)
axes[3].imshow(img, alpha=0.3)
axes[3].set_title(f"Eroded (shrink 3px)\n({int(eroded_3.sum())} pixels)", fontsize=11)
axes[3].axis("off")

result_map = np.zeros((*mask_binary.shape, 3), dtype=np.uint8)
result_map[boundary] = [255, 255, 0]
result_map[interior] = [0, 0, 255]

axes[4].imshow(result_map)
axes[4].set_title(f"Final Result\nYellow=Boundary ({int(boundary.sum())} px)\nBlue=Interior ({int(interior.sum())} px)", fontsize=11, fontweight="bold")
axes[4].axis("off")

fig.suptitle("Final Boundary Extraction: boundary = dilated & ~eroded, interior = eroded", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/boundary_extraction_final.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/boundary_extraction_final.png")
plt.close()

print("\nSummary:")
print(f"  Original mask pixels: {int(mask_binary.sum())}")
print(f"  After dilation (expand 3px): {int(dilated_3.sum())} pixels (+{int(dilated_3.sum() - mask_binary.sum())})")
print(f"  After erosion (shrink 3px): {int(eroded_3.sum())} pixels (-{int(mask_binary.sum() - eroded_3.sum())})")
print(f"  Boundary pixels: {int(boundary.sum())}")
print(f"  Interior pixels: {int(interior.sum())}")
print(f"  Boundary ratio: {boundary.sum() / mask_binary.sum() * 100:.1f}% of original mask")
