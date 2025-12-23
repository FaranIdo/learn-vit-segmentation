#!/bin/bash

# Dataset Setup Script for ViT Segmentation Tutorial
# Downloads and prepares the Oxford-IIIT Pets dataset

set -e  # Exit on error

echo "========================================"
echo "Oxford-IIIT Pets Dataset Setup"
echo "========================================"

# Configuration
DATA_DIR="./data/oxford-iiit-pet"
IMAGES_URL="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

# Create data directory
echo ""
echo "1. Creating data directory..."
mkdir -p "$DATA_DIR"
echo "   ✓ Created $DATA_DIR"

# Download images
echo ""
echo "2. Downloading images (755 MB)..."
if [ -f "$DATA_DIR/images.tar.gz" ]; then
    echo "   ⊙ images.tar.gz already exists, skipping download"
else
    curl -k -L -o "$DATA_DIR/images.tar.gz" "$IMAGES_URL"
    echo "   ✓ Downloaded images.tar.gz"
fi

# Download annotations
echo ""
echo "3. Downloading annotations (20 MB)..."
if [ -f "$DATA_DIR/annotations.tar.gz" ]; then
    echo "   ⊙ annotations.tar.gz already exists, skipping download"
else
    curl -k -L -o "$DATA_DIR/annotations.tar.gz" "$ANNOTATIONS_URL"
    echo "   ✓ Downloaded annotations.tar.gz"
fi

# Extract images
echo ""
echo "4. Extracting images..."
if [ -d "$DATA_DIR/images" ]; then
    echo "   ⊙ images/ already exists, skipping extraction"
else
    tar -xzf "$DATA_DIR/images.tar.gz" -C "$DATA_DIR"
    echo "   ✓ Extracted images/"
fi

# Extract annotations
echo ""
echo "5. Extracting annotations..."
if [ -d "$DATA_DIR/annotations" ]; then
    echo "   ⊙ annotations/ already exists, skipping extraction"
else
    tar -xzf "$DATA_DIR/annotations.tar.gz" -C "$DATA_DIR"
    echo "   ✓ Extracted annotations/"
fi

# Count files
echo ""
echo "6. Verifying dataset..."
IMAGE_COUNT=$(find "$DATA_DIR/images" -type f -name "*.jpg" | wc -l | tr -d ' ')
MASK_COUNT=$(find "$DATA_DIR/annotations/trimaps" -type f -name "*.png" | wc -l | tr -d ' ')

echo "   ✓ Found $IMAGE_COUNT images"
echo "   ✓ Found $MASK_COUNT masks"

if [ "$IMAGE_COUNT" -lt 7000 ] || [ "$MASK_COUNT" -lt 7000 ]; then
    echo ""
    echo "   ⚠ Warning: Expected ~7,390 images and masks"
    echo "   ⚠ Dataset may be incomplete"
else
    echo "   ✓ Dataset verification passed"
fi

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo "Dataset location: $DATA_DIR"
echo ""
echo "Directory structure:"
echo "  $DATA_DIR/"
echo "  ├── images/           # Pet images"
echo "  ├── annotations/"
echo "  │   └── trimaps/     # Segmentation masks"
echo "  ├── images.tar.gz"
echo "  └── annotations.tar.gz"
echo ""
echo "Next steps:"
echo "  1. Activate your virtual environment: source venv/bin/activate"
echo "  2. Verify setup: python examples/sanity_check.py"
echo "  3. Start training: python train.py"
echo ""
echo "========================================"

