# Future Work

Potential improvements and experiments to try after completing the tutorial.

## Positional Encoding

### RoPE (Rotary Position Embedding)

Replace learnable positional embeddings with RoPE for:
- Zero additional parameters
- Built-in relative position awareness
- Potential to handle variable image sizes

**Implementation idea**: Apply rotation to query and key vectors in attention.

### 2D Positional Embeddings

Separate row and column embeddings instead of flattened 1D:
- 14 row embeddings + 14 column embeddings = 28 vectors
- More explicit 2D structure
- Fewer parameters (28 vs 196)

## Architecture Improvements

### Swin Transformer

- Hierarchical feature maps (like CNN)
- Shifted window attention for efficiency
- Better for dense prediction tasks like segmentation

### Multi-Scale Features

- Extract features from multiple transformer layers
- Combine early (fine detail) and late (semantic) features
- Common in segmentation models like SegFormer

### SETR (Segmentation Transformer)

Use skip connections from intermediate ViT layers:

"""python
# Modify encoder to return intermediate outputs
intermediates = []
for i, block in enumerate(self.blocks):
    x = block(x)
    if i in [2, 4]:  # Save layers 2 and 4
        intermediates.append(x)
"""

Benefits:
- Skip connections without changing ViT architecture
- Earlier layers have more local information
- Later layers have more semantic information
- All layers are 14×14, but fused during upsampling

### U-Net Style Decoder for ViT

Options to add skip connections:
1. **Intermediate layer skips**: Use blocks 2, 4, 6 outputs
2. **Fake pyramid**: Upsample 14×14 to multiple scales
3. **Hierarchical ViT**: Use Swin or SegFormer (true multi-scale)

### Relative Position Bias

- Add learnable bias based on relative position in attention
- Used by Swin Transformer
- Better captures local vs global relationships

## Training Improvements

### Data Augmentation

- MixUp, CutMix for regularization
- RandAugment for automatic augmentation policy
- Multi-scale training

### Knowledge Distillation

- Train small ViT using predictions from large pretrained model
- DeiT approach: distillation token

### Self-Supervised Pretraining

- MAE (Masked Autoencoder): Mask patches, reconstruct them
- DINO: Self-distillation without labels
- Can improve downstream task performance

## Efficiency

### Token Pruning

- Remove uninformative patches during inference
- Reduce computation for easy images

### Quantization

- INT8 or lower precision for faster inference
- Especially useful for deployment

### Flash Attention

- Memory-efficient attention implementation
- Faster training and inference

