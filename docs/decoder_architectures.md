# Decoder Architectures for Segmentation

## The Challenge

Segmentation requires per-pixel predictions. The encoder compresses spatial information:

- Input: 224×224 pixels
- Encoder output: 14×14 tokens (16× smaller)

The decoder must upsample back to full resolution.

## Our Approach: Progressive Decoder

We upsample in 4 steps, each doubling spatial size:

"""
14×14 → 28×28 → 56×56 → 112×112 → 224×224
"""

Each step:
1. Bilinear interpolation (2× upsample)
2. Convolution (refine features)
3. Reduce channels (384 → 192 → 96 → 48 → num_classes)

**Pros**:
- Simple and stable
- No checkerboard artifacts
- Works well with ViT's single-scale output

**Cons**:
- Must reconstruct details from compressed features
- No direct access to high-resolution information

### Reshaping Tokens to Spatial Grid

The encoder outputs tokens in sequence format. Convolutions need spatial format:

"""
Encoder output: [B, 196, 384]   (sequence: 196 tokens, 384 features each)
Conv2d expects: [B, 384, 14, 14] (spatial: 384 channels, 14×14 grid)
"""

The reshape operation:

"""python
x = x.transpose(1, 2)              # [B, 196, 384] → [B, 384, 196]
x = x.reshape(B, 384, 14, 14)      # [B, 384, 196] → [B, 384, 14, 14]
"""

This "unfolds" the 196 tokens back into their original 14×14 spatial arrangement.

### The Spatial-Channel Trade-off

As spatial size increases, channels decrease:

"""
Step      Spatial     Channels         Values per image
──────────────────────────────────────────────────────────
Input     14×14       384              75,264
Step 1    28×28       192              150,528
Step 2    56×56       96               301,056
Step 3    112×112     48               602,112
Output    224×224     num_classes      224×224×num_classes
"""

The final layer outputs `num_classes` channels (not always 3):
- Oxford Pets: 3 classes (background, foreground, border)
- Cityscapes: 19 classes (road, car, person, etc.)
- ADE20K: 150 classes
- Binary segmentation: 1 or 2 classes

**Why decrease channels?**

1. **Memory efficiency**: 384 channels at 224×224 = 19.3M values. Much smaller with fewer output classes.

2. **Semantic to spatial**: Early layers encode rich semantics ("cat ear", "dog nose"). Final layer only needs class logits per pixel.

3. **Convention**: Same pattern as CNN decoders (U-Net, ResNet). Encoder compresses space and expands channels. Decoder does the opposite.

"""
Encoder: compress space, expand channels (capture meaning)
Decoder: expand space, compress channels (localize predictions)
"""

## U-Net: Skip Connections

U-Net uses a symmetric encoder-decoder with skip connections:

"""
Encoder                    Decoder
224×224 ──────────────────→ 224×224
   ↓                           ↑
112×112 ──────────────────→ 112×112
   ↓                           ↑
56×56   ──────────────────→ 56×56
   ↓                           ↑
28×28   ──────────────────→ 28×28
   ↓                           ↑
14×14 (bottleneck) ───────────┘
"""

Skip connections concatenate encoder features directly to decoder:

"""python
decoder_features = upsample(bottleneck)
decoder_features = concat(decoder_features, encoder_features)  # Skip!
decoder_features = conv(decoder_features)
"""

**Pros**:
- Preserves fine details (edges, textures)
- Excellent for medical imaging
- Gradients flow easily through skip connections

**Cons**:
- Requires hierarchical encoder (not standard ViT)
- More memory (storing encoder features)
- More complex architecture

## Transposed Convolution

Alternative to bilinear + conv upsampling:

"""python
# Transposed conv: single operation
x = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)(x)

# Our approach: two operations
x = F.interpolate(x, scale_factor=2, mode='bilinear')
x = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)(x)
"""

**Pros**:
- Learnable upsampling
- Single operation (faster)

**Cons**:
- Checkerboard artifacts from uneven overlap
- Harder to train

## SegFormer: Best of Both Worlds

SegFormer (Phase 5 of tutorial) combines:
- Hierarchical ViT encoder (multi-scale features)
- Lightweight MLP decoder (fuses all scales)

"""
Encoder produces 4 scales:
  1/4  resolution → 
  1/8  resolution → → MLP fusion → Upsample → Prediction
  1/16 resolution →
  1/32 resolution →
"""

This is the current state-of-the-art approach for ViT segmentation.

## Why We Use Progressive Decoder

For this tutorial:

1. **Standard ViT** - Our encoder produces single-scale output (14×14), so U-Net skip connections aren't possible without modifying the encoder.

2. **Simplicity** - Easy to understand and implement.

3. **Foundation** - Understanding this helps appreciate why SegFormer's hierarchical approach is an improvement.

## Comparison Table

| Approach | Skip Connections | Artifacts | Complexity | Best For |
|----------|-----------------|-----------|------------|----------|
| Progressive (ours) | No | None | Low | ViT single-scale |
| U-Net | Yes | None | Medium | CNNs, medical |
| Transposed Conv | Optional | Checkerboard | Low | Speed-critical |
| SegFormer | Implicit (fusion) | None | Medium | Production ViT |

