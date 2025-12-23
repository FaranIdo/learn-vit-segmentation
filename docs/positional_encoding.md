# Positional Encoding in Vision Transformers

## The Problem

Transformers process sequences but have no built-in notion of position. When we flatten a 14×14 grid of patches into 196 tokens, the model doesn't know their spatial arrangement.

## Three Approaches

### 1. Learnable Positional Embeddings (What We Use)

"""python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
"""

- Each position gets a learnable 384-dim vector
- Starts as zeros, learns structure during training
- Used by: ViT, BERT, GPT-2

**Pros**: Best performance for fixed-length sequences
**Cons**: Cannot extrapolate to longer sequences

### 2. Sinusoidal Positional Encoding (Original Transformer)

"""python
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
"""

- Fixed mathematical formula, no learning
- Nearby positions have similar encodings
- Used by: Original Transformer (2017)

**Pros**: Zero parameters, can extrapolate
**Cons**: Slightly worse performance than learnable

### 3. RoPE (Rotary Position Embedding)

- Rotates embedding dimensions based on position
- Relative position emerges naturally in attention computation
- Used by: LLaMA, Mistral, GPT-NeoX

**Pros**: Zero parameters, built-in relative position, can extrapolate
**Cons**: More complex attention implementation

## Comparison Table

| Method | Parameters | Relative Position | Extrapolation | Best For |
|--------|------------|-------------------|---------------|----------|
| Learnable | num_pos × embed_dim | Learned implicitly | No | Fixed-length (images) |
| Sinusoidal | 0 | Absolute only | Yes | Variable-length |
| RoPE | 0 | Built-in | Yes | Modern LLMs |

## Why We Chose Learnable

1. **Fixed image size**: 224×224 always produces 196 patches
2. **Best performance**: ViT paper showed learnable works best for vision
3. **Simplicity**: Just add a parameter tensor
4. **2D structure**: Model learns spatial relationships during training

## The 2D Problem

Flattening a 2D grid to 1D loses spatial information:

"""
2D grid:           1D sequence:
[0  1  2  3 ]      [0, 1, 2, 3, 4, 5, 6, 7, ...]
[4  5  6  7 ]
"""

Patch 3 and patch 4 are far apart in 2D (different rows) but adjacent in 1D.

**Solution options**:
- Separate row + column embeddings (28 vectors instead of 196)
- Let the model learn 2D structure from 1D embeddings (what we do)

The ViT authors tested both and found 1D learnable works just as well.

