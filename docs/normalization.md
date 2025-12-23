# Normalization in Transformers

## Why Normalize?

Deep networks suffer from unstable activations - values can explode or vanish as data flows through layers. Normalization keeps values in a stable range.

## LayerNorm vs BatchNorm

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes across | Batch dimension | Feature dimension |
| For shape [B, 196, 384] | Stats from B samples | Stats from 384 features |
| Works with batch=1? | No | Yes |
| Used in | CNNs | Transformers |

Transformers use LayerNorm because:
- Works with any batch size
- Each token normalized independently
- No running statistics needed

## How LayerNorm Works

For each token (384 values):

"""python
mean = token.mean()
std = token.std()
normalized = (token - mean) / std
output = gamma * normalized + beta  # learnable scale/shift
"""

Input: `[2.5, -0.3, 100.0, ...]` (wild range)
Output: `[0.02, -0.01, 1.2, ...]` (mean≈0, std≈1)

## Pre-LN vs Post-LN

### Post-LN (Original Transformer 2017)

"""python
x = self.norm1(x + self.attn(x))  # norm AFTER
x = self.norm2(x + self.mlp(x))
"""

- Norm applied after residual addition
- Can have training instability with deep networks
- Needs careful learning rate warmup

### Pre-LN (What We Use)

"""python
x = x + self.attn(self.norm1(x))  # norm BEFORE
x = x + self.mlp(self.norm2(x))
"""

- Norm applied before attention/MLP
- More stable training
- Used by GPT-2, ViT, most modern transformers

## Why Final LayerNorm in Encoder?

With Pre-LN, each block's output is NOT normalized (it's residual + transformed value). After 6 blocks, values can drift.

"""python
def forward(self, x):
    x = self.patch_embed(x)
    for block in self.blocks:
        x = block(x)
    x = self.norm(x)  # Clean up before decoder
    return x
"""

The final LayerNorm:
- Normalizes accumulated residuals
- Gives decoder consistent input distribution
- Acts as stabilization at encoder-decoder boundary

## Pre-LN Stability

Why Pre-LN trains more stably:

1. **Gradient flow**: Residual path has clean gradient (just addition)
2. **Scale consistency**: Normalized values enter attention/MLP
3. **No accumulation**: Each sub-layer receives normalized input

Post-LN can accumulate large values through residuals before normalization, causing gradient issues in deep networks.

