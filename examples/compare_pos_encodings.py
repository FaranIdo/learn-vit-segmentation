"""Compare learnable vs sinusoidal positional encodings."""

import sys

sys.path.insert(0, ".")

import torch
import math


def create_sinusoidal_encoding(num_positions: int, embed_dim: int) -> torch.Tensor:
    """Original Transformer positional encoding using sin/cos."""
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

    pe = torch.zeros(num_positions, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos

    return pe


def main():
    num_positions = 196
    embed_dim = 384

    sinusoidal = create_sinusoidal_encoding(num_positions, embed_dim)
    learnable = torch.zeros(num_positions, embed_dim)

    print("=== Sinusoidal Positional Encoding ===")
    print(f"Shape: {sinusoidal.shape}")
    print(f"\nPosition 0 (first 8 values):")
    print(f"  {sinusoidal[0, :8].tolist()}")
    print(f"\nPosition 1 (first 8 values):")
    print(f"  {sinusoidal[1, :8].tolist()}")
    print(f"\nPosition 195 (first 8 values):")
    print(f"  {sinusoidal[195, :8].tolist()}")

    print("\n=== Key Properties ===")
    print("- Values are between -1 and 1 (sin/cos range)")
    print("- Each position has a unique pattern")
    print("- Nearby positions have similar patterns")

    sim_0_1 = torch.cosine_similarity(sinusoidal[0], sinusoidal[1], dim=0).item()
    sim_0_10 = torch.cosine_similarity(sinusoidal[0], sinusoidal[10], dim=0).item()
    sim_0_195 = torch.cosine_similarity(sinusoidal[0], sinusoidal[195], dim=0).item()

    print(f"\nCosine similarity (sinusoidal):")
    print(f"  Position 0 ↔ 1:   {sim_0_1:.4f} (neighbors)")
    print(f"  Position 0 ↔ 10:  {sim_0_10:.4f} (10 apart)")
    print(f"  Position 0 ↔ 195: {sim_0_195:.4f} (opposite ends)")

    print("\n=== Why ViT Uses Learnable Instead ===")
    print("1. Images have fixed size - no need to extrapolate")
    print("2. 2D spatial structure is different from 1D text sequence")
    print("3. Learnable embeddings can capture image-specific patterns")
    print("4. Slightly better performance in practice")


if __name__ == "__main__":
    main()
