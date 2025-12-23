"""Explore positional embeddings - what do they look like?"""

import sys
sys.path.insert(0, ".")

import torch
from vit.patch_embed import PatchEmbedding
from config import IMAGE_SIZE, PATCH_SIZE, EMBED_DIM


def explore_positional_embeddings():
    patch_embed = PatchEmbedding(IMAGE_SIZE, PATCH_SIZE, EMBED_DIM)
    
    pos = patch_embed.pos_embed
    print(f"Positional embedding shape: {pos.shape}")
    print(f"  - 1 = batch dimension (shared across all images)")
    print(f"  - {pos.shape[1]} = number of positions (patches)")
    print(f"  - {pos.shape[2]} = embedding dimension")
    
    print(f"\n--- Initial values (before training) ---")
    print(f"Position 0 (first 10 values): {pos[0, 0, :10].tolist()}")
    print(f"Position 1 (first 10 values): {pos[0, 1, :10].tolist()}")
    print("(All zeros because we initialized with zeros!)")
    
    print(f"\n--- Simulating trained positional embeddings ---")
    torch.manual_seed(42)
    trained_pos = torch.randn_like(pos) * 0.02
    
    print(f"Position 0 (top-left):     {trained_pos[0, 0, :5].tolist()}")
    print(f"Position 1 (next to it):   {trained_pos[0, 1, :5].tolist()}")
    print(f"Position 14 (row below):   {trained_pos[0, 14, :5].tolist()}")
    print(f"Position 195 (bottom-right): {trained_pos[0, 195, :5].tolist()}")
    
    print(f"\n--- Similarity between positions ---")
    pos_0 = trained_pos[0, 0]
    pos_1 = trained_pos[0, 1]
    pos_14 = trained_pos[0, 14]
    pos_195 = trained_pos[0, 195]
    
    sim_0_1 = torch.cosine_similarity(pos_0, pos_1, dim=0).item()
    sim_0_14 = torch.cosine_similarity(pos_0, pos_14, dim=0).item()
    sim_0_195 = torch.cosine_similarity(pos_0, pos_195, dim=0).item()
    
    print(f"Cosine similarity (0 ↔ 1, neighbors):    {sim_0_1:.4f}")
    print(f"Cosine similarity (0 ↔ 14, one row down): {sim_0_14:.4f}")
    print(f"Cosine similarity (0 ↔ 195, opposite corner): {sim_0_195:.4f}")
    print("\n(After training, nearby positions often have similar embeddings)")


if __name__ == "__main__":
    explore_positional_embeddings()

