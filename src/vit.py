import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.5):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        #self attention projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, attn_mask=None):
        #if not specified, default to self-attention
        if key is None: key = query
        if value is None: value = key

        B, N_q, D = query.shape
        _, N_kv, _ = key.shape

        #linear projection and reshape to (B, num_heads, seq_len, head_dim)
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        ## mask out the affinity matrix that shouldn't contribute to the training/inference
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        # softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #weighted aggregate of values
        out = torch.matmul(attn_weights, V)

        #concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        return self.out_proj(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout: float = 0.5):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim) #prelayer normalization to facilitate convergence
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #pre-LN self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        #pre-LN feed-forward network with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, Embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, dropout):
        super().__init__()

        #patch projection
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        #learnable token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        #transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        #classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        #weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x) #(B, Num_patches, embed_dim)

        #prepare cls token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        #add spatial position embeddings
        x = self.pos_drop(x + self.pos_embed)

        #pass through transformer encoders
        for block in self.blocks:
            x = block(x)

        #extract cls token representation & classify
        #the last token is the classification token. so as for sentence representation/generation
        x = self.norm(x)
        cls_rep = x[:, 0]
        logits = self.head(cls_rep)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images = torch.randn(2, 3, 224, 224).to(device)
    model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                               embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1).to(device)

    logits = model(images)
    print(f"input shape: {images.shape}")
    print(f"output shape: {logits.shape}") #should be (2, 1000)

    #verification of cross-attention module usage
    cross_attn = MultiHeadAttention(embed_dim=768, num_heads=12).to(device)
    query_seq = torch.randn(2, 10, 768).to(device)
    kv_seq = torch.randn(2, 197, 768).to(device)

    cross_out = cross_attn(query=query_seq, key=kv_seq, value=kv_seq)
    print(f"cross attention output shape: {cross_out.shape}") #should be (2, 10, 768)
