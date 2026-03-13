"""CIPNet: neural network approximation for CIP solutions."""

import torch
from torch import nn
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CIPEmbedLayer(nn.Module):
    def __init__(self, num_neighbors, point_dim, embed_dim):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.point_dim = point_dim
        self.embed_dim = embed_dim

        self.base_embed = nn.Linear(point_dim, embed_dim)

        self.neighbor_embed = nn.Sequential(
            nn.Linear(point_dim, embed_dim),
        )

        self.dist_encoding = nn.Parameter(torch.randn(num_neighbors, embed_dim))

    def forward(self, base_points, neighbors):
        base_embed = self.base_embed(base_points).unsqueeze(1)

        neighbor_embed = self.neighbor_embed(neighbors.view(-1, self.point_dim))
        neighbor_embed = neighbor_embed.view(-1, self.num_neighbors, self.embed_dim)

        x = torch.cat((base_embed, neighbor_embed), dim=1)

        x[:, 1:, :] += self.dist_encoding

        return x

class Classifier(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x

class CIPNet(nn.Module):
    def __init__(self, num_neighbors, point_dim, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0.):
        super().__init__()

        self.embedding = CIPEmbedLayer(num_neighbors, point_dim, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.classifier = Classifier(dim, point_dim)

    def forward(self, base_points, neighbors):
        x = self.embedding(base_points, neighbors)
        x = self.transformer(x)

        x = self.classifier(x)
        return x
