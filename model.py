# model.py
import torch
from torch import nn
from einops import rearrange, repeat

# ---- Positional encodings ----
class PositionalEncoding(nn.Module):
    """1D sinusoidal positional encoding used for side (PPS) data."""
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / (10000 ** (2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
        sinusoid_table = torch.tensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)], dtype=torch.float32)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table.unsqueeze(0)
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncoding2d(nn.Module):
    """2D sinusoidal positional encoding for EEG arranged on a grid (e.g., 9x9)."""
    def __init__(self, d_hid, d_wid, n_position=200):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid, d_wid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid, d_wid):
        import numpy as np
        def get_position_angle_vec(position):
            return [position / (10000 ** (2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table_x = np.zeros_like(sinusoid_table)
        sinusoid_table_y = np.zeros_like(sinusoid_table)
        sinusoid_table_x[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table_x[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        sinusoid_table_y[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table_y[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        table = np.zeros((n_position, d_hid, d_wid))
        for i in range(d_hid):
            for j in range(d_wid):
                for k in range(n_position):
                    table[k][i][j] = sinusoid_table_x[k, i] * sinusoid_table_y[k, j]
        return torch.tensor(table, dtype=torch.float32).unsqueeze(0)
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

# ---- Blocks ----
class PreNorm(nn.Module):
    """LayerNorm applied before fn. Supports single or bi-modal input."""
    def __init__(self, dim, side_embedding_dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(side_embedding_dim)
        self.fn = fn
    def forward(self, inp, **kwargs):
        if isinstance(inp, (tuple, list)) and len(inp) == 2:
            x, y = inp
            return self.fn((self.norm1(x), self.norm2(y)), **kwargs)
        else:
            x = inp
            return self.fn(self.norm1(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Dropout(dropout)
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Cross_Attention(nn.Module):
    """TACO Cross Attention: jointly models temporal-spatial dependencies across modalities."""
    def __init__(self, dim, side_embedding_dim, heads, dim_head, side_dim_head, cross_heads, cross_dim_head, dropout=0.):
        super().__init__()
        inner_dim = cross_dim_head * cross_heads
        self.heads = cross_heads
        self.scale = (dim_head * side_dim_head) ** -0.5
        self.attend_chan = nn.Softmax(dim=-1)
        self.attend_token = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_side = nn.Linear(side_embedding_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Dropout(dropout)
    def forward(self, inp):
        x, y = inp
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        qs, ks, vs = self.to_qkv_side(y).chunk(3, dim=-1)
        q  = rearrange(q,  'b n (h d) -> b h n d', h=self.heads)
        k  = rearrange(k,  'b n (h d) -> b h n d', h=self.heads)
        v  = rearrange(v,  'b n (h d) -> b h n d', h=self.heads)
        qs = rearrange(qs, 'b n (h d) -> b h n d', h=self.heads)
        ks = rearrange(ks, 'b n (h d) -> b h n d', h=self.heads)
        vs = rearrange(vs, 'b n (h d) -> b h n d', h=self.heads)

        sig_len = x.size(1)
        chan_dots  = torch.matmul(q.transpose(-1, -2), ks) * self.scale
        token_dots = torch.matmul(qs, k.transpose(-1, -2)) * (sig_len ** -0.5)
        chan_attn  = self.dropout(self.attend_chan(chan_dots))
        token_attn = self.dropout(self.attend_token(token_dots))

        chan_out = torch.matmul(v, chan_attn)
        out = torch.matmul(token_attn, chan_out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        qs_res = rearrange(qs, 'b h n d -> b n (h d)')
        v_res  = rearrange(v,  'b h n d -> b n (h d)')
        return v_res, out

class Transformer(nn.Module):
    def __init__(self, embedding_dim, side_embedding_dim, depth, side_depth, heads, side_heads,
                 cross_heads, cross_dim_head, dim_head, side_dim_head, mlp_dim, side_mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.side_layers = nn.ModuleList([])
        self.cross_modal = Cross_Attention(embedding_dim, side_embedding_dim, heads, dim_head,
                                           side_dim_head, cross_heads, cross_dim_head, dropout=0.)
        self.cross_prenorm = PreNorm(embedding_dim, side_embedding_dim, self.cross_modal)
        self.cross_post = PreNorm(embedding_dim, side_embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout=dropout))
        self.back_cross_modal = Cross_Attention(side_embedding_dim, embedding_dim, heads, dim_head,
                                                side_dim_head, cross_heads, cross_dim_head, dropout=0.)
        self.back_cross_prenorm = PreNorm(side_embedding_dim, embedding_dim, self.back_cross_modal)
        self.back_cross_post = PreNorm(embedding_dim, embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout=dropout))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embedding_dim, side_embedding_dim, Attention(embedding_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(embedding_dim, side_embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout=dropout))
            ]))
        for _ in range(side_depth):
            self.side_layers.append(nn.ModuleList([
                PreNorm(side_embedding_dim, side_embedding_dim,
                        Attention(side_embedding_dim, heads=side_heads, dim_head=side_dim_head, dropout=dropout)),
                PreNorm(side_embedding_dim, side_embedding_dim,
                        FeedForward(side_embedding_dim, side_mlp_dim, dropout=dropout))
            ]))

    def forward(self, inp):
        x, y = inp  # x: EEG, y: PPS
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        for attn, ff in self.side_layers:
            y = attn(y) + y
            y = ff(y) + y
        # pre-normalize both, do TACO cross attention
        vs, out = self.cross_prenorm((x, y))
        k = vs + out                    # residual
        z = k + self.cross_post(k)      # FFN + residual
        return z

class ViT(nn.Module):
    def __init__(self, *, num_classes=2, height=9, width=9, embedding_dim=1024, side_dim=8, side_mid_dim=64,
                 side_embedding_dim=128, depth=8, side_depth=4, heads=8, side_heads=4, cross_heads=8, dim_head=64,
                 side_dim_head=32, cross_dim_head=64, mlp_dim=2048, side_mlp_dim=256, pool='cls', dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.side_dim = side_dim
        self.height = height
        self.width = width
        patch_dim = height * width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, embedding_dim))
        self.to_patch_side_embedding = nn.Sequential(
            nn.Linear(side_dim, side_mid_dim),
            nn.Linear(side_mid_dim, side_embedding_dim)
        )
        self.position_enc = PositionalEncoding2d(height, width, n_position=200)
        self.side_position_enc = PositionalEncoding(side_embedding_dim, n_position=200)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.side_cls_token = nn.Parameter(torch.randn(1, 1, side_embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(embedding_dim, side_embedding_dim, depth, side_depth, heads, side_heads,
                                       cross_heads, cross_dim_head, dim_head, side_dim_head, mlp_dim, side_mlp_dim,
                                       dropout=dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.laynorm = nn.LayerNorm(embedding_dim)
        self.side_laynorm = nn.LayerNorm(side_embedding_dim)
        self.mlp_head = nn.Sequential(nn.Linear(embedding_dim, num_classes))

    def forward(self, img):
        # img: [B, C, T] as in your original code; then we permute to [B, T, C]
        img = img.permute(0, 2, 1)
        y = img[:, :, -self.side_dim:]
        img = img[:, :, :-self.side_dim]

        # reshape 81 EEG channels to 9x9 grid per time step
        x = img.reshape((img.shape[0], img.shape[1], self.height, self.width))
        x = self.position_enc(x)
        x = x.reshape(x.shape[0], x.shape[1], self.height * self.width)

        # linear embeddings
        x = self.to_patch_embedding(x)
        y = self.to_patch_side_embedding(y)

        # prepend CLS tokens
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.laynorm(x)

        side_cls_tokens = repeat(self.side_cls_token, '1 1 d -> b 1 d', b=b)
        y = torch.cat((side_cls_tokens, y), dim=1)
        y = self.side_position_enc(y)
        y = self.dropout(y)
        y = self.side_laynorm(y)

        # separate encoders + TACO cross attention
        x = self.transformer((x, y))
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0, :]
        x = self.to_latent(x)
        return self.mlp_head(x)
