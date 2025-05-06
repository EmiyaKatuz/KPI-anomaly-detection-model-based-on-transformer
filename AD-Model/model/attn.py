import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

# 保留原 TriangularCausalMask（可选备用）
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

# New dynamic neighbor mask generation function added.
def create_diag_symmetric_mask(seq_len, diagonal, device="cpu"):
    mask = torch.zeros(seq_len, seq_len).to(device)
    for i in range(seq_len):
        for j in range(i + diagonal, seq_len):
            mask[i, j] = 1
            mask[j, i] = 1
    return mask.bool()

class AnomalyAttention(nn.Module):
    def __init__(self, win_size, input_c, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False, dynamic_mask_scale=None):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.dynamic_mask_scale = dynamic_mask_scale  # 新增参数控制动态掩码窗口

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 使用动态邻居掩码替换上三角掩码
        if self.dynamic_mask_scale is not None:
            dyn_mask = create_diag_symmetric_mask(L, self.dynamic_mask_scale, device=queries.device)
            scores = scores.masked_fill(dyn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L), float('-inf'))
        elif self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores = scores.masked_fill(attn_mask.mask, -np.inf)

        attn = scale * scores
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        return V.contiguous(), series

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, num_proto, len_map, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.num_proto = num_proto
        self.len_map = len_map
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.keys = nn.Parameter(torch.randn(num_proto, d_model))
        self.values = nn.Parameter(torch.randn(num_proto, d_model))
        self.attn_map = nn.Parameter(torch.randn(len_map, num_proto))

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        H = self.n_heads
        x = queries

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.keys.view(1, self.num_proto, H, -1).repeat(B, 1, 1, 1)
        values = self.values.view(1, self.num_proto, H, -1).repeat(B, 1, 1, 1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, attn = self.inner_attention(queries, keys, values, sigma, attn_mask)

        attn_map = torch.softmax(self.attn_map, dim=-1).view(1, 1, self.len_map, -1).repeat(B, H, 1, 1)
        sim = torch.einsum("bhln,bhmn->bhlm", attn, attn_map)
        sim_l = torch.sum(torch.sum(sim, dim=-1), dim=1)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn, sim_l
