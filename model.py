# Copyright (c) 2026 Zhaoxin Yan.
"""Model architecture for the geoeffective CME prediction fusion network.

Builds the multimodal fusion model with ResNet50 +
EfficientNet-B0 backbones, per-modality EM modules, cross-modal fusion, a
2-block Vision Transformer, and an MLP head producing a scalar logit for CME geoeffectiveness.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

MODALITIES = ["lasco", "aia193", "aia211", "hmi"]


class ChannelAttention(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c // r), nn.ReLU(inplace=True),
            nn.Linear(c // r, c), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class ModalityEM(nn.Module):
    def __init__(self, in_c=2048 + 1280, out_c=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
        self.attn = ChannelAttention(out_c)

    def forward(self, rn, en):
        return self.attn(self.conv(torch.cat([rn, en], dim=1)))


class CrossModalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(4))
        self.proj = nn.Conv2d(512, 768, 1)

    def forward(self, maps):
        w = F.softmax(self.weights, dim=0)
        x = sum(w[i] * maps[i] for i in range(4))
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    def __init__(self, d=768, heads=12, mlp_ratio=4.0):
        super().__init__()
        self.heads, self.head_dim = heads, d // heads
        self.norm1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, int(d * mlp_ratio))
        self.fc2 = nn.Linear(int(d * mlp_ratio), d)

    def _attn(self, x):
        B, N, D = x.shape
        h, hd = self.heads, self.head_dim
        q = self.q(x).reshape(B, N, h, hd).transpose(1, 2)
        k = self.k(x).reshape(B, N, h, hd).transpose(1, 2)
        v = self.v(x).reshape(B, N, h, hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        return self.o((a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D))

    def forward(self, x):
        x = x + self._attn(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class PredictionNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.fuse = CrossModalFusion()
        self.pos_emb = nn.Parameter(torch.zeros(1, 49, 768))
        self.blocks = nn.ModuleList([ViTBlock() for _ in range(2)])
        self.norm = nn.LayerNorm(768)
        self.head_proj = nn.Linear(768, 512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, 1),
        )

    def forward(self, maps):
        x = self.fuse(maps) + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        return self.mlp(self.head_proj(self.norm(x).mean(1))).squeeze(-1)


class FusionModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        rn = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(rn.children())[:-2])
        en = models.efficientnet_b0(weights=None)
        self.effnet = en.features
        self.em = nn.ModuleDict({m: ModalityEM() for m in MODALITIES})
        self.pn = PredictionNet(dropout=dropout)

    def _modal_feat(self, modality, imgs):
        # Process images one at a time to bound peak activation memory; lets
        # the demo run on a laptop CPU when a modality has many frames.
        n = imgs.shape[0]
        rn_sum = en_sum = None
        for i in range(n):
            x = imgs[i:i + 1]
            rn = self.resnet(x)
            en = self.effnet(x)
            rn_sum = rn if rn_sum is None else rn_sum + rn
            en_sum = en if en_sum is None else en_sum + en
        return self.em[modality](rn_sum / n, en_sum / n)

    def forward(self, batch_modalities):
        dev = next(self.parameters()).device
        bsz = len(next(iter(batch_modalities.values())))
        stacked = []
        for m in MODALITIES:
            feats = [self._modal_feat(m, batch_modalities[m][b].to(dev))
                     for b in range(bsz)]
            stacked.append(torch.cat(feats, dim=0))
        return self.pn(stacked)
