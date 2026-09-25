#!/usr/bin/env python3
# Copyright (c) 2026 Zhaoxin Yan.
"""Train the multimodal CME classifier on prepared event tensors.

Shared ImageNet CNN backbones feed four modality ensemble modules, weighted
fusion, 49 spatial tokens, two transformer blocks and a scalar-logit MLP.
The CLI defaults to pretrained transformer blocks and CNN fine-tuning.
Validation holds out ceil(12% of each class); threshold selection uses validation
TSS with BS as the epoch-selection tie breaker.
"""
from __future__ import annotations
import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models


# ----------------------------- Config ---------------------------------------

MODALITIES = ("lasco", "aia193", "aia211", "hmi")   # the four data types of the paper

# Generic tensor loader (see dnet_data.py for the csv and tensor layout):
#   read_labels(csv)               -> [(id, label)]
#   CMEDataset(events, cache_dir)  -> ({modality: (n_images, 3, 224, 224)}, label)
from dnet_data import read_labels, CMEDataset  # noqa: E402


# ----------------------------- Model ----------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
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
    """Fuse modality maps into 49 spatial tokens."""

    def __init__(self, grid=7):
        super().__init__()
        if grid != 7:
            raise ValueError("This model uses a 7x7 token grid (49 tokens).")
        self.grid = grid
        self.weights = nn.Parameter(torch.ones(4))
        self.proj = nn.Conv2d(512, 768, 1)

    def forward(self, maps):
        w = F.softmax(self.weights, dim=0)
        x = sum(w[i] * maps[i] for i in range(4))
        if x.shape[-2:] != (self.grid, self.grid):
            x = F.interpolate(x, size=(self.grid, self.grid), mode="bilinear",
                              align_corners=False)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # (B, grid*grid, 768)


class ViTBlock(nn.Module):
    def __init__(self, d=768, heads=12, mlp_ratio=4.0):
        super().__init__()
        self.d, self.heads, self.head_dim = d, heads, d // heads
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


def load_pretrained_vit(blocks):
    import timm
    print(f"[pretrained-vit] loading vit_base_patch16_224 into {len(blocks)} blocks")
    src = timm.create_model("vit_base_patch16_224", pretrained=True)
    for i, blk in enumerate(blocks):
        s = src.blocks[i]
        with torch.no_grad():
            qw, kw, vw = s.attn.qkv.weight.chunk(3, 0)
            qb, kb, vb = s.attn.qkv.bias.chunk(3, 0)
            blk.q.weight.copy_(qw); blk.q.bias.copy_(qb)
            blk.k.weight.copy_(kw); blk.k.bias.copy_(kb)
            blk.v.weight.copy_(vw); blk.v.bias.copy_(vb)
            blk.o.weight.copy_(s.attn.proj.weight); blk.o.bias.copy_(s.attn.proj.bias)
            blk.fc1.weight.copy_(s.mlp.fc1.weight); blk.fc1.bias.copy_(s.mlp.fc1.bias)
            blk.fc2.weight.copy_(s.mlp.fc2.weight); blk.fc2.bias.copy_(s.mlp.fc2.bias)
            blk.norm1.weight.copy_(s.norm1.weight); blk.norm1.bias.copy_(s.norm1.bias)
            blk.norm2.weight.copy_(s.norm2.weight); blk.norm2.bias.copy_(s.norm2.bias)

class PredictionNet(nn.Module):
    def __init__(self, dropout=0.3, vit_grid=7):
        super().__init__()
        self.fuse = CrossModalFusion(grid=vit_grid)
        self.pos_emb = nn.Parameter(torch.zeros(1, vit_grid * vit_grid, 768))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
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
    def __init__(self, pretrained_vit=False,
                 freeze_backbones=False, dropout=0.3, vit_grid=7, pretrained_backbones=True):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbones else None)
        self.resnet = nn.Sequential(*list(rn.children())[:-2])
        en = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbones else None)
        self.effnet = en.features
        if freeze_backbones:
            for p in self.resnet.parameters(): p.requires_grad = False
            for p in self.effnet.parameters(): p.requires_grad = False
            # Keep BN in eval mode so running stats stay stable.
            self.resnet.eval(); self.effnet.eval()
        self.freeze_backbones = freeze_backbones
        self.em = nn.ModuleDict({m: ModalityEM() for m in MODALITIES})
        self.pn = PredictionNet(dropout=dropout, vit_grid=vit_grid)
        if pretrained_vit:
            # Initialize Transformer block weights; positional embeddings are initialized separately.
            load_pretrained_vit(self.pn.blocks)

    def train(self, mode=True):
        """Override so frozen backbones stay in eval mode during training.

        When the backbones are unfrozen, their weights train but BatchNorm
        stays in eval mode to retain its current running statistics.
        Each event supplies a batch of images for each modality.
        """
        super().train(mode)
        if self.freeze_backbones:
            self.resnet.eval(); self.effnet.eval()
        else:
            for m in list(self.resnet.modules()) + list(self.effnet.modules()):
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def _modal_feat(self, modality, imgs):
        """Process one modality's frames together, average features, and apply EM."""
        rn = self.resnet(imgs).mean(dim=0, keepdim=True)   # (1, 2048, 7, 7)
        en = self.effnet(imgs).mean(dim=0, keepdim=True)   # (1, 1280, 7, 7)
        return self.em[modality](rn, en)

    def forward(self, batch_modalities):
        dev = next(self.parameters()).device
        bsz = len(next(iter(batch_modalities.values())))
        stacked = []
        for m in MODALITIES:
            feats = [self._modal_feat(m, batch_modalities[m][b].to(dev))
                     for b in range(bsz)]
            stacked.append(torch.cat(feats, dim=0))
        return self.pn(stacked)


# ----------------------------- Metrics --------------------------------------

from metrics import compute_metrics


def grid_thr(y, p):
    if set(np.asarray(y).tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both validation classes.")
    best_t, best = 0.5, -1.0
    for t in (i / 100 for i in range(30, 71, 5)):
        m = compute_metrics(y, p, float(t))
        if m.tss > best:
            best, best_t = m.tss, float(t)
    return best_t


# ----------------------------- Train / Eval ---------------------------------

def run_epoch(model, loader, opt, device, train, sched=None, pos_weight=None):
    """Run one epoch in float32 using binary event labels."""
    model.train(train)
    total, n = 0.0, 0
    ys, ps = [], []
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for batch_x, y in loader:
        y = y.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logit = model(batch_x)
            if not torch.isfinite(logit).all():
                raise ValueError("Model produced non-finite logits.")
            loss = bce(logit, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0, error_if_nonfinite=True)
            opt.step()
            if sched is not None:
                sched.step()
        total += loss.item() * y.size(0); n += y.size(0)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit.detach()).cpu().numpy())
    return total / max(n, 1), np.concatenate(ys), np.concatenate(ps)


def train_loop(tr_evs, val_evs, device, args):
    model = FusionModel(pretrained_vit=args.pretrained_vit,
                        freeze_backbones=not args.unfreeze_backbones,
                        dropout=args.dropout, vit_grid=args.vit_grid,
                        pretrained_backbones=not args.random_init).to(device)
    n_tot = sum(p.numel() for p in model.parameters()) / 1e6
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"params total={n_tot:.2f}M  trainable={n_tr:.2f}M", flush=True)
    # Class weighting from training set imbalance.
    n_pos = sum(1 for _, y in tr_evs if y == 1)
    n_neg = len(tr_evs) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"class balance: pos={n_pos} neg={n_neg}  pos_weight={pos_weight.item():.3f}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    tr = DataLoader(CMEDataset(tr_evs, args.cache_dir), batch_size=1, shuffle=True,
                    num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))
    va = DataLoader(CMEDataset(val_evs, args.cache_dir), batch_size=1, shuffle=False,
                    num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))
    # Linear warmup over first 5% of total steps, then cosine decay.
    steps_per_epoch = max(1, len(tr))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best, best_bs, best_ep, state, thr = -2.0, 9.0, 0, None, 0.6
    for ep in range(1, args.epochs + 1):
        loss, _, _ = run_epoch(model, tr, opt, device, True, sched=sched,
                               pos_weight=pos_weight)
        _, y, p = run_epoch(model, va, None, device, False)
        t = grid_thr(y, p); m = compute_metrics(y, p, t)
        cur_lr = opt.param_groups[0]["lr"]
        print(f"ep{ep:02d}  lr={cur_lr:.2e}  loss={loss:.4f}  val TSS={m.tss:.3f} "
              f"(thr={t:.2f})  F1={m.f1:.3f}  BS={m.bs:.3f}", flush=True)
        if m.tss > best or (m.tss == best and m.bs < best_bs):
            best, best_bs, best_ep, thr = m.tss, m.bs, ep, t
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"selected epoch {best_ep}: val TSS={best:.3f} BS={best_bs:.3f} "
          f"thr={thr:.2f}", flush=True)
    model.load_state_dict(state)
    model.val_info = {"val_tss": best, "val_bs": best_bs, "epoch": best_ep,
                      "thr": thr, "selection": "best_validation"}
    return model, thr


def evaluate_probs(model, evs, device, thr, num_workers=4, cache_dir=None):
    loader = DataLoader(CMEDataset(evs, cache_dir), batch_size=1, shuffle=False,
                        num_workers=num_workers,
                        persistent_workers=(num_workers > 0))
    _, y, p = run_epoch(model, loader, None, device, False)
    return compute_metrics(y, p, thr), y, p


def evaluate(model, evs, device, thr, num_workers=4, cache_dir=None):
    return evaluate_probs(model, evs, device, thr, num_workers, cache_dir)[0]


def print_confusion(m, label):
    print(f"\n[{label}] confusion matrix")
    print(f"           pred N   pred Y")
    print(f"  true N   {m.tn:>6}   {m.fp:>6}")
    print(f"  true Y   {m.fn:>6}   {m.tp:>6}")
    print(f"  TSS={m.tss:.3f}  F1={m.f1:.3f}  Acc={m.accuracy:.3f}  "
          f"Rec={m.recall:.3f}  Prec={m.precision:.3f}  "
          f"BS={m.bs:.3f}  BSS={m.bss:.3f}")


def save_checkpoint(checkpoint, destination):
    """Write a checkpoint atomically in its destination directory."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ----------------------------- Entry ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="training events (csv: id,label)")
    ap.add_argument("--test-csv",  required=True, help="test events (csv: id,label)")
    ap.add_argument("--cache-dir", required=True,
                    help="directory with one <id>.pt tensor file per event (see dnet_data.py)")
    ap.add_argument("--random-init", action="store_true",
                    help="initialize all networks locally without pretrained downloads (e.g. smoke tests)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained-vit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--unfreeze-backbones", action=argparse.BooleanOptionalAction, default=True,
                    help="fine-tune ResNet50 and EfficientNet-B0 (default: enabled)")
    ap.add_argument("--weight-decay", type=float, default=1e-3,
                    help="AdamW weight decay")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="dropout in MLP head")
    ap.add_argument("--vit-grid", type=int, default=7, choices=(7,),
                    help="ViT token grid: 7 -> 49 tokens")
    ap.add_argument("--save", default="checkpoints/dnet_best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.epochs < 1 or args.num_workers < 0:
        ap.error('--epochs must be positive and --num-workers must be nonnegative')
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error('--lr must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        ap.error('--weight-decay must be finite and nonnegative')
    if not math.isfinite(args.dropout) or not 0 <= args.dropout <= 1:
        ap.error('--dropout must be finite and within [0, 1]')
    if args.random_init:
        args.pretrained_vit = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  cuda: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}  "
              f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    train_all = read_labels(args.train_csv)
    test_evs = read_labels(args.test_csv)

    overlap = {ev for ev, _ in train_all} & {ev for ev, _ in test_evs}
    if overlap:
        raise ValueError(f'Training and test CSVs overlap: {len(overlap)} event id(s).')
    if any(sum(y == c for _, y in train_all) < 2 for c in (0, 1)):
        raise ValueError('Training CSV needs at least two events per class for train/validation splitting.')
    rng = np.random.default_rng(args.seed)
    pos = [e for e in train_all if e[1] == 1]; neg = [e for e in train_all if e[1] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    nvp = max(1, math.ceil(0.12 * len(pos)))
    nvn = max(1, math.ceil(0.12 * len(neg)))
    val_evs = pos[:nvp] + neg[:nvn]; tr_evs = pos[nvp:] + neg[nvn:]
    print(f"train={len(tr_evs)}  val={len(val_evs)}  test={len(test_evs)}")

    model, thr = train_loop(tr_evs, val_evs, device, args)
    checkpoint = {"state_dict": model.state_dict(), "threshold": thr,
                  "val": getattr(model, "val_info", None), "args": vars(args),
                  "model_config": {"vit_grid": args.vit_grid, "dropout": args.dropout}}
    save_checkpoint(checkpoint, args.save)
    print(f"saved -> {args.save}")
    m, y, p = evaluate_probs(model, test_evs, device, thr, args.num_workers, args.cache_dir)
    print_confusion(m, f"TEST (thr={thr:.2f})")
    m6 = compute_metrics(y, p, 0.60)
    print_confusion(m6, "TEST (fixed thr=0.60)")
    print("\nper-event test probabilities:")
    for (ev, _), yi, pi in zip(test_evs, y, p):
        print(f"  {ev}  true={int(yi)}  prob={pi:.4f}")
    checkpoint["test"] = {"tss": m.tss, "bs": m.bs, "bss": m.bss, "tn": m.tn,
                          "fp": m.fp, "fn": m.fn, "tp": m.tp,
                          "events": [e for e, _ in test_evs],
                          "y": y.tolist(), "p": p.tolist()}
    save_checkpoint(checkpoint, args.save)
    print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()

