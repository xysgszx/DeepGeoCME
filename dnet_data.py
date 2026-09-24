#!/usr/bin/env python3
# Copyright (c) 2026 Zhaoxin Yan.
"""Generic loader for user-prepared tensors; no raw-observation processing.

CSV: id,label. Each id names <cache_dir>/<id>.pt containing a dictionary of
lasco, aia193, aia211, hmi tensors, each shaped (n_images, 3, 224, 224), n>=1.
Inputs must already be prepared for the model as finite floating-point tensors.
The loader checks tensor shape and finite values without image transformations.
"""
from __future__ import annotations
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset

MODALITIES = ('lasco', 'aia193', 'aia211', 'hmi')


def validate_events(events):
    if not events:
        raise ValueError('Event list must not be empty.')
    seen = set()
    for ev, label in events:
        if not isinstance(ev, str) or not ev or ev in ('.', '..') or any(c in ev for c in '/\\\x00'):
            raise ValueError(f'Invalid event id: {ev!r}')
        if ev in seen:
            raise ValueError(f'Duplicate event id: {ev}')
        if label not in (0, 1):
            raise ValueError(f'{ev}: label must be 0 or 1.')
        seen.add(ev)


def read_labels(csv_path):
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'id', 'label'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain id,label columns.')
        events = []
        for row in reader:
            label = (row.get('label') or '').strip()
            if label not in ('0', '1'):
                raise ValueError(f'Invalid label at CSV line {reader.line_num}: {label!r}')
            events.append(((row.get('id') or '').strip(), int(label)))
    validate_events(events)
    return events


class CMEDataset(Dataset):
    def __init__(self, events, cache_dir):
        validate_events(events)
        if cache_dir is None:
            raise ValueError('A tensor cache directory is required.')
        self.events = events
        self.cache_dir = Path(cache_dir)
        missing = [ev for ev, _ in events if not (self.cache_dir / f'{ev}.pt').is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} tensor file(s) missing in {self.cache_dir}, e.g. {missing[0]}.pt')

    def __len__(self):
        return len(self.events)

    def __getitem__(self, i):
        ev, y = self.events[i]
        obj = torch.load(self.cache_dir / f'{ev}.pt', map_location='cpu', weights_only=True)
        if not isinstance(obj, dict):
            raise ValueError(f'{ev}: expected a dictionary of modality tensors.')
        out = {}
        for m in MODALITIES:
            if m not in obj:
                raise KeyError(f'{ev}.pt has no {m!r} entry')
            t = obj[m]
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                raise ValueError(f'{ev}/{m}: expected a floating-point tensor.')
            if t.ndim != 4 or t.shape[0] < 1 or tuple(t.shape[1:]) != (3, 224, 224):
                raise ValueError(f'{ev}/{m}: expected (n>=1, 3, 224, 224), got {tuple(t.shape)}.')
            t = t.float()
            if not torch.isfinite(t).all():
                raise ValueError(f'{ev}/{m}: non-finite input values.')
            out[m] = t
        return out, torch.tensor(y, dtype=torch.float32)
