#!/usr/bin/env python3
# Copyright (c) 2026 Zhaoxin Yan.
"""Run the fusion model on the test events and plot the Figure 7 confusion matrix."""
from __future__ import annotations
import argparse
import csv
import os
import hashlib
import json
import tempfile
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import FusionModel            # noqa: E402
from data import CMEDataset, read_csv    # noqa: E402

THRESHOLD = 0.6                                  # paper, Sec. 3
PAPER_FIG7 = {"TN": 14, "FP": 2, "FN": 2, "TP": 15}  # for the final comparison only
ASSET_FILES = ("dnet_best.pt", "test_tensors.zip")


# ----------------------------------------------------------------------------- assets
def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_assets(assets_dir: Path, base_url: str | None) -> tuple[Path, Path]:
    """Download and verify the weights and test inputs, then extract the tensors."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    expected = json.loads((HERE / "assets_manifest.json").read_text())
    for name in ASSET_FILES:
        f = assets_dir / name
        if not f.exists():
            if not base_url:
                raise FileNotFoundError(f"Missing {f}. Supply --assets-url or download the release assets.")
            temporary = f.with_suffix(f.suffix + ".part")
            try:
                print(f"downloading {name} ...", flush=True)
                urllib.request.urlretrieve(f"{base_url.rstrip('/')}/{name}", temporary)
                if file_sha256(temporary) != expected[name]:
                    raise ValueError(f"Download checksum mismatch: {name}")
                temporary.replace(f)
            finally:
                temporary.unlink(missing_ok=True)
        if file_sha256(f) != expected[name]:
            raise ValueError(f"Checksum mismatch: {f}. Download the release file again.")
    # Extract on every run so an interrupted earlier extraction cannot be reused.
    with zipfile.ZipFile(assets_dir / "test_tensors.zip") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if len(path.parts) != 2 or path.parts[0] != "test_tensors" or path.suffix != ".pt":
                raise ValueError(f"Unexpected tensor archive entry: {info.filename}")
            target = assets_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as inp, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                temporary = Path(out.name)
                try:
                    import shutil
                    shutil.copyfileobj(inp, out)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(target)
    return assets_dir / "dnet_best.pt", assets_dir / "test_tensors"


# ----------------------------------------------------------------------------- inference
def predict(ckpt: Path, tensor_dir: Path, test_csv: Path, device: str):
    """Run the model on every test event. Returns (ids, labels, probabilities)."""
    events = read_csv(str(test_csv))
    ds = CMEDataset(events, cache_dir=tensor_dir)
    dev = torch.device(device)
    model = FusionModel().to(dev)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.eval()
    ids, labels, probs = [], [], []
    with torch.no_grad():
        for i, (ev, y) in enumerate(events):
            x, _ = ds[i]
            logit = model({k: [v] for k, v in x.items()}).item()
            if not np.isfinite(logit):
                raise ValueError(f"Non-finite model output for {ev}")
            ids.append(ev); labels.append(int(y)); probs.append(float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64))))
            print(f"  [{i + 1:2d}/{len(events)}] {ev:18s} label={int(y)} p={probs[-1]:.3f}", flush=True)
    return ids, np.array(labels), np.array(probs)


def confusion_counts(labels: np.ndarray, probs: np.ndarray, thr: float = THRESHOLD) -> dict:
    """TN/FP/FN/TP computed from the labels and the thresholded predictions."""
    pred = (probs >= thr).astype(int)
    return {"TN": int(((pred == 0) & (labels == 0)).sum()),
            "FP": int(((pred == 1) & (labels == 0)).sum()),
            "FN": int(((pred == 0) & (labels == 1)).sum()),
            "TP": int(((pred == 1) & (labels == 1)).sum())}


def metrics(cm: dict, labels: np.ndarray, probs: np.ndarray) -> dict:
    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    rec = tp / max(tp + fn, 1); pre = tp / max(tp + fp, 1)
    return {"recall": rec, "precision": pre, "accuracy": (tp + tn) / len(labels),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1), "TSS": rec - fp / max(fp + tn, 1),
            "BS": float(np.mean((labels - probs) ** 2))}


# ----------------------------------------------------------------------------- figure
def plot_fig7(cm: dict, out: Path):
    """2x2 confusion matrix in the style of Fig. 7: rows = true class (N, Y), columns =
    predicted class (N, Y); cells TN (dark red), FP (light pink), FN (light blue),
    TP (navy); class name in the upper-left corner, count in the centre; no axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cells = [  # (name, column, row, face colour, text colour)  row 0 = top
        ("TN", 0, 0, "#8b0000", "white"),
        ("FP", 1, 0, "#ffb6c1", "black"),
        ("FN", 0, 1, "#add8e6", "black"),
        ("TP", 1, 1, "#000080", "white"),
    ]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    for name, c, r, face, txt in cells:
        y0 = 1 - r                                   # row 0 at the top
        ax.add_patch(Rectangle((c, y0), 1, 1, facecolor=face, edgecolor="none"))
        ax.text(c + 0.05, y0 + 0.92, name, ha="left", va="top", fontsize=17, color=txt)
        ax.text(c + 0.5, y0 + 0.5, str(cm[name]), ha="center", va="center", fontsize=17, color=txt)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"figure saved -> {out.name}")
    return fig


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-dir", default=str(HERE / "assets"))
    ap.add_argument("--assets-url", default=os.environ.get("ASSETS_URL", "https://github.com/xysgszx/DeepGeoCME/releases/download/v1.0"),
                    help="base URL holding dnet_best.pt and test_tensors.zip (GitHub release, ...)")
    ap.add_argument("--test-csv", default=str(HERE / "test.csv"))
    ap.add_argument("--out", default=str(HERE / "fig7_confusion_matrix.png"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    ckpt, tdir = fetch_assets(Path(args.assets_dir), args.assets_url)
    print(f"weights: {ckpt}\ntensors: {tdir}\ndevice:  {args.device}\n\nrunning the model on the test set:")
    ids, labels, probs = predict(ckpt, tdir, Path(args.test_csv), args.device)

    cm = confusion_counts(labels, probs)
    m = metrics(cm, labels, probs)
    print(f"\nconfusion matrix (threshold {THRESHOLD}):")
    print(f"             pred N   pred Y")
    print(f"  true N    {cm['TN']:>6}   {cm['FP']:>6}")
    print(f"  true Y    {cm['FN']:>6}   {cm['TP']:>6}")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    with open(Path(args.out).with_name("predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label", "probability", "prediction"])
        w.writerows((i, int(l), f"{p:.4f}", int(p >= THRESHOLD)) for i, l, p in zip(ids, labels, probs))
    plot_fig7(cm, Path(args.out))

    same = all(cm[k] == PAPER_FIG7[k] for k in PAPER_FIG7)
    print(f"\ncomparison with the paper's Fig. 7 {PAPER_FIG7}: {'MATCH' if same else 'MISMATCH'}")
    return cm


if __name__ == "__main__":
    main()
