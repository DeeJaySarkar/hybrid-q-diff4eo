"""
Plot training and validation loss curves from DiffCR / QDiffCR train.log files.

The training loop logs, per epoch:
    train/mse_loss: <value>   +   epoch: <n>
and (when val_loss_epoch is set) every val_loss_epoch epochs:
    val/mse_loss: <value>     +   val_epoch: <n>

Usage:
    python plot_curves.py \
        --run "Classical:/home/deejay/DiffCR/experiments/train_.../train.log" \
        --run "Quantum:/home/deejay/QDiffCR-basic/experiments/train_.../train.log" \
        --out /home/deejay/loss_curves.png

Produces a 2-panel figure:
  (left)  full train + val loss vs epoch, log-y
  (right) zoomed to later epochs to see the converged regime and any train/val divergence
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    txt = open(path).read()
    train, val = [], []
    tr_loss = None
    va_loss = None
    for line in txt.splitlines():
        m = re.search(r'train/mse_loss:\s*([0-9.eE+-]+)', line)
        if m:
            tr_loss = float(m.group(1)); continue
        m = re.search(r'val/mse_loss:\s*([0-9.eE+-]+)', line)
        if m:
            va_loss = float(m.group(1)); continue
        m = re.search(r'val_epoch:\s*(\d+)', line)
        if m and va_loss is not None:
            val.append((int(m.group(1)), va_loss)); va_loss = None; continue
        m = re.search(r'epoch:\s*(\d+)', line)
        if m and tr_loss is not None:
            train.append((int(m.group(1)), tr_loss)); tr_loss = None; continue
    train = np.array(train) if train else np.empty((0, 2))
    val = np.array(val) if val else np.empty((0, 2))
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help='LABEL:/path/to/train.log  (repeatable)')
    ap.add_argument("--out", default="loss_curves.png")
    ap.add_argument("--zoom-from", type=int, default=200,
                    help="epoch to start the zoomed panel from")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, path = spec.split(":", 1)
        tr, va = parse_log(path)
        runs.append((label, tr, va))
        print(f"{label}: {len(tr)} train points, {len(va)} val points")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for (label, tr, va), c in zip(runs, colors):
        if len(tr):
            ax1.plot(tr[:, 0], tr[:, 1], '-', color=c, label=f'{label} train')
        if len(va):
            ax1.plot(va[:, 0], va[:, 1], '--', color=c, label=f'{label} val')
        if len(tr):
            mask = tr[:, 0] >= args.zoom_from
            ax2.plot(tr[mask, 0], tr[mask, 1], '-', color=c, label=f'{label} train')
        if len(va):
            mask = va[:, 0] >= args.zoom_from
            ax2.plot(va[mask, 0], va[mask, 1], '--', color=c, label=f'{label} val')

    for ax, title in [(ax1, 'Full training (log scale)'),
                      (ax2, f'Zoom: epoch >= {args.zoom_from} (converged regime)')]:
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('MSE loss (x0 objective)')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
