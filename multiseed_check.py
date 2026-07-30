"""
Quick multi-seed sanity check of the PRISTINE repo variants (architecture code
untouched -- only seed and n_epoch are overridden in the config).

Runs DiffCR-basic (classical) and QDiffCR-basic (quantum) for a few epochs at
seeds 0,1,2 and prints the per-epoch training loss so we can see whether the
quantum variant is consistently lower (its claimed advantage) or whether the
earlier single-run result was seed noise.
"""
import json, os, re, subprocess, sys
from collections import OrderedDict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
N_EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEEDS = [0, 1, 2]
VARIANTS = OrderedDict([
    ("classical", ("DiffCR-basic",  "config/ours_sigmoid_w32.json")),
    ("quantum",   ("QDiffCR-basic", "config/ours_sigmoid_w32_quantum.json")),
])


def parse_train(log):
    txt = open(log).read()
    tl = [float(x) for x in re.findall(r'train/mse_loss:\s*([0-9.eE+-]+)', txt)]
    te = [int(x) for x in re.findall(r'(?<!val_)epoch:\s*(\d+)', txt)]
    n = min(len(tl), len(te))
    return dict(zip(te[:n], tl[:n]))


def newest_train_log(folder):
    import glob
    logs = glob.glob(os.path.join(folder, "experiments", "*", "train.log"))
    return max(logs, key=os.path.getmtime) if logs else None


results = {}   # (variant, seed) -> {epoch: loss}
for vname, (folder_rel, template) in VARIANTS.items():
    folder = os.path.join(HERE, folder_rel)
    for seed in SEEDS:
        tag = f"{vname}_seed{seed}"
        cfg = json.load(open(os.path.join(folder, template)), object_pairs_hook=OrderedDict)
        cfg["name"] = tag
        cfg["seed"] = seed
        cfg["train"]["n_epoch"] = N_EPOCH
        cfg["train"]["val_epoch"] = N_EPOCH * 100          # skip val sampling
        cfg["train"]["save_checkpoint_epoch"] = N_EPOCH * 100
        os.makedirs(os.path.join(folder, "config", "_ms"), exist_ok=True)
        cpath = f"config/_ms/{tag}.json"
        json.dump(cfg, open(os.path.join(folder, cpath), "w"), indent=4)
        print(f"\n===== RUN {tag} ({N_EPOCH} epochs) =====")
        subprocess.run([PYTHON, "run.py", "-c", cpath, "-p", "train"], cwd=folder, check=True)
        log = newest_train_log(folder)
        results[(vname, seed)] = parse_train(log)

# ---- comparison table ----
print("\n\n==================== TRAIN LOSS BY EPOCH ====================")
header = "epoch | " + " | ".join(f"{v}_s{s}" for v in VARIANTS for s in SEEDS)
print(header)
for ep in range(1, N_EPOCH + 1):
    row = f"{ep:>5} | " + " | ".join(
        f"{results[(v,s)].get(ep, float('nan')):>10.3f}" for v in VARIANTS for s in SEEDS)
    print(row)

print("\n==================== MEAN OVER SEEDS ====================")
print(f"{'epoch':>5} | {'classical':>12} | {'quantum':>12} | {'q - c':>10}")
for ep in range(1, N_EPOCH + 1):
    c = np.nanmean([results[('classical', s)].get(ep, np.nan) for s in SEEDS])
    q = np.nanmean([results[('quantum', s)].get(ep, np.nan) for s in SEEDS])
    print(f"{ep:>5} | {c:>12.3f} | {q:>12.3f} | {q-c:>+10.3f}")
print("\n(q - c) negative => quantum lower (better) at that epoch; positive => classical lower.")
