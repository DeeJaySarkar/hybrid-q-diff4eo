"""
Run ONLY the quantum (QDiffCR-basic) model on the full 32x32 dataset.
  - seed 1, 3000 epochs
  - val loss every 100 epochs, checkpoint every 100 epochs
  - best-val-checkpoint used for testing (EMA weights)
  - archived to archive/quantum_full32_ep3000_seed1/
"""
import json, os, sys, glob, shutil, subprocess
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_results import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
FOLDER = os.path.join(HERE, "QDiffCR")
TEMPLATE = "config/ours_sigmoid_w32_quantum.json"
DATA_ROOT = os.environ.get("CTGAN_DATA_32", "data/CTGAN_full_32/CTGAN/Sen2_MTC/dataset")
SEED = 1
N_EPOCH = 3000
TAG = "quantum_full32_seed1"

def load(p):
    with open(p) as f: return json.load(f, object_pairs_hook=OrderedDict)
def dump(c, p):
    with open(p, "w") as f: json.dump(c, f, indent=4)
def exp_set(): return set(glob.glob(os.path.join(FOLDER, "experiments", "*")))
def newest_since(before):
    new = sorted(exp_set() - before, key=os.path.getmtime)
    return new[-1] if new else None

# --- train config ---
cfg = load(os.path.join(FOLDER, TEMPLATE))
cfg["name"] = TAG
cfg["seed"] = SEED
cfg["path"]["resume_state"] = "None"
for sp in ["train", "val", "test"]:
    a = cfg["datasets"][sp]["which_dataset"]["args"]
    a["data_root"] = DATA_ROOT
    a["image_size"] = 32
    a.pop("frame_stride", None)
cfg["model"]["which_networks"][0]["args"]["unet"]["quantum"] = {
    "enabled": True, "n_qubits": 4, "n_layers": 2, "mode": "global"
}
cfg["train"]["n_epoch"] = N_EPOCH
cfg["train"]["val_loss_epoch"] = 100
cfg["train"]["save_checkpoint_epoch"] = 100
cfg["train"]["val_epoch"] = N_EPOCH * 10  # skip expensive full-sampling val
os.makedirs(os.path.join(FOLDER, "config", "_pipe"), exist_ok=True)
train_cfg = f"config/_pipe/{TAG}_train.json"
dump(cfg, os.path.join(FOLDER, train_cfg))

# --- train ---
print(f"### TRAINING {TAG} | {N_EPOCH} epochs | seed {SEED} | full 32x32 dataset ###")
before = exp_set()
subprocess.run([PYTHON, "run.py", "-c", train_cfg, "-p", "train"], cwd=FOLDER, check=True)
train_root = newest_since(before)
print(f"  train_root={train_root}")

# --- find best-val checkpoint ---
import re, numpy as np
txt = open(os.path.join(train_root, "train.log")).read()
vl = [float(x) for x in re.findall(r'val/mse_loss:\s*([0-9.eE+-]+)', txt)]
ve = [int(x) for x in re.findall(r'val_epoch:\s*(\d+)', txt)]
n = min(len(vl), len(ve))
val_dict = dict(zip(ve[:n], vl[:n]))
# only consider epochs where a checkpoint was saved
saved = []
for p in glob.glob(os.path.join(train_root, "checkpoint", "*_Network.pth")):
    b = os.path.basename(p)
    if "_ema" not in b:
        try: saved.append(int(b.split("_")[0]))
        except: pass
cands = [(val_dict.get(e, float("inf")), e) for e in saved]
cands.sort()
best_ep = cands[0][1] if cands else N_EPOCH
print(f"  best-val epoch: {best_ep} (val_loss={val_dict.get(best_ep)})")

# --- use EMA weights for testing ---
ckpt_dir = os.path.join(train_root, "checkpoint")
ema_src = os.path.join(ckpt_dir, f"{best_ep}_Network_ema.pth")
if os.path.exists(ema_src):
    ema_dst = os.path.join(ckpt_dir, f"{best_ep}ema_Network.pth")
    if not os.path.exists(ema_dst): shutil.copy(ema_src, ema_dst)
    ckpt_prefix = os.path.join(ckpt_dir, f"{best_ep}ema")
else:
    ckpt_prefix = os.path.join(ckpt_dir, str(best_ep))

# --- test config ---
tcfg = load(os.path.join(FOLDER, TEMPLATE))
tcfg["name"] = TAG
tcfg["seed"] = SEED
tcfg["path"]["resume_state"] = ckpt_prefix
for sp in ["train", "val", "test"]:
    a = tcfg["datasets"][sp]["which_dataset"]["args"]
    a["data_root"] = DATA_ROOT
    a["image_size"] = 32
    a.pop("frame_stride", None)
tcfg["model"]["which_networks"][0]["args"]["unet"]["quantum"] = {
    "enabled": True, "n_qubits": 4, "n_layers": 2, "mode": "global"
}
test_cfg = f"config/_pipe/{TAG}_test.json"
dump(tcfg, os.path.join(FOLDER, test_cfg))

print(f"### TESTING {TAG} | checkpoint ep {best_ep} (EMA) ###")
before = exp_set()
subprocess.run([PYTHON, "run.py", "-c", test_cfg, "-p", "test"], cwd=FOLDER, check=True)
test_root = newest_since(before)
# find results dir
rdir = None
for d in sorted(glob.glob(os.path.join(test_root, "results", "*", "*")), key=os.path.getmtime, reverse=True):
    if os.path.isdir(d) and glob.glob(os.path.join(d, "GT_*")):
        rdir = d; break

metrics = evaluate(rdir, use_lpips=True) if rdir else {}
print(f"  test results: {rdir}")
print(f"  PSNR={metrics.get('psnr_mean'):.4f}  SSIM={metrics.get('ssim_mean'):.4f}  LPIPS={metrics.get('lpips_mean'):.4f}")

# --- archive ---
archive = os.path.join(HERE, "archive", "quantum_full32_ep3000_seed1")
os.makedirs(os.path.join(archive, "QDiffCR", "experiments"), exist_ok=True)
for d in exp_set():
    dest = os.path.join(archive, "QDiffCR", "experiments", os.path.basename(d))
    if not os.path.exists(dest): shutil.move(d, dest)
print(f"\nARCHIVED to {archive}")
print("DONE.")
