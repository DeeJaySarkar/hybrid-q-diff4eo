"""
Rigorous experiment pipeline: classical vs quantum vs param-matched-control,
at 32x32 and 64x64, multi-seed, with early-stopping-by-selection, EMA evaluation,
and quality-vs-epoch convergence curves.

MODELS (all share identical data / splits / seed; only the bottleneck differs):
  * classical : plain DiffCR (no bottleneck module)          [DiffCR folder]
  * quantum   : QuantumBottleneckLayer (PQC)                  [QDiffCR folder, mode=global]
  * control   : ClassicalBottleneckLayer (param-matched MLP)  [QDiffCR folder, mode=classical_control]

For every (model, resolution, seed):
  1. train n_epoch epochs, saving a checkpoint every SAVE_EVERY and val-loss every VAL_EVERY
  2. select the BEST-val-loss checkpoint  (early stopping by selection -> avoids the overfit tail)
  3. evaluate that checkpoint with EMA weights on the full test set (PSNR/SSIM/LPIPS)
  4. (optional) evaluate every saved checkpoint on a test subset -> quality-vs-epoch curve

Then aggregate across seeds (mean +/- std), run paired significance tests
(quantum vs classical, quantum vs control), and produce all figures.

FAIRNESS: identical data_root, frame_stride, seed, and splits across every model;
resolution and the bottleneck module are the only things that vary.

Usage:
    python run_all.py                        # full: seeds 0,1,2 ; 2000 epochs
    python run_all.py --seeds 0              # single seed (fastest full-quality pass)
    python run_all.py --n-epoch 2000 --seeds 0 1 2
    python run_all.py --quality-curve        # also make quality-vs-epoch curves (seed 0)
    python run_all.py --quick                # 2-epoch smoke test of the whole thing
"""
import argparse
import json
import os
import glob
import shutil
import subprocess
import sys
from collections import OrderedDict, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate_results import evaluate

PYTHON = sys.executable
DATA_ROOT = os.environ.get("CTGAN_DATA", "data/CTGAN/Sen2_MTC/dataset")
FRAME_STRIDE = 12
VAL_EVERY = 100
SAVE_EVERY = 100          # save a checkpoint every 100 epochs (aligned with val grid)
RESULTS = os.path.join(HERE, "results")

# model -> (folder, template config, quantum-block override or None)
MODELS = OrderedDict([
    ("classical", ("DiffCR",  "config/ours_sigmoid_w32.json", None)),
    ("quantum",   ("QDiffCR", "config/ours_sigmoid_w32_quantum.json",
                   {"enabled": True, "n_qubits": 4, "n_layers": 2, "mode": "global"})),
    ("quantum_matched", ("QDiffCR", "config/ours_sigmoid_w32_quantum.json",
                   {"enabled": True, "n_qubits": 5, "n_layers": 5, "mode": "quantum_matched"})),
    ("control",   ("QDiffCR", "config/ours_sigmoid_w32_quantum.json",
                   {"enabled": True, "n_qubits": 4, "n_layers": 2, "mode": "classical_control"})),
])
RESOLUTIONS = [32, 64]


def load(p):
    with open(p) as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def dump(cfg, p):
    with open(p, "w") as f:
        json.dump(cfg, f, indent=4)


def set_dataset_args(cfg, res, test_stride=None):
    for split in ["train", "val", "test"]:
        a = cfg["datasets"][split]["which_dataset"]["args"]
        a["data_root"] = DATA_ROOT
        a["image_size"] = res
        a.pop("frame_stride", None)
        if split == "train":
            a["frame_stride"] = FRAME_STRIDE
        elif split == "test" and test_stride:
            a["frame_stride"] = test_stride


def exp_set(folder):
    return set(glob.glob(os.path.join(folder, "experiments", "*")))


def newest_since(folder, before):
    new = sorted(exp_set(folder) - before, key=os.path.getmtime)
    return new[-1] if new else None


def saved_ckpt_epochs(train_root):
    eps = []
    for p in glob.glob(os.path.join(train_root, "checkpoint", "*_Network.pth")):
        b = os.path.basename(p)
        if "_ema" in b:
            continue
        try:
            eps.append(int(b.split("_")[0]))
        except ValueError:
            pass
    return sorted(eps)


def parse_val(train_log):
    import re
    txt = open(train_log).read()
    vl = [float(x) for x in re.findall(r'val/mse_loss:\s*([0-9.eE+-]+)', txt)]
    ve = [int(x) for x in re.findall(r'val_epoch:\s*(\d+)', txt)]
    n = min(len(vl), len(ve))
    return dict(zip(ve[:n], vl[:n]))


def best_ckpt_epoch(train_root):
    """Best-val-loss checkpoint among SAVED epochs (early stopping by selection)."""
    val = parse_val(os.path.join(train_root, "train.log"))
    saved = saved_ckpt_epochs(train_root)
    cands = [(val.get(e, float("inf")), e) for e in saved]
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def ema_prefix(train_root, epoch):
    """Copy <epoch>_Network_ema.pth -> <epoch>ema_Network.pth so load_network picks EMA."""
    ck = os.path.join(train_root, "checkpoint")
    src = os.path.join(ck, f"{epoch}_Network_ema.pth")
    if not os.path.exists(src):
        return os.path.join(ck, str(epoch))          # fall back to raw weights
    dst = os.path.join(ck, f"{epoch}ema_Network.pth")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    return os.path.join(ck, f"{epoch}ema")


def run_test(folder, template, quantum, res, seed, ckpt_prefix, tag, test_stride=None):
    cfg = load(os.path.join(folder, template))
    cfg["name"] = tag
    cfg["seed"] = seed
    cfg["path"]["resume_state"] = ckpt_prefix
    set_dataset_args(cfg, res, test_stride=test_stride)
    if quantum is not None:
        cfg["model"]["which_networks"][0]["args"]["unet"]["quantum"] = quantum
    cpath = f"config/_pipe/{tag}_test.json"
    dump(cfg, os.path.join(folder, cpath))
    before = exp_set(folder)
    subprocess.run([PYTHON, "run.py", "-c", cpath, "-p", "test"], cwd=folder, check=True)
    troot = newest_since(folder, before)
    for d in sorted(glob.glob(os.path.join(troot, "results", "*", "*")), key=os.path.getmtime, reverse=True):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "GT_*")):
            return d
    return None


def train_one(model, res, seed, n_epoch):
    folder_name, template, quantum = MODELS[model]
    folder = os.path.join(HERE, folder_name)
    tag = f"{model}_res{res}_seed{seed}"
    print(f"\n{'#'*72}\n# TRAIN {tag}\n{'#'*72}")
    cfg = load(os.path.join(folder, template))
    cfg["name"] = tag
    cfg["seed"] = seed
    cfg["path"]["resume_state"] = "None"
    set_dataset_args(cfg, res)
    if quantum is not None:
        cfg["model"]["which_networks"][0]["args"]["unet"]["quantum"] = quantum
    cfg["train"]["n_epoch"] = n_epoch
    cfg["train"]["val_loss_epoch"] = min(VAL_EVERY, n_epoch)
    cfg["train"]["val_epoch"] = n_epoch * 10          # skip expensive full-sampling val
    cfg["train"]["save_checkpoint_epoch"] = min(SAVE_EVERY, n_epoch)
    os.makedirs(os.path.join(folder, "config", "_pipe"), exist_ok=True)
    cpath = f"config/_pipe/{tag}_train.json"
    dump(cfg, os.path.join(folder, cpath))
    before = exp_set(folder)
    subprocess.run([PYTHON, "run.py", "-c", cpath, "-p", "train"], cwd=folder, check=True)
    return folder, template, quantum, newest_since(folder, before)


def main():
    global DATA_ROOT, FRAME_STRIDE, RESOLUTIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-epoch", type=int, default=2000)
    ap.add_argument("--quality-curve", action="store_true",
                    help="test every saved checkpoint (seed 0) for quality-vs-epoch curves")
    ap.add_argument("--quality-test-stride", type=int, default=4,
                    help="subsample test set for the quality curve to save time")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--data-root", default=DATA_ROOT, help="dataset root")
    ap.add_argument("--frame-stride", type=int, default=FRAME_STRIDE,
                    help="train frame subsample stride (1 = use all frames)")
    ap.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS)
    ap.add_argument("--run-name", default=None,
                    help="archive folder name under archive/ (default: auto from res/epochs/seeds/time)")
    args = ap.parse_args()
    DATA_ROOT = args.data_root
    FRAME_STRIDE = args.frame_stride
    RESOLUTIONS = args.resolutions
    n_epoch = 2 if args.quick else args.n_epoch
    use_lpips = not args.no_lpips
    os.makedirs(RESULTS, exist_ok=True)

    # default run-name for the archive folder
    import time
    run_name = args.run_name or (
        "res" + "-".join(map(str, RESOLUTIONS)) +
        f"_ep{n_epoch}_seeds" + "-".join(map(str, args.seeds)) +
        "_" + time.strftime("%y%m%d_%H%M%S"))

    # snapshot existing experiment dirs so we can archive ONLY this run's dirs later
    pre_exp = set()
    for m in MODELS:
        folder = os.path.join(HERE, MODELS[m][0])
        pre_exp |= set(glob.glob(os.path.join(folder, "experiments", "*")))

    # records[(model,res)] = list over seeds of dict(seed, metrics, best_epoch, train_log, val_at_best)
    records = defaultdict(list)
    quality = {}   # (model,res) -> list[(epoch, metrics)]  (seed 0 only)

    for res in RESOLUTIONS:
        for model in MODELS:
            for seed in args.seeds:
                try:
                    folder, template, quantum, troot = train_one(model, res, seed, n_epoch)
                    be = best_ckpt_epoch(troot)
                    val = parse_val(os.path.join(troot, "train.log"))
                    print(f"  best-val checkpoint: epoch {be} (val={val.get(be)})")
                    # EMA eval of best checkpoint on full test set
                    prefix = ema_prefix(troot, be)
                    rdir = run_test(folder, template, quantum, res, seed, prefix,
                                    f"{model}_res{res}_seed{seed}_best")
                    m = evaluate(rdir, use_lpips=use_lpips) if rdir else {}
                    records[(model, res)].append(
                        {"seed": seed, "metrics": m, "best_epoch": be,
                         "train_log": os.path.join(troot, "train.log"),
                         "val_at_best": val.get(be)})
                    print(f"  [{model}_res{res}_seed{seed}] best_ep={be} "
                          f"PSNR={m.get('psnr_mean',float('nan')):.3f} "
                          f"SSIM={m.get('ssim_mean',float('nan')):.4f} "
                          f"LPIPS={m.get('lpips_mean',float('nan')):.4f}")

                    # quality-vs-epoch curve (seed 0 only)
                    if args.quality_curve and seed == args.seeds[0]:
                        pts = []
                        for ep in saved_ckpt_epochs(troot):
                            pr = ema_prefix(troot, ep)
                            rd = run_test(folder, template, quantum, res, seed, pr,
                                          f"{model}_res{res}_ep{ep}_qc",
                                          test_stride=args.quality_test_stride)
                            mm = evaluate(rd, use_lpips=use_lpips) if rd else {}
                            pts.append((ep, mm))
                        quality[(model, res)] = pts
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] {model}_res{res}_seed{seed} failed: {e}")
                except Exception as e:
                    print(f"[ERROR] {model}_res{res}_seed{seed}: {e}")

    summarize(records, quality, use_lpips)

    # ---- archive EVERYTHING for this run into a labeled folder (never delete) ----
    archive_dir = os.path.join(HERE, "archive", run_name)
    os.makedirs(archive_dir, exist_ok=True)
    moved = 0
    for m in MODELS:
        folder = os.path.join(HERE, MODELS[m][0])
        cur = set(glob.glob(os.path.join(folder, "experiments", "*")))
        new_dirs = cur - pre_exp                      # only dirs created by THIS run
        arch_exp = os.path.join(archive_dir, MODELS[m][0], "experiments")
        os.makedirs(arch_exp, exist_ok=True)
        for d in new_dirs:
            dest = os.path.join(arch_exp, os.path.basename(d))
            if not os.path.exists(dest):
                shutil.move(d, dest)                  # move (preserves weights + train.log + test images)
                moved += 1
    # copy the summary figures/CSVs into the archive too
    arch_results = os.path.join(archive_dir, "results")
    os.makedirs(arch_results, exist_ok=True)
    for f in glob.glob(os.path.join(RESULTS, "*")):
        if os.path.isfile(f):
            shutil.copy(f, arch_results)
    print(f"\nARCHIVED {moved} experiment dirs (weights + logs + test images) + results to:")
    print(f"   {archive_dir}")
    print("\nPIPELINE COMPLETE. Results in", RESULTS, "| full run archived in", archive_dir)


def summarize(records, quality, use_lpips):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    metric_keys = ["psnr_mean", "ssim_mean"] + (["lpips_mean"] if use_lpips else [])

    # ---- summary table: mean +/- std over seeds ----
    lines = ["model,res,seeds,best_epoch_mean,psnr_mean,psnr_std,ssim_mean,ssim_std,lpips_mean,lpips_std"]
    agg = {}
    for (model, res), recs in sorted(records.items()):
        vals = {k: np.array([r["metrics"].get(k, np.nan) for r in recs]) for k in metric_keys}
        best_eps = np.array([r["best_epoch"] for r in recs if r["best_epoch"]])
        agg[(model, res)] = {k: (np.nanmean(v), np.nanstd(v)) for k, v in vals.items()}
        agg[(model, res)]["best_epoch_mean"] = best_eps.mean() if len(best_eps) else float("nan")
        def ms(k):
            a = agg[(model, res)].get(k, (np.nan, np.nan))
            return f"{a[0]},{a[1]}"
        lines.append(f'{model},{res},{len(recs)},{agg[(model,res)]["best_epoch_mean"]},'
                     f'{ms("psnr_mean")},{ms("ssim_mean")},'
                     f'{ms("lpips_mean") if use_lpips else ",,"}')
    with open(os.path.join(RESULTS, "SUMMARY_metrics.csv"), "w") as f:
        f.write("\n".join(lines))
    print("\n=== SUMMARY (mean over seeds) ===")
    print("\n".join(lines))

    # ---- convergence: epochs to reach val-loss thresholds (mean +/- std over seeds) ----
    import re
    def val_series(log):
        txt = open(log).read()
        vl = [float(x) for x in re.findall(r'val/mse_loss:\s*([0-9.eE+-]+)', txt)]
        ve = [int(x) for x in re.findall(r'val_epoch:\s*(\d+)', txt)]
        n = min(len(vl), len(ve)); return np.array(ve[:n]), np.array(vl[:n])
    def epochs_to(log, thr):
        e, l = val_series(log); idx = np.where(l < thr)[0]
        return int(e[idx[0]]) if len(idx) else np.nan
    conv_lines = ["model,res,threshold,epochs_mean,epochs_std"]
    for (model, res), recs in sorted(records.items()):
        for thr in [0.01, 0.005, 0.004, 0.0035]:
            ee = np.array([epochs_to(r["train_log"], thr) for r in recs], float)
            conv_lines.append(f"{model},{res},{thr},{np.nanmean(ee)},{np.nanstd(ee)}")
    with open(os.path.join(RESULTS, "SUMMARY_convergence.csv"), "w") as f:
        f.write("\n".join(conv_lines))

    # ---- paired significance: quantum vs classical, quantum vs control (per res) ----
    sig_lines = ["comparison,res,metric,mean_A,mean_B,wilcoxon_p_over_seeds"]
    for res in RESOLUTIONS:
        for a, b in [("quantum", "classical"), ("quantum", "control")]:
            ra, rb = records.get((a, res)), records.get((b, res))
            if not ra or not rb:
                continue
            for k in metric_keys:
                va = [r["metrics"].get(k, np.nan) for r in ra]
                vb = [r["metrics"].get(k, np.nan) for r in rb]
                p = np.nan
                if len(va) == len(vb) and len(va) >= 2:
                    try:
                        p = wilcoxon(va, vb).pvalue
                    except Exception:
                        pass
                sig_lines.append(f"{a}_vs_{b},{res},{k.replace('_mean','')},"
                                 f"{np.nanmean(va)},{np.nanmean(vb)},{p}")
    with open(os.path.join(RESULTS, "SUMMARY_significance.csv"), "w") as f:
        f.write("\n".join(sig_lines))

    # ---- bar chart of final metrics with error bars ----
    groups = [f"{m}_{r}" for r in RESOLUTIONS for m in MODELS if (m, r) in agg]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(6 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, metric_keys):
        means = [agg[(m, r)][k][0] for r in RESOLUTIONS for m in MODELS if (m, r) in agg]
        stds = [agg[(m, r)][k][1] for r in RESOLUTIONS for m in MODELS if (m, r) in agg]
        ax.bar(groups, means, yerr=stds, capsize=4)
        ax.set_title(k.replace("_mean", "").upper()); ax.tick_params(axis="x", rotation=40)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "SUMMARY_metrics_bar.png"), dpi=140); plt.close()

    # ---- quality-vs-epoch curves ----
    if quality:
        for k in metric_keys:
            plt.figure(figsize=(9, 6))
            for (model, res), pts in sorted(quality.items()):
                eps = [e for e, _ in pts]; ys = [mm.get(k, np.nan) for _, mm in pts]
                plt.plot(eps, ys, marker="o", label=f"{model}_res{res}")
            plt.xlabel("epoch"); plt.ylabel(k.replace("_mean", "")); plt.grid(alpha=0.3); plt.legend()
            plt.title(f"{k.replace('_mean','').upper()} vs training epoch (seed {0})")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, f"QUALITY_{k.replace('_mean','')}_vs_epoch.png"), dpi=140)
            plt.close()

    # ---- train & validation curves: all variants, mean +/- std over seeds ----
    def series(log, which):
        txt = open(log).read()
        if which == "train":
            v = [float(x) for x in re.findall(r'train/mse_loss:\s*([0-9.eE+-]+)', txt)]
            e = [int(x) for x in re.findall(r'(?<!val_)epoch:\s*(\d+)', txt)]
        else:
            v = [float(x) for x in re.findall(r'val/mse_loss:\s*([0-9.eE+-]+)', txt)]
            e = [int(x) for x in re.findall(r'val_epoch:\s*(\d+)', txt)]
        n = min(len(v), len(e))
        return dict(zip(e[:n], v[:n]))
    colors = {"classical": "#1f77b4", "quantum": "#17becf", "control": "#e377c2"}
    for which in ["train", "val"]:
        for res in RESOLUTIONS:
            plt.figure(figsize=(9, 6))
            any_data = False
            for model in MODELS:
                recs = records.get((model, res))
                if not recs:
                    continue
                ds = [series(r["train_log"], which) for r in recs]
                eps = sorted(set().union(*[set(d.keys()) for d in ds])) if ds else []
                if not eps:
                    continue
                mean = np.array([np.nanmean([d.get(e, np.nan) for d in ds]) for e in eps])
                std = np.array([np.nanstd([d.get(e, np.nan) for d in ds]) for e in eps])
                c = colors.get(model, None)
                plt.plot(eps, mean, color=c, lw=2, label=model,
                         marker="o" if which == "val" else None, ms=3)
                plt.fill_between(eps, mean - std, mean + std, color=c, alpha=0.2)
                any_data = True
            if not any_data:
                plt.close(); continue
            plt.yscale("log"); plt.xlabel("epoch")
            plt.ylabel(f"{which} MSE loss (mean +/- std over seeds)")
            plt.title(f"{which.capitalize()} loss @ {res}x{res} (all variants)")
            plt.grid(True, which="both", alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, f"{which.upper()}_curves_res{res}.png"), dpi=140)
            plt.close()

    # ---- loss curves (all runs) ----
    cmd = [PYTHON, os.path.join(HERE, "plot_curves.py")]
    for (model, res), recs in sorted(records.items()):
        cmd += ["--run", f'{model}_res{res}:{recs[0]["train_log"]}']
    cmd += ["--out", os.path.join(RESULTS, "ALL_loss_curves.png")]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("loss curve plot failed:", e)


if __name__ == "__main__":
    main()
