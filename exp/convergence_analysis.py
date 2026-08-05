"""
Convergence analysis for low-data fraction experiments.

For each experiment, tests the model at multiple checkpoint epochs and computes
PSNR/SSIM/LPIPS. Produces:
1. Convergence curves: PSNR vs epoch for each model at each data fraction
2. Data efficiency plot: final PSNR vs data fraction for classical vs quantum
3. Convergence speed: epoch at which each model reaches X% of its final PSNR

Usage:
    python convergence_analysis.py                    # auto-find experiments
    python convergence_analysis.py --epochs 200 500 1000 1500 2000
    python convergence_analysis.py --fractions 5 10 20
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, "configs")
LOG_DIR = os.path.join(HERE, "logs")
RESULTS_DIR = os.path.join(HERE, "results")

CLASSICAL_CODE = "/home/deejay/exp/DiffCR"
HQCONV_CODE = "/home/deejay/QDiffCR-hqconv"

DATA_ROOT = "/home/deejay/CTGAN_full_32/CTGAN/Sen2_MTC/dataset"


def find_experiment_dir(code_dir, name_pattern):
    """Find the most recent experiment directory matching a pattern."""
    matches = sorted(glob.glob(os.path.join(code_dir, "experiments", f"train_{name_pattern}_*")))
    return matches[-1] if matches else None


def get_checkpoints(exp_dir, target_epochs=None):
    """List available checkpoint epochs in an experiment."""
    ckpt_dir = os.path.join(exp_dir, "checkpoint")
    if not os.path.isdir(ckpt_dir):
        return []
    epochs = set()
    for f in os.listdir(ckpt_dir):
        if f.endswith("_Network_ema.pth"):
            ep = f.split("_")[0]
            if ep.isdigit():
                epochs.add(int(ep))
    epochs = sorted(epochs)
    if target_epochs:
        epochs = [e for e in epochs if e in target_epochs]
    return epochs


def run_test(code_dir, config_path, resume_state, test_tag):
    """Run test phase and return the results directory."""
    # Build a test config
    cfg = json.load(open(config_path), object_pairs_hook=OrderedDict)
    cfg['name'] = test_tag
    cfg['path']['resume_state'] = resume_state

    test_cfg_path = os.path.join(HERE, "configs", f"_test_{test_tag}.json")
    json.dump(cfg, open(test_cfg_path, 'w'), indent=4)

    # Run test
    result = subprocess.run(
        [PYTHON, "run.py", "-c", test_cfg_path, "-p", "test"],
        cwd=code_dir, capture_output=True, text=True, timeout=600
    )

    if result.returncode != 0:
        print(f"  [WARN] Test failed for {test_tag}: {result.stderr[-200:]}")
        return None

    # Find results
    test_dirs = sorted(glob.glob(os.path.join(code_dir, "experiments", f"test_{test_tag}_*")))
    if not test_dirs:
        return None
    results_path = os.path.join(test_dirs[-1], "results", "test", "0")
    if os.path.isdir(results_path):
        return results_path
    return None


def evaluate_dir(results_dir):
    """Compute PSNR, SSIM, LPIPS for a results directory."""
    gt_paths = sorted(glob.glob(os.path.join(results_dir, "GT_*")))
    pairs = []
    for gt_path in gt_paths:
        out_path = os.path.join(results_dir, "Out_" + os.path.basename(gt_path)[3:])
        if os.path.exists(out_path):
            pairs.append((gt_path, out_path))

    if not pairs:
        return None

    psnr_list, ssim_list = [], []
    for gt_path, out_path in pairs:
        gt = np.array(Image.open(gt_path).convert("RGB"))
        out = np.array(Image.open(out_path).convert("RGB"))
        psnr_list.append(sk_psnr(gt, out, data_range=255))
        ssim_list.append(sk_ssim(gt, out, channel_axis=-1, data_range=255))

    result = {
        "n": len(pairs),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
    }

    # LPIPS if available
    try:
        import torch
        import lpips
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        lpips_list = []
        for gt_path, out_path in pairs:
            gt = np.array(Image.open(gt_path).convert("RGB"))
            out = np.array(Image.open(out_path).convert("RGB"))
            gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            out_t = torch.from_numpy(out).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            with torch.no_grad():
                lpips_list.append(lpips_fn(gt_t.to(device), out_t.to(device)).item())
        result["lpips_mean"] = float(np.mean(lpips_list))
        result["lpips_std"] = float(np.std(lpips_list))
    except ImportError:
        pass

    return result


def convergence_speed(epoch_metrics, threshold_pct=90):
    """Find the epoch at which the model reaches threshold% of its final PSNR."""
    if not epoch_metrics:
        return None
    final_psnr = epoch_metrics[-1][1]["psnr_mean"]
    target = final_psnr * threshold_pct / 100.0
    for ep, m in epoch_metrics:
        if m["psnr_mean"] >= target:
            return ep
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--epochs", nargs="+", type=int, default=[100, 200, 500, 1000, 1500, 2000])
    parser.add_argument("--skip-test", action="store_true", help="Only analyze existing results")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = [
        ("classical", CLASSICAL_CODE, "classical_32x32_{pct}pct_seed1"),
        ("hqconv_bn", HQCONV_CODE, "hqconv_bn_32x32_{pct}pct_seed1"),
    ]

    all_results = {}

    for model_name, code_dir, name_template in models:
        all_results[model_name] = {}

        for pct in args.fractions:
            tag = name_template.format(pct=pct)
            exp_dir = find_experiment_dir(code_dir, tag)

            if not exp_dir:
                print(f"[SKIP] No experiment found for {tag}")
                continue

            print(f"\n{'='*60}")
            print(f"Model: {model_name}, Data: {pct}%, Experiment: {os.path.basename(exp_dir)}")
            print(f"{'='*60}")

            checkpoints = get_checkpoints(exp_dir, target_epochs=args.epochs)
            print(f"  Available checkpoints: {checkpoints}")

            epoch_metrics = []
            for ep in checkpoints:
                ckpt_path = os.path.join(exp_dir, "checkpoint", str(ep))
                test_tag = f"{tag}_ep{ep}"

                if args.skip_test:
                    # Look for existing test results
                    test_dirs = sorted(glob.glob(os.path.join(code_dir, "experiments", f"test_{test_tag}_*")))
                    if test_dirs:
                        results_path = os.path.join(test_dirs[-1], "results", "test", "0")
                        if os.path.isdir(results_path):
                            metrics = evaluate_dir(results_path)
                            if metrics:
                                epoch_metrics.append((ep, metrics))
                                print(f"  ep{ep}: PSNR={metrics['psnr_mean']:.3f}, SSIM={metrics['ssim_mean']:.4f}")
                    continue

                print(f"  Testing epoch {ep}...", end=" ", flush=True)
                cfg_path = os.path.join(CFG_DIR, f"{tag.split('_seed')[0]}_seed1.json")
                results_path = run_test(code_dir, cfg_path, ckpt_path, test_tag)

                if results_path:
                    metrics = evaluate_dir(results_path)
                    if metrics:
                        epoch_metrics.append((ep, metrics))
                        print(f"PSNR={metrics['psnr_mean']:.3f}, SSIM={metrics['ssim_mean']:.4f}")
                    else:
                        print("no pairs found")
                else:
                    print("test failed")

            all_results[model_name][pct] = epoch_metrics

            # Convergence speed
            speed_90 = convergence_speed(epoch_metrics, 90)
            if speed_90:
                print(f"  90% convergence at epoch: {speed_90}")

    # Save raw results
    results_file = os.path.join(RESULTS_DIR, "convergence_data.json")
    # Convert to serializable format
    serializable = {}
    for model, fracs in all_results.items():
        serializable[model] = {}
        for pct, epoch_list in fracs.items():
            serializable[model][str(pct)] = [(ep, m) for ep, m in epoch_list]
    json.dump(serializable, open(results_file, 'w'), indent=2)
    print(f"\nRaw results saved to: {results_file}")

    # Generate summary
    print_summary(all_results, args.fractions)

    # Plot if matplotlib available
    try:
        plot_results(all_results, args.fractions)
    except ImportError:
        print("\n[INFO] Install matplotlib for plots: pip install matplotlib")


def print_summary(all_results, fractions):
    """Print a comparison table."""
    print("\n" + "=" * 70)
    print("SUMMARY: Final PSNR (epoch 2000) by data fraction")
    print("=" * 70)
    print(f"{'Data %':<10}{'Classical':<25}{'HQConv Bottleneck':<25}{'Delta':<10}")
    print("-" * 70)

    for pct in fractions:
        cl_metrics = all_results.get("classical", {}).get(pct, [])
        hq_metrics = all_results.get("hqconv_bn", {}).get(pct, [])

        cl_final = cl_metrics[-1][1]["psnr_mean"] if cl_metrics else None
        hq_final = hq_metrics[-1][1]["psnr_mean"] if hq_metrics else None

        cl_str = f"{cl_final:.3f}" if cl_final else "pending"
        hq_str = f"{hq_final:.3f}" if hq_final else "pending"
        delta = f"{hq_final - cl_final:+.3f}" if (cl_final and hq_final) else "—"

        print(f"{pct}%{'':<7}{cl_str:<25}{hq_str:<25}{delta:<10}")

    print("\n" + "=" * 70)
    print("CONVERGENCE SPEED: Epoch to reach 90% of final PSNR")
    print("=" * 70)
    print(f"{'Data %':<10}{'Classical':<20}{'HQConv':<20}{'Speedup':<15}")
    print("-" * 70)

    for pct in fractions:
        cl_metrics = all_results.get("classical", {}).get(pct, [])
        hq_metrics = all_results.get("hqconv_bn", {}).get(pct, [])

        cl_speed = convergence_speed(cl_metrics, 90)
        hq_speed = convergence_speed(hq_metrics, 90)

        cl_str = f"ep {cl_speed}" if cl_speed else "pending"
        hq_str = f"ep {hq_speed}" if hq_speed else "pending"
        if cl_speed and hq_speed and hq_speed > 0:
            speedup = f"{cl_speed / hq_speed:.2f}x"
        else:
            speedup = "—"

        print(f"{pct}%{'':<7}{cl_str:<20}{hq_str:<20}{speedup:<15}")


def plot_results(all_results, fractions):
    """Generate convergence and data efficiency plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {"classical": "tab:blue", "hqconv_bn": "tab:red"}
    labels = {"classical": "Classical", "hqconv_bn": "HQConv Bottleneck"}

    # Plot 1-3: convergence curves per fraction
    for idx, pct in enumerate(fractions):
        ax = axes[idx]
        for model in ["classical", "hqconv_bn"]:
            metrics = all_results.get(model, {}).get(pct, [])
            if metrics:
                epochs = [ep for ep, _ in metrics]
                psnrs = [m["psnr_mean"] for _, m in metrics]
                ax.plot(epochs, psnrs, 'o-', color=colors[model], label=labels[model])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"{pct}% data ({int(2380 * pct / 100)} frames)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "convergence_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nConvergence plot saved: {out_path}")
    plt.close()

    # Data efficiency plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for model in ["classical", "hqconv_bn"]:
        final_psnrs = []
        valid_fracs = []
        for pct in fractions:
            metrics = all_results.get(model, {}).get(pct, [])
            if metrics:
                final_psnrs.append(metrics[-1][1]["psnr_mean"])
                valid_fracs.append(pct)
        if final_psnrs:
            ax.plot(valid_fracs, final_psnrs, 'o-', color=colors[model],
                    label=labels[model], markersize=8)

    ax.set_xlabel("Training data fraction (%)")
    ax.set_ylabel("PSNR at epoch 2000 (dB)")
    ax.set_title("Data Efficiency: Classical vs Quantum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(fractions)

    out_path = os.path.join(RESULTS_DIR, "data_efficiency.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Data efficiency plot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
