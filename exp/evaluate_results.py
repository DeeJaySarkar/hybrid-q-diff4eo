"""
Robust evaluation for DiffCR / QDiffCR test outputs.

The test phase saves images to:
    experiments/<run>/results/test/<epoch>/GT_<name>.png
    experiments/<run>/results/test/<epoch>/Out_<name>.png

This script pairs GT_<name> with Out_<name> by name, computes per-image
PSNR and SSIM (modern scikit-image API), optionally LPIPS, and reports
mean +/- std so results can be compared across runs/seeds.

Usage:
    # Single results directory
    python evaluate_results.py --dir experiments/<run>/results/test/<epoch>

    # Compare two directories (e.g. classical vs quantum)
    python evaluate_results.py --dir <quantum_dir> --compare <classical_dir>

    # Disable LPIPS (if the package is unavailable)
    python evaluate_results.py --dir <dir> --no-lpips
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

# Modern scikit-image API (compare_psnr/compare_ssim were removed in >=0.18)
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _load(path):
    img = np.array(Image.open(path).convert("RGB"))
    return img


def _pair_images(result_dir):
    """Return list of (gt_path, out_path) pairs matched by suffix name."""
    gt_paths = glob.glob(os.path.join(result_dir, "GT_*"))
    pairs = []
    for gt_path in sorted(gt_paths):
        fname = os.path.basename(gt_path)
        out_path = os.path.join(result_dir, "Out_" + fname[len("GT_"):])
        if os.path.exists(out_path):
            pairs.append((gt_path, out_path))
    return pairs


def evaluate(result_dir, use_lpips=True):
    pairs = _pair_images(result_dir)
    if not pairs:
        raise RuntimeError(
            f"No GT_/Out_ image pairs found in {result_dir}. "
            "Make sure you point at results/test/<epoch>."
        )

    lpips_fn = None
    device = None
    if use_lpips:
        try:
            import torch
            import lpips
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        except Exception as e:
            print(f"[warn] LPIPS disabled ({e}); reporting PSNR/SSIM only.")
            lpips_fn = None

    psnr_list, ssim_list, lpips_list = [], [], []
    for gt_path, out_path in pairs:
        gt = _load(gt_path)
        out = _load(out_path)
        if gt.shape != out.shape:
            # Resize Out to GT if needed (shouldn't happen, but be safe)
            out = np.array(Image.fromarray(out).resize((gt.shape[1], gt.shape[0])))

        psnr_list.append(sk_psnr(gt, out, data_range=255))
        ssim_list.append(
            sk_ssim(gt, out, channel_axis=-1, gaussian_weights=True,
                    use_sample_covariance=False, sigma=1.5, data_range=255)
        )

        if lpips_fn is not None:
            import torch
            # to [-1,1] CHW tensors
            def to_t(x):
                t = torch.from_numpy(x).float().permute(2, 0, 1) / 127.5 - 1.0
                return t.unsqueeze(0).to(device)
            with torch.no_grad():
                lpips_list.append(lpips_fn(to_t(gt), to_t(out)).item())

    result = {
        "n": len(pairs),
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
    }
    if lpips_list:
        result["lpips_mean"] = float(np.mean(lpips_list))
        result["lpips_std"] = float(np.std(lpips_list))
    return result


def _print_result(tag, r):
    print(f"\n=== {tag} ===")
    print(f"  images:  {r['n']}")
    print(f"  PSNR:    {r['psnr_mean']:.4f} +/- {r['psnr_std']:.4f}")
    print(f"  SSIM:    {r['ssim_mean']:.4f} +/- {r['ssim_std']:.4f}")
    if "lpips_mean" in r:
        print(f"  LPIPS:   {r['lpips_mean']:.4f} +/- {r['lpips_std']:.4f}  (lower is better)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="results/test/<epoch> directory")
    parser.add_argument("--compare", default=None, help="second results dir to compare")
    parser.add_argument("--no-lpips", action="store_true", help="skip LPIPS")
    args = parser.parse_args()

    use_lpips = not args.no_lpips

    r1 = evaluate(args.dir, use_lpips=use_lpips)
    _print_result(args.dir, r1)

    if args.compare:
        r2 = evaluate(args.compare, use_lpips=use_lpips)
        _print_result(args.compare, r2)

        print("\n=== DELTA (dir - compare) ===")
        print(f"  PSNR:  {r1['psnr_mean'] - r2['psnr_mean']:+.4f}  (higher favors --dir)")
        print(f"  SSIM:  {r1['ssim_mean'] - r2['ssim_mean']:+.4f}  (higher favors --dir)")
        if "lpips_mean" in r1 and "lpips_mean" in r2:
            print(f"  LPIPS: {r1['lpips_mean'] - r2['lpips_mean']:+.4f}  (lower favors --dir)")
