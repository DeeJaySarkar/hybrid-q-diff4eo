"""
Multi-seed significance analysis for low-data experiments.
Compares Classical vs HQConv BN at 5% and 10% data across seeds 1-4.
Reports mean ± std, paired t-test, and effect size (Cohen's d).
"""
import glob
import os
import numpy as np
from scipy import stats
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def evaluate_dir(d):
    gt_paths = sorted(glob.glob(os.path.join(d, 'GT_*')))
    pairs = [(g, os.path.join(d, 'Out_' + os.path.basename(g)[3:])) for g in gt_paths]
    pairs = [(g, o) for g, o in pairs if os.path.exists(o)]
    if len(pairs) < 687:
        return None
    psnrs, ssims = [], []
    for g, o in pairs:
        gt = np.array(Image.open(g).convert('RGB'))
        out = np.array(Image.open(o).convert('RGB'))
        psnrs.append(sk_psnr(gt, out, data_range=255))
        ssims.append(sk_ssim(gt, out, channel_axis=-1, data_range=255))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2) / (na+nb-2))
    if pooled_std == 0:
        return 0
    return (np.mean(a) - np.mean(b)) / pooled_std


def main():
    seeds = [1, 2, 3, 4]

    print("=" * 75)
    print("MULTI-SEED SIGNIFICANCE ANALYSIS: Classical vs HQConv Bottleneck")
    print("32×32, 1000 epochs, seeds 1-4")
    print("=" * 75)

    for pct in [5, 10]:
        cl_psnrs, hq_psnrs = [], []
        sc_psnrs, qb_psnrs = [], []
        cl_ssims, hq_ssims = [], []

        print(f"\n{'─' * 75}")
        print(f"  {pct}% DATA ({int(2380 * pct / 100)} training frames)")
        print(f"{'─' * 75}")

        for seed in seeds:
            # Classical
            cl_dirs = sorted(glob.glob(
                f'/home/deejay/exp/DiffCR/experiments/test_classical_32x32_{pct}pct_seed{seed}_ep1000_*/results/test/0'))
            if cl_dirs:
                r = evaluate_dir(cl_dirs[-1])
                if r:
                    cl_psnrs.append(r[0])
                    cl_ssims.append(r[1])

            # HQConv BN
            hq_dirs = sorted(glob.glob(
                f'/home/deejay/QDiffCR-hqconv/experiments/test_hqconv_bn_32x32_{pct}pct_seed{seed}_ep1000_*/results/test/0'))
            if hq_dirs:
                r = evaluate_dir(hq_dirs[-1])
                if r:
                    hq_psnrs.append(r[0])
                    hq_ssims.append(r[1])

            # Spatial control
            sc_dirs = sorted(glob.glob(
                f'/home/deejay/QDiffCR-hqconv/experiments/test_spatial_ctrl_32x32_{pct}pct_seed{seed}_ep1000_*/results/test/0'))
            if sc_dirs:
                r = evaluate_dir(sc_dirs[-1])
                if r:
                    sc_psnrs.append(r[0])

            # Quantum basic (global pool)
            qb_dirs = sorted(glob.glob(
                f'/home/deejay/QDiffCR-hqconv/experiments/test_quantum_basic_32x32_{pct}pct_seed{seed}_ep1000_*/results/test/0'))
            if qb_dirs:
                r = evaluate_dir(qb_dirs[-1])
                if r:
                    qb_psnrs.append(r[0])

        print(f"\n  Results (PSNR, mean ± std):")
        if cl_psnrs:
            print(f"    Classical:        {np.mean(cl_psnrs):6.3f} ± {np.std(cl_psnrs):.3f}  (n={len(cl_psnrs)})")
        if sc_psnrs:
            print(f"    Spatial control:  {np.mean(sc_psnrs):6.3f} ± {np.std(sc_psnrs):.3f}  (n={len(sc_psnrs)})")
        if qb_psnrs:
            print(f"    Quantum basic:    {np.mean(qb_psnrs):6.3f} ± {np.std(qb_psnrs):.3f}  (n={len(qb_psnrs)})")
        if hq_psnrs:
            print(f"    HQConv BN:        {np.mean(hq_psnrs):6.3f} ± {np.std(hq_psnrs):.3f}  (n={len(hq_psnrs)})")

        # Pairwise t-tests
        if len(cl_psnrs) >= 2 and len(hq_psnrs) >= 2:
            print(f"\n  Pairwise comparisons (independent t-test):")
            pairs = [
                ("HQConv BN vs Classical", hq_psnrs, cl_psnrs),
                ("HQConv BN vs Spatial ctrl", hq_psnrs, sc_psnrs),
                ("Spatial ctrl vs Classical", sc_psnrs, cl_psnrs),
                ("Quantum basic vs Classical", qb_psnrs, cl_psnrs),
            ]
            for label, a, b in pairs:
                if len(a) >= 2 and len(b) >= 2:
                    t_stat, p_val = stats.ttest_ind(a, b)
                    d = cohens_d(a, b)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    print(f"    {label:<30}: delta={np.mean(a)-np.mean(b):+.3f} dB, p={p_val:.4f} {sig}, d={d:.2f}")

    print(f"\n{'=' * 75}")


if __name__ == "__main__":
    main()
