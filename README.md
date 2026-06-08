# Quantum-Enhanced Cloud-Conditioned Diffusion Models for Cloud Removal in Sentinel Imagery: A Proof of Concept Study

This project is primarily a bridge between two distinct lines of work. Built on DiffCR (Zou et al., 2023) — a fast conditional diffusion framework for cloud removal using temporally adjacent multi-temporal Sentinel-2 imagery — we integrate quanvolutional layers inspired by the QCU-Net architecture (Mauro et al., 2025) into DiffCR's condition encoder.

Both papers and their GitHub are available at:

DiffCR

Paper: https://arxiv.org/abs/2308.04417
Code: https://github.com/XavierJiezou/DiffCR

QHD-EO

Paper: https://arxiv.org/abs/2512.20448
Code: https://github.com/NesyaLab/Quantum-Hybrid-Diffusion-Models-for-EO

Training and evaluation are performed on a subsampled Sen2_MTC dataset (CTGAN.zip from DiffCR repo) at 32×32 resolution, deliberately constrained to reflect the low-data scenarios where quantum feature extractors are hypothesised to offer the greatest advantage. The primary comparison metric is convergence speed — specifically, how many training epochs each variant requires to reach equivalent validation MAE — alongside final MAE, PSNR, and SSIM.
