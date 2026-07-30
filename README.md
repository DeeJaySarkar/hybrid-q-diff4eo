# Quantum-Enhanced Diffusion Models for Satellite Cloud Removal

This repository investigates whether parameterized quantum circuits (PQCs) can improve conditional diffusion models for cloud removal in Sentinel-2 imagery.

## Architecture

Built on **DiffCR** (Zou et al., 2024) — a NAFNet-based conditional UNet with DPM-Solver++ for fast 20-step sampling — we integrate quantum layers inspired by the **QHD-EO** architecture (Mauro et al., 2025).

## Repository Structure

```
DiffCR-basic/       Classical baseline (unmodified DiffCR, width=32)
QDiffCR-basic/      Quantum variant (4-qubit PQC at UNet bottleneck)
data_resize.ipynb   Dataset preprocessing (256x256 -> target resolution)
data_subsample.py   Frame subsampling utility (10% subsample, seed 42)
multiseed_check.py  Multi-seed sanity check script
```

## Variants

| Variant | Architecture | Quantum Params |
|---------|-------------|----------------|
| DiffCR-basic | NAFNet UNet, double encoder, split channel attention | None |
| QDiffCR-basic | Same + QuantumBottleneckLayer | 4 qubits, 8 params (BasicEntanglerLayers) |

The quantum layer acts as a global feature modulator at the bottleneck:
`Global Pool -> Linear(512->4) -> PQC (AngleEmbedding + BasicEntangler) -> Linear(4->512) -> Residual Add`

## Dataset

Training uses the **Sen2_MTC** dataset (CTGAN) with multi-temporal cloudy/cloud-free Sentinel-2 pairs. Experiments run at 32x32 and 64x64 resolutions with various data fractions.

- Sen2_MTC_New: [CTGAN.zip](https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view?usp=share_link)

## Quick Start

```bash
# Install dependencies
pip install -r QDiffCR-basic/requirements.txt

# Train the quantum variant
cd QDiffCR-basic
python run.py -c config/ours_sigmoid_w32_quantum.json -p train

# Train the classical baseline
cd DiffCR-basic
python run.py -c config/ours_sigmoid_w32.json -p train

# Test
python run.py -c config/ours_sigmoid_w32.json -p test
```

**Note:** Update `data_root` in the config JSON to point to your local dataset path before training.

## References

- **DiffCR:** Zou et al., "DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal from Optical Satellite Images," IEEE TGRS, 2024. [Paper](https://arxiv.org/abs/2308.04417) | [Code](https://github.com/XavierJiezou/DiffCR)
- **QHD-EO:** Mauro et al., "Quantum Hybrid Diffusion Models for Earth Observation," 2025. [Paper](https://arxiv.org/abs/2512.20448) | [Code](https://github.com/NesyaLab/Quantum-Hybrid-Diffusion-Models-for-EO)

## Requirements

- PyTorch >= 1.6 with CUDA
- PennyLane >= 0.38.0 (for quantum variant)
- pennylane-lightning[gpu] (optional, for GPU-accelerated circuit simulation)
