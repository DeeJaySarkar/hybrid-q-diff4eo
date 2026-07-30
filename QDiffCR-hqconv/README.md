# QDiffCR-HQConv: Quanvolutional Diffusion Model with HQConv Ansatz

A quantum-enhanced cloud removal diffusion model using **NesyaLab's HQConv ansatz** — the
best-performing quanvolutional circuit from [Quantum-Hybrid-Diffusion-Models-for-EO](https://github.com/NesyaLab/Quantum-Hybrid-Diffusion-Models-for-EO) — adapted for the DiffCR architecture.

## Key Differences from QDiffCR-basic

| | QDiffCR-basic (global) | QDiffCR-HQConv (this) |
|---|---|---|
| **Placement** | UNet bottleneck | **Encoder stage 2** (16×16 spatial) |
| **Input to circuit** | Global-pooled channel vector | **2×2 spatial patches × 2 channels** |
| **Qubits** | 4 | **8** |
| **Ansatz** | BasicEntanglerLayers (RX + CNOT ring) | **HQConv** (CRZ+CRX intra-group + inter-group) |
| **Circuit layers** | 2 | **3** |
| **Quantum params** | 8 | **84** |
| **Spatial awareness** | None | **Yes** (processes local 2×2 patches) |
| **Circuit calls/sample** | 1 | 64 (at 64×64 input) |

## Architecture

```
Input [B,12,64,64]
  ↓
Encoder Stage 0: NAFBlock (32ch, 64×64)  →  downsample
Encoder Stage 1: NAFBlock (64ch, 32×32)  →  downsample
Encoder Stage 2: NAFBlock (128ch, 16×16) →  ★ HQConv Layer ★  →  downsample
Encoder Stage 3: NAFBlock (256ch, 8×8)   →  downsample
  ↓
Bottleneck: NAFBlock (512ch, 4×4)
  ↓
Decoder (with skip connections) → Output [B,3,64,64]
```

## HQConv Circuit (8 qubits, 84 params)

```
|0⟩ ─ RX(x₀) ─┬─ CRZ ─ CRX ─┬─────────── CRZ ─ CRX ─┬─ ... ─ ⟨Z⟩
|1⟩ ─ RX(x₁) ─┤  block_A     ├─────────── CRZ ─ CRX ─┤        ⟨Z⟩
|2⟩ ─ RX(x₂) ─┤  (intra-     ├─────────── CRZ ─ CRX ─┤        ⟨Z⟩
|3⟩ ─ RX(x₃) ─┴─ group)      ┴──┐        ╱            │        ⟨Z⟩
|4⟩ ─ RX(x₄) ─┬─ CRZ ─ CRX ─┬──┤ block_B             │        ⟨Z⟩
|5⟩ ─ RX(x₅) ─┤  block_A     ├──┤ (inter-group)       │        ⟨Z⟩
|6⟩ ─ RX(x₆) ─┤  (intra-     ├──┤                     │        ⟨Z⟩
|7⟩ ─ RX(x₇) ─┴─ group)      ┴──┘                     ┴        ⟨Z⟩
                                    × 3 layers
```

## Setup & Training

```bash
pip install -r requirements.txt

# Training (64×64, full dataset)
python run.py -c config/ours_sigmoid_w32_hqconv.json -p train
```

## Configuration

```json
"quantum": {
    "enabled": true,
    "mode": "hqconv_encoder",
    "encoder_stage": 2,
    "n_layers": 3
}
```

- `encoder_stage`: which encoder stage to place the quanvolution (2 = 16×16 at 64×64 input)
- `n_layers`: number of HQConv ansatz repetitions (3 = 84 quantum params)

## Requirements

- PyTorch ≥ 1.6 with CUDA
- PennyLane ≥ 0.38.0
- NVIDIA GPU

## Reference

HQConv ansatz from: [NesyaLab/Quantum-Hybrid-Diffusion-Models-for-EO](https://github.com/NesyaLab/Quantum-Hybrid-Diffusion-Models-for-EO)
