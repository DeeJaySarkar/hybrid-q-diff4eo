# QDiffCR: Quantum-Enhanced Diffusion Model for Cloud Removal

A hybrid quantum-classical diffusion model for satellite image cloud removal. Extends [DiffCR](https://github.com/XavierJiezworkin/DiffCR) by inserting a parameterized quantum circuit (PQC) at the UNet bottleneck.

## Architecture

```
Input → Cond Encoder + Main Encoder → Classical Bottleneck → Quantum Layer → Decoder → Output
```

**Quantum Layer:** Global Avg Pool → Linear(512→4) → 4-qubit PQC → Linear(4→512) → Residual Add

- 4 qubits, AngleEmbedding (RX) + BasicEntanglerLayers
- GPU-accelerated simulation via PennyLane `lightning.gpu` (NVIDIA cuQuantum)
- Differentiable via adjoint method — full end-to-end backpropagation

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python run.py -c config/ours_sigmoid_w32_quantum.json -p train
```

## Configuration

In `config/ours_sigmoid_w32_quantum.json`, the quantum layer is controlled by:

```json
"quantum": {
    "enabled": true,
    "n_qubits": 4,
    "n_layers": 2
}
```

## Files Changed from Classical DiffCR

| File | Change |
|------|--------|
| `models/ours/quantum_layer.py` | **New** — QuantumBottleneckLayer (43 lines) |
| `models/ours/nafnet_..._quantum.py` | **New** — UNet + quantum layer at bottleneck |
| `models/network_x0_dpm_solver.py` | 2 lines added (register new module_name) |
| `config/ours_sigmoid_w32_quantum.json` | **New** — config with quantum params |
| `requirements.txt` | 2 lines added (pennylane, pennylane-lightning[gpu]) |

## Requirements

- PyTorch ≥ 1.6 with CUDA
- PennyLane ≥ 0.38.0
- pennylane-lightning[gpu]
- NVIDIA GPU with CUDA

## Acknowledgements

Based on [DiffCR](https://github.com/XavierJiezworkin/DiffCR).
