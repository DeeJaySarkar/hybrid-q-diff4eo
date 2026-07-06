# DiffCR: Conditional Diffusion Model for Cloud Removal (Classical Baseline)

Classical DiffCR baseline — the unmodified architecture before quantum integration.

Based on [DiffCR](https://github.com/XavierJiezworkin/DiffCR).

## Training

```bash
python run.py -c config/ours_sigmoid_w32.json -p train
```

## Architecture

NAFNet-based conditional UNet with double encoder and split channel attention, using DPM-Solver++ for fast sampling.
