#!/bin/bash
# Test a trained model and compute PSNR/SSIM/LPIPS metrics.
#
# Usage:
#   ./scripts/test_and_evaluate.sh --checkpoint PATH [--data PATH] [--image-size N]
#
# Examples:
#   ./scripts/test_and_evaluate.sh --checkpoint experiments/train_hqconv_additive_.../checkpoint/2000
#   ./scripts/test_and_evaluate.sh --checkpoint /path/to/checkpoint/2000 --data /path/to/dataset

set -e
cd "$(dirname "$0")/.."

# Defaults
CHECKPOINT=""
DATA="data/CTGAN/Sen2_MTC/dataset"
IMAGE_SIZE=64
MODE="hqconv_encoder"
ENCODER_STAGE=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --data) DATA="$2"; shift 2;;
        --image-size) IMAGE_SIZE="$2"; shift 2;;
        --mode) MODE="$2"; shift 2;;
        --encoder-stage) ENCODER_STAGE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: --checkpoint is required"
    echo "Usage: ./scripts/test_and_evaluate.sh --checkpoint experiments/.../checkpoint/2000"
    exit 1
fi

echo "=== Test & Evaluate ==="
echo "  Checkpoint: $CHECKPOINT"
echo "  Mode:       $MODE"
echo "  Data:       $DATA"
echo "  Image size: ${IMAGE_SIZE}×${IMAGE_SIZE}"
echo ""

# Build quantum config based on mode
if [ "$MODE" = "hqconv_encoder" ]; then
    QUANTUM_CFG="{\"enabled\": true, \"mode\": \"hqconv_encoder\", \"encoder_stage\": ${ENCODER_STAGE}, \"n_layers\": 3}"
elif [ "$MODE" = "hqconv_bottleneck" ]; then
    QUANTUM_CFG="{\"enabled\": true, \"mode\": \"hqconv_bottleneck\", \"n_layers\": 3}"
else
    echo "ERROR: Unknown mode '$MODE'. Use 'hqconv_encoder' or 'hqconv_bottleneck'."
    exit 1
fi

# Generate test config
TEST_CFG="config/_pipe/test_$(basename "$CHECKPOINT").json"
mkdir -p config/_pipe

python -c "
import json, os
from collections import OrderedDict

cfg = json.load(open('config/ours_sigmoid_w32_hqconv.json'), object_pairs_hook=OrderedDict)
cfg['name'] = 'test_eval'
cfg['path']['resume_state'] = '${CHECKPOINT}'

for split in ['train', 'val', 'test']:
    if split in cfg['datasets']:
        cfg['datasets'][split]['which_dataset']['args']['data_root'] = '${DATA}'
        cfg['datasets'][split]['which_dataset']['args']['image_size'] = ${IMAGE_SIZE}

cfg['model']['which_networks'][0]['args']['unet']['quantum'] = json.loads('${QUANTUM_CFG}')

os.makedirs(os.path.dirname('${TEST_CFG}'), exist_ok=True)
json.dump(cfg, open('${TEST_CFG}', 'w'), indent=4)
print(f'Test config: ${TEST_CFG}')
"

echo ""
echo "--- Running test (DPM-Solver++ 20-step sampling) ---"
python run.py -c "$TEST_CFG" -p test

# Find the results directory
RESULTS_DIR=$(find experiments/test_test_eval_*/results/test/0 -maxdepth 0 -type d 2>/dev/null | sort | tail -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: No test results found"
    exit 1
fi

echo ""
echo "--- Computing metrics ---"
echo "Results dir: $RESULTS_DIR"
NUM_IMAGES=$(ls "$RESULTS_DIR"/GT_*.png 2>/dev/null | wc -l)
echo "Image pairs: $NUM_IMAGES"
echo ""

python -c "
import glob, os, numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

result_dir = '${RESULTS_DIR}'
gt_paths = sorted(glob.glob(os.path.join(result_dir, 'GT_*')))
pairs = []
for gt_path in gt_paths:
    out_path = os.path.join(result_dir, 'Out_' + os.path.basename(gt_path)[3:])
    if os.path.exists(out_path):
        pairs.append((gt_path, out_path))

psnr_list, ssim_list = [], []
for gt_path, out_path in pairs:
    gt = np.array(Image.open(gt_path).convert('RGB'))
    out = np.array(Image.open(out_path).convert('RGB'))
    psnr_list.append(psnr(gt, out, data_range=255))
    ssim_list.append(ssim(gt, out, channel_axis=-1, data_range=255))

print(f'  PSNR:  {np.mean(psnr_list):.4f} +/- {np.std(psnr_list):.4f}')
print(f'  SSIM:  {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}')

try:
    import torch, lpips
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_list = []
    for gt_path, out_path in pairs:
        gt = np.array(Image.open(gt_path).convert('RGB'))
        out = np.array(Image.open(out_path).convert('RGB'))
        gt_t = torch.from_numpy(gt).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1.0
        out_t = torch.from_numpy(out).float().permute(2,0,1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            lpips_list.append(lpips_fn(gt_t.to(device), out_t.to(device)).item())
    print(f'  LPIPS: {np.mean(lpips_list):.4f} +/- {np.std(lpips_list):.4f}  (lower is better)')
except ImportError:
    print('  LPIPS: skipped (pip install lpips)')
"
