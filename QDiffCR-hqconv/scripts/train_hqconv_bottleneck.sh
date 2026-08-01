#!/bin/bash
# Train HQConv BOTTLENECK variant (4×4 spatial, 4 patches)
#
# This places the 8-qubit HQConv quanvolutional layer at the UNet bottleneck,
# processing 2×2 patches over the 4×4 feature map (4 quantum circuit calls
# per forward pass). Faster than additive but fewer spatial patches.
#
# Usage:
#   ./scripts/train_hqconv_bottleneck.sh [--seed SEED] [--epochs N] [--batch BATCH] [--data PATH]
#
# Examples:
#   ./scripts/train_hqconv_bottleneck.sh                         # defaults: seed=1, 3000 epochs, batch=4
#   ./scripts/train_hqconv_bottleneck.sh --seed 42 --epochs 2000
#   ./scripts/train_hqconv_bottleneck.sh --data /path/to/CTGAN_full_64/CTGAN/Sen2_MTC/dataset

set -e
cd "$(dirname "$0")/.."

# Defaults
SEED=1
N_EPOCH=3000
BATCH=4
DATA="data/CTGAN/Sen2_MTC/dataset"
IMAGE_SIZE=64

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2;;
        --epochs) N_EPOCH="$2"; shift 2;;
        --batch) BATCH="$2"; shift 2;;
        --data) DATA="$2"; shift 2;;
        --image-size) IMAGE_SIZE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

TAG="hqconv_bottleneck_${IMAGE_SIZE}x${IMAGE_SIZE}_seed${SEED}"
CFG="config/_pipe/${TAG}_train.json"

echo "=== HQConv Bottleneck (4×4 spatial) ==="
echo "  Seed:       $SEED"
echo "  Epochs:     $N_EPOCH"
echo "  Batch size: $BATCH"
echo "  Image size: ${IMAGE_SIZE}×${IMAGE_SIZE}"
echo "  Data:       $DATA"
echo "  Config:     $CFG"
echo ""

# Generate training config
mkdir -p config/_pipe
python -c "
import json, os
from collections import OrderedDict

cfg = json.load(open('config/ours_sigmoid_w32_hqconv_bottleneck.json'), object_pairs_hook=OrderedDict)
cfg['name'] = '${TAG}'
cfg['seed'] = ${SEED}
cfg['path']['resume_state'] = 'None'

for split in ['train', 'val', 'test']:
    if split in cfg['datasets']:
        cfg['datasets'][split]['which_dataset']['args']['data_root'] = '${DATA}'
        cfg['datasets'][split]['which_dataset']['args']['image_size'] = ${IMAGE_SIZE}

cfg['datasets']['train']['dataloader']['args']['batch_size'] = ${BATCH}
cfg['train']['n_epoch'] = ${N_EPOCH}
cfg['train']['save_checkpoint_epoch'] = 100
cfg['train']['val_loss_epoch'] = 100

os.makedirs(os.path.dirname('${CFG}'), exist_ok=True)
json.dump(cfg, open('${CFG}', 'w'), indent=4)
print(f'Config written: ${CFG}')
"

# Check for existing checkpoint to resume from
LATEST=$(ls experiments/train_${TAG}_*/checkpoint/*_Network.pth 2>/dev/null \
    | xargs -n1 basename 2>/dev/null \
    | grep -v ema | grep -oE "^[0-9]+" | sort -n | tail -1)

if [ -n "$LATEST" ]; then
    CKPT_DIR=$(find experiments/train_${TAG}_*/checkpoint -name "${LATEST}_Network.pth" 2>/dev/null | head -1 | xargs dirname)
    CKPT_PATH="$(realpath "$CKPT_DIR/$LATEST")"
    echo "Resuming from checkpoint: epoch $LATEST"
    python -c "
import json
from collections import OrderedDict
c = json.load(open('${CFG}'), object_pairs_hook=OrderedDict)
c['path']['resume_state'] = '${CKPT_PATH}'
json.dump(c, open('${CFG}', 'w'), indent=4)
"
else
    echo "Starting fresh training"
fi

echo ""
python run.py -c "$CFG" -p train
