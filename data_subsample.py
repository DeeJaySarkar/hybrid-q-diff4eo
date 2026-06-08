from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
#%%
DATA_ROOT = Path("CTGAN_resized")
SPLIT_ROOT = Path("CTGAN/CTGAN/Sen2_MTC/dataset")
OUTPUT_ROOT = Path("CTGAN_resized_subsampled")

KEEP_FRACTION = 0.1
SEED = 42
random.seed(SEED)
#%%
def read_split_file(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
#%%
def process_split(split_name):
    split_file = SPLIT_ROOT / f"{split_name}.txt"
    tiles = read_split_file(split_file)

    print(f"\nProcessing split: {split_name} | tiles: {len(tiles)}")
    total_groups_kept = 0
    total_tiff_kept = 0

    for tile in tiles:
        tile_dir = DATA_ROOT / "CTGAN" / "Sen2_MTC" / "dataset" / "Sen2_MTC" / tile
        cloudless_dir = tile_dir / "cloudless"
        cloud_dir = tile_dir / "cloud"

        if not cloudless_dir.exists():
            print(f"Missing cloudless folder: {cloudless_dir}")
            continue

        cloudless_files = sorted([
            p for p in cloudless_dir.glob("*.tif")
            if not p.name.endswith(("_0.tif", "_1.tif", "_2.tif"))
        ])

        if not cloudless_files:
            print(f"No cloudless TIFFs in: {cloudless_dir}")
            continue

        k = max(1, int(len(cloudless_files) * KEEP_FRACTION))
        sampled = set(random.sample(cloudless_files, k))

        kept_here = 0

        for cloudless_src in sampled:
            base_name = cloudless_src.stem
            group_files = [
                cloudless_src,
                cloud_dir / f"{base_name}_0.tif",
                cloud_dir / f"{base_name}_1.tif",
                cloud_dir / f"{base_name}_2.tif",
            ]

            if not all(p.exists() for p in group_files):
                continue

            for src in group_files:
                rel = src.relative_to(DATA_ROOT)
                dst = OUTPUT_ROOT / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            kept_here += 1

        total_groups_kept += kept_here
        total_tiff_kept += kept_here * 4
        print(f"{tile}: kept {kept_here}/{len(cloudless_files)} groups")

    print(f"Total groups kept for {split_name}: {total_groups_kept}")
    print(f"Total TIFF files kept for {split_name}: {total_tiff_kept}")
    return total_groups_kept, total_tiff_kept
#%%
grand_groups = 0
grand_tiffs = 0

for split in ["train", "val", "test"]:
    g, t = process_split(split)
    grand_groups += g
    grand_tiffs += t

print(f"\nGrand total groups kept: {grand_groups}")
print(f"Grand total TIFF files kept: {grand_tiffs}")