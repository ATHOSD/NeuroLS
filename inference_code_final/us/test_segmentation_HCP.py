#!/usr/bin/env python3
"""
Segmentation of T1w and T2w images for HCP-A, HCP-D, HCP-YA using MoE_foundation_v3.

Input (skull-stripped per dataset):
  HCP-A:  /labs/wanglab/projects/lifespan-T1T2FA/HCP-A/Skull-stripped/sub-{id}/ses-01/anat/
  HCP-D:  /labs/wanglab/projects/lifespan-T1T2FA/HCP-D/Skull-stripped/sub-{id}/ses-01/anat/
  HCP-YA: /labs/wanglab/projects/lifespan-T1T2FA/HCP-YA/Skull-stripped/sub-{id}/ses-01/anat/

Output:
  HCP-A  -> /opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue/HCP/HCP-A/
  HCP-D  -> /opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue/HCP/HCP-D/
  HCP-YA -> /opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue/HCP/HCP-YA/

Usage:
    python test_segmentation_HCP.py [--checkpoint <path>] [--datasets HCP-A HCP-D HCP-YA]
"""

import sys
import os
_V3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _V3_ROOT not in sys.path:
    sys.path.insert(0, _V3_ROOT)

import torch
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from cfg.lifespan_config import get_cfg_defaults
from model.segmentation_model import MoESegModel
from utils import util
from data.lifespan_datasets import parse_age_to_years, age_years_to_group
import torchio as tio


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

RESULT_ROOT = Path("/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/Result/HCP")

DATASET_CFG = {
    "HCP-A": {
        "skull_stripped": Path("/labs/wanglab/projects/lifespan-T1T2FA/HCP-A/Resampled"),
        "demographics":   Path("/labs/wanglab/projects/lifespan-T1T2FA/HCP-A/demographics.csv"),
        "output_dir":     RESULT_ROOT / "HCP-A",
    },
    "HCP-D": {
        "skull_stripped": Path("/labs/wanglab/projects/lifespan-T1T2FA/HCP-D/Resampled"),
        "demographics":   Path("/labs/wanglab/projects/lifespan-T1T2FA/HCP-D/demographics.csv"),
        "output_dir":     RESULT_ROOT / "HCP-D",
    },
    "HCP-YA": {
        "skull_stripped": Path("/labs/wanglab/projects/lifespan-T1T2FA/HCP-YA/Resampled"),
        "demographics":   Path("/labs/wanglab/projects/lifespan-T1T2FA/HCP-YA/demographics.csv"),
        "output_dir":     RESULT_ROOT / "HCP-YA",
    },
}

CHECKPOINT     = "/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/segmentation_checkpoints/best_model.pth"
DEFAULT_CONFIG = "/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/cfg/lifespan_segmentation.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_modality(filename):
    """Extract T1w or T2w from filename. Returns None for non-T1/T2 files."""
    stem = filename.replace(".nii.gz", "")
    for part in stem.split("_"):
        if part.startswith("T1"):
            return "T1w"
        if part.startswith("T2"):
            return "T2w"
    return None


def get_sub_id(filepath):
    """Get sub_id from the grandparent directory name (sub-{id}/ses-01/anat/)."""
    return filepath.parent.parent.parent.name


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_single_scan(model, cfg, scan_data, modality="T1w", age_group="adult"):
    model.eval()
    pad_flag = False
    x, y, z = cfg.data.patch_size

    if cfg.data.normalize:
        scan_data = util.norm_img(scan_data, cfg.data.norm_perc)

    if min(scan_data.shape) < min(x, y, z):
        x_ori_size, y_ori_size, z_ori_size = scan_data.shape
        pad_flag = True
        x_diff = x - x_ori_size
        y_diff = y - y_ori_size
        z_diff = z - z_ori_size
        scan_data = np.pad(
            scan_data,
            ((max(0, int(x_diff / 2)), max(0, x_diff - int(x_diff / 2))),
             (max(0, int(y_diff / 2)), max(0, y_diff - int(y_diff / 2))),
             (max(0, int(z_diff / 2)), max(0, z_diff - int(z_diff / 2)))),
            constant_values=1e-4
        )

    pred     = np.zeros((cfg.train.cls_num,) + scan_data.shape)
    tmp_norm = np.zeros((cfg.train.cls_num,) + scan_data.shape)

    scan_patches, _, tmp_idx = util.patch_slicer(
        scan_data, scan_data, cfg.data.patch_size,
        (x - 16, y - 16, z - 16),
        remove_bg=cfg.data.remove_bg, test=True, ori_path=None
    )

    bound       = util.get_bounds(torch.from_numpy(scan_data))
    global_scan = torch.unsqueeze(torch.from_numpy(scan_data).to(dtype=torch.float), dim=0)

    with torch.no_grad():
        for idx, patch in enumerate(scan_patches):
            ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
            ipt = ipt.reshape((1, 1,) + ipt.shape)

            patch_idx = tmp_idx[idx]
            location  = torch.zeros_like(torch.from_numpy(scan_data)).float()
            location  = torch.unsqueeze(location, 0)
            location[:, patch_idx[0]:patch_idx[1],
                        patch_idx[2]:patch_idx[3],
                        patch_idx[4]:patch_idx[5]] = 1

            sbj = tio.Subject(
                one_image=tio.ScalarImage(
                    tensor=global_scan[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]
                ),
                a_segmentation=tio.LabelMap(
                    tensor=location[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]
                )
            )
            transforms = tio.transforms.Resize(target_shape=(x, y, z))
            sbj        = transforms(sbj)
            down_scan  = sbj['one_image'].data
            loc        = sbj['a_segmentation'].data

            tmp_coor = util.get_bounds(loc)
            coordinates_A = np.array([
                np.floor(tmp_coor[0] / 4), np.ceil(tmp_coor[1] / 4),
                np.floor(tmp_coor[2] / 4), np.ceil(tmp_coor[3] / 4),
                np.floor(tmp_coor[4] / 4), np.ceil(tmp_coor[5] / 4)
            ]).astype(int)
            coordinates_A = torch.unsqueeze(torch.from_numpy(coordinates_A), 0)

            tmp_pred, _ = model(ipt, down_scan.cuda().reshape([1, 1, x, y, z]),
                                coordinates_A, modality=modality, age_group=age_group)

            patch_slice = (slice(0, cfg.train.cls_num),) + (
                slice(patch_idx[0], patch_idx[1]),
                slice(patch_idx[2], patch_idx[3]),
                slice(patch_idx[4], patch_idx[5])
            )
            pred[patch_slice]     += torch.squeeze(tmp_pred).detach().cpu().numpy()
            tmp_norm[patch_slice] += 1

    pred[tmp_norm > 0] = pred[tmp_norm > 0] / tmp_norm[tmp_norm > 0]

    sf       = torch.nn.Softmax(dim=0)
    pred_vol = sf(torch.from_numpy(pred)).numpy()
    pred_vol = np.argmax(pred_vol, axis=0)

    if pad_flag:
        pred_vol = pred_vol[
            max(0, int(x_diff / 2)): max(0, int(x_diff / 2)) + x_ori_size,
            max(0, int(y_diff / 2)): max(0, int(y_diff / 2)) + y_ori_size,
            max(0, int(z_diff / 2)): max(0, int(z_diff / 2)) + z_ori_size
        ]
        assert pred_vol.shape == (x_ori_size, y_ori_size, z_ori_size)

    return pred_vol


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def test_dataset(model, cfg, dataset_name, resume=True):
    dcfg       = DATASET_CFG[dataset_name]
    ss_root    = dcfg["skull_stripped"]
    demo_path  = dcfg["demographics"]
    output_dir = dcfg["output_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset:    {dataset_name}")
    print(f"Input root: {ss_root}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}")

    # Load demographics
    demo_lookup = {}
    if demo_path.exists():
        df = pd.read_csv(demo_path)
        for _, row in df.iterrows():
            demo_lookup[str(row['sub_id'])] = row
        print(f"  Loaded {len(demo_lookup)} subjects from demographics")
    else:
        print(f"  [WARNING] Demographics not found: {demo_path}")

    # Collect T1w and T2w files (exclude masks)
    img_files = sorted(
        p for p in ss_root.rglob("*.nii.gz")
        if not p.name.endswith("_mask.nii.gz")
        and get_modality(p.name) in ("T1w", "T2w")
    )
    print(f"  Found {len(img_files)} T1w/T2w images")

    processed = skipped_exist = skipped_err = 0

    for img_path in tqdm(img_files, desc=dataset_name):
        out_path = output_dir / img_path.name

        if resume and out_path.exists():
            skipped_exist += 1
            continue

        modality = get_modality(img_path.name)
        sub_id   = get_sub_id(img_path)

        demo_row  = demo_lookup.get(sub_id)
        if demo_row is not None:
            age_group = age_years_to_group(parse_age_to_years(demo_row.get('age', None)))
        else:
            age_group = 'adult'
            tqdm.write(f"  [NO DEMO] {img_path.name} ({sub_id}), defaulting age_group=adult")

        try:
            img_nii  = nib.load(img_path)
            img_data = np.squeeze(img_nii.get_fdata())
        except Exception as e:
            tqdm.write(f"  [SKIP load] {img_path.name}: {e}")
            skipped_err += 1
            continue

        img_data[img_data < 0] = 0

        try:
            pred_vol = infer_single_scan(model, cfg, img_data.copy(),
                                         modality=modality, age_group=age_group)
        except Exception as e:
            tqdm.write(f"  [SKIP infer] {img_path.name}: {e}")
            skipped_err += 1
            continue

        nib.save(
            nib.Nifti1Image(pred_vol.astype(np.uint8), affine=img_nii.affine, header=img_nii.header),
            str(out_path)
        )
        processed += 1

    print(f"  Done: processed={processed}  skipped(exist)={skipped_exist}  errors={skipped_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Segment T1w/T2w images for HCP-A/D/YA (MoE_v3)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--datasets",   type=str, nargs="+",
                        default=["HCP-A", "HCP-D","HCP-YA"],
                        choices=["HCP-A", "HCP-D","HCP-YA"])
    parser.add_argument("--resume",     action="store_true", default=True,
                        help="Skip files whose output already exists (default: True)")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), "train_cfg.yaml")
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
        print(f"Config: {config_path}")
    elif os.path.exists(DEFAULT_CONFIG):
        cfg.merge_from_file(DEFAULT_CONFIG)
        print(f"Config: {DEFAULT_CONFIG}")
    else:
        print("Warning: no config file found, using defaults")
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print("Loading model...")
    model = MoESegModel(cfg, pretrained_encoder_path=None)
    model = model.to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded.\n")

    for dataset in args.datasets:
        test_dataset(model, cfg, dataset, resume=args.resume)

    print(f"\n{'='*60}")
    print("All done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
