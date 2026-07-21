#!/usr/bin/env python3
"""
Segmentation script for ADNI follow-up T1w dataset using MoE_foundation_v3.

Input:
  /labs/wanglab/projects/lifespan-T1T2FA/ADNI/Skull-stripped_fellowup/
  Processes T1w skull-stripped scans only (*_T1w_brain.nii.gz).

Output (mirrors input structure):
  /opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue/us_other/ADNI/

Demographics:
  /labs/wanglab/projects/lifespan-T1T2FA/ADNI/demographics_fellowup.csv
  columns: sub_id, ses_id, dataset, age, gender, ...

Usage:
    python test_segmentation_adni_t1.py [--checkpoint <path>] [--resume]
"""

import sys
import os
# Ensure v3 modules are imported (not v1), regardless of working directory
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
from data.preprocess import reorient_to_ras
import torchio as tio


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DATA   = Path("/labs/wanglab/projects/lifespan-T1T2FA")
OUTPUT_ROOT = Path("/opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue_final/us_other/ADNI_AD_add")
CHECKPOINT  = "/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_v3/segmentation_finetune_checkpoints_final/PROJ/Lifespan_Segmentation/best_model.pth"
DEFAULT_CONFIG = "/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_v3/cfg/lifespan_segmentation.yaml"

# Per-dataset configuration
DATASET_CONFIGS = {
    'ADNI_fellowup_T1': {
        'input_dir':  BASE_DATA / 'ADNI' / 'Resampled_sMRI_add',
        'output_dir': OUTPUT_ROOT / 'T1w',
        'demo_file':  BASE_DATA / 'ADNI' / 'demographics_sMRI_followup.csv',
    },
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_age_group(age_years: float) -> str:
    """
    Map age in years to age group string matching model training groups.
      fetal:    < 0   (gestational, not used here)
      neonatal: 0 – 0.25 y
      infant:   0.25 – 2 y
      child:    2 – 18 y
      adult:    18 – 65 y
      elderly:  > 65 y
    """
    if age_years is None:
        return 'adult'
    if age_years < 0:
        return 'fetal'
    if age_years < 0.25:
        return 'neonatal'
    if age_years < 2:
        return 'infant'
    if age_years < 18:
        return 'child'
    if age_years <= 65:
        return 'adult'
    return 'elderly'


def _safe_age(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_age_lookup(demo_file: Path) -> dict:
    """Return {(sub_id, ses_id): age_group} from demographics.csv."""
    if not demo_file.exists():
        print(f"  Warning: demographics not found: {demo_file}")
        return {}
    df = pd.read_csv(demo_file)
    lookup = {}
    for _, row in df.iterrows():
        sub = str(row['sub_id']).strip()
        ses = str(row['ses_id']).strip()
        age = _safe_age(row['age'])
        lookup[(sub, ses)] = get_age_group(age)
    return lookup


def is_wanted_modality(filename):
    """Return True for T1w scans only."""
    stem = filename.replace('.nii.gz', '').replace('.nii', '')
    for part in stem.split('_'):
        if part.startswith('T1'):
            return True
    return False


def extract_modality_from_filename(filename):
    """Return 'T1w', 'T2w', 'FA', or 'MD' based on filename."""
    stem = filename.replace('.nii.gz', '').replace('.nii', '')
    for part in stem.split('_'):
        if part.startswith('T1'):
            return part
        if part.startswith('T2'):
            return part
        if part == 'FA':
            return 'FA'
        if part == 'MD':
            return 'MD'
    return 'T1w'


def infer_single_scan(model, cfg, scan_data, modality=None, age_group=None):
    model.eval()
    pad_flag = False
    x, y, z = cfg.data.patch_size

    if cfg.data.normalize:
        scan_data = util.norm_img(scan_data, cfg.data.norm_perc)
        if not np.isfinite(scan_data).all():
            raise ValueError("Normalization produced NaN/Inf (likely constant image)")

    if min(scan_data.shape) < min(x, y, z):
        x_ori_size, y_ori_size, z_ori_size = scan_data.shape
        pad_flag = True
        x_diff = x - x_ori_size
        y_diff = y - y_ori_size
        z_diff = z - z_ori_size
        scan_data = np.pad(
            scan_data,
            ((max(0, int(x_diff/2)), max(0, x_diff - int(x_diff/2))),
             (max(0, int(y_diff/2)), max(0, y_diff - int(y_diff/2))),
             (max(0, int(z_diff/2)), max(0, z_diff - int(z_diff/2)))),
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

            gx0 = patch_idx[0] if (bound[1] - bound[0] < x) else bound[0]
            gx1 = patch_idx[1] if (bound[1] - bound[0] < x) else bound[1]
            gy0 = patch_idx[2] if (bound[3] - bound[2] < y) else bound[2]
            gy1 = patch_idx[3] if (bound[3] - bound[2] < y) else bound[3]
            gz0 = patch_idx[4] if (bound[5] - bound[4] < z) else bound[4]
            gz1 = patch_idx[5] if (bound[5] - bound[4] < z) else bound[5]

            sbj = tio.Subject(
                one_image=tio.ScalarImage(
                    tensor=global_scan[:, gx0:gx1, gy0:gy1, gz0:gz1]
                ),
                a_segmentation=tio.LabelMap(
                    tensor=location[:, gx0:gx1, gy0:gy1, gz0:gz1]
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
            max(0, int(x_diff/2)): max(0, int(x_diff/2)) + x_ori_size,
            max(0, int(y_diff/2)): max(0, int(y_diff/2)) + y_ori_size,
            max(0, int(z_diff/2)): max(0, int(z_diff/2)) + z_ori_size
        ]
        assert pred_vol.shape == (x_ori_size, y_ori_size, z_ori_size)

    return pred_vol


# ---------------------------------------------------------------------------
# Dataset walker
# ---------------------------------------------------------------------------

def collect_files(input_dir: Path):
    """Return list of (img_path, relative_path) for T1w brain .nii.gz files (excluding masks)."""
    entries = []
    for nii in sorted(input_dir.rglob("*_T1w_brain.nii.gz")):
        entries.append((nii, nii.relative_to(input_dir)))
    return entries


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(dataset_name, cfg, model, resume):
    dcfg       = DATASET_CONFIGS[dataset_name]
    input_dir  = dcfg['input_dir']
    output_dir = dcfg['output_dir']
    demo_file  = dcfg['demo_file']

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Demo  : {demo_file}")

    age_lookup = load_age_lookup(demo_file)
    if age_lookup:
        groups = set(age_lookup.values())
        print(f"  Age groups found: {groups}  ({len(age_lookup)} subjects)")
    else:
        print("  Warning: no demographics loaded, defaulting all to 'adult'")

    all_entries = collect_files(input_dir)
    print(f"  Images found on disk: {len(all_entries)}")

    # Only process subjects/sessions listed in demographics CSV
    if age_lookup:
        entries = [
            (p, r) for p, r in all_entries
            if (r.parts[0] if len(r.parts) >= 1 else '', r.parts[1] if len(r.parts) >= 2 else '') in age_lookup
        ]
        print(f"  Images after demographics filter: {len(entries)}")
    else:
        entries = all_entries
        print(f"  Images: {len(entries)}")

    skipped_exist = 0
    skipped_err   = 0
    processed     = 0

    for img_path, rel_path in tqdm(entries, desc=dataset_name):
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip non-target modalities (AD, RD, etc.)
        if not is_wanted_modality(img_path.name):
            continue

        if resume and out_path.exists():
            skipped_exist += 1
            continue

        sub_id    = rel_path.parts[0] if len(rel_path.parts) >= 1 else ''
        ses_id    = rel_path.parts[1] if len(rel_path.parts) >= 2 else ''
        age_group = age_lookup.get((sub_id, ses_id), 'adult')

        try:
            img_nii  = reorient_to_ras(nib.load(img_path))
            img_data = np.squeeze(img_nii.get_fdata(dtype=np.float32))
        except Exception as e:
            tqdm.write(f"  [SKIP load] {rel_path}: {e}")
            skipped_err += 1
            continue

        img_data[img_data < 0] = 0
        modality = extract_modality_from_filename(img_path.name)

        if img_data.max() == img_data.min():
            tqdm.write(f"  [SKIP constant] {rel_path}: image is constant, skipping")
            skipped_err += 1
            continue

        try:
            pred_vol = infer_single_scan(model, cfg, img_data.copy(),
                                         modality=modality, age_group=age_group)
        except Exception as e:
            tqdm.write(f"  [SKIP infer] {rel_path}: {e}")
            skipped_err += 1
            continue

        pred_vol = pred_vol.astype(np.uint8)
        nib.save(nib.Nifti1Image(pred_vol, affine=img_nii.affine, header=img_nii.header),
                 str(out_path))
        processed += 1

    print(f"  Done — Processed: {processed}  |  Already existed: {skipped_exist}  |  Errors: {skipped_err}")
    return processed, skipped_exist, skipped_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Segment ADNI follow-up dataset (MoE_v3) — T1w only')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('--config',     type=str, default=None)
    parser.add_argument('--resume',     action='store_true',default=True,
                        help='Skip files whose output already exists')
    parser.add_argument('--dataset',    type=str, nargs='+',
                        choices=list(DATASET_CONFIGS.keys()),
                        default=list(DATASET_CONFIGS.keys()),
                        help='Which datasets to process (default: all)')
    args = parser.parse_args()

    # Load config
    cfg = get_cfg_defaults()
    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), 'train_cfg.yaml')
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
        print(f"Config: {config_path}")
    elif os.path.exists(DEFAULT_CONFIG):
        cfg.merge_from_file(DEFAULT_CONFIG)
        print(f"Config: {DEFAULT_CONFIG}")
    else:
        print("Warning: no config file found, using defaults")
    cfg.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print("Loading model...")
    model = MoESegModel(cfg, pretrained_encoder_path=None)
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.")

    total_processed = total_skipped_exist = total_skipped_err = 0
    for ds_name in args.dataset:
        p, se, serr = process_dataset(ds_name, cfg, model, resume=args.resume)
        total_processed     += p
        total_skipped_exist += se
        total_skipped_err   += serr

    print(f"\n{'='*60}")
    print(f"ALL DONE.  Processed: {total_processed}  |  "
          f"Already existed: {total_skipped_exist}  |  Errors: {total_skipped_err}")
    print(f"Results root: {OUTPUT_ROOT}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
