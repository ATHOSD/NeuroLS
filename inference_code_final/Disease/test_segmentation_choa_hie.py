#!/usr/bin/env python3
"""
Segmentation script for CHOA-HIE dataset using MoE_foundation_v3.

Input:
  /labs/wanglab/projects/lifespan-T1T2FA/CHOA-hie/Resampled/
  Structure: sub-{subject_id}/ses-01/sub-{subject_id}_ses-01_T1w_*.nii.gz

Output (mirrors input structure):
  /opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue/choa_disease/hie/

Demographics:
  /labs/wanglab/projects/lifespan-T1T2FA/CHOA-hie/demographics.csv
  columns: subject_id, sex, birth_date, study_date, age_at_scan_days, accession_number

All subjects are neonatal age group (0-59 days), all scans are T1w.

Usage:
    python test_segmentation_choa_hie.py [--checkpoint <path>] [--resume]
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

INPUT_DIR      = Path("/labs/wanglab/projects/lifespan-T1T2FA/CHOA-hie/Resampled")
OUTPUT_DIR     = Path("/opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue/choa_disease/hie")
DEMO_PATH      = Path("/labs/wanglab/projects/lifespan-T1T2FA/CHOA-hie/demographics.csv")
CHECKPOINT     = "/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_v3/segmentation_choa_finetune_checkpoints_v2/PROJ/Lifespan_Segmentation/model_final.pth"
DEFAULT_CONFIG = "/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_v3/cfg/lifespan_segmentation.yaml"

# Fixed for all HIE scans
FIXED_AGE_GROUP = 'neonatal'
FIXED_MODALITY  = 'T1w'


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def load_demo_lookup(demo_path: Path) -> dict:
    """Return {subject_id_str: {'sex', 'age_at_scan_days'}} ."""
    if not demo_path.exists():
        print(f"  Warning: demographics not found: {demo_path}")
        return {}
    df = pd.read_csv(demo_path)
    lookup = {}
    for _, row in df.iterrows():
        try:
            subj_key = str(int(float(row['subject_id'])))
        except (ValueError, TypeError):
            continue
        lookup[subj_key] = {
            'sex':              str(row['sex']).strip()              if pd.notna(row['sex'])              else None,
            'age_at_scan_days': float(row['age_at_scan_days'])       if pd.notna(row['age_at_scan_days']) else None,
        }
    return lookup


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_single_scan(model, cfg, scan_data, modality=None, age_group=None):
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
# Processing
# ---------------------------------------------------------------------------

def process_hie(cfg, model, resume):
    print(f"\n{'='*60}")
    print(f"Dataset: CHOA-HIE")
    print(f"  Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Demo  : {DEMO_PATH}")
    print(f"  Age group: {FIXED_AGE_GROUP}  |  Modality: {FIXED_MODALITY}")

    demo_lookup = load_demo_lookup(DEMO_PATH)
    print(f"  Demographics subjects loaded: {len(demo_lookup)}")

    # Collect all T1w images: sub-{id}/ses-01/sub-{id}_ses-01_T1w_*.nii.gz
    entries = sorted(INPUT_DIR.glob("sub-*/ses-01/*_T1w_*.nii.gz"))
    print(f"  T1w images found: {len(entries)}")

    skipped_exist = 0
    skipped_err   = 0
    processed     = 0

    for img_path in tqdm(entries, desc="HIE"):
        # subject id from folder name "sub-102476002644841" -> "102476002644841"
        subj_folder = img_path.parent.parent.name   # e.g. "sub-102476002644841"
        subj_id     = subj_folder.replace("sub-", "")
        ses_folder  = img_path.parent.name           # "ses-01"
        out_path    = OUTPUT_DIR / subj_folder / ses_folder / (img_path.stem.replace(".nii", "") + "_seg.nii.gz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and out_path.exists():
            skipped_exist += 1
            continue

        demo = demo_lookup.get(subj_id, {})
        if not demo:
            tqdm.write(f"  [WARN] {subj_id}: not in demographics, processing anyway")

        try:
            img_nii  = reorient_to_ras(nib.load(str(img_path)))
            img_data = np.squeeze(img_nii.get_fdata(dtype=np.float32))
        except Exception as e:
            tqdm.write(f"  [SKIP load] {scan_id}: {e}")
            skipped_err += 1
            continue

        img_data[img_data < 0] = 0

        if img_data.max() == img_data.min():
            tqdm.write(f"  [SKIP constant] {scan_id}: image is constant")
            skipped_err += 1
            continue

        try:
            pred_vol = infer_single_scan(model, cfg, img_data.copy(),
                                         modality=FIXED_MODALITY,
                                         age_group=FIXED_AGE_GROUP)
        except Exception as e:
            tqdm.write(f"  [SKIP infer] {scan_id}: {e}")
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
        description='Segment CHOA-HIE dataset (MoE_v3) — T1w neonatal')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('--config',     type=str, default=None)
    parser.add_argument('--resume',     action='store_true', default=True,
                        help='Skip files whose output already exists')
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.")

    p, se, serr = process_hie(cfg, model, resume=args.resume)

    print(f"\n{'='*60}")
    print(f"ALL DONE.  Processed: {p}  |  Already existed: {se}  |  Errors: {serr}")
    print(f"Results: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
