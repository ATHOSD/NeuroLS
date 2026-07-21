#!/usr/bin/env python3
"""
Segmentation script for LNSP and SPINS datasets using MoE_foundation_v3.

Inputs  : /labs/wanglab/projects/lifespan-T1T2FA/{LNSP,SPINS}/Resampled/
Outputs : /opt/localdata/data/usr-envs/ruiying/Code/foundation/result_tissue/disease/{LNSP,SPINS}/
          (directory structure mirrored from Resampled/)

Demographics:
  SPINS : /labs/wanglab/projects/lifespan-T1T2FA/SPINS/demographics.csv  (sub_id, ses_id, age)
  LNSP  : /labs/wanglab/projects/lifespan-T1T2FA/LNSP/demographics.csv   (sub_id, ses_id, age)

Usage:
    python test_segmentation_disease.py [--checkpoint <path>] [--resume]
                                        [--dataset LNSP SPINS]
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
BASE_DATA   = Path('/labs/wanglab/projects/lifespan-T1T2FA')
OUTPUT_ROOT = Path('/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/Result/disease')
CHECKPOINT  = '/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/segmentation_checkpoints/best_model.pth'
DEFAULT_CONFIG = '/opt/localdata/data/usr-envs/ruiying/Code/foundation/MoE_foundation_final/cfg/lifespan_segmentation.yaml'

DATASET_CONFIGS = {
    # 'SPINS': {
    #     'input_dir':  BASE_DATA / 'SPINS' / 'Resampled',
    #     'output_dir': OUTPUT_ROOT / 'SPINS',
    #     'demo_file':  BASE_DATA / 'SPINS' / 'demographics.csv',
    # },
    # 'LNSP': {
    #     'input_dir':  BASE_DATA / 'LNSP' / 'Resampled',
    #     'output_dir': OUTPUT_ROOT / 'LNSP',
    #     'demo_file':  BASE_DATA / 'LNSP' / 'demographics.csv',
    # },
    # 'ABIDE2': {
    #     'input_dir':  BASE_DATA / 'ABIDE2' / 'Resampled',
    #     'output_dir': OUTPUT_ROOT / 'ABIDE2',
    #     'demo_file':  BASE_DATA / 'ABIDE2' / 'demographics.csv',
    # },
    # 'Bonbid-Hie': {
    #     'input_dir':  BASE_DATA / 'Bonbid-Hie' / 'Resampled',
    #     'output_dir': OUTPUT_ROOT / 'Bonbid-Hie',
    #     'demo_file':  BASE_DATA / 'Bonbid-Hie' / 'demographics.csv',
    # },
    'Parkinson': {
        'input_dir':  BASE_DATA / 'Parkinson' / 'Resampled',
        'output_dir': OUTPUT_ROOT / 'Parkinson_old',
        'demo_file':  BASE_DATA / 'Parkinson' / 'demographics.csv',
    },
    # 'liege-pd': {
    #     'input_dir':  BASE_DATA / 'liege-pd' / 'Resampled',
    #     'output_dir': OUTPUT_ROOT / 'liege-pd',
    #     'demo_file':  BASE_DATA / 'liege-pd' / 'demographics.csv',
    # },
    # 'Infant-PWMI': {
    #     'input_dir':  BASE_DATA / 'Infant-PWMI' / 'Skull-stripped-CP',
    #     'output_dir': OUTPUT_ROOT / 'Infant-PWMI',
    #     'demo_file':  BASE_DATA / 'Infant-PWMI' / 'demographics_CP.csv',
    # },
    # 'lInfant-PWMI': {
    #     'input_dir':  BASE_DATA / 'Infant-PWMI' / 'Skull-stripped-Normal',
    #     'output_dir': OUTPUT_ROOT / 'Infant-PWMI',
    #     'demo_file':  BASE_DATA / 'Infant-PWMI' / 'demographics_Normal.csv',
    # },
}

WANTED_MODALITIES = {'T1', 'T1w', 'T2', 'T2w', 'FA', 'MD'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_wanted_modality(filename):
    stem = filename.replace('.nii.gz', '').replace('.nii', '')
    for part in stem.split('_'):
        if part.startswith('T1') or part.startswith('T2') or part in ('FA', 'MD'):
            return True
    return False


def extract_modality(filename):
    for part in filename.split('_'):
        if part.startswith('T1'):
            return 'T1w'
        if part.startswith('T2'):
            return 'T2w'
        if part in ('FA', 'MD'):
            return part
    return 'T1w'


def load_age_lookup(demo_file: Path) -> dict:
    """Return {(sub_id, ses_id): age_group} and {sub_id: age_group} fallback."""
    if not demo_file.exists():
        print(f'  [WARNING] Demographics not found: {demo_file}')
        return {}, {}
    df = pd.read_csv(demo_file)
    full = {}   # (sub_id, ses_id) -> age_group
    sub_only = {}  # sub_id -> age_group (first seen)
    for _, row in df.iterrows():
        sub = str(row.get('sub_id', '')).strip()
        ses_raw = row.get('ses_id', '')
        ses = '' if (ses_raw is None or (isinstance(ses_raw, float) and np.isnan(ses_raw))) else str(ses_raw).strip()
        try:
            age_grp = age_years_to_group(parse_age_to_years(row.get('age')))
        except Exception:
            age_grp = 'adult'
        full[(sub, ses)] = age_grp
        sub_only.setdefault(sub, age_grp)
    return full, sub_only


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
# Per-dataset processing
# ---------------------------------------------------------------------------
def process_dataset(dataset_name, cfg, model, resume):
    dcfg       = DATASET_CONFIGS[dataset_name]
    input_dir  = dcfg['input_dir']
    output_dir = dcfg['output_dir']
    demo_file  = dcfg['demo_file']

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Demo  : {demo_file}")

    age_lookup, age_lookup_sub = load_age_lookup(demo_file)
    print(f"  Loaded {len(age_lookup)} (sub, ses) entries from demographics")

    # Collect all .nii.gz files recursively
    entries = [(p, p.relative_to(input_dir))
               for p in sorted(input_dir.rglob('*.nii.gz'))]
    print(f"  Images: {len(entries)}")

    processed = skipped_exist = skipped_err = 0

    for img_path, rel_path in tqdm(entries, desc=dataset_name):
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if 'mask' in img_path.name.lower():
            continue

        if not is_wanted_modality(img_path.name):
            continue

        if resume and out_path.exists():
            skipped_exist += 1
            continue

        # Extract sub_id and ses_id from path (e.g. sub-CMH0001/ses-01/...)
        parts = rel_path.parts
        sub_id = parts[0] if len(parts) > 0 else ''
        ses_id = parts[1] if len(parts) > 1 else ''

        age_group = age_lookup.get((sub_id, ses_id),
                    age_lookup_sub.get(sub_id, 'adult'))

        modality = extract_modality(img_path.name)

        try:
            img_nii  = reorient_to_ras(nib.load(str(img_path)))
            img_data = np.squeeze(img_nii.get_fdata(dtype=np.float32))
        except Exception as e:
            tqdm.write(f'  [SKIP load] {rel_path}: {e}')
            skipped_err += 1
            continue

        img_data[img_data < 0] = 0

        try:
            pred_vol = infer_single_scan(model, cfg, img_data.copy(),
                                         modality=modality, age_group=age_group)
        except Exception as e:
            tqdm.write(f'  [SKIP infer] {rel_path}: {e}')
            skipped_err += 1
            continue

        nib.save(nib.Nifti1Image(pred_vol.astype(np.uint8),
                                  affine=img_nii.affine,
                                  header=img_nii.header),
                 str(out_path))
        processed += 1

    print(f"  Done — Processed: {processed}  |  Already existed: {skipped_exist}  |  Errors: {skipped_err}")
    return processed, skipped_exist, skipped_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Segment LNSP/SPINS datasets (MoE_v3)')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    parser.add_argument('--config',     type=str, default=None)
    parser.add_argument('--resume',     action='store_true',
                        help='Skip files whose output already exists')
    parser.add_argument('--dataset',    type=str, nargs='+',
                        choices=list(DATASET_CONFIGS.keys()),
                        default=list(DATASET_CONFIGS.keys()),
                        help='Which datasets to process (default: all)')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), 'train_cfg.yaml')
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
        print(f'Config: {config_path}')
    elif os.path.exists(DEFAULT_CONFIG):
        cfg.merge_from_file(DEFAULT_CONFIG)
        print(f'Config: {DEFAULT_CONFIG}')
    else:
        print('Warning: no config file found, using defaults')
    cfg.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    print('Loading model...')
    model = MoESegModel(cfg, pretrained_encoder_path=None)
    model = model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print('Model loaded.')

    total_p = total_se = total_err = 0
    for ds in args.dataset:
        p, se, err = process_dataset(ds, cfg, model, resume=args.resume)
        total_p   += p
        total_se  += se
        total_err += err

    print(f"\n{'='*60}")
    print(f'ALL DONE.  Processed: {total_p}  |  Already existed: {total_se}  |  Errors: {total_err}')
    print(f'Results: {OUTPUT_ROOT}')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
