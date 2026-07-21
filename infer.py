"""
Lifespan Brain Segmentation Inference
--------------------------------------
Pipeline:
  1. Age prediction  (NewAgePredictionModel)  → predicted age_group
  2. Tissue segmentation (MoESegModel) using predicted age_group for MoE routing

Usage:
    from infer import LifespanPredictor
    predictor = LifespanPredictor()
    pred_nii, age_group = predictor.predict_nifti("brain.nii.gz", modality="T1w")
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
import torchio as tio

# Make sure local modules are importable when this script is run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfg.lifespan_config import get_cfg_defaults
from model.segmentation_model import MoESegModel
from model.new_age_prediction_model import NewAgePredictionModel
from utils import util

# Maps predicted class index → age_group string expected by segmentation MoE encoder
AGE_CLASS_NAMES = ['fetal', 'neonatal', 'infant', 'child', 'adult', 'elderly']

_BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SEG_CKPT  = os.path.join(_BASE_DIR, 'checkpoints', 'seg', 'best_model.pth')
_DEFAULT_AGE_CKPT  = os.path.join(_BASE_DIR, 'checkpoints', 'age', 'best_age.pth')
_DEFAULT_CFG_YAML  = os.path.join(_BASE_DIR, 'cfg', 'infer_config.yaml')

# Set HF_REPO_ID env var (or pass to LifespanPredictor) to download checkpoints from
# Hugging Face Hub if they are not present on disk.
_HF_REPO_ID = os.environ.get("HF_REPO_ID", "")


def _ensure_checkpoints(seg_ckpt: str, age_ckpt: str, hf_repo_id: str = ""):
    """Download checkpoints from HF Hub if they don't exist locally."""
    missing = not os.path.exists(seg_ckpt) or not os.path.exists(age_ckpt)
    if not missing:
        return

    repo = hf_repo_id or _HF_REPO_ID
    if not repo:
        raise FileNotFoundError(
            f"Checkpoints not found and HF_REPO_ID is not set.\n"
            f"  seg : {seg_ckpt}\n  age : {age_ckpt}\n"
            "Set env var HF_REPO_ID=<hf-username>/<repo-name> to auto-download."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub  to enable auto-download of checkpoints.")

    print(f"Downloading checkpoints from {repo} ...", flush=True)
    os.makedirs(os.path.dirname(seg_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(age_ckpt), exist_ok=True)

    if not os.path.exists(seg_ckpt):
        hf_hub_download(repo_id=repo, filename="checkpoints/seg/best_model.pth",
                        local_dir=_BASE_DIR)
    if not os.path.exists(age_ckpt):
        hf_hub_download(repo_id=repo, filename="checkpoints/age/best_age.pth",
                        local_dir=_BASE_DIR)


class LifespanPredictor:
    def __init__(
        self,
        seg_checkpoint: str = _DEFAULT_SEG_CKPT,
        age_checkpoint: str = _DEFAULT_AGE_CKPT,
        config_file: str = None,
        device: torch.device = None,
        hf_repo_id: str = "",
    ):
        """
        Args:
            seg_checkpoint: Path to segmentation model checkpoint.
            age_checkpoint: Path to age prediction model checkpoint.
            config_file:    Optional YAML to override default config.
            device:         Torch device. Defaults to CUDA if available.
            hf_repo_id:     HF Hub repo (e.g. "athosd/lifespan-seg") to
                            download checkpoints from if not present locally.
        """
        _ensure_checkpoints(seg_checkpoint, age_checkpoint, hf_repo_id)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Config: always load inference defaults, optionally override further
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(_DEFAULT_CFG_YAML)
        if config_file is not None:
            self.cfg.merge_from_file(config_file)
        self.cfg.freeze()

        # Segmentation model
        self.seg_model = MoESegModel(self.cfg)
        _load_state(self.seg_model, seg_checkpoint, device)
        self.seg_model.to(device).eval()

        # Age prediction model
        self.age_model = NewAgePredictionModel(self.cfg)
        _load_state(self.age_model, age_checkpoint, device)
        self.age_model.to(device).eval()

        print(f"LifespanPredictor ready on {device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, scan_data: np.ndarray, modality: str = 'T1w'):
        """
        Segment a 3-D brain scan.

        Args:
            scan_data: float32 numpy array, shape (X, Y, Z).
            modality:  One of 'T1w', 'T2w', 'FA', 'MD'.

        Returns:
            pred_vol:  uint8 numpy array (X, Y, Z) with tissue label indices.
            age_group: Predicted age group string.
        """
        cfg = self.cfg
        x, y, z = cfg.data.patch_size

        # Step 1 – predict age group from the whole scan (one forward pass)
        age_group = self._predict_age_group(scan_data, modality)
        print(f"  Predicted age group: {age_group}")

        # Step 2 – normalise intensity
        if cfg.data.normalize:
            scan_data = util.norm_img(scan_data, cfg.data.norm_perc)

        # Step 3 – pad if the scan is smaller than the patch size
        pad_flag = False
        ori_shape = scan_data.shape
        if min(scan_data.shape) < min(x, y, z):
            pad_flag = True
            x_diff = max(0, x - ori_shape[0])
            y_diff = max(0, y - ori_shape[1])
            z_diff = max(0, z - ori_shape[2])
            scan_data = np.pad(
                scan_data,
                (
                    (x_diff // 2, x_diff - x_diff // 2),
                    (y_diff // 2, y_diff - y_diff // 2),
                    (z_diff // 2, z_diff - z_diff // 2),
                ),
                constant_values=1e-4,
            )

        # Step 4 – sliding-window segmentation
        pred = np.zeros((cfg.train.cls_num,) + scan_data.shape, dtype=np.float32)
        count = np.zeros((cfg.train.cls_num,) + scan_data.shape, dtype=np.float32)

        scan_patches, _, patch_indices = util.patch_slicer(
            scan_data, scan_data, cfg.data.patch_size,
            (x - 16, y - 16, z - 16),
            remove_bg=cfg.data.remove_bg, test=True, ori_path=None,
        )

        bound = util.get_bounds(torch.from_numpy(scan_data))
        global_scan = torch.from_numpy(scan_data).float().unsqueeze(0)   # (1, X, Y, Z)
        cropped_global = global_scan[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]

        # Pre-compute the downsampled global scan once (shared across all patches)
        down_scan_shared = torch.nn.functional.interpolate(
            cropped_global.unsqueeze(0), size=(x, y, z), mode='trilinear', align_corners=False
        ).squeeze(0).to(self.device)  # (1, x, y, z)

        # Pre-compute all location masks and coordinates on CPU, then batch on GPU
        all_coords = []
        for idx, p_idx in enumerate(patch_indices):
            loc = torch.zeros_like(global_scan)
            loc[:, p_idx[0]:p_idx[1], p_idx[2]:p_idx[3], p_idx[4]:p_idx[5]] = 1.0
            loc_crop = loc[:, bound[0]:bound[1], bound[2]:bound[3], bound[4]:bound[5]]
            loc_down = torch.nn.functional.interpolate(
                loc_crop.unsqueeze(0), size=(x, y, z), mode='nearest'
            ).squeeze(0)
            tmp_coor = util.get_bounds(loc_down)
            coords = np.array([
                int(np.floor(tmp_coor[0] / 4)), int(np.ceil(tmp_coor[1] / 4)),
                int(np.floor(tmp_coor[2] / 4)), int(np.ceil(tmp_coor[3] / 4)),
                int(np.floor(tmp_coor[4] / 4)), int(np.ceil(tmp_coor[5] / 4)),
            ], dtype=np.int64)
            all_coords.append(coords)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
            for idx, patch in enumerate(scan_patches):
                p_idx  = patch_indices[idx]
                coords = all_coords[idx]
                coords_t = torch.from_numpy(coords).unsqueeze(0)

                ipt = torch.from_numpy(patch).float().to(self.device).reshape(1, 1, x, y, z)

                tmp_pred, _ = self.seg_model(
                    ipt,
                    down_scan_shared.unsqueeze(0),
                    coords_t,
                    modality=modality,
                    age_group=age_group,
                )

                slc = (
                    slice(None),
                    slice(p_idx[0], p_idx[1]),
                    slice(p_idx[2], p_idx[3]),
                    slice(p_idx[4], p_idx[5]),
                )
                pred[slc]  += tmp_pred.squeeze().detach().cpu().numpy()
                count[slc] += 1.0

        # Average overlapping patches → softmax → argmax
        nonzero = count > 0
        pred[nonzero] /= count[nonzero]
        pred_vol = torch.nn.Softmax(dim=0)(torch.from_numpy(pred)).numpy()
        pred_vol = np.argmax(pred_vol, axis=0).astype(np.uint8)

        # Remove padding
        if pad_flag:
            pred_vol = pred_vol[
                x_diff // 2: x_diff // 2 + ori_shape[0],
                y_diff // 2: y_diff // 2 + ori_shape[1],
                z_diff // 2: z_diff // 2 + ori_shape[2],
            ]

        return pred_vol, age_group

    def predict_nifti(self, nifti_path: str, modality: str = 'T1w', output_path: str = None):
        """
        Convenience wrapper: load NIfTI → predict → optionally save result.

        Returns:
            pred_nii:  nibabel NIfTI1Image with predicted labels.
            age_group: Predicted age group string.
        """
        img = nib.load(nifti_path)
        scan_data = np.squeeze(img.get_fdata().astype(np.float32))
        scan_data[scan_data < 0] = 0.0

        pred_vol, age_group = self.predict(scan_data, modality=modality)

        pred_nii = nib.Nifti1Image(pred_vol, affine=img.affine, header=img.header)
        if output_path is not None:
            nib.save(pred_nii, output_path)
            print(f"  Saved prediction to {output_path}")

        return pred_nii, age_group

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_age_group(self, scan_data: np.ndarray, modality: str) -> str:
        """Forward pass through the age prediction model."""
        x, y, z = self.cfg.data.patch_size
        scan_t = torch.from_numpy(scan_data).float().unsqueeze(0).unsqueeze(0).to(self.device)
        # Resize to 96³ expected by age model
        if scan_t.shape[2:] != (x, y, z):
            scan_t = torch.nn.functional.interpolate(
                scan_t, size=(x, y, z), mode='trilinear', align_corners=False
            )
        with torch.no_grad():
            logits = self.age_model(scan_t, modality_ids=modality)
            predicted_idx = int(torch.argmax(logits, dim=1).item())
        return AGE_CLASS_NAMES[predicted_idx]


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _load_state(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Lifespan brain segmentation inference')
    parser.add_argument('input',  help='Input NIfTI file (.nii or .nii.gz)')
    parser.add_argument('output', help='Output NIfTI file for predicted labels')
    parser.add_argument('--modality', default='T1w', choices=['T1w', 'T2w', 'FA', 'MD'])
    parser.add_argument('--seg_checkpoint', default=_DEFAULT_SEG_CKPT)
    parser.add_argument('--age_checkpoint', default=_DEFAULT_AGE_CKPT)
    args = parser.parse_args()

    predictor = LifespanPredictor(
        seg_checkpoint=args.seg_checkpoint,
        age_checkpoint=args.age_checkpoint,
    )
    _, age_group = predictor.predict_nifti(args.input, modality=args.modality, output_path=args.output)
    print(f"Done. Age group: {age_group}")
