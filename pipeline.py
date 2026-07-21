"""
Lifespan Brain MRI Pipeline
----------------------------
Assumes input is already skull-stripped and reconstructed.

Steps:
  1. Resampling to 1mm³  (SimpleITK)
  2. Reorientation to RAS (nibabel)
  3. Segmentation         (LifespanPredictor)
  4. Volume calculation   (per-class mm³ and % ICV)

Usage:
    from pipeline import LifespanPipeline
    pipeline = LifespanPipeline()
    result = pipeline.run("brain.nii.gz", modality="T1w")
    # result: {
    #   "seg_path":    "/path/to/brain_seg.nii.gz",
    #   "age_group":   "adult",
    #   "volumes":     {"Background": 0, "Label_1": 12345.6, ...},
    #   "volumes_pct": {"Label_1": 1.23, ...},
    # }
"""

import os, sys, tempfile, shutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.preprocess import reorient_to_ras, resample_to_spacing
from infer import LifespanPredictor

# ── Label map: fill in your 15-class names ────────────────────────────────
LABEL_NAMES = {
    0:  "Background",
    1:  "Label_1",
    2:  "Label_2",
    3:  "Label_3",
    4:  "Label_4",
    5:  "Label_5",
    6:  "Label_6",
    7:  "Label_7",
    8:  "Label_8",
    9:  "Label_9",
    10: "Label_10",
    11: "Label_11",
    12: "Label_12",
    13: "Label_13",
    14: "Label_14",
}


class LifespanPipeline:
    def __init__(
        self,
        seg_checkpoint: str = None,
        age_checkpoint: str = None,
        target_spacing: tuple = (1.0, 1.0, 1.0),
        device=None,
        hf_repo_id: str = "",
    ):
        self.target_spacing = target_spacing
        self.predictor = LifespanPredictor(
            **({} if seg_checkpoint is None else {"seg_checkpoint": seg_checkpoint}),
            **({} if age_checkpoint is None else {"age_checkpoint": age_checkpoint}),
            **({"device": device} if device is not None else {}),
            hf_repo_id=hf_repo_id,
        )

    def run(self, input_path: str, modality: str = "T1w", output_dir: str = None):
        """
        Full pipeline: preprocess → segment → volumes.

        Args:
            input_path: Path to skull-stripped NIfTI (.nii or .nii.gz).
            modality:   'T1w' | 'T2w' | 'FA' | 'MD'
            output_dir: Where to save outputs (default: same dir as input).

        Returns dict with keys:
            preprocessed_path, seg_path, age_group, volumes, volumes_pct
        """
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(input_path))
        os.makedirs(output_dir, exist_ok=True)

        basename = os.path.basename(input_path).replace(".nii.gz", "").replace(".nii", "")
        work_dir = tempfile.mkdtemp(prefix="lifespan_")

        try:
            print(f"[1/3] Resampling to {self.target_spacing} mm ...", flush=True)
            resampled_path = os.path.join(work_dir, f"{basename}_resampled.nii.gz")
            self._resample(input_path, resampled_path)

            print(f"[2/3] Reorienting to RAS ...", flush=True)
            preprocessed_path = os.path.join(output_dir, f"{basename}_preprocessed.nii.gz")
            self._reorient(resampled_path, preprocessed_path)

            print(f"[3/3] Running segmentation ...", flush=True)
            seg_path = os.path.join(output_dir, f"{basename}_seg.nii.gz")
            pred_nii, age_group = self.predictor.predict_nifti(
                preprocessed_path, modality=modality, output_path=seg_path
            )

            volumes, volumes_pct = self._compute_volumes(pred_nii)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        return {
            "preprocessed_path": preprocessed_path,
            "seg_path":          seg_path,
            "age_group":         age_group,
            "volumes":           volumes,
            "volumes_pct":       volumes_pct,
        }

    # ── Resample ──────────────────────────────────────────────────────────
    def _resample(self, input_path: str, output_path: str):
        img = nib.load(input_path)
        resampled = resample_to_spacing(img, self.target_spacing, is_label=False)
        nib.save(resampled, output_path)

    # ── Reorient to RAS ───────────────────────────────────────────────────
    def _reorient(self, input_path: str, output_path: str):
        img = nib.load(input_path)
        ras = reorient_to_ras(img)
        nib.save(ras, output_path)

    # ── Volume calculation ─────────────────────────────────────────────────
    def _compute_volumes(self, seg_nii: nib.Nifti1Image):
        """
        Returns:
            volumes:     {label_name: volume_mm3}
            volumes_pct: {label_name: pct_of_ICV}  (excludes background)
        """
        data    = np.asarray(seg_nii.dataobj).astype(np.int32)
        zooms   = seg_nii.header.get_zooms()[:3]
        vox_mm3 = float(zooms[0] * zooms[1] * zooms[2])

        volumes = {}
        for label_id, label_name in LABEL_NAMES.items():
            count = int(np.sum(data == label_id))
            volumes[label_name] = round(count * vox_mm3, 2)

        icv = sum(v for k, v in volumes.items() if k != "Background")
        volumes_pct = {}
        if icv > 0:
            for name, vol in volumes.items():
                if name != "Background":
                    volumes_pct[name] = round(100.0 * vol / icv, 4)

        return volumes, volumes_pct


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lifespan brain MRI pipeline")
    parser.add_argument("input",      help="Input skull-stripped NIfTI file")
    parser.add_argument("--modality", default="T1w", choices=["T1w", "T2w", "FA", "MD"])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    pipeline = LifespanPipeline()
    result   = pipeline.run(args.input, modality=args.modality, output_dir=args.output_dir)

    print("\n=== Results ===")
    print(f"Age group : {result['age_group']}")
    print(f"Seg mask  : {result['seg_path']}")
    print(f"\nVolumes (mm³):")
    for name, vol in result["volumes"].items():
        if name != "Background" and vol > 0:
            pct = result["volumes_pct"].get(name, 0)
            print(f"  {name:<20} {vol:>12.1f} mm³  ({pct:.2f}% ICV)")
