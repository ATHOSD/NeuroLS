#!/usr/bin/env python3
"""
Basic preprocessing utilities for MAE training
Only includes reorientation and resampling functions
"""

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff
from typing import Tuple
import warnings


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Reorient image to RAS (Right-Anterior-Superior) orientation.

    Args:
        img: Input nibabel image

    Returns:
        Reoriented image in RAS space
    """
    try:
        # Get current orientation and target RAS orientation
        cur_ornt = io_orientation(img.affine)
        ras_ornt = axcodes2ornt(('R', 'A', 'S'))

        # Calculate transformation
        xform = ornt_transform(cur_ornt, ras_ornt)

        # Apply transformation to data and affine
        data = img.get_fdata(dtype=np.float32)
        data_ras = apply_orientation(data, xform)
        aff_ras = img.affine @ inv_ornt_aff(xform, img.shape)

        # Create new image with RAS orientation
        return nib.Nifti1Image(data_ras, aff_ras, img.header)

    except Exception as e:
        warnings.warn(f"RAS reorientation failed: {e}, returning original image")
        return img


def resample_to_spacing(
    img: nib.Nifti1Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label: bool = False
) -> nib.Nifti1Image:
    """
    Resample image to target isotropic spacing using scipy.

    Args:
        img: Input nibabel image
        target_spacing: Target spacing in mm (x, y, z)
        is_label: If True, use nearest neighbor interpolation (order=0) to preserve label values.
                  If False, use linear interpolation (order=1) for images.

    Returns:
        Resampled image
    """
    try:
        from scipy import ndimage

        # Get image data
        data = img.get_fdata().astype(np.float32)

        # Handle 4D images by squeezing or selecting first volume
        if data.ndim > 3:
            data = np.squeeze(data)
            if data.ndim > 3:
                # If still 4D after squeeze, take first volume
                data = data[..., 0]

        # Get current spacing from header
        current_spacing = img.header.get_zooms()[:3]

        # Calculate zoom factors - ensure it matches data dimensions
        zoom_factors = np.array(current_spacing) / np.array(target_spacing)

        # Ensure zoom_factors matches data rank
        if len(zoom_factors) != data.ndim:
            raise ValueError(f"Zoom factors length {len(zoom_factors)} doesn't match data rank {data.ndim}")

        # Resample data using appropriate interpolation method
        # order=0 (nearest neighbor) for labels to preserve exact values
        # order=1 (linear) for images to get smooth results
        interpolation_order = 0 if is_label else 1
        resampled_data = ndimage.zoom(data, zoom_factors, order=interpolation_order, prefilter=False)

        # Update affine matrix
        new_affine = img.affine.copy()
        # Scale the first 3 columns by inverse zoom factors
        new_affine[:3, :3] = new_affine[:3, :3] / zoom_factors

        return nib.Nifti1Image(resampled_data, new_affine, img.header)

    except Exception as e:
        # Include more context in the error
        import traceback
        warnings.warn(f"Resampling failed: {e}\nData shape: {data.shape if 'data' in locals() else 'unknown'}, "
                     f"Zoom factors: {zoom_factors if 'zoom_factors' in locals() else 'unknown'}\n"
                     f"Returning original image")
        return img


def preprocess_image(
    img_path: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    reorient: bool = True,
    resample: bool = False,
    is_label: bool = False
) -> np.ndarray:
    """
    Load and preprocess image with reorientation and resampling.

    Args:
        img_path: Path to image file
        target_spacing: Target isotropic spacing in mm
        reorient: Whether to reorient to RAS
        resample: Whether to resample to target spacing
        is_label: If True, use nearest neighbor interpolation for labels.
                  If False, use linear interpolation for images.

    Returns:
        Preprocessed image data as numpy array
    """
    # Load image
    img = nib.load(img_path)

    # Reorient to RAS if requested
    if reorient:
        img = reorient_to_ras(img)

    # Resample if requested
    if resample:
        img = resample_to_spacing(img, target_spacing, is_label=is_label)

    # Get data and return
    data = img.get_fdata().astype(np.float32)

    return data