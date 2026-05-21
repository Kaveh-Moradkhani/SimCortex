from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandGaussianNoised,
)
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Label mapping
# -----------------------------------------------------------------------------
# Output classes:
#   0 = background / ignored non-target tissue
#   1 = left inner cerebrum / WM-seed region
#   2 = right inner cerebrum / WM-seed region
#   3 = left cortical gray-matter ribbon
#   4 = right cortical gray-matter ribbon
#   5 = left hippocampus/amygdala
#   6 = right hippocampus/amygdala
#   7 = left lateral ventricle
#   8 = right lateral ventricle
#
# Note:
#   - Classes 3/4 are cortical ribbon labels, not "pial" labels. The pial surface
#     is a boundary inferred later, not a voxel label.
#   - FreeSurfer labels 3 and 42 are included as fallback non-parcellated cortex
#     labels. In fully parcellated aparc+aseg volumes, cortex is usually encoded
#     by 1000+/2000+ labels, but including 3/42 avoids silent cortical holes.
#   - Labels 77 and 80 are disambiguated using filled.mgz below.
LABEL_GROUPS: Dict[int, Set[int]] = {
    1: {2, 5, 10, 11, 12, 13, 26, 28, 30, 31},
    2: {41, 44, 49, 50, 51, 52, 58, 60, 62, 63},
    3: {3} | set(range(1000, 1004)) | set(range(1005, 1036)),
    4: {42} | set(range(2000, 2004)) | set(range(2005, 2036)),
    5: {17, 18},
    6: {53, 54},
    7: {4},
    8: {43},
}

VALID_SEG9_LABELS = np.arange(9, dtype=np.int64)


def map_labels(seg_arr: np.ndarray, filled_arr: np.ndarray) -> np.ndarray:
    """
    Map FreeSurfer aparc+aseg labels to SimCortex's 9-class segmentation.

    Parameters
    ----------
    seg_arr:
        MNI-space aparc+aseg array. Values are rounded before integer mapping to
        protect against tiny floating-point artifacts after image IO.

    filled_arr:
        MNI-space filled.mgz-derived array. Used to split ambiguous FreeSurfer
        hypointensity labels 77/80 into left/right inner cerebrum classes.

    Returns
    -------
    np.ndarray
        int64 array with labels in {0, ..., 8}, same spatial shape as seg_arr.
    """
    if seg_arr.shape != filled_arr.shape:
        raise ValueError(
            f"seg_arr and filled_arr must have the same shape, got "
            f"{seg_arr.shape} and {filled_arr.shape}."
        )

    seg_i = np.rint(seg_arr).astype(np.int32, copy=False)
    filled_i = np.rint(filled_arr).astype(np.int32, copy=False)

    seg_mapped = np.zeros(seg_i.shape, dtype=np.int64)
    for cls, labels in LABEL_GROUPS.items():
        seg_mapped[np.isin(seg_i, list(labels))] = int(cls)

    ambiguous = np.isin(seg_i, [77, 80])
    seg_mapped[ambiguous & (filled_i == 255)] = 1
    seg_mapped[ambiguous & (filled_i == 127)] = 2

    return seg_mapped


def robust_normalize(vol: np.ndarray) -> np.ndarray:
    """
    Foreground-percentile normalization for MNI T1w volumes.

    Background remains zero. Values above the foreground 99th percentile are
    clipped, then the image is scaled to approximately [0, 1].
    """
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    positive = vol[vol > 0]
    if positive.size == 0:
        return vol

    p99 = float(np.percentile(positive, 99))
    if not np.isfinite(p99) or p99 <= 0:
        return vol

    vol = np.clip(vol, 0.0, p99)
    return vol / p99


def get_augmentations() -> Compose:
    """
    Training augmentations for MNI-space segmentation.

    Translation is intentionally not enabled: the input is already registered to
    an MNI template, and large random translations can destroy template-space
    correspondence. Image uses trilinear/bilinear interpolation and labels use
    nearest-neighbor interpolation.
    """
    return Compose(
        [
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandBiasFieldd(keys=["image"], prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.1, std=0.05),
        ]
    )


def _validate_pad_mult(mult: int) -> None:
    if int(mult) < 1:
        raise ValueError(f"pad_mult must be >= 1, got {mult}.")


def _pad_spatial_3d(
    x: torch.Tensor,
    *,
    mult: int,
    mode: str,
    value: float = 0.0,
) -> torch.Tensor:
    """
    Pad a [D,H,W] or [C,D,H,W] tensor so D/H/W are divisible by mult.
    """
    _validate_pad_mult(mult)

    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected tensor with shape [D,H,W] or [C,D,H,W], got {tuple(x.shape)}.")

    _, d, h, w = x.shape
    pads = (
        0,
        (mult - w % mult) % mult,
        0,
        (mult - h % mult) % mult,
        0,
        (mult - d % mult) % mult,
    )

    if all(p == 0 for p in pads):
        return x

    if mode == "constant":
        return F.pad(x, pads, mode=mode, value=value)
    return F.pad(x, pads, mode=mode)


def pad_vol_to_multiple(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    """Pad an image tensor using replicated border values."""
    return _pad_spatial_3d(x, mult=mult, mode="replicate")


def pad_seg_to_multiple(x: torch.Tensor, mult: int = 16) -> torch.Tensor:
    """Pad a label tensor with background label 0."""
    return _pad_spatial_3d(x, mult=mult, mode="constant", value=0.0)


# -----------------------------------------------------------------------------
# Path helpers for sc-preproc derivative layout
# -----------------------------------------------------------------------------
def _sub_id(subject_label: str) -> str:
    sub = str(subject_label).strip()
    return sub if sub.startswith("sub-") else f"sub-{sub}"


def _ses_id(session_label: str) -> str:
    ses = str(session_label).strip()
    return ses if ses.startswith("ses-") else f"ses-{ses}"


def _stem(sub: str, ses: str) -> str:
    return f"{sub}_{ses}"


def _anat_dir(deriv_root: Path, sub: str, ses: str) -> Path:
    return deriv_root / sub / ses / "anat"


def _t1_mni_path(deriv_root: Path, sub: str, ses: str, space: str) -> Path:
    st = _stem(sub, ses)
    return _anat_dir(deriv_root, sub, ses) / f"{st}_space-{space}_desc-preproc_T1w.nii.gz"


def _aparc_aseg_mni_path(deriv_root: Path, sub: str, ses: str, space: str) -> Path:
    st = _stem(sub, ses)
    return _anat_dir(deriv_root, sub, ses) / f"{st}_space-{space}_desc-aparcaseg_dseg.nii.gz"


def _filled_mni_path(deriv_root: Path, sub: str, ses: str, space: str) -> Path:
    st = _stem(sub, ses)
    return _anat_dir(deriv_root, sub, ses) / f"{st}_space-{space}_desc-filled_dseg.nii.gz"


def _pred_seg9_candidates(pred_root: Path, sub: str, ses: str, space: str) -> Tuple[Path, Path]:
    st = _stem(sub, ses)
    prefix = pred_root / sub / ses / "anat" / f"{st}_space-{space}_desc-seg9"
    return (
        Path(str(prefix) + "_dseg.nii.gz"),  # BIDS-style current output
        Path(str(prefix) + "_pred.nii.gz"),  # legacy fallback
    )


def _resolve_pred_seg9_path(pred_root: Path, sub: str, ses: str, space: str) -> Path:
    cands = _pred_seg9_candidates(pred_root, sub, ses, space)
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing prediction. Tried: {', '.join(str(p) for p in cands)}")


def _read_split_subjects(
    split_csv: Path,
    split_name: str,
    dataset: Optional[str] = None,
) -> list[str]:
    """
    Read subjects from a split CSV.

    Required columns:
      - subject
      - split

    Optional column:
      - dataset, used when `dataset` is provided.

    `split_name` supports:
      - "train", "val", "test"
      - "train+val"
      - "train,val"
      - "all", "*", "any"
    """
    df = pd.read_csv(split_csv)

    if "subject" not in df.columns or "split" not in df.columns:
        raise ValueError(
            f"split_csv must have columns ['subject', 'split', ...], got: {list(df.columns)}"
        )

    if dataset is not None:
        if "dataset" not in df.columns:
            raise ValueError(
                f"split_csv has no 'dataset' column, but dataset='{dataset}' was provided. "
                f"Columns: {list(df.columns)}"
            )
        dataset_key = str(dataset).strip()
        df = df[df["dataset"].astype(str).str.strip() == dataset_key]

    split_name_raw = str(split_name).strip()
    split_name_s = split_name_raw.lower()

    if split_name_s in {"all", "*", "any"}:
        subs = df["subject"].astype(str).tolist()
    else:
        parts = [
            p.strip().lower()
            for p in split_name_s.replace("+", ",").split(",")
            if p.strip()
        ]
        if len(parts) == 1:
            mask = df["split"].astype(str).str.strip().str.lower() == parts[0]
        else:
            mask = df["split"].astype(str).str.strip().str.lower().isin(parts)
        subs = df.loc[mask, "subject"].astype(str).tolist()

    # Normalize to BIDS-style subject IDs to match preprocessing outputs.
    subs = sorted(_sub_id(s) for s in subs)

    if not subs:
        extra = f" and dataset='{dataset}'" if dataset is not None else ""
        raise ValueError(f"No subjects found for split='{split_name_raw}'{extra} in {split_csv}")

    return subs


# -----------------------------------------------------------------------------
# Image loading and validation helpers
# -----------------------------------------------------------------------------
def _load_nifti(path: Path) -> nib.spatialimages.SpatialImage:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def _check_same_grid(
    ref_img: nib.spatialimages.SpatialImage,
    other_img: nib.spatialimages.SpatialImage,
    *,
    ref_name: str,
    other_name: str,
    subject: str,
    affine_atol: float = 1e-4,
) -> None:
    ref_shape = tuple(ref_img.shape[:3])
    other_shape = tuple(other_img.shape[:3])
    if ref_shape != other_shape:
        raise ValueError(
            f"Shape mismatch for {subject}: {ref_name}={ref_shape}, "
            f"{other_name}={other_shape}."
        )

    if not np.allclose(np.asarray(ref_img.affine), np.asarray(other_img.affine), atol=affine_atol):
        raise ValueError(
            f"Affine mismatch for {subject}: {ref_name} and {other_name} are not on the same grid."
        )


def _load_training_triplet(
    *,
    deriv_root: Path,
    sub: str,
    ses: str,
    space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1_path = _t1_mni_path(deriv_root, sub, ses, space)
    seg_path = _aparc_aseg_mni_path(deriv_root, sub, ses, space)
    fill_path = _filled_mni_path(deriv_root, sub, ses, space)

    if not t1_path.exists():
        raise FileNotFoundError(f"Missing T1 (MNI): {t1_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing aparc+aseg (MNI): {seg_path}")
    if not fill_path.exists():
        raise FileNotFoundError(f"Missing filled (MNI): {fill_path}")

    t1_img = _load_nifti(t1_path)
    seg_img = _load_nifti(seg_path)
    fill_img = _load_nifti(fill_path)

    _check_same_grid(t1_img, seg_img, ref_name="T1", other_name="aparc+aseg", subject=sub)
    _check_same_grid(t1_img, fill_img, ref_name="T1", other_name="filled", subject=sub)

    vol = t1_img.get_fdata(dtype=np.float32)
    seg_arr = np.asanyarray(seg_img.dataobj)
    fill_arr = np.asanyarray(fill_img.dataobj)
    affine = np.asarray(t1_img.affine, dtype=np.float64)

    return vol, seg_arr, fill_arr, affine


def _assert_valid_label_range(arr: np.ndarray, *, name: str, subject: str) -> None:
    bad = np.setdiff1d(np.unique(arr), VALID_SEG9_LABELS)
    if bad.size > 0:
        raise ValueError(
            f"{name} for {subject} contains invalid labels outside 0..8: {bad.tolist()}"
        )


# -----------------------------------------------------------------------------
# Dataset classes
# -----------------------------------------------------------------------------
class SegDataset(Dataset):
    """
    Training/validation segmentation dataset.

    Returns
    -------
    image:
        torch.float32 tensor of shape [1, D, H, W], padded to `pad_mult`.

    label:
        torch.long tensor of shape [D, H, W], padded to `pad_mult`.
    """

    def __init__(
        self,
        deriv_root: str,
        split_csv: str,
        split: str = "train",
        dataset: Optional[str] = None,
        session_label: str = "01",
        space: str = "MNI152",
        pad_mult: int = 16,
        augment: bool = False,
    ) -> None:
        super().__init__()
        _validate_pad_mult(pad_mult)

        self.deriv_root = Path(deriv_root)
        self.split_csv = Path(split_csv)
        self.split = str(split).strip()
        self.dataset = dataset
        self.ses = _ses_id(session_label)
        self.space = str(space).strip()
        self.pad_mult = int(pad_mult)

        self.subjects = _read_split_subjects(self.split_csv, self.split, dataset=self.dataset)
        self.transforms = (
            get_augmentations()
            if (self.split.lower() == "train" and bool(augment))
            else None
        )

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sub = self.subjects[idx]

        vol, seg_arr, fill_arr, _ = _load_training_triplet(
            deriv_root=self.deriv_root,
            sub=sub,
            ses=self.ses,
            space=self.space,
        )

        vol = robust_normalize(vol)
        seg9 = map_labels(seg_arr, fill_arr)

        data = {
            "image": vol[None, ...],   # [C,D,H,W]
            "label": seg9[None, ...],  # [C,D,H,W]
        }
        if self.transforms is not None:
            data = self.transforms(data)

        vol_t = torch.as_tensor(data["image"], dtype=torch.float32)
        seg_t = torch.as_tensor(data["label"], dtype=torch.long)

        vol_t = torch.nan_to_num(vol_t, nan=0.0, posinf=0.0, neginf=0.0)

        vol_t = pad_vol_to_multiple(vol_t, self.pad_mult)
        seg_t = pad_seg_to_multiple(seg_t, self.pad_mult)

        # Label channel is used only for joint MONAI transforms; CrossEntropyLoss
        # expects target shape [B,D,H,W] after DataLoader batching.
        seg_t = seg_t.squeeze(0)
        _assert_valid_label_range(seg_t.cpu().numpy(), name="Mapped GT label", subject=sub)

        return vol_t, seg_t


class PredictSegDataset(Dataset):
    """
    Inference segmentation dataset.

    Returns
    -------
    image:
        torch.float32 tensor of shape [1,D,H,W], padded to `pad_mult`.

    sub:
        BIDS subject ID, e.g. "sub-100307".

    ses:
        BIDS session ID, e.g. "ses-01".

    affine:
        Original MNI-space affine before padding.

    orig_shape:
        Original unpadded spatial shape as int32 [D,H,W].
    """

    def __init__(
        self,
        deriv_root: str,
        split_csv: str,
        split_name: str = "test",
        dataset: Optional[str] = None,
        session_label: str = "01",
        space: str = "MNI152",
        pad_mult: int = 16,
    ) -> None:
        super().__init__()
        _validate_pad_mult(pad_mult)

        self.deriv_root = Path(deriv_root)
        self.split_csv = Path(split_csv)
        self.ses = _ses_id(session_label)
        self.dataset = dataset
        self.space = str(space).strip()
        self.pad_mult = int(pad_mult)

        self.subjects = _read_split_subjects(self.split_csv, split_name, dataset=self.dataset)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sub = self.subjects[idx]
        t1_path = _t1_mni_path(self.deriv_root, sub, self.ses, self.space)
        if not t1_path.exists():
            raise FileNotFoundError(f"Missing T1 (MNI): {t1_path}")

        img = _load_nifti(t1_path)
        vol = img.get_fdata(dtype=np.float32)
        affine = np.asarray(img.affine, dtype=np.float64)
        orig_shape = np.asarray(vol.shape[:3], dtype=np.int32)

        vol = robust_normalize(vol)
        vol_t = torch.from_numpy(vol[None, ...]).float()
        vol_t = torch.nan_to_num(vol_t, nan=0.0, posinf=0.0, neginf=0.0)
        vol_t = pad_vol_to_multiple(vol_t, mult=self.pad_mult)

        return vol_t, sub, self.ses, affine, orig_shape


class EvalSegDataset(Dataset):
    """
    Segmentation evaluation dataset.

    Loads mapped 9-class GT labels and saved predicted 9-class labels.
    Prediction may be larger than GT due to padding; it is cropped only when it
    safely contains the full GT grid.
    """

    def __init__(
        self,
        deriv_root: str,
        split_csv: str,
        pred_root: str,
        split_name: str = "test",
        dataset: Optional[str] = None,
        session_label: str = "01",
        space: str = "MNI152",
    ) -> None:
        super().__init__()
        self.deriv_root = Path(deriv_root)
        self.split_csv = Path(split_csv)
        self.pred_root = Path(pred_root)
        self.ses = _ses_id(session_label)
        self.dataset = dataset
        self.space = str(space).strip()

        self.subjects = _read_split_subjects(self.split_csv, split_name, dataset=self.dataset)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        sub = self.subjects[idx]

        gt_path = _aparc_aseg_mni_path(self.deriv_root, sub, self.ses, self.space)
        fill_path = _filled_mni_path(self.deriv_root, sub, self.ses, self.space)
        pred_path = _resolve_pred_seg9_path(self.pred_root, sub, self.ses, self.space)

        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT aparc+aseg (MNI): {gt_path}")
        if not fill_path.exists():
            raise FileNotFoundError(f"Missing filled (MNI): {fill_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        gt_img = _load_nifti(gt_path)
        fill_img = _load_nifti(fill_path)
        pred_img = _load_nifti(pred_path)

        _check_same_grid(gt_img, fill_img, ref_name="GT aparc+aseg", other_name="filled", subject=sub)

        gt_arr = np.asanyarray(gt_img.dataobj)
        fill_arr = np.asanyarray(fill_img.dataobj)
        pred_arr = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int64, copy=False)

        gt9 = map_labels(gt_arr, fill_arr)
        _assert_valid_label_range(gt9, name="Mapped GT label", subject=sub)

        d, h, w = gt9.shape
        if pred_arr.ndim != 3:
            raise ValueError(f"Prediction for {sub} must be 3D, got shape {pred_arr.shape}.")

        if pred_arr.shape[0] < d or pred_arr.shape[1] < h or pred_arr.shape[2] < w:
            raise ValueError(
                f"Prediction is smaller than GT for {sub}/{self.ses}: "
                f"pred={pred_arr.shape}, gt={gt9.shape}. Cannot safely evaluate."
            )

        pred_arr = pred_arr[:d, :h, :w]

        if pred_arr.shape != gt9.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch after crop for {sub}/{self.ses}: "
                f"pred={pred_arr.shape}, gt={gt9.shape}."
            )

        _assert_valid_label_range(pred_arr, name="Prediction", subject=sub)

        # If prediction was saved at the unpadded GT shape, its affine should match.
        # If it was saved at padded shape, affine can still match; checking the affine
        # here catches accidental wrong-space predictions.
        if not np.allclose(np.asarray(gt_img.affine), np.asarray(pred_img.affine), atol=1e-4):
            raise ValueError(f"Prediction affine does not match GT affine for {sub}/{self.ses}: {pred_path}")

        return gt9, pred_arr, sub, self.ses
