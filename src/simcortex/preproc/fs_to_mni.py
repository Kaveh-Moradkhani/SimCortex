#!/usr/bin/env python3
"""
SimCortex preprocessing: FreeSurfer native outputs -> MNI152 derivatives.

This module is intended for the training/full-pipeline path where FreeSurfer
volumes and surfaces are available. It exports native volumes, estimates a
linear native-T1w -> MNI image registration with ANTsPy, resamples label volumes,
and exports native scanner-RAS plus MNI152 RAS surface PLY files.

Important transform convention
------------------------------
ANTs/ITK affine files from ``reg["fwdtransforms"]`` are used by ANTs to resample
moving images into the fixed image space. However, when that affine is read as a
POINT transform with ``ants.read_transform()``, its physical point direction is
fixed/MNI -> moving/native. Therefore:

    .mat point matrix                 : MNI152 RAS-mm -> native scannerRAS RAS-mm
    inverse(.mat point matrix)        : native scannerRAS RAS-mm -> MNI152 RAS-mm

The code below writes explicit ``mode-surface_xfm.txt`` files in an ``xfm/``
folder so downstream surface code never has to infer this convention from image
resampling files.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import nibabel as nib
import numpy as np
import typer
from nibabel.freesurfer.io import read_geometry

try:
    import ants  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("ANTsPy is required. Install it with: pip install antspyx") from e

APP_NAME = "SimCortex-Preproc-ANTsPy"
__version__ = "0.1"
PIPELINE_NAME = "sc-preproc-0.1"
RAS_TO_LPS_4 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="FreeSurfer -> MNI preprocessing with ANTsPy + nibabel.",
)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(verbosity: int = 0, log_file: Optional[Path] = None) -> logging.Logger:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logger = logging.getLogger("sc-preproc-antspy")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def strip_prefix(x: str, prefix: str) -> str:
    x = str(x).strip()
    return x[len(prefix):] if x.startswith(prefix) else x


def bids_sub_id(label: str) -> str:
    return f"sub-{strip_prefix(label, 'sub-')}"


def bids_ses_id(label: str) -> str:
    return f"ses-{strip_prefix(label, 'ses-')}"


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_fs_subject_dir(d: Path) -> bool:
    return d.is_dir() and (d / "mri").is_dir() and (d / "surf").is_dir()


def has_nested_fs_session_dir(d: Path) -> bool:
    return d.is_dir() and any(is_fs_subject_dir(ses) for ses in d.glob("ses-*"))


def discover_subjects(fs_root: Path) -> list[str]:
    """Discover subject labels from flat or BIDS-session FreeSurfer layouts."""
    if not fs_root.exists():
        raise FileNotFoundError(f"FreeSurfer root not found: {fs_root}")

    subjects: list[str] = []
    for d in fs_root.iterdir():
        if not d.is_dir():
            continue
        if is_fs_subject_dir(d) or has_nested_fs_session_dir(d):
            subjects.append(bids_sub_id(d.name))

    return sorted(set(subjects))


def find_fs_subject_dir(fs_root: Path, sub_label: str, session_label: str = "ses-01") -> Path:
    """
    Resolve a FreeSurfer subject/session directory.

    Supported layouts:
      1. <fs_root>/<sub>/mri and <fs_root>/<sub>/surf
      2. <fs_root>/<sub>/<ses>/mri and <fs_root>/<sub>/<ses>/surf
      3. same as above with subject label stored without the ``sub-`` prefix.
    """
    sub = bids_sub_id(sub_label)
    sub_no_prefix = strip_prefix(sub_label, "sub-")
    ses = bids_ses_id(session_label)

    candidates = [
        fs_root / sub / ses,
        fs_root / sub,
        fs_root / sub_no_prefix / ses,
        fs_root / sub_no_prefix,
        fs_root / str(sub_label) / ses,
        fs_root / str(sub_label),
    ]
    for cand in candidates:
        if is_fs_subject_dir(cand):
            return cand

    raise FileNotFoundError(
        f"Could not find FreeSurfer subject directory for {sub_label}/{ses} under {fs_root}. "
        "Expected either <root>/<sub>/mri+surf or <root>/<sub>/<ses>/mri+surf."
    )


def resolve_fs_surface_path(fs_surf_dir: Path, fs_hemi: str, surf_type: str) -> Optional[Path]:
    cands: list[Path] = []
    if surf_type == "white":
        cands.append(fs_surf_dir / f"{fs_hemi}.white")
    elif surf_type == "pial":
        cands.append(fs_surf_dir / f"{fs_hemi}.pial")
        cands.append(fs_surf_dir / f"{fs_hemi}.pial.T1")
    else:
        cands.append(fs_surf_dir / f"{fs_hemi}.{surf_type}")

    for p in cands:
        if p.exists():
            return p
    return None


def apply_affine(mat4: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    mat4 = np.asarray(mat4, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    if mat4.shape != (4, 4):
        raise ValueError(f"Expected affine shape (4, 4), got {mat4.shape}")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (N, 3), got {xyz.shape}")

    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz1 = np.concatenate([xyz, ones], axis=1)
    out = (mat4 @ xyz1.T).T
    return out[:, :3]


def write_dataset_description(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    dd = {
        "Name": PIPELINE_NAME,
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": APP_NAME,
                "Version": __version__,
                "Description": "FreeSurfer-derived preprocessing using nibabel and ANTsPy linear registration.",
            }
        ],
    }
    (out_root / "dataset_description.json").write_text(
        json.dumps(dd, indent=2) + "\n",
        encoding="utf-8",
    )


def save_like_ply_ascii(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write an ASCII PLY mesh preserving double precision coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Vertices must be (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"Faces must be (M, 3), got {f.shape}")

    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {v.shape[0]}\n")
        fp.write("property double x\n")
        fp.write("property double y\n")
        fp.write("property double z\n")
        fp.write(f"element face {f.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for row in v:
            fp.write(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")
        for tri in f:
            fp.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


# -----------------------------------------------------------------------------
# NIfTI / MGZ helpers
# -----------------------------------------------------------------------------
def load_nib(path: Path) -> nib.spatialimages.SpatialImage:
    return nib.load(str(path))


def save_nifti_lossless_like(img: nib.spatialimages.SpatialImage, out_path: Path) -> None:
    """Export a nibabel image as NIfTI without reorienting or resampling."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=np.float64)
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(data.dtype)
    hdr.set_qform(affine, code=1)
    hdr.set_sform(affine, code=1)
    out = nib.Nifti1Image(data, affine, header=hdr)
    nib.save(out, str(out_path))


def get_tkras_to_scanner_ras_from_orig(orig_mgz: Path) -> np.ndarray:
    img = load_nib(orig_mgz)
    hdr = img.header
    if not hasattr(hdr, "get_vox2ras") or not hasattr(hdr, "get_vox2ras_tkr"):
        raise RuntimeError("orig.mgz header does not expose get_vox2ras / get_vox2ras_tkr")
    norig = np.asarray(hdr.get_vox2ras(), dtype=np.float64)
    torig = np.asarray(hdr.get_vox2ras_tkr(), dtype=np.float64)
    return norig @ np.linalg.inv(torig)


# -----------------------------------------------------------------------------
# ANTsPy transform helpers
# -----------------------------------------------------------------------------
def ants_lps_matrix_to_ras_matrix(m_lps: np.ndarray) -> np.ndarray:
    """Convert a 4x4 physical point transform matrix from LPS to RAS."""
    m_lps = np.asarray(m_lps, dtype=np.float64)
    if m_lps.shape != (4, 4):
        raise ValueError(f"Expected matrix shape (4, 4), got {m_lps.shape}")
    return RAS_TO_LPS_4 @ m_lps @ RAS_TO_LPS_4


def ants_affine_to_homogeneous_lps(tx_path: Path) -> np.ndarray:
    """
    Convert an ANTs/ITK affine transform to a 4x4 homogeneous point matrix in LPS.

    ANTs/ITK affine parameters are interpreted as:
        y = A @ (x - c) + c + t
    where c is the fixed transform center. The equivalent homogeneous matrix is:
        y = A @ x + (t + c - A @ c)
    """
    tx = ants.read_transform(str(tx_path))
    params = np.asarray(tx.parameters, dtype=np.float64)
    fixed = np.asarray(tx.fixed_parameters, dtype=np.float64)

    if params.size < 12:
        raise ValueError(f"Expected at least 12 affine parameters, got {params.size} from {tx_path}")
    if fixed.size < 3:
        raise ValueError(f"Expected at least 3 fixed parameters, got {fixed.size} from {tx_path}")

    a = params[:9].reshape(3, 3)
    t = params[9:12]
    c = fixed[:3]

    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = a
    m[:3, 3] = t + c - a @ c
    return m


def compute_surface_point_matrices_from_ants_mat(ants_mat: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (native_to_mni_ras, mni_to_native_ras) point matrices for surfaces.

    The ANTs forward .mat from image registration is interpreted as an MNI->native
    POINT transform after ants.read_transform(), so native->MNI is its inverse.
    """
    mni_to_native_lps = ants_affine_to_homogeneous_lps(ants_mat)
    mni_to_native_ras = ants_lps_matrix_to_ras_matrix(mni_to_native_lps)
    native_to_mni_ras = np.linalg.inv(mni_to_native_ras)
    return native_to_mni_ras, mni_to_native_ras


def normalize_ants_transform_type(name: str) -> str:
    key = str(name).strip().lower()
    mapping = {"rigid": "Rigid", "affine": "Affine"}
    if key not in mapping:
        raise ValueError("--transform-type must be 'rigid' or 'affine'")
    return mapping[key]


def pick_affine_transform_path(transform_list: Sequence[str]) -> Path:
    cands = [Path(t) for t in transform_list if str(t).lower().endswith(".mat")]
    if len(cands) != 1:
        raise RuntimeError(f"Expected exactly one affine transform, got: {list(transform_list)}")
    return cands[0]


def save_matrix_txt(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=np.float64), fmt="%.10f")


def save_surface_transform_json(
    path: Path,
    *,
    source_ants_transform: Path,
    native_to_mni_ras: np.ndarray,
    mni_to_native_ras: np.ndarray,
    transform_type: str,
    space: str,
    use_n4: bool,
    n4_use_mask: bool,
    random_seed: Optional[int],
) -> None:
    payload = {
        "Description": "Surface point transforms in RAS-mm coordinates for SimCortex meshes.",
        "SourceANTsTransform": str(source_ants_transform.name),
        "ANTsCoordinateConvention": "LPS",
        "SurfaceCoordinateConvention": "RAS",
        "TransformType": normalize_ants_transform_type(transform_type),
        "MatrixApplication": "column homogeneous points: y = M @ x; code may equivalently use row_hom @ M.T",
        "NativeToTargetSpace": f"native scannerRAS -> {space} RAS",
        "TargetSpaceToNative": f"{space} RAS -> native scannerRAS",
        "NativeToTargetMatrix": native_to_mni_ras.tolist(),
        "TargetToNativeMatrix": mni_to_native_ras.tolist(),
        "N4BiasCorrection": bool(use_n4),
        "N4UsedMask": bool(n4_use_mask),
        "RandomSeed": random_seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_n4_bias_correction(
    t1_native_path: Path,
    out_path: Path,
    use_mask: bool,
    shrink_factor: int,
    spline_distance: float,
    logger: logging.Logger,
) -> None:
    logger.info("N4 bias-field correction: %s -> %s", t1_native_path.name, out_path.name)
    img = ants.image_read(str(t1_native_path))
    mask = ants.get_mask(img) if use_mask else None
    corrected = ants.n4_bias_field_correction(
        img,
        mask=mask,
        shrink_factor=int(shrink_factor),
        spline_param=float(spline_distance),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(corrected, str(out_path))


def estimate_linear_registration(
    fixed_mni_path: Path,
    moving_t1_path: Path,
    transform_type: str,
    random_seed: Optional[int],
    out_affine_mat_path: Path,
    logger: logging.Logger,
) -> Any:
    fixed = ants.image_read(str(fixed_mni_path))
    moving = ants.image_read(str(moving_t1_path))

    ants_type = normalize_ants_transform_type(transform_type)
    kwargs: dict[str, Any] = {
        "fixed": fixed,
        "moving": moving,
        "type_of_transform": ants_type,
    }
    if random_seed is not None:
        kwargs["random_seed"] = int(random_seed)

    logger.info(
        "ANTs registration: %s (moving=%s -> fixed=%s)",
        ants_type,
        moving_t1_path.name,
        fixed_mni_path.name,
    )
    with tempfile.TemporaryDirectory(prefix="fs_to_mni_antspy_") as tmpdir:
        kwargs["outprefix"] = str(Path(tmpdir) / "ants_")
        reg = ants.registration(**kwargs)
        aff_path = pick_affine_transform_path(reg["fwdtransforms"])
        out_affine_mat_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(aff_path, out_affine_mat_path)
        return reg


def apply_linear_transform_to_volume(
    moving_path: Path,
    fixed_path: Path,
    affine_mat_path: Path,
    out_path: Path,
    interpolation: str,
    logger: logging.Logger,
) -> None:
    logger.info("Apply transform to volume: %s -> %s (%s)", moving_path.name, out_path.name, interpolation)
    moving = ants.image_read(str(moving_path))
    fixed = ants.image_read(str(fixed_path))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(affine_mat_path)],
        interpolator=interpolation,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(warped, str(out_path))


# -----------------------------------------------------------------------------
# Core per-subject processing
# -----------------------------------------------------------------------------
def process_one(
    *,
    fs_root: Path,
    out_root: Path,
    participant_label: str,
    session_label: str,
    mni_template: Path,
    space: str,
    surface_names: Sequence[str],
    hemis: Sequence[str],
    write_aparc_aseg: bool,
    write_filled: bool,
    strict_volumes: bool,
    use_n4: bool,
    n4_use_mask: bool,
    n4_shrink_factor: int,
    n4_spline_distance: float,
    transform_type: str,
    save_raw_t1_export: bool,
    random_seed: Optional[int],
    skip_existing: bool,
    strict_surfaces: bool,
    write_legacy_anat_xfms: bool,
    logger: logging.Logger,
) -> None:
    sub = bids_sub_id(participant_label)
    ses = bids_ses_id(session_label)
    stem = f"{sub}_{ses}"

    fs_sub_dir = find_fs_subject_dir(fs_root, sub, ses)
    fs_mri = fs_sub_dir / "mri"
    fs_surf = fs_sub_dir / "surf"

    out_sub = out_root / sub / ses
    out_anat = out_sub / "anat"
    out_surfaces = out_sub / "surfaces"
    out_xfm = out_sub / "xfm"
    safe_mkdir(out_anat)
    safe_mkdir(out_surfaces)
    safe_mkdir(out_xfm)

    # Volume outputs
    f_t1_raw = out_anat / f"{stem}_desc-fsraw_T1w.nii.gz"
    f_t1_native = out_anat / f"{stem}_desc-preproc_T1w.nii.gz"
    f_t1_mni = out_anat / f"{stem}_space-{space}_desc-preproc_T1w.nii.gz"

    f_aseg_native = out_anat / f"{stem}_desc-aseg_dseg.nii.gz"
    f_aseg_mni = out_anat / f"{stem}_space-{space}_desc-aseg_dseg.nii.gz"

    f_aparc_native = out_anat / f"{stem}_desc-aparcaseg_dseg.nii.gz"
    f_aparc_mni = out_anat / f"{stem}_space-{space}_desc-aparcaseg_dseg.nii.gz"

    f_filled_native = out_anat / f"{stem}_desc-filled_dseg.nii.gz"
    f_filled_mni = out_anat / f"{stem}_space-{space}_desc-filled_dseg.nii.gz"

    # Transform outputs. The .mat is the raw ANTs image-resampling transform.
    f_aff_mat = out_xfm / f"{stem}_from-native_to-{space}_mode-image_desc-antsAffine.mat"
    f_surface_native_to_mni = out_xfm / f"{stem}_from-native_to-{space}_mode-surface_xfm.txt"
    f_surface_mni_to_native = out_xfm / f"{stem}_from-{space}_to-native_mode-surface_xfm.txt"
    f_surface_json = out_xfm / f"{stem}_desc-surface_xfm.json"

    # Legacy compatibility names. These are point matrices, despite the old
    # mode-image naming; new downstream code should prefer xfm/*mode-surface*.
    f_legacy_native_to_mni = out_anat / f"{stem}_from-T1w_to-{space}_mode-image_xfm.txt"
    f_legacy_mni_to_native = out_anat / f"{stem}_from-{space}_to-T1w_mode-image_xfm.txt"
    f_legacy_json = out_anat / f"{stem}_from-T1w_to-{space}_mode-image_xfm.json"

    orig_mgz = fs_mri / "orig.mgz"
    aseg_mgz = fs_mri / "aseg.mgz"
    aparc_mgz = fs_mri / "aparc+aseg.mgz"
    filled_mgz = fs_mri / "filled.mgz"

    for required in (orig_mgz, aseg_mgz):
        if not required.exists():
            raise FileNotFoundError(f"Missing required FreeSurfer file: {required}")

    if write_aparc_aseg and not aparc_mgz.exists():
        msg = f"[{stem}] Requested aparc+aseg but missing: {aparc_mgz}"
        if strict_volumes:
            raise FileNotFoundError(msg)
        logger.warning(msg)

    if write_filled and not filled_mgz.exists():
        msg = f"[{stem}] Requested filled.mgz but missing: {filled_mgz}"
        if strict_volumes:
            raise FileNotFoundError(msg)
        logger.warning(msg)

    # ---- 1) Export native volumes directly from MGZ using nibabel ----
    if save_raw_t1_export and ((not f_t1_raw.exists()) or (not skip_existing)):
        logger.info("[%s] Export raw T1: orig.mgz -> %s", stem, f_t1_raw.name)
        save_nifti_lossless_like(load_nib(orig_mgz), f_t1_raw)

    if use_n4:
        if (not f_t1_native.exists()) or (not skip_existing):
            if save_raw_t1_export:
                if (not f_t1_raw.exists()) or (not skip_existing):
                    save_nifti_lossless_like(load_nib(orig_mgz), f_t1_raw)
                source_for_n4 = f_t1_raw
                run_n4_bias_correction(
                    t1_native_path=source_for_n4,
                    out_path=f_t1_native,
                    use_mask=n4_use_mask,
                    shrink_factor=n4_shrink_factor,
                    spline_distance=n4_spline_distance,
                    logger=logger,
                )
            else:
                with tempfile.TemporaryDirectory(prefix=f"{stem}_fsraw_") as tmpdir:
                    source_for_n4 = Path(tmpdir) / f"{stem}_desc-fsraw_T1w.nii.gz"
                    save_nifti_lossless_like(load_nib(orig_mgz), source_for_n4)
                    run_n4_bias_correction(
                        t1_native_path=source_for_n4,
                        out_path=f_t1_native,
                        use_mask=n4_use_mask,
                        shrink_factor=n4_shrink_factor,
                        spline_distance=n4_spline_distance,
                        logger=logger,
                    )
    else:
        if (not f_t1_native.exists()) or (not skip_existing):
            logger.info("[%s] Export native T1 without N4: orig.mgz -> %s", stem, f_t1_native.name)
            save_nifti_lossless_like(load_nib(orig_mgz), f_t1_native)

    if (not f_aseg_native.exists()) or (not skip_existing):
        logger.info("[%s] Export native aseg: aseg.mgz -> %s", stem, f_aseg_native.name)
        save_nifti_lossless_like(load_nib(aseg_mgz), f_aseg_native)

    if write_aparc_aseg and aparc_mgz.exists() and ((not f_aparc_native.exists()) or (not skip_existing)):
        logger.info("[%s] Export native aparc+aseg -> %s", stem, f_aparc_native.name)
        save_nifti_lossless_like(load_nib(aparc_mgz), f_aparc_native)

    if write_filled and filled_mgz.exists() and ((not f_filled_native.exists()) or (not skip_existing)):
        logger.info("[%s] Export native filled -> %s", stem, f_filled_native.name)
        save_nifti_lossless_like(load_nib(filled_mgz), f_filled_native)

    # ---- 2) Estimate/reuse linear registration and explicit surface matrices ----
    need_register = (not skip_existing) or (not f_aff_mat.exists()) or (not f_t1_mni.exists())
    if need_register:
        reg = estimate_linear_registration(
            fixed_mni_path=mni_template,
            moving_t1_path=f_t1_native,
            transform_type=transform_type,
            random_seed=random_seed,
            out_affine_mat_path=f_aff_mat,
            logger=logger,
        )
        logger.info("[%s] Write warped MNI T1: %s", stem, f_t1_mni.name)
        ants.image_write(reg["warpedmovout"], str(f_t1_mni))
    else:
        logger.debug("[%s] Reusing existing ANTs affine and T1 MNI output.", stem)

    need_surface_xfm = (
        (not skip_existing)
        or (not f_surface_native_to_mni.exists())
        or (not f_surface_mni_to_native.exists())
        or (not f_surface_json.exists())
    )
    if need_surface_xfm:
        native_to_mni_ras, mni_to_native_ras = compute_surface_point_matrices_from_ants_mat(f_aff_mat)
        save_matrix_txt(f_surface_native_to_mni, native_to_mni_ras)
        save_matrix_txt(f_surface_mni_to_native, mni_to_native_ras)
        save_surface_transform_json(
            f_surface_json,
            source_ants_transform=f_aff_mat,
            native_to_mni_ras=native_to_mni_ras,
            mni_to_native_ras=mni_to_native_ras,
            transform_type=transform_type,
            space=space,
            use_n4=use_n4,
            n4_use_mask=n4_use_mask,
            random_seed=random_seed,
        )
        logger.info("[%s] Surface transforms written under: %s", stem, out_xfm)
    else:
        native_to_mni_ras = np.loadtxt(f_surface_native_to_mni, dtype=np.float64).reshape(4, 4)
        mni_to_native_ras = np.loadtxt(f_surface_mni_to_native, dtype=np.float64).reshape(4, 4)

    if write_legacy_anat_xfms:
        if (not skip_existing) or (not f_legacy_native_to_mni.exists()) or (not f_legacy_mni_to_native.exists()):
            save_matrix_txt(f_legacy_native_to_mni, native_to_mni_ras)
            save_matrix_txt(f_legacy_mni_to_native, mni_to_native_ras)
            legacy_payload = {
                "Description": "Compatibility copy of surface point transforms. Prefer xfm/*_mode-surface_xfm.txt for new code.",
                "ForwardMatrix": str(f_legacy_native_to_mni.name),
                "ForwardDirection": f"native scannerRAS -> {space} RAS",
                "InverseMatrix": str(f_legacy_mni_to_native.name),
                "InverseDirection": f"{space} RAS -> native scannerRAS",
                "CanonicalSurfaceTransformJson": str(f_surface_json.relative_to(out_sub)),
            }
            f_legacy_json.write_text(json.dumps(legacy_payload, indent=2) + "\n", encoding="utf-8")

    # ---- 3) Resample label / auxiliary volumes to MNI ----
    if (not f_aseg_mni.exists()) or (not skip_existing):
        apply_linear_transform_to_volume(
            moving_path=f_aseg_native,
            fixed_path=mni_template,
            affine_mat_path=f_aff_mat,
            out_path=f_aseg_mni,
            interpolation="nearestNeighbor",
            logger=logger,
        )

    if write_aparc_aseg and f_aparc_native.exists() and ((not f_aparc_mni.exists()) or (not skip_existing)):
        apply_linear_transform_to_volume(
            moving_path=f_aparc_native,
            fixed_path=mni_template,
            affine_mat_path=f_aff_mat,
            out_path=f_aparc_mni,
            interpolation="nearestNeighbor",
            logger=logger,
        )

    if write_filled and f_filled_native.exists() and ((not f_filled_mni.exists()) or (not skip_existing)):
        apply_linear_transform_to_volume(
            moving_path=f_filled_native,
            fixed_path=mni_template,
            affine_mat_path=f_aff_mat,
            out_path=f_filled_mni,
            interpolation="nearestNeighbor",
            logger=logger,
        )

    # ---- 4) Surface handling: tkRAS -> scanner RAS -> MNI152 RAS ----
    tkras_to_scanner_ras = get_tkras_to_scanner_ras_from_orig(orig_mgz)

    for hemi in hemis:
        hemi_u = str(hemi).upper()
        if hemi_u not in {"L", "R"}:
            raise ValueError(f"Unsupported hemi '{hemi}'. Use L and/or R.")
        fs_hemi = "lh" if hemi_u == "L" else "rh"

        for surf_name in surface_names:
            fs_path = resolve_fs_surface_path(fs_surf, fs_hemi, str(surf_name))
            if fs_path is None:
                msg = f"[{stem}] Missing surface for hemi={hemi_u}, type={surf_name}"
                if strict_surfaces:
                    raise FileNotFoundError(msg)
                logger.warning(msg)
                continue

            out_native = out_surfaces / f"{stem}_hemi-{hemi_u}_{surf_name}.surf.ply"
            out_mni = out_surfaces / f"{stem}_space-{space}_hemi-{hemi_u}_{surf_name}.surf.ply"

            if skip_existing and out_native.exists() and out_mni.exists():
                logger.debug("[%s] Skip existing surfaces: %s / %s", stem, out_native.name, out_mni.name)
                continue

            logger.info("[%s] Surface %s %s -> native/scanner and space-%s PLY", stem, hemi_u, surf_name, space)
            verts_tkras, faces = read_geometry(str(fs_path))
            verts_scanner_ras = apply_affine(tkras_to_scanner_ras, verts_tkras)
            verts_mni_ras = apply_affine(native_to_mni_ras, verts_scanner_ras)

            if (not out_native.exists()) or (not skip_existing):
                save_like_ply_ascii(out_native, verts_scanner_ras, faces)
            if (not out_mni.exists()) or (not skip_existing):
                save_like_ply_ascii(out_mni, verts_mni_ras, faces)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
@app.callback(invoke_without_command=True)
def main(
    freesurfer_root: Path = typer.Option(..., "--freesurfer-root", exists=True, file_okay=False, dir_okay=True, help="Root containing FreeSurfer subject directories."),
    out_deriv_root: Path = typer.Option(..., "--out-deriv-root", help="Output BIDS-derivatives-like root, typically derivatives/sc-preproc-0.1."),
    mni_template: Path = typer.Option(..., "--mni-template", exists=True, file_okay=True, dir_okay=False, help="Reference MNI template image."),
    participant_label: Optional[list[str]] = typer.Option(None, "--participant-label", "-p", help="One or more participants, e.g. sub-100307 or 100307."),
    session_label: str = typer.Option("01", "--session-label", "-s", help="Session label, written as ses-<label>."),
    space: str = typer.Option("MNI152", "--space", help="Target space label to use in output filenames."),
    hemi: list[str] = typer.Option(["L", "R"], "--hemi", help="Hemisphere(s) to export: L and/or R."),
    surface: list[str] = typer.Option(["white", "pial"], "--surface", help="Surface name(s) to export."),
    with_aparc_aseg: bool = typer.Option(True, "--with-aparc-aseg/--no-aparc-aseg", help="Also export/resample aparc+aseg.mgz when available."),
    with_filled: bool = typer.Option(True, "--with-filled/--no-filled", help="Also export/resample filled.mgz when available."),
    strict_volumes: bool = typer.Option(False, "--strict-volumes", help="Fail if requested optional volumes are missing."),
    n4: bool = typer.Option(True, "--n4/--no-n4", help="Apply N4 bias-field correction to orig.mgz before registration."),
    n4_use_mask: bool = typer.Option(True, "--n4-use-mask/--no-n4-use-mask", help="Use ants.get_mask(image) as the N4 mask."),
    n4_shrink_factor: int = typer.Option(4, "--n4-shrink-factor", min=1, help="N4 shrink factor."),
    n4_spline_distance: float = typer.Option(200.0, "--n4-spline-distance", min=1.0, help="N4 spline distance parameter."),
    transform_type: str = typer.Option("affine", "--transform-type", help="Linear registration type: rigid or affine."),
    save_raw_t1_export: bool = typer.Option(True, "--save-raw-t1-export/--no-save-raw-t1-export", help="Also save orig.mgz as desc-fsraw_T1w.nii.gz."),
    random_seed: Optional[int] = typer.Option(None, "--random-seed", help="Optional ANTsPy random seed for more deterministic behavior."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Recompute outputs even if files exist."),
    strict_surfaces: bool = typer.Option(False, "--strict-surfaces", help="Fail the subject if any requested surface is missing."),
    write_legacy_anat_xfms: bool = typer.Option(True, "--write-legacy-anat-xfms/--no-write-legacy-anat-xfms", help="Also write compatibility transform copies in anat/."),
    start: Optional[int] = typer.Option(None, "--start", help="Process subjects from this index (0-based) after sorting."),
    stop: Optional[int] = typer.Option(None, "--stop", help="Stop before this index (0-based) after sorting."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Optional path to write logs."),
    verbosity: int = typer.Option(0, "-v", count=True),
) -> None:
    logger = setup_logger(verbosity=verbosity, log_file=log_file)

    transform_type = transform_type.lower().strip()
    if transform_type not in {"rigid", "affine"}:
        raise typer.BadParameter("--transform-type must be 'rigid' or 'affine'.")

    safe_mkdir(out_deriv_root)
    write_dataset_description(out_deriv_root)

    if participant_label is None or len(participant_label) == 0:
        logger.info("No --participant-label provided. Discovering subjects in %s", freesurfer_root)
        subjects = discover_subjects(freesurfer_root)
        logger.info("Discovered %d subject(s).", len(subjects))
    else:
        subjects = [bids_sub_id(x) for x in participant_label]

    subjects = sorted(subjects)
    if start is not None or stop is not None:
        subjects = subjects[start:stop]
        logger.info("After slicing (--start/--stop), processing %d subject(s).", len(subjects))

    if not subjects:
        logger.error("No subjects to process.")
        raise typer.Exit(code=1)

    skip_existing = not overwrite
    failed: list[str] = []

    for subject in subjects:
        try:
            process_one(
                fs_root=freesurfer_root,
                out_root=out_deriv_root,
                participant_label=subject,
                session_label=session_label,
                mni_template=mni_template,
                space=space,
                surface_names=surface,
                hemis=hemi,
                write_aparc_aseg=with_aparc_aseg,
                write_filled=with_filled,
                strict_volumes=strict_volumes,
                use_n4=n4,
                n4_use_mask=n4_use_mask,
                n4_shrink_factor=n4_shrink_factor,
                n4_spline_distance=n4_spline_distance,
                transform_type=transform_type,
                save_raw_t1_export=save_raw_t1_export,
                random_seed=random_seed,
                skip_existing=skip_existing,
                strict_surfaces=strict_surfaces,
                write_legacy_anat_xfms=write_legacy_anat_xfms,
                logger=logger,
            )
        except Exception as e:
            failed.append(subject)
            logger.exception("[%s] FAILED: %s", subject, e)

    if failed:
        logger.error("Done with failures (%d): %s", len(failed), ", ".join(failed))
        raise typer.Exit(code=1)

    logger.info("Done. Outputs written under: %s", out_deriv_root)


if __name__ == "__main__":
    app()
