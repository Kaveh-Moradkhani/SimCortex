#!/usr/bin/env python3
"""
MRI-only SimCortex preprocessing for inference: native T1w -> MNI152.

This path is used when FreeSurfer surfaces are not needed/available. It writes
MNI-space T1w images, raw ANTs image transforms, explicit RAS-mm surface point
transforms, and success/failure manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import ants
import nibabel as nib
import numpy as np

log = logging.getLogger("simcortex.preproc.mri_to_mni_inference")

SUPPORTED_EXTS = (".nii.gz", ".nii", ".mgz", ".mgh")
PREPROC_DERIVATIVE_NAME = "scpp-preproc-0.1"
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float64)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_sub(label: str) -> str:
    label = str(label).strip()
    return label if label.startswith("sub-") else f"sub-{label}"


def ensure_ses(label: str) -> str:
    label = str(label).strip()
    return label if label.startswith("ses-") else f"ses-{label}"


def strip_nii_gz_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def is_supported_image(path: Path) -> bool:
    s = str(path)
    return any(s.endswith(ext) for ext in SUPPORTED_EXTS)


def is_probably_raw_bids_t1w(path: Path) -> bool:
    """Reject common derivative/preprocessed T1w files during BIDS discovery."""
    if not path.is_file() or not is_supported_image(path):
        return False
    name = path.name
    stem = strip_nii_gz_name(path)
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        return False
    if not stem.endswith("_T1w"):
        return False
    tokens = stem.split("_")
    if any(tok.startswith("space-") for tok in tokens):
        return False
    blocked_desc = {"desc-preproc", "desc-fsraw", "desc-brain", "desc-skullstripped"}
    if any(tok in blocked_desc for tok in tokens):
        return False
    return True


def infer_subject_session_from_path(t1w: Path) -> tuple[str, str]:
    subject = None
    session = None

    for part in t1w.parts:
        if part.startswith("sub-"):
            subject = part
        if part.startswith("ses-"):
            session = part

    if subject is None:
        for tok in strip_nii_gz_name(t1w).split("_"):
            if tok.startswith("sub-"):
                subject = tok
                break

    if session is None:
        for tok in strip_nii_gz_name(t1w).split("_"):
            if tok.startswith("ses-"):
                session = tok
                break

    if subject is None:
        log.warning("Could not infer subject from %s; using sub-single", t1w)
    if session is None:
        log.warning("Could not infer session from %s; using ses-01", t1w)

    return subject or "sub-single", session or "ses-01"


def find_bids_t1w_files(
    bids_root: Path,
    participant_labels: Optional[list[str]] = None,
    session_labels: Optional[list[str]] = None,
) -> list[tuple[str, str, Path]]:
    """
    Find raw T1w files in a BIDS-like layout.

    Preferred:
      bids_root/sub-*/ses-*/anat/*_T1w.nii.gz
      bids_root/sub-*/anat/*_T1w.nii.gz

    Fallback, for checked FreeSurfer-style development inputs:
      bids_root/sub-*/ses-*/mri/orig.mgz
    """
    if participant_labels:
        subjects = [ensure_sub(x) for x in participant_labels]
    else:
        subjects = sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())

    wanted_sessions = {ensure_ses(x) for x in session_labels} if session_labels else None
    items: list[tuple[str, str, Path]] = []

    for sub in subjects:
        sub_dir = bids_root / sub
        if not sub_dir.exists():
            log.warning("Subject folder not found: %s", sub_dir)
            continue

        ses_dirs = sorted(p for p in sub_dir.glob("ses-*") if p.is_dir())
        if ses_dirs:
            for ses_dir in ses_dirs:
                ses = ses_dir.name
                if wanted_sessions is not None and ses not in wanted_sessions:
                    continue

                added = False
                anat_dir = ses_dir / "anat"
                if anat_dir.exists():
                    candidates = sorted(p for p in anat_dir.glob("*_T1w.nii*") if is_probably_raw_bids_t1w(p))
                    for p in candidates:
                        items.append((sub, ses, p))
                        added = True

                mri_dir = ses_dir / "mri"
                if not added and mri_dir.exists():
                    for fname in ("orig.mgz", "T1.mgz", "norm.mgz", "brain.mgz"):
                        p = mri_dir / fname
                        if p.exists():
                            log.warning("No raw anat/*_T1w.nii* found for %s/%s; using fallback MRI file: %s", sub, ses, p)
                            items.append((sub, ses, p))
                            break
        else:
            ses = "ses-01"
            if wanted_sessions is not None and ses not in wanted_sessions:
                continue
            anat_dir = sub_dir / "anat"
            if anat_dir.exists():
                candidates = sorted(p for p in anat_dir.glob("*_T1w.nii*") if is_probably_raw_bids_t1w(p))
                for p in candidates:
                    items.append((sub, ses, p))

            mri_dir = sub_dir / "mri"
            if mri_dir.exists() and not any(row[0] == sub and row[1] == ses for row in items):
                p = mri_dir / "orig.mgz"
                if p.exists():
                    log.warning("No raw anat/*_T1w.nii* found for %s/%s; using fallback MRI file: %s", sub, ses, p)
                    items.append((sub, ses, p))

    return items


def convert_to_temp_nifti_if_needed(src: Path, tmpdir: Path, canonicalize_mgz: bool = False) -> Path:
    if str(src).endswith((".nii", ".nii.gz")):
        return src
    if str(src).endswith((".mgz", ".mgh")):
        img = nib.load(str(src))
        if canonicalize_mgz:
            log.warning("Canonicalizing MGZ/MGH to closest RAS orientation before ANTs: %s", src)
            img = nib.as_closest_canonical(img)
        out = tmpdir / f"{src.stem}.nii.gz"
        nib.save(img, str(out))
        return out
    raise ValueError(f"Unsupported image format: {src}")


def save_float32_nifti_like_ants(img: ants.ANTsImage, out_path: Path) -> None:
    """Write an ANTs image as float32 NIfTI without leaving temp files in anat/."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ants_write_") as td:
        tmp = Path(td) / "tmp.nii.gz"
        ants.image_write(img, str(tmp))
        nii = nib.load(str(tmp))
        data = np.asarray(nii.get_fdata(dtype=np.float32), dtype=np.float32)
        out_img = nib.Nifti1Image(data, affine=nii.affine, header=nii.header)
        out_img.set_data_dtype(np.float32)
        nib.save(out_img, str(out_path))


def create_compat_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.name)
    except Exception:
        shutil.copy2(src, dst)


def write_dataset_description(deriv_root: Path) -> None:
    deriv_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "Name": PREPROC_DERIVATIVE_NAME,
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "SimCortex MRI-only preprocessing",
                "Description": "ANTsPy N4 and linear registration of native T1w images to MNI152.",
            }
        ],
    }
    (deriv_root / "dataset_description.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# =============================================================================
# Surface-space affine export
# =============================================================================
def ants_affine_to_homogeneous_lps(transform_path: Path) -> np.ndarray:
    """
    Convert an ANTs/ITK affine transform to a 4x4 homogeneous matrix in LPS mm.

    ANTs/ITK affine parameters are interpreted as:
        y = A @ (x - c) + c + t
    where c is the fixed transform center. The equivalent homogeneous matrix is:
        y = A @ x + (t + c - A @ c)
    """
    tx = ants.read_transform(str(transform_path))
    params = np.asarray(tx.parameters, dtype=np.float64)
    fixed = np.asarray(tx.fixed_parameters, dtype=np.float64)

    if params.size < 12:
        raise ValueError(
            f"Expected at least 12 affine parameters in {transform_path}, got {params.size}. "
            "Surface matrix export only supports affine-like ANTs transforms."
        )
    if fixed.size < 3:
        raise ValueError(f"Expected 3 fixed parameters in {transform_path}, got {fixed.size}.")

    a = params[:9].reshape(3, 3)
    t = params[9:12]
    c = fixed[:3]

    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = a
    m[:3, 3] = t + c - a @ c
    return m


def ants_lps_matrix_to_ras_matrix(m_lps: np.ndarray) -> np.ndarray:
    """Convert a 4x4 point transform matrix from ITK/ANTs LPS to neuroimaging RAS."""
    m_lps = np.asarray(m_lps, dtype=np.float64)
    if m_lps.shape != (4, 4):
        raise ValueError(f"Expected matrix shape (4, 4), got {m_lps.shape}")
    return LPS_TO_RAS @ m_lps @ LPS_TO_RAS


def save_matrix_txt(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=np.float64), fmt="%.10g")


def save_surface_transform_json(
    path: Path,
    *,
    source_transform: Path,
    native_to_mni_ras: np.ndarray,
    mni_to_native_ras: np.ndarray,
    transform_type: str,
    n4: bool,
    n4_use_mask: bool,
    n4_shrink_factor: int,
    n4_spline_distance: float,
    random_seed: Optional[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Surface point transforms in RAS-mm coordinates. Apply to homogeneous row/column points consistently as documented in SimCortex.",
        "source_ants_transform": str(source_transform),
        "ants_coordinate_convention": "LPS",
        "surface_coordinate_convention": "RAS",
        "transform_type": transform_type,
        "matrix_application": "column homogeneous points: y = M @ x; code may equivalently use row_hom @ M.T",
        "native_to_MNI152_mode_surface_xfm": native_to_mni_ras.tolist(),
        "MNI152_to_native_mode_surface_xfm": mni_to_native_ras.tolist(),
        "n4": bool(n4),
        "n4_use_mask": bool(n4_use_mask),
        "n4_shrink_factor": int(n4_shrink_factor),
        "n4_spline_distance": float(n4_spline_distance),
        "random_seed": random_seed,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_surface_affine_transforms(
    *,
    subject: str,
    session: str,
    xfm_out: Path,
    native_to_mni_ants_transform: Optional[Path],
    transform_type: str,
    n4: bool,
    n4_use_mask: bool,
    n4_shrink_factor: int,
    n4_spline_distance: float,
    random_seed: Optional[int],
) -> dict:
    """
    Write explicit RAS-mm surface transforms for affine-like registrations.

    The copied ANTs forward .mat is an image-resampling transform. Read as a
    point transform, it maps MNI/fixed -> native/moving; its inverse maps
    native scannerRAS -> MNI152 RAS.
    """
    if native_to_mni_ants_transform is None:
        log.warning("No ANTs affine transform available; surface transform matrices were not written.")
        return {}

    if not native_to_mni_ants_transform.exists():
        log.warning("ANTs affine transform does not exist: %s", native_to_mni_ants_transform)
        return {}

    if native_to_mni_ants_transform.suffix != ".mat":
        log.warning("Surface matrix export currently supports affine .mat transforms only; got: %s", native_to_mni_ants_transform)
        return {}

    mni_to_native_lps = ants_affine_to_homogeneous_lps(native_to_mni_ants_transform)
    mni_to_native_ras = ants_lps_matrix_to_ras_matrix(mni_to_native_lps)
    native_to_mni_ras = np.linalg.inv(mni_to_native_ras)

    native_to_mni_txt = xfm_out / f"{subject}_{session}_from-native_to-MNI152_mode-surface_xfm.txt"
    mni_to_native_txt = xfm_out / f"{subject}_{session}_from-MNI152_to-native_mode-surface_xfm.txt"
    json_path = xfm_out / f"{subject}_{session}_desc-surface_xfm.json"

    save_matrix_txt(native_to_mni_txt, native_to_mni_ras)
    save_matrix_txt(mni_to_native_txt, mni_to_native_ras)
    save_surface_transform_json(
        json_path,
        source_transform=native_to_mni_ants_transform,
        native_to_mni_ras=native_to_mni_ras,
        mni_to_native_ras=mni_to_native_ras,
        transform_type=transform_type,
        n4=n4,
        n4_use_mask=n4_use_mask,
        n4_shrink_factor=n4_shrink_factor,
        n4_spline_distance=n4_spline_distance,
        random_seed=random_seed,
    )

    log.info("Surface transform written: %s", native_to_mni_txt)
    log.info("Surface transform written: %s", mni_to_native_txt)

    return {
        "surface_xfm_native_to_mni": str(native_to_mni_txt),
        "surface_xfm_mni_to_native": str(mni_to_native_txt),
        "surface_xfm_json": str(json_path),
    }


# =============================================================================
# Main preprocessing
# =============================================================================
def expected_output_paths(
    *,
    out_root: Path,
    derivative_name: str,
    subject: str,
    session: str,
) -> tuple[Path, Path, Path, Path]:
    deriv_root = out_root / derivative_name
    anat_out = deriv_root / subject / session / "anat"
    xfm_out = deriv_root / subject / session / "xfm"
    out_t1 = anat_out / f"{subject}_{session}_space-MNI152_desc-preproc_T1w.nii.gz"
    compat_t1 = anat_out / f"{subject}_{session}_space-MNI152_T1w.nii.gz"
    native_to_mni = xfm_out / f"{subject}_{session}_from-native_to-MNI152_mode-surface_xfm.txt"
    mni_to_native = xfm_out / f"{subject}_{session}_from-MNI152_to-native_mode-surface_xfm.txt"
    return out_t1, compat_t1, native_to_mni, mni_to_native


def preprocess_one_t1w_to_mni(
    t1w_path: Path,
    subject: str,
    session: str,
    out_root: Path,
    mni_path: Path,
    transform_type: str = "Affine",
    do_n4: bool = True,
    n4_use_mask: bool = True,
    n4_shrink_factor: int = 4,
    n4_spline_distance: float = 200.0,
    random_seed: Optional[int] = None,
    output_derivative_name: str = PREPROC_DERIVATIVE_NAME,
    overwrite: bool = False,
    canonicalize_mgz: bool = False,
) -> dict:
    """
    MRI-only inference preprocessing:
      T1w native/original space -> MNI152 1mm space
    """
    subject = ensure_sub(subject)
    session = ensure_ses(session)

    deriv_root = out_root / output_derivative_name
    anat_out = deriv_root / subject / session / "anat"
    xfm_out = deriv_root / subject / session / "xfm"
    anat_out.mkdir(parents=True, exist_ok=True)
    xfm_out.mkdir(parents=True, exist_ok=True)
    write_dataset_description(deriv_root)

    out_t1 = anat_out / f"{subject}_{session}_space-MNI152_desc-preproc_T1w.nii.gz"
    compat_t1 = anat_out / f"{subject}_{session}_space-MNI152_T1w.nii.gz"
    native_to_mni_txt = xfm_out / f"{subject}_{session}_from-native_to-MNI152_mode-surface_xfm.txt"
    mni_to_native_txt = xfm_out / f"{subject}_{session}_from-MNI152_to-native_mode-surface_xfm.txt"

    base_row = {
        "subject": subject,
        "session": session,
        "input_t1w": str(t1w_path),
        "mni_t1w": str(out_t1),
        "mni_t1w_compat": str(compat_t1),
        "xfm_dir": str(xfm_out),
        "transform_type": transform_type,
        "n4": str(bool(do_n4)),
        "n4_use_mask": str(bool(n4_use_mask)),
        "n4_shrink_factor": str(int(n4_shrink_factor)),
        "n4_spline_distance": str(float(n4_spline_distance)),
        "random_seed": "" if random_seed is None else str(random_seed),
    }

    required_for_skip = [out_t1, compat_t1, native_to_mni_txt, mni_to_native_txt]
    if not overwrite and all(p.exists() for p in required_for_skip):
        log.info("Skipping existing preprocessing for %s/%s", subject, session)
        row = dict(base_row)
        row.update({
            "status": "skipped_existing",
            "surface_xfm_native_to_mni": str(native_to_mni_txt),
            "surface_xfm_mni_to_native": str(mni_to_native_txt),
            "surface_xfm_json": str(xfm_out / f"{subject}_{session}_desc-surface_xfm.json"),
        })
        return row

    log.info("Subject/session: %s / %s", subject, session)
    log.info("Input T1w: %s", t1w_path)
    log.info("MNI template: %s", mni_path)
    log.info("Output T1w MNI: %s", out_t1)

    copied_fwd: list[Path] = []
    copied_inv: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="scpp_mri_to_mni_") as td:
        tmpdir = Path(td)
        moving_nifti = convert_to_temp_nifti_if_needed(t1w_path, tmpdir, canonicalize_mgz=canonicalize_mgz)

        fixed = ants.image_read(str(mni_path)).clone("float")
        moving = ants.image_read(str(moving_nifti)).clone("float")

        if do_n4:
            log.info("Running N4 bias correction on moving T1w...")
            mask = ants.get_mask(moving) if n4_use_mask else None
            moving = ants.n4_bias_field_correction(
                moving,
                mask=mask,
                shrink_factor=int(n4_shrink_factor),
                spline_param=float(n4_spline_distance),
            )

        kwargs = {
            "fixed": fixed,
            "moving": moving,
            "type_of_transform": transform_type,
            "verbose": False,
        }
        if random_seed is not None:
            kwargs["random_seed"] = int(random_seed)

        log.info("Running ANTs registration: %s", transform_type)
        reg = ants.registration(**kwargs)

        save_float32_nifti_like_ants(reg["warpedmovout"], out_t1)
        create_compat_symlink_or_copy(out_t1, compat_t1)

        for idx, src in enumerate(reg.get("fwdtransforms", [])):
            src_path = Path(src)
            if not src_path.exists():
                continue
            suffix = "".join(src_path.suffixes)
            dst = xfm_out / f"{subject}_{session}_from-native_to-MNI152_mode-image_desc-ants_{idx}{suffix}"
            shutil.copy2(src_path, dst)
            copied_fwd.append(dst)

        for idx, src in enumerate(reg.get("invtransforms", [])):
            src_path = Path(src)
            if not src_path.exists():
                continue
            suffix = "".join(src_path.suffixes)
            dst = xfm_out / f"{subject}_{session}_from-MNI152_to-native_mode-image_desc-ants_{idx}{suffix}"
            shutil.copy2(src_path, dst)
            copied_inv.append(dst)

    first_fwd_mat = next((p for p in copied_fwd if p.suffix == ".mat"), None)
    surface_xfm_info = write_surface_affine_transforms(
        subject=subject,
        session=session,
        xfm_out=xfm_out,
        native_to_mni_ants_transform=first_fwd_mat,
        transform_type=transform_type,
        n4=do_n4,
        n4_use_mask=n4_use_mask,
        n4_shrink_factor=n4_shrink_factor,
        n4_spline_distance=n4_spline_distance,
        random_seed=random_seed,
    )

    row = dict(base_row)
    row.update({
        "status": "ok",
        "ants_fwd_transforms": ";".join(str(p) for p in copied_fwd),
        "ants_inv_transforms": ";".join(str(p) for p in copied_inv),
    })
    row.update(surface_xfm_info)
    return row


def write_tsv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)

    if not fields:
        return

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log.info("TSV written: %s", out_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MRI-only SimCortex preprocessing for inference: T1w -> MNI152 derivative."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--t1w", type=Path, help="Path to one T1w image (.nii.gz, .nii, .mgz, .mgh).")
    mode.add_argument("--bids-root", type=Path, help="BIDS root. Finds raw sub-*/ses-*/anat/*_T1w.nii.gz automatically.")

    p.add_argument("--participant-label", nargs="+", default=None)
    p.add_argument("--session-label", nargs="+", default=None)
    p.add_argument("--subject", default=None, help="Override subject label for --t1w mode.")
    p.add_argument("--session", default=None, help="Override session label for --t1w mode.")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--mni", type=Path, required=True)
    p.add_argument("--transform-type", default="Affine", choices=["Rigid", "Affine"])
    p.add_argument("--no-n4", action="store_true", help="Disable N4 bias-field correction.")
    p.add_argument("--n4-use-mask", dest="n4_use_mask", action="store_true", default=True, help="Use ants.get_mask(image) for N4.")
    p.add_argument("--no-n4-use-mask", dest="n4_use_mask", action="store_false", help="Do not use a mask for N4.")
    p.add_argument("--n4-shrink-factor", type=int, default=4)
    p.add_argument("--n4-spline-distance", type=float, default=200.0)
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--derivative-name", default=PREPROC_DERIVATIVE_NAME)
    p.add_argument("--overwrite", action="store_true", help="Recompute even if all expected outputs exist.")
    p.add_argument("--canonicalize-mgz", action="store_true", help="Canonicalize MGZ/MGH to closest RAS before ANTs. Off by default to avoid hidden train/inference drift.")
    p.add_argument("--verbose", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(args.verbose)

    if args.n4_shrink_factor < 1:
        raise ValueError("--n4-shrink-factor must be >= 1")
    if args.n4_spline_distance <= 0:
        raise ValueError("--n4-spline-distance must be > 0")

    mni_path = args.mni.expanduser().resolve()
    if not mni_path.exists():
        raise FileNotFoundError(f"MNI template not found: {mni_path}")

    out_root = args.out_root.expanduser().resolve()
    rows: list[dict] = []
    failed: list[dict] = []

    if args.t1w is not None:
        t1w = args.t1w.expanduser().resolve()
        if not t1w.exists():
            raise FileNotFoundError(f"T1w not found: {t1w}")

        inferred_sub, inferred_ses = infer_subject_session_from_path(t1w)
        subject = ensure_sub(args.subject) if args.subject else inferred_sub
        session = ensure_ses(args.session) if args.session else inferred_ses
        items = [(subject, session, t1w)]
    else:
        bids_root = args.bids_root.expanduser().resolve()
        if not bids_root.exists():
            raise FileNotFoundError(f"BIDS root not found: {bids_root}")

        items = find_bids_t1w_files(
            bids_root=bids_root,
            participant_labels=args.participant_label,
            session_labels=args.session_label,
        )
        if not items:
            raise RuntimeError(
                f"No raw T1w files found under {bids_root}. Expected sub-*/ses-*/anat/*_T1w.nii.gz "
                "or use --t1w directly."
            )

    log.info("Found %d T1w file(s).", len(items))
    for subject, session, t1w in items:
        try:
            rows.append(
                preprocess_one_t1w_to_mni(
                    t1w_path=t1w,
                    subject=subject,
                    session=session,
                    out_root=out_root,
                    mni_path=mni_path,
                    transform_type=args.transform_type,
                    do_n4=not args.no_n4,
                    n4_use_mask=args.n4_use_mask,
                    n4_shrink_factor=args.n4_shrink_factor,
                    n4_spline_distance=args.n4_spline_distance,
                    random_seed=args.random_seed,
                    output_derivative_name=args.derivative_name,
                    overwrite=args.overwrite,
                    canonicalize_mgz=args.canonicalize_mgz,
                )
            )
        except Exception as e:
            log.exception("FAILED %s/%s (%s): %s", subject, session, t1w, e)
            failed.append({
                "subject": ensure_sub(subject),
                "session": ensure_ses(session),
                "input_t1w": str(t1w),
                "error": repr(e),
            })

    deriv_root = out_root / args.derivative_name
    write_tsv(rows, deriv_root / "mri_to_mni_manifest.tsv")
    write_tsv(failed, deriv_root / "mri_to_mni_failed.tsv")

    if failed:
        log.error("Done with %d failure(s).", len(failed))
        raise SystemExit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
