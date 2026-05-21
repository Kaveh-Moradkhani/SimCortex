from __future__ import annotations

from pathlib import Path


def _sub_id(subject: str) -> str:
    s = str(subject).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def _ses_id(ses: str = "01") -> str:
    s = str(ses).strip()
    return s if s.startswith("ses-") else f"ses-{s}"


def _stem(subject: str, ses: str = "01") -> str:
    sub = _sub_id(subject)
    ses_id = _ses_id(ses)
    return f"{sub}_{ses_id}"


def anat_dir(root: str | Path, subject: str, ses: str = "01") -> Path:
    return Path(root) / _sub_id(subject) / _ses_id(ses) / "anat"


def surf_dir(root: str | Path, subject: str, ses: str = "01") -> Path:
    return Path(root) / _sub_id(subject) / _ses_id(ses) / "surfaces"


def t1_mni_path(preproc_root: str | Path, subject: str, ses: str = "01", space: str = "MNI152") -> str:
    sub = _sub_id(subject)
    ses_id = _ses_id(ses)
    return str(
        anat_dir(preproc_root, sub, ses_id)
        / f"{sub}_{ses_id}_space-{space}_desc-preproc_T1w.nii.gz"
    )


def seg9_dseg_path(seg_root: str | Path, subject: str, ses: str = "01", space: str = "MNI152") -> str:
    sub = _sub_id(subject)
    ses_id = _ses_id(ses)
    return str(
        anat_dir(seg_root, sub, ses_id)
        / f"{sub}_{ses_id}_space-{space}_desc-seg9_dseg.nii.gz"
    )


def out_anat_dir(out_root: str | Path, subject: str, ses: str = "01") -> str:
    return str(anat_dir(out_root, subject, ses))


def out_surf_dir(out_root: str | Path, subject: str, ses: str = "01") -> str:
    return str(surf_dir(out_root, subject, ses))
