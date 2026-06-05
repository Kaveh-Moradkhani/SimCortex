#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


log = logging.getLogger("simcortex.run_pipeline")


# =============================================================================
# Data structures
# =============================================================================

@dataclasses.dataclass(frozen=True)
class SubjectInput:
    subject: str
    session: str
    t1w_path: Path


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    out_root: Path
    work_root: Path
    project_root: Path
    mni: Path
    seg_ckpt: Path
    deform_ckpt: Path
    device: str
    space: str
    transform_type: str
    overwrite: bool
    keep_work: bool
    qc_collisions: bool
    initsurf_workers: int
    export_native: bool


@dataclasses.dataclass(frozen=True)
class SubjectLayout:
    subject: str
    session: str
    preproc_root: Path
    seg_root: Path
    initsurf_root: Path
    deform_root: Path
    tmp_dir: Path
    log_dir: Path
    final_dir: Path
    split_file: Path


# =============================================================================
# Basic helpers
# =============================================================================

def normalize_subject_id(x: str) -> str:
    x = str(x).strip()
    return x if x.startswith("sub-") else f"sub-{x}"


def normalize_session_id(x: str) -> str:
    x = str(x).strip()
    return x if x.startswith("ses-") else f"ses-{x}"


def session_value_for_hydra(session: str) -> str:
    return normalize_session_id(session)


def seconds_to_min(x: float) -> float:
    return round(float(x) / 60.0, 4)


def setup_logging(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = out_root / "run_pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")],
    )
    log.info("Logging to %s", log_file)


def build_layout(sub: SubjectInput, cfg: PipelineConfig) -> SubjectLayout:
    subject = normalize_subject_id(sub.subject)
    session = normalize_session_id(sub.session)

    return SubjectLayout(
        subject=subject,
        session=session,
        preproc_root=cfg.work_root / "sc-preproc-0.1",
        seg_root=cfg.work_root / "sc-seg-0.1",
        initsurf_root=cfg.work_root / "sc-initsurf-0.1",
        deform_root=cfg.work_root / "sc-deform-0.1",
        tmp_dir=cfg.work_root / "tmp" / subject / session,
        log_dir=cfg.work_root / "logs" / subject / session,
        final_dir=cfg.out_root / subject / session / "surfaces",
        split_file=cfg.work_root / "tmp" / subject / session / "split_one_subject.csv",
    )


def ensure_layout_dirs(layout: SubjectLayout) -> None:
    for p in [
        layout.preproc_root,
        layout.seg_root,
        layout.initsurf_root,
        layout.deform_root,
        layout.tmp_dir,
        layout.log_dir,
        layout.final_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def write_split_file(layout: SubjectLayout) -> None:
    layout.tmp_dir.mkdir(parents=True, exist_ok=True)
    layout.split_file.write_text(f"subject,split\n{layout.subject},test\n")


def check_required_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def subprocess_env(cfg: PipelineConfig) -> dict:
    env = os.environ.copy()
    src = str(cfg.project_root / "src")
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not old else f"{src}:{old}"
    return env


# =============================================================================
# Input discovery
# =============================================================================

def discover_single_subject(t1w: Path, subject: str, session: str) -> list[SubjectInput]:
    t1w = t1w.expanduser().resolve()
    check_required_file(t1w, "Input T1w MRI")
    return [SubjectInput(normalize_subject_id(subject), normalize_session_id(session), t1w)]


def choose_mri_candidate(subject: str, session: str, session_dir: Path) -> Optional[Path]:
    """
    Prefer raw-BIDS anat/*T1w* when present. Fall back to existing FreeSurfer MRI files.
    This does not run FreeSurfer; it only reads an available MRI image.
    """
    anat_dir = session_dir / "anat"
    raw_bids = sorted(anat_dir.glob(f"{subject}_{session}*T1w.nii*")) if anat_dir.exists() else []
    if raw_bids:
        if len(raw_bids) > 1:
            log.info("Multiple BIDS T1w candidates found for %s %s; using: %s", subject, session, raw_bids[0])
        return raw_bids[0]

    fs_priority = [
        session_dir / "mri" / "orig.mgz",
        session_dir / "mri" / "T1.mgz",
        session_dir / "mri" / "brain.mgz",
        session_dir / "mri" / "nu.mgz",
    ]
    fs_existing = [p for p in fs_priority if p.exists()]
    if not fs_existing:
        return None

    if len(fs_existing) > 1:
        log.info("Multiple MRI candidates found for %s %s; using priority candidate: %s", subject, session, fs_existing[0])
    return fs_existing[0]


def discover_bids_subjects(
    bids_root: Path,
    participant_labels: Optional[list[str]],
    sessions: Optional[list[str]],
) -> list[SubjectInput]:
    bids_root = bids_root.expanduser().resolve()
    check_required_file(bids_root, "BIDS root")

    subjects = [normalize_subject_id(x) for x in participant_labels] if participant_labels else sorted(
        p.name for p in bids_root.glob("sub-*") if p.is_dir()
    )
    requested_sessions = [normalize_session_id(s) for s in sessions] if sessions else None

    found: list[SubjectInput] = []
    for subject in subjects:
        subject_dir = bids_root / subject
        if not subject_dir.exists():
            log.warning("Requested subject not found: %s", subject_dir)
            continue

        session_dirs = [subject_dir / s for s in requested_sessions] if requested_sessions else sorted(
            p for p in subject_dir.glob("ses-*") if p.is_dir()
        )
        if not session_dirs:
            session_dirs = [subject_dir]

        for session_dir in session_dirs:
            if not session_dir.exists():
                log.warning("Requested session not found: %s", session_dir)
                continue

            session = session_dir.name if session_dir.name.startswith("ses-") else "ses-01"
            mri = choose_mri_candidate(subject, session, session_dir)
            if mri is None:
                log.warning("No usable MRI found for %s %s under %s", subject, session, session_dir)
                continue

            found.append(SubjectInput(subject=subject, session=session, t1w_path=mri.resolve()))

    if not found:
        raise RuntimeError("No usable subject/session MRI inputs found.")
    return found


# =============================================================================
# Command execution
# =============================================================================

def run_cmd(stage: str, cmd: list[str], log_path: Path, cwd: Path, cfg: PipelineConfig) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("[%s] started", stage)
    log.info("[%s] full log: %s", stage, log_path)

    t0 = time.time()
    with log_path.open("w") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(str(x) for x in cmd))
        f.write("\n\n")
        f.flush()

        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=subprocess_env(cfg),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.error("[%s] failed after %.2f sec. See: %s", stage, elapsed, log_path)
        raise RuntimeError(f"Stage '{stage}' failed with exit code {proc.returncode}. Full log: {log_path}")

    log.info("[%s] finished in %.2f sec", stage, elapsed)
    return elapsed


# =============================================================================
# Stage runners
# =============================================================================

def run_preprocessing_mri_only(sub: SubjectInput, layout: SubjectLayout, cfg: PipelineConfig) -> float:
    cmd = [
        sys.executable,
        "-m",
        "simcortex.preproc.mri_to_mni_inference",
        "--t1w",
        sub.t1w_path,
        "--subject",
        layout.subject,
        "--session",
        layout.session,
        "--out-root",
        cfg.work_root,
        "--mni",
        cfg.mni,
        "--transform-type",
        cfg.transform_type,
        "--derivative-name",
        layout.preproc_root.name,
    ]
    elapsed = run_cmd("preproc", cmd, layout.log_dir / "preproc.log", cfg.project_root, cfg)

    expected = layout.preproc_root / layout.subject / layout.session / "anat" / (
        f"{layout.subject}_{layout.session}_space-{cfg.space}_desc-preproc_T1w.nii.gz"
    )
    check_required_file(expected, "Preprocessed MNI T1w")

    alias = layout.preproc_root / layout.subject / layout.session / "anat" / (
        f"{layout.subject}_{layout.session}_space-{cfg.space}_T1w.nii.gz"
    )
    if not alias.exists():
        shutil.copy2(expected, alias)

    if cfg.export_native:
        check_required_file(
            mni_to_native_surface_xfm(layout),
            "MNI152-to-native surface transform matrix. Re-run preprocessing with the updated mri_to_mni_inference.py",
        )

    return elapsed


def run_segmentation(layout: SubjectLayout, cfg: PipelineConfig) -> float:
    cmd = [
        sys.executable,
        "-m",
        "simcortex.seg.inference",
        f"dataset.path={layout.preproc_root}",
        f"dataset.split_file={layout.split_file}",
        "dataset.split_name=test",
        f"dataset.session_label={session_value_for_hydra(layout.session)}",
        f"dataset.space={cfg.space}",
        f"model.ckpt_path={cfg.seg_ckpt}",
        f"trainer.device={cfg.device}",
        "trainer.batch_size=1",
        "trainer.num_workers=0",
        f"outputs.out_root={layout.seg_root}",
        f"outputs.log_dir={layout.log_dir / 'segmentation_internal'}",
    ]
    elapsed = run_cmd("segmentation", cmd, layout.log_dir / "segmentation.log", cfg.project_root, cfg)

    expected = layout.seg_root / layout.subject / layout.session / "anat" / (
        f"{layout.subject}_{layout.session}_space-{cfg.space}_desc-seg9_dseg.nii.gz"
    )
    check_required_file(expected, "Segmentation output")
    return elapsed


def run_initsurf(layout: SubjectLayout, cfg: PipelineConfig) -> float:
    cmd = [
        sys.executable,
        "-m",
        "simcortex.initsurf.generate",
        f"dataset.path={layout.preproc_root}",
        f"dataset.seg_root={layout.seg_root}",
        f"dataset.split_file={layout.split_file}",
        "dataset.split_name=test",
        f"dataset.session_label={session_value_for_hydra(layout.session)}",
        f"dataset.space={cfg.space}",
        f"outputs.out_root={layout.initsurf_root}",
        f"outputs.log_dir={layout.log_dir / 'initsurf_internal'}",
        f"n_workers={cfg.initsurf_workers}",
    ]
    elapsed = run_cmd("initsurf", cmd, layout.log_dir / "initsurf.log", cfg.project_root, cfg)

    surf_dir = layout.initsurf_root / layout.subject / layout.session / "surfaces"
    for p in [
        surf_dir / f"{layout.subject}_{layout.session}_space-{cfg.space}_hemi-L_pial.surf.ply",
        surf_dir / f"{layout.subject}_{layout.session}_space-{cfg.space}_hemi-L_white.surf.ply",
        surf_dir / f"{layout.subject}_{layout.session}_space-{cfg.space}_hemi-R_pial.surf.ply",
        surf_dir / f"{layout.subject}_{layout.session}_space-{cfg.space}_hemi-R_white.surf.ply",
    ]:
        check_required_file(p, "InitSurf output")
    return elapsed


def run_deform(layout: SubjectLayout, cfg: PipelineConfig) -> float:
    cmd = [
        sys.executable,
        "-m",
        "simcortex.deform.inference",
        f"dataset.path={layout.preproc_root}",
        f"dataset.initsurf_root={layout.initsurf_root}",
        f"dataset.split_file={layout.split_file}",
        "dataset.split_name=test",
        f"dataset.session_label={session_value_for_hydra(layout.session)}",
        f"dataset.space={cfg.space}",
        f"inference.device={cfg.device}",
        "inference.batch_size=1",
        "inference.num_workers=0",
        f"inference.overwrite={str(cfg.overwrite).lower()}",
        f"model.ckpt_path={cfg.deform_ckpt}",
        f"outputs.out_root={layout.deform_root}",
        f"outputs.log_dir={layout.log_dir / 'deform_internal'}",
    ]
    elapsed = run_cmd("deform", cmd, layout.log_dir / "deform.log", cfg.project_root, cfg)

    for p in expected_deform_surface_paths(layout.deform_root, layout.subject, layout.session, cfg.space).values():
        check_required_file(p, "Deform output")
    return elapsed


# =============================================================================
# Surface paths and export
# =============================================================================

def expected_deform_surface_paths(root: Path, subject: str, session: str, space: str) -> dict[str, Path]:
    surf_dir = root / subject / session / "surfaces"
    return surface_paths(surf_dir, subject, session, space)


def expected_final_surface_paths(final_dir: Path, subject: str, session: str, space: str) -> dict[str, Path]:
    return surface_paths(final_dir, subject, session, space)


def surface_paths(surf_dir: Path, subject: str, session: str, space: str) -> dict[str, Path]:
    return {
        "lh_pial": surf_dir / f"{subject}_{session}_space-{space}_desc-deform_hemi-L_pial.surf.ply",
        "lh_white": surf_dir / f"{subject}_{session}_space-{space}_desc-deform_hemi-L_white.surf.ply",
        "rh_pial": surf_dir / f"{subject}_{session}_space-{space}_desc-deform_hemi-R_pial.surf.ply",
        "rh_white": surf_dir / f"{subject}_{session}_space-{space}_desc-deform_hemi-R_white.surf.ply",
    }


def mni_to_native_surface_xfm(layout: SubjectLayout) -> Path:
    return layout.preproc_root / layout.subject / layout.session / "xfm" / (
        f"{layout.subject}_{layout.session}_from-MNI152_to-native_mode-surface_xfm.txt"
    )


def native_to_mni_surface_xfm(layout: SubjectLayout) -> Path:
    return layout.preproc_root / layout.subject / layout.session / "xfm" / (
        f"{layout.subject}_{layout.session}_from-native_to-MNI152_mode-surface_xfm.txt"
    )


def load_affine_matrix(path: Path) -> np.ndarray:
    check_required_file(path, "Surface transform matrix")
    M = np.loadtxt(path, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}, got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError(f"Non-finite values in transform matrix: {path}")
    return M


def apply_affine_to_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    ones = np.ones((verts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([verts, ones], axis=1)
    out = hom @ matrix.T
    return out[:, :3].astype(np.float32)


def transform_surface_with_matrix(src_ply: Path, dst_ply: Path, matrix: np.ndarray) -> None:
    import trimesh

    mesh = trimesh.load(src_ply, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values()]
        if not geoms:
            raise RuntimeError(f"Empty mesh scene: {src_ply}")
        mesh = trimesh.util.concatenate(geoms)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
        raise RuntimeError(f"Invalid vertices in {src_ply}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"Invalid faces in {src_ply}")

    out_verts = apply_affine_to_vertices(verts, matrix)
    if not np.isfinite(out_verts).all():
        raise RuntimeError(f"Non-finite vertices after transform: {src_ply}")

    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=out_verts, faces=faces, process=False).export(dst_ply)


def collect_final_surfaces(layout: SubjectLayout, cfg: PipelineConfig) -> float:
    t0 = time.time()
    layout.final_dir.mkdir(parents=True, exist_ok=True)

    srcs = expected_deform_surface_paths(layout.deform_root, layout.subject, layout.session, cfg.space)
    dsts = expected_final_surface_paths(layout.final_dir, layout.subject, layout.session, cfg.space)
    for key, src in srcs.items():
        check_required_file(src, f"Final source surface {key}")
        shutil.copy2(src, dsts[key])

    elapsed = time.time() - t0
    log.info("[collect] Copied 4 surfaces to %s", layout.final_dir)
    return elapsed


def native_surface_paths(final_dir: Path, subject: str, session: str) -> dict[str, Path]:
    return expected_final_surface_paths(final_dir, subject, session, "native")


def write_export_manifest(
    layout: SubjectLayout,
    cfg: PipelineConfig,
    matrix_path: Path,
    mni_paths: dict[str, Path],
    native_paths: dict[str, Path],
) -> None:
    manifest = cfg.out_root / layout.subject / layout.session / "export_manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for surf_name in ["lh_pial", "lh_white", "rh_pial", "rh_white"]:
        rows.append(
            {
                "subject": layout.subject,
                "session": layout.session,
                "surface_name": surf_name,
                "source_space": cfg.space,
                "target_space": "native",
                "source_path": str(mni_paths[surf_name]),
                "output_path": str(native_paths[surf_name]),
                "surface_transform_matrix": str(matrix_path),
                "surface_transform_direction": "MNI152RAS_to_nativeScannerRAS",
                "matrix_application": "row_homogeneous_vertices @ matrix.T",
                "coordinate_convention": "RAS-mm",
            }
        )

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    log.info("[export-native] Manifest written: %s", manifest)


def export_native_surfaces(layout: SubjectLayout, cfg: PipelineConfig) -> float:
    """
    Export MNI-space mesh vertices to native scanner-RAS space using the explicit
    RAS-mm surface matrix written during preprocessing.
    """
    t0 = time.time()
    matrix_path = mni_to_native_surface_xfm(layout)
    M = load_affine_matrix(matrix_path)

    mni_paths = expected_final_surface_paths(layout.final_dir, layout.subject, layout.session, cfg.space)
    out_paths = native_surface_paths(layout.final_dir, layout.subject, layout.session)

    for surf_name, src in mni_paths.items():
        check_required_file(src, f"MNI surface for native export: {surf_name}")
        dst = out_paths[surf_name]
        if dst.exists() and not cfg.overwrite:
            continue
        transform_surface_with_matrix(src, dst, M)

    write_export_manifest(layout, cfg, matrix_path, mni_paths, out_paths)

    elapsed = time.time() - t0
    log.info("[export-native] Exported 4 native-space surfaces in %.2f sec", elapsed)
    return elapsed


# =============================================================================
# QC
# =============================================================================

def run_collision_qc(final_dir: Path, subject: str, session: str, space: str, prefix: str) -> dict:
    qc = {
        f"{prefix}_lh_white_pial_collision": "NA",
        f"{prefix}_lh_white_pial_contacts": "NA",
        f"{prefix}_rh_white_pial_collision": "NA",
        f"{prefix}_rh_white_pial_contacts": "NA",
        f"{prefix}_lh_pial_rh_pial_collision": "NA",
        f"{prefix}_lh_pial_rh_pial_contacts": "NA",
    }

    try:
        import trimesh
        from trimesh.collision import CollisionManager
    except Exception as e:
        log.warning("[qc:%s] Collision QC skipped because trimesh/FCL is unavailable: %r", space, e)
        return qc

    paths = expected_final_surface_paths(final_dir, subject, session, space)
    for p in paths.values():
        if not p.exists():
            log.warning("[qc:%s] skipped because missing file: %s", space, p)
            return qc

    meshes = {k: trimesh.load(p, process=False) for k, p in paths.items()}

    def check(a: str, b: str) -> tuple[bool, int]:
        cm = CollisionManager()
        cm.add_object("a", meshes[a])
        cm.add_object("b", meshes[b])
        hit, contacts = cm.in_collision_internal(return_names=False, return_data=True)
        return bool(hit), 0 if contacts is None else int(len(contacts))

    pairs = [
        ("lh_white", "lh_pial", f"{prefix}_lh_white_pial_collision", f"{prefix}_lh_white_pial_contacts"),
        ("rh_white", "rh_pial", f"{prefix}_rh_white_pial_collision", f"{prefix}_rh_white_pial_contacts"),
        ("lh_pial", "rh_pial", f"{prefix}_lh_pial_rh_pial_collision", f"{prefix}_lh_pial_rh_pial_contacts"),
    ]
    for a, b, hit_key, n_key in pairs:
        hit, n = check(a, b)
        qc[hit_key] = int(hit)
        qc[n_key] = n
        log.info("[qc:%s] %s vs %s | collision=%s | contacts=%d", space, a, b, hit, n)
    return qc


# =============================================================================
# Summary
# =============================================================================

def write_summary_row(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subject", "session", "input_t1w", "status",
        "preproc_sec", "segmentation_sec", "initsurf_sec", "deform_sec", "collect_sec", "export_native_sec",
        "total_sec", "total_min", "final_dir",
        "lh_pial", "lh_white", "rh_pial", "rh_white",
        "native_lh_pial", "native_lh_white", "native_rh_pial", "native_rh_white",
        "qc_lh_white_pial_collision", "qc_lh_white_pial_contacts",
        "qc_rh_white_pial_collision", "qc_rh_white_pial_contacts",
        "qc_lh_pial_rh_pial_collision", "qc_lh_pial_rh_pial_contacts",
        "qc_native_lh_white_pial_collision", "qc_native_lh_white_pial_contacts",
        "qc_native_rh_white_pial_collision", "qc_native_rh_white_pial_contacts",
        "qc_native_lh_pial_rh_pial_collision", "qc_native_lh_pial_rh_pial_contacts",
        "error",
    ]

    exists = summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Run one subject/session
# =============================================================================

def run_one_subject(sub: SubjectInput, cfg: PipelineConfig, index: int, total: int) -> None:
    layout = build_layout(sub, cfg)
    ensure_layout_dirs(layout)
    write_split_file(layout)

    summary_path = cfg.out_root / "pipeline_summary.tsv"
    stage_times = {
        "preproc": 0.0,
        "segmentation": 0.0,
        "initsurf": 0.0,
        "deform": 0.0,
        "collect": 0.0,
        "export_native": 0.0,
    }
    qc_row: dict = {}
    status = "FAILED"
    error_msg = ""
    t_all0 = time.time()

    log.info("Starting item %d/%d", index, total)
    log.info("============================================================")
    log.info("Running subject/session: %s / %s", layout.subject, layout.session)
    log.info("Input MRI: %s", sub.t1w_path)
    log.info("Work root: %s", cfg.work_root)
    log.info("Final dir: %s", layout.final_dir)
    log.info("Stage logs: %s", layout.log_dir)
    log.info("============================================================")

    try:
        stage_times["preproc"] = run_preprocessing_mri_only(sub, layout, cfg)
        stage_times["segmentation"] = run_segmentation(layout, cfg)
        stage_times["initsurf"] = run_initsurf(layout, cfg)
        stage_times["deform"] = run_deform(layout, cfg)
        stage_times["collect"] = collect_final_surfaces(layout, cfg)

        if cfg.export_native:
            stage_times["export_native"] = export_native_surfaces(layout, cfg)

        if cfg.qc_collisions:
            qc_row.update(run_collision_qc(layout.final_dir, layout.subject, layout.session, cfg.space, prefix="qc"))
            if cfg.export_native:
                qc_row.update(run_collision_qc(layout.final_dir, layout.subject, layout.session, "native", prefix="qc_native"))

        status = "OK"

    except Exception as e:
        error_msg = repr(e)
        status = "FAILED"
        raise

    finally:
        total_sec = time.time() - t_all0
        final_paths = expected_final_surface_paths(layout.final_dir, layout.subject, layout.session, cfg.space)
        native_paths = native_surface_paths(layout.final_dir, layout.subject, layout.session)

        row = {
            "subject": layout.subject,
            "session": layout.session,
            "input_t1w": str(sub.t1w_path),
            "status": status,
            "preproc_sec": round(stage_times["preproc"], 3),
            "segmentation_sec": round(stage_times["segmentation"], 3),
            "initsurf_sec": round(stage_times["initsurf"], 3),
            "deform_sec": round(stage_times["deform"], 3),
            "collect_sec": round(stage_times["collect"], 3),
            "export_native_sec": round(stage_times["export_native"], 3),
            "total_sec": round(total_sec, 3),
            "total_min": seconds_to_min(total_sec),
            "final_dir": str(layout.final_dir),
            "lh_pial": str(final_paths["lh_pial"]),
            "lh_white": str(final_paths["lh_white"]),
            "rh_pial": str(final_paths["rh_pial"]),
            "rh_white": str(final_paths["rh_white"]),
            "native_lh_pial": str(native_paths["lh_pial"]) if cfg.export_native else "",
            "native_lh_white": str(native_paths["lh_white"]) if cfg.export_native else "",
            "native_rh_pial": str(native_paths["rh_pial"]) if cfg.export_native else "",
            "native_rh_white": str(native_paths["rh_white"]) if cfg.export_native else "",
            "error": error_msg,
        }
        row.update(qc_row)
        write_summary_row(summary_path, row)

    log.info("Finished %s %s in %.2f minutes", layout.subject, layout.session, total_sec / 60.0)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SimCortex end-to-end MRI-only inference.")
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(x: argparse.ArgumentParser) -> None:
        x.add_argument("--out-root", required=True, type=Path)
        x.add_argument("--work-root", default=None, type=Path)
        x.add_argument("--project-root", required=True, type=Path)
        x.add_argument("--mni", required=True, type=Path)
        x.add_argument("--seg-ckpt", required=True, type=Path)
        x.add_argument("--deform-ckpt", required=True, type=Path)
        x.add_argument("--device", default="cuda:0")
        x.add_argument("--space", default="MNI152")
        x.add_argument("--transform-type", default="Affine", choices=["Rigid", "Affine"])
        x.add_argument("--overwrite", action="store_true")
        x.add_argument("--keep-work", action="store_true")
        x.add_argument("--qc-collisions", action="store_true")
        x.add_argument("--initsurf-workers", type=int, default=1)
        x.add_argument("--export-native", action="store_true")

    s1 = sub.add_parser("single", help="Run one standalone MRI.")
    add_common(s1)
    s1.add_argument("--t1w", required=True, type=Path)
    s1.add_argument("--subject", required=True)
    s1.add_argument("--session", default="ses-01")

    s2 = sub.add_parser("bids", help="Run a BIDS-like dataset or selected subject/session items.")
    add_common(s2)
    s2.add_argument("--bids-root", required=True, type=Path)
    s2.add_argument("--participant-label", nargs="*", default=None)
    s2.add_argument("--session", nargs="*", default=None)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    out_root = args.out_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve() if args.work_root else (out_root / "work")
    setup_logging(out_root)

    project_root = args.project_root.expanduser().resolve()
    mni = args.mni.expanduser().resolve()
    seg_ckpt = args.seg_ckpt.expanduser().resolve()
    deform_ckpt = args.deform_ckpt.expanduser().resolve()

    check_required_file(project_root, "Project root")
    check_required_file(project_root / "src" / "simcortex", "SimCortex source package")
    check_required_file(mni, "MNI template")
    check_required_file(seg_ckpt, "Segmentation checkpoint")
    check_required_file(deform_ckpt, "Deformation checkpoint")

    cfg = PipelineConfig(
        out_root=out_root,
        work_root=work_root,
        project_root=project_root,
        mni=mni,
        seg_ckpt=seg_ckpt,
        deform_ckpt=deform_ckpt,
        device=str(args.device),
        space=str(args.space),
        transform_type=str(args.transform_type),
        overwrite=bool(args.overwrite),
        keep_work=bool(args.keep_work),
        qc_collisions=bool(args.qc_collisions),
        initsurf_workers=max(1, int(args.initsurf_workers)),
        export_native=bool(args.export_native),
    )

    if args.mode == "single":
        subjects = discover_single_subject(args.t1w, args.subject, args.session)
    elif args.mode == "bids":
        subjects = discover_bids_subjects(args.bids_root, args.participant_label, args.session)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    log.info("Discovered %d subject/session input(s).", len(subjects))
    for i, sub in enumerate(subjects, start=1):
        run_one_subject(sub, cfg, index=i, total=len(subjects))

    log.info("Pipeline finished successfully for all inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
