from __future__ import annotations

import os
import time
import json
import logging
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import trimesh
from simcortex.deform.data.dataloader import CSRDeformInferDataset, collate_csr_deform_infer
from simcortex.deform.utils.coords import voxel_to_world
from simcortex.deform.models.surfdeform import SurfDeform

log = logging.getLogger(__name__)

_SURF_MAP = {
    "lh_pial": ("L", "pial"),
    "lh_white": ("L", "white"),
    "rh_pial": ("R", "pial"),
    "rh_white": ("R", "white"),
}


def _ses(session_label: str) -> str:
    s = str(session_label)
    return s if s.startswith("ses-") else f"ses-{s}"


def ensure_derivative_description(out_root: str, name: str = "scpp-deform"):
    p = os.path.join(out_root, "dataset_description.json")
    if os.path.isfile(p):
        return
    os.makedirs(out_root, exist_ok=True)
    desc = {
        "Name": name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "SimCortex", "Description": "Surface deformation stage"}],
    }
    with open(p, "w") as f:
        json.dump(desc, f, indent=2)

def _get_map(cfg_node, keys: Tuple[str, ...]) -> Dict[str, str] | None:
    for k in keys:
        v = getattr(cfg_node, k, None)
        if v is not None and hasattr(v, "items"):
            return {str(kk): str(vv) for kk, vv in v.items()}
    return None


def normalize_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject" not in df.columns:
        raise ValueError(f"split_file must contain column 'subject'. Got: {list(df.columns)}")
    df["subject"] = df["subject"].astype(str)
    df["subject"] = df["subject"].apply(lambda x: x if x.startswith("sub-") else f"sub-{x}")
    return df


def _validate_single_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Single-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")

    if "dataset" in df.columns:
        vals = sorted(df["dataset"].dropna().astype(str).str.strip().unique().tolist())
        if len(vals) > 1:
            raise ValueError(
                "Single-dataset deform inference received a split_file with multiple dataset values. "
                "Please provide a split CSV for one dataset only."
            )


def _validate_multi_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Multi-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")


def _detect_deform_infer_mode(cfg):
    single_preproc_root = OmegaConf.select(cfg, "dataset.path", default=None)
    single_initsurf_root = OmegaConf.select(cfg, "dataset.initsurf_root", default=None)
    single_out_root = OmegaConf.select(cfg, "outputs.out_root", default=None)

    single_preproc_root = None if single_preproc_root in (None, "") else str(single_preproc_root)
    single_initsurf_root = None if single_initsurf_root in (None, "") else str(single_initsurf_root)
    single_out_root = None if single_out_root in (None, "") else str(single_out_root)

    roots_map = _get_map(cfg.dataset, ("roots",))
    initsurf_roots_map = _get_map(cfg.dataset, ("initsurf_roots",))
    out_roots_map = _get_map(cfg.outputs, ("out_roots",))

    if single_preproc_root is not None:
        log.info("Deform inference mode: SINGLE-DATASET")
        log.info(f"dataset.path = {single_preproc_root}")
        log.info(f"dataset.initsurf_root = {single_initsurf_root}")
        log.info(f"outputs.out_root = {single_out_root}")

        if single_initsurf_root is None:
            raise ValueError("Single-dataset deform inference requires dataset.initsurf_root")
        if single_out_root is None:
            raise ValueError("Single-dataset deform inference requires outputs.out_root")

        if roots_map is not None:
            log.warning(
                "Both dataset.path and dataset.roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.roots."
            )
        if initsurf_roots_map is not None:
            log.warning(
                "Both dataset.initsurf_root and dataset.initsurf_roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.initsurf_roots."
            )
        if out_roots_map is not None:
            log.warning(
                "Both outputs.out_root and outputs.out_roots are present. "
                "Using SINGLE-DATASET mode and ignoring outputs.out_roots."
            )

        return "single", single_preproc_root, single_initsurf_root, single_out_root, None, None, None

    if roots_map is not None:
        log.info("Deform inference mode: MULTI-DATASET")
        log.info(f"dataset.roots keys = {list(roots_map.keys())}")

        if initsurf_roots_map is None:
            raise ValueError("Multi-dataset deform inference requires dataset.initsurf_roots")
        if out_roots_map is None:
            raise ValueError("Multi-dataset deform inference requires outputs.out_roots")

        missing_init = sorted(set(roots_map.keys()) - set(initsurf_roots_map.keys()))
        if missing_init:
            raise KeyError(f"dataset.initsurf_roots missing keys required by dataset.roots: {missing_init}")

        missing_out = sorted(set(roots_map.keys()) - set(out_roots_map.keys()))
        if missing_out:
            raise KeyError(f"outputs.out_roots missing keys required by dataset.roots: {missing_out}")

        return "multi", None, None, None, roots_map, initsurf_roots_map, out_roots_map

    raise ValueError(
        "Could not determine deform inference mode. Provide either:\n"
        "  - dataset.path + dataset.initsurf_root + outputs.out_root   (single-dataset)\n"
        "or\n"
        "  - dataset.roots + dataset.initsurf_roots + outputs.out_roots (multi-dataset)"
    )

def load_checkpoint(model: torch.nn.Module, ckpt_path: str, strict: bool = True):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
        sd = sd.get("state_dict", sd.get("model", sd))

    # strip DDP prefix if any
    sd = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }

    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(sd, strict=strict)
    log.info("Loaded checkpoint: %s (strict=%s)", ckpt_path, strict)


def build_unified_init(batch: Dict, device: torch.device, surface_names: List[str]):
    B = len(batch["subject"])

    unified_list = []
    per_counts = []
    faces_per_subj = []

    affines = batch["affine"].to(device)     # [B,4,4]
    shifts  = batch["shift_ijk"].to(device)  # [B,3]

    for i in range(B):
        verts_cat = []
        counts_i = []
        faces_i = []

        for s in surface_names:
            v = batch["init_verts_vox"][i][s].to(device)      # [Ni,3] voxel in cropped/padded space
            f = batch["init_faces"][i][s].to(device).long()   # [Fi,3]
            verts_cat.append(v)
            counts_i.append(int(v.shape[0]))
            faces_i.append(f.detach().cpu().numpy().astype(np.int64))

        merged = torch.cat(verts_cat, dim=0)
        unified_list.append(merged)
        per_counts.append(counts_i)
        faces_per_subj.append(faces_i)

    lengths = torch.tensor([v.shape[0] for v in unified_list], device=device, dtype=torch.long)
    padded = pad_sequence(unified_list, batch_first=True).to(device)  # [B,Nmax,3]
    return padded, lengths, per_counts, faces_per_subj, affines, shifts


def out_surface_path(out_root: str, subj: str, session_label: str, space: str, surf_name: str) -> str:
    ses = _ses(session_label)
    hemi, surf = _SURF_MAP[surf_name]
    return os.path.join(
        out_root, subj, ses, "surfaces",
        f"{subj}_{ses}_space-{space}_desc-deform_hemi-{hemi}_{surf}.surf.ply"
    )


@hydra.main(version_base=None, config_path="pkg://simcortex.configs.deform", config_name="inference")
def main(cfg: DictConfig):

    if cfg.user_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.user_config))

    level = getattr(logging, str(getattr(cfg.inference, "log_level", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    os.makedirs(str(cfg.outputs.log_dir), exist_ok=True)
    log_file = os.path.join(str(cfg.outputs.log_dir), "inference.log")

    root_logger = logging.getLogger()
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root_logger.addHandler(fh)

    log.info("Logging to %s", log_file)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))


    surface_names = list(cfg.dataset.surface_name)

    unknown_surfs = [s for s in surface_names if s not in _SURF_MAP]
    if unknown_surfs:
        raise KeyError(f"Unknown surface names: {unknown_surfs}. Supported: {list(_SURF_MAP.keys())}")

    # add_prob_grad forced if c_in==3
    add_prob_grad = bool(getattr(cfg.dataset, "add_prob_grad", False))
    if int(cfg.model.c_in) == 3:
        add_prob_grad = True

    device_str = str(getattr(cfg.inference, "device", "cuda:0"))
    device = torch.device(device_str if (("cuda" not in device_str) or torch.cuda.is_available()) else "cpu")

    split_file = str(cfg.dataset.split_file)
    split_name = str(cfg.dataset.split_name)
    session_label = str(getattr(cfg.dataset, "session_label", "01"))
    space = str(getattr(cfg.dataset, "space", "MNI152"))

    mode, single_preproc_root, single_initsurf_root, single_out_root, roots_map, initsurf_roots_map, out_roots_map = _detect_deform_infer_mode(cfg)

    df = pd.read_csv(split_file)
    df = normalize_subject_column(df)
    if split_name != "all":
        df = df[df["split"].astype(str).str.strip() == split_name].copy()
    else:
        df = df.copy()
    if len(df) == 0:
        raise RuntimeError(f"No subjects found for split_name='{split_name}' in {split_file}")
    if mode == "multi":
        df = df[["dataset", "subject", "split"]].drop_duplicates(["dataset", "subject"]).reset_index(drop=True)
    else:
        df = df[["subject", "split"]].drop_duplicates(["subject"]).reset_index(drop=True)

    # model
    model = SurfDeform(
        C_hid=cfg.model.c_hid,
        C_in=int(cfg.model.c_in),
        inshape=list(cfg.model.inshape),
        sigma=float(cfg.model.sigma),
        device=device,
        geom_ratio=float(getattr(cfg.model, "geom_ratio", 0.5)),
        geom_depth=int(getattr(cfg.model, "geom_depth", 6)),
        gn_groups=int(getattr(cfg.model, "gn_groups", 8)),
        gate_init=float(getattr(cfg.model, "gate_init", -3.0)),
    ).to(device)

    load_checkpoint(model, str(cfg.model.ckpt_path), strict=bool(getattr(cfg.model, "strict_load", True)))
    model.eval()

    overwrite = bool(getattr(cfg.inference, "overwrite", False))
    bs = int(getattr(cfg.inference, "batch_size", 1))
    nw = int(getattr(cfg.inference, "num_workers", 2))

    times = []

    # ---------------- SINGLE ----------------
    if mode == "single":
        _validate_single_split_df(df)

        ensure_derivative_description(str(single_out_root))
        subjects = df["subject"].astype(str).tolist()


        ds = CSRDeformInferDataset(
            preproc_root=str(single_preproc_root),
            initsurf_root=str(single_initsurf_root),
            subjects=subjects,
            session_label=session_label,
            space=space,
            surface_names=surface_names,
            inshape_dhw=list(cfg.model.inshape),
            prob_clip_min=float(cfg.dataset.prob_clip_min),
            prob_clip_max=float(cfg.dataset.prob_clip_max),
            prob_gamma=float(cfg.dataset.prob_gamma),
            add_prob_grad=add_prob_grad,
        )

        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            collate_fn=collate_csr_deform_infer,
        )

        log.info("[SINGLE] subjects=%d | out_root=%s", len(ds), single_out_root)

        with torch.inference_mode ():
            for batch in tqdm(loader, desc="Infer SINGLE", leave=False):
                vol = batch["vol"].to(device)
                B = vol.shape[0]

                padded_init, lengths, per_counts, faces_per_subj, affines, shifts = build_unified_init(
                    batch, device, surface_names
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()

                pred_all = model(padded_init, vol, int(cfg.model.n_steps))

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                times.extend([(t1 - t0) / max(B, 1)] * B)

                for i in range(B):
                    subj = str(batch["subject"][i])
                    A = affines[i]
                    sh = shifts[i]

                    pred_unified = pred_all[i, : int(lengths[i].item())]
                    splits = torch.split(pred_unified, per_counts[i], dim=0)

                    for j, surf in enumerate(surface_names):
                        out_path = out_surface_path(str(single_out_root), subj, session_label, space, surf)
                        if (not overwrite) and os.path.isfile(out_path):
                            continue

                        os.makedirs(os.path.dirname(out_path), exist_ok=True)

                        v_vox_cp = splits[j]
                        v_vox_orig = v_vox_cp - sh
                        v_mm = voxel_to_world(v_vox_orig, A).detach().cpu().numpy().astype(np.float32)

                        f = faces_per_subj[i][j]
                        trimesh.Trimesh(vertices=v_mm, faces=f, process=False).export(out_path)

    # ---------------- MULTI ----------------
    else:
        _validate_multi_split_df(df)
        df["dataset"] = df["dataset"].astype(str).str.strip()

        for ds_key, ds_df in df.groupby("dataset"):
            if ds_key not in roots_map or ds_key not in initsurf_roots_map or ds_key not in out_roots_map:
                raise KeyError(f"Missing dataset key in config: {ds_key}")

            preproc_root = str(roots_map[ds_key])
            initsurf_root = str(initsurf_roots_map[ds_key])
            out_root = str(out_roots_map[ds_key])

            ensure_derivative_description(out_root)
            subjects = ds_df["subject"].astype(str).tolist()

            ds = CSRDeformInferDataset(
                preproc_root=preproc_root,
                initsurf_root=initsurf_root,
                subjects=subjects,
                session_label=session_label,
                space=space,
                surface_names=surface_names,
                inshape_dhw=list(cfg.model.inshape),
                prob_clip_min=float(cfg.dataset.prob_clip_min),
                prob_clip_max=float(cfg.dataset.prob_clip_max),
                prob_gamma=float(cfg.dataset.prob_gamma),
                add_prob_grad=add_prob_grad,
            )

            loader = DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
                collate_fn=collate_csr_deform_infer,
            )

            log.info("[%s] subjects=%d | out_root=%s", ds_key, len(ds), out_root)

            with torch.inference_mode ():
                for batch in tqdm(loader, desc=f"Infer {ds_key}", leave=False):
                    vol = batch["vol"].to(device)
                    B = vol.shape[0]

                    padded_init, lengths, per_counts, faces_per_subj, affines, shifts = build_unified_init(
                        batch, device, surface_names
                    )

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()

                    pred_all = model(padded_init, vol, int(cfg.model.n_steps))

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.time()
                    times.extend([(t1 - t0) / max(B, 1)] * B)

                    for i in range(B):
                        subj = str(batch["subject"][i])
                        A = affines[i]
                        sh = shifts[i]

                        pred_unified = pred_all[i, : int(lengths[i].item())]
                        splits = torch.split(pred_unified, per_counts[i], dim=0)

                        for j, surf in enumerate(surface_names):
                            out_path = out_surface_path(out_root, subj, session_label, space, surf)
                            if (not overwrite) and os.path.isfile(out_path):
                                continue

                            os.makedirs(os.path.dirname(out_path), exist_ok=True)

                            v_vox_cp = splits[j]
                            v_vox_orig = v_vox_cp - sh
                            v_mm = voxel_to_world(v_vox_orig, A).detach().cpu().numpy().astype(np.float32)

                            f = faces_per_subj[i][j]
                            trimesh.Trimesh(vertices=v_mm, faces=f, process=False).export(out_path)

    if times:
        log.info("Avg inference time/subject: %.4fs", float(sum(times) / len(times)))
    log.info("Done.")


if __name__ == "__main__":
    main()