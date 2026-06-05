#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from simcortex.seg.data.dataloader import PredictSegDataset
from simcortex.seg.models.unet import Unet


# -----------------------------------------------------------------------------
# Logging / path helpers
# -----------------------------------------------------------------------------
def _to_abs_path(path_like: Any) -> Path:
    """Resolve config paths robustly under Hydra while preserving absolute paths."""
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return p
    return Path(to_absolute_path(str(p)))


def setup_logger(log_dir: str | Path, filename: str = "inference.log") -> None:
    log_dir = _to_abs_path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / filename
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

    root = logging.getLogger("")
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(str(log_file))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    root.addHandler(sh)


def _norm_ses(s: Any) -> str:
    s = str(s)
    return s if s.startswith("ses-") else f"ses-{s}"


def _norm_sub(s: Any) -> str:
    s = str(s)
    return s if s.startswith("sub-") else f"sub-{s}"


# -----------------------------------------------------------------------------
# Checkpoint / model helpers
# -----------------------------------------------------------------------------
def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load checkpoints saved with DataParallel/DDP that may contain 'module.' prefixes."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Support all checkpoint formats used in SimCortex:
      1. raw model state_dict
      2. {"state_dict": ...}
      3. corrected full checkpoint {"model": ..., "optimizer": ..., ...}
      4. common fallback {"model_state_dict": ...}
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict/state_dict, got {type(checkpoint)!r}")

    for key in ("model", "state_dict", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            return _strip_module_prefix(state)

    # Raw state_dict: all values should be tensors or tensor-like state entries.
    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        return _strip_module_prefix(checkpoint)  # type: ignore[arg-type]

    raise ValueError("Could not find model weights in checkpoint.")


def _checkpoint_cfg(checkpoint: Any) -> Optional[DictConfig]:
    if isinstance(checkpoint, dict) and "cfg" in checkpoint:
        try:
            return OmegaConf.create(checkpoint["cfg"])
        except Exception:
            return None
    return None


def build_model(cfg: DictConfig, checkpoint_cfg: Optional[DictConfig] = None) -> Unet:
    """
    Build a U-Net using current inference config, with optional fallback to model
    settings stored in a full training checkpoint.

    Inference must use the exact same architecture as training. This matters after
    switching from the old single-conv U-Net to the corrected normalized double-conv U-Net.
    """
    num_classes = int(OmegaConf.select(cfg, "model.out_channels", default=9))
    kwargs: dict[str, Any] = {
        "c_in": int(OmegaConf.select(cfg, "model.in_channels", default=1)),
        "c_out": num_classes,
    }

    sig = inspect.signature(Unet.__init__)
    optional_names = ("features", "norm", "negative_slope", "dropout")

    for name in optional_names:
        value = OmegaConf.select(cfg, f"model.{name}", default=None)
        if value is None and checkpoint_cfg is not None:
            value = OmegaConf.select(checkpoint_cfg, f"model.{name}", default=None)
        if value is not None and name in sig.parameters:
            # Convert OmegaConf containers to plain Python containers for constructors.
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            kwargs[name] = value

    return Unet(**kwargs)


def load_model_from_checkpoint(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    ckpt_path_raw = OmegaConf.select(cfg, "model.ckpt_path", default=None)
    if ckpt_path_raw in (None, ""):
        raise ValueError("model.ckpt_path must be provided for segmentation inference.")

    ckpt_path = _to_abs_path(ckpt_path_raw)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info("Loading checkpoint: %s", ckpt_path)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    checkpoint_cfg = _checkpoint_cfg(checkpoint)

    model = build_model(cfg, checkpoint_cfg=checkpoint_cfg)
    state = _extract_state_dict(checkpoint)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load segmentation checkpoint strictly. This usually means the "
            "inference config model settings do not match the training architecture. "
            "Check model.features, model.norm, model.dropout, and model.out_channels. "
            f"Checkpoint: {ckpt_path}"
        ) from exc

    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# BIDS derivative helpers
# -----------------------------------------------------------------------------
def _get_pkg_version(pkg_name: str) -> str:
    try:
        import importlib.metadata as importlib_metadata

        return importlib_metadata.version(pkg_name)
    except Exception:
        return "0.0.0"


def _write_dataset_description(deriv_root: Path, name: str, version: str, overwrite: bool = False) -> None:
    """Create/update dataset_description.json for the segmentation derivative."""
    deriv_root.mkdir(parents=True, exist_ok=True)
    p = deriv_root / "dataset_description.json"
    if p.exists() and not overwrite:
        return

    desc = {
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "SimCortex",
                "Version": version,
                "Description": "3D U-Net 9-class segmentation inference in MNI152 space.",
            }
        ],
        "GeneratedOn": str(date.today()),
    }
    p.write_text(json.dumps(desc, indent=2) + "\n", encoding="utf-8")


def _seg_out_path(out_root: Path, sub: str, ses: str, space: str) -> Path:
    sub = _norm_sub(sub)
    ses = _norm_ses(ses)
    stem = f"{sub}_{ses}"
    return out_root / sub / ses / "anat" / f"{stem}_space-{space}_desc-seg9_dseg.nii.gz"


def save_segmentation_nifti(pred: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Labels are 0..8, so uint8 is enough and compact. If future labels exceed
    # this range, change this to int16.
    pred = np.asarray(pred, dtype=np.uint8)
    affine = np.asarray(affine, dtype=np.float64)

    img = nib.Nifti1Image(pred, affine)
    img.set_data_dtype(np.uint8)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(out_path))


# -----------------------------------------------------------------------------
# Dataset routing / multi-dataset support
# -----------------------------------------------------------------------------
def _get_roots_map(ds_cfg: Any) -> Optional[Dict[str, str]]:
    for key in ("roots", "dataset_roots", "deriv_roots"):
        val = getattr(ds_cfg, key, None)
        if val is not None and hasattr(val, "items"):
            out = {str(k): str(_to_abs_path(v)) for k, v in val.items() if v not in (None, "")}
            return out if out else None
    return None


def _get_out_roots_map(outputs_cfg: Any) -> Optional[Dict[str, str]]:
    val = getattr(outputs_cfg, "out_roots", None)
    if val is not None and hasattr(val, "items"):
        out = {str(k): str(_to_abs_path(v)) for k, v in val.items() if v not in (None, "")}
        return out if out else None
    return None


def _detect_inference_mode(cfg: DictConfig) -> tuple[str, Optional[str], Optional[Dict[str, str]]]:
    """
    Decide between:
      - single mode: dataset.path + outputs.out_root
      - multi mode:  dataset.roots + outputs.out_roots

    If dataset.path is provided, single mode wins to avoid accidental multi-dataset runs.
    """
    single_path_raw = OmegaConf.select(cfg, "dataset.path", default=None)
    single_out_raw = OmegaConf.select(cfg, "outputs.out_root", default=None)
    split_file = OmegaConf.select(cfg, "dataset.split_file", default=None)

    single_path = None if single_path_raw in (None, "") else str(_to_abs_path(single_path_raw))
    single_out = None if single_out_raw in (None, "") else str(_to_abs_path(single_out_raw))

    if split_file in (None, ""):
        raise ValueError("dataset.split_file must be provided.")

    roots_map = _get_roots_map(cfg.dataset)
    out_roots_map = _get_out_roots_map(cfg.outputs)

    if single_path is not None:
        if single_out is None:
            raise ValueError("Single-dataset inference requires outputs.out_root.")
        if roots_map is not None:
            logging.warning("dataset.path is set; ignoring dataset.roots.")
        if out_roots_map is not None:
            logging.warning("outputs.out_root is set; ignoring outputs.out_roots.")
        logging.info("Inference mode: SINGLE-DATASET")
        logging.info("dataset.path = %s", single_path)
        logging.info("outputs.out_root = %s", single_out)
        return "single", single_path, None

    if roots_map is not None:
        if out_roots_map is None:
            raise ValueError("Multi-dataset inference requires outputs.out_roots.")
        missing = sorted(set(roots_map.keys()) - set(out_roots_map.keys()))
        if missing:
            raise KeyError(f"outputs.out_roots missing keys required by dataset.roots: {missing}")
        extra = sorted(set(out_roots_map.keys()) - set(roots_map.keys()))
        if extra:
            logging.warning("outputs.out_roots has extra keys not present in dataset.roots: %s", extra)
        logging.info("Inference mode: MULTI-DATASET")
        logging.info("dataset.roots keys = %s", list(roots_map.keys()))
        logging.info("outputs.out_roots keys = %s", list(out_roots_map.keys()))
        return "multi", None, roots_map

    raise ValueError(
        "Could not determine inference mode. Provide either dataset.path + outputs.out_root "
        "or dataset.roots + outputs.out_roots."
    )


def _cache_per_dataset_csvs(split_csv: str | Path, cache_dir: Path, roots: Dict[str, str]) -> Dict[str, str]:
    """
    Given a combined CSV with columns [subject, split, dataset], write per-dataset
    CSVs with columns [subject, split]. Stale files for configured roots are removed.
    """
    split_csv = _to_abs_path(split_csv)
    cache_dir = _to_abs_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")

    out_map: Dict[str, str] = {}
    for ds_name in roots.keys():
        out = cache_dir / f"split_{ds_name}.csv"
        if out.exists():
            out.unlink()

        df_ds = df[df["dataset"].astype(str).str.strip() == ds_name][["subject", "split"]]
        if df_ds.empty:
            logging.warning("No rows for dataset='%s' in %s", ds_name, split_csv)
            continue

        df_ds.to_csv(out, index=False)
        out_map[ds_name] = str(out)

    if not out_map:
        raise RuntimeError(f"No per-dataset split files created in: {cache_dir}")

    return out_map


class _TagDataset(Dataset):
    """Attach a dataset name to each sample so outputs can be routed per dataset."""

    def __init__(self, base: Dataset, ds_name: str) -> None:
        self.base = base
        self.ds_name = ds_name

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        vol, sub, ses, affine, orig_shape = self.base[idx]
        return vol, sub, ses, affine, orig_shape, self.ds_name


def _resolve_out_root(cfg: DictConfig, ds_name: Optional[str]) -> Path:
    if ds_name is not None:
        out_roots = _get_out_roots_map(cfg.outputs)
        if out_roots is None or ds_name not in out_roots:
            raise KeyError(f"outputs.out_roots missing key '{ds_name}'")
        return Path(out_roots[ds_name])

    out_root = OmegaConf.select(cfg, "outputs.out_root", default=None)
    if out_root in (None, ""):
        raise ValueError("Single-dataset inference requires outputs.out_root.")
    return _to_abs_path(out_root)


def build_dataset(cfg: DictConfig, single_path: Optional[str], roots_map: Optional[Dict[str, str]]) -> Dataset:
    split_name = str(OmegaConf.select(cfg, "dataset.split_name", default="test"))
    session_label = str(OmegaConf.select(cfg, "dataset.session_label", default="01"))
    space = str(OmegaConf.select(cfg, "dataset.space", default="MNI152"))
    pad_mult = int(OmegaConf.select(cfg, "dataset.pad_mult", default=16))
    split_file = _to_abs_path(OmegaConf.select(cfg, "dataset.split_file"))

    if roots_map is not None:
        cache_dir = _to_abs_path(OmegaConf.select(cfg, "outputs.log_dir")) / "split_cache"
        per_ds_csv = _cache_per_dataset_csvs(split_file, cache_dir, roots_map)

        dsets: list[Dataset] = []
        for ds_name, root in roots_map.items():
            if ds_name not in per_ds_csv:
                continue
            base = PredictSegDataset(
                deriv_root=str(root),
                split_csv=str(per_ds_csv[ds_name]),
                split_name=split_name,
                session_label=session_label,
                space=space,
                pad_mult=pad_mult,
            )
            dsets.append(_TagDataset(base, ds_name))

        if not dsets:
            raise RuntimeError("No datasets constructed. Check dataset.roots and split_file dataset values.")
        return dsets[0] if len(dsets) == 1 else ConcatDataset(dsets)

    if single_path is None:
        raise ValueError("single_path is required in single-dataset mode.")

    return PredictSegDataset(
        deriv_root=str(single_path),
        split_csv=str(split_file),
        split_name=split_name,
        session_label=session_label,
        space=space,
        pad_mult=pad_mult,
    )


def _as_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _item_from_batch(x: Any, idx: int) -> Any:
    # Default PyTorch collation keeps strings as lists/tuples.
    if isinstance(x, (list, tuple)):
        return x[idx]
    if torch.is_tensor(x):
        return x[idx]
    arr = np.asarray(x)
    return arr[idx]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_path="pkg://simcortex.configs.seg", config_name="inference")
def main(cfg: DictConfig) -> None:
    setup_logger(OmegaConf.select(cfg, "outputs.log_dir"), "inference.log")
    logging.info("=== Segmentation inference config ===")
    logging.info("\n%s", OmegaConf.to_yaml(cfg))

    mode, single_path, roots_map = _detect_inference_mode(cfg)

    device_cfg = str(OmegaConf.select(cfg, "trainer.device", default="cuda:0"))
    device = torch.device(device_cfg if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    amp_enabled = bool(OmegaConf.select(cfg, "trainer.amp", default=(device.type == "cuda"))) and device.type == "cuda"
    overwrite = bool(OmegaConf.select(cfg, "outputs.overwrite", default=False))
    write_desc_overwrite = bool(OmegaConf.select(cfg, "outputs.overwrite_dataset_description", default=False))
    space = str(OmegaConf.select(cfg, "dataset.space", default="MNI152"))

    # Prepare BIDS derivative descriptions.
    version = _get_pkg_version("simcortex")
    if roots_map is not None:
        for ds_name in roots_map.keys():
            out_root = _resolve_out_root(cfg, ds_name)
            _write_dataset_description(
                out_root,
                name=f"SimCortex Segmentation ({ds_name})",
                version=version,
                overwrite=write_desc_overwrite,
            )
    else:
        out_root = _resolve_out_root(cfg, None)
        _write_dataset_description(
            out_root,
            name="SimCortex Segmentation",
            version=version,
            overwrite=write_desc_overwrite,
        )

    ds = build_dataset(cfg, single_path=single_path, roots_map=roots_map)

    num_workers = int(OmegaConf.select(cfg, "trainer.num_workers", default=0))
    dl = DataLoader(
        ds,
        batch_size=int(OmegaConf.select(cfg, "trainer.batch_size", default=1)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model = load_model_from_checkpoint(cfg, device)

    log_dir = _to_abs_path(OmegaConf.select(cfg, "outputs.log_dir"))
    writer = SummaryWriter(str(log_dir)) if bool(OmegaConf.select(cfg, "outputs.tensorboard", default=True)) else None

    manifest_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    processed = 0
    saved = 0
    skipped = 0

    logging.info("Dataset samples: %d | batches: %d", len(ds), len(dl))
    logging.info("Device: %s | AMP: %s | overwrite: %s", device, amp_enabled, overwrite)

    with torch.inference_mode():
        pbar = tqdm(dl, desc="Inferring", total=len(dl))
        for step, batch in enumerate(pbar):
            if roots_map is not None:
                vol, sub, ses, affine, orig_shape, ds_name = batch
            else:
                vol, sub, ses, affine, orig_shape = batch
                ds_name = [None] * int(vol.shape[0])

            vol = vol.to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(vol)
                pred = logits.argmax(dim=1).detach().cpu().numpy()

            shapes = _as_numpy(orig_shape)
            affines = _as_numpy(affine)

            for b in range(pred.shape[0]):
                ds_b = _item_from_batch(ds_name, b)
                ds_b = None if ds_b in (None, "None", "") else str(ds_b)

                sid = _norm_sub(_item_from_batch(sub, b))
                ses_b = _norm_ses(_item_from_batch(ses, b))

                try:
                    shape_b = tuple(int(v) for v in np.asarray(shapes[b]).tolist())
                    if len(shape_b) != 3:
                        raise ValueError(f"Expected orig_shape length 3, got {shape_b}")

                    if pred[b].shape[0] < shape_b[0] or pred[b].shape[1] < shape_b[1] or pred[b].shape[2] < shape_b[2]:
                        raise ValueError(
                            f"Prediction shape {pred[b].shape} is smaller than original shape {shape_b}."
                        )

                    pred_b = pred[b, : shape_b[0], : shape_b[1], : shape_b[2]]
                    bad = np.setdiff1d(np.unique(pred_b), np.arange(int(cfg.model.out_channels)))
                    if bad.size > 0:
                        raise ValueError(f"Predicted labels outside expected range: {bad.tolist()}")

                    out_root = _resolve_out_root(cfg, ds_b)
                    out_path = _seg_out_path(out_root, sid, ses_b, space)

                    if out_path.exists() and not overwrite:
                        status = "skipped_existing"
                        skipped += 1
                    else:
                        save_segmentation_nifti(pred_b, affines[b], out_path)
                        status = "saved"
                        saved += 1
                        logging.info("[%s] Saved: %s", ds_b or "SINGLE", out_path)

                    processed += 1
                    manifest_rows.append(
                        {
                            "dataset": ds_b or "",
                            "subject": sid,
                            "session": ses_b,
                            "status": status,
                            "output": str(out_path),
                            "shape_D": shape_b[0],
                            "shape_H": shape_b[1],
                            "shape_W": shape_b[2],
                        }
                    )

                except Exception as exc:
                    failed_rows.append(
                        {
                            "dataset": ds_b or "",
                            "subject": sid,
                            "session": ses_b,
                            "error": repr(exc),
                        }
                    )
                    logging.error("[%s] FAILED %s/%s: %s", ds_b or "SINGLE", sid, ses_b, exc)

            if writer is not None:
                writer.add_scalar("inference/processed_subjects", processed, step)
                writer.add_scalar("inference/saved_subjects", saved, step)
                writer.add_scalar("inference/skipped_subjects", skipped, step)

            pbar.set_postfix(saved=saved, skipped=skipped, failed=len(failed_rows))

    if writer is not None:
        writer.close()

    manifest_path = log_dir / "seg_inference_manifest.tsv"
    failed_path = log_dir / "seg_inference_failed.tsv"

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
        logging.info("Manifest written: %s", manifest_path)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_path, sep="\t", index=False)
        logging.error("Inference finished with %d failed sample(s). See %s", len(failed_rows), failed_path)
        raise SystemExit(1)

    logging.info("Inference finished. processed=%d saved=%d skipped=%d", processed, saved, skipped)


if __name__ == "__main__":
    main()
