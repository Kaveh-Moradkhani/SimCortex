from __future__ import annotations

import inspect
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from simcortex.seg.data.dataloader import SegDataset
from simcortex.seg.models.unet import Unet


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def _state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return an unwrappped state_dict from plain, DataParallel, or DDP models."""
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _to_abs_str(path_like: Any) -> str:
    """Resolve config paths robustly under Hydra while preserving absolute paths."""
    p = Path(str(path_like)).expanduser()
    if p.is_absolute():
        return str(p)
    return to_absolute_path(str(p))


def _get_bool(node: Any, key: str, default: bool = False) -> bool:
    val = getattr(node, key, default)
    return bool(val)


def _main_process(rank: int) -> bool:
    return rank == 0


# -----------------------------------------------------------------------------
# Metrics / losses
# -----------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Soft multiclass Dice loss for logits.

    Notes
    -----
    - Background can be excluded from the Dice term.
    - By default, classes absent in the target for a batch are ignored in the Dice
      average. Cross entropy still penalizes false positives for those classes.
    """

    def __init__(
        self,
        num_classes: int,
        exclude_bg: bool = True,
        eps: float = 1e-6,
        ignore_absent_target_classes: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = int(num_classes)
        self.exclude_bg = bool(exclude_bg)
        self.eps = float(eps)
        self.ignore_absent_target_classes = bool(ignore_absent_target_classes)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5:
            raise ValueError(f"logits must be [B,C,D,H,W], got shape {tuple(logits.shape)}")
        if y.ndim != 4:
            raise ValueError(f"target must be [B,D,H,W], got shape {tuple(y.shape)}")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits channel count {logits.shape[1]} does not match num_classes={self.num_classes}"
            )

        p = F.softmax(logits, dim=1)
        y_1h = F.one_hot(y.long(), self.num_classes).permute(0, 4, 1, 2, 3).to(dtype=p.dtype)

        if self.exclude_bg:
            p = p[:, 1:]
            y_1h = y_1h[:, 1:]

        p_f = p.flatten(2)
        y_f = y_1h.flatten(2)

        inter = (p_f * y_f).sum(dim=-1)  # [B, C']
        union = p_f.sum(dim=-1) + y_f.sum(dim=-1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        loss = 1.0 - dice

        if self.ignore_absent_target_classes:
            present = y_f.sum(dim=-1) > 0
            if present.any():
                return loss[present].mean()
            return loss.new_tensor(0.0)

        return loss.mean()


def _new_metric_state(num_classes: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "loss_sum": torch.zeros((), dtype=torch.float64, device=device),
        "sample_count": torch.zeros((), dtype=torch.float64, device=device),
        "correct_vox": torch.zeros((), dtype=torch.float64, device=device),
        "total_vox": torch.zeros((), dtype=torch.float64, device=device),
        "inter": torch.zeros(num_classes, dtype=torch.float64, device=device),
        "union": torch.zeros(num_classes, dtype=torch.float64, device=device),
    }


@torch.no_grad()
def _update_metric_state(
    state: dict[str, torch.Tensor],
    *,
    logits: torch.Tensor,
    y: torch.Tensor,
    loss: torch.Tensor,
    num_classes: int,
) -> None:
    """Accumulate exact hard-Dice components without constructing large one-hot tensors."""
    pred = logits.argmax(dim=1)
    y = y.long()
    b = int(y.shape[0])

    state["loss_sum"] += loss.detach().double() * float(b)
    state["sample_count"] += float(b)
    state["correct_vox"] += (pred == y).sum(dtype=torch.float64)
    state["total_vox"] += float(y.numel())

    for cls in range(num_classes):
        pred_c = pred == cls
        y_c = y == cls
        state["inter"][cls] += (pred_c & y_c).sum(dtype=torch.float64)
        state["union"][cls] += pred_c.sum(dtype=torch.float64) + y_c.sum(dtype=torch.float64)


def _reduce_metric_state(state: dict[str, torch.Tensor], is_ddp: bool) -> None:
    if not is_ddp:
        return
    for value in state.values():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)


def _finalize_metric_state(
    state: dict[str, torch.Tensor],
    *,
    exclude_bg: bool = True,
    eps: float = 1e-6,
) -> dict[str, float | list[float]]:
    n = max(float(state["sample_count"].item()), 1.0)
    vox = max(float(state["total_vox"].item()), 1.0)

    inter = state["inter"]
    union = state["union"]
    dice = (2.0 * inter + eps) / (union + eps)
    present = union > 0

    if exclude_bg and dice.numel() > 1:
        dice_used = dice[1:]
        present_used = present[1:]
    else:
        dice_used = dice
        present_used = present

    if present_used.any():
        mean_dice = float(dice_used[present_used].mean().item())
    else:
        mean_dice = 0.0

    per_class = [float(v) if bool(p) else float("nan") for v, p in zip(dice.detach().cpu(), present.detach().cpu())]

    return {
        "loss": float((state["loss_sum"] / n).item()),
        "dice": mean_dice,
        "pixel_acc": float((state["correct_vox"] / vox).item()),
        "per_class_dice": per_class,
        "samples": float(state["sample_count"].item()),
    }


def _format_per_class_dice(per_class: Sequence[float]) -> str:
    vals = []
    for i, v in enumerate(per_class):
        vals.append(f"c{i}=nan" if np.isnan(v) else f"c{i}={v:.4f}")
    return ", ".join(vals)


# -----------------------------------------------------------------------------
# DDP
# -----------------------------------------------------------------------------
def setup_ddp(cfg: DictConfig) -> Tuple[int, int, bool, int]:
    """
    Set up DDP safely.

    Important safety rule:
    If torchrun environment variables are present but trainer.use_ddp=false, fail fast.
    Otherwise every torchrun process may think it is rank 0 and overwrite logs/checkpoints.
    """
    use_ddp = bool(OmegaConf.select(cfg, "trainer.use_ddp", default=False))
    launched_with_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if launched_with_torchrun and not use_ddp:
        raise RuntimeError(
            "torchrun environment detected, but trainer.use_ddp=false. "
            "Set trainer.use_ddp=true for DDP, or run with plain python."
        )

    if use_ddp and launched_with_torchrun:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return rank, world, True, local_rank

    return 0, 1, False, 0


def cleanup_ddp(is_ddp: bool) -> None:
    if is_ddp and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class DistributedEvalSampler(Sampler[int]):
    """Non-padding distributed sampler for validation/evaluation.

    PyTorch's DistributedSampler pads by default when len(dataset) is not divisible
    by world size. That duplicates validation cases and slightly biases metrics.
    This sampler partitions indices by rank without duplication.
    """

    def __init__(self, dataset: Dataset[Any], num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        n = len(self.dataset)
        if n <= self.rank:
            return 0
        return (n - 1 - self.rank) // self.num_replicas + 1


# -----------------------------------------------------------------------------
# Logging / reproducibility
# -----------------------------------------------------------------------------
def setup_logging(log_dir: str | Path, rank: int) -> None:
    log_dir = Path(log_dir)
    if _main_process(rank):
        log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    root = logging.getLogger("")
    root.handlers.clear()
    root.setLevel(logging.INFO if _main_process(rank) else logging.WARNING)

    if _main_process(rank):
        fh = logging.FileHandler(str(log_dir / "train_seg.log"), mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    else:
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------
def _get_roots_map(ds_cfg: Any) -> Optional[Dict[str, str]]:
    for key in ("roots", "dataset_roots", "deriv_roots"):
        val = getattr(ds_cfg, key, None)
        if val is not None and hasattr(val, "items"):
            return {str(k): _to_abs_str(v) for k, v in val.items()}
    return None


def _validate_single_split_csv(split_csv: str) -> None:
    df = pd.read_csv(split_csv)
    req = {"subject", "split"}
    if not req.issubset(set(df.columns)):
        raise ValueError(
            f"Single-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}"
        )

    if "dataset" in df.columns:
        vals = sorted(df["dataset"].dropna().astype(str).str.strip().unique().tolist())
        if len(vals) > 1:
            raise ValueError(
                "Single-dataset mode received a split_file with multiple dataset values. "
                "Please provide a split CSV for one dataset only, or use dataset.roots."
            )


def _cache_per_dataset_csvs(
    split_csv: str,
    cache_dir: Path,
    roots: Dict[str, str],
    rank: int,
    is_ddp: bool,
) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(
            f"Multi-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}"
        )

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()

    if _main_process(rank):
        for ds_name in roots.keys():
            p = cache_dir / f"split_{ds_name}.csv"
            if p.exists():
                p.unlink()

    if is_ddp:
        dist.barrier()  # make sure stale files are gone before any rank checks the cache

    if _main_process(rank):
        for ds_name in roots.keys():
            out = cache_dir / f"split_{ds_name}.csv"
            df_ds = df[df["dataset"] == str(ds_name).strip()][["subject", "split"]]
            if df_ds.empty:
                logging.warning("No rows for dataset='%s' in %s", ds_name, split_csv)
                continue
            df_ds.to_csv(out, index=False)

    if is_ddp:
        dist.barrier()  # make sure rank 0 has finished writing before other ranks read

    out_map: Dict[str, str] = {}
    for ds_name in roots.keys():
        p = cache_dir / f"split_{ds_name}.csv"
        if p.exists():
            out_map[ds_name] = str(p)

    if not out_map:
        raise RuntimeError(f"No cached per-dataset split files found in {cache_dir}")

    return out_map


def _detect_train_dataset_mode(cfg: DictConfig) -> tuple[str, Optional[str], Optional[Dict[str, str]]]:
    single_path = OmegaConf.select(cfg, "dataset.path", default=None)
    split_file = OmegaConf.select(cfg, "dataset.split_file", default=None)

    single_path = None if single_path in (None, "") else _to_abs_str(single_path)
    split_file = None if split_file in (None, "") else _to_abs_str(split_file)

    roots_map = _get_roots_map(cfg.dataset)

    # Store resolved split_file back into cfg for downstream calls.
    if split_file is not None:
        OmegaConf.update(cfg, "dataset.split_file", split_file, merge=False)

    if single_path is not None:
        logging.info("Segmentation training mode: SINGLE-DATASET")
        logging.info("dataset.path = %s", single_path)
        logging.info("dataset.split_file = %s", split_file)

        if split_file is None:
            raise ValueError("Single-dataset training requires dataset.split_file")

        if roots_map is not None:
            logging.warning(
                "Both dataset.path and dataset.roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.roots."
            )

        return "single", single_path, None

    if roots_map is not None:
        logging.info("Segmentation training mode: MULTI-DATASET")
        logging.info("dataset.roots keys = %s", list(roots_map.keys()))
        logging.info("dataset.split_file = %s", split_file)

        if split_file is None:
            raise ValueError("Multi-dataset training requires dataset.split_file")

        return "multi", None, roots_map

    raise ValueError(
        "Could not determine segmentation training mode. Provide either:\n"
        "  - dataset.path + dataset.split_file   (single-dataset)\n"
        "or\n"
        "  - dataset.roots + dataset.split_file  (multi-dataset)"
    )


def build_dataset(
    cfg: DictConfig,
    split: str,
    rank: int,
    is_ddp: bool,
    mode: str,
    single_path: Optional[str],
    roots_map: Optional[Dict[str, str]],
) -> Dataset[Any]:
    ds_cfg = cfg.dataset
    pad_mult = int(getattr(ds_cfg, "pad_mult", 16))
    session_label = str(getattr(ds_cfg, "session_label", "01"))
    space = str(getattr(ds_cfg, "space", "MNI152"))
    train_split = str(getattr(ds_cfg, "train_split", "train")).strip().lower()
    augment = bool(getattr(ds_cfg, "augment", False)) and split.strip().lower() == train_split
    split_file = _to_abs_str(cfg.dataset.split_file)

    if mode == "multi":
        if roots_map is None:
            raise ValueError("Multi-dataset mode requires dataset.roots")

        cache_dir = Path(_to_abs_str(cfg.outputs.log_dir)) / "split_cache"
        per_ds_csv = _cache_per_dataset_csvs(split_file, cache_dir, roots_map, rank, is_ddp)

        dsets: list[Dataset[Any]] = []
        for ds_name, root in roots_map.items():
            if ds_name not in per_ds_csv:
                continue
            dsets.append(
                SegDataset(
                    deriv_root=str(root),
                    split_csv=str(per_ds_csv[ds_name]),
                    split=split,
                    session_label=session_label,
                    space=space,
                    pad_mult=pad_mult,
                    augment=augment,
                )
            )

        if not dsets:
            raise RuntimeError(
                "No datasets constructed. Check dataset names in split_file vs cfg.dataset.roots keys."
            )

        return dsets[0] if len(dsets) == 1 else ConcatDataset(dsets)

    if mode == "single":
        if single_path in (None, ""):
            raise ValueError("Single-dataset mode requires dataset.path")

        _validate_single_split_csv(split_file)
        return SegDataset(
            deriv_root=str(single_path),
            split_csv=split_file,
            split=split,
            session_label=session_label,
            space=space,
            pad_mult=pad_mult,
            augment=augment,
        )

    raise ValueError(f"Unknown dataset mode: {mode}")


# -----------------------------------------------------------------------------
# Model / optimizer / scheduler / checkpoints
# -----------------------------------------------------------------------------
def build_model(cfg: DictConfig, num_classes: int) -> nn.Module:
    """Build Unet while staying compatible with older and newer Unet signatures."""
    kwargs: dict[str, Any] = {
        "c_in": int(cfg.model.in_channels),
        "c_out": int(num_classes),
    }

    sig = inspect.signature(Unet.__init__)
    optional_model_args = {
        "base_channels": OmegaConf.select(cfg, "model.base_channels", default=None),
        "features": OmegaConf.select(cfg, "model.features", default=None),
        "norm": OmegaConf.select(cfg, "model.norm", default=None),
        "dropout": OmegaConf.select(cfg, "model.dropout", default=None),
    }
    for name, value in optional_model_args.items():
        if value is not None and name in sig.parameters:
            kwargs[name] = value

    return Unet(**kwargs)


def build_ce_loss(cfg: DictConfig, num_classes: int, device: torch.device) -> nn.Module:
    weights = OmegaConf.select(cfg, "trainer.ce_class_weights", default=None)
    if weights in (None, ""):
        return nn.CrossEntropyLoss()

    w = torch.as_tensor(list(weights), dtype=torch.float32, device=device)
    if w.numel() != num_classes:
        raise ValueError(f"trainer.ce_class_weights must have {num_classes} values, got {w.numel()}")
    return nn.CrossEntropyLoss(weight=w)


def build_scheduler(cfg: DictConfig, optimizer: optim.Optimizer) -> tuple[Optional[Any], str]:
    raw = OmegaConf.select(cfg, "trainer.scheduler", default=None)
    num_epochs = int(cfg.trainer.num_epochs)
    base_lr = float(cfg.trainer.learning_rate)

    # Backward-compatible default: use cosine unless explicitly disabled.
    if raw is None:
        name = "cosine"
        params: dict[str, Any] = {}
    elif isinstance(raw, str):
        name = raw.lower().strip()
        params = {}
    else:
        name = str(OmegaConf.select(raw, "name", default="cosine")).lower().strip()
        params = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
        params = dict(params) if isinstance(params, dict) else {}

    if name in {"", "none", "null", "off", "false"}:
        return None, "none"

    if name in {"cosine", "cosineannealing", "cosine_annealing"}:
        eta_min = float(params.get("eta_min", min(base_lr * 0.01, 1e-6)))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min), "cosine"

    if name in {"plateau", "reduce_on_plateau", "reduce_lr_on_plateau"}:
        mode = str(params.get("mode", "max"))
        factor = float(params.get("factor", 0.5))
        patience = int(params.get("patience", 10))
        min_lr = float(params.get("min_lr", 1e-6))
        return (
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            ),
            "plateau",
        )

    if name in {"step", "steplr"}:
        step_size = int(params.get("step_size", max(num_epochs // 3, 1)))
        gamma = float(params.get("gamma", 0.5))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma), "step"

    raise ValueError(f"Unsupported trainer.scheduler.name='{name}'")


def current_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _optimizer_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_full_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[GradScaler],
    epoch: int,
    best_dice: float,
    best_epoch: int,
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_dice": float(best_dice),
        "best_epoch": int(best_epoch),
        "model": _state_dict(model),
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def save_model_only(path: Path, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict(model), path)


def cleanup_old_checkpoints(ckpt_dir: Path, pattern: str, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    files = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    old = files[:-keep_last_n]
    for p in old:
        try:
            p.unlink()
        except OSError as exc:
            logging.warning("Could not remove old checkpoint %s: %s", p, exc)


def load_resume_checkpoint(
    resume_from: Optional[str],
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[GradScaler],
    device: torch.device,
) -> tuple[int, float, int]:
    if resume_from in (None, ""):
        return 0, -1.0, -1

    ckpt_path = Path(_to_abs_str(resume_from))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    logging.info("Loading resume checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Full checkpoint format.
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            _optimizer_to_device(optimizer, device)
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_dice", -1.0)), int(ckpt.get("best_epoch", -1))

    # Model-only state_dict format.
    model.load_state_dict(ckpt, strict=True)
    return 0, -1.0, -1


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------
def run_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    loss_ce: nn.Module,
    loss_dice: DiceLoss,
    dice_weight: float,
    optimizer: Optional[optim.Optimizer],
    scaler: Optional[GradScaler],
    amp_enabled: bool,
    grad_clip_norm: float,
    train: bool,
    epoch: int,
    rank: int,
    is_ddp: bool,
) -> dict[str, float | list[float]]:
    model.train(mode=train)
    state = _new_metric_state(num_classes, device)

    desc = f"[train] ep{epoch}" if train else f"[val] ep{epoch}"
    pbar = tqdm(dataloader, desc=desc, leave=False, disable=not _main_process(rank))

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        if train:
            if optimizer is None:
                raise RuntimeError("optimizer must be provided for training")
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                logits = model(x)
                loss = loss_ce(logits, y) + dice_weight * loss_dice(logits, y)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        _update_metric_state(
            state,
            logits=logits.detach(),
            y=y,
            loss=loss.detach(),
            num_classes=num_classes,
        )

        if _main_process(rank):
            partial = _finalize_metric_state(state, exclude_bg=True)
            pbar.set_postfix(
                loss=f"{float(partial['loss']):.4f}",
                dice=f"{float(partial['dice']):.3f}",
                pixacc=f"{float(partial['pixel_acc']):.3f}",
            )

    _reduce_metric_state(state, is_ddp)
    return _finalize_metric_state(state, exclude_bg=True)


# -----------------------------------------------------------------------------
# Train entry
# -----------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_path="pkg://simcortex.configs.seg", config_name="train")
def main(cfg: DictConfig) -> None:
    rank, world, is_ddp, local_rank = setup_ddp(cfg)

    # Resolve outputs early. Keep config values updated so downstream helpers use the same paths.
    log_dir = Path(_to_abs_str(cfg.outputs.log_dir))
    ckpt_dir = Path(_to_abs_str(cfg.outputs.ckpt_dir))
    OmegaConf.update(cfg, "outputs.log_dir", str(log_dir), merge=False)
    OmegaConf.update(cfg, "outputs.ckpt_dir", str(ckpt_dir), merge=False)

    setup_logging(log_dir, rank)

    try:
        if _main_process(rank):
            logging.info("=== Segmentation config ===")
            logging.info("\n%s", OmegaConf.to_yaml(cfg))
            if bool(OmegaConf.select(cfg, "trainer.use_ddp", default=False)) and not is_ddp:
                logging.warning(
                    "trainer.use_ddp=true but torchrun env vars were not found; running single-process."
                )

        seed = OmegaConf.select(cfg, "trainer.seed", default=None)
        deterministic = bool(OmegaConf.select(cfg, "trainer.deterministic", default=False))
        if seed is not None:
            set_seed(int(seed), deterministic)

        if is_ddp:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            requested_device = str(OmegaConf.select(cfg, "trainer.device", default="cuda:0"))
            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                logging.warning("CUDA requested but unavailable; falling back to CPU.")
                requested_device = "cpu"
            device = torch.device(requested_device)

        if _main_process(rank):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
        if is_ddp:
            dist.barrier()

        train_split = str(getattr(cfg.dataset, "train_split", "train"))
        val_split = str(getattr(cfg.dataset, "val_split", "val"))

        mode, single_path, roots_map = _detect_train_dataset_mode(cfg)
        train_ds = build_dataset(cfg, train_split, rank, is_ddp, mode, single_path, roots_map)
        val_ds = build_dataset(cfg, val_split, rank, is_ddp, mode, single_path, roots_map)

        if _main_process(rank):
            logging.info("Train samples=%d | Val samples=%d", len(train_ds), len(val_ds))

        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=False)
            if is_ddp
            else None
        )
        val_sampler = DistributedEvalSampler(val_ds, num_replicas=world, rank=rank) if is_ddp else None

        num_workers = int(cfg.trainer.num_workers)
        batch_size = int(cfg.trainer.batch_size)
        pin_memory = device.type == "cuda"
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed) + rank)

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_seed_worker if seed is not None else None,
            generator=generator,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            worker_init_fn=_seed_worker if seed is not None else None,
        )

        num_classes = int(cfg.model.out_channels)
        model = build_model(cfg, num_classes=num_classes).to(device)

        if is_ddp:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )
        elif torch.cuda.device_count() > 1 and bool(OmegaConf.select(cfg, "trainer.data_parallel", default=False)):
            model = nn.DataParallel(model)

        loss_ce = build_ce_loss(cfg, num_classes, device)
        loss_dice = DiceLoss(
            num_classes=num_classes,
            exclude_bg=bool(OmegaConf.select(cfg, "trainer.dice_exclude_bg", default=True)),
            ignore_absent_target_classes=bool(
                OmegaConf.select(cfg, "trainer.dice_ignore_absent_target_classes", default=True)
            ),
        )
        dice_weight = float(OmegaConf.select(cfg, "trainer.dice_weight", default=1.0))

        optimizer_name = str(OmegaConf.select(cfg, "trainer.optimizer", default="adamw")).lower().strip()
        lr = float(cfg.trainer.learning_rate)
        weight_decay = float(OmegaConf.select(cfg, "trainer.weight_decay", default=0.0))
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported trainer.optimizer='{optimizer_name}'")

        scheduler, scheduler_kind = build_scheduler(cfg, optimizer)

        amp_enabled = bool(OmegaConf.select(cfg, "trainer.amp", default=(device.type == "cuda"))) and device.type == "cuda"
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None
        grad_clip_norm = float(OmegaConf.select(cfg, "trainer.grad_clip_norm", default=0.0) or 0.0)

        writer = SummaryWriter(str(log_dir)) if _main_process(rank) else None

        start_epoch = 0
        best_dice = -1.0
        best_epoch = -1
        resume_from = OmegaConf.select(cfg, "trainer.resume_from", default=None)
        if resume_from not in (None, ""):
            start_epoch, best_dice, best_epoch = load_resume_checkpoint(
                str(resume_from),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
            if _main_process(rank):
                logging.info(
                    "Resumed from epoch=%d | best_epoch=%d | best_dice=%.4f",
                    start_epoch,
                    best_epoch,
                    best_dice,
                )

        num_epochs = int(cfg.trainer.num_epochs)
        save_interval = int(OmegaConf.select(cfg, "trainer.save_interval", default=0) or 0)
        val_every = int(OmegaConf.select(cfg, "trainer.validation_interval", default=1) or 1)
        keep_last_n = int(OmegaConf.select(cfg, "trainer.keep_last_n_checkpoints", default=0) or 0)
        early_stop_patience = int(OmegaConf.select(cfg, "trainer.early_stop_patience", default=0) or 0)
        epochs_without_improvement = 0

        if _main_process(rank):
            logging.info(
                "Training setup | device=%s | ddp=%s world=%d | amp=%s | optimizer=%s | scheduler=%s | grad_clip=%.3f",
                device,
                is_ddp,
                world,
                amp_enabled,
                optimizer_name,
                scheduler_kind,
                grad_clip_norm,
            )

        for epoch in range(start_epoch + 1, num_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = run_one_epoch(
                model=model,
                dataloader=train_dl,
                device=device,
                num_classes=num_classes,
                loss_ce=loss_ce,
                loss_dice=loss_dice,
                dice_weight=dice_weight,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=grad_clip_norm,
                train=True,
                epoch=epoch,
                rank=rank,
                is_ddp=is_ddp,
            )

            if _main_process(rank):
                logging.info(
                    "Epoch %03d TRAIN | loss=%.4f dice=%.4f pixel_acc=%.4f lr=%.3e",
                    epoch,
                    float(train_metrics["loss"]),
                    float(train_metrics["dice"]),
                    float(train_metrics["pixel_acc"]),
                    current_lr(optimizer),
                )
                logging.debug("Train per-class Dice: %s", _format_per_class_dice(train_metrics["per_class_dice"]))
                if writer is not None:
                    writer.add_scalar("train/loss", float(train_metrics["loss"]), epoch)
                    writer.add_scalar("train/dice", float(train_metrics["dice"]), epoch)
                    writer.add_scalar("train/pixel_acc", float(train_metrics["pixel_acc"]), epoch)
                    writer.add_scalar("train/lr", current_lr(optimizer), epoch)

            did_validation = epoch % val_every == 0
            vdice_m: Optional[float] = None

            if did_validation:
                val_metrics = run_one_epoch(
                    model=model,
                    dataloader=val_dl,
                    device=device,
                    num_classes=num_classes,
                    loss_ce=loss_ce,
                    loss_dice=loss_dice,
                    dice_weight=dice_weight,
                    optimizer=None,
                    scaler=None,
                    amp_enabled=amp_enabled,
                    grad_clip_norm=0.0,
                    train=False,
                    epoch=epoch,
                    rank=rank,
                    is_ddp=is_ddp,
                )
                vdice_m = float(val_metrics["dice"])

                if _main_process(rank):
                    logging.info(
                        "Epoch %03d VAL   | loss=%.4f dice=%.4f pixel_acc=%.4f",
                        epoch,
                        float(val_metrics["loss"]),
                        float(val_metrics["dice"]),
                        float(val_metrics["pixel_acc"]),
                    )
                    logging.info("VAL per-class Dice: %s", _format_per_class_dice(val_metrics["per_class_dice"]))
                    if writer is not None:
                        writer.add_scalar("val/loss", float(val_metrics["loss"]), epoch)
                        writer.add_scalar("val/dice", float(val_metrics["dice"]), epoch)
                        writer.add_scalar("val/pixel_acc", float(val_metrics["pixel_acc"]), epoch)
                        for cls_idx, value in enumerate(val_metrics["per_class_dice"]):
                            if not np.isnan(value):
                                writer.add_scalar(f"val/dice_class_{cls_idx}", float(value), epoch)

                    if vdice_m > best_dice:
                        best_dice = vdice_m
                        best_epoch = epoch
                        epochs_without_improvement = 0
                        best_model_path = ckpt_dir / "seg_best_dice.pt"
                        best_full_path = ckpt_dir / "seg_best_dice_full.pt"
                        save_model_only(best_model_path, model)
                        save_full_checkpoint(
                            best_full_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            epoch=epoch,
                            best_dice=best_dice,
                            best_epoch=best_epoch,
                            cfg=cfg,
                        )
                        logging.info(
                            "Best updated: epoch=%d dice=%.4f -> %s",
                            best_epoch,
                            best_dice,
                            best_model_path,
                        )
                    else:
                        epochs_without_improvement += 1

            # Step scheduler after validation so ReduceLROnPlateau can use validation Dice.
            if scheduler is not None:
                if scheduler_kind == "plateau":
                    if did_validation and vdice_m is not None:
                        scheduler.step(vdice_m)
                else:
                    scheduler.step()

            if _main_process(rank):
                # Always keep a resumable last checkpoint.
                save_full_checkpoint(
                    ckpt_dir / "seg_last_full.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_dice=best_dice,
                    best_epoch=best_epoch,
                    cfg=cfg,
                )

                if save_interval and epoch % save_interval == 0:
                    model_path = ckpt_dir / f"seg_epoch_{epoch:03d}.pt"
                    full_path = ckpt_dir / f"seg_epoch_{epoch:03d}_full.pt"
                    save_model_only(model_path, model)
                    save_full_checkpoint(
                        full_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_dice=best_dice,
                        best_epoch=best_epoch,
                        cfg=cfg,
                    )
                    logging.info("Saved checkpoint: %s", model_path)
                    cleanup_old_checkpoints(ckpt_dir, "seg_epoch_[0-9][0-9][0-9].pt", keep_last_n)
                    cleanup_old_checkpoints(ckpt_dir, "seg_epoch_[0-9][0-9][0-9]_full.pt", keep_last_n)

            stop_flag = torch.zeros((), dtype=torch.int64, device=device)
            if _main_process(rank) and early_stop_patience > 0 and did_validation:
                if epochs_without_improvement >= early_stop_patience:
                    logging.info(
                        "Early stopping triggered at epoch=%d after %d validation checks without improvement.",
                        epoch,
                        epochs_without_improvement,
                    )
                    stop_flag.fill_(1)

            if is_ddp:
                dist.broadcast(stop_flag, src=0)
            if int(stop_flag.item()) == 1:
                break

        if writer is not None:
            writer.close()

        if _main_process(rank):
            logging.info("Done. Best epoch=%d, best val dice=%.4f", best_epoch, best_dice)

    finally:
        cleanup_ddp(is_ddp)


if __name__ == "__main__":
    main()
