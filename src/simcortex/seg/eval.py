#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

try:
    from monai.metrics import compute_surface_dice
except Exception:  # pragma: no cover
    compute_surface_dice = None  # type: ignore[assignment]

from simcortex.seg.data.dataloader import EvalSegDataset


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(log_dir: str | Path, filename: str = "seg_eval.log") -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / filename

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


# -----------------------------------------------------------------------------
# Path/config helpers
# -----------------------------------------------------------------------------
def _as_abs_path(path_like: Any) -> str:
    if path_like is None:
        raise ValueError("Cannot resolve None as a path.")
    return to_absolute_path(str(path_like))


def _as_optional_abs_path(path_like: Any) -> Optional[str]:
    if path_like in (None, ""):
        return None
    return _as_abs_path(path_like)


def _get_map(cfg_node: Any, keys: Tuple[str, ...]) -> Optional[Dict[str, str]]:
    for k in keys:
        v = getattr(cfg_node, k, None)
        if v is not None and hasattr(v, "items"):
            out = {str(kk): _as_abs_path(vv) for kk, vv in v.items()}
            return out if out else None
    return None


def _validate_single_split_csv(split_csv: str, dataset_name: Optional[str] = None) -> Optional[Path]:
    """
    Validate a single-dataset split CSV.

    If the split file has a dataset column with multiple datasets, dataset_name may be
    supplied to create a filtered temporary CSV. This keeps single-dataset mode usable
    with a combined split file.
    """
    split_csv_p = Path(split_csv)
    df = pd.read_csv(split_csv_p)

    req = {"subject", "split"}
    if not req.issubset(set(df.columns)):
        raise ValueError(
            f"Single-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}"
        )

    if "dataset" not in df.columns:
        return None

    vals = sorted(df["dataset"].dropna().astype(str).str.strip().unique().tolist())
    if len(vals) <= 1:
        return None

    if dataset_name is None:
        raise ValueError(
            "Single-dataset eval received a split_file with multiple dataset values. "
            "Either provide a split CSV for one dataset only or set dataset.name to filter it."
        )

    dataset_name = str(dataset_name).strip()
    df_ds = df[df["dataset"].astype(str).str.strip() == dataset_name][["subject", "split"]]
    if df_ds.empty:
        raise ValueError(f"No rows for dataset.name='{dataset_name}' in split file: {split_csv}")

    filtered = split_csv_p.parent / f".single_eval_split_{dataset_name}.csv"
    df_ds.to_csv(filtered, index=False)
    logging.info("Filtered single-dataset split CSV written: %s", filtered)
    return filtered


def _cache_per_dataset_csvs(split_csv: str, cache_dir: Path, roots: Dict[str, str]) -> Dict[str, str]:
    """
    Given a combined CSV with columns [subject, split, dataset], write one CSV per
    configured dataset with columns [subject, split].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")

    # Remove stale cache files from previous runs.
    for ds_name in roots.keys():
        p = cache_dir / f"split_{ds_name}.csv"
        if p.exists():
            p.unlink()

    out_map: Dict[str, str] = {}
    for ds_name in roots.keys():
        out = cache_dir / f"split_{ds_name}.csv"
        df_ds = df[df["dataset"].astype(str).str.strip() == ds_name][["subject", "split"]]
        if df_ds.empty:
            logging.warning("No rows for dataset='%s' in %s", ds_name, split_csv)
            continue
        df_ds.to_csv(out, index=False)
        out_map[ds_name] = str(out)

    if not out_map:
        raise RuntimeError(f"No cached per-dataset split files found in {cache_dir}")

    return out_map


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _validate_label_range(arr: np.ndarray, num_classes: int, name: str, subject: str) -> None:
    labels = np.unique(arr)
    bad = labels[(labels < 0) | (labels >= num_classes)]
    if bad.size:
        raise ValueError(
            f"{name} for {subject} contains labels outside [0, {num_classes - 1}]: "
            f"{bad[:20].tolist()}"
        )


def per_class_dice_np(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    exclude_background: bool = True,
    eps: float = 1e-6,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    start_cls = 1 if exclude_background else 0

    for c in range(start_cls, num_classes):
        gt_c = gt == c
        pred_c = pred == c
        inter = np.logical_and(gt_c, pred_c).sum(dtype=np.float64)
        union = gt_c.sum(dtype=np.float64) + pred_c.sum(dtype=np.float64)

        key = f"dice_c{c}"
        if union == 0:
            out[key] = float("nan")
        else:
            out[key] = float((2.0 * inter + eps) / (union + eps))

    return out


def mean_dice_from_per_class(per_class: Dict[str, float]) -> float:
    vals = np.asarray([v for v in per_class.values()], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def foreground_pixel_accuracy_np(gt: np.ndarray, pred: np.ndarray) -> float:
    mask = gt > 0
    if not np.any(mask):
        return 0.0
    return float((gt[mask] == pred[mask]).sum() / mask.sum())


def pixel_accuracy_np(gt: np.ndarray, pred: np.ndarray) -> float:
    return float((gt == pred).sum() / gt.size)


def nsd_monai(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    tolerance_vox: float = 1.0,
    include_background: bool = False,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Tuple[float, Dict[str, float]]:
    """
    Mean Normalized Surface Dice using MONAI.

    Returns
    -------
    mean_nsd:
        Mean over finite per-class NSD values.
    per_class:
        Dictionary with keys nsd_c1, nsd_c2, ... when include_background=False.
    """
    if compute_surface_dice is None:
        raise RuntimeError(
            "MONAI compute_surface_dice is unavailable. Install MONAI or set evaluation.compute_nsd=false."
        )

    gt_t = torch.from_numpy(gt.astype(np.int64, copy=False)).unsqueeze(0)
    pred_t = torch.from_numpy(pred.astype(np.int64, copy=False)).unsqueeze(0)

    gt_1h = F.one_hot(gt_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    pred_1h = F.one_hot(pred_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    n_thresholds = num_classes if include_background else num_classes - 1
    class_thresholds = [float(tolerance_vox)] * n_thresholds

    nsd = compute_surface_dice(
        y_pred=pred_1h,
        y=gt_1h,
        class_thresholds=class_thresholds,
        include_background=include_background,
        distance_metric="euclidean",
        spacing=tuple(float(s) for s in spacing),
        use_subvoxels=False,
    )

    # MONAI returns [B, C_selected]. We evaluate one subject at a time.
    nsd_vals = nsd.detach().cpu().float().reshape(-1).numpy()
    finite = nsd_vals[np.isfinite(nsd_vals)]
    mean_nsd = float(finite.mean()) if finite.size else 0.0

    start_cls = 0 if include_background else 1
    per_class = {
        f"nsd_c{c}": float(v) if np.isfinite(v) else float("nan")
        for c, v in zip(range(start_cls, start_cls + len(nsd_vals)), nsd_vals)
    }
    return mean_nsd, per_class


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------
def build_eval_datasets(cfg) -> List[Tuple[str, EvalSegDataset]]:
    ds_cfg = cfg.dataset
    split_csv = _as_abs_path(ds_cfg.split_file)
    split_name = str(ds_cfg.split_name)
    session_label = str(getattr(ds_cfg, "session_label", "01"))
    space = str(getattr(ds_cfg, "space", "MNI152"))

    single_path = _as_optional_abs_path(OmegaConf.select(cfg, "dataset.path", default=None))
    single_pred_root = _as_optional_abs_path(OmegaConf.select(cfg, "outputs.pred_root", default=None))
    single_dataset_name = OmegaConf.select(cfg, "dataset.name", default=None)
    single_dataset_name = None if single_dataset_name in (None, "") else str(single_dataset_name)

    roots_map = _get_map(ds_cfg, ("roots", "dataset_roots", "deriv_roots"))
    pred_roots_map = _get_map(cfg.outputs, ("pred_roots", "out_roots", "pred_out_roots"))

    # SINGLE mode takes priority if dataset.path is explicitly provided.
    if single_path is not None:
        logging.info("Segmentation eval mode: SINGLE-DATASET")
        logging.info("dataset.path = %s", single_path)
        logging.info("outputs.pred_root = %s", single_pred_root)

        if single_pred_root is None:
            raise ValueError("Single-dataset eval requires outputs.pred_root")

        if roots_map is not None:
            logging.warning(
                "Both dataset.path and dataset.roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.roots."
            )

        if pred_roots_map is not None:
            logging.warning(
                "Both outputs.pred_root and outputs.pred_roots are present. "
                "Using SINGLE-DATASET mode and ignoring outputs.pred_roots."
            )

        filtered_csv = _validate_single_split_csv(split_csv, dataset_name=single_dataset_name)
        split_for_dataset = str(filtered_csv) if filtered_csv is not None else split_csv

        ds = EvalSegDataset(
            deriv_root=single_path,
            split_csv=split_for_dataset,
            pred_root=single_pred_root,
            split_name=split_name,
            session_label=session_label,
            space=space,
        )
        return [(single_dataset_name or "SINGLE", ds)]

    # MULTI mode.
    if roots_map is not None:
        logging.info("Segmentation eval mode: MULTI-DATASET")
        logging.info("dataset.roots keys = %s", list(roots_map.keys()))

        if pred_roots_map is None:
            raise ValueError("For multi-dataset eval, set outputs.pred_roots.")

        missing = sorted(set(roots_map.keys()) - set(pred_roots_map.keys()))
        if missing:
            raise ValueError(f"outputs.pred_roots missing keys: {missing}")

        extra = sorted(set(pred_roots_map.keys()) - set(roots_map.keys()))
        if extra:
            logging.warning("outputs.pred_roots has extra keys not in dataset.roots: %s", extra)

        cache_dir = Path(_as_abs_path(cfg.outputs.log_dir)) / "split_cache"
        per_ds_csv = _cache_per_dataset_csvs(split_csv, cache_dir, roots_map)

        items: List[Tuple[str, EvalSegDataset]] = []
        for ds_name, deriv_root in roots_map.items():
            if ds_name not in per_ds_csv:
                continue
            pred_root = pred_roots_map[ds_name]
            ds = EvalSegDataset(
                deriv_root=str(deriv_root),
                split_csv=str(per_ds_csv[ds_name]),
                pred_root=str(pred_root),
                split_name=split_name,
                session_label=session_label,
                space=space,
            )
            items.append((ds_name, ds))

        if not items:
            raise RuntimeError(
                "No datasets constructed for eval. Check dataset names in split_file vs cfg.dataset.roots keys."
            )
        return items

    raise ValueError(
        "Could not determine segmentation eval mode. "
        "Provide either:\n"
        "  - dataset.path + outputs.pred_root   (single-dataset)\n"
        "or\n"
        "  - dataset.roots + outputs.pred_roots (multi-dataset)"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_path="pkg://simcortex.configs.seg", config_name="eval")
def main(cfg) -> None:
    log_dir = Path(_as_abs_path(cfg.outputs.log_dir))
    setup_logger(log_dir, "seg_eval.log")

    logging.info("=== Segmentation Eval config ===")
    logging.info("\n%s", OmegaConf.to_yaml(cfg))

    num_classes = int(cfg.evaluation.num_classes)
    exclude_bg = bool(cfg.evaluation.exclude_background)
    eps = float(getattr(cfg.evaluation, "eps", 1e-6))

    compute_nsd = bool(getattr(cfg.evaluation, "compute_nsd", True))
    nsd_tol = float(getattr(cfg.evaluation, "nsd_tolerance_vox", 1.0))
    nsd_include_bg = bool(getattr(cfg.evaluation, "nsd_include_background", False))
    spacing = tuple(float(x) for x in getattr(cfg.evaluation, "spacing", (1.0, 1.0, 1.0)))
    if len(spacing) != 3:
        raise ValueError(f"evaluation.spacing must have length 3, got: {spacing}")

    datasets = build_eval_datasets(cfg)

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_total = 0

    for ds_name, ds in datasets:
        logging.info("[%s] Evaluating %d subjects on split=%s", ds_name, len(ds), cfg.dataset.split_name)

        for i in range(len(ds)):
            try:
                gt9, pred_arr, sub, ses = ds[i]

                if gt9.shape != pred_arr.shape:
                    raise ValueError(
                        f"GT/pred shape mismatch after dataloader crop: gt={gt9.shape}, pred={pred_arr.shape}"
                    )

                _validate_label_range(gt9, num_classes, "GT", str(sub))
                _validate_label_range(pred_arr, num_classes, "Prediction", str(sub))

                dice_per_class = per_class_dice_np(
                    gt9,
                    pred_arr,
                    num_classes=num_classes,
                    exclude_background=exclude_bg,
                    eps=eps,
                )
                mean_dice = mean_dice_from_per_class(dice_per_class)
                pix_acc = pixel_accuracy_np(gt9, pred_arr)
                fg_acc = foreground_pixel_accuracy_np(gt9, pred_arr)

                row: Dict[str, Any] = {
                    "dataset": ds_name,
                    "subject": sub,
                    "session": ses,
                    "dice_mean": mean_dice,
                    "pixel_acc": pix_acc,
                    "foreground_pixel_acc": fg_acc,
                }
                row.update(dice_per_class)

                if compute_nsd:
                    nsd_mean, nsd_per_class = nsd_monai(
                        gt9,
                        pred_arr,
                        num_classes=num_classes,
                        tolerance_vox=nsd_tol,
                        include_background=nsd_include_bg,
                        spacing=spacing,
                    )
                    row["nsd_mean"] = nsd_mean
                    row.update(nsd_per_class)

                records.append(row)
                n_total += 1

                msg = (
                    f"[{ds_name}] {sub} {ses}: "
                    f"Dice={mean_dice:.4f}, pixelAcc={pix_acc:.4f}, fgAcc={fg_acc:.4f}"
                )
                if compute_nsd:
                    msg += f", NSD={row['nsd_mean']:.4f}"
                logging.info(msg)

            except Exception as e:
                failures.append(
                    {
                        "dataset": ds_name,
                        "index": i,
                        "error": repr(e),
                    }
                )
                logging.warning("[%s] Failed: index=%d err=%r", ds_name, i, e)

    eval_csv = Path(_as_abs_path(cfg.outputs.eval_csv))
    eval_csv.parent.mkdir(parents=True, exist_ok=True)

    failed_csv = Path(
        _as_abs_path(getattr(cfg.outputs, "failed_csv", eval_csv.with_name(eval_csv.stem + "_failed.csv")))
    )
    summary_csv = Path(
        _as_abs_path(getattr(cfg.outputs, "summary_csv", eval_csv.with_name(eval_csv.stem + "_summary_by_dataset.csv")))
    )
    summary_json = Path(
        _as_abs_path(getattr(cfg.outputs, "summary_json", eval_csv.with_name(eval_csv.stem + "_summary.json")))
    )

    if failures:
        pd.DataFrame(failures).to_csv(failed_csv, index=False)
        logging.info("Saved failure report: %s", failed_csv)
    else:
        # Write an empty failure report with stable columns.
        pd.DataFrame(columns=["dataset", "index", "error"]).to_csv(failed_csv, index=False)

    if not records:
        logging.warning("No subjects evaluated successfully.")
        return

    df = pd.DataFrame(records)
    df.to_csv(eval_csv, index=False)
    logging.info("Saved per-subject metrics to %s", eval_csv)

    metric_cols = [
        c
        for c in df.columns
        if c.startswith("dice_")
        or c in {"pixel_acc", "foreground_pixel_acc", "nsd_mean"}
        or c.startswith("nsd_c")
    ]

    by_dataset = df.groupby("dataset")[metric_cols].agg(["count", "mean", "std"])
    by_dataset.columns = [f"{m}_{s}" for (m, s) in by_dataset.columns]
    by_dataset = by_dataset.reset_index()
    by_dataset.to_csv(summary_csv, index=False)
    logging.info("Saved dataset summary to %s", summary_csv)

    overall_stats = df[metric_cols].agg(["mean", "std", "count"])
    overall_payload = {
        "evaluated": int(n_total),
        "failed": int(len(failures)),
        "metrics": {
            stat: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
            for stat, row in overall_stats.to_dict(orient="index").items()
        },
    }
    summary_json.write_text(json.dumps(overall_payload, indent=2), encoding="utf-8")
    logging.info("Saved overall summary to %s", summary_json)

    logging.info(
        "OVERALL | Dice=%s ± %s | FG-pixel-acc=%s ± %s | Evaluated=%d Failed=%d",
        f"{overall_stats.loc['mean', 'dice_mean']:.4f}" if "dice_mean" in overall_stats else "NA",
        f"{overall_stats.loc['std', 'dice_mean']:.4f}" if "dice_mean" in overall_stats else "NA",
        f"{overall_stats.loc['mean', 'foreground_pixel_acc']:.4f}"
        if "foreground_pixel_acc" in overall_stats
        else "NA",
        f"{overall_stats.loc['std', 'foreground_pixel_acc']:.4f}"
        if "foreground_pixel_acc" in overall_stats
        else "NA",
        n_total,
        len(failures),
    )

    out_xlsx = getattr(cfg.outputs, "eval_xlsx", None)
    if out_xlsx:
        out_xlsx_p = Path(_as_abs_path(out_xlsx))
        out_xlsx_p.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_xlsx_p, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="per_subject", index=False)
            by_dataset.to_excel(w, sheet_name="summary_by_dataset", index=False)
            overall_stats.reset_index().rename(columns={"index": "stat"}).to_excel(
                w, sheet_name="summary_overall", index=False
            )
            if failures:
                pd.DataFrame(failures).to_excel(w, sheet_name="failures", index=False)
        logging.info("Saved Excel report to %s", out_xlsx_p)

    logging.info("Done. Evaluated=%d Failed=%d", n_total, len(failures))


if __name__ == "__main__":
    main()
